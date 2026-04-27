"""
runner/self_probe_register.py — v17 Phase 8: self-registration probe.

When a user submits a run for a model_id that ``model_registry`` has
never seen, the orchestrator schedules one short *self-probe* against
the user's endpoint and records the resulting fingerprint in the
registry with ``data_source='self_probed'`` and ``confidence=0.85``.
The probe outputs are explicitly *not* eligible to act as official
baselines (Phase 11 query filters self_probed out).

Four miniature probes (~1k tokens total budget):

  1. Cutoff binary-search   — yes/no events at quarterly cadence
  2. Tokenizer fingerprint  — 4 known special-token boundaries
  3. Timing                 — N short calls to derive p50 TTFT/TPS
  4. Self-report            — direct identity probe (banner-grab)

Public surface:

  run_self_probe(model_id, *, ask_fn, tokenize_fn, time_fn, identity_fn,
                 vendor_hint=None, persist=True) -> dict

All four probe functions are injected so tests can run the orchestrator
fully offline.  Production wiring lives in the orchestrator and supplies
real adapter-backed callables.

Output dict mirrors the registry record schema and includes a
``fingerprint_sha256`` derived from a stable subset of fields.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.logging import get_logger
from app.repository import registry_repo

logger = get_logger(__name__)


# ── Probe 1: cutoff binary search ───────────────────────────────────────────


# Each anchor is (iso_date, short_human_description).  Anchors are listed in
# strict chronological order so binary search is well-defined.
_DEFAULT_CUTOFF_ANCHORS: tuple[tuple[str, str], ...] = (
    ("2023-09-30", "OpenAI's DevDay 2023 announced GPT-4 Turbo"),
    ("2024-03-31", "Anthropic released the Claude 3 family"),
    ("2024-09-30", "OpenAI released o1-preview"),
    ("2025-03-31", "DeepSeek-V3 reasoning models gained adoption"),
    ("2025-09-30", "GPT-5 family general availability"),
)


def _render_event_prompt(date: str, event: str) -> str:
    return (
        "Reply with EXACTLY 'yes' or 'no' (lowercase, no punctuation). "
        "Did this event occur BEFORE your knowledge cutoff?\n"
        f"Event ({date}): {event}"
    )


@dataclass
class CutoffProbeResult:
    cutoff_date: str | None
    answers: list[tuple[str, bool]] = field(default_factory=list)
    total_calls: int = 0


def probe_cutoff(
    ask_fn: Callable[[str], bool],
    anchors: tuple[tuple[str, str], ...] = _DEFAULT_CUTOFF_ANCHORS,
) -> CutoffProbeResult:
    """Locate the largest anchor the model still recognises.

    ``ask_fn(prompt) -> bool`` must answer deterministically (caller
    typically wraps temperature=0 + parsed yes/no).
    """
    if not anchors:
        return CutoffProbeResult(cutoff_date=None)

    answers: list[tuple[str, bool]] = []
    calls = 0

    # Earliest / latest fast paths
    last_date, last_event = anchors[-1]
    knows_latest = ask_fn(_render_event_prompt(last_date, last_event))
    answers.append((last_date, knows_latest))
    calls += 1
    if knows_latest:
        return CutoffProbeResult(
            cutoff_date=f">= {last_date}", answers=answers, total_calls=calls,
        )

    first_date, first_event = anchors[0]
    knows_first = ask_fn(_render_event_prompt(first_date, first_event))
    answers.append((first_date, knows_first))
    calls += 1
    if not knows_first:
        return CutoffProbeResult(
            cutoff_date=f"< {first_date}", answers=answers, total_calls=calls,
        )

    # Standard binary search: find max index whose anchor the model knows.
    lo, hi = 0, len(anchors) - 1
    cutoff_idx = lo
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        date, event = anchors[mid]
        knows = ask_fn(_render_event_prompt(date, event))
        answers.append((date, knows))
        calls += 1
        if knows:
            lo = mid
            cutoff_idx = mid
        else:
            hi = mid
    return CutoffProbeResult(
        cutoff_date=anchors[cutoff_idx][0],
        answers=answers,
        total_calls=calls,
    )


# ── Probe 2: tokenizer fingerprint ─────────────────────────────────────────
#
# Each family parses certain special tokens differently; sending the
# probe string with stream=False and observing prompt_tokens already
# discriminates several families.  We rely on *relative* counts: the
# adapter only needs to return ``prompt_tokens`` per probe.

# Known fingerprints (probe_id -> {family: prompt_tokens lower bound, ...})
_TOKENIZER_PROBE_PROMPTS: tuple[tuple[str, str], ...] = (
    ("cl100k_marker", "<|im_start|>system\nyou are.<|im_end|>"),
    ("o200k_marker",  "<|start|>system<|message|>you are.<|end|>"),
    ("claude_marker", "Human: hi\n\nAssistant:"),
    ("llama_marker",  "<|begin_of_text|>hi"),
)


# Reference signatures: what prompt_tokens look like for each family on
# the four probes.  Treated as approximate matches: we score by L1
# distance and pick the best-fitting family.
_TOKENIZER_REFERENCE: dict[str, dict[str, int]] = {
    "cl100k":  {"cl100k_marker": 12, "o200k_marker": 22, "claude_marker": 7, "llama_marker": 6},
    "o200k":   {"cl100k_marker": 14, "o200k_marker":  9, "claude_marker": 7, "llama_marker": 6},
    "claude":  {"cl100k_marker": 22, "o200k_marker": 28, "claude_marker": 6, "llama_marker": 9},
    "llama":   {"cl100k_marker": 18, "o200k_marker": 24, "claude_marker": 7, "llama_marker": 4},
    "gemini":  {"cl100k_marker": 16, "o200k_marker": 20, "claude_marker": 7, "llama_marker": 7},
}


@dataclass
class TokenizerProbeResult:
    tokenizer_id: str | None
    distance: float
    samples: dict[str, int] = field(default_factory=dict)


def probe_tokenizer(
    tokenize_fn: Callable[[str], int],
    probes: tuple[tuple[str, str], ...] = _TOKENIZER_PROBE_PROMPTS,
) -> TokenizerProbeResult:
    """Match the model's tokenizer against known families.

    ``tokenize_fn(probe_text) -> prompt_tokens`` should return the
    number of input tokens the upstream charged for ``probe_text``.
    """
    samples: dict[str, int] = {}
    for probe_id, probe_text in probes:
        try:
            samples[probe_id] = max(0, int(tokenize_fn(probe_text)))
        except Exception as e:
            logger.warning("probe_tokenizer: tokenize_fn raised", probe=probe_id, error=str(e))
            samples[probe_id] = -1

    best_id: str | None = None
    best_dist = float("inf")
    for tid, ref in _TOKENIZER_REFERENCE.items():
        # L1 distance, ignoring failed probes (-1)
        dist = 0.0
        used = 0
        for k, expected in ref.items():
            v = samples.get(k, -1)
            if v < 0:
                continue
            dist += abs(v - expected)
            used += 1
        if used == 0:
            continue
        # Normalise by used count to keep partial samples comparable
        norm = dist / used
        if norm < best_dist:
            best_dist = norm
            best_id = tid
    return TokenizerProbeResult(
        tokenizer_id=best_id,
        distance=round(best_dist, 3) if best_id is not None else float("inf"),
        samples=samples,
    )


# ── Probe 3: timing ──────────────────────────────────────────────────────


@dataclass
class TimingProbeResult:
    n: int
    ttft_p50_ms: float | None
    tps_p50: float | None
    ttft_samples: list[float] = field(default_factory=list)
    tps_samples: list[float] = field(default_factory=list)


def probe_timing(
    time_fn: Callable[[], tuple[float, float] | None],
    n: int = 20,
) -> TimingProbeResult:
    """Run ``n`` timing samples via ``time_fn`` and return p50 quantiles.

    ``time_fn() -> (ttft_ms, tps) | None`` is invoked ``n`` times.
    A None return is treated as a failed sample and skipped.
    """
    ttfts: list[float] = []
    tpss: list[float] = []
    for _ in range(max(0, int(n))):
        try:
            r = time_fn()
        except Exception as e:
            logger.warning("probe_timing: time_fn raised", error=str(e))
            r = None
        if r is None:
            continue
        try:
            ttft, tps = float(r[0]), float(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if ttft <= 0 or tps <= 0:
            continue
        ttfts.append(ttft)
        tpss.append(tps)

    return TimingProbeResult(
        n=len(ttfts),
        ttft_p50_ms=round(statistics.median(ttfts), 2) if ttfts else None,
        tps_p50=round(statistics.median(tpss), 2) if tpss else None,
        ttft_samples=ttfts,
        tps_samples=tpss,
    )


# ── Probe 4: self-report banner grab ───────────────────────────────────────


@dataclass
class IdentityProbeResult:
    self_report_id: str
    raw_text: str = ""


def probe_identity(identity_fn: Callable[[], str]) -> IdentityProbeResult:
    """Ask the model what it is.  Returns lower-cased, trimmed first 80 chars."""
    try:
        text = identity_fn() or ""
    except Exception as e:
        logger.warning("probe_identity: identity_fn raised", error=str(e))
        text = ""
    raw = text[:400]
    sid = " ".join(text.split()).strip().lower()[:80]
    return IdentityProbeResult(self_report_id=sid, raw_text=raw)


# ── Orchestration ──────────────────────────────────────────────────────────


@dataclass
class SelfProbeReport:
    model_id: str
    vendor: str
    cutoff: CutoffProbeResult
    tokenizer: TokenizerProbeResult
    timing: TimingProbeResult
    identity: IdentityProbeResult
    fingerprint_sha256: str
    persisted: bool = False

    def to_registry_record(self) -> dict[str, Any]:
        # Strip the ">=" / "<" prefixes so the registry stores a clean date when possible.
        cutoff = self.cutoff.cutoff_date
        cutoff_clean: str | None = None
        if isinstance(cutoff, str):
            tok = cutoff.replace(">=", "").replace("<", "").strip()
            cutoff_clean = tok or None
        rec: dict[str, Any] = {
            "model_id": self.model_id,
            "vendor": self.vendor,
            "data_source": "self_probed",
            "confidence": 0.85,
            "status": "self_probed",
            "self_report_id": self.identity.self_report_id,
            "fingerprint_sha256": self.fingerprint_sha256,
            "raw_metadata": {
                "cutoff": {
                    "raw": cutoff,
                    "answers": self.cutoff.answers,
                    "calls": self.cutoff.total_calls,
                },
                "tokenizer": {
                    "samples": self.tokenizer.samples,
                    "distance": self.tokenizer.distance,
                },
                "timing": {
                    "n": self.timing.n,
                    "ttft_p50_ms": self.timing.ttft_p50_ms,
                    "tps_p50": self.timing.tps_p50,
                },
                "identity_raw": self.identity.raw_text,
            },
        }
        if cutoff_clean and cutoff_clean[0].isdigit():
            rec["cutoff_date"] = cutoff_clean
        if self.tokenizer.tokenizer_id is not None:
            rec["tokenizer_id"] = self.tokenizer.tokenizer_id
        if self.timing.ttft_p50_ms is not None:
            rec["ttft_p50_ms"] = self.timing.ttft_p50_ms
        if self.timing.tps_p50 is not None:
            rec["tps_p50"] = self.timing.tps_p50
        return rec

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "vendor": self.vendor,
            "cutoff": {
                "cutoff_date": self.cutoff.cutoff_date,
                "answers": self.cutoff.answers,
                "calls": self.cutoff.total_calls,
            },
            "tokenizer": {
                "tokenizer_id": self.tokenizer.tokenizer_id,
                "distance": self.tokenizer.distance,
                "samples": self.tokenizer.samples,
            },
            "timing": {
                "n": self.timing.n,
                "ttft_p50_ms": self.timing.ttft_p50_ms,
                "tps_p50": self.timing.tps_p50,
            },
            "identity": {
                "self_report_id": self.identity.self_report_id,
                "raw_text": self.identity.raw_text,
            },
            "fingerprint_sha256": self.fingerprint_sha256,
            "persisted": self.persisted,
        }


def _compute_fingerprint(
    cutoff: CutoffProbeResult,
    tokenizer: TokenizerProbeResult,
    timing: TimingProbeResult,
    identity: IdentityProbeResult,
) -> str:
    payload = json.dumps(
        {
            "cutoff": cutoff.cutoff_date,
            "tokenizer_id": tokenizer.tokenizer_id,
            "tokenizer_samples": tokenizer.samples,
            # Round timings into coarse buckets so trivial jitter doesn't change the hash
            "ttft_p50_bucket": (
                None if timing.ttft_p50_ms is None
                else int(timing.ttft_p50_ms // 50) * 50
            ),
            "tps_p50_bucket": (
                None if timing.tps_p50 is None
                else int(timing.tps_p50 // 5) * 5
            ),
            "self_report_id": identity.self_report_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_known_model(model_id: str) -> bool:
    """Return True if the model already has a registry row."""
    return registry_repo.get_model_card(model_id) is not None


def run_self_probe(
    model_id: str,
    *,
    ask_fn: Callable[[str], bool],
    tokenize_fn: Callable[[str], int],
    time_fn: Callable[[], tuple[float, float] | None],
    identity_fn: Callable[[], str],
    vendor_hint: str | None = None,
    timing_samples: int = 20,
    persist: bool = True,
) -> SelfProbeReport:
    """Run all four probes and (optionally) upsert into ``model_registry``."""
    if not model_id:
        raise ValueError("run_self_probe: model_id is required")

    cutoff = probe_cutoff(ask_fn)
    tokenizer = probe_tokenizer(tokenize_fn)
    timing = probe_timing(time_fn, n=timing_samples)
    identity = probe_identity(identity_fn)

    fingerprint = _compute_fingerprint(cutoff, tokenizer, timing, identity)

    report = SelfProbeReport(
        model_id=model_id,
        vendor=(vendor_hint or "unknown").lower(),
        cutoff=cutoff,
        tokenizer=tokenizer,
        timing=timing,
        identity=identity,
        fingerprint_sha256=fingerprint,
    )

    if persist:
        try:
            now = int(time.time())
            rec = report.to_registry_record()
            rec["last_synced_at"] = now
            registry_repo.upsert_model(rec)
            report.persisted = True
        except Exception as e:
            logger.warning("self_probe_register: upsert failed", model_id=model_id, error=str(e))

    return report


__all__ = [
    "run_self_probe",
    "is_known_model",
    "probe_cutoff",
    "probe_tokenizer",
    "probe_timing",
    "probe_identity",
    "CutoffProbeResult",
    "TokenizerProbeResult",
    "TimingProbeResult",
    "IdentityProbeResult",
    "SelfProbeReport",
]
