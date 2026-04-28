"""
authenticity/price_evidence.py — v17 Phase 3: claimed-vs-official price comparison.

The cheapest, most reliable wrapper-detection signal is plain economics:
no proxy can sustainably resell GPT-4o for less than 30% of the official
USD/Mtok price.  Users can declare the per-token rate they're being
charged (or that the upstream advertises) when launching a run; this
module compares it against the corresponding official rate from
``_data/official_prices.yaml`` and emits a structured ``PriceEvidence``
that the verdict engine treats as a hard rule.

Schema of the loaded price table is documented in the YAML header.

This module does **no I/O at import time** — the price table is loaded
lazily on first ``get_official_price`` call and cached in process memory.
"""
from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


_PRICE_TABLE_PATH = pathlib.Path(__file__).parent.parent / "_data" / "official_prices.yaml"
_PRICE_CACHE_LOCK = threading.Lock()
_PRICE_CACHE: dict[str, dict] | None = None


# ── YAML loader (uses PyYAML if available, falls back to safe parsing) ───────


def _load_yaml(path: pathlib.Path) -> dict:
    """Load YAML preferring PyYAML; raise on failure."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to parse official_prices.yaml; install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _ensure_loaded() -> dict[str, dict]:
    """Return a model_id-keyed price map, loading from disk on first call."""
    global _PRICE_CACHE
    if _PRICE_CACHE is not None:
        return _PRICE_CACHE
    with _PRICE_CACHE_LOCK:
        if _PRICE_CACHE is not None:
            return _PRICE_CACHE
        if not _PRICE_TABLE_PATH.exists():
            logger.warning("official_prices.yaml not found", path=str(_PRICE_TABLE_PATH))
            _PRICE_CACHE = {}
            return _PRICE_CACHE
        try:
            data = _load_yaml(_PRICE_TABLE_PATH)
        except Exception as exc:
            logger.warning("failed to load official_prices.yaml", error=str(exc))
            _PRICE_CACHE = {}
            return _PRICE_CACHE
        models = data.get("models") or []
        cache: dict[str, dict] = {}
        for entry in models:
            mid = (entry or {}).get("model_id")
            if not isinstance(mid, str) or not mid:
                continue
            cache[mid.lower()] = entry
        _PRICE_CACHE = cache
        return _PRICE_CACHE


# Public helper used by Phase 6 sync to invalidate the cache after rewrite
def _invalidate_cache() -> None:
    global _PRICE_CACHE
    with _PRICE_CACHE_LOCK:
        _PRICE_CACHE = None


def get_official_price(model_id: str) -> dict | None:
    """Return the canonical price record for ``model_id`` or ``None``.

    Lookup is case-insensitive on ``model_id``; the returned dict matches
    the YAML schema (``input_per_mtok_usd``, ``output_per_mtok_usd``,
    optional ``cache_read_per_mtok_usd``, etc.).
    """
    if not model_id:
        return None
    return _ensure_loaded().get(model_id.lower())


# ── Evidence structure ──────────────────────────────────────────────────────


# Hard-rule thresholds (registered in _data/SOURCES.yaml as
# pricing.below_official_30pct and pricing.below_official_60pct).
PRICE_BELOW_30PCT_THRESHOLD = 0.30
PRICE_BELOW_60PCT_THRESHOLD = 0.60
PRICE_ABOVE_120PCT_THRESHOLD = 1.20


@dataclass
class PriceEvidence:
    """Result of comparing claimed price against official rate."""

    model_id: str
    has_claim: bool                 # True if user supplied any claim
    has_official: bool              # True if official rate found in registry

    claimed_input_usd_per_mtok: float | None = None
    claimed_output_usd_per_mtok: float | None = None
    official_input_usd_per_mtok: float | None = None
    official_output_usd_per_mtok: float | None = None

    input_ratio: float | None = None              # claimed / official
    output_ratio: float | None = None
    blended_ratio: float | None = None            # geometric mean (input * output) ** 0.5

    severity: str = "none"                        # none | suspicious | fake_high_confidence | overpaid
    reasons: list[str] | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "has_claim": self.has_claim,
            "has_official": self.has_official,
            "claimed_input_usd_per_mtok": self.claimed_input_usd_per_mtok,
            "claimed_output_usd_per_mtok": self.claimed_output_usd_per_mtok,
            "official_input_usd_per_mtok": self.official_input_usd_per_mtok,
            "official_output_usd_per_mtok": self.official_output_usd_per_mtok,
            "input_ratio": (
                round(self.input_ratio, 4) if self.input_ratio is not None else None
            ),
            "output_ratio": (
                round(self.output_ratio, 4) if self.output_ratio is not None else None
            ),
            "blended_ratio": (
                round(self.blended_ratio, 4) if self.blended_ratio is not None else None
            ),
            "severity": self.severity,
            "reasons": self.reasons or [],
            "source_url": self.source_url,
        }


# ── Comparison entry point ──────────────────────────────────────────────────


def evaluate_pricing(
    model_id: str,
    claimed_input_usd_per_mtok: float | None,
    claimed_output_usd_per_mtok: float | None,
    currency: str = "USD",
) -> PriceEvidence:
    """Compare a user-supplied claimed price against the official rate.

    Either ``claimed_input`` or ``claimed_output`` may be ``None``; if both
    are missing, the result has ``severity='none'`` and ``has_claim=False``.

    Severity ladder (hard rules):
      * ``blended_ratio < 0.30`` → ``fake_high_confidence``
      * ``0.30 ≤ blended_ratio < 0.60`` → ``suspicious``
      * ``blended_ratio > 1.20`` → ``overpaid`` (informational; common for
        managed proxies, NOT a wrapper signal)
      * otherwise → ``none``
    """
    has_claim = (
        claimed_input_usd_per_mtok is not None
        or claimed_output_usd_per_mtok is not None
    )

    # Currency safety: official prices are denominated in USD. A user who
    # supplies CNY 1.4 / Mtok would otherwise be compared against USD 1.4
    # and falsely trigger the <30% cap. Refuse non-USD claims rather than
    # silently misinterpret them.
    cur = (currency or "USD").upper()
    if has_claim and cur != "USD":
        ev = PriceEvidence(
            model_id=model_id,
            has_claim=has_claim,
            has_official=False,
            claimed_input_usd_per_mtok=claimed_input_usd_per_mtok,
            claimed_output_usd_per_mtok=claimed_output_usd_per_mtok,
            reasons=[
                f"声明价格币种为 {cur}，价格证据仅支持 USD（官方价基准为 USD/Mtok）；"
                f"请将报价折算为 USD 后重新提交"
            ],
        )
        ev.severity = "none"
        return ev

    official = get_official_price(model_id) if model_id else None
    has_official = official is not None

    ev = PriceEvidence(
        model_id=model_id,
        has_claim=has_claim,
        has_official=has_official,
        claimed_input_usd_per_mtok=claimed_input_usd_per_mtok,
        claimed_output_usd_per_mtok=claimed_output_usd_per_mtok,
        reasons=[],
    )

    if not (has_claim and has_official):
        return ev

    ev.official_input_usd_per_mtok = float(official.get("input_per_mtok_usd") or 0.0) or None
    ev.official_output_usd_per_mtok = float(official.get("output_per_mtok_usd") or 0.0) or None
    ev.source_url = official.get("source_url")

    # Compute per-direction ratios where both sides exist.
    if (
        claimed_input_usd_per_mtok is not None
        and ev.official_input_usd_per_mtok
        and ev.official_input_usd_per_mtok > 0
    ):
        ev.input_ratio = float(claimed_input_usd_per_mtok) / ev.official_input_usd_per_mtok
    if (
        claimed_output_usd_per_mtok is not None
        and ev.official_output_usd_per_mtok
        and ev.official_output_usd_per_mtok > 0
    ):
        ev.output_ratio = float(claimed_output_usd_per_mtok) / ev.official_output_usd_per_mtok

    # Blended ratio: geometric mean if both directions exist; else the one we have.
    if ev.input_ratio is not None and ev.output_ratio is not None:
        ev.blended_ratio = (ev.input_ratio * ev.output_ratio) ** 0.5
    elif ev.input_ratio is not None:
        ev.blended_ratio = ev.input_ratio
    elif ev.output_ratio is not None:
        ev.blended_ratio = ev.output_ratio

    if ev.blended_ratio is None:
        return ev

    pct = ev.blended_ratio * 100.0
    if ev.blended_ratio < PRICE_BELOW_30PCT_THRESHOLD:
        ev.severity = "fake_high_confidence"
        ev.reasons.append(
            f"声称价格仅为官方价的 {pct:.0f}%（< 30%），经济上无法持续维持，"
            f"强烈提示为中转/聚合代理"
        )
    elif ev.blended_ratio < PRICE_BELOW_60PCT_THRESHOLD:
        ev.severity = "suspicious"
        ev.reasons.append(
            f"声称价格仅为官方价的 {pct:.0f}%（< 60%），疑似中转代理"
        )
    elif ev.blended_ratio > PRICE_ABOVE_120PCT_THRESHOLD:
        ev.severity = "overpaid"
        ev.reasons.append(
            f"声称价格为官方价的 {pct:.0f}%（> 120%），高于官方但合规"
            f"（典型托管代理加价场景）"
        )
    return ev


__all__ = [
    "PriceEvidence",
    "evaluate_pricing",
    "get_official_price",
    "PRICE_BELOW_30PCT_THRESHOLD",
    "PRICE_BELOW_60PCT_THRESHOLD",
    "PRICE_ABOVE_120PCT_THRESHOLD",
]
