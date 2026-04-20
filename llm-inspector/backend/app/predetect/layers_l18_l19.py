"""
predetect/layers_l18_l19.py — L18 Response Timing Side-Channel & L19 Token Distribution Side-Channel.

L18 Reference:
    Yu et al. (2024) "Timing Attacks on LLM APIs"
    Note: specific arXiv pending verification; timing fingerprinting via TTFT/TPS patterns.

L19 Reference:
    Carlini et al. (2023) "Stealing Part of a Production Language Model"
    arXiv: https://arxiv.org/abs/2403.06634
"""
from __future__ import annotations

import math
from collections import Counter

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Timing reference profiles ────────────────────────────────────────────────

_TIMING_REFS: dict[str, dict] = {
    "claude":    {"ttft_ms_mean": 800,  "ttft_ms_std": 200, "tps_mean": 45},
    "gpt":       {"ttft_ms_mean": 600,  "ttft_ms_std": 150, "tps_mean": 50},
    "gemini":    {"ttft_ms_mean": 700,  "ttft_ms_std": 180, "tps_mean": 55},
    "qwen":      {"ttft_ms_mean": 500,  "ttft_ms_std": 120, "tps_mean": 60},
    "deepseek":  {"ttft_ms_mean": 1200, "ttft_ms_std": 300, "tps_mean": 35},
    "llama":     {"ttft_ms_mean": 400,  "ttft_ms_std": 100, "tps_mean": 70},
}

# ── Distribution reference profiles ─────────────────────────────────────────

_DIST_REFS: dict[str, dict] = {
    "claude":    {"avg_len": 480, "len_cv": 0.4,  "repetition_rate": 0.05},
    "gpt":       {"avg_len": 350, "len_cv": 0.5,  "repetition_rate": 0.06},
    "qwen":      {"avg_len": 400, "len_cv": 0.45, "repetition_rate": 0.07},
    "deepseek":  {"avg_len": 600, "len_cv": 0.35, "repetition_rate": 0.04},
}


# ── Helper: compute KL-like distance (Gaussian approximation) ────────────────

def _kl_gaussian(obs_mean: float, ref_mean: float, ref_std: float) -> float:
    """
    Gaussian KL-divergence approximation:
        kl = (obs_mean - ref_mean)^2 / (2 * ref_std^2)
    """
    if ref_std <= 0:
        return float("inf")
    return ((obs_mean - ref_mean) ** 2) / (2.0 * ref_std ** 2)


# ── Helper: Wasserstein-like 1D distance ─────────────────────────────────────

def _wasserstein_1d(obs_avg_len: float, ref_avg_len: float) -> float:
    """Wasserstein-like 1D distance: abs(obs - ref) / max(ref, 1)."""
    return abs(obs_avg_len - ref_avg_len) / max(ref_avg_len, 1.0)


# ── Helper: compute 4-gram repetition rate ───────────────────────────────────

def _repetition_rate(texts: list[str]) -> float:
    """
    Ratio of repeated 4-grams across all response texts.
    Returns 0.0 if no 4-grams can be formed.
    """
    all_ngrams: list[str] = []
    for text in texts:
        words = text.split()
        for i in range(len(words) - 3):
            ngram = " ".join(words[i:i + 4])
            all_ngrams.append(ngram)
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(all_ngrams)


# ── Layer 18: Timing Side-Channel ────────────────────────────────────────────

class Layer18TimingSideChannel:
    """
    Zero-token layer. Re-analyzes timing data already collected by prior layers.

    Computes:
      - mean TTFT, std TTFT, mean TPS from collected samples
      - KL-divergence-like distance against known timing profiles
      - Confidence capped at 0.50 (timing is weak evidence)
    """

    LAYER = 18
    NAME = "timing_side_channel"

    def run(
        self,
        adapter,                           # noqa: ARG002  (not used — zero-token layer)
        model_name: str,
        layer_results_so_far: list,
    ) -> dict:
        # Extract timing samples from prior layer dicts
        ttft_samples: list[float] = []
        tps_samples: list[float] = []

        for lr in layer_results_so_far:
            raw = lr if isinstance(lr, dict) else (lr.to_dict() if hasattr(lr, "to_dict") else {})
            ttft = raw.get("ttft_ms")
            tps_ = raw.get("tps")
            if ttft is not None:
                try:
                    ttft_samples.append(float(ttft))
                except (TypeError, ValueError):
                    pass
            if tps_ is not None:
                try:
                    tps_samples.append(float(tps_))
                except (TypeError, ValueError):
                    pass

        if not ttft_samples:
            logger.debug("Layer18: no timing data available", model=model_name)
            return {
                "layer": self.LAYER,
                "name": self.NAME,
                "tokens": 0,
                "skipped": True,
                "reason": "no_timing_data",
                "confidence": 0.0,
                "evidence": [],
            }

        # Compute statistics
        n = len(ttft_samples)
        mean_ttft = sum(ttft_samples) / n
        variance_ttft = sum((x - mean_ttft) ** 2 for x in ttft_samples) / n if n > 1 else 0.0
        std_ttft = math.sqrt(variance_ttft)
        cv_ttft = std_ttft / mean_ttft if mean_ttft > 0 else 0.0

        mean_tps = sum(tps_samples) / len(tps_samples) if tps_samples else None

        # KL distances against reference profiles
        kl_scores: dict[str, float] = {}
        for family, ref in _TIMING_REFS.items():
            kl = _kl_gaussian(mean_ttft, ref["ttft_ms_mean"], ref["ttft_ms_std"])
            kl_scores[family] = kl

        closest_family = min(kl_scores, key=kl_scores.get)
        min_kl = kl_scores[closest_family]

        # Confidence capped at 0.50
        raw_confidence = 1.0 / (1.0 + min_kl)
        confidence = min(round(raw_confidence, 3), 0.50)

        evidence = [
            f"mean_ttft={mean_ttft:.1f}ms, std={std_ttft:.1f}ms, cv={cv_ttft:.2f}",
            f"closest_family={closest_family}, kl_distance={min_kl:.3f}",
        ]
        if mean_tps is not None:
            evidence.append(f"mean_tps={mean_tps:.1f} tokens/s")

        logger.info(
            "Layer18/TimingSideChannel complete",
            model=model_name,
            ttft_samples=n,
            mean_ttft_ms=mean_ttft,
            closest_family=closest_family,
            kl_distance=min_kl,
            confidence=confidence,
        )

        return {
            "layer": self.LAYER,
            "name": self.NAME,
            "tokens": 0,
            "skipped": False,
            "ttft_samples": n,
            "mean_ttft_ms": round(mean_ttft, 2),
            "std_ttft_ms": round(std_ttft, 2),
            "mean_tps": round(mean_tps, 2) if mean_tps is not None else None,
            "cv_ttft": round(cv_ttft, 3),
            "closest_family": closest_family,
            "kl_distance": round(min_kl, 4),
            "confidence": confidence,
            "evidence": evidence,
        }


# ── Layer 19: Token Distribution Side-Channel ─────────────────────────────────

class Layer19TokenDistribution:
    """
    Zero-token layer. Re-analyzes response texts from prior layers.

    Computes:
      - avg_response_len, len_cv
      - repetition_rate (4-gram overlap)
      - stop_token_freq
      - Wasserstein-like distance against known distribution profiles
      - Confidence capped at 0.45 (distribution is weak/indirect evidence)
    """

    LAYER = 19
    NAME = "token_distribution"

    # Common stop patterns (end of response)
    _STOP_PATTERNS = (".", "\n", "?", "!")

    def run(
        self,
        adapter,                           # noqa: ARG002  (not used — zero-token layer)
        model_name: str,
        layer_results_so_far: list,
    ) -> dict:
        # Extract response texts from prior layer dicts
        texts: list[str] = []

        for lr in layer_results_so_far:
            raw = lr if isinstance(lr, dict) else (lr.to_dict() if hasattr(lr, "to_dict") else {})
            # Try multiple text fields that prior layers might populate
            for key in ("response_text", "response", "text", "content"):
                val = raw.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append(val)
                    break
            # Also check nested evidence entries that contain text
            for ev in raw.get("evidence", []):
                if isinstance(ev, str) and len(ev) > 20:
                    texts.append(ev)

        if not texts:
            logger.debug("Layer19: no response text data available", model=model_name)
            return {
                "layer": self.LAYER,
                "name": self.NAME,
                "tokens": 0,
                "skipped": True,
                "reason": "no_response_data",
                "confidence": 0.0,
                "evidence": [],
            }

        # Compute statistics
        n = len(texts)
        lengths = [len(t) for t in texts]
        avg_len = sum(lengths) / n
        std_len = math.sqrt(sum((x - avg_len) ** 2 for x in lengths) / n) if n > 1 else 0.0
        len_cv = std_len / avg_len if avg_len > 0 else 0.0

        rep_rate = _repetition_rate(texts)

        stop_count = sum(
            1 for t in texts if t.rstrip().endswith(self._STOP_PATTERNS)
        )
        stop_token_freq = stop_count / n

        # Wasserstein distances against reference profiles
        w_scores: dict[str, float] = {}
        for family, ref in _DIST_REFS.items():
            w = _wasserstein_1d(avg_len, ref["avg_len"])
            w_scores[family] = w

        closest_family = min(w_scores, key=w_scores.get)
        min_w = w_scores[closest_family]

        # Confidence capped at 0.45
        raw_confidence = 1.0 / (1.0 + min_w * 2.0)
        confidence = min(round(raw_confidence, 3), 0.45)

        evidence = [
            f"avg_response_len={avg_len:.1f} chars (n={n})",
            f"len_cv={len_cv:.3f}, repetition_rate={rep_rate:.4f}",
            f"stop_token_freq={stop_token_freq:.3f}",
            f"closest_family={closest_family}, wasserstein_distance={min_w:.4f}",
        ]

        logger.info(
            "Layer19/TokenDistribution complete",
            model=model_name,
            samples=n,
            avg_response_len=avg_len,
            closest_family=closest_family,
            wasserstein_distance=min_w,
            confidence=confidence,
        )

        return {
            "layer": self.LAYER,
            "name": self.NAME,
            "tokens": 0,
            "skipped": False,
            "samples": n,
            "avg_response_len": round(avg_len, 2),
            "std_response_len": round(std_len, 2),
            "len_cv": round(len_cv, 3),
            "repetition_rate": round(rep_rate, 4),
            "stop_token_freq": round(stop_token_freq, 3),
            "closest_family": closest_family,
            "wasserstein_distance": round(min_w, 4),
            "confidence": confidence,
            "evidence": evidence,
        }
