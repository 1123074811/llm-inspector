"""
judge/semantic_entailment.py — Local NLI-based semantic entailment judge.

Uses sentence-transformers cross-encoder for NLI when available;
falls back to lightweight cosine similarity, then to semantic_v2 rules.

Primary model:
    cross-encoder/nli-deberta-v3-base (Reimers & Gurevych 2019)
    URL: https://huggingface.co/cross-encoder/nli-deberta-v3-base

Fallback model:
    cross-encoder/nli-MiniLM2-L6-H768
    URL: https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768

Reference:
    Reimers, N. & Gurevych, I. (2019). Sentence-BERT.
    arXiv: https://arxiv.org/abs/1908.10084
"""
from __future__ import annotations

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# -- Availability flag --------------------------------------------------------

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False

# -- Singleton NLI model ------------------------------------------------------

_nli_model = None
_nli_model_name: str | None = None

# Default entailment threshold
# (Reimers & Gurevych 2019 report ~0.70 as reliable NLI entailment cutoff)
_DEFAULT_THRESHOLD = 0.70

# Cosine fallback threshold (word-overlap Jaccard is coarser — lower bar)
_COSINE_FALLBACK_THRESHOLD = 0.50

_PRIMARY_MODEL = "cross-encoder/nli-deberta-v3-base"
_FALLBACK_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


def _get_nli_model():
    """
    Lazy singleton: load cross-encoder NLI model.

    Tries primary model first; if loading fails, falls back to lighter model.
    Returns (model, model_name) or (None, None) on failure.
    """
    global _nli_model, _nli_model_name
    if _nli_model is not None:
        return _nli_model, _nli_model_name

    if not HAS_ST:
        return None, None

    try:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(_PRIMARY_MODEL)
        _nli_model_name = _PRIMARY_MODEL
        logger.info("NLI model loaded", model=_PRIMARY_MODEL)
    except Exception as e:
        logger.warning("Primary NLI model failed", model=_PRIMARY_MODEL, error=str(e))
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder(_FALLBACK_MODEL)
            _nli_model_name = _FALLBACK_MODEL
            logger.info("NLI fallback model loaded", model=_FALLBACK_MODEL)
        except Exception as e2:
            logger.warning("Fallback NLI model also failed", error=str(e2))
            _nli_model = None
            _nli_model_name = None

    return _nli_model, _nli_model_name


# -- Cosine (word-overlap Jaccard) fallback -----------------------------------

def _tokenize_simple(text: str) -> set[str]:
    """Lowercase word tokens."""
    return set(re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower()))


def _cosine_similarity_fallback(premise: str, hypothesis: str) -> float:
    """
    Lightweight word-overlap Jaccard similarity as fallback when
    sentence-transformers is unavailable.

    Returns value in [0, 1].
    """
    tok_p = _tokenize_simple(premise)
    tok_h = _tokenize_simple(hypothesis)
    if not tok_p or not tok_h:
        return 0.0
    intersection = len(tok_p & tok_h)
    union = len(tok_p | tok_h)
    return intersection / union if union > 0 else 0.0


# -- Main judge function -------------------------------------------------------

def semantic_entailment_judge(
    response: str,
    params: dict,
) -> tuple[bool, dict]:
    """
    Semantic entailment judge.

    Determines whether *response* entails / is semantically consistent with
    the reference answer.

    Args:
        response: Model response text (premise).
        params:
            expected_answer or reference — reference text (hypothesis, required)
            entailment_threshold         — NLI entailment score threshold
                                           (default 0.70; Reimers & Gurevych 2019)

    Returns:
        (passed, detail_dict)
    """
    reference = params.get("expected_answer") or params.get("reference")
    if not reference:
        return False, {
            "method": "semantic_entailment",
            "error": "missing_reference",
            "note": "params['expected_answer'] or params['reference'] is required",
        }

    threshold = float(params.get("entailment_threshold", _DEFAULT_THRESHOLD))

    # -- Path 1: sentence-transformers NLI cross-encoder ----------------------
    model, model_name = _get_nli_model()
    if model is not None:
        try:
            # Cross-encoder expects (premise, hypothesis)
            scores = model.predict([(response, reference)])
            # DeBERTa / MiniLM NLI models output [contradiction, neutral, entailment]
            # If raw probabilities are available use them; otherwise fall through
            if hasattr(scores, "__iter__"):
                arr = list(scores)
                if len(arr) == 3:
                    # labels: contradiction=0, neutral=1, entailment=2
                    entailment_score = float(arr[2])
                else:
                    entailment_score = float(arr[0])
            else:
                entailment_score = float(scores)

            passed = entailment_score >= threshold
            return passed, {
                "method": "semantic_entailment",
                "backend": "nli",
                "model": model_name,
                "score": entailment_score,
                "threshold": threshold,
                "passed": passed,
            }
        except Exception as e:
            logger.warning("NLI cross-encoder inference failed", error=str(e))
            # Fall through to cosine fallback

    # -- Path 2: word-overlap cosine fallback ---------------------------------
    if not HAS_ST:
        cosine_score = _cosine_similarity_fallback(response, reference)
        eff_threshold = _COSINE_FALLBACK_THRESHOLD
        passed = cosine_score >= eff_threshold
        return passed, {
            "method": "semantic_entailment",
            "backend": "cosine",
            "score": cosine_score,
            "threshold": eff_threshold,
            "passed": passed,
            "note": "sentence-transformers not installed; using word-overlap Jaccard",
        }

    # -- Path 3: semantic_v2 rule fallback ------------------------------------
    try:
        from app.judge.semantic_v2 import semantic_judge_v2
        rule_params = dict(params)
        rule_params.setdefault("reference_answer", reference)
        sv2_passed, sv2_detail = semantic_judge_v2(response, rule_params)
        return sv2_passed, {
            "method": "semantic_entailment",
            "backend": "semantic_v2_fallback",
            "score": sv2_detail.get("score"),
            "threshold": threshold,
            "passed": sv2_passed,
            "semantic_v2_detail": sv2_detail,
        }
    except Exception as e:
        logger.warning("semantic_v2 fallback failed in semantic_entailment_judge", error=str(e))
        return False, {
            "method": "semantic_entailment",
            "backend": "error",
            "error": str(e),
            "passed": False,
        }
