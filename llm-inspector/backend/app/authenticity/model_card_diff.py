"""
authenticity/model_card_diff.py — Compare Claimed vs Suspected Model Card.

Generates a side-by-side diff card for the report.
"""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field
from app.core.logging import get_logger

logger = get_logger(__name__)

_TAXONOMY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_data", "model_taxonomy.yaml")
_taxonomy_cache: dict | None = None


def _load_taxonomy() -> dict:
    global _taxonomy_cache
    if _taxonomy_cache is None:
        try:
            with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            _taxonomy_cache = {}
            for item in (data if isinstance(data, list) else data.get("families", [])):
                if isinstance(item, dict) and "family" in item:
                    _taxonomy_cache[item["family"]] = item
        except Exception as e:
            logger.warning("Could not load model_taxonomy.yaml", error=str(e))
            _taxonomy_cache = {}
    return _taxonomy_cache


@dataclass
class ModelCardField:
    """A single field in a model card comparison."""
    field_name: str
    claimed_value: str | None
    suspected_value: str | None
    match: bool | None = None  # True=match, False=mismatch, None=unknown


@dataclass
class ModelCardDiff:
    """Side-by-side comparison of claimed vs suspected actual model."""
    claimed_model: str
    suspected_model: str | None
    wrapper_probability: float
    risk_level: str
    fields: list[ModelCardField] = field(default_factory=list)
    collision_snippets: list[str] = field(default_factory=list)
    evidence_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "claimed_model": self.claimed_model,
            "suspected_model": self.suspected_model,
            "wrapper_probability": self.wrapper_probability,
            "risk_level": self.risk_level,
            "fields": [
                {
                    "field": f.field_name,
                    "claimed": f.claimed_value,
                    "suspected": f.suspected_value,
                    "match": f.match,
                }
                for f in self.fields
            ],
            "collision_snippets": self.collision_snippets,
            "evidence_summary": self.evidence_summary,
        }


def build_model_card_diff(
    claimed_model: str,
    suspected_model: str | None,
    wrapper_probability: float,
    risk_level: str,
    evidence_list: list[dict],
) -> ModelCardDiff:
    """
    Build a ModelCardDiff from taxonomy data and evidence.
    """
    taxonomy = _load_taxonomy()

    # Determine family keys
    claimed_family = _find_family(claimed_model, taxonomy)
    suspected_family = _find_family(suspected_model or "", taxonomy)

    claimed_info = taxonomy.get(claimed_family, {}) if claimed_family else {}
    suspected_info = taxonomy.get(suspected_family, {}) if suspected_family else {}

    fields = []

    # Knowledge cutoff
    claimed_cutoff = claimed_info.get("knowledge_cutoff_claims") or claimed_info.get("knowledge_cutoff")
    suspected_cutoff = suspected_info.get("knowledge_cutoff_claims") or suspected_info.get("knowledge_cutoff")
    if claimed_cutoff or suspected_cutoff:
        fields.append(ModelCardField(
            "knowledge_cutoff",
            str(claimed_cutoff) if claimed_cutoff else "unknown",
            str(suspected_cutoff) if suspected_cutoff else "unknown",
            match=(claimed_cutoff == suspected_cutoff) if (claimed_cutoff and suspected_cutoff) else None,
        ))

    # Vendor
    claimed_vendor = claimed_info.get("vendor") or claimed_info.get("provider")
    suspected_vendor = suspected_info.get("vendor") or suspected_info.get("provider")
    if claimed_vendor or suspected_vendor:
        fields.append(ModelCardField(
            "vendor",
            str(claimed_vendor) if claimed_vendor else "unknown",
            str(suspected_vendor) if suspected_vendor else "unknown",
            match=(claimed_vendor == suspected_vendor) if (claimed_vendor and suspected_vendor) else None,
        ))

    # Typical TTFT
    claimed_ttft = claimed_info.get("typical_ttft_ms")
    suspected_ttft = suspected_info.get("typical_ttft_ms")
    if claimed_ttft or suspected_ttft:
        fields.append(ModelCardField(
            "typical_ttft_ms",
            str(claimed_ttft) if claimed_ttft else "unknown",
            str(suspected_ttft) if suspected_ttft else "unknown",
            match=None,  # Can't compare without measurement
        ))

    # Refusal style
    claimed_refusal = claimed_info.get("refusal_style") or (claimed_info.get("rejection_patterns") or [""])[0]
    suspected_refusal = suspected_info.get("refusal_style") or (suspected_info.get("rejection_patterns") or [""])[0]
    if claimed_refusal or suspected_refusal:
        fields.append(ModelCardField(
            "refusal_style",
            str(claimed_refusal)[:80] if claimed_refusal else "unknown",
            str(suspected_refusal)[:80] if suspected_refusal else "unknown",
            match=(claimed_refusal == suspected_refusal) if (claimed_refusal and suspected_refusal) else None,
        ))

    # Collect collision snippets
    collision_snippets = [
        e.get("raw_snippet", "")
        for e in evidence_list
        if e.get("type") == "identity_collision" and e.get("raw_snippet")
    ]

    n_contradicting = sum(1 for e in evidence_list if e.get("direction") == "contradicts_claim")
    n_supporting = sum(1 for e in evidence_list if e.get("direction") == "supports_claim")
    evidence_summary = (
        f"{n_contradicting} 条反对证据 / {n_supporting} 条支持证据 / "
        f"包装风险 P={wrapper_probability:.2f} ({risk_level})"
    )

    return ModelCardDiff(
        claimed_model=claimed_model,
        suspected_model=suspected_model,
        wrapper_probability=wrapper_probability,
        risk_level=risk_level,
        fields=fields,
        collision_snippets=collision_snippets[:5],  # limit to 5
        evidence_summary=evidence_summary,
    )


def _find_family(model_name: str, taxonomy: dict) -> str | None:
    """Find the taxonomy family key that best matches model_name."""
    if not model_name:
        return None
    lower = model_name.lower()
    # Direct family name match
    for fam in taxonomy:
        if fam in lower:
            return fam
    # Check official names in taxonomy
    for fam, info in taxonomy.items():
        names = info.get("official_names", []) or []
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name.lower() in lower or lower in name.lower():
                return fam
    return None
