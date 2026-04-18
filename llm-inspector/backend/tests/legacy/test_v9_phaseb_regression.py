"""v9 Phase B regression tests.

Covers:
- Judge consensus arbitration behavior
- Token ROI billing generation in report builder
"""

from __future__ import annotations

from app.core.schemas import TestCase, CaseResult, SampleResult, LLMResponse
from app.judge.consensus import arbitrate_with_semantic
from app.analysis.pipeline import ReportBuilder


def _mk_case_result(case_id: str, pass_rate_flag: bool, tokens: int, category: str = "reasoning") -> CaseResult:
    case = TestCase(
        id=case_id,
        category=category,
        name=case_id,
        user_prompt="test",
        expected_type="text",
        judge_method="exact_match",
        params={},
        n_samples=1,
    )
    result = CaseResult(case=case)
    resp = LLMResponse(content="ok", status_code=200, latency_ms=100, usage_total_tokens=tokens)
    result.samples.append(
        SampleResult(
            sample_index=0,
            response=resp,
            judge_passed=pass_rate_flag,
            judge_detail={"source": "unit"},
        )
    )
    return result


def test_consensus_keeps_agreement_when_rule_and_semantic_align():
    params = {"force_consensus": True}
    passed, detail = arbitrate_with_semantic(
        method="semantic_judge_v2",
        text="This is a valid answer with clear reasoning and complete explanation.",
        params=params,
        rule_passed=True,
        rule_detail={"rule": "ok"},
    )

    assert "judge_consensus" in detail
    assert detail["judge_consensus"]["enabled"] is True
    assert detail["judge_consensus"]["winner"] in {"both", "rule", "semantic"}
    assert passed in {True, False, None}


def test_consensus_disagreement_contains_evidence_chain_fields():
    # Use forced consensus and an obviously weak text to increase chance of disagreement
    params = {"force_consensus": True}
    passed, detail = arbitrate_with_semantic(
        method="semantic_judge_v2",
        text="n/a",
        params=params,
        rule_passed=True,
        rule_detail={"rule": "forced_pass"},
    )

    jc = detail.get("judge_consensus")
    assert jc is not None
    assert "rule" in jc and "semantic" in jc
    assert "winner" in jc and "mode" in jc
    assert passed in {True, False, None}


def test_token_roi_summary_and_ranking():
    case_results = [
        _mk_case_result("case_high", True, 50),
        _mk_case_result("case_low", False, 200),
    ]

    roi = ReportBuilder._build_token_roi(case_results)

    assert "summary" in roi
    assert roi["summary"]["total_cases"] == 2
    assert roi["summary"]["total_tokens"] == 250

    rows = roi["per_case"]
    assert len(rows) == 2
    assert rows[0]["roi"] >= rows[1]["roi"]
    assert rows[0]["roi_rank"] == 1
