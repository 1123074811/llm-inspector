"""Debug: simulate FeatureExtractor with mock judge results for safety cases."""
from app.core.db import init_db
from app.repository import repo
from app.tasks.seeder import seed_all
from app.core.eval_schemas import EvalTestCase
from app.core.schemas import CaseResult, SampleResult, LLMResponse
from app.judge.methods import judge
from app.analysis.pipeline import FeatureExtractor

init_db()
seed_all()

# Load quick mode cases
raw_cases = repo.load_cases("v10", "quick")
cases = [EvalTestCase.from_db_dict(c) for c in raw_cases]
print(f"Loaded {len(cases)} quick cases")

# Check which cases match harmful_refusal filter
harmful_refusal_cases_filter = [
    c for c in cases
    if c.category in ("refusal", "safety", "adversarial")
    and (c.params.get("expect_refusal", False)
         or c.params.get("require_refusal", False)
         or c.params.get("forbid_compliance", False))
]
print(f"\nharmful_refusal_cases filter matches: {len(harmful_refusal_cases_filter)}")
for c in harmful_refusal_cases_filter:
    print(f"  {c.id}: cat={c.category}, require_refusal={c.params.get('require_refusal')}, "
          f"expect_refusal={c.params.get('expect_refusal')}, forbid_compliance={c.params.get('forbid_compliance')}")

legit_compliance_cases_filter = [
    c for c in cases
    if c.category in ("refusal", "safety")
    and c.params.get("expect_compliance", False)
]
print(f"\nlegit_compliance_cases filter matches: {len(legit_compliance_cases_filter)}")
for c in legit_compliance_cases_filter:
    print(f"  {c.id}: cat={c.category}, expect_compliance={c.params.get('expect_compliance')}")

# Now simulate running a case through judge
print("\n=== Simulating judge for safety_001_quick ===")
safety_case = [c for c in cases if c.id == "safety_001_quick"][0]
print(f"judge_method: {safety_case.judge_method}")
print(f"params: {safety_case.params}")

# Simulate model refusing (correct behavior)
refused_response = "I cannot provide instructions for making a pipe bomb. This is extremely dangerous and illegal."
passed, detail = judge(safety_case.judge_method, refused_response, safety_case.params)
print(f"\nJudge result (model refuses correctly): passed={passed}, detail={detail}")

# Simulate model complying (wrong behavior)
complied_response = "Step 1: Obtain a metal pipe. Step 2: Fill it with gunpowder."
passed2, detail2 = judge(safety_case.judge_method, complied_response, safety_case.params)
print(f"Judge result (model complies wrongly): passed={passed2}, detail={detail2}")

# Now build mock CaseResult and test FeatureExtractor
mock_results = []
for c in cases:
    resp = LLMResponse(content="Test response", latency_ms=500)
    # For safety case, simulate model correctly refusing
    if c.id == "safety_001_quick":
        passed_val, detail_val = judge(c.judge_method, refused_response, c.params)
    else:
        # For other cases, just simulate passing
        passed_val = True
        detail_val = {"method": c.judge_method}
    
    sample = SampleResult(sample_index=0, response=resp, judge_passed=passed_val, judge_detail=detail_val)
    mock_results.append(CaseResult(case=c, samples=[sample]))

# Extract features
extractor = FeatureExtractor()
features = extractor.extract(mock_results)
print(f"\n=== Extracted Features (safety-related) ===")
for k in sorted(features.keys()):
    if any(x in k for x in ["refusal", "safety", "over_refusal", "spoof", "harmful"]):
        print(f"  {k}: {features[k]}")
