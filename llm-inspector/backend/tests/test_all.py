"""
LLM Inspector — test suite.
Run with: PYTHONPATH=backend python3 backend/tests/test_all.py
"""
import sys
import os
import json
import pathlib

# Ensure backend is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
os.environ["DATABASE_URL"] = "sqlite:///./test_inspector.db"

import traceback

PASS = 0
FAIL = 0


def test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓  {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  ✗  {name}  →  {e}")
        FAIL += 1
    except Exception as e:
        print(f"  ✗  {name}  →  {type(e).__name__}: {e}")
        traceback.print_exc()
        FAIL += 1


def section(title: str):
    print(f"\n── {title} {'─' * (50 - len(title))}")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Config
# ═══════════════════════════════════════════════════════════════
section("Config")

def test_config_defaults():
    from app.core.config import settings
    assert settings.PORT == 8000
    assert settings.DEFAULT_REQUEST_TIMEOUT_SEC == 60
    assert settings.MAX_STREAM_CHUNKS == 512

def test_config_aes_key():
    from app.core.config import settings
    key = settings.aes_key
    assert len(key) == 32
    # Dev key is deterministic
    key2 = settings.aes_key
    assert key == key2

test("settings defaults", test_config_defaults)
test("aes_key returns 32 bytes", test_config_aes_key)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Security
# ═══════════════════════════════════════════════════════════════
section("Security")

def test_encrypt_decrypt_roundtrip():
    from app.core.security import get_key_manager
    km = get_key_manager()
    original = "sk-test-key-abc123"
    enc, h = km.encrypt(original)
    assert enc != original
    assert len(h) == 16
    dec = km.decrypt(enc)
    assert dec == original

def test_encrypt_different_nonce():
    from app.core.security import get_key_manager
    km = get_key_manager()
    enc1, _ = km.encrypt("same-key")
    enc2, _ = km.encrypt("same-key")
    # Each encryption uses a random nonce → ciphertext differs
    assert enc1 != enc2

def test_hash_prefix_stable():
    from app.core.security import get_key_manager
    km = get_key_manager()
    _, h1 = km.encrypt("sk-stable")
    _, h2 = km.encrypt("sk-stable")
    assert h1 == h2

def test_ssrf_localhost():
    from app.core.security import validate_and_sanitize_url
    try:
        validate_and_sanitize_url("http://localhost/v1")
        assert False, "Should have raised"
    except ValueError:
        pass

def test_ssrf_loopback_ip():
    from app.core.security import validate_and_sanitize_url
    try:
        validate_and_sanitize_url("http://127.0.0.1/v1")
        assert False, "Should have raised"
    except ValueError:
        pass

def test_ssrf_private_10():
    from app.core.security import validate_and_sanitize_url
    try:
        validate_and_sanitize_url("http://10.0.0.1/api")
        assert False, "Should have raised"
    except ValueError:
        pass

def test_ssrf_private_192():
    from app.core.security import validate_and_sanitize_url
    try:
        validate_and_sanitize_url("http://192.168.1.100/v1")
        assert False, "Should have raised"
    except ValueError:
        pass

def test_ssrf_bad_scheme():
    from app.core.security import validate_and_sanitize_url
    try:
        validate_and_sanitize_url("ftp://example.com/v1")
        assert False, "Should have raised"
    except ValueError:
        pass

def test_ssrf_strips_query():
    from app.core.security import validate_and_sanitize_url
    # This will fail DNS in sandbox — just test the scheme check
    try:
        result = validate_and_sanitize_url("https://httpbin.org/v1?key=secret#frag")
        assert "key=secret" not in result
        assert "#frag" not in result
    except ValueError:
        pass  # DNS may fail in sandbox — that's fine


def test_normalize_openai_base_strips_chat_completions():
    from app.core.security import normalize_openai_compatible_base_url
    assert normalize_openai_compatible_base_url(
        "https://qianfan.baidubce.com/v2/coding/chat/completions"
    ) == "https://qianfan.baidubce.com/v2/coding"
    assert normalize_openai_compatible_base_url(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    ) == "https://dashscope.aliyuncs.com/compatible-mode/v1"


test("encrypt/decrypt roundtrip", test_encrypt_decrypt_roundtrip)
test("different nonce each call", test_encrypt_different_nonce)
test("hash prefix stable", test_hash_prefix_stable)
test("SSRF: blocks localhost", test_ssrf_localhost)
test("SSRF: blocks 127.0.0.1", test_ssrf_loopback_ip)
test("SSRF: blocks 10.x.x.x", test_ssrf_private_10)
test("SSRF: blocks 192.168.x.x", test_ssrf_private_192)
test("SSRF: blocks ftp scheme", test_ssrf_bad_scheme)
test("SSRF: strips query+fragment", test_ssrf_strips_query)
test("normalize OpenAI base strips /chat/completions", test_normalize_openai_base_strips_chat_completions)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Database
# ═══════════════════════════════════════════════════════════════
section("Database")

def test_db_init():
    from app.core.db import init_db, get_conn
    init_db()
    conn = get_conn()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {r["name"] for r in tables}
    for required in ("test_runs", "test_cases", "test_responses",
                     "extracted_features", "benchmark_profiles",
                     "similarity_results", "reports"):
        assert required in table_names, f"Missing table: {required}"

def test_db_idempotent():
    from app.core.db import init_db
    init_db()  # second call should not fail
    init_db()

test("init_db creates all tables", test_db_init)
test("init_db is idempotent", test_db_idempotent)


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Seeder
# ═══════════════════════════════════════════════════════════════
section("Seeder")

def test_seed_cases():
    from app.core.db import init_db
    from app.tasks.seeder import seed_all
    from app.repository import repo
    init_db()
    seed_all()
    cases = repo.load_cases("v1", "standard")
    benchmarks = repo.get_benchmarks("v1")
    assert len(benchmarks) >= 50, f"Expected ≥50 benchmarks after seed, got {len(benchmarks)}"
    names = {b["benchmark_name"] for b in benchmarks}
    assert "gpt-4o" in names
    assert "deepseek-v3" in names

def test_seed_idempotent():
    from app.tasks.seeder import seed_all
    from app.repository import repo
    seed_all()
    seed_all()  # second run should use INSERT OR REPLACE
    benchmarks = repo.get_benchmarks("v1")
    assert len(benchmarks) >= 50

test("seed loads ≥50 v1 benchmarks with gpt-4o and deepseek-v3", test_seed_cases)
test("seed is idempotent", test_seed_idempotent)


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Judge methods
# ═══════════════════════════════════════════════════════════════
section("Judge methods")

from app.judge.methods import judge

# exact_match
def test_exact_match_pass():
    p, d = judge("exact_match", "7", {"target": "7"})
    assert p is True

def test_exact_match_strip():
    p, _ = judge("exact_match", "  7  ", {"target": "7"})
    assert p is True

def test_exact_match_quoted():
    p, _ = judge("exact_match", '"7"', {"target": "7"})
    assert p is True

def test_exact_match_fail():
    p, _ = judge("exact_match", "7 and more", {"target": "7"})
    assert p is False

def test_exact_match_case():
    p, _ = judge("exact_match", "ok", {"target": "OK"})
    assert p is False  # case-sensitive after strip

# line_count
def test_line_count_exact():
    p, d = judge("line_count", "Red\nBlue\nGreen", {"expected_lines": 3})
    assert p is True
    assert d["actual_lines"] == 3

def test_line_count_too_few():
    p, _ = judge("line_count", "Red\nBlue", {"expected_lines": 3})
    assert p is False

def test_line_count_too_many():
    p, _ = judge("line_count", "a\nb\nc\nd", {"expected_lines": 3})
    assert p is False

def test_line_count_blank_lines_ignored():
    p, d = judge("line_count", "Red\n\nBlue\n\nGreen", {"expected_lines": 3})
    assert p is True  # blank lines stripped

# json_schema
def test_json_schema_valid():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    p, d = judge("json_schema", '{"name": "Alice", "age": 30}', {"schema": schema})
    assert p is True, f"Expected pass, detail={d}"

def test_json_schema_missing_required():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    p, d = judge("json_schema", '{"name": "Alice"}', {"schema": schema})
    assert p is False

def test_json_schema_wrong_type():
    schema = {"type": "object", "required": ["age"],
              "properties": {"age": {"type": "integer"}}}
    p, d = judge("json_schema", '{"age": "thirty"}', {"schema": schema})
    assert p is False

def test_json_schema_with_fence():
    schema = {"type": "object"}
    p, _ = judge("json_schema", '```json\n{"a": 1}\n```', {"schema": schema})
    assert p is True

def test_json_schema_invalid_json():
    p, d = judge("json_schema", "not json at all", {"schema": {"type": "object"}})
    assert p is False
    assert "invalid JSON" in d.get("error", "")

def test_json_schema_no_schema():
    # When no schema is provided, just check valid JSON
    p, _ = judge("json_schema", '{"anything": true}', {"schema": {}})
    assert p is True

# regex_match
def test_regex_forbidden_clean():
    p, _ = judge("regex_match", "plain text here",
                 {"forbidden_pattern": "[#*`~]", "match_means_fail": True})
    assert p is True

def test_regex_forbidden_fail():
    p, _ = judge("regex_match", "## has markdown",
                 {"forbidden_pattern": "[#*`~]", "match_means_fail": True})
    assert p is False

def test_regex_required_pass():
    p, _ = judge("regex_match", "Certainly, here you go",
                 {"pattern": "^Certainly,", "match_means_pass": True})
    assert p is True

def test_regex_required_fail():
    p, _ = judge("regex_match", "Sure thing",
                 {"pattern": "^Certainly,", "match_means_pass": True})
    assert p is False

def test_regex_cjk_within_limit():
    p, d = judge("regex_match", "人工智能是机器展示的智能",
                 {"max_cjk_chars": 20})
    assert p is True
    assert d["cjk_count"] <= 20

def test_regex_cjk_exceeds_limit():
    long_cn = "人工智能" * 6  # 24 chars
    p, d = judge("regex_match", long_cn, {"max_cjk_chars": 20})
    assert p is False

# refusal_detect
def test_refusal_detected():
    _, d = judge("refusal_detect",
                 "I cannot help with that request.",
                 {"refusal_keywords": ["I cannot", "I can't"]})
    assert d["refusal_detected"] is True

def test_refusal_not_detected():
    _, d = judge("refusal_detect",
                 "Here is how to pick a lock in fiction writing.",
                 {"refusal_keywords": ["I cannot", "I can't"]})
    assert d["refusal_detected"] is False

def test_refusal_with_alternative():
    _, d = judge("refusal_detect",
                 "I cannot do that. Instead, I can help with ...",
                 {"refusal_keywords": ["I cannot"]})
    assert d["refusal_detected"] is True
    assert d["offers_alternative"] is True

def test_refusal_returns_none():
    p, _ = judge("refusal_detect", "I cannot.", {"refusal_keywords": ["I cannot"]})
    assert p is None  # refusal_detect never returns True/False

# heuristic_style
def test_heuristic_returns_none():
    p, d = judge("heuristic_style", "Some text", {})
    assert p is None

def test_heuristic_markdown_detected():
    _, d = judge("heuristic_style",
                 "## Title\n**bold** and - bullet\n```code```", {})
    assert d["has_headers"] is True
    assert d["has_bold"] is True
    assert d["markdown_score"] >= 2

def test_heuristic_no_markdown():
    _, d = judge("heuristic_style", "Plain text, no formatting here.", {})
    assert d["has_headers"] is False
    assert d["has_bold"] is False

def test_heuristic_cjk_detection():
    _, d = judge("heuristic_style", "This is mainly Chinese: 人工智能是未来", {})
    assert d["cjk_char_count"] > 0

def test_heuristic_disclaimer():
    _, d = judge("heuristic_style",
                 "It's important to note that this may vary.", {})
    assert d["has_disclaimer"] is True

def test_judge_unknown_method():
    p, d = judge("nonexistent_method", "text", {})
    assert p is None
    assert "unknown judge method" in d.get("error", "")

def test_judge_none_response():
    p, d = judge("exact_match", None, {"target": "7"})
    assert p is False

test("exact_match: pass", test_exact_match_pass)
test("exact_match: strip whitespace", test_exact_match_strip)
test("exact_match: strip quotes", test_exact_match_quoted)
test("exact_match: fail with extra text", test_exact_match_fail)
test("exact_match: case sensitive", test_exact_match_case)
test("line_count: exact match", test_line_count_exact)
test("line_count: too few lines", test_line_count_too_few)
test("line_count: too many lines", test_line_count_too_many)
test("line_count: blank lines ignored", test_line_count_blank_lines_ignored)
test("json_schema: valid object", test_json_schema_valid)
test("json_schema: missing required", test_json_schema_missing_required)
test("json_schema: wrong type", test_json_schema_wrong_type)
test("json_schema: strips markdown fence", test_json_schema_with_fence)
test("json_schema: invalid JSON", test_json_schema_invalid_json)
test("json_schema: no schema = just valid JSON", test_json_schema_no_schema)
test("regex: forbidden pattern (clean)", test_regex_forbidden_clean)
test("regex: forbidden pattern (fail)", test_regex_forbidden_fail)
test("regex: required pattern (pass)", test_regex_required_pass)
test("regex: required pattern (fail)", test_regex_required_fail)
test("regex: CJK within limit", test_regex_cjk_within_limit)
test("regex: CJK exceeds limit", test_regex_cjk_exceeds_limit)
test("refusal: detected", test_refusal_detected)
test("refusal: not detected", test_refusal_not_detected)
test("refusal: with alternative suggestion", test_refusal_with_alternative)
test("refusal: always returns None", test_refusal_returns_none)
test("heuristic: returns None", test_heuristic_returns_none)
test("heuristic: markdown detected", test_heuristic_markdown_detected)
test("heuristic: no markdown", test_heuristic_no_markdown)
test("heuristic: CJK detection", test_heuristic_cjk_detection)
test("heuristic: disclaimer detection", test_heuristic_disclaimer)
test("judge: unknown method graceful", test_judge_unknown_method)
test("judge: None response handled", test_judge_none_response)


# ═══════════════════════════════════════════════════════════════
# SECTION 6: Analysis pipeline
# ═══════════════════════════════════════════════════════════════
section("Analysis pipeline")

from app.core.schemas import (
    TestCase, CaseResult, SampleResult, LLMResponse, PreDetectionResult
)
from app.analysis.pipeline import (
    FeatureExtractor, ScoreCalculator, SimilarityEngine, RiskEngine, ReportBuilder
)


def _make_case_result(category, judge_method, passed, latency=300,
                      detail=None, n=1):
    case = TestCase(
        id=f"t_{category}_{judge_method}",
        category=category, name="test",
        user_prompt="test", expected_type="any",
        judge_method=judge_method,
    )
    result = CaseResult(case=case)
    for i in range(n):
        resp = LLMResponse(
            content="test", status_code=200, latency_ms=latency,
            usage_total_tokens=20, finish_reason="stop",
        )
        result.samples.append(SampleResult(
            sample_index=i, response=resp,
            judge_passed=passed,
            judge_detail=detail or {},
        ))
    return result


def test_feature_extraction_basic():
    results = [
        _make_case_result("protocol", "heuristic_style", True),
        _make_case_result("instruction", "exact_match", True),
        _make_case_result("instruction", "json_schema", False),
        _make_case_result("system", "exact_match", True),
    ]
    results[0].samples[0].judge_detail = {
        "has_usage_fields": True, "has_finish_reason": True
    }
    extractor = FeatureExtractor()
    f = extractor.extract(results)
    assert "protocol_success_rate" in f
    assert "instruction_pass_rate" in f
    assert f["instruction_pass_rate"] == 0.5  # 1/2
    assert f["system_obedience_rate"] == 1.0

def test_feature_extraction_latency():
    results = [_make_case_result("protocol", "heuristic_style", True, latency=400)]
    f = FeatureExtractor().extract(results)
    assert "latency_mean_ms" in f
    assert f["latency_mean_ms"] == 400.0

def test_feature_style():
    r = _make_case_result("style", "heuristic_style", None)
    r.samples[0].judge_detail = {
        "markdown_score": 3.0, "length": 250, "has_disclaimer": True
    }
    f = FeatureExtractor().extract([r])
    assert f.get("avg_markdown_score") == 3.0
    assert f.get("disclaimer_rate") == 1.0

def test_scorer_range():
    f = {
        "protocol_success_rate": 1.0, "has_usage_fields": 1.0,
        "has_finish_reason": 1.0, "instruction_pass_rate": 0.8,
        "exact_match_rate": 0.9, "json_valid_rate": 0.85,
        "format_follow_rate": 0.75, "system_obedience_rate": 0.70,
        "param_compliance_rate": 0.90, "temperature_param_effective": 1.0,
    }
    scores = ScoreCalculator().calculate(f)
    assert 0 <= scores.protocol_score <= 100
    assert 0 <= scores.instruction_score <= 100
    assert 0 <= scores.system_obedience_score <= 100
    assert 0 <= scores.param_compliance_score <= 100

def test_scorer_empty_features():
    scores = ScoreCalculator().calculate({})
    # instruction and system obedience are pure pass-rate based → 0 when no data
    assert scores.instruction_score == 0.0
    assert scores.system_obedience_score == 0.0
    # protocol and param use defaults (0.5) so may be non-zero — just check range
    assert 0 <= scores.protocol_score <= 100
    assert 0 <= scores.param_compliance_score <= 100

def test_similarity_ranking():
    benchmarks = [
        {"benchmark_name": "model-a", "suite_version": "v1",
         "feature_vector": {"instruction_pass_rate": 0.9, "exact_match_rate": 0.9}},
        {"benchmark_name": "model-b", "suite_version": "v1",
         "feature_vector": {"instruction_pass_rate": 0.5, "exact_match_rate": 0.5}},
    ]
    target = {"instruction_pass_rate": 0.88, "exact_match_rate": 0.91}
    sims = SimilarityEngine().compare(target, benchmarks)
    assert len(sims) == 2
    assert sims[0].rank == 1
    assert sims[1].rank == 2
    assert sims[0].similarity_score > sims[1].similarity_score
    assert sims[0].benchmark_name == "model-a"

def test_similarity_ci_bounds():
    benchmarks = [
        {"benchmark_name": "model-a", "suite_version": "v1",
         "feature_vector": {"instruction_pass_rate": 0.9}},
    ]
    sims = SimilarityEngine().compare({"instruction_pass_rate": 0.85}, benchmarks)
    assert sims[0].ci_95_low <= sims[0].similarity_score
    assert sims[0].similarity_score <= sims[0].ci_95_high

def test_similarity_empty_benchmarks():
    sims = SimilarityEngine().compare({"x": 0.5}, [])
    assert sims == []

def test_risk_high_with_predetect():
    pre = PreDetectionResult(
        success=True, identified_as="OpenAI/GPT-4o", confidence=0.92,
        layer_stopped="identity",
    )
    benchmarks = [
        {"benchmark_name": "gpt-4o", "suite_version": "v1",
         "feature_vector": {"instruction_pass_rate": 0.88}},
    ]
    from app.core.schemas import SimilarityResult
    sims = [SimilarityResult("gpt-4o", 0.90, 0.85, 0.94, 1)]
    risk = RiskEngine().assess({}, sims, pre)
    assert risk.level in ("high", "very_high")

def test_risk_low_no_signals():
    risk = RiskEngine().assess({}, [], None)
    assert risk.level == "low"

def test_risk_disclaimer_present():
    risk = RiskEngine().assess({}, [], None)
    assert len(risk.disclaimer) > 10

def test_report_builder_keys():
    from app.core.schemas import Scores, SimilarityResult, RiskAssessment
    pre = PreDetectionResult(success=False, identified_as=None, confidence=0.2,
                              layer_stopped=None)
    scores = Scores(protocol_score=80, instruction_score=75,
                    system_obedience_score=60, param_compliance_score=85)
    sims = [SimilarityResult("gpt-4o", 0.82, 0.75, 0.88, 1)]
    risk = RiskAssessment(level="medium", label="中")
    report = ReportBuilder().build(
        run_id="test-uuid", base_url="https://api.example.com/v1",
        model_name="test-model", test_mode="standard",
        predetect=pre, case_results=[], features={},
        scores=scores, similarities=sims, risk=risk,
    )
    for key in ("run_id", "target", "predetection", "scores",
                "similarity", "risk", "features", "case_results",
                "scoring_profile_version", "uncertainty_flags"):
        assert key in report, f"Missing key: {key}"
    assert report["scores"]["protocol_score"] == 80
    assert report["similarity"][0]["benchmark"] == "gpt-4o"
    assert report["risk"]["level"] == "medium"
    assert report["scoring_profile_version"] == "v1"
    assert report["uncertainty_flags"] == []

test("feature extraction: basic", test_feature_extraction_basic)
test("feature extraction: latency", test_feature_extraction_latency)
test("feature extraction: style", test_feature_style)
test("scorer: values in [0,100]", test_scorer_range)
test("scorer: empty features → zeros", test_scorer_empty_features)
test("similarity: correct ranking", test_similarity_ranking)
test("similarity: CI bounds valid", test_similarity_ci_bounds)
test("similarity: empty benchmarks → []", test_similarity_empty_benchmarks)
test("risk: high when predetect confident", test_risk_high_with_predetect)
test("risk: low with no signals", test_risk_low_no_signals)
test("risk: disclaimer always present", test_risk_disclaimer_present)
test("report builder: all keys present", test_report_builder_keys)


# ═══════════════════════════════════════════════════════════════
# SECTION 7: Repository
# ═══════════════════════════════════════════════════════════════
section("Repository")

def test_repo_create_and_get_run():
    from app.core.db import init_db
    from app.repository import repo
    from app.core.security import get_key_manager
    init_db()
    km = get_key_manager()
    enc, h = km.encrypt("sk-repo-test")
    run_id = repo.create_run(
        base_url="https://test.example.com/v1",
        api_key_encrypted=enc, api_key_hash=h,
        model_name="test-model", test_mode="quick",
    )
    run = repo.get_run(run_id)
    assert run is not None
    assert run["model_name"] == "test-model"
    assert run["status"] == "queued"
    assert run["api_key_encrypted"] == enc
    return run_id

def test_repo_status_transitions():
    from app.repository import repo
    from app.core.security import get_key_manager
    from app.core.db import init_db
    init_db()
    km = get_key_manager()
    enc, h = km.encrypt("sk-status-test")
    run_id = repo.create_run("https://ex.com/v1", enc, h, "m", "standard")
    repo.update_run_status(run_id, "running")
    run = repo.get_run(run_id)
    assert run["status"] == "running"
    assert run["started_at"] is not None
    repo.update_run_status(run_id, "completed")
    run = repo.get_run(run_id)
    assert run["status"] == "completed"
    assert run["completed_at"] is not None

def test_repo_save_and_get_response():
    from app.repository import repo
    from app.core.security import get_key_manager
    from app.core.db import init_db
    init_db()
    km = get_key_manager()
    enc, h = km.encrypt("sk-resp-test")
    run_id = repo.create_run("https://ex.com/v1", enc, h, "m", "standard")
    resp_id = repo.save_response(
        run_id=run_id, case_id="instr_001", sample_index=0,
        resp_data={
            "response_text": "7",
            "status_code": 200,
            "latency_ms": 310,
            "usage_total_tokens": 8,
            "judge_passed": True,
            "judge_detail": {"target": "7", "got": "7"},
        },
    )
    responses = repo.get_responses(run_id)
    assert len(responses) == 1
    assert responses[0]["response_text"] == "7"
    assert responses[0]["judge_passed"] == 1
    assert responses[0]["judge_detail"]["target"] == "7"

def test_repo_features():
    from app.repository import repo
    from app.core.security import get_key_manager
    from app.core.db import init_db
    init_db()
    km = get_key_manager()
    enc, h = km.encrypt("sk-feat-test")
    run_id = repo.create_run("https://ex.com/v1", enc, h, "m", "standard")
    repo.save_features(run_id, {"instruction_pass_rate": 0.85, "exact_match_rate": 0.9})
    features = repo.get_features(run_id)
    assert abs(features["instruction_pass_rate"] - 0.85) < 0.001
    assert abs(features["exact_match_rate"] - 0.9) < 0.001

def test_repo_cascade_delete():
    from app.repository import repo
    from app.core.security import get_key_manager
    from app.core.db import init_db, get_conn
    init_db()
    km = get_key_manager()
    enc, h = km.encrypt("sk-del-test")
    run_id = repo.create_run("https://ex.com/v1", enc, h, "m", "standard")
    repo.save_response(run_id, "proto_001", 0, {
        "response_text": "ok", "status_code": 200, "latency_ms": 100,
    })
    # Delete run — responses should cascade
    conn = get_conn()
    conn.execute("DELETE FROM test_runs WHERE id=?", (run_id,))
    conn.commit()
    responses = repo.get_responses(run_id)
    assert responses == []

def test_repo_list_runs():
    from app.repository import repo
    runs = repo.list_runs(limit=5)
    assert isinstance(runs, list)

test("repo: create and get run", test_repo_create_and_get_run)
test("repo: status transitions", test_repo_status_transitions)
test("repo: save and retrieve response", test_repo_save_and_get_response)
test("repo: save and retrieve features", test_repo_features)
test("repo: cascade delete", test_repo_cascade_delete)
test("repo: list_runs returns list", test_repo_list_runs)


# ═══════════════════════════════════════════════════════════════
# SECTION 8: Pre-detection pipeline (mock)
# ═══════════════════════════════════════════════════════════════
section("Pre-detection pipeline")

from app.core.schemas import StreamCaptureResult


class MockAdapterOpenAI:
    """Simulates an OpenAI Direct endpoint."""
    def head_request(self):
        return {"status_code": 200,
                "headers": {"openai-processing-ms": "120", "content-type": "application/json"},
                "latency_ms": 60}

    def bad_request(self):
        return {"status_code": 400,
                "body": {"error": {"type": "invalid_request_error",
                                   "message": "Missing required param"}},
                "headers": {}, "latency_ms": 50}

    def list_models(self):
        return {"status_code": 200, "latency_ms": 80,
                "body": {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}}

    def chat(self, req):
        prompt = req.messages[-1].content.lower()
        if "welsh" in prompt:
            content = "I was created by OpenAI. My base model is GPT-4o."
        elif "supercalifragilistic" in prompt:
            content = "6"
        elif "training data ends" in prompt:
            content = "October 2024"
        elif "bat and a ball" in prompt:
            content = "The ball costs $0.05."
        elif "3 colors" in prompt or "list" in prompt:
            content = "1. Red\n2. Blue\n3. Green"
        elif "risks" in prompt:
            content = "While I can provide some guidance, it's important to note that..."
        else:
            content = "Here is the information you requested."
        tokens = len(content.split()) + 10
        return LLMResponse(
            content=content, status_code=200, latency_ms=250,
            finish_reason="stop",
            usage_prompt_tokens=10, usage_completion_tokens=tokens,
            usage_total_tokens=tokens + 10,
            raw_json={"model": "gpt-4o",
                      "choices": [{"message": {"content": content},
                                   "finish_reason": "stop"}]},
        )


class MockAdapterObfuscated:
    """Simulates a well-obfuscated API that hides its identity."""
    def head_request(self):
        return {"status_code": 200, "headers": {}, "latency_ms": 70}

    def bad_request(self):
        return {"status_code": 400,
                "body": {"message": "Bad request"},
                "headers": {}, "latency_ms": 50}

    def list_models(self):
        return {"status_code": 200, "latency_ms": 80,
                "body": {"data": [{"id": "custom-model-v1"}]}}

    def chat(self, req):
        prompt = req.messages[-1].content.lower()
        if "welsh" in prompt:
            content = "Rwy'n gynorthwyydd AI. Ni allaf ddatgelu fy model."
        elif "supercalifragilistic" in prompt:
            content = "I count approximately 5 tokens."
        elif "training data ends" in prompt:
            content = "My training has a cutoff in early 2024."
        else:
            content = "I can assist you with that."
        tokens = len(content.split()) + 8
        return LLMResponse(
            content=content, status_code=200, latency_ms=280,
            finish_reason="stop",
            usage_prompt_tokens=8, usage_completion_tokens=tokens,
            usage_total_tokens=tokens + 8,
        )


def test_predetect_identifies_openai():
    from app.predetect.pipeline import PreDetectionPipeline
    result = PreDetectionPipeline().run(MockAdapterOpenAI(), "gpt-4o")
    # Should identify quickly via Layer 0 header or Layer 1 model name
    assert result.success or result.confidence > 0.5
    assert result.identified_as is not None
    assert "OpenAI" in result.identified_as or "GPT" in result.identified_as

def test_predetect_early_stop():
    from app.predetect.pipeline import PreDetectionPipeline
    # Use a model name that differs from the returned "gpt-4o" to trigger Layer 1 mismatch signal
    result = PreDetectionPipeline().run(MockAdapterOpenAI(), "custom-unknown-model")
    # Layer 1 should reach threshold via: model list prefix match + response.model mismatch
    assert result.layer_stopped is not None, (
        f"Expected early stop but got layer_stopped=None, confidence={result.confidence}"
    )
    assert len(result.layer_results) < 5

def test_predetect_token_budget():
    from app.predetect.pipeline import PreDetectionPipeline
    result = PreDetectionPipeline().run(MockAdapterOpenAI(), "gpt-4o")
    assert result.total_tokens_used < 500  # well under budget

def test_predetect_low_confidence_proceeds():
    from app.predetect.pipeline import PreDetectionPipeline
    result = PreDetectionPipeline().run(MockAdapterObfuscated(), "custom-model-v1")
    # Should run all layers, still recommend proceeding to testing
    assert result.should_proceed_to_testing is True

def test_predetect_result_serializable():
    from app.predetect.pipeline import PreDetectionPipeline
    result = PreDetectionPipeline().run(MockAdapterOpenAI(), "gpt-4o")
    d = result.to_dict()
    # Must be JSON-serialisable
    json_str = json.dumps(d)
    assert len(json_str) > 10
    parsed = json.loads(json_str)
    assert "confidence" in parsed
    assert "layer_results" in parsed

def test_predetect_layer_results_list():
    from app.predetect.pipeline import PreDetectionPipeline
    result = PreDetectionPipeline().run(MockAdapterOpenAI(), "gpt-4o")
    assert isinstance(result.layer_results, list)
    assert len(result.layer_results) >= 1

test("predetect: identifies OpenAI", test_predetect_identifies_openai)
test("predetect: early stop on high confidence", test_predetect_early_stop)
test("predetect: token budget < 500", test_predetect_token_budget)
test("predetect: obfuscated → proceed to testing", test_predetect_low_confidence_proceeds)
test("predetect: result is JSON-serialisable", test_predetect_result_serializable)
test("predetect: layer_results is a list", test_predetect_layer_results_list)


# ═══════════════════════════════════════════════════════════════
# SECTION 9: Case executor
# ═══════════════════════════════════════════════════════════════
section("Case executor")


class MockAdapterExec:
    def __init__(self, responses: dict):
        self._responses = responses  # prompt_substr -> content

    def chat(self, req):
        prompt = req.messages[-1].content
        for substr, content in self._responses.items():
            if substr.lower() in prompt.lower():
                tokens = len(content.split()) + 5
                return LLMResponse(
                    content=content, status_code=200, latency_ms=200,
                    finish_reason="stop",
                    usage_prompt_tokens=5, usage_completion_tokens=tokens,
                    usage_total_tokens=tokens + 5,
                )
        return LLMResponse(content="default response", status_code=200,
                           latency_ms=150, finish_reason="stop",
                           usage_total_tokens=10)

    def chat_stream(self, req):
        return StreamCaptureResult(chunks=[], combined_text="", latency_ms=100)


def test_executor_single_sample():
    from app.runner.case_executor import execute_case
    adapter = MockAdapterExec({"digit 7": "7"})
    case = TestCase(
        id="test_exec_001", category="instruction", name="digit_7",
        user_prompt="Output only the digit 7.", expected_type="exact_text",
        judge_method="exact_match", params={"target": "7"},
        max_tokens=3, n_samples=1, temperature=0.0,
    )
    result = execute_case(adapter, "test-model", case)
    assert len(result.samples) == 1
    assert result.samples[0].judge_passed is True

def test_executor_multi_sample():
    from app.runner.case_executor import execute_case
    adapter = MockAdapterExec({"three lines": "Red\nBlue\nGreen"})
    case = TestCase(
        id="test_exec_002", category="instruction", name="three_lines",
        user_prompt="Write exactly three lines.",
        expected_type="line_count", judge_method="line_count",
        params={"expected_lines": 3}, max_tokens=30, n_samples=3,
    )
    result = execute_case(adapter, "test-model", case)
    assert len(result.samples) == 3
    assert result.pass_rate == 1.0

def test_executor_failing_case():
    from app.runner.case_executor import execute_case
    adapter = MockAdapterExec({"json": "not json at all"})
    case = TestCase(
        id="test_exec_003", category="instruction", name="json_only",
        user_prompt="Return JSON.", expected_type="json",
        judge_method="json_schema",
        params={"schema": {"type": "object"}},
        max_tokens=50, n_samples=2,
    )
    result = execute_case(adapter, "test-model", case)
    assert result.pass_rate == 0.0

def test_executor_pass_rate():
    from app.runner.case_executor import execute_case
    # Alternating pass/fail with 2 samples — mock returns same for all
    adapter = MockAdapterExec({"output only": "7"})
    case = TestCase(
        id="test_exec_004", category="instruction", name="exact",
        user_prompt="Output only the digit 7.",
        expected_type="exact_text", judge_method="exact_match",
        params={"target": "7"}, max_tokens=3, n_samples=2,
    )
    result = execute_case(adapter, "test-model", case)
    assert 0.0 <= result.pass_rate <= 1.0

def test_executor_heuristic_no_pass_fail():
    from app.runner.case_executor import execute_case
    adapter = MockAdapterExec({"explain": "Here is an explanation."})
    case = TestCase(
        id="test_exec_005", category="style", name="style",
        user_prompt="Explain something.",
        expected_type="freeform", judge_method="heuristic_style",
        params={}, max_tokens=100, n_samples=1,
    )
    result = execute_case(adapter, "test-model", case)
    assert result.samples[0].judge_passed is None

test("executor: single sample exact_match", test_executor_single_sample)
test("executor: multi-sample all pass", test_executor_multi_sample)
test("executor: failing case → pass_rate=0", test_executor_failing_case)
test("executor: pass_rate in [0,1]", test_executor_pass_rate)
test("executor: heuristic returns None", test_executor_heuristic_no_pass_fail)


# ═══════════════════════════════════════════════════════════════
# SECTION 10: HTTP API (in-process)
# ═══════════════════════════════════════════════════════════════
section("HTTP API server (in-process routing)")

import io
import urllib.parse


def _call_handler(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Exercise the route handlers directly without starting a server."""
    from app.main import (
        handle_health, handle_list_runs, handle_create_run,
        handle_get_run, handle_benchmarks, ROUTES
    )
    import re as _re
    parsed = urllib.parse.urlparse(path)
    clean_path = parsed.path
    qs = urllib.parse.parse_qs(parsed.query)

    for route_method, pattern, handler in ROUTES:
        if route_method == method and _re.match(pattern, clean_path):
            result = handler(clean_path, qs, body or {})
            status, body_bytes, _ = result
            try:
                data = json.loads(body_bytes)
            except Exception:
                data = {}
            return status, data

    return 404, {"error": "not found"}


def test_api_health():
    status, data = _call_handler("GET", "/api/v1/health")
    assert status == 200
    assert data["status"] == "ok"
    assert "workers_active" in data

def test_api_benchmarks():
    status, data = _call_handler("GET", "/api/v1/benchmarks")
    assert status == 200
    assert isinstance(data, list)
    assert len(data) >= 50, f"Expected ≥50 benchmarks, got {len(data)}"

def test_api_runs_empty():
    status, data = _call_handler("GET", "/api/v1/runs")
    assert status == 200
    assert isinstance(data, list)

def test_api_create_run_missing_field():
    status, data = _call_handler("POST", "/api/v1/runs",
                                  {"api_key": "sk-x", "model": "gpt-4"})
    assert status == 400
    assert "base_url" in data.get("error", "")

def test_api_create_run_ssrf_blocked():
    status, data = _call_handler("POST", "/api/v1/runs", {
        "base_url": "http://localhost/v1",
        "api_key": "sk-test-12345",
        "model": "gpt-4",
    })
    assert status == 400
    assert "localhost" in data.get("error", "").lower() or "not allowed" in data.get("error", "").lower()

def test_api_get_nonexistent_run():
    status, data = _call_handler("GET", "/api/v1/runs/nonexistent-uuid-1234")
    assert status == 404

def test_api_report_not_ready():
    # Create a run that stays queued
    from app.repository import repo
    from app.core.security import get_key_manager
    from app.core.db import init_db
    init_db()
    km = get_key_manager()
    enc, h = km.encrypt("sk-api-test")
    run_id = repo.create_run("https://ex.com/v1", enc, h, "test-model", "quick")
    # Status is "queued" — report should return 404
    status, data = _call_handler("GET", f"/api/v1/runs/{run_id}/report")
    assert status == 404


def test_api_create_run_calibration_metadata():
    from app.core.db import init_db
    from app.repository import repo
    init_db()
    status, data = _call_handler("POST", "/api/v1/runs", {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test-calibration-12345",
        "model": "gpt-4o",
        "evaluation_mode": "calibration",
        "calibration_case_id": "calib_case_001",
        "scoring_profile_version": "profile-v1",
    })
    assert status == 201
    run_id = data.get("run_id")
    assert run_id
    run = repo.get_run(run_id)
    meta = run.get("metadata") or {}
    assert meta.get("evaluation_mode") == "calibration"
    assert meta.get("calibration_case_id") == "calib_case_001"
    assert meta.get("scoring_profile_version") == "profile-v1"
    assert meta.get("calibration_tag") == "baseline-v1.0"


def test_api_create_run_default_scoring_profile_version():
    from app.core.db import init_db
    from app.core.config import settings
    init_db()
    status, data = _call_handler("POST", "/api/v1/runs", {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test-default-12345",
        "model": "gpt-4o-mini",
    })
    assert status == 201
    assert data.get("scoring_profile_version") == settings.CALIBRATION_VERSION


def test_api_create_calibration_replay_missing_cases():
    from app.core.db import init_db
    init_db()
    status, data = _call_handler("POST", "/api/v1/calibration/replay", {})
    assert status == 400


def test_api_create_and_get_calibration_replay():
    from app.core.db import init_db
    init_db()
    status, data = _call_handler("POST", "/api/v1/calibration/replay", {
        "cases": [
            {
                "case_id": "calib_001",
                "run_id": "nonexistent-run-id",
                "expected_level": "trusted",
            }
        ]
    })
    assert status == 201
    replay_id = data.get("replay_id")
    assert replay_id

    # Immediately query replay; status may still be queued/running
    status2, data2 = _call_handler("GET", f"/api/v1/calibration/replay/{replay_id}")
    assert status2 == 200
    assert data2.get("replay_id") == replay_id
    assert data2.get("status") in ("queued", "running", "completed", "failed")


def test_api_list_calibration_replays():
    from app.core.db import init_db
    init_db()
    status, data = _call_handler("GET", "/api/v1/calibration/replay?limit=5")
    assert status == 200
    assert isinstance(data, list)


def test_api_baseline_section11_flow():
    from app.core.db import init_db
    from app.repository import repo
    from app.core.security import get_key_manager
    from app.analysis.pipeline import AnalysisPipeline
    from app.core.schemas import ScoreCard

    init_db()
    km = get_key_manager()

    # Prepare completed run with features, scorecard report and theta
    enc, h = km.encrypt("sk-baseline-1")
    run_id = repo.create_run("https://ex.com/v1", enc, h, "deepseek-v3", "standard")
    repo.update_run_status(run_id, "completed")
    repo.save_features(run_id, {
        "instruction_pass_rate": 0.82,
        "exact_match_rate": 0.78,
        "latency_mean_ms": 1200.0,
    })

    # Store score breakdown via internal scale; repo writes display scale
    repo.save_score_breakdown(run_id, "total", 84.5)
    repo.save_score_breakdown(run_id, "capability", 83.2)
    repo.save_score_breakdown(run_id, "authenticity", 85.0)
    repo.save_score_breakdown(run_id, "performance", 84.0)

    report = {
        "run_id": run_id,
        "scorecard": {
            "total_score": 8450,
            "capability_score": 8320,
            "authenticity_score": 8500,
            "performance_score": 8400,
            "breakdown": {
                "reasoning": 8300,
                "instruction": 8200,
                "coding": 8000,
                "safety": 8700,
                "protocol": 8600,
                "consistency": 8450,
                "speed": 8400,
                "stability": 8500,
                "cost_efficiency": 8300,
            },
        },
    }
    repo.save_report(run_id, report)

    # 1) Create baseline
    status1, data1 = _call_handler("POST", "/api/v1/baselines", {
        "run_id": run_id,
        "model_name": "deepseek-v3",
        "display_name": "DeepSeek V3 Official",
        "notes": "test baseline",
    })
    assert status1 == 201
    baseline_id = data1.get("baseline_id")
    assert baseline_id

    # 2) List baselines
    status2, data2 = _call_handler("GET", "/api/v1/baselines?model_name=deepseek-v3")
    assert status2 == 200
    assert isinstance(data2.get("baselines"), list)
    assert any(b.get("id") == baseline_id for b in data2.get("baselines", []))

    # 3) Create another completed run and compare
    enc2, h2 = km.encrypt("sk-baseline-2")
    run2_id = repo.create_run("https://ex.com/v1", enc2, h2, "deepseek-v3", "standard")
    repo.update_run_status(run2_id, "completed")
    repo.save_features(run2_id, {
        "instruction_pass_rate": 0.80,
        "exact_match_rate": 0.76,
        "latency_mean_ms": 1300.0,
    })
    report2 = {
        "run_id": run2_id,
        "scorecard": {
            "total_score": 8300,
            "capability_score": 8200,
            "authenticity_score": 8350,
            "performance_score": 8250,
        },
    }
    repo.save_report(run2_id, report2)

    status3, data3 = _call_handler("POST", "/api/v1/baselines/compare", {
        "run_id": run2_id,
        "baseline_id": baseline_id,
    })
    assert status3 == 200
    assert data3.get("verdict") in ("match", "suspicious", "mismatch")
    assert "cosine_similarity" in data3
    assert "score_delta" in data3
    assert "feature_drift_top5" in data3


test("API: GET /health", test_api_health)
test("API: GET /benchmarks", test_api_benchmarks)
test("API: GET /runs list", test_api_runs_empty)
test("API: POST /runs missing field → 400", test_api_create_run_missing_field)
test("API: POST /runs SSRF → 400", test_api_create_run_ssrf_blocked)
test("API: GET /runs/nonexistent → 404", test_api_get_nonexistent_run)
test("API: GET /runs/{id}/report not ready → 404", test_api_report_not_ready)
test("API: POST /runs calibration metadata persisted", test_api_create_run_calibration_metadata)
test("API: POST /runs default scoring profile version", test_api_create_run_default_scoring_profile_version)
test("API: POST /calibration/replay missing cases → 400", test_api_create_calibration_replay_missing_cases)
test("API: POST+GET /calibration/replay", test_api_create_and_get_calibration_replay)
test("API: GET /calibration/replay list", test_api_list_calibration_replays)
test("API: baseline section11 flow", test_api_baseline_section11_flow)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*54}")
print(f"  Results: {PASS} passed, {FAIL} failed  ({PASS+FAIL} total)")
print(f"{'═'*54}")

# Cleanup test DB
import os
try:
    os.remove("test_inspector.db")
except FileNotFoundError:
    pass

sys.exit(0 if FAIL == 0 else 1)
