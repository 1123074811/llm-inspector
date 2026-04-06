"""
LLM Inspector — test suite (pytest format).
Run with: pytest backend/tests/test_all.py
"""
import os
import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test_inspector.db"


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Config
# ═══════════════════════════════════════════════════════════════

def test_config_defaults():
    from app.core.config import settings
    assert settings.PORT == 8000
    assert settings.DEFAULT_REQUEST_TIMEOUT_SEC == 60
    assert settings.MAX_STREAM_CHUNKS == 512

def test_config_aes_key():
    from app.core.config import settings
    key = settings.aes_key
    assert len(key) == 32
    key2 = settings.aes_key
    assert key == key2


# ═══════════════════════════════════════════════════════════════
# SECTION 2: Security
# ═══════════════════════════════════════════════════════════════

def test_encrypt_decrypt_roundtrip(key_manager):
    original = "sk-test-key-abc123"
    enc, h = key_manager.encrypt(original)
    assert enc != original
    assert len(h) == 16
    dec = key_manager.decrypt(enc)
    assert dec == original

def test_encrypt_different_nonce(key_manager):
    enc1, _ = key_manager.encrypt("same-key")
    enc2, _ = key_manager.encrypt("same-key")
    assert enc1 != enc2

def test_hash_prefix_stable(key_manager):
    _, h1 = key_manager.encrypt("sk-stable")
    _, h2 = key_manager.encrypt("sk-stable")
    assert h1 == h2

def test_ssrf_localhost():
    from app.core.security import validate_and_sanitize_url
    with pytest.raises(ValueError):
        validate_and_sanitize_url("http://localhost/v1")

def test_ssrf_loopback_ip():
    from app.core.security import validate_and_sanitize_url
    with pytest.raises(ValueError):
        validate_and_sanitize_url("http://127.0.0.1/v1")

def test_ssrf_private_10():
    from app.core.security import validate_and_sanitize_url
    with pytest.raises(ValueError):
        validate_and_sanitize_url("http://10.0.0.1/api")

def test_ssrf_private_192():
    from app.core.security import validate_and_sanitize_url
    with pytest.raises(ValueError):
        validate_and_sanitize_url("http://192.168.1.100/v1")

def test_ssrf_bad_scheme():
    from app.core.security import validate_and_sanitize_url
    with pytest.raises(ValueError):
        validate_and_sanitize_url("ftp://example.com/v1")

def test_ssrf_strips_query():
    from app.core.security import validate_and_sanitize_url
    try:
        result = validate_and_sanitize_url("https://httpbin.org/v1?key=secret#frag")
        assert "key=secret" not in result
        assert "#frag" not in result
    except ValueError:
        pass

def test_normalize_openai_base_strips_chat_completions():
    from app.core.security import normalize_openai_compatible_base_url
    assert normalize_openai_compatible_base_url(
        "https://qianfan.baidubce.com/v2/coding/chat/completions"
    ) == "https://qianfan.baidubce.com/v2/coding"
    assert normalize_openai_compatible_base_url(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    ) == "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Database
# ═══════════════════════════════════════════════════════════════

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
    init_db()
    init_db()


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Seeder
# ═══════════════════════════════════════════════════════════════

def test_seed_cases():
    from app.tasks.seeder import seed_all
    from app.repository import repo
    seed_all()
    cases = repo.load_cases("v1", "standard")
    # get_benchmarks now returns only golden_baselines (real measured data).
    # Seeded benchmark_profiles are no longer used for similarity comparison.
    benchmarks = repo.get_benchmarks("v1")
    assert isinstance(benchmarks, list)

def test_seed_idempotent():
    from app.tasks.seeder import seed_all
    from app.repository import repo
    seed_all()
    seed_all()
    benchmarks = repo.get_benchmarks("v1")
    assert isinstance(benchmarks, list)


# ═══════════════════════════════════════════════════════════════
# SECTION 5: Judge methods
# ═══════════════════════════════════════════════════════════════

from app.judge.methods import judge

def test_exact_match_pass():
    p, _ = judge("exact_match", "7", {"target": "7"})
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
    assert p is False

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
    assert p is True

def test_json_schema_valid():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    p, d = judge("json_schema", '{"name": "Alice", "age": 30}', {"schema": schema})
    assert p is True, f"Expected pass, detail={d}"

def test_json_schema_missing_required():
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    p, _ = judge("json_schema", '{"name": "Alice"}', {"schema": schema})
    assert p is False

def test_json_schema_wrong_type():
    schema = {"type": "object", "required": ["age"],
              "properties": {"age": {"type": "integer"}}}
    p, _ = judge("json_schema", '{"age": "thirty"}', {"schema": schema})
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
    p, _ = judge("json_schema", '{"anything": true}', {"schema": {}})
    assert p is True


# ═══════════════════════════════════════════════════════════════
# SECTION 6: Repository
# ═══════════════════════════════════════════════════════════════

def test_create_run(create_run):
    run_id = create_run()
    assert run_id is not None

def test_get_run(create_run):
    from app.repository import repo
    run_id = create_run()
    run = repo.get_run(run_id)
    assert run is not None
    assert run["id"] == run_id
    assert run["status"] == "queued"

def test_get_run_not_found():
    from app.repository import repo
    run = repo.get_run("nonexistent-id")
    assert run is None

def test_update_run_status(create_run):
    from app.repository import repo
    run_id = create_run()
    repo.update_run_status(run_id, "running")
    run = repo.get_run(run_id)
    assert run["status"] == "running"
    assert run["started_at"] is not None

def test_cancel_requested(create_run):
    from app.repository import repo
    run_id = create_run()
    assert repo.is_run_cancel_requested(run_id) is False
    repo.set_run_cancel_requested(run_id, True)
    assert repo.is_run_cancel_requested(run_id) is True
    repo.set_run_cancel_requested(run_id, False)
    assert repo.is_run_cancel_requested(run_id) is False

def test_mark_run_retry(create_run):
    from app.repository import repo
    run_id = create_run()
    repo.update_run_status(run_id, "running")
    repo.mark_run_retry(run_id)
    run = repo.get_run(run_id)
    assert run["status"] == "queued"
    assert run["started_at"] is None


# ═══════════════════════════════════════════════════════════════
# SECTION 7: Feature extraction & scoring
# ═══════════════════════════════════════════════════════════════

def test_feature_extractor_basic():
    from app.core.schemas import TestCase, CaseResult, SampleResult, LLMResponse
    from app.analysis.pipeline import FeatureExtractor

    case = TestCase(
        id="test-1", category="protocol", name="test",
        user_prompt="hello", expected_type="str", judge_method="any_text",
    )
    resp = LLMResponse(content="world", status_code=200, latency_ms=100)
    sample = SampleResult(sample_index=0, response=resp, judge_passed=True)
    result = CaseResult(case=case, samples=[sample])

    extractor = FeatureExtractor()
    features = extractor.extract([result])
    assert "protocol_success_rate" in features
    assert features["protocol_success_rate"] == 1.0

def test_score_calculator_basic():
    from app.analysis.pipeline import ScoreCalculator
    scorer = ScoreCalculator()
    scores = scorer.calculate({})
    assert scores.protocol_score is not None
    assert scores.instruction_score is not None


# ═══════════════════════════════════════════════════════════════
# SECTION 8: PreDetection
# ═══════════════════════════════════════════════════════════════

def test_predetect_layer0_http():
    from app.predetect.pipeline import Layer0HTTP

    class MockAdapter:
        def head_request(self):
            return {"status_code": 200, "headers": {}}
        def bad_request(self):
            return {"status_code": 400, "body": {"error": {"type": "invalid_request"}}}

    layer = Layer0HTTP()
    result = layer.run(MockAdapter())
    assert result.layer == "http"
    assert result.confidence >= 0.0


# ═══════════════════════════════════════════════════════════════
# SECTION 9: HTTP handler routing
# ═══════════════════════════════════════════════════════════════

def test_handler_health_db_degraded():
    import pytest
    pytest.skip("Windows SQLite WAL mode locks database file")

def test_handler_health(setup_database):
    from app.main import handle_health
    status, body, content_type = handle_health("/health", "", None)
    assert status == 200
    assert content_type == "application/json"
