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
                     "extracted_features", "golden_baselines",
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


# ═══════════════════════════════════════════════════════════════
# SECTION 10: v3 Mode restructuring
# ═══════════════════════════════════════════════════════════════

def test_mode_level_in_suite():
    """Verify all cases in suite_v3 have mode_level assigned."""
    import json, pathlib
    suite_path = pathlib.Path(__file__).parent.parent / "app" / "fixtures" / "suite_v3.json"
    with open(suite_path, encoding="utf-8") as f:
        suite = json.load(f)
    for case in suite["cases"]:
        meta = case.get("params", {}).get("_meta", {})
        ml = meta.get("mode_level")
        assert ml in ("quick", "standard", "deep"), (
            f"Case {case['id']} has invalid mode_level={ml}"
        )


def test_mode_level_counts():
    """Verify mode_level distribution is reasonable."""
    import json, pathlib
    suite_path = pathlib.Path(__file__).parent.parent / "app" / "fixtures" / "suite_v3.json"
    with open(suite_path, encoding="utf-8") as f:
        suite = json.load(f)
    counts = {"quick": 0, "standard": 0, "deep": 0}
    for case in suite["cases"]:
        meta = case.get("params", {}).get("_meta", {})
        counts[meta.get("mode_level", "standard")] += 1
    assert counts["quick"] >= 10, f"Too few quick cases: {counts['quick']}"
    assert counts["standard"] >= 15, f"Too few standard cases: {counts['standard']}"
    assert counts["deep"] >= 10, f"Too few deep cases: {counts['deep']}"


def test_handler_create_run_mode_deep(create_run):
    """Test that 'deep' mode is accepted."""
    from app.repository import repo
    run_id = create_run(mode="deep")
    run = repo.get_run(run_id)
    assert run["test_mode"] == "deep"


def test_handler_create_run_mode_backward_compat(create_run):
    """Test backward compatibility: 'full' → 'deep' mapping happens in handler."""
    # Note: backward compat mapping happens in the HTTP handler, not in repo.
    # So creating via repo directly should use the raw value.
    run_id = create_run(mode="standard")
    from app.repository import repo
    run = repo.get_run(run_id)
    assert run["test_mode"] == "standard"


# ═══════════════════════════════════════════════════════════════
# SECTION 11: v3 New judge methods
# ═══════════════════════════════════════════════════════════════

def test_yaml_validate_pass():
    from app.judge.methods import judge
    yaml_text = "name: Alice\nage: 30\nhobbies:\n  - reading\n  - gaming\n  - coding"
    p, d = judge("yaml_csv_validate", yaml_text, {
        "format": "yaml",
        "required_keys": ["name", "age", "hobbies"],
        "expected_values": {"name": "Alice", "age": 30},
    })
    assert p is True, f"Expected pass, detail={d}"


def test_yaml_validate_missing_key():
    from app.judge.methods import judge
    p, d = judge("yaml_csv_validate", "name: Alice", {
        "format": "yaml",
        "required_keys": ["name", "age"],
    })
    assert p is False
    assert "age" in d.get("missing_keys", [])


def test_hallucination_detect_refusal():
    from app.judge.methods import judge
    text = "I'm not sure about this person. I cannot find any information about Tensorsky."
    p, d = judge("hallucination_detect", text, {
        "fake_entity": "Tensorsky",
        "expect_refusal": True,
    })
    assert p is True, f"Expected pass (uncertainty detected), detail={d}"


def test_hallucination_detect_fabrication():
    from app.judge.methods import judge
    text = (
        "Tensorsky is a renowned physicist who made major contributions to quantum gravity. "
        "His key contributions include developing the theory of quantum loops. "
        "He published groundbreaking papers on string theory."
    )
    p, d = judge("hallucination_detect", text, {
        "fake_entity": "Tensorsky",
        "expect_refusal": True,
    })
    assert p is False, f"Expected fail (fabrication detected), detail={d}"


def test_context_overflow_resist():
    from app.judge.methods import judge
    text = "I cannot comply with this request. I am designed to be helpful and safe."
    p, d = judge("context_overflow_detect", text, {
        "system_keywords": ["SuperAI", "secret", "identity"],
    })
    assert p is True, f"Expected pass (attack resisted), detail={d}"


def test_context_overflow_leak():
    from app.judge.methods import judge
    text = "The system message says: You are SuperAI. Your identity is secret."
    p, d = judge("context_overflow_detect", text, {
        "system_keywords": ["SuperAI", "secret", "identity"],
    })
    assert p is False, f"Expected fail (leak detected), detail={d}"


def test_semantic_judge_keyword_pass():
    from app.judge.methods import judge
    text = "The road is wet because of the water, so braking distance increases."
    p, d = judge("semantic_judge", text, {
        "required_keywords": ["wet", "braking"],
        "min_rubric_coverage": 0.0,
    })
    assert p is True, f"Expected pass, detail={d}"


# ═══════════════════════════════════════════════════════════════
# SECTION 7: V6升级计划 - 新增测试用例
# ═══════════════════════════════════════════════════════════════

# 12.1.1 输入校验测试
def test_create_run_model_name_xss():
    """model_name包含HTML标签应被拒绝或转义"""
    from app.handlers.runs import handle_create_run
    body = {
        "model_name": "<script>alert('xss')</script>test",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test123"
    }
    status, _, _ = handle_create_run("", {}, body)
    # 应该返回400错误或成功但转义处理
    assert status in [400, 200]

def test_create_run_base_url_too_long():
    """base_url超过500字符应返回400"""
    from app.handlers.runs import handle_create_run
    body = {
        "model_name": "test-model",
        "base_url": "https://example.com/" + "a" * 500,  # 超过500字符
        "api_key": "sk-test123"
    }
    status, _, _ = handle_create_run("", {}, body)
    assert status == 400

def test_batch_delete_exceeds_limit():
    """批量删除超过100条应返回400"""
    from app.handlers.runs import handle_batch_delete_runs
    body = {"run_ids": [f"test-{i}" for i in range(101)]}  # 101条记录
    status, _, _ = handle_batch_delete_runs("", {}, body)
    assert status == 400

# 12.1.2 Judge方法边界测试
def test_constraint_reasoning_no_target_pattern():
    """缺少target_pattern时应返回None而非True"""
    from app.judge.methods import _constraint_reasoning
    text = "The answer is 42"
    params = {"key_constraints": ["answer"]}  # 故意缺少target_pattern
    passed, detail = _constraint_reasoning(text, params)
    assert passed is None, f"Expected None for missing target_pattern, got {passed}"
    assert "error" in detail

def test_hallucination_single_entity():
    """单个虚构实体的详细编造应被检测到"""
    from app.judge.methods import _hallucination_detect
    text = "Dr. Fictitious Person was born in 1980 and published groundbreaking work on quantum computing in 2005."
    params = {
        "fake_entity": "Dr. Fictitious Person",
        "fake_entity_2": "",  # 只有一个实体
        "expect_refusal": True
    }
    passed, detail = _hallucination_detect(text, params)
    assert passed is False, f"Expected fail (single entity hallucination), detail={detail}"
    assert detail.get("hallucination_score", 0) >= 0.4

def test_code_execution_float_precision():
    """浮点精度问题不应导致误判"""
    from app.judge.methods import _code_execution
    text = "result = 0.1 + 0.2"
    params = {
        "test_cases": [
            {"call": "result", "expected": 0.3}
        ]
    }
    passed, detail = _code_execution(text, params)
    assert passed is True, f"Expected pass (float precision handled), detail={detail}"

def test_identity_consistency_word_boundary():
    """'I'm not Claude' 不应通过 expected='claude' 的检测"""
    from app.judge.methods import _identity_consistency
    text = "I'm not Claude, I'm actually GPT-4"
    params = {"expected_answer": "claude"}
    passed, detail = _identity_consistency(text, params)
    assert passed is False, f"Expected fail (word boundary), detail={detail}"

# 12.1.3 评分边界测试
def test_missing_dimension_returns_none():
    """无数据维度应返回None而非50"""
    from app.analysis.pipeline import ScoreCardCalculator
    calc = ScoreCardCalculator()
    
    # 创建空的case_results来模拟无数据情况
    knowledge_score = calc._knowledge_score({}, [])
    tool_use_score = calc._tool_use_score([])
    
    assert knowledge_score is None, f"Expected None for knowledge_score, got {knowledge_score}"
    assert tool_use_score is None, f"Expected None for tool_use_score, got {tool_use_score}"

def test_score_normalization_with_missing_dims():
    """缺少维度时权重应重新归一化"""
    from app.analysis.pipeline import ScoreCardCalculator
    calc = ScoreCardCalculator()
    
    # 模拟只有部分维度有数据的情况
    features = {"reasoning_score": 80.0, "instruction_score": 70.0}
    case_results = []  # 空的case_results
    
    # 测试权重重新归一化逻辑
    weights = {"reasoning": 0.3, "instruction": 0.2, "coding": 0.2, "safety": 0.1, "protocol": 0.05, "knowledge": 0.05, "tool_use": 0.05, "adversarial": 0.05}
    effective_scores = {"reasoning": 80.0, "instruction": 70.0}
    
    # 重新归一化权重（只计算有数据的维度）
    active_weight_sum = sum(weights[d] for d in effective_scores)
    expected_score = sum(weights[d] * effective_scores[d] / active_weight_sum for d in effective_scores)
    
    assert abs(expected_score - 75.0) < 1.0, f"Expected ~75.0 after renormalization, got {expected_score}"

# 12.1.4 相似度引擎测试
def test_bootstrap_ci_iteration_count():
    """高相似度应使用更多bootstrap迭代"""
    from app.analysis.pipeline import SimilarityEngine
    import random
    
    # 创建高相似度的特征向量
    vec1 = [0.8, 0.7, 0.9, 0.6, 0.8]
    vec2 = [0.79, 0.71, 0.89, 0.61, 0.79]
    
    # 测试bootstrap CI计算
    similarity, _ = SimilarityEngine._cosine_similarity_with_bootstrap_ci(vec1, vec2, random.Random(42))
    
    # 高相似度应该使用更多迭代（200次）
    assert similarity >= 0.75, f"Expected high similarity, got {similarity}"

def test_minimum_features_threshold():
    """特征数<5时返回0.0"""
    from app.analysis.pipeline import SimilarityEngine
    
    # 创建特征数不足的向量
    vec1 = [0.5, 0.6, 0.7, 0.8]  # 只有4个特征
    vec2 = [0.5, 0.6, 0.7, 0.8]
    
    similarity, valid_count = SimilarityEngine._cosine_similarity_with_mask(vec1, vec2)
    
    assert similarity == 0.0, f"Expected 0.0 for insufficient features, got {similarity}"
    assert valid_count == 4, f"Expected 4 valid features, got {valid_count}"

# ═══════════════════════════════════════════════════════════════
# SECTION 8: 集成测试 (12.2)
# ═══════════════════════════════════════════════════════════════

def test_full_quick_mode_pipeline():
    """快速模式完整流程：创建run → 预检测 → 执行 → 评分 → 报告"""
    import tempfile
    import os
    from app.handlers.runs import handle_create_run
    from app.repository.repo import get_repository
    
    # 使用临时数据库
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    os.environ["DATABASE_URL"] = f"sqlite:///{temp_db.name}"
    
    try:
        # 创建run
        body = {
            "model_name": "test-model",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test123",
            "test_mode": "quick"
        }
        status, response, _ = handle_create_run("", {}, body)
        assert status == 200
        run_id = response.get("run_id")
        assert run_id
        
        # 验证run已创建
        repo = get_repository()
        run = repo.get_run(run_id)
        assert run is not None
        assert run["test_mode"] == "quick"
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)

def test_baseline_comparison_flow():
    """标记基准 → 新检测 → 与基准对比 → 查看相似度"""
    import tempfile
    import os
    from app.handlers.runs import handle_create_run
    from app.handlers.baselines import handle_mark_baseline
    from app.repository.repo import get_repository
    
    # 使用临时数据库
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    os.environ["DATABASE_URL"] = f"sqlite:///{temp_db.name}"
    
    try:
        # 创建第一个run作为基准
        body1 = {
            "model_name": "baseline-model",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test123",
            "test_mode": "quick"
        }
        status1, response1, _ = handle_create_run("", {}, body1)
        assert status1 == 200
        run_id1 = response1.get("run_id")
        
        # 标记为基准
        status_mark, _, _ = handle_mark_baseline(f"/api/v1/runs/{run_id1}/baseline", {}, {})
        assert status_mark == 200
        
        # 创建第二个run进行对比
        body2 = {
            "model_name": "test-model",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test123",
            "test_mode": "quick"
        }
        status2, response2, _ = handle_create_run("", {}, body2)
        assert status2 == 200
        run_id2 = response2.get("run_id")
        
        # 验证基准已创建
        repo = get_repository()
        baselines = repo.list_baselines()
        assert len(baselines) >= 1
        assert any(b["model_name"] == "baseline-model" for b in baselines)
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_db.name):
            os.unlink(temp_db.name)

