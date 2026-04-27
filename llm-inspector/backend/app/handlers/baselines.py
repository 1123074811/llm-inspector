"""Baseline and benchmark handlers."""
from __future__ import annotations

from app.handlers.helpers import _json, _error, _extract_id
from app.repository import repo
from app.tasks.worker import submit_run
from app.core.logging import get_logger

logger = get_logger(__name__)


def handle_benchmarks(_path, qs, _body) -> tuple:
    suite_version = qs.get("suite_version", ["v1"])[0]
    benchmarks = repo.get_benchmarks(suite_version)
    return _json([
        {
            "name": b.get("benchmark_name") or b.get("name", "unknown"),
            "suite_version": b.get("suite_version", ""),
            "data_source": b.get("data_source", "not_measured"),
            "sample_count": b.get("sample_count", 3),
        }
        for b in benchmarks
    ])


def handle_create_baseline(_path, _qs, body: dict) -> tuple:
    # Mode 1: Mark an existing run as baseline (from frontend "标记为基准模型")
    if body.get("run_id") and not body.get("base_url"):
        run_id = body["run_id"]
        run = repo.get_run(run_id)
        if not run:
            return _error("Run not found", 404)
        if run["status"] not in ("completed", "partial_failed"):
            return _error("Run not completed yet", 400)

        model_name = body.get("model_name") or run["model_name"]
        display_name = body.get("display_name") or model_name

        try:
            result = repo.create_baseline(
                run_id=run_id,
                model_name=model_name,
                display_name=display_name,
            )
        except ValueError as e:
            return _error(str(e), 400)
        # Critical: invalidate the in-process benchmark cache so the next run
        # actually sees the just-created baseline. Without this, runs started
        # within BENCHMARK_CACHE_TTL_SEC of the call below will silently use
        # the previous benchmark list (root cause of v16 acceptance bug:
        # "set deepseek-v4-flash run #1 as baseline → run #2 still ranks
        # qianfan-code-latest #1 because run #2 saw the stale baseline list").
        try:
            from app.runner.case_prep import invalidate_benchmark_cache
            invalidate_benchmark_cache()
        except Exception as e:
            logger.warning("Failed to invalidate benchmark cache after baseline create", error=str(e))
        logger.info("Baseline created from run", display_name=display_name, run_id=run_id)
        return _json({"baseline_id": result["id"], "status": "created"}, 201)

    # Mode 2: Create a new run and mark as baseline (full params)
    for field in ("name", "base_url", "api_key", "model"):
        if not body.get(field):
            return _error(f"Missing required field: {field}")

    from app.core.security import validate_and_sanitize_url, get_key_manager
    try:
        clean_url = validate_and_sanitize_url(body["base_url"])
    except ValueError as e:
        return _error(str(e))

    api_key = str(body["api_key"]).strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()

    km = get_key_manager()
    encrypted, key_hash = km.encrypt(api_key)

    test_mode = body.get("test_mode", "standard")
    suite_version = body.get("suite_version", "v3")

    run_metadata = {
        "evaluation_mode": "normal",
        "scoring_profile_version": "v1",
        "calibration_tag": "baseline-v1.0",
    }

    run_id = repo.create_run(
        base_url=clean_url,
        api_key_encrypted=encrypted,
        api_key_hash=key_hash,
        model_name=body["model"],
        test_mode=test_mode,
        suite_version=suite_version,
        metadata=run_metadata,
    )

    repo.create_baseline(run_id=run_id, model_name=body["model"], display_name=body["name"])
    try:
        from app.runner.case_prep import invalidate_benchmark_cache
        invalidate_benchmark_cache()
    except Exception as e:
        logger.warning("Failed to invalidate benchmark cache after baseline create (mode2)", error=str(e))
    submit_run(run_id)
    logger.info("Baseline created", name=body["name"], run_id=run_id)
    return _json({"baseline_id": run_id, "status": "queued"}, 201)


def handle_list_baselines(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["50"])[0])
    baselines = repo.list_baselines(limit=min(limit, 100))
    return _json({"baselines": baselines})


def handle_compare_baseline(_path, _qs, body: dict) -> tuple:
    try:
        baseline_id = body.get("baseline_id")
        run_id = body.get("run_id")
        if not baseline_id or not run_id:
            return _error("baseline_id and run_id required")

        baseline = repo.get_baseline(baseline_id)
        if not baseline:
            return _error("Baseline not found", 404)

        # 安全获取baseline的run_id (字段名是source_run_id)
        baseline_run_id = baseline.get("source_run_id")
        logger.info("Baseline data", baseline_id=baseline_id, baseline_keys=list(baseline.keys()), baseline_run_id=baseline_run_id)
        if not baseline_run_id:
            return _error("Baseline missing source_run_id", 500)
        
        baseline_run = repo.get_run(baseline_run_id)
        target_run = repo.get_run(run_id)
        if not target_run:
            return _error("Target run not found", 404)

        if target_run["status"] not in ("completed", "partial_failed"):
            return _error("Target run not completed", 400)

        report_row = repo.get_report(run_id)
        if not report_row:
            return _error("Target report not found", 404)

        baseline_report = repo.get_report(baseline_run_id)
        if not baseline_report:
            return _error("Baseline report not found", 404)
        # 创建安全的字典访问包装器
        class SafeDict:
            def __init__(self, data, default_run_id):
                self.data = data if isinstance(data, dict) else {}
                self.default_run_id = default_run_id
            
            def get(self, key, default=None):
                if key == "run_id" and key not in self.data:
                    return self.default_run_id
                return self.data.get(key, default)
            
            def __getitem__(self, key):
                if key == "run_id" and key not in self.data:
                    return self.default_run_id
                return self.data[key]
            
            def __contains__(self, key):
                return True if key == "run_id" else key in self.data
            
            def keys(self):
                keys = set(self.data.keys())
                keys.add("run_id")  # 确保run_id总是在keys中
                return keys
        
        # 创建一个完全独立的基准对比实现，避免所有可能的run_id访问问题
        try:
            # 安全提取报告数据
            target_details_raw = report_row["details"]
            baseline_details_raw = baseline_report["details"]
            
            if not target_details_raw:
                return _error("Target report details are empty", 500)
            if not baseline_details_raw:
                return _error("Baseline report details are empty", 500)
            
            # 直接使用get方法避免KeyError
            target_scorecard = target_details_raw.get("scorecard", {}) or {}
            baseline_scorecard = baseline_details_raw.get("scorecard", {}) or {}
            
            target_features = target_details_raw.get("features", {}) or {}
            baseline_features = baseline_details_raw.get("features", {}) or {}
            
            logger.info("Comparing reports", target_features_count=len(target_features), baseline_features_count=len(baseline_features))
            
            # 计算基准对比结果（完全独立实现）
            all_keys = sorted(set(target_features.keys()) | set(baseline_features.keys()))
            
            # 计算余弦相似度
            vec_curr = [float(target_features.get(k, 0.0) or 0.0) for k in all_keys]
            vec_base = [float(baseline_features.get(k, 0.0) or 0.0) for k in all_keys]
            
            import math
            dot = sum(x * y for x, y in zip(vec_curr, vec_base))
            norm_curr = math.sqrt(sum(x * x for x in vec_curr))
            norm_base = math.sqrt(sum(y * y for y in vec_base))
            denom = norm_curr * norm_base
            cosine_sim = (dot / denom) if denom > 0 else 0.0
            
            # 计算分数差异
            target_total = float(target_scorecard.get("total_score", 0) or 0)
            baseline_total = float(baseline_scorecard.get("total_score", 0) or 0)
            target_cap = float(target_scorecard.get("capability_score", 0) or 0)
            baseline_cap = float(baseline_scorecard.get("capability_score", 0) or 0)
            target_auth = float(target_scorecard.get("authenticity_score", 0) or 0)
            baseline_auth = float(baseline_scorecard.get("authenticity_score", 0) or 0)
            target_perf = float(target_scorecard.get("performance_score", 0) or 0)
            baseline_perf = float(baseline_scorecard.get("performance_score", 0) or 0)
            
            delta_total = target_total - baseline_total
            delta_cap = target_cap - baseline_cap
            delta_auth = target_auth - baseline_auth
            delta_perf = target_perf - baseline_perf
            
            # 计算特征漂移
            feature_drift = {}
            for k in all_keys:
                base_val = float(baseline_features.get(k, 0.0) or 0.0)
                curr_val = float(target_features.get(k, 0.0) or 0.0)
                if base_val != 0:
                    pct = (curr_val - base_val) / abs(base_val) * 100
                else:
                    pct = 0.0
                feature_drift[k] = {
                    "baseline": round(base_val, 4),
                    "current": round(curr_val, 4),
                    "delta_pct": round(pct, 2),
                }
            
            top5 = dict(sorted(feature_drift.items(), key=lambda x: abs(x[1]["delta_pct"]), reverse=True)[:5])
            
            # 判断结果
            abs_delta_total_display = abs(delta_total) * 100
            from app.core.config import settings
            if (
                cosine_sim >= settings.BASELINE_MATCH_COSINE_THRESHOLD
                and abs_delta_total_display <= settings.BASELINE_MATCH_SCORE_DELTA_MAX
            ):
                verdict = "match"
            elif cosine_sim >= 0.85 or abs_delta_total_display <= 1500:
                verdict = "suspicious"
            else:
                verdict = "mismatch"
            
            comparison = {
                "cosine_similarity": round(cosine_sim, 4),
                "score_delta": {
                    "total": round(delta_total * 100),
                    "capability": round(delta_cap * 100),
                    "authenticity": round(delta_auth * 100),
                    "performance": round(delta_perf * 100),
                },
                "feature_drift_top5": top5,
                "verdict": verdict,
            }
            
            # Add baseline name to comparison result
            comparison["baseline_name"] = baseline.get("display_name", baseline.get("model_name", "Unknown"))
            comparison["baseline_id"] = baseline.get("id", baseline_id)
            
            return _json(comparison)
            
        except Exception as inner_e:
            logger.error("Independent comparison failed", error=str(inner_e), baseline_id=baseline_id, run_id=run_id)
            # 返回一个基本的结果
            return _json({
                "cosine_similarity": 0.0,
                "score_delta": {"total": 0.0, "capability": 0.0, "authenticity": 0.0, "performance": 0.0},
                "feature_drift_top5": {},
                "verdict": "mismatch",
                "baseline_name": baseline.get("display_name", baseline.get("model_name", "Unknown")),
                "baseline_id": baseline.get("id", baseline_id),
                "error": f"Comparison failed: {str(inner_e)}",
            })
        
    except KeyError as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error("KeyError in baseline comparison", missing_key=str(e), baseline_id=baseline_id, run_id=run_id, traceback=error_trace)
        return _error(f"KeyError: missing key '{str(e)}'", 500)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error("Baseline comparison failed", baseline_id=baseline_id, run_id=run_id, error=str(e), traceback=error_trace)
        return _error(f"Internal error: {str(e)}", 500)


def handle_delete_baseline(path, _qs, _body) -> tuple:
    baseline_id = _extract_id(path, r"/api/v1/baselines/([^/]+)$")
    if not baseline_id:
        return _error("Invalid baseline ID", 400)
    repo.delete_baseline(baseline_id)
    try:
        from app.runner.case_prep import invalidate_benchmark_cache
        invalidate_benchmark_cache()
    except Exception as e:
        logger.warning("Failed to invalidate benchmark cache after baseline delete", error=str(e))
    return _json({"deleted": baseline_id})


def handle_get_baseline(path, _qs, _body) -> tuple:
    baseline_id = _extract_id(path, r"/api/v1/baselines/([^/]+)$")
    if not baseline_id:
        return _error("Invalid baseline ID", 400)
    baseline = repo.get_baseline(baseline_id)
    if not baseline:
        return _error("Baseline not found", 404)
    return _json(baseline)
