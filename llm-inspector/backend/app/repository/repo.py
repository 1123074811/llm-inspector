"""
Repository — all database read/write operations.
Uses stdlib sqlite3 via app.core.db.
"""
from __future__ import annotations

import json
from app.core.db import get_conn, now_iso, new_id, json_col, from_json_col


# ── Runs ──────────────────────────────────────────────────────────────────────

def create_run(
    base_url: str,
    api_key_encrypted: str,
    api_key_hash: str,
    model_name: str,
    test_mode: str = "standard",
    suite_version: str = "v1",
    metadata: dict | None = None,
) -> str:
    run_id = new_id()
    conn = get_conn()
    meta = metadata or {}
    conn.execute(
        """INSERT INTO test_runs
           (id, base_url, api_key_encrypted, api_key_hash,
            model_name, test_mode, suite_version, status, created_at,
            evaluation_mode, calibration_case_id, scoring_profile_version,
            calibration_tag, metadata)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, base_url, api_key_encrypted, api_key_hash,
         model_name, test_mode, suite_version, "queued", now_iso(),
         meta.get("evaluation_mode", "normal"),
         meta.get("calibration_case_id"),
         meta.get("scoring_profile_version", "v1"),
         meta.get("calibration_tag"),
         json_col(meta)),
    )
    conn.commit()
    return run_id


def get_run(run_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM test_runs WHERE id=?", (run_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["predetect_result"] = from_json_col(d.get("predetect_result"))
    d["metadata"] = from_json_col(d.get("metadata"))
    return d


def update_run_status(run_id: str, status: str, **kwargs) -> None:
    sets = ["status=?"]
    vals: list = [status]
    if status == "running" and "started_at" not in kwargs:
        sets.append("started_at=?")
        vals.append(now_iso())
    if status in ("completed", "failed") and "completed_at" not in kwargs:
        sets.append("completed_at=?")
        vals.append(now_iso())
    for k, v in kwargs.items():
        sets.append(f"{k}=?")
        vals.append(v)
    vals.append(run_id)
    conn = get_conn()
    conn.execute(f"UPDATE test_runs SET {','.join(sets)} WHERE id=?", vals)
    conn.commit()


def save_identity_exposure(run_id: str, report_dict: dict) -> None:
    """v14 Phase 3: persist IdentityExposureReport for a run."""
    conn = get_conn()
    conn.execute(
        "UPDATE test_runs SET identity_exposure_result=? WHERE id=?",
        (json_col(report_dict), run_id),
    )
    conn.commit()


def get_identity_exposure(run_id: str) -> dict | None:
    """v14 Phase 3: retrieve IdentityExposureReport for a run."""
    conn = get_conn()
    row = conn.execute(
        "SELECT identity_exposure_result FROM test_runs WHERE id=?", (run_id,)
    ).fetchone()
    if not row:
        return None
    return from_json_col(row["identity_exposure_result"])


def get_predetect_trace_path(run_id: str) -> str:
    """v14 Phase 5: return path to predetect JSONL trace file for a run."""
    import pathlib
    from app.core.config import settings
    return str(pathlib.Path(settings.DATA_DIR) / "traces" / run_id / "predetect.jsonl")


def read_predetect_trace(run_id: str, offset: int = 0, limit: int = 50) -> list[dict]:
    """v14 Phase 5: read JSONL predetect trace for a run, with pagination.
    Returns empty list if file doesn't exist or on any I/O error.
    """
    import pathlib
    from app.core.config import settings
    trace_file = pathlib.Path(settings.DATA_DIR) / "traces" / run_id / "predetect.jsonl"
    if not trace_file.exists():
        return []
    try:
        lines: list[dict] = []
        with trace_file.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if raw_line:
                    try:
                        lines.append(json.loads(raw_line))
                    except json.JSONDecodeError:
                        pass
        return lines[offset: offset + limit]
    except Exception:
        return []


def save_predetect_result(run_id: str, result_dict: dict) -> None:
    confidence = result_dict.get("confidence", 0.0)
    identified = result_dict.get("success", False)
    conn = get_conn()
    conn.execute(
        """UPDATE test_runs
           SET predetect_result=?, predetect_confidence=?,
               predetect_identified=?, status=?
           WHERE id=?""",
        (json_col(result_dict), confidence, 1 if identified else 0,
         "pre_detected", run_id),
    )
    conn.commit()


def update_predetect_progress(
    run_id: str,
    current_layer: str,
    layer_results: list = None,
    current_probe: str = None,
    probe_detail: dict = None,
    evidence: list = None,
    tokens_used: int = 0,
) -> None:
    """Update current layer progress during pre-detection for real-time UI feedback.
    
    Args:
        current_layer: Current layer being executed (e.g., "Layer2/Identity")
        current_probe: Specific probe/step within the layer (e.g., "tokenizer_count_probe")
        probe_detail: Dict with request/response preview for the current probe
        evidence: List of evidence items collected so far
        tokens_used: Cumulative tokens used in this layer
    """
    conn = get_conn()
    result_partial = {
        "current_layer": current_layer,
        "current_probe": current_probe,
        "probe_detail": probe_detail or {},
        "evidence": evidence or [],
        "layer_results": layer_results or [],
        "tokens_used": tokens_used,
        "success": False,
        "identified_as": None,
        "confidence": 0.0,
    }
    conn.execute(
        """UPDATE test_runs
           SET predetect_result=?, status=?
           WHERE id=?""",
        (json_col(result_partial), "pre_detecting", run_id),
    )
    conn.commit()


def list_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_run(run_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM test_runs WHERE id=?", (run_id,))
    conn.commit()


def set_run_cancel_requested(run_id: str, requested: bool = True) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE test_runs SET cancel_requested=? WHERE id=?",
        (1 if requested else 0, run_id),
    )
    conn.commit()


def is_run_cancel_requested(run_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT cancel_requested FROM test_runs WHERE id=?", (run_id,)
    ).fetchone()
    if not row:
        return False
    return bool(row["cancel_requested"])


def mark_run_retry(run_id: str) -> None:
    conn = get_conn()
    conn.execute(
        """UPDATE test_runs
           SET status=?, started_at=NULL, completed_at=NULL,
               error_message=NULL, cancel_requested=0, resume_from_existing=1
           WHERE id=?""",
        ("queued", run_id),
    )
    conn.commit()


# ── Test Cases ────────────────────────────────────────────────────────────────

def upsert_test_case(case: dict) -> None:
    conn = get_conn()
    params = dict(case.get("params", {}))
    params.setdefault("_meta", {})
    meta = params.get("_meta") or {}
    meta.update({
        "dimension": meta.get("dimension") or case.get("dimension"),
        "tags": meta.get("tags") or case.get("tags", []),
        "judge_rubric": meta.get("judge_rubric") or case.get("judge_rubric", {}),
        "anchor": bool(meta.get("anchor", False)),
        "info_gain_prior": float(meta.get("info_gain_prior", case.get("weight", 1.0))),
    })
    if "difficulty" in case:
        meta["difficulty"] = case["difficulty"]
    params["_meta"] = meta

    conn.execute(
        """INSERT OR REPLACE INTO test_cases
           (id, category, name, system_prompt, user_prompt,
            expected_type, judge_method, params,
            max_tokens, n_samples, temperature, weight, enabled, suite_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            case.get("id"), case.get("category"), case.get("name"),
            case.get("system_prompt"), case.get("user_prompt"),
            case.get("expected_type", "any"), case.get("judge_method", "any_text"),
            json_col(params),
            case.get("max_tokens", 4096), case.get("n_samples", 1),
            case.get("temperature", 0.0), case.get("weight", 1.0),
            1 if case.get("enabled", True) else 0,
            case.get("suite_version", "v1"),
        ),
    )
    conn.commit()


def load_cases(suite_version: str = "v1", test_mode: str = "standard") -> list[dict]:
    conn = get_conn()

    rows = conn.execute(
        "SELECT * FROM test_cases WHERE suite_version=? AND enabled=1",
        (suite_version,),
    ).fetchall()

    if not rows and suite_version != "v2":
        rows = conn.execute(
            "SELECT * FROM test_cases WHERE suite_version='v2' AND enabled=1"
        ).fetchall()

    if not rows:
        rows = conn.execute(
            "SELECT * FROM test_cases WHERE enabled=1"
        ).fetchall()

    mode_filter = {"quick": ["quick"], "standard": ["quick", "standard"], "deep": ["quick", "standard", "deep"]}
    allowed_levels = mode_filter.get(test_mode, ["quick", "standard"])

    cases = []
    for row in rows:
        c = dict(row)
        c["params"] = from_json_col(c.get("params")) or {}
        if c.get("difficulty") is None:
            c["difficulty"] = (c["params"].get("_meta") or {}).get("difficulty")
        mode_level = (c["params"].get("_meta") or {}).get("mode_level", "standard")
        if mode_level not in allowed_levels:
            continue
        cases.append(c)
    return cases


# ── Responses ─────────────────────────────────────────────────────────────────

def save_response(run_id: str, case_id: str, sample_index: int,
                  resp_data: dict) -> str:
    resp_id = new_id()
    conn = get_conn()
    conn.execute(
        """INSERT INTO test_responses
           (id, run_id, case_id, sample_index, request_payload,
            response_text, raw_response, raw_headers,
            status_code, latency_ms, first_token_ms, finish_reason,
            usage_prompt_tokens, usage_completion_tokens, usage_total_tokens,
            error_type, error_message, judge_passed, judge_detail, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            resp_id, run_id, case_id, sample_index,
            json_col(resp_data.get("request")),
            resp_data.get("response_text"),
            json_col(resp_data.get("raw_response")),
            json_col(resp_data.get("raw_headers")),
            resp_data.get("status_code"),
            resp_data.get("latency_ms"),
            resp_data.get("first_token_ms"),
            resp_data.get("finish_reason"),
            resp_data.get("usage_prompt_tokens"),
            resp_data.get("usage_completion_tokens"),
            resp_data.get("usage_total_tokens"),
            resp_data.get("error_type"),
            resp_data.get("error_message"),
            1 if resp_data.get("judge_passed") else (
                0 if resp_data.get("judge_passed") is False else None
            ),
            json_col(resp_data.get("judge_detail", {})),
            now_iso(),
        ),
    )
    conn.commit()
    return resp_id


def save_response_batch(rows_data: list[dict]) -> list[str]:
    """Batch insert test_responses rows in one transaction."""
    if not rows_data:
        return []
    conn = get_conn()
    ids: list[str] = []
    ts = now_iso()
    for item in rows_data:
        resp_id = new_id()
        ids.append(resp_id)
        resp_data = item["resp_data"]
        conn.execute(
            """INSERT INTO test_responses
               (id, run_id, case_id, sample_index, request_payload,
                response_text, raw_response, raw_headers,
                status_code, latency_ms, first_token_ms, finish_reason,
                usage_prompt_tokens, usage_completion_tokens, usage_total_tokens,
                error_type, error_message, judge_passed, judge_detail, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                resp_id, item["run_id"], item["case_id"], item["sample_index"],
                json_col(resp_data.get("request")),
                resp_data.get("response_text"),
                json_col(resp_data.get("raw_response")),
                json_col(resp_data.get("raw_headers")),
                resp_data.get("status_code"),
                resp_data.get("latency_ms"),
                resp_data.get("first_token_ms"),
                resp_data.get("finish_reason"),
                resp_data.get("usage_prompt_tokens"),
                resp_data.get("usage_completion_tokens"),
                resp_data.get("usage_total_tokens"),
                resp_data.get("error_type"),
                resp_data.get("error_message"),
                1 if resp_data.get("judge_passed") else (
                    0 if resp_data.get("judge_passed") is False else None
                ),
                json_col(resp_data.get("judge_detail", {})),
                ts,
            ),
        )
    conn.commit()
    return ids


def get_responses(run_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_responses WHERE run_id=? ORDER BY created_at",
        (run_id,),
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["raw_response"] = from_json_col(d.get("raw_response"))
        d["judge_detail"] = from_json_col(d.get("judge_detail"))
        d["request_payload"] = from_json_col(d.get("request_payload"))
        result.append(d)
    return result


# ── Features ──────────────────────────────────────────────────────────────────

def save_features(run_id: str, features: dict[str, float]) -> None:
    conn = get_conn()
    ts = now_iso()
    for name, value in features.items():
        fid = new_id()
        conn.execute(
            """INSERT OR REPLACE INTO extracted_features
               (id, run_id, feature_name, feature_value, created_at)
               VALUES (?,?,?,?,?)""",
            (fid, run_id, name, float(value), ts),
        )
    conn.commit()


def get_features(run_id: str) -> dict[str, float]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT feature_name, feature_value FROM extracted_features WHERE run_id=?",
        (run_id,),
    ).fetchall()
    return {row["feature_name"]: row["feature_value"] for row in rows}


# ── Benchmarks (from golden_baselines) ────────────────────────────────────────

def get_benchmarks(suite_version: str = "v1") -> list[dict]:
    """
    Load benchmark profiles for similarity comparison.
    Priority:
      1. golden_baselines table (real measured data marked by user) — always preferred
      2. default_profiles.json (BenchmarkCollector output, data_source="measured") — fallback only
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM golden_baselines WHERE is_active=1 ORDER BY created_at DESC"
    ).fetchall()

    result: list[dict] = []
    seen_names: set[str] = set()

    for row in rows:
        d = dict(row)
        name = d.get("model_name", "")
        if name in seen_names:
            continue
        seen_names.add(name)
        result.append({
            "benchmark_name": name,
            "name": d.get("display_name", name),
            "suite_version": d.get("suite_version", suite_version),
            "feature_vector": from_json_col(d.get("feature_vector")) or {},
            "data_source": "measured",
            "sample_count": d.get("sample_count", 1),
        })

    # ── Fallback: load from default_profiles.json when golden_baselines is empty ──
    if not result:
        result = _load_seed_profiles(suite_version, exclude_names=seen_names)

    return result


def _load_seed_profiles(suite_version: str, exclude_names: set[str]) -> list[dict]:
    """
    Load profiles from default_profiles.json as a bootstrap fallback.
    Accepts both "measured" and "estimated" profiles — estimated profiles are
    tagged with data_source="estimated_seed" so downstream scoring can
    apply lower confidence weights when comparing against them.
    Having estimated baselines is much better than having none at all.
    """
    import json as _json
    import pathlib as _pathlib

    profiles_path = (
        _pathlib.Path(__file__).parent.parent
        / "fixtures" / "benchmarks" / "default_profiles.json"
    )
    if not profiles_path.exists():
        return []

    try:
        data = _json.loads(profiles_path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError):
        return []

    out = []
    for p in data.get("benchmarks", []):
        name = p.get("name") or p.get("benchmark_name", "")
        if not name or name in exclude_names:
            continue
        source = p.get("data_source", "estimated")
        # Tag seed profiles distinctly so scoring can weight them appropriately
        seed_source = "measured_seed" if source == "measured" else "estimated_seed"
        out.append({
            "benchmark_name": name,
            "name": p.get("display_name", name),
            "suite_version": p.get("suite_version", suite_version),
            "feature_vector": p.get("feature_vector", {}),
            "data_source": seed_source,
            "sample_count": p.get("sample_count", 1),
        })
    return out


# ── Similarity Results ────────────────────────────────────────────────────────

def save_similarities(run_id: str, results: list[dict]) -> None:
    conn = get_conn()
    for r in results:
        conn.execute(
            """INSERT INTO similarity_results
               (id, run_id, benchmark_name, similarity_score,
                ci_95_low, ci_95_high, rank_pos)
               VALUES (?,?,?,?,?,?,?)""",
            (new_id(), run_id, r["benchmark"],
             r["score"], r.get("ci_95_low"), r.get("ci_95_high"), r["rank"]),
        )
    conn.commit()


# ── Reports ───────────────────────────────────────────────────────────────────

def save_report(run_id: str, report: dict) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO reports
           (id, run_id, summary, details, created_at)
           VALUES (?,?,?,?,?)""",
        (
            new_id(), run_id,
            json_col({
                "risk_level": report.get("risk", {}).get("level"),
                "identified_as": (report.get("predetection") or {}).get("identified_as"),
                "protocol_score": report.get("scores", {}).get("protocol_score"),
                "instruction_score": report.get("scores", {}).get("instruction_score"),
            }),
            json_col(report),
            now_iso(),
        ),
    )
    conn.commit()


def get_report(run_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM reports WHERE run_id=?", (run_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["details"] = from_json_col(d.get("details"))
    d["summary"] = from_json_col(d.get("summary"))
    return d


# ── Score Breakdown ──────────────────────────────────────────────────────────

def save_score_breakdown(run_id: str, dimension: str, score: float,
                         max_score: float = 10000.0, details: dict | None = None) -> None:
    conn = get_conn()
    display_score = round(float(score) * 100)
    conn.execute(
        """INSERT OR REPLACE INTO score_breakdown
           (id, run_id, dimension, score, max_score, details, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (new_id(), run_id, dimension, display_score, max_score,
         json_col(details) if details else None, now_iso()),
    )
    conn.commit()


def save_score_breakdowns(run_id: str, breakdowns: dict[str, float]) -> None:
    for dimension, score in breakdowns.items():
        save_score_breakdown(run_id, dimension, score)


def get_score_breakdown(run_id: str) -> dict[str, float]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT dimension, score FROM score_breakdown WHERE run_id=?",
        (run_id,),
    ).fetchall()
    return {row["dimension"]: row["score"] for row in rows}


# ── Compare Runs ─────────────────────────────────────────────────────────────

def create_compare_run(golden_run_id: str, candidate_run_id: str) -> str:
    cid = new_id()
    conn = get_conn()
    conn.execute(
        """INSERT INTO compare_runs
           (id, golden_run_id, candidate_run_id, status, created_at)
           VALUES (?,?,?,?,?)""",
        (cid, golden_run_id, candidate_run_id, "queued", now_iso()),
    )
    conn.commit()
    return cid


def get_compare_run(compare_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM compare_runs WHERE id=?", (compare_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["details"] = from_json_col(d.get("details"))
    return d


def update_compare_run(compare_id: str, **kwargs) -> None:
    sets = []
    vals: list = []
    for k, v in kwargs.items():
        sets.append(f"{k}=?")
        vals.append(json_col(v) if isinstance(v, dict) else v)
    vals.append(compare_id)
    conn = get_conn()
    conn.execute(f"UPDATE compare_runs SET {','.join(sets)} WHERE id=?", vals)
    conn.commit()


def list_compare_runs(limit: int = 20) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM compare_runs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Model Scores History ─────────────────────────────────────────────────────

def save_score_history(model_name: str, base_url: str, run_id: str,
                       total: float, capability: float,
                       authenticity: float, performance: float) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO model_scores_history
           (id, model_name, base_url, run_id, total_score,
            capability, authenticity, performance, recorded_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (new_id(), model_name, base_url, run_id,
         round(float(total) * 100),
         round(float(capability) * 100),
         round(float(authenticity) * 100),
         round(float(performance) * 100),
         now_iso()),
    )
    conn.commit()


def get_model_trend(model_name: str, limit: int = 20) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT * FROM model_scores_history
           WHERE model_name=? ORDER BY recorded_at DESC LIMIT ?""",
        (model_name, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_leaderboard(sort_by: str = "total_score", limit: int = 50) -> list[dict]:
    """Return latest score for each model, sorted by specified field."""
    valid_sorts = {"total_score", "capability", "authenticity", "performance"}
    if sort_by not in valid_sorts:
        sort_by = "total_score"
    conn = get_conn()
    rows = conn.execute(
        f"""SELECT h.* FROM model_scores_history h
            INNER JOIN (
                SELECT model_name, MAX(recorded_at) as latest
                FROM model_scores_history GROUP BY model_name
            ) g ON h.model_name = g.model_name AND h.recorded_at = g.latest
            ORDER BY h.{sort_by} DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Golden Baselines ────────────────────────────────────────────────────────

def get_baseline_by_run_id(run_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM golden_baselines WHERE source_run_id=? AND is_active=1",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return dict(row)


def create_baseline(run_id: str, model_name: str, display_name: str, notes: str = "") -> dict:
    conn = get_conn()

    # model_name 是唯一索引：如果已存在同名基准，覆盖它
    existing = conn.execute(
        "SELECT id FROM golden_baselines WHERE model_name=? AND is_active=1",
        (model_name,),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE golden_baselines SET is_active=0, updated_at=? WHERE id=?",
            (now_iso(), existing["id"]),
        )
        conn.commit()

    run = conn.execute(
        "SELECT id, suite_version, status FROM test_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    if not run:
        raise ValueError("run_id not found")
    
    # 允许在 queued 状态下创建（Mode 2：创建并排队）
    if run["status"] not in ("completed", "partial_failed", "queued"):
        raise ValueError("run is not completed")

    feature_rows = conn.execute(
        "SELECT feature_name, feature_value FROM extracted_features WHERE run_id=?",
        (run_id,),
    ).fetchall()
    feature_vector = {r["feature_name"]: float(r["feature_value"] or 0.0) for r in feature_rows} if feature_rows else {}

    score_rows = conn.execute(
        "SELECT dimension, score FROM score_breakdown WHERE run_id=?",
        (run_id,),
    ).fetchall()
    score_map_display = {r["dimension"]: float(r["score"] or 0.0) for r in score_rows}
    score_map_internal = {k: v / 100.0 for k, v in score_map_display.items()}

    theta_row = conn.execute(
        "SELECT theta_global FROM model_theta_history WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    theta = float(theta_row["theta_global"]) if theta_row and theta_row["theta_global"] is not None else None

    bid = new_id()
    ts = now_iso()
    conn.execute(
        """INSERT INTO golden_baselines
           (id, model_name, display_name, source_run_id, suite_version,
            feature_vector, total_score, capability_score, authenticity_score,
            performance_score, score_breakdown, theta, sample_count, notes,
            created_at, updated_at, is_active)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            bid,
            model_name,
            display_name,
            run_id,
            run["suite_version"],
            json_col(feature_vector),
            score_map_internal.get("total", 0.0),
            score_map_internal.get("capability", 0.0),
            score_map_internal.get("authenticity", 0.0),
            score_map_internal.get("performance", 0.0),
            json_col(score_map_internal),
            theta,
            1,
            notes,
            ts,
            ts,
            1,
        ),
    )
    conn.commit()
    return {
        "id": bid,
        "model_name": model_name,
        "display_name": display_name,
        "total_score": round(score_map_internal.get("total", 0.0) * 100),
        "capability_score": round(score_map_internal.get("capability", 0.0) * 100),
        "authenticity_score": round(score_map_internal.get("authenticity", 0.0) * 100),
        "performance_score": round(score_map_internal.get("performance", 0.0) * 100),
        "sample_count": 1,
        "notes": notes,
        "created_at": ts,
    }


def list_baselines(model_name: str | None = None, active_only: bool = True, limit: int = 100) -> list[dict]:
    conn = get_conn()
    where = []
    vals: list = []
    if model_name:
        where.append("model_name=?")
        vals.append(model_name)
    if active_only:
        where.append("is_active=1")
    sql = "SELECT * FROM golden_baselines"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    rows = conn.execute(sql, tuple(vals + [limit])).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["feature_vector"] = from_json_col(d.get("feature_vector")) or {}
        d["score_breakdown"] = from_json_col(d.get("score_breakdown")) or {}
        d["total_score"] = round(float(d.get("total_score", 0.0)) * 100)
        d["capability_score"] = round(float(d.get("capability_score", 0.0)) * 100)
        d["authenticity_score"] = round(float(d.get("authenticity_score", 0.0)) * 100)
        d["performance_score"] = round(float(d.get("performance_score", 0.0)) * 100)
        out.append(d)
    return out


def get_baseline(baseline_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM golden_baselines WHERE id=?", (baseline_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["feature_vector"] = from_json_col(d.get("feature_vector")) or {}
    d["score_breakdown"] = from_json_col(d.get("score_breakdown")) or {}
    return d


def get_latest_active_baseline(model_name: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        """SELECT * FROM golden_baselines
           WHERE model_name=? AND is_active=1
           ORDER BY created_at DESC LIMIT 1""",
        (model_name,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["feature_vector"] = from_json_col(d.get("feature_vector")) or {}
    d["score_breakdown"] = from_json_col(d.get("score_breakdown")) or {}
    return d


def delete_baseline(baseline_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM golden_baselines WHERE id=?", (baseline_id,))
    conn.commit()


def save_baseline_comparison(
    run_id: str,
    baseline_id: str,
    cosine_similarity: float,
    score_delta_total: float,
    score_delta_capability: float,
    score_delta_authenticity: float,
    score_delta_performance: float,
    verdict: str,
    details: dict,
    p_value: float | None = None,
) -> str:
    cid = new_id()
    conn = get_conn()
    conn.execute(
        """INSERT INTO baseline_comparisons
           (id, run_id, baseline_id, cosine_similarity,
            score_delta_total, score_delta_capability,
            score_delta_authenticity, score_delta_performance,
            verdict, p_value, details, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            cid,
            run_id,
            baseline_id,
            cosine_similarity,
            score_delta_total,
            score_delta_capability,
            score_delta_authenticity,
            score_delta_performance,
            verdict,
            p_value,
            json_col(details),
            now_iso(),
        ),
    )
    conn.commit()
    return cid


# ── Item bank / IRT stats / Theta history / Pairwise ───────────────────────

def upsert_item_bank(item_id: str, dimension: str,
                     anchor_flag: bool = False,
                     active: bool = True,
                     version: str = "v1") -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO item_bank
           (item_id, dimension, anchor_flag, active, version, created_at)
           VALUES (?,?,?,?,?,COALESCE((SELECT created_at FROM item_bank WHERE item_id=?), ?))""",
        (item_id, dimension, 1 if anchor_flag else 0, 1 if active else 0, version, item_id, now_iso()),
    )
    conn.commit()


def upsert_item_stat(item_id: str, dimension: str, a: float = 1.0,
                     b: float = 0.0, c: float | None = None,
                     info_score: float = 0.0, sample_size: int = 0) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO item_stats
           (id, item_id, dimension, irt_a, irt_b, irt_c, info_score, sample_size, last_calibrated_at)
           VALUES (COALESCE((SELECT id FROM item_stats WHERE item_id=?), ?),?,?,?,?,?,?,?,?)""",
        (item_id, new_id(), item_id, dimension, a, b, c, info_score, sample_size, now_iso()),
    )
    conn.commit()


def get_item_stat(item_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM item_stats WHERE item_id=?", (item_id,)).fetchone()
    return dict(row) if row else None


def list_item_stats(dimension: str | None = None) -> list[dict]:
    conn = get_conn()
    if dimension:
        rows = conn.execute(
            "SELECT * FROM item_stats WHERE dimension=? ORDER BY item_id", (dimension,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM item_stats ORDER BY item_id").fetchall()
    return [dict(r) for r in rows]


def save_theta_history(run_id: str, model_name: str, base_url: str,
                       theta_global: float, theta_global_ci_low: float,
                       theta_global_ci_high: float, theta_dims: dict,
                       percentile_global: float | None,
                       percentile_dims: dict | None,
                       calibration_version: str,
                       method: str = "rasch_1pl") -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO model_theta_history
           (id, run_id, model_name, base_url,
            theta_global, theta_global_ci_low, theta_global_ci_high,
            theta_dims_json, percentile_global, percentile_dims_json,
            calibration_version, method, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            new_id(), run_id, model_name, base_url,
            theta_global, theta_global_ci_low, theta_global_ci_high,
            json_col(theta_dims), percentile_global,
            json_col(percentile_dims) if percentile_dims is not None else None,
            calibration_version, method, now_iso(),
        ),
    )
    conn.commit()


def get_theta_by_run(run_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM model_theta_history WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["theta_dims_json"] = from_json_col(d.get("theta_dims_json")) or {}
    d["percentile_dims_json"] = from_json_col(d.get("percentile_dims_json")) or {}
    return d


def get_model_theta_trend(model_name: str, limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT * FROM model_theta_history
           WHERE model_name=? ORDER BY created_at DESC LIMIT ?""",
        (model_name, limit),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["theta_dims_json"] = from_json_col(d.get("theta_dims_json")) or {}
        d["percentile_dims_json"] = from_json_col(d.get("percentile_dims_json")) or {}
        out.append(d)
    return out


def get_theta_leaderboard(dimension: str = "global", limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT h.* FROM model_theta_history h
            INNER JOIN (
                SELECT model_name, MAX(created_at) as latest
                FROM model_theta_history GROUP BY model_name
            ) g ON h.model_name = g.model_name AND h.created_at = g.latest
            ORDER BY h.theta_global DESC LIMIT ?""",
        (limit,),
    ).fetchall()

    data = []
    for r in rows:
        d = dict(r)
        d["theta_dims_json"] = from_json_col(d.get("theta_dims_json")) or {}
        if dimension and dimension != "global":
            d["sort_theta"] = float(d["theta_dims_json"].get(dimension, {}).get("theta", d.get("theta_global", 0.0)))
        else:
            d["sort_theta"] = float(d.get("theta_global", 0.0))
        data.append(d)

    data.sort(key=lambda x: x.get("sort_theta", 0.0), reverse=True)
    return data[:limit]


def save_pairwise_result(run_id: str, model_a: str, model_b: str,
                         delta_theta: float, win_prob_a: float,
                         method: str = "bradley_terry",
                         details: dict | None = None) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO pairwise_results
           (id, run_id, model_a, model_b, delta_theta, win_prob_a, method, details, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            new_id(), run_id, model_a, model_b,
            delta_theta, win_prob_a, method,
            json_col(details) if details else None,
            now_iso(),
        ),
    )
    conn.commit()


def get_pairwise_by_run(run_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM pairwise_results WHERE run_id=? ORDER BY created_at DESC",
        (run_id,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = from_json_col(d.get("details"))
        out.append(d)
    return out


def save_calibration_snapshot(version: str, item_params_json: dict, notes: str = "") -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO calibration_snapshots
           (id, version, item_params_json, notes, created_at)
           VALUES (COALESCE((SELECT id FROM calibration_snapshots WHERE version=?), ?), ?, ?, ?, ?)
        """,
        (version, new_id(), version, json_col(item_params_json), notes, now_iso()),
    )
    conn.commit()


def get_latest_calibration_snapshot() -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM calibration_snapshots ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["item_params_json"] = from_json_col(d.get("item_params_json")) or {}
    return d


# ── Calibration Replay ───────────────────────────────────────────────────────

def create_calibration_replay(cases_json: dict) -> str:
    rid = new_id()
    conn = get_conn()
    conn.execute(
        """INSERT INTO calibration_replays
           (id, status, cases_json, created_at)
           VALUES (?,?,?,?)""",
        (rid, "queued", json_col(cases_json), now_iso()),
    )
    conn.commit()
    return rid


def get_calibration_replay(replay_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM calibration_replays WHERE id=?", (replay_id,)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["cases_json"] = from_json_col(d.get("cases_json")) or {}
    d["result_json"] = from_json_col(d.get("result_json"))
    return d


def list_calibration_replays(limit: int = 20) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM calibration_replays ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["cases_json"] = from_json_col(d.get("cases_json")) or {}
        d["result_json"] = from_json_col(d.get("result_json"))
        out.append(d)
    return out


def update_calibration_replay(replay_id: str, status: str, **kwargs) -> None:
    sets = ["status=?"]
    vals: list = [status]

    if status == "running" and "started_at" not in kwargs:
        sets.append("started_at=?")
        vals.append(now_iso())
    if status in ("completed", "failed") and "completed_at" not in kwargs:
        sets.append("completed_at=?")
        vals.append(now_iso())

    for k, v in kwargs.items():
        sets.append(f"{k}=?")
        if k in ("result_json",):
            vals.append(json_col(v) if v is not None else None)
        else:
            vals.append(v)

    vals.append(replay_id)
    conn = get_conn()
    conn.execute(f"UPDATE calibration_replays SET {','.join(sets)} WHERE id=?", vals)
    conn.commit()


# ── ELO and Pairwise ─────────────────────────────────────────────────────────

def save_pairwise_result(
    run_id: str,
    model_a: str,
    model_b: str,
    delta_theta: float,
    win_prob_a: float,
    method: str,
    details: dict,
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO pairwise_results 
        (id, run_id, model_a, model_b, delta_theta, win_prob_a, method, details, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            new_id(), run_id, model_a, model_b,
            delta_theta, win_prob_a, method,
            json_col(details), now_iso()
        )
    )
    conn.commit()


from app.analysis.elo import EloRecord

def get_elo(model_name: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM model_elo WHERE model_name = ?", (model_name,)).fetchone()
    return dict(row) if row else None


def upsert_elo(record: EloRecord) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO model_elo 
        (model_name, display_name, elo_rating, games_played, wins, losses, draws, peak_elo, last_run_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET
            display_name = excluded.display_name,
            elo_rating = excluded.elo_rating,
            games_played = excluded.games_played,
            wins = excluded.wins,
            losses = excluded.losses,
            draws = excluded.draws,
            peak_elo = excluded.peak_elo,
            last_run_id = excluded.last_run_id,
            updated_at = excluded.updated_at
        """,
        (
            record.model_name,
            record.display_name,
            record.elo_rating,
            record.games_played,
            record.wins,
            record.losses,
            record.draws,
            record.peak_elo,
            record.last_run_id,
            now_iso()
        )
    )
    conn.commit()


def list_elo(limit: int = 100) -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM model_elo ORDER BY elo_rating DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
