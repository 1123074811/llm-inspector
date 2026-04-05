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
    conn.execute(
        """INSERT INTO test_runs
           (id, base_url, api_key_encrypted, api_key_hash,
            model_name, test_mode, suite_version, status, created_at, metadata)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (run_id, base_url, api_key_encrypted, api_key_hash,
         model_name, test_mode, suite_version, "queued", now_iso(), json_col(metadata or {})),
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


def list_runs(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM test_runs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def set_run_cancel_requested(run_id: str, requested: bool = True) -> None:
    run = get_run(run_id)
    if not run:
        return
    metadata = run.get("metadata") or {}
    metadata["cancel_requested"] = bool(requested)
    conn = get_conn()
    conn.execute(
        "UPDATE test_runs SET metadata=? WHERE id=?",
        (json_col(metadata), run_id),
    )
    conn.commit()


def is_run_cancel_requested(run_id: str) -> bool:
    run = get_run(run_id)
    if not run:
        return False
    metadata = run.get("metadata") or {}
    return bool(metadata.get("cancel_requested", False))


def mark_run_retry(run_id: str) -> None:
    run = get_run(run_id)
    if not run:
        return
    metadata = run.get("metadata") or {}
    metadata["cancel_requested"] = False
    metadata["resume_from_existing"] = True
    conn = get_conn()
    conn.execute(
        """UPDATE test_runs
           SET status=?, started_at=NULL, completed_at=NULL,
               error_message=NULL, metadata=?
           WHERE id=?""",
        ("queued", json_col(metadata), run_id),
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
    params["_meta"] = meta

    conn.execute(
        """INSERT OR REPLACE INTO test_cases
           (id, category, name, system_prompt, user_prompt,
            expected_type, judge_method, params,
            max_tokens, n_samples, temperature, weight, enabled, suite_version)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            case["id"], case["category"], case["name"],
            case.get("system_prompt"), case["user_prompt"],
            case["expected_type"], case["judge_method"],
            json_col(params),
            case.get("max_tokens", 5), case.get("n_samples", 1),
            case.get("temperature", 0.0), case.get("weight", 1.0),
            1 if case.get("enabled", True) else 0,
            case.get("suite_version", "v1"),
        ),
    )
    conn.commit()


def load_cases(suite_version: str = "v1", test_mode: str = "standard") -> list[dict]:
    conn = get_conn()

    # Primary query for requested suite version
    rows = conn.execute(
        "SELECT * FROM test_cases WHERE suite_version=? AND enabled=1",
        (suite_version,),
    ).fetchall()

    # Fallback chain to avoid hard failure when requested suite has not been seeded yet
    if not rows and suite_version != "v2":
        rows = conn.execute(
            "SELECT * FROM test_cases WHERE suite_version='v2' AND enabled=1"
        ).fetchall()

    # Final fallback: any enabled cases
    if not rows:
        rows = conn.execute(
            "SELECT * FROM test_cases WHERE enabled=1"
        ).fetchall()

    cases = []
    for row in rows:
        c = dict(row)
        c["params"] = from_json_col(c.get("params")) or {}
        # quick mode: skip style and some refusal cases
        if test_mode == "quick" and c["category"] in ("style",):
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


# ── Benchmark Profiles ────────────────────────────────────────────────────────

def upsert_benchmark(name: str, suite_version: str,
                     feature_vector: dict, sample_count: int = 3) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO benchmark_profiles
           (id, benchmark_name, suite_version, feature_vector, sample_count, generated_at)
           VALUES (?,?,?,?,?,?)""",
        (new_id(), name, suite_version, json_col(feature_vector),
         sample_count, now_iso()),
    )
    conn.commit()


def get_benchmarks(suite_version: str = "v1") -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """SELECT * FROM benchmark_profiles
           WHERE suite_version=? ORDER BY benchmark_name""",
        (suite_version,),
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["feature_vector"] = from_json_col(d.get("feature_vector")) or {}
        result.append(d)
    return result


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
                         max_score: float = 100.0, details: dict | None = None) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO score_breakdown
           (id, run_id, dimension, score, max_score, details, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (new_id(), run_id, dimension, score, max_score,
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
         total, capability, authenticity, performance, now_iso()),
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
