"""Miscellaneous handlers (health, static, tools)."""
from __future__ import annotations

import json
import pathlib
from app.handlers.helpers import _json, _error
from app.core.logging import get_logger

logger = get_logger(__name__)


def handle_health(_path, _qs, _body) -> tuple:
    db_state = "ok"
    case_stats: dict[str, int] = {}
    try:
        from app.repository import repo
        conn = repo.get_conn()
        rows = conn.execute(
            "SELECT suite_version, COUNT(*) as n FROM test_cases GROUP BY suite_version"
        ).fetchall()
        case_stats = {row["suite_version"]: row["n"] for row in rows}
    except Exception:
        db_state = "degraded"

    from app.tasks.worker import active_count
    return _json({
        "status": "ok",
        "version": "1.0.0",
        "db": db_state,
        "cases": case_stats,
        "active_workers": active_count(),
    })


def _build_isomorphic_cases() -> list[dict]:
    return [
        {
            "id": "reason_candy_iso_007",
            "category": "reasoning",
            "dimension": "reasoning",
            "tags": ["constraint_reasoning", "isomorphic"],
            "name": "candy_shape_flavor_iso_a",
            "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
            "user_prompt": "袋中有三种口味糖果，每种有圆形和星形可触摸区分。请给出最少抓取总数，保证手里一定同时存在跨形状的苹果味与桃子味配对（圆苹果+星桃子，或圆桃子+星苹果）。请给出最不利边界证明。",
            "expected_type": "text",
            "judge_method": "constraint_reasoning",
            "max_tokens": 220,
            "n_samples": 2,
            "temperature": 0.0,
            "params": {
                "target_pattern": "\\b21\\b",
                "key_constraints": ["可触摸区分", "最不利", "边界", "形状"],
                "boundary_signals": ["最不利", "上界", "保证", "边界"],
                "anti_pattern_signals": ["盲抽", "随机抽取即可"],
                "min_constraint_hits": 1,
                "require_boundary": True,
                "allow_anti_pattern": False,
            },
            "weight": 2.4,
        },
        {
            "id": "reason_rope_iso_007",
            "category": "reasoning",
            "dimension": "reasoning",
            "tags": ["constraint_reasoning", "isomorphic"],
            "name": "rope_unsat_iso_a",
            "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
            "user_prompt": "有一根燃烧不均匀的绳子，整根烧完60分钟，只有一个打火机。如何精确测15分钟？若无解请说明约束冲突与边界理由。",
            "expected_type": "text",
            "judge_method": "constraint_reasoning",
            "max_tokens": 220,
            "n_samples": 2,
            "temperature": 0.0,
            "params": {
                "target_pattern": "无解|无法",
                "key_constraints": ["不均匀", "一根", "一个打火机"],
                "boundary_signals": ["无解", "无法", "约束冲突"],
                "anti_pattern_signals": ["对折", "两头点燃后再"],
                "min_constraint_hits": 1,
                "require_boundary": True,
                "allow_anti_pattern": False,
            },
            "weight": 2.4,
        },
        {
            "id": "reason_mice_iso_007",
            "category": "reasoning",
            "dimension": "reasoning",
            "tags": ["constraint_reasoning", "isomorphic"],
            "name": "mice_ternary_iso_a",
            "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
            "user_prompt": "240瓶酒有1瓶毒酒，毒发固定24小时。离宴会48小时，最少需要几只小白鼠确保定位毒酒？要求说明状态数建模。",
            "expected_type": "text",
            "judge_method": "constraint_reasoning",
            "max_tokens": 220,
            "n_samples": 2,
            "temperature": 0.0,
            "params": {
                "target_pattern": "\\b5\\b",
                "key_constraints": ["48小时", "24小时", "两轮", "三种状态"],
                "boundary_signals": ["3^5", "243", "编码"],
                "anti_pattern_signals": ["二进制", "8只"],
                "min_constraint_hits": 1,
                "require_boundary": True,
                "allow_anti_pattern": False,
            },
            "weight": 2.4,
        },
        {
            "id": "instr_token_iso_007",
            "category": "instruction",
            "dimension": "instruction",
            "tags": ["token_control", "isomorphic", "char_count", "forbidden_chars"],
            "name": "apple_50char_no_de_shi_iso",
            "system_prompt": "严格执行硬约束，不要解释。",
            "user_prompt": '写一段介绍苹果的中文短文，要求：恰好50个中文字符（不计标点），且不能出现"的"和"是"。',
            "expected_type": "text",
            "judge_method": "text_constraints",
            "max_tokens": 180,
            "n_samples": 3,
            "temperature": 0.0,
            "params": {"exact_chars": 50, "forbidden_chars": ["的", "是"]},
            "weight": 2.2,
        },
    ]


def handle_generate_isomorphic(_path, qs, _body) -> tuple:
    apply_flag = (qs.get("apply", ["false"])[0] or "false").lower() == "true"
    suite_path = pathlib.Path(__file__).parent.parent / "fixtures" / "suite_v2.json"
    if not suite_path.exists():
        return _error("suite_v2.json not found", 404)

    data = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    existing_ids = {c.get("id") for c in cases}
    generated = _build_isomorphic_cases()
    to_add = [c for c in generated if c["id"] not in existing_ids]

    if apply_flag and to_add:
        data["cases"] = cases + to_add
        suite_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return _json({
        "apply": apply_flag,
        "added_count": len(to_add) if apply_flag else 0,
        "preview_count": len(to_add),
        "to_add_ids": [c["id"] for c in to_add],
    })


def handle_static(path) -> tuple:
    # __file__ is at backend/app/handlers/misc.py
    # target is at llm-inspector/llm-inspector/frontend
    static_dir = pathlib.Path(__file__).parents[3] / "frontend"
    if path == "/" or path == "":
        target = static_dir / "index.html"
    else:
        target = static_dir / path.lstrip("/")

    if not target.is_file() or not str(target.resolve()).startswith(str(static_dir.resolve())):
        return 404, b"Not found", "text/plain"

    content_type = "text/html; charset=utf-8"
    if path.endswith(".css"):
        content_type = "text/css; charset=utf-8"
    elif path.endswith(".js"):
        content_type = "application/javascript; charset=utf-8"

    body = target.read_bytes()
    return 200, body, content_type
