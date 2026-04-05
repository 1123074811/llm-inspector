"""
Generate isomorphic benchmark cases for reasoning/token-control dimensions.

Usage:
  python backend/tools/generate_isomorphic_cases.py --preview
  python backend/tools/generate_isomorphic_cases.py --apply
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = ROOT / "backend" / "app" / "fixtures" / "suite_v2.json"


def _new_reason_case(
    case_id: str,
    name: str,
    prompt: str,
    target_pattern: str,
    key_constraints: list[str],
    boundary_signals: list[str],
    anti_pattern_signals: list[str],
) -> dict:
    return {
        "id": case_id,
        "category": "reasoning",
        "dimension": "reasoning",
        "tags": ["constraint_reasoning", "isomorphic"],
        "name": name,
        "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
        "user_prompt": prompt,
        "expected_type": "text",
        "judge_method": "constraint_reasoning",
        "max_tokens": 220,
        "n_samples": 2,
        "temperature": 0.0,
        "params": {
            "target_pattern": target_pattern,
            "key_constraints": key_constraints,
            "boundary_signals": boundary_signals,
            "anti_pattern_signals": anti_pattern_signals,
            "min_constraint_hits": 1,
            "require_boundary": True,
            "allow_anti_pattern": False,
        },
        "weight": 2.4,
    }


def _new_token_case(
    case_id: str,
    name: str,
    prompt: str,
    exact_chars: int,
    forbidden_chars: list[str],
) -> dict:
    return {
        "id": case_id,
        "category": "instruction",
        "dimension": "instruction",
        "tags": ["token_control", "isomorphic", "char_count", "forbidden_chars"],
        "name": name,
        "system_prompt": "严格执行硬约束，不要解释。",
        "user_prompt": prompt,
        "expected_type": "text",
        "judge_method": "text_constraints",
        "max_tokens": 180,
        "n_samples": 3,
        "temperature": 0.0,
        "params": {
            "exact_chars": exact_chars,
            "forbidden_chars": forbidden_chars,
        },
        "weight": 2.2,
    }


def build_cases() -> list[dict]:
    out: list[dict] = []

    # candy-isomorphic
    out.append(_new_reason_case(
        "reason_candy_iso_007",
        "candy_shape_flavor_iso_a",
        "袋中有三种口味糖果，每种有圆形和星形可触摸区分。请给出最少抓取总数，保证手里一定同时存在跨形状的苹果味与桃子味配对（圆苹果+星桃子，或圆桃子+星苹果）。请给出最不利边界证明。",
        r"\b21\b",
        ["可触摸区分", "最不利", "边界", "形状"],
        ["最不利", "上界", "保证", "边界"],
        ["盲抽", "随机抽取即可"],
    ))

    out.append(_new_reason_case(
        "reason_rope_iso_007",
        "rope_unsat_iso_a",
        "有一根燃烧不均匀的绳子，整根烧完60分钟，只有一个打火机。如何精确测15分钟？若无解请说明约束冲突与边界理由。",
        r"无解|无法",
        ["不均匀", "一根", "一个打火机"],
        ["无解", "无法", "约束冲突"],
        ["对折", "两头点燃后再"],
    ))

    out.append(_new_reason_case(
        "reason_mice_iso_007",
        "mice_ternary_iso_a",
        "240瓶酒有1瓶毒酒，毒发固定24小时。离宴会48小时，最少需要几只小白鼠确保定位毒酒？要求说明状态数建模。",
        r"\b5\b",
        ["48小时", "24小时", "两轮", "三种状态"],
        ["3^5", "243", "编码"],
        ["二进制", "8只"],
    ))

    # token-control isomorphic
    out.append(_new_token_case(
        "instr_token_iso_007",
        "apple_50char_no_de_shi_iso",
        "写一段介绍苹果的中文短文，要求：恰好50个中文字符（不计标点），且不能出现“的”和“是”。",
        50,
        ["的", "是"],
    ))

    out.append(_new_token_case(
        "instr_token_iso_008",
        "pear_42char_no_de_iso",
        "写一段介绍梨香气与口感的中文短文，要求：恰好42个中文字符（不计标点），且不能出现“的”。",
        42,
        ["的"],
    ))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes to suite_v2.json")
    parser.add_argument("--preview", action="store_true", help="print generated cases")
    args = parser.parse_args()

    data = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    existing_ids = {c.get("id") for c in cases}

    generated = build_cases()
    to_add = [c for c in generated if c["id"] not in existing_ids]

    if args.preview or (not args.apply):
        print(json.dumps({"to_add": to_add, "count": len(to_add)}, ensure_ascii=False, indent=2))

    if args.apply:
        new_data = copy.deepcopy(data)
        new_data["cases"] = cases + to_add
        SUITE_PATH.write_text(json.dumps(new_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Applied. Added {len(to_add)} cases to {SUITE_PATH}")


if __name__ == "__main__":
    main()
