"""
Export a simple radar chart (SVG) from run report scorecard.
No third-party dependency required.

Usage:
  python backend/tools/export_radar_svg.py --run-id <RUN_ID> --out backend/output/<RUN_ID>-radar.svg
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "llm_inspector.db"


def _load_report(run_id: str) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT details FROM reports WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"report not found for run_id={run_id}")
    return json.loads(row["details"])


def _polar_to_xy(cx: float, cy: float, r: float, angle: float) -> tuple[float, float]:
    return cx + r * math.cos(angle), cy + r * math.sin(angle)


def render_svg(report: dict) -> str:
    score = report.get("scorecard", {})
    breakdown = score.get("breakdown", {})
    # v4: 雷达图升级为9维度
    dims = [
        ("推理", float(breakdown.get("reasoning", 0.0))),
        ("编码", float(breakdown.get("coding", 0.0))),
        ("指令遵循", float(breakdown.get("instruction", 0.0))),
        ("安全性", float(breakdown.get("safety", 0.0))),
        ("知识", float(breakdown.get("knowledge_score", 0.0))),
        ("工具使用", float(breakdown.get("tool_use_score", 0.0))),
        ("一致性", float(breakdown.get("consistency", 0.0))),
        ("抗提取", float(breakdown.get("extraction_resistance", 0.0))),
        ("响应速度", float(breakdown.get("speed", 0.0))),
    ]

    w, h = 760, 560
    cx, cy = 380, 280
    max_r = 210

    n = len(dims)
    angles = [(-math.pi / 2) + i * (2 * math.pi / n) for i in range(n)]

    rings = []
    for lv in [2000, 4000, 6000, 8000, 10000]:
        r = max_r * lv / 10000.0
        pts = ["{:.1f},{:.1f}".format(*_polar_to_xy(cx, cy, r, a)) for a in angles]
        rings.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="#ddd" stroke-width="1" />')

    axes = []
    labels = []
    for (name, val), a in zip(dims, angles):
        x2, y2 = _polar_to_xy(cx, cy, max_r, a)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#bbb" stroke-width="1" />')
        lx, ly = _polar_to_xy(cx, cy, max_r + 24, a)
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="14" fill="#333" text-anchor="middle">{name}</text>')

    poly_pts = []
    val_labels = []
    for (name, val), a in zip(dims, angles):
        r = max_r * max(0.0, min(10000.0, val)) / 10000.0
        x, y = _polar_to_xy(cx, cy, r, a)
        poly_pts.append(f"{x:.1f},{y:.1f}")
        val_labels.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb" />')

    target = report.get("target") or {}
    title = f"综合能力雷达图 · {target.get('model', 'unknown')}"

    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">\n  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n  <text x=\"24\" y=\"36\" font-size=\"24\" font-weight=\"700\" fill=\"#111\">{title}</text>\n  <text x=\"24\" y=\"62\" font-size=\"14\" fill=\"#666\">run_id: {report.get('run_id')}</text>\n  {''.join(rings)}\n  {''.join(axes)}\n  <polygon points=\"{' '.join(poly_pts)}\" fill=\"rgba(37,99,235,0.20)\" stroke=\"#2563eb\" stroke-width=\"2\"/>\n  {''.join(val_labels)}\n  {''.join(labels)}\n  <g transform=\"translate(560,120)\">\n    <text x=\"0\" y=\"0\" font-size=\"18\" font-weight=\"700\" fill=\"#111\">评分摘要</text>\n    <text x=\"0\" y=\"32\" font-size=\"14\" fill=\"#333\">总分:{score.get('total_score', 0)}</text>\n    <text x=\"0\" y=\"56\" font-size=\"14\" fill=\"#333\">能力分:{score.get('capability_score', 0)}</text>\n    <text x=\"0\" y=\"80\" font-size=\"14\" fill=\"#333\">真实性:{score.get('authenticity_score', 0)}</text>\n    <text x=\"0\" y=\"104\" font-size=\"14\" fill=\"#333\">性能分:{score.get('performance_score', 0)}</text>\n    <text x=\"0\" y=\"128\" font-size=\"12\" fill=\"#666\">推理:{breakdown.get('reasoning', 0)}</text>\n    <text x=\"0\" y=\"146\" font-size=\"12\" fill=\"#666\">编码:{breakdown.get('coding', 0)}</text>\n    <text x=\"0\" y=\"164\" font-size=\"12\" fill=\"#666\">指令遵循:{breakdown.get('instruction', 0)}</text>\n    <text x=\"0\" y=\"182\" font-size=\"12\" fill=\"#666\">安全性:{breakdown.get('safety', 0)}</text>\n    <text x=\"0\" y=\"200\" font-size=\"12\" fill=\"#666\">工具使用:{breakdown.get('tool_use_score', 0)}</text>\n    <text x=\"0\" y=\"218\" font-size=\"12\" fill=\"#666\">一致性:{breakdown.get('consistency', 0)}</text>\n    <text x=\"0\" y=\"236\" font-size=\"12\" fill=\"#666\">抗提取:{breakdown.get('extraction_resistance', 0)}</text>\n    <text x=\"0\" y=\"254\" font-size=\"12\" fill=\"#666\">响应速度:{breakdown.get('speed', 0)}</text>\n  </g>\n</svg>\n"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = _load_report(args.run_id)
    svg = render_svg(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"SVG exported: {out}")
