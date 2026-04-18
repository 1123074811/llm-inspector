"""
scripts/build_suite_v13.py — Build suite_v13.json from benchmark datasets.

Usage:
    python build_suite_v13.py [--offline]   # offline = skip HF download, use embedded examples
    python build_suite_v13.py --from-hf     # try to download from HuggingFace

Output:
    backend/app/fixtures/suite_v13.json
    (manifest is at backend/app/_data/benchmarks/manifest.yaml)

The embedded fallback data exactly mirrors the suite_v13.json already written by
Phase 3 Stream A.  When --from-hf is given and the `datasets` package is installed,
the script will attempt to pull additional rows from HuggingFace and merge them with
the embedded base set (deduplicating by `id`).  If the download fails for any reason
the script falls back silently to the embedded data and exits with code 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent          # backend/
_FIXTURES_DIR = _REPO_ROOT / "app" / "fixtures"
_MANIFEST_PATH = _REPO_ROOT / "app" / "_data" / "benchmarks" / "manifest.yaml"
_OUTPUT_PATH = _FIXTURES_DIR / "suite_v13.json"
_EMBEDDED_PATH = _FIXTURES_DIR / "suite_v13.json"   # same file — rebuild in place

VERSION = "v13"
GENERATED_AT = "2026-04-18T00:00:00Z"
DESCRIPTION = (
    "LLM Inspector v13 test suite — includes cases from GPQA Diamond, AIME 2024, "
    "MATH-500, SWE-bench Lite, MMLU-Pro, CMMLU, JailbreakBench. "
    "All cases include source_ref and license. "
    "IRT params are preliminary (calibrated=false) pending Phase 2 fitting."
)

# -- Default IRT params for un-calibrated cases
_DEFAULT_IRT = {"irt_a": 1.0, "irt_b": 0.0, "irt_c": 0.25, "calibrated": False}


# ---------------------------------------------------------------------------
# HuggingFace helpers (optional — only used with --from-hf)
# ---------------------------------------------------------------------------

def _hf_available() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def _load_gpqa_from_hf(n: int = 15) -> list[dict[str, Any]]:
    """Pull n cases from GPQA Diamond via HuggingFace datasets library."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
    cases: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        correct_letter = row.get("Correct Answer", "A")
        options = {
            "A": row.get("Incorrect Answer 1", ""),
            "B": row.get("Incorrect Answer 2", ""),
            "C": row.get("Incorrect Answer 3", ""),
            "D": row.get("Correct Answer", ""),
        }
        # Shuffle so correct answer isn't always D
        import random
        letters = list(options.keys())
        random.shuffle(letters)
        mapping = {letters[j]: list(options.values())[j] for j in range(4)}
        correct_new = [k for k, v in mapping.items() if v == row.get("Correct Answer", "")][0]

        prompt = row.get("Question", "")
        for letter, text in sorted(mapping.items()):
            prompt += f"\n{letter}) {text}"
        prompt += "\n\nAnswer with the letter only."

        cases.append({
            "id": f"gpqa_hf_{i+1:03d}",
            "category": "reasoning",
            "dimension": "reasoning",
            "name": f"gpqa_hf_q{i+1}",
            "source_ref": "gpqa_diamond",
            "license": "MIT",
            "system_prompt": None,
            "user_prompt": prompt,
            "expected_type": "text",
            "judge_method": "regex_match",
            "max_tokens": 10,
            "n_samples": 1,
            "temperature": 0.0,
            "params": {
                "pattern": f"\\b{correct_new}\\b",
                "match_means_pass": True,
                "_meta": {
                    "mode_level": "deep",
                    "subject": row.get("Subdomain", "science"),
                    "skill_vector": {"scientific_reasoning": 1},
                },
            },
            "weight": 3.0,
            **_DEFAULT_IRT,
        })
    return cases


def _load_mmlu_pro_from_hf(per_subject: int = 2) -> list[dict[str, Any]]:
    """Pull per_subject cases per subject from MMLU-Pro."""
    from datasets import load_dataset  # type: ignore

    subjects = [
        "math", "physics", "chemistry", "biology", "computer science",
        "law", "economics", "psychology", "history", "engineering",
    ]
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    cases: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for row in ds:
        subj = row.get("category", "").lower()
        matched = next((s for s in subjects if s in subj), None)
        if matched is None:
            continue
        if counts.get(matched, 0) >= per_subject:
            continue
        counts[matched] = counts.get(matched, 0) + 1

        options = row.get("options", [])
        letters = "ABCDEFGHIJ"
        prompt = row.get("question", "")
        for j, opt in enumerate(options):
            prompt += f"\n{letters[j]}) {opt}"
        correct_idx = row.get("answer_index", 0)
        correct_letter = letters[correct_idx] if correct_idx < len(letters) else "A"
        prompt += "\n\nAnswer with the letter only."

        slug = matched.replace(" ", "_")
        idx = counts[matched]
        cases.append({
            "id": f"mmlu_pro_hf_{slug}_{idx:02d}",
            "category": "knowledge",
            "dimension": "knowledge",
            "name": f"mmlu_pro_hf_{slug}_{idx}",
            "source_ref": "mmlu_pro",
            "license": "MIT",
            "system_prompt": None,
            "user_prompt": prompt,
            "expected_type": "text",
            "judge_method": "regex_match",
            "max_tokens": 10,
            "n_samples": 1,
            "temperature": 0.0,
            "params": {
                "pattern": f"\\b{correct_letter}\\b",
                "match_means_pass": True,
                "_meta": {
                    "mode_level": "standard",
                    "subject": matched,
                    "skill_vector": {"knowledge_recall": 1},
                },
            },
            "weight": 2.0,
            **_DEFAULT_IRT,
        })
        if all(counts.get(s, 0) >= per_subject for s in subjects):
            break

    return cases


# ---------------------------------------------------------------------------
# Build logic
# ---------------------------------------------------------------------------

def _load_embedded() -> list[dict[str, Any]]:
    """Load the already-written suite_v13.json cases as the base set."""
    if _EMBEDDED_PATH.exists():
        with open(_EMBEDDED_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("cases", [])
    return []


def build(use_hf: bool = False) -> list[dict[str, Any]]:
    """Return the merged case list."""
    cases = _load_embedded()
    existing_ids = {c["id"] for c in cases}

    if use_hf and _hf_available():
        print("[build_suite_v13] Attempting HuggingFace download …", flush=True)
        try:
            new_cases: list[dict[str, Any]] = []
            new_cases.extend(_load_gpqa_from_hf(n=5))
            new_cases.extend(_load_mmlu_pro_from_hf(per_subject=1))
            added = 0
            for c in new_cases:
                if c["id"] not in existing_ids:
                    cases.append(c)
                    existing_ids.add(c["id"])
                    added += 1
            print(f"[build_suite_v13] Added {added} cases from HuggingFace.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[build_suite_v13] HF download failed ({exc}); using embedded data.", flush=True)
    elif use_hf:
        print(
            "[build_suite_v13] `datasets` package not installed. "
            "Run `pip install datasets` for HF support. Using embedded data.",
            flush=True,
        )

    return cases


def write_suite(cases: list[dict[str, Any]], output: Path = _OUTPUT_PATH) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suite = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": DESCRIPTION,
        "manifest_ref": "app/_data/benchmarks/manifest.yaml",
        "cases": cases,
    }
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(suite, fh, ensure_ascii=False, indent=2)
    print(f"[build_suite_v13] Wrote {len(cases)} cases to {output}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build backend/app/fixtures/suite_v13.json from benchmark datasets."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Skip HuggingFace download, use embedded example data only (default).",
    )
    parser.add_argument(
        "--from-hf",
        dest="from_hf",
        action="store_true",
        default=False,
        help="Attempt to download additional cases from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_PATH,
        help=f"Output path (default: {_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="After writing, validate that all required fields are present.",
    )
    args = parser.parse_args()

    cases = build(use_hf=args.from_hf and not args.offline)
    write_suite(cases, output=args.output)

    if args.validate:
        _validate(args.output)


def _validate(path: Path) -> None:
    """Basic structural validation of the output file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data.get("cases", [])
    required_top = ["id", "category", "dimension", "name", "source_ref", "license",
                    "user_prompt", "judge_method", "irt_a", "irt_b", "irt_c", "calibrated"]
    errors: list[str] = []
    for c in cases:
        for field in required_top:
            if field not in c:
                errors.append(f"[{c.get('id', '?')}] missing field: {field}")
        meta = c.get("params", {}).get("_meta", {})
        if "mode_level" not in meta:
            errors.append(f"[{c.get('id', '?')}] missing params._meta.mode_level")
        ml = meta.get("mode_level", "")
        if ml not in ("quick", "standard", "deep"):
            errors.append(f"[{c.get('id', '?')}] invalid mode_level: {ml!r}")
    if errors:
        print(f"[validate] {len(errors)} error(s):", flush=True)
        for e in errors[:20]:
            print(f"  {e}", flush=True)
        sys.exit(1)
    else:
        print(f"[validate] All {len(cases)} cases passed validation.", flush=True)


if __name__ == "__main__":
    main()
