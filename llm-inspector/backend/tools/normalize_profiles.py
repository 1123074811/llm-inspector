#!/usr/bin/env python3
"""
Normalize benchmark profiles: fill missing feature dimensions with vendor-level
means, and add required core fields.

Usage:
    python -m tools.normalize_profiles [--in-place]
    python -m tools.normalize_profiles --dry-run

Run from: llm-inspector/backend/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "app" / "fixtures"
PROFILES_PATH = FIXTURES_DIR / "benchmarks" / "default_profiles.json"

# Core feature keys that every profile should have for similarity comparison.
CORE_FEATURE_KEYS: list[str] = [
    "protocol_success_rate",
    "instruction_pass_rate",
    "exact_match_rate",
    "json_valid_rate",
    "system_obedience_rate",
    "param_compliance_rate",
    "temperature_param_effective",
    "refusal_rate",
    "disclaimer_rate",
    "identity_consistency_pass_rate",
    "antispoof_identity_detect_rate",
    "antispoof_override_leak_rate",
    "avg_markdown_score",
    "avg_response_length",
    "adversarial_spoof_signal_rate",
    "latency_mean_ms",
]

# Global fallback means (used when no vendor-level mean is available).
GLOBAL_MEANS: dict[str, float] = {
    "identity_consistency_pass_rate": 0.75,
    "antispoof_identity_detect_rate": 0.35,
    "antispoof_override_leak_rate": 0.20,
    "adversarial_spoof_signal_rate": 0.15,
    "latency_mean_ms": 1500.0,
}

# Vendor-level defaults for fields that may be absent.
VENDOR_DEFAULTS: dict[str, dict[str, float]] = {
    "OpenAI": {
        "identity_consistency_pass_rate": 0.80,
        "antispoof_identity_detect_rate": 0.28,
        "antispoof_override_leak_rate": 0.15,
        "adversarial_spoof_signal_rate": 0.08,
        "latency_mean_ms": 800.0,
    },
    "Anthropic": {
        "identity_consistency_pass_rate": 0.82,
        "antispoof_identity_detect_rate": 0.22,
        "antispoof_override_leak_rate": 0.12,
        "adversarial_spoof_signal_rate": 0.05,
        "latency_mean_ms": 1200.0,
    },
    "Google": {
        "identity_consistency_pass_rate": 0.76,
        "antispoof_identity_detect_rate": 0.33,
        "antispoof_override_leak_rate": 0.20,
        "adversarial_spoof_signal_rate": 0.12,
        "latency_mean_ms": 1100.0,
    },
    "DeepSeek": {
        "identity_consistency_pass_rate": 0.74,
        "antispoof_identity_detect_rate": 0.40,
        "antispoof_override_leak_rate": 0.24,
        "adversarial_spoof_signal_rate": 0.10,
        "latency_mean_ms": 900.0,
    },
    "Alibaba": {
        "identity_consistency_pass_rate": 0.74,
        "antispoof_identity_detect_rate": 0.39,
        "antispoof_override_leak_rate": 0.23,
        "adversarial_spoof_signal_rate": 0.10,
        "latency_mean_ms": 950.0,
    },
    "Zhipu": {
        "identity_consistency_pass_rate": 0.73,
        "antispoof_identity_detect_rate": 0.42,
        "antispoof_override_leak_rate": 0.25,
        "adversarial_spoof_signal_rate": 0.12,
        "latency_mean_ms": 1300.0,
    },
    "Meta": {
        "identity_consistency_pass_rate": 0.70,
        "antispoof_identity_detect_rate": 0.50,
        "antispoof_override_leak_rate": 0.32,
        "adversarial_spoof_signal_rate": 0.18,
        "latency_mean_ms": 2000.0,
    },
    "Mistral AI": {
        "identity_consistency_pass_rate": 0.72,
        "antispoof_identity_detect_rate": 0.46,
        "antispoof_override_leak_rate": 0.28,
        "adversarial_spoof_signal_rate": 0.15,
        "latency_mean_ms": 1600.0,
    },
    "default": {
        "identity_consistency_pass_rate": 0.75,
        "antispoof_identity_detect_rate": 0.35,
        "antispoof_override_leak_rate": 0.20,
        "adversarial_spoof_signal_rate": 0.15,
        "latency_mean_ms": 1500.0,
    },
}

# Known provider names (partial match).
PROVIDER_KEYWORDS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "alibaba": "Alibaba",
    "qwen": "Alibaba",
    "zhipu": "Zhipu",
    "glm": "Zhipu",
    "meta": "Meta",
    "llama": "Meta",
    "mistral": "Mistral AI",
    "moonshot": "Moonshot AI",
    "kimi": "Moonshot AI",
}


def _detect_provider(profile_name: str) -> str:
    name_lower = profile_name.lower()
    for kw, provider in PROVIDER_KEYWORDS.items():
        if kw in name_lower:
            return provider
    return "default"


def _get_defaults_for_profile(profile_name: str) -> dict[str, float]:
    provider = _detect_provider(profile_name)
    vendor_defaults = VENDOR_DEFAULTS.get(provider, VENDOR_DEFAULTS["default"])
    result = {}
    for key in CORE_FEATURE_KEYS:
        if key in GLOBAL_MEANS:
            result[key] = vendor_defaults.get(key, GLOBAL_MEANS[key])
    return result


def normalize_profiles(data: dict) -> tuple[dict, list[dict]]:
    """
    Fill missing core feature keys in all profiles.
    Returns (normalized_data, change_log).
    """
    profiles = data.get("benchmarks", [])
    change_log: list[dict] = []

    for profile in profiles:
        name = profile.get("name", "?")
        vector = profile.setdefault("feature_vector", {})
        added: dict[str, float] = {}
        defaults = _get_defaults_for_profile(name)

        for key in CORE_FEATURE_KEYS:
            if key not in vector:
                value = defaults.get(key, GLOBAL_MEANS.get(key, 0.5))
                vector[key] = value
                added[key] = value

        if added:
            change_log.append({
                "profile": name,
                "provider": _detect_provider(name),
                "added": added,
            })

    # Update metadata
    data["normalized_at"] = "2026-04-05"
    data["normalized_version"] = "v1.1"

    return data, change_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize benchmark profiles")
    parser.add_argument("--in-place", action="store_true",
                        help="Write changes back to the input file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without writing")
    parser.add_argument("--input", type=pathlib.Path,
                        default=PROFILES_PATH,
                        help=f"Input JSON file (default: {PROFILES_PATH})")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    normalized, changes = normalize_profiles(data)

    print(f"Loaded {len(normalized.get('benchmarks', []))} profiles from {input_path.name}")
    print(f"Changes needed: {len(changes)} profiles")

    if changes:
        print(f"\nTop 5 changes:")
        for ch in changes[:5]:
            print(f"  {ch['profile']}: +{list(ch['added'].keys())}")
        if len(changes) > 5:
            print(f"  ... and {len(changes) - 5} more")

    if args.dry_run:
        print("\n(Dry run — no files written)")
        return

    if args.in_place:
        backup = input_path.with_suffix(".json.bak")
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        print(f"\nWritten: {input_path}")
        print(f"Backup:  {backup}")
    else:
        output = pathlib.Path("normalized_profiles.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        print(f"\nWritten: {output}")


if __name__ == "__main__":
    main()
