#!/usr/bin/env python3
"""
scripts/sample_timing_references.py — Sample real timing and distribution data for L18/L19.

Usage (single family):
    python backend/scripts/sample_timing_references.py \\
        --base-url https://api.openai.com/v1 --api-key sk-... \\
        --family gpt --model gpt-4o-mini --samples 100

Usage (batch, env-driven, v17 Phase 4):
    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \\
    python backend/scripts/sample_timing_references.py --all --samples 100

This script sends N timing probes to the specified API and computes:
  - TTFT (Time To First Token) mean/std
  - TPS (Tokens Per Second) mean/std + p10/p25/p50/p75/p90 quantiles
  - Response length statistics
  - 4-gram repetition rate
  - Raw-data SHA256 for tamper-evidence

Results are merged into ``backend/app/_data/timing_refs.json`` and the
``_provenance`` section is rewritten so the placeholder marker disappears
as soon as at least one real sample lands.  v17 L18/L19 layers refuse to
treat KL/Wasserstein scores as evidence until ``sampled: True`` is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime


# Default per-family sampling targets for the --all batch mode.  Each entry
# names the env var carrying the API key and a representative model id.
DEFAULT_FAMILY_TARGETS = {
    "gpt": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "auth": "bearer",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-5-haiku-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "auth": "x-api-key",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
        "env_key": "GOOGLE_API_KEY",
        "auth": "google-key",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-2",
        "env_key": "XAI_API_KEY",
        "auth": "bearer",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "auth": "bearer",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "auth": "bearer",
    },
}


def _quantiles(values: list[float]) -> dict[str, float]:
    """Return p10/p25/p50/p75/p90 of ``values`` (empty → all zeros)."""
    if not values:
        return {k: 0.0 for k in ("p10", "p25", "p50", "p75", "p90")}
    s = sorted(values)
    n = len(s)

    def q(p):
        # Type 7 / NumPy default linear interpolation.
        if n == 1:
            return float(s[0])
        h = (n - 1) * p
        lo = int(h)
        hi = min(lo + 1, n - 1)
        frac = h - lo
        return float(s[lo]) * (1 - frac) + float(s[hi]) * frac

    return {
        "p10": round(q(0.10), 3),
        "p25": round(q(0.25), 3),
        "p50": round(q(0.50), 3),
        "p75": round(q(0.75), 3),
        "p90": round(q(0.90), 3),
    }


def _sha256_of_records(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PROBE_PROMPT = "Explain briefly what a large language model is. Be concise."


def send_probe(base_url: str, api_key: str, model: str, timeout: float = 30) -> dict | None:
    """Send one timing probe request and return timing stats."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": 100,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            t_first_byte = time.time()
            body = json.loads(resp.read())
            t_done = time.time()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
        ttft_ms = (t_first_byte - t0) * 1000
        total_s = t_done - t0
        tps = completion_tokens / total_s if total_s > 0 else 0
        return {"ttft_ms": ttft_ms, "tps": tps, "content": content,
                "completion_tokens": completion_tokens}
    except Exception as e:
        print(f"  Probe failed: {e}", file=sys.stderr)
        return None


def compute_4gram_repetition(texts: list[str]) -> float:
    from collections import Counter
    all_ngrams = []
    for text in texts:
        words = text.split()
        for i in range(len(words) - 3):
            all_ngrams.append(" ".join(words[i:i+4]))
    if not all_ngrams:
        return 0.0
    counts = Counter(all_ngrams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(all_ngrams)


def _sample_one_family(family: str, base_url: str, api_key: str,
                       model: str, samples: int) -> dict | None:
    """Run ``samples`` probes against one family.  Returns family_data dict."""
    print(f"Sampling {samples} probes for family={family} model={model}")
    results = []
    for i in range(samples):
        print(f"  Probe {i+1}/{samples}...", end=" ", flush=True)
        r = send_probe(base_url, api_key, model)
        if r:
            results.append(r)
            print(f"TTFT={r['ttft_ms']:.0f}ms TPS={r['tps']:.1f}")
        else:
            print("FAILED")
        time.sleep(0.5)

    if not results:
        print(f"  ! family={family}: no successful probes", file=sys.stderr)
        return None

    ttfts = [r["ttft_ms"] for r in results]
    tpss = [r["tps"] for r in results]
    texts = [r["content"] for r in results]
    lengths = [len(t.split()) for t in texts]

    return {
        "sampled": True,
        "sample_size": len(results),
        "sampled_at": datetime.utcnow().isoformat() + "Z",
        "model_version": model,
        "ttft_ms_mean": round(statistics.mean(ttfts), 2),
        "ttft_ms_std": round(statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0, 2),
        "ttft_ms_quantiles": _quantiles(ttfts),
        "tps_mean": round(statistics.mean(tpss), 2),
        "tps_std": round(statistics.stdev(tpss) if len(tpss) > 1 else 0.0, 2),
        "tps_quantiles": _quantiles(tpss),
        "avg_response_len_words": round(statistics.mean(lengths), 1),
        "repetition_rate_4gram": round(compute_4gram_repetition(texts), 4),
        "raw_data_sha256": _sha256_of_records(
            [{"ttft": round(r["ttft_ms"], 2),
              "tps": round(r["tps"], 3),
              "len": len(r["content"].split())} for r in results]
        ),
    }


def _save_results(output: str, updates: dict[str, dict]) -> None:
    """Merge ``updates`` (family → family_data) into the timing_refs file and
    refresh the ``_provenance`` block so PLACEHOLDER markers go away.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    existing: dict = {}
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing.setdefault("families", {})
    for family, data in updates.items():
        existing["families"][family] = data

    # Refresh provenance: drop the v15/v16 PLACEHOLDER marker once any real
    # sample lands, but record the original placeholder timestamp for audit.
    prov = existing.setdefault("_provenance", {})
    if prov.get("sampling_required"):
        prov["placeholder_replaced_at"] = datetime.utcnow().isoformat() + "Z"
    prov["note"] = (
        "v17 Phase 4: real samples collected via "
        "scripts/sample_timing_references.py.  See "
        "UPGRADE_PLAN_V17.md Phase 4 for refresh cadence."
    )
    prov["last_updated"] = datetime.utcnow().isoformat() + "Z"
    prov["sampling_required"] = False
    prov["version"] = "v17.0-self-measurement"
    prov["sampling_script"] = "backend/scripts/sample_timing_references.py"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def _print_summary(family: str, data: dict) -> None:
    print(f"\n[{family}] sample_size={data['sample_size']}")
    print(f"  TTFT mean={data['ttft_ms_mean']:.1f}ms std={data['ttft_ms_std']:.1f}ms "
          f"p50={data['ttft_ms_quantiles']['p50']:.0f}ms "
          f"p90={data['ttft_ms_quantiles']['p90']:.0f}ms")
    print(f"  TPS  mean={data['tps_mean']:.1f}  p50={data['tps_quantiles']['p50']:.1f}")
    print(f"  raw_data_sha256={data['raw_data_sha256'][:16]}...")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--family", help="Model family name (e.g. gpt, claude)")
    parser.add_argument("--model")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default="backend/app/_data/timing_refs.json")
    parser.add_argument("--all", action="store_true",
                        help="Batch mode: sample every family from DEFAULT_FAMILY_TARGETS "
                             "using API keys read from environment variables.")
    args = parser.parse_args()

    updates: dict[str, dict] = {}

    if args.all:
        for family, target in DEFAULT_FAMILY_TARGETS.items():
            key = os.environ.get(target["env_key"])
            if not key:
                print(f"  - skip family={family}: env var {target['env_key']} unset")
                continue
            data = _sample_one_family(
                family, target["base_url"], key, target["model"], args.samples
            )
            if data:
                updates[family] = data
                _print_summary(family, data)
    else:
        if not (args.base_url and args.api_key and args.family and args.model):
            parser.error("--base-url, --api-key, --family, --model are required "
                         "unless --all is given")
        data = _sample_one_family(args.family, args.base_url, args.api_key,
                                  args.model, args.samples)
        if data:
            updates[args.family] = data
            _print_summary(args.family, data)

    if not updates:
        print("No families sampled successfully. Exiting.", file=sys.stderr)
        sys.exit(1)

    _save_results(args.output, updates)
    print(f"\n✓ Saved {len(updates)} families to {args.output}")


if __name__ == "__main__":
    main()
