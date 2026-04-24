#!/usr/bin/env python3
"""
scripts/sample_timing_references.py — Sample real timing and distribution data for L18/L19.

Usage:
    python backend/scripts/sample_timing_references.py \\
        --base-url https://api.openai.com/v1 \\
        --api-key sk-... \\
        --family gpt \\
        --model gpt-4o \\
        --samples 100 \\
        --output backend/app/_data/timing_refs.json

This script sends N timing probes to the specified API and computes:
  - TTFT (Time To First Token) mean/std
  - TPS (Tokens Per Second) mean/std
  - Response length statistics
  - 4-gram repetition rate

Results are merged into the existing timing_refs.json file.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--family", required=True, help="Model family name (e.g. gpt, claude)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", default="backend/app/_data/timing_refs.json")
    args = parser.parse_args()

    print(f"Sampling {args.samples} probes for family={args.family} model={args.model}")
    results = []
    for i in range(args.samples):
        print(f"  Probe {i+1}/{args.samples}...", end=" ", flush=True)
        r = send_probe(args.base_url, args.api_key, args.model)
        if r:
            results.append(r)
            print(f"TTFT={r['ttft_ms']:.0f}ms TPS={r['tps']:.1f}")
        else:
            print("FAILED")
        time.sleep(0.5)  # Rate limit courtesy

    if not results:
        print("No successful probes. Exiting.", file=sys.stderr)
        sys.exit(1)

    ttfts = [r["ttft_ms"] for r in results]
    tpss = [r["tps"] for r in results]
    texts = [r["content"] for r in results]
    lengths = [len(t.split()) for t in texts]

    family_data = {
        "sampled": True,
        "sample_size": len(results),
        "sampled_at": datetime.utcnow().isoformat() + "Z",
        "model_version": args.model,
        "ttft_ms_mean": statistics.mean(ttfts),
        "ttft_ms_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
        "tps_mean": statistics.mean(tpss),
        "tps_std": statistics.stdev(tpss) if len(tpss) > 1 else 0,
        "avg_response_len_words": statistics.mean(lengths),
        "repetition_rate_4gram": compute_4gram_repetition(texts),
    }

    # Load existing file and merge
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    existing = {}
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)

    if "families" not in existing:
        existing["families"] = {}
    existing["families"][args.family] = family_data
    existing.setdefault("_provenance", {})["last_updated"] = datetime.utcnow().isoformat() + "Z"

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.output}")
    print(f"  TTFT: mean={family_data['ttft_ms_mean']:.1f}ms std={family_data['ttft_ms_std']:.1f}ms")
    print(f"  TPS:  mean={family_data['tps_mean']:.1f} std={family_data['tps_std']:.1f}")


if __name__ == "__main__":
    main()
