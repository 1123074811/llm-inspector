"""
predetect/layer_l22_prompt_reconstruct.py — Layer 22: System Prompt Reconstruction

Passively reconstructs received system prompts from the existing v14
system_prompt_harvester output. Does NOT send new API requests.

If the upstream service injected a system prompt (common in proxy/wrapper
setups), the harvester will have captured fragments. This layer compiles
those fragments into a best-effort reconstruction for the report.

Default mode: Deep only (zero additional tokens).
"""
from __future__ import annotations

import json
import pathlib
from app.core.config import settings
from app.core.logging import get_logger
from app.core.schemas import LayerResult

logger = get_logger(__name__)


class Layer22PromptReconstruct:
    """Reconstruct system prompt from harvested fragments."""

    def run(self, adapter, model_name: str, run_id: str = "") -> LayerResult:
        evidence: list[str] = []
        fragments: list[str] = []

        # Try reading from the harvester's stored output
        try:
            fragments = self._load_harvester_output(run_id)
        except Exception as exc:
            logger.warning(
                "Layer22: could not load harvester output",
                run_id=run_id, error=str(exc),
            )

        if not fragments:
            evidence.append("No system prompt fragments found from harvester")
            return LayerResult(
                layer="Layer22/PromptReconstruct",
                confidence=0.0,
                identified_as=None,
                evidence=evidence,
                tokens_used=0,
            )

        # De-duplicate and compile
        unique_fragments = list(dict.fromkeys(fragments))  # preserve order, dedupe
        compiled = "\n".join(unique_fragments)

        # Heuristic: if fragments contain routing/system prompt instructions,
        # it strongly suggests a wrapper
        suspicious_keywords = [
            "你是一个", "you are", "你是", "你的任务",
            "system prompt", "你叫", "你的名字",
            "forward", "proxy", "route", "中转",
        ]
        hit_count = sum(1 for kw in suspicious_keywords if kw.lower() in compiled.lower())

        confidence = min(hit_count / len(suspicious_keywords) * 0.9, 0.7)
        if hit_count > 0:
            evidence.append(f"System prompt reconstruction found ({len(unique_fragments)} fragments)")
            for i, frag in enumerate(unique_fragments[:5]):
                evidence.append(f"  Fragment[{i}]: {frag[:120]}")
            if len(unique_fragments) > 5:
                evidence.append(f"  ... and {len(unique_fragments) - 5} more fragments")
            evidence.append(f"  Suspicious keyword hits: {hit_count}/{len(suspicious_keywords)}")

        return LayerResult(
            layer="Layer22/PromptReconstruct",
            confidence=round(confidence, 3),
            identified_as=None,
            evidence=evidence,
            tokens_used=0,
        )

    @staticmethod
    def _load_harvester_output(run_id: str) -> list[str]:
        """Load system prompt fragments from harvester stored data."""
        fragments: list[str] = []

        # Try trace file first (predetect traces)
        trace_path = pathlib.Path(settings.DATA_DIR) / "traces" / run_id / "predetect.jsonl"
        if trace_path.exists():
            with trace_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        evidence_list = record.get("evidence", [])
                        for ev in evidence_list:
                            if isinstance(ev, str) and len(ev) > 20:
                                fragments.append(ev)
                    except json.JSONDecodeError:
                        continue

        # Try system prompt result from DB (via adapter injection)
        try:
            from app.repository import repo
            run = repo.get_run(run_id)
            if run:
                sys_prompt_raw = run.get("system_prompt_result")
                if sys_prompt_raw:
                    if isinstance(sys_prompt_raw, str):
                        sys_prompt_raw = json.loads(sys_prompt_raw)
                    if isinstance(sys_prompt_raw, dict):
                        for val in sys_prompt_raw.values():
                            if isinstance(val, str) and len(val) > 20:
                                fragments.append(val)
        except Exception:
            pass

        return fragments
