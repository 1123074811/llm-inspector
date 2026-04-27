"""
runner/model_registry_sync.py — v17 Phase 6: Tier 1+2 model registry sync.

Pulls model metadata from upstream sources and merges it into
``model_registry`` via :mod:`app.repository.registry_repo`.  Designed to
run every 6 hours (UPGRADE_PLAN_V17.md §9), but can also be invoked
ad-hoc via ``python -m app.runner.model_registry_sync --once``.

Sources (in order of priority — see ``registry_repo._SOURCE_PRIORITY``):

  Tier 1 (official endpoints, highest authority):
    GET https://api.openai.com/v1/models                  ``OPENAI_API_KEY``
    GET https://api.anthropic.com/v1/models               ``ANTHROPIC_API_KEY``
    GET https://api.x.ai/v1/models                        ``XAI_API_KEY``
    GET https://api.deepseek.com/v1/models                ``DEEPSEEK_API_KEY``
    GET https://api.mistral.ai/v1/models                  ``MISTRAL_API_KEY``
    GET https://generativelanguage.googleapis.com/...     ``GOOGLE_API_KEY``

  Tier 2 (aggregator, broad coverage including pricing):
    GET https://openrouter.ai/api/v1/models               ``OPENROUTER_API_KEY``
                                                          (or unauthenticated)

Diff semantics:
  * model seen in this sync         → upsert + status='active', last_seen_at=now
  * model previously known but not seen → status remains; deprecation logic
    is intentionally conservative — we only flip status='deprecated' when
    ``deprecated_after_misses`` consecutive sync windows have elapsed
    without seeing the model.  ``last_synced_at`` on the registry row is
    used to count missed windows, so a single misfire from one source
    does not deprecate a still-live model.

Network behaviour:
  * stdlib urllib only (matches ``adapters/openai_compat.py``)
  * each upstream is best-effort: connection / 4xx / 5xx errors log a
    warning but do not abort the overall sync
  * SSL context uses certifi when available
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from app.core.logging import get_logger
from app.repository import registry_repo

logger = get_logger(__name__)


# ── HTTP helpers ────────────────────────────────────────────────────────────


def _build_ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import]
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _build_ssl_ctx()


def _http_get_json(url: str, headers: dict[str, str], timeout: int = 20) -> Any | None:
    """GET ``url`` and return parsed JSON, or None on any failure."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.warning(
            "model_registry_sync: HTTP error",
            url=url, status=e.code, reason=str(e.reason),
        )
        return None
    except urllib.error.URLError as e:
        logger.warning("model_registry_sync: URL error", url=url, error=str(e.reason))
        return None
    except Exception as e:
        logger.warning("model_registry_sync: parse error", url=url, error=str(e)[:200])
        return None


# ── Adapters ────────────────────────────────────────────────────────────────


@dataclass
class SyncResult:
    """Outcome of a single sync source."""
    source: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    errors: int = 0
    skipped_reason: str | None = None        # set when the source is bypassed
    seen_model_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "fetched": self.fetched,
            "inserted": self.inserted,
            "updated": self.updated,
            "errors": self.errors,
            "skipped_reason": self.skipped_reason,
            "seen_model_ids_count": len(self.seen_model_ids),
        }


# Tier 1: OpenAI-compatible vendors (id, created, owned_by) ──────────────────

_OPENAI_COMPATIBLE_VENDORS = (
    # (vendor_key, base_url, env_key, family_default)
    ("openai",   "https://api.openai.com/v1",       "OPENAI_API_KEY",   "openai_api"),
    ("xai",      "https://api.x.ai/v1",             "XAI_API_KEY",      "xai_api"),
    ("deepseek", "https://api.deepseek.com/v1",     "DEEPSEEK_API_KEY", "deepseek_api"),
    ("mistral",  "https://api.mistral.ai/v1",       "MISTRAL_API_KEY",  "mistral_api"),
)


def _sync_openai_compat(
    vendor: str, base_url: str, api_key: str, data_source: str
) -> SyncResult:
    """Sync vendors that follow the OpenAI ``GET /v1/models`` schema."""
    res = SyncResult(source=data_source)
    payload = _http_get_json(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "LLMInspector/17.0 RegistrySync",
        },
    )
    if payload is None:
        res.errors += 1
        return res
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        res.errors += 1
        logger.warning(
            "model_registry_sync: unexpected payload shape",
            source=data_source, top_keys=list(payload.keys()) if isinstance(payload, dict) else None,
        )
        return res

    now = int(time.time())
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        record = {
            "model_id": mid,
            "vendor": vendor,
            "data_source": data_source,
            "last_synced_at": now,
            "raw_metadata": item,
        }
        # OpenAI returns "created" as a unix timestamp; surface it as
        # first_seen if registry has no row yet (registry_repo handles
        # this safely — first_seen_at is ignored on update).
        created = item.get("created")
        if isinstance(created, (int, float)) and created > 0:
            record["first_seen_at"] = int(created)
        try:
            before = registry_repo.get_model_card(mid)
            registry_repo.upsert_model(record)
            res.fetched += 1
            res.seen_model_ids.add(mid)
            if before is None:
                res.inserted += 1
            else:
                res.updated += 1
        except Exception as e:
            res.errors += 1
            logger.warning("registry_repo.upsert_model failed", model_id=mid, error=str(e))
    return res


# Tier 1: Anthropic (id, display_name, created_at) ──────────────────────────


def _sync_anthropic(api_key: str) -> SyncResult:
    res = SyncResult(source="anthropic_api")
    payload = _http_get_json(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Accept": "application/json",
            "User-Agent": "LLMInspector/17.0 RegistrySync",
        },
    )
    if payload is None:
        res.errors += 1
        return res
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        res.errors += 1
        return res

    now = int(time.time())
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        record = {
            "model_id": mid,
            "vendor": "anthropic",
            "data_source": "anthropic_api",
            "last_synced_at": now,
            "raw_metadata": item,
        }
        # Anthropic uses ISO date strings — parse leniently
        created_at = item.get("created_at")
        if isinstance(created_at, str):
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                record["first_seen_at"] = int(dt.timestamp())
            except Exception:
                pass
        try:
            before = registry_repo.get_model_card(mid)
            registry_repo.upsert_model(record)
            res.fetched += 1
            res.seen_model_ids.add(mid)
            if before is None:
                res.inserted += 1
            else:
                res.updated += 1
        except Exception as e:
            res.errors += 1
            logger.warning("anthropic upsert failed", model_id=mid, error=str(e))
    return res


# Tier 1: Google Gemini ─────────────────────────────────────────────────────


def _sync_google(api_key: str) -> SyncResult:
    res = SyncResult(source="google_api")
    payload = _http_get_json(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(api_key)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMInspector/17.0 RegistrySync",
        },
    )
    if payload is None:
        res.errors += 1
        return res
    items = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        res.errors += 1
        return res

    now = int(time.time())
    for item in items:
        if not isinstance(item, dict):
            continue
        # Google returns name="models/gemini-2.0-flash"; strip prefix
        full_name = item.get("name", "")
        mid = full_name.split("/")[-1] if isinstance(full_name, str) else ""
        if not mid:
            continue
        record = {
            "model_id": mid,
            "vendor": "google",
            "data_source": "google_api",
            "last_synced_at": now,
            "context_window": item.get("inputTokenLimit"),
            "max_output_tokens": item.get("outputTokenLimit"),
            "raw_metadata": item,
        }
        try:
            before = registry_repo.get_model_card(mid)
            registry_repo.upsert_model(record)
            res.fetched += 1
            res.seen_model_ids.add(mid)
            if before is None:
                res.inserted += 1
            else:
                res.updated += 1
        except Exception as e:
            res.errors += 1
            logger.warning("google upsert failed", model_id=mid, error=str(e))
    return res


# Tier 2: OpenRouter aggregator ─────────────────────────────────────────────

# Heuristic: pull vendor from OpenRouter's "<vendor>/<model>" id format.
def _vendor_from_openrouter_id(model_id: str) -> str:
    if "/" in model_id:
        prefix = model_id.split("/", 1)[0].lower()
        # OpenRouter prefixes: "openai", "anthropic", "google", "meta-llama",
        # "mistralai", "deepseek", "x-ai", "cohere", "moonshotai", "qwen",
        # "01-ai", "perplexity", etc.  Normalise common ones.
        return {
            "x-ai": "xai",
            "meta-llama": "meta",
            "mistralai": "mistral",
            "moonshotai": "moonshot",
            "01-ai": "yi",
        }.get(prefix, prefix)
    return "unknown"


def _sync_openrouter(api_key: str | None = None) -> SyncResult:
    res = SyncResult(source="openrouter")
    headers = {
        "Accept": "application/json",
        "User-Agent": "LLMInspector/17.0 RegistrySync",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = _http_get_json("https://openrouter.ai/api/v1/models", headers=headers)
    if payload is None:
        res.errors += 1
        return res
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        res.errors += 1
        return res

    now = int(time.time())
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        # OpenRouter pricing is in USD per token (not per Mtok), as strings.
        pricing = item.get("pricing") or {}
        input_per_mtok = _to_per_mtok(pricing.get("prompt"))
        output_per_mtok = _to_per_mtok(pricing.get("completion"))
        cache_read_per_mtok = _to_per_mtok(pricing.get("input_cache_read"))

        arch = item.get("architecture") or {}
        modality = arch.get("modality")
        tokenizer = arch.get("tokenizer")

        record = {
            "model_id": mid,
            "vendor": _vendor_from_openrouter_id(mid),
            "family": arch.get("instruct_type") or None,
            "data_source": "openrouter",
            "last_synced_at": now,
            "context_window": item.get("context_length"),
            "modality": modality,
            "tokenizer_id": tokenizer,
            "input_price_usd": input_per_mtok,
            "output_price_usd": output_per_mtok,
            "cache_read_price_usd": cache_read_per_mtok,
            "raw_metadata": item,
        }
        try:
            before = registry_repo.get_model_card(mid)
            registry_repo.upsert_model(record)
            res.fetched += 1
            res.seen_model_ids.add(mid)
            if before is None:
                res.inserted += 1
            else:
                res.updated += 1
        except Exception as e:
            res.errors += 1
            logger.warning("openrouter upsert failed", model_id=mid, error=str(e))
    return res


def _to_per_mtok(value: Any) -> float | None:
    """OpenRouter prices come as USD-per-token strings (e.g. ``"0.0000025"``).

    Convert to USD per million tokens.  Empty / non-numeric → None.
    """
    if value in (None, "", 0, "0"):
        return None
    try:
        per_token = float(value)
    except (TypeError, ValueError):
        return None
    return round(per_token * 1_000_000, 6)


# ── Orchestration ───────────────────────────────────────────────────────────


@dataclass
class FullSyncReport:
    started_at: int
    finished_at: int
    duration_ms: int
    per_source: list[SyncResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "per_source": [s.to_dict() for s in self.per_source],
            "total_fetched": sum(s.fetched for s in self.per_source),
            "total_inserted": sum(s.inserted for s in self.per_source),
            "total_updated": sum(s.updated for s in self.per_source),
            "total_errors": sum(s.errors for s in self.per_source),
        }


def run_full_sync(env: dict[str, str] | None = None) -> FullSyncReport:
    """Run all configured upstream syncs (best-effort, never raises).

    Sources whose env keys are missing are recorded as ``skipped_reason``
    rather than treated as errors.  OpenRouter is always attempted (no
    key required).
    """
    env = env if env is not None else os.environ
    started = int(time.time())
    t0 = time.monotonic()
    results: list[SyncResult] = []

    for vendor, base_url, env_key, data_source in _OPENAI_COMPATIBLE_VENDORS:
        key = env.get(env_key)
        if not key:
            results.append(SyncResult(source=data_source, skipped_reason=f"{env_key} not set"))
            continue
        results.append(_sync_openai_compat(vendor, base_url, key, data_source))

    if env.get("ANTHROPIC_API_KEY"):
        results.append(_sync_anthropic(env["ANTHROPIC_API_KEY"]))
    else:
        results.append(SyncResult(source="anthropic_api", skipped_reason="ANTHROPIC_API_KEY not set"))

    if env.get("GOOGLE_API_KEY"):
        results.append(_sync_google(env["GOOGLE_API_KEY"]))
    else:
        results.append(SyncResult(source="google_api", skipped_reason="GOOGLE_API_KEY not set"))

    # OpenRouter (Tier 2) — always attempt
    results.append(_sync_openrouter(env.get("OPENROUTER_API_KEY")))

    finished = int(time.time())
    return FullSyncReport(
        started_at=started,
        finished_at=finished,
        duration_ms=int((time.monotonic() - t0) * 1000),
        per_source=results,
    )


# Optional: deprecation sweep (Phase 6 §9.2 §3-cycle rule) ──────────────────


def deprecate_stale_models(
    now: int | None = None,
    miss_window_days: int = 14,
) -> list[str]:
    """Mark models as deprecated when they have not been seen in any
    upstream sync for ``miss_window_days`` days.

    Returns a list of model_ids that were transitioned.
    """
    now = int(now or time.time())
    cutoff = now - miss_window_days * 86400
    from app.core.db import get_conn
    rows = get_conn().execute(
        "SELECT model_id FROM model_registry "
        "WHERE status='active' AND last_seen_at < ? "
        "AND data_source != 'manual'",
        (cutoff,),
    ).fetchall()
    transitioned: list[str] = []
    for row in rows:
        if registry_repo.mark_deprecated(row["model_id"], ts=now):
            transitioned.append(row["model_id"])
    if transitioned:
        logger.info(
            "model_registry_sync: deprecated stale models",
            count=len(transitioned),
            sample=transitioned[:5],
        )
    return transitioned


# ── CLI entrypoint ──────────────────────────────────────────────────────────


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="Run one sync cycle then exit (default).")
    parser.add_argument("--sweep-deprecated", action="store_true",
                        help="After sync, deprecate models not seen in the last 14 days.")
    parser.add_argument("--miss-window-days", type=int, default=14)
    args = parser.parse_args()

    # Initialise schema before touching the table
    from app.core.db import init_db, get_conn
    from app.core.db_migrations import migrate
    init_db()
    migrate(get_conn())

    report = run_full_sync()
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    if args.sweep_deprecated:
        deprecated = deprecate_stale_models(miss_window_days=args.miss_window_days)
        print(f"\n{len(deprecated)} model(s) deprecated by sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "run_full_sync",
    "deprecate_stale_models",
    "FullSyncReport",
    "SyncResult",
    # Adapters (exposed for testing)
    "_sync_openai_compat",
    "_sync_anthropic",
    "_sync_google",
    "_sync_openrouter",
    "_to_per_mtok",
    "_vendor_from_openrouter_id",
]
