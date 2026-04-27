"""
runner/dataset_sync.py — v17 Phase 9: live benchmark suite ingestion.

Pulls a fresh slice of public benchmark datasets (LiveBench, SWE-bench
Verified, HLE, GPQA Diamond Live) and merges *new* entries into the
``test_cases`` table.  Designed to run weekly (UPGRADE_PLAN_V17.md §12)
or ad-hoc via ``python -m app.runner.dataset_sync --pull``.

Architecture
------------

A ``DatasetSource`` is a pure data definition: a name, an HF dataset
identifier, and a *transformer* callable that turns one upstream record
into the inspector's local ``test_case`` schema (the same shape used by
``app.repository.repo.upsert_test_case``).

Each sync cycle:

  1. ``fetcher(source) -> list[dict]`` returns the upstream records.
  2. ``transformer(record, source) -> dict`` converts to local schema.
  3. The resulting case_id is checked against ``test_cases``.  Only
     genuinely new ids are inserted; existing ids are *skipped* (versioned
     replacement is left to a follow-up upgrade — see Phase 10).
  4. Per-source counts are aggregated into a ``DatasetSyncReport``.

The fetcher is fully injectable so unit tests stay offline; the default
implementation hits HF's ``datasets-server`` JSON API which works
without authentication for public datasets and avoids importing the
full ``datasets`` Python library.

Schema additions
----------------
Every imported case is tagged in ``params._meta``::

  {
    "source_dataset": "LiveBench" | "SWE-bench-Verified" | "HLE" | ...,
    "source_id":      <upstream id, ideally stable across versions>,
    "ingested_at":    <unix epoch seconds>,
  }

This lets Phase 10's suite_pruner attribute pass-rates back to a
specific dataset and surface "suite exhaustion" warnings per family.
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from app.core.logging import get_logger
from app.core.db import get_conn
from app.repository import repo

logger = get_logger(__name__)


# ── HTTP helper ─────────────────────────────────────────────────────────────


def _build_ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import]
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _build_ssl_ctx()


def _http_get_json(url: str, timeout: int = 30) -> Any | None:
    """Best-effort HTTP GET → parsed JSON.  Returns None on any failure."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMInspector/17.0 DatasetSync",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        logger.warning("dataset_sync: fetch failed", url=url, error=str(getattr(e, "reason", e)))
        return None
    except Exception as e:
        logger.warning("dataset_sync: parse error", url=url, error=str(e)[:200])
        return None


# ── Source registry + transformers ──────────────────────────────────────────


@dataclass(frozen=True)
class DatasetSource:
    name: str
    hf_repo: str            # e.g. "LiveBench/LiveBench"
    config: str             # subset / configuration name (or empty string)
    split: str              # train / test / live etc.
    category: str           # local category mapping (reasoning / coding / safety / ...)
    transformer: Callable[[dict, "DatasetSource"], dict | None]
    license_url: str = ""


# ---------------------------------------------------------------------------
# Transformers below are intentionally lenient: they pull common fields and
# fall back to the upstream id.  Real datasets shapes differ; transformers
# are the single source of "shape knowledge" and should be expanded as the
# inspector starts ingesting more datasets.
# ---------------------------------------------------------------------------


def _slugify(s: str, max_len: int = 80) -> str:
    """Lowercase + safe-char filter for use inside a case id.

    Underscores and hyphens are preserved verbatim because upstream
    benchmark ids commonly contain them (e.g. ``django__django-12345``);
    other separators collapse to ``-``.
    """
    out = []
    for ch in (s or "").strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch in (" ", "/"):
            out.append("-")
    return "".join(out)[:max_len].strip("-") or "x"


def _make_case_id(prefix: str, upstream_id: str | int) -> str:
    return f"{prefix}_{_slugify(str(upstream_id), 60)}"


def _livebench_transformer(record: dict, src: DatasetSource) -> dict | None:
    """LiveBench → test_case.  Pulls question/answer/category."""
    question = record.get("question") or record.get("prompt") or record.get("turns")
    if isinstance(question, list) and question:
        question = str(question[0])
    if not question or not isinstance(question, str):
        return None
    upstream_id = record.get("question_id") or record.get("id") or _slugify(question[:24])
    answer = (
        record.get("ground_truth")
        or record.get("answer")
        or record.get("solution")
        or ""
    )
    return {
        "id": _make_case_id("livebench", upstream_id),
        "category": src.category,
        "name": f"LiveBench {upstream_id}",
        "user_prompt": question,
        "expected_type": "text",
        "judge_method": "semantic_judge_v2" if not answer else "exact_match",
        "params": {
            "expected_output": answer,
            "_meta": {
                "source_dataset": src.name,
                "source_id": str(upstream_id),
                "ingested_at": int(time.time()),
            },
        },
        "weight": 1.0,
        "enabled": True,
        "suite_version": "v17_livebench",
        "max_tokens": 1024,
    }


def _swebench_transformer(record: dict, src: DatasetSource) -> dict | None:
    """SWE-bench Verified → coding case."""
    instance_id = record.get("instance_id") or record.get("id")
    problem = record.get("problem_statement")
    if not instance_id or not problem:
        return None
    return {
        "id": _make_case_id("swebench", instance_id),
        "category": "coding",
        "name": f"SWE-bench {instance_id}",
        "user_prompt": problem,
        "expected_type": "code_diff",
        "judge_method": "code_execution",
        "params": {
            "repo": record.get("repo"),
            "base_commit": record.get("base_commit"),
            "test_patch": record.get("test_patch"),
            "_meta": {
                "source_dataset": src.name,
                "source_id": str(instance_id),
                "ingested_at": int(time.time()),
            },
        },
        "weight": 1.5,
        "enabled": True,
        "suite_version": "v17_swebench",
        "max_tokens": 4096,
    }


def _hle_transformer(record: dict, src: DatasetSource) -> dict | None:
    """HLE (Humanity's Last Exam) → reasoning case."""
    qid = record.get("id") or record.get("question_id")
    question = record.get("question") or record.get("prompt")
    answer = record.get("answer") or ""
    if not qid or not question:
        return None
    return {
        "id": _make_case_id("hle", qid),
        "category": "reasoning",
        "name": f"HLE {qid}",
        "user_prompt": question,
        "expected_type": "text",
        "judge_method": "exact_match" if answer else "semantic_judge_v2",
        "params": {
            "expected_output": answer,
            "_meta": {
                "source_dataset": src.name,
                "source_id": str(qid),
                "ingested_at": int(time.time()),
            },
        },
        "weight": 2.0,
        "enabled": True,
        "suite_version": "v17_hle",
        "max_tokens": 2048,
    }


DEFAULT_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="LiveBench",
        hf_repo="LiveBench/LiveBench",
        config="default",
        split="test",
        category="reasoning",
        transformer=_livebench_transformer,
        license_url="https://huggingface.co/datasets/LiveBench/LiveBench",
    ),
    DatasetSource(
        name="SWE-bench-Verified",
        hf_repo="princeton-nlp/SWE-bench_Verified",
        config="default",
        split="test",
        category="coding",
        transformer=_swebench_transformer,
        license_url="https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified",
    ),
    DatasetSource(
        name="HLE",
        hf_repo="cais/hle",
        config="default",
        split="test",
        category="reasoning",
        transformer=_hle_transformer,
        license_url="https://huggingface.co/datasets/cais/hle",
    ),
)


# ── HF datasets-server fetcher (default) ────────────────────────────────────


def _hf_fetcher(source: DatasetSource, max_rows: int = 1000) -> list[dict] | None:
    """Default fetcher hitting Hugging Face's ``datasets-server`` API.

    Returns None on transport failure (caller treats as "fetch failed").
    Public datasets do not require auth.
    """
    base = "https://datasets-server.huggingface.co/rows"
    params = urllib.parse.urlencode({
        "dataset": source.hf_repo,
        "config": source.config,
        "split": source.split,
        "offset": 0,
        "length": min(max_rows, 100),       # API caps at 100/req
    })
    payload = _http_get_json(f"{base}?{params}")
    if payload is None:
        return None
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    out: list[dict] = []
    for r in rows:
        # HF wraps each row as {"row": {...}, "row_idx": N}
        if isinstance(r, dict) and isinstance(r.get("row"), dict):
            out.append(r["row"])
        elif isinstance(r, dict):
            out.append(r)
    return out


# ── Sync core ───────────────────────────────────────────────────────────────


@dataclass
class DatasetSyncResult:
    source_name: str
    fetched_rows: int = 0
    transformed: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    skipped_invalid: int = 0
    error: str | None = None
    sample_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "fetched_rows": self.fetched_rows,
            "transformed": self.transformed,
            "inserted": self.inserted,
            "skipped_existing": self.skipped_existing,
            "skipped_invalid": self.skipped_invalid,
            "error": self.error,
            "sample_ids": self.sample_ids[:10],
        }


def _case_id_exists(case_id: str) -> bool:
    row = get_conn().execute(
        "SELECT 1 FROM test_cases WHERE id = ?", (case_id,)
    ).fetchone()
    return row is not None


def sync_one_source(
    source: DatasetSource,
    fetcher: Callable[[DatasetSource], list[dict] | None] = _hf_fetcher,
    max_rows: int = 1000,
) -> DatasetSyncResult:
    """Fetch + transform + upsert one ``DatasetSource``.  Never raises."""
    res = DatasetSyncResult(source_name=source.name)
    try:
        rows = fetcher(source) if fetcher is _hf_fetcher else fetcher(source)
    except Exception as e:
        res.error = f"fetcher_raised: {str(e)[:200]}"
        return res
    if rows is None:
        res.error = "fetch_failed"
        return res
    res.fetched_rows = len(rows)

    for row in rows[:max_rows]:
        try:
            case = source.transformer(row, source)
        except Exception as e:
            res.skipped_invalid += 1
            logger.warning("transformer raised", source=source.name, error=str(e)[:200])
            continue
        if case is None or not case.get("id"):
            res.skipped_invalid += 1
            continue
        res.transformed += 1
        if _case_id_exists(case["id"]):
            res.skipped_existing += 1
            continue
        try:
            repo.upsert_test_case(case)
            res.inserted += 1
            if len(res.sample_ids) < 10:
                res.sample_ids.append(case["id"])
        except Exception as e:
            res.skipped_invalid += 1
            logger.warning("upsert_test_case failed", case_id=case.get("id"), error=str(e))
    return res


@dataclass
class DatasetSyncReport:
    started_at: int
    finished_at: int
    duration_sec: int
    per_source: list[DatasetSyncResult]

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "per_source": [r.to_dict() for r in self.per_source],
            "total_inserted": sum(r.inserted for r in self.per_source),
            "total_skipped_existing": sum(r.skipped_existing for r in self.per_source),
            "total_skipped_invalid": sum(r.skipped_invalid for r in self.per_source),
            "errors": [r.source_name for r in self.per_source if r.error],
        }


def run_dataset_sync(
    sources: tuple[DatasetSource, ...] = DEFAULT_SOURCES,
    fetcher: Callable[[DatasetSource], list[dict] | None] = _hf_fetcher,
    max_rows_per_source: int = 1000,
) -> DatasetSyncReport:
    """Run dataset sync across ``sources``.  Best-effort; never raises."""
    started = int(time.time())
    t0 = time.monotonic()
    results = [
        sync_one_source(src, fetcher=fetcher, max_rows=max_rows_per_source)
        for src in sources
    ]
    finished = int(time.time())
    return DatasetSyncReport(
        started_at=started,
        finished_at=finished,
        duration_sec=int(time.monotonic() - t0),
        per_source=results,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true",
                        help="Pull fresh datasets and ingest new rows.")
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args()
    _ = args
    from app.core.db import init_db, get_conn
    from app.core.db_migrations import migrate
    init_db()
    migrate(get_conn())
    report = run_dataset_sync(max_rows_per_source=args.max_rows)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "DatasetSource",
    "DEFAULT_SOURCES",
    "DatasetSyncResult",
    "DatasetSyncReport",
    "sync_one_source",
    "run_dataset_sync",
    "_livebench_transformer",
    "_swebench_transformer",
    "_hle_transformer",
    "_make_case_id",
]
