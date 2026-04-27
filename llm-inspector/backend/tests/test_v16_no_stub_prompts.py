"""
v16 acceptance regression: ensure no stub-prompt cases sneak into a real run.

Background
----------
suite_v16_test_comm.json and suite_v16_test_nc.json shipped with placeholder
prompts ("GPQA #1", "LCB #1", "MMLU #1", "TQ #1" — i.e. ID strings, not real
questions). Loading them caused 0% pass on coding/safety/knowledge/reasoning
because the model has nothing meaningful to answer.

These tests guard against three regressions:
1. The repo composite for ``suite_version="v16"`` does not include the stub
   suites (component list is hard-pinned to v10+v13+v15).
2. The seeder force-disables any case from a stub suite at seed time.
3. ``repo.load_cases("v16", ...)`` returns zero cases whose user_prompt looks
   like a stub ID (very short and matches /^[A-Z]+ #\\d+$/).

If real prompts are ever imported into these suites, remove this test file
and re-add the suite versions to ``_V16_COMPONENT_SUITES`` in ``repo.py``.
"""
from __future__ import annotations

import re

# Pattern that matches the v16 stub IDs: e.g. "GPQA #1", "LCB #12", "TQ #5".
# Real questions are paragraphs, never this short.
_STUB_PROMPT_RE = re.compile(r"^[A-Z]{2,5}\s*#\d+$")


def test_v16_composite_excludes_stub_suites():
    """The hard-pinned composite list must not include any v16_test_* suite."""
    from app.repository import repo
    # Use introspection on the function source so we don't have to expose it.
    import inspect
    src = inspect.getsource(repo.load_cases)
    assert "_V16_COMPONENT_SUITES" in src, "load_cases() lost the composite tuple"
    # Stub suite versions must be commented out / removed.
    assert '"v16_test_comm"' not in src or "stub" in src.lower(), \
        "v16_test_comm re-added without stub-disabling note"
    assert '"v16_test_nc"' not in src or "stub" in src.lower(), \
        "v16_test_nc re-added without stub-disabling note"


def test_v16_load_cases_has_no_stub_prompts():
    """A v16 standard run must not surface any stub-prompt case."""
    from app.repository import repo
    cases = repo.load_cases(suite_version="v16", test_mode="standard")
    # We require at least some cases to be loaded — otherwise the test is
    # vacuously passing.
    assert len(cases) > 0, "v16 standard run loaded zero cases"
    offenders = [
        c for c in cases
        if _STUB_PROMPT_RE.match((c.get("user_prompt") or "").strip())
    ]
    assert not offenders, (
        f"Found {len(offenders)} stub-prompt cases that would waste tokens. "
        f"First offenders: "
        + ", ".join(f"{c.get('id')!r}={c.get('user_prompt')!r}" for c in offenders[:5])
    )


def test_seeder_disables_stub_suite_cases():
    """The seeder must mark cases from stub suites as enabled=False."""
    from app.tasks import seeder
    import inspect
    src = inspect.getsource(seeder._seed_test_cases)
    assert "_STUB_SUITE_FILES" in src, \
        "Seeder lost its stub-suite gate; stub cases will be re-enabled at next boot"
    assert 'case["enabled"] = False' in src, \
        "Seeder no longer force-disables stub cases"


def test_no_stub_prompts_in_database_at_rest():
    """
    Direct DB invariant: no enabled case has a stub-style user_prompt.

    This catches the case where someone manually re-enables a stub row in DB
    or imports a fresh stub fixture into a different suite_version.
    """
    from app.core.db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, suite_version, user_prompt FROM test_cases "
        "WHERE enabled=1 AND user_prompt IS NOT NULL"
    ).fetchall()
    offenders = [
        dict(r) for r in rows
        if _STUB_PROMPT_RE.match((r["user_prompt"] or "").strip())
    ]
    assert not offenders, (
        f"Found {len(offenders)} enabled cases with stub-style prompts in DB. "
        f"First: {offenders[:3]}"
    )
