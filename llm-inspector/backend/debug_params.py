"""Debug script: check params in DB for safety/refusal cases."""
from app.core.db import init_db
from app.repository import repo
from app.tasks.seeder import seed_all

init_db()
seed_all()

# Load cases for quick mode
for mode in ("quick", "standard"):
    cases = repo.load_cases("v10", mode)
    print(f"\n=== Mode: {mode} ({len(cases)} cases) ===")
    for c in cases:
        cat = c.get("category")
        params = c.get("params", {})
        if cat in ("safety", "refusal", "adversarial"):
            print(f"  {c['id']:30s} cat={cat:12s} "
                  f"require_refusal={params.get('require_refusal')} "
                  f"expect_refusal={params.get('expect_refusal')} "
                  f"forbid_compliance={params.get('forbid_compliance')} "
                  f"expect_compliance={params.get('expect_compliance')}")

# Also check what EvalTestCase.params looks like
from app.core.eval_schemas import EvalTestCase
cases = repo.load_cases("v10", "quick")
for c in cases:
    ec = EvalTestCase.from_db_dict(c)
    if ec.category in ("safety", "refusal", "adversarial"):
        print(f"\n  EvalTestCase: {ec.id} params={ec.params}")
        print(f"    require_refusal={ec.params.get('require_refusal')} "
              f"expect_refusal={ec.params.get('expect_refusal')} "
              f"forbid_compliance={ec.params.get('forbid_compliance')}")
