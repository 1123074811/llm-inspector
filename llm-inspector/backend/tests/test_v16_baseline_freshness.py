"""
v16 acceptance regression: similarity ranking must respect

  (a) baselines created mid-session — invalidate_benchmark_cache must be
      called by every baseline mutation handler so a run started right after
      the user clicks "Set as baseline" actually compares against the new
      baseline.

  (b) coverage of the comparison — a baseline whose feature_vector overlaps
      FEATURE_ORDER on only ~10 dims must not out-rank one with rich overlap
      (~23 dims) just because cosine over a tiny subspace is inflated.

These two bugs together produced the v16 acceptance symptom:
"set deepseek-v4-flash run #1 as baseline → run #2 ranks qianfan-code-latest
 #1 above its own baseline".
"""
from __future__ import annotations


def test_invalidate_benchmark_cache_clears_all():
    """All cache entries must be removed when called without a suite_version."""
    from app.runner import case_prep
    case_prep._benchmark_cache["v16"] = (1.0, [{"x": 1}])
    case_prep._benchmark_cache["v15"] = (1.0, [{"y": 2}])
    case_prep.invalidate_benchmark_cache()
    assert case_prep._benchmark_cache == {}


def test_invalidate_benchmark_cache_targeted():
    """Targeted invalidation must only drop the requested suite_version."""
    from app.runner import case_prep
    case_prep._benchmark_cache.clear()
    case_prep._benchmark_cache["v16"] = (1.0, [{"x": 1}])
    case_prep._benchmark_cache["v15"] = (1.0, [{"y": 2}])
    case_prep.invalidate_benchmark_cache("v16")
    assert "v16" not in case_prep._benchmark_cache
    assert "v15" in case_prep._benchmark_cache
    case_prep._benchmark_cache.clear()


def test_baseline_mutation_handlers_invalidate_cache():
    """create/delete baseline handlers must call invalidate_benchmark_cache."""
    import inspect
    from app.handlers import baselines as bl_mod
    create_src = inspect.getsource(bl_mod.handle_create_baseline)
    delete_src = inspect.getsource(bl_mod.handle_delete_baseline)
    assert "invalidate_benchmark_cache" in create_src, (
        "handle_create_baseline must invalidate the in-process benchmark cache "
        "or new baselines will not appear in the next run within TTL"
    )
    assert "invalidate_benchmark_cache" in delete_src, (
        "handle_delete_baseline must invalidate the in-process benchmark cache"
    )


def test_similarity_ranking_prefers_high_coverage():
    """
    Reproduces the v16 acceptance symptom and asserts the fix.

    Setup:
      - target run features: 23 dims of FEATURE_ORDER populated (modern run)
      - "rich" baseline:    same 23 dims (high confidence overlap)
      - "sparse" baseline:  only 10 dims overlap (low confidence) but those
                            10 dims are crafted to give a slightly higher
                            cosine in the sparse subspace
    Expected: rich baseline ranks #1.
    """
    from app.analysis.similarity import SimilarityEngine, FEATURE_ORDER

    rng = list(FEATURE_ORDER)
    target_features = {k: 0.7 for k in rng[:23]}

    rich_features = {k: 0.71 for k in rng[:23]}      # 23-dim overlap
    sparse_features = {k: 0.999 for k in rng[:10]}   # 10-dim overlap; cosine inflated

    benchmarks = [
        {"benchmark_name": "rich-baseline", "feature_vector": rich_features,
         "data_source": "measured"},
        {"benchmark_name": "sparse-baseline", "feature_vector": sparse_features,
         "data_source": "measured"},
    ]

    engine = SimilarityEngine()
    results = engine.compare(target_features, benchmarks)
    by_name = {r.benchmark_name: r for r in results}
    assert by_name["rich-baseline"].rank == 1, (
        f"rich baseline must rank #1 but got rank={by_name['rich-baseline'].rank}; "
        f"sparse rank={by_name['sparse-baseline'].rank}; "
        f"this means a 10-dim baseline can still out-rank a 23-dim one — "
        f"the v16 qianfan-vs-deepseek bug is back."
    )
    # Coverage tiers must be set
    assert by_name["rich-baseline"].confidence_level in ("medium", "high"), \
        f"rich (23 dims) should be medium/high confidence, got {by_name['rich-baseline'].confidence_level}"
    assert by_name["sparse-baseline"].confidence_level in ("insufficient", "low"), \
        f"sparse (10 dims) should be low/insufficient, got {by_name['sparse-baseline'].confidence_level}"
