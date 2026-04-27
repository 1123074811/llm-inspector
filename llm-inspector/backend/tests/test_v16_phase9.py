"""
test_v16_phase9.py — v16 Phase 9 regression tests.

Validates:
  - index.html contains report-real-model-card template
  - styles.css contains RMC styles and mobile breakpoint
  - app.js contains renderRealModelCard and upgraded showToast
  - RealModelCard to_dict compatibility with frontend
"""
import pytest
import pathlib as _pl

_FRONTEND = _pl.Path(__file__).resolve().parent.parent.parent / "frontend"


class TestIndexHTML:
    def test_real_model_card_template_exists(self):
        content = (_FRONTEND / "index.html").read_text(encoding="utf-8")
        assert 'id="report-real-model-card"' in content or 'report-real-model-card' in content

    def test_rmc_fields_present(self):
        content = (_FRONTEND / "index.html").read_text(encoding="utf-8")
        assert "rmc-claimed" in content
        assert "rmc-suspected" in content
        assert "rmc-posterior" in content
        assert "rmc-evidence" in content
        assert "rmc-leaked-prompt" in content
        assert "rmc-model-list" in content

    def test_old_identity_template_removed(self):
        content = (_FRONTEND / "index.html").read_text(encoding="utf-8")
        # The old template tag itself should be gone (comment references are OK)
        import re
        old_template = re.search(r'<template\s+id=["\']identity-exposure-tpl["\']', content)
        assert old_template is None, "Old identity-exposure-tpl template tag still exists"

    def test_rmc_official_field(self):
        content = (_FRONTEND / "index.html").read_text(encoding="utf-8")
        assert "rmc-official" in content


class TestStylesCSS:
    def test_rmc_row_style(self):
        content = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
        assert ".rmc-row" in content

    def test_risk_badge_styles(self):
        content = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
        assert ".risk-badge.trusted" in content
        assert ".risk-badge.high_risk" in content

    def test_toast_fatal_style(self):
        content = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
        assert ".toast-fatal" in content

    def test_mobile_breakpoint(self):
        content = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
        assert "@media (max-width: 768px)" in content

    def test_toast_actions_style(self):
        content = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
        assert ".toast-actions" in content


class TestAppJS:
    def test_render_real_model_card_function(self):
        content = (_FRONTEND / "app.js").read_text(encoding="utf-8")
        assert "function renderRealModelCard" in content

    def test_showtoast_supports_hint(self):
        content = (_FRONTEND / "app.js").read_text(encoding="utf-8")
        # showToast should accept hint parameter
        assert "hint" in content.split("function showToast")[1].split("}")[0]

    def test_showtoast_supports_actions(self):
        content = (_FRONTEND / "app.js").read_text(encoding="utf-8")
        assert "toast-actions" in content

    def test_rmc_tier_logic(self):
        content = (_FRONTEND / "app.js").read_text(encoding="utf-8")
        # Should have tier classification logic
        assert "trusted" in content
        assert "high_risk" in content
        assert "fake" in content


class TestRealModelCardFrontendCompat:
    def test_to_dict_matches_frontend_fields(self):
        from app.core.schemas import RealModelCard, Evidence
        rmc = RealModelCard(
            claimed_model="gpt-4o",
            suspected_family="qwen2",
            posterior=0.85,
            evidence=[
                Evidence(source_layer="model_discovery", snippet="qwen2 found", confidence=0.9),
            ],
            is_official=False,
            leaked_system_prompt="You are a helpful assistant",
            model_list_report={"available": ["qwen2-72b"]},
        )
        d = rmc.to_dict()
        # Frontend expects these keys
        assert "claimed_model" in d
        assert "suspected_family" in d
        assert "posterior" in d
        assert "evidence" in d
        assert "is_official" in d
        assert "leaked_system_prompt" in d
        assert "model_list_report" in d
