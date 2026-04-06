"""Model trend and leaderboard handlers."""
from __future__ import annotations

import urllib.parse
from app.handlers.helpers import _json, _error, _extract_id
from app.repository import repo

__all__ = [
    "handle_model_trend",
    "handle_leaderboard",
    "handle_model_theta_trend",
    "handle_theta_leaderboard",
]


def handle_model_trend(path, qs, _body) -> tuple:
    model_name = _extract_id(path, r"/api/v1/models/([^/]+)/trend$")
    if not model_name:
        return _error("Invalid model name", 400)
    model_name = urllib.parse.unquote(model_name)
    limit = int(qs.get("limit", ["20"])[0])
    return _json(repo.get_model_trend(model_name, min(limit, 200)))


def handle_leaderboard(_path, qs, _body) -> tuple:
    sort_by = qs.get("sort_by", ["total_score"])[0]
    limit = int(qs.get("limit", ["50"])[0])
    return _json(repo.get_leaderboard(sort_by=sort_by, limit=min(limit, 200)))


def handle_model_theta_trend(path, qs, _body) -> tuple:
    model_name = _extract_id(path, r"/api/v1/models/([^/]+)/theta-trend$")
    if not model_name:
        return _error("Invalid model name", 400)
    model_name = urllib.parse.unquote(model_name)
    limit = int(qs.get("limit", ["50"])[0])
    return _json(repo.get_model_theta_trend(model_name, min(limit, 200)))


def handle_theta_leaderboard(_path, qs, _body) -> tuple:
    dimension = qs.get("dimension", ["global"])[0]
    limit = int(qs.get("limit", ["50"])[0])
    return _json(repo.get_theta_leaderboard(dimension=dimension, limit=min(limit, 200)))
