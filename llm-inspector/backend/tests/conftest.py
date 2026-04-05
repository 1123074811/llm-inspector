"""
LLM Inspector — pytest configuration.
Handles test database isolation and shared fixtures.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

# Ensure backend is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Use isolated test database
os.environ["DATABASE_URL"] = "sqlite:///./test_inspector.db"


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialize and seed the test database once per session."""
    from app.core.db import init_db
    from app.tasks.seeder import seed_all
    init_db()
    seed_all()
    yield
    # Cleanup
    try:
        os.remove("test_inspector.db")
    except FileNotFoundError:
        pass


@pytest.fixture()
def key_manager():
    """Provide a KeyManager instance."""
    from app.core.security import get_key_manager
    return get_key_manager()


@pytest.fixture()
def create_run(key_manager):
    """Factory fixture to create a test run and return its ID."""
    from app.repository import repo

    def _create(base_url="https://ex.com/v1", api_key="sk-test",
                model="test-model", mode="standard"):
        enc, h = key_manager.encrypt(api_key)
        return repo.create_run(base_url, enc, h, model, mode)

    return _create
