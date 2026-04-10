"""
Shared pytest fixtures.
Populated further by Claude Code at Step 13 (tests/unit/).
"""

import sqlite3
import pytest


@pytest.fixture(scope="session")
def db_path() -> str:
    """Path to the SQLite database."""
    return "data/mlb.db"


@pytest.fixture(scope="session")
def db_conn(db_path: str):
    """
    Read-only session-scoped DB connection.
    Shared across the entire test session — do not write to this connection.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def test_db():
    """
    Isolated in-memory DB for tests that write data.
    Fresh schema per test function — no cross-test state.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
