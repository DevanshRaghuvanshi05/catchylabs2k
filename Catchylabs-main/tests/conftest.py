"""Test DB isolation and rollback fixtures.

This module resolves TEST_DATABASE_URL and overwrites DATABASE_URL before any
project module (database, config, app) is imported. The live URL is inspected
only to compare host, database name, and username; credentials are never emitted.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _identity(url: str | None) -> tuple[str | None, str | None, str | None]:
    """Return host, database name, and username; never include credentials."""
    if not url:
        return None, None, None
    parsed = make_url(url)
    return parsed.host, parsed.database, parsed.username


def _supabase_project_ref(url: str | None) -> str | None:
    """Extract Supabase project ref from host (db.<ref>.supabase.co) or username (postgres.<ref>)."""
    if not url:
        return None
    parsed = make_url(str(url))
    if parsed.host:
        m = re.search(r"\bdb\.([a-zA-Z0-9_-]+)\.supabase\.co\b", parsed.host, re.IGNORECASE)
        if m:
            return m.group(1)
    if parsed.username:
        m = re.search(r"\bpostgres\.([a-zA-Z0-9_-]+)\b", parsed.username, re.IGNORECASE)
        if m:
            return m.group(1)
        if "." in parsed.username:
            parts = parsed.username.split(".", 1)
            if parts[0] in ("postgres", "default"):
                return parts[1]
    return None


# Read the live endpoint identity without exposing or logging the URL. This is
# used solely as a canary to ensure test host/database cannot target live data.
_live_env = dotenv_values(PROJECT_ROOT / ".env")
_live_db_url = _live_env.get("DATABASE_URL")
LIVE_DATABASE_IDENTITY = _identity(_live_db_url)
LIVE_DATABASE_REF = _supabase_project_ref(_live_db_url)
del _live_env
_pretest_database_identity = _identity(os.environ.get("DATABASE_URL"))


# Prefer the process environment. For local development, allow a separate
# .env.test file to provide only TEST_DATABASE_URL.
_test_database_url = os.environ.get("TEST_DATABASE_URL")
if not _test_database_url:
    _test_env = dotenv_values(PROJECT_ROOT / ".env.test")
    _test_database_url = _test_env.get("TEST_DATABASE_URL")
    del _test_env

if _test_database_url:
    os.environ["TEST_DATABASE_URL"] = _test_database_url
    # config.py calls load_dotenv(), which does not override existing values.
    # Force all project modules imported by tests to use the test endpoint.
    os.environ["DATABASE_URL"] = _test_database_url


def pytest_configure(config):
    """Abort at collection start unless the isolated test endpoint is set."""
    if (not LIVE_DATABASE_IDENTITY or not any(LIVE_DATABASE_IDENTITY)) and os.environ.get("CI") != "1":
        pytest.exit(
            "Live database identity cannot be determined from .env and CI=1 is not set. "
            "Aborting to prevent running tests against unknown database.",
            returncode=2,
        )

    if getattr(config.option, "collectonly", False):
        return

    test_url = os.environ.get("TEST_DATABASE_URL")
    if not test_url:
        pytest.exit(
            "TEST_DATABASE_URL is unset. Set it in the process environment or "
            "in .env.test; tests were not collected.",
            returncode=2,
        )

    if os.environ.get("DATABASE_URL") != test_url:
        pytest.exit(
            "Test isolation setup failed: DATABASE_URL was not redirected to "
            "TEST_DATABASE_URL before project imports.",
            returncode=2,
        )


def pytest_sessionstart(session):
    """Assert the imported application engine resolves to the test DB."""
    if getattr(session.config.option, "collectonly", False):
        return

    import database

    test_url = make_url(os.environ["TEST_DATABASE_URL"])
    engine_url = database.engine.url
    if engine_url != test_url and _identity(str(engine_url)) != _identity(str(test_url)):
        pytest.exit(
            "Test isolation failure: database.engine.url differs from "
            f"TEST_DATABASE_URL (engine identity={_identity(str(engine_url))}, "
            f"test identity={_identity(str(test_url))}).",
            returncode=2,
        )

    engine_identity = _identity(str(engine_url))
    if (
        (LIVE_DATABASE_IDENTITY[0] and engine_identity == LIVE_DATABASE_IDENTITY)
        or (_pretest_database_identity[0] and engine_identity == _pretest_database_identity)
    ):
        pytest.exit(
            "Test isolation failure: test engine host/database/username matches the live "
            f".env endpoint ({engine_identity}).",
            returncode=2,
        )

    engine_ref = _supabase_project_ref(str(engine_url))
    test_ref = _supabase_project_ref(str(test_url))
    if (
        (LIVE_DATABASE_REF and test_ref and test_ref == LIVE_DATABASE_REF)
        or (LIVE_DATABASE_REF and engine_ref and engine_ref == LIVE_DATABASE_REF)
    ):
        pytest.exit(
            "Test isolation failure: Supabase project ref matches the live "
            f".env endpoint ({LIVE_DATABASE_REF}).",
            returncode=2,
        )


@pytest.fixture(scope="session")
def test_engine():
    """Prepare the explicit test DB using init_db(test_engine)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import database

    test_url = os.environ["TEST_DATABASE_URL"]
    engine = create_engine(test_url, pool_pre_ping=True)
    if engine.url != make_url(test_url) and _identity(str(engine.url)) != _identity(test_url):
        pytest.fail("Created test engine does not match TEST_DATABASE_URL.")

    # Bind both application globals to the test engine before any helper is
    # called. init_db accepts the engine explicitly and cannot use a global.
    database.engine = engine
    database.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    database.init_db(engine)

    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def TestSessionLocal(test_engine):
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db_session(test_engine):
    """Roll back all writes, including application-helper commits."""
    connection = test_engine.connect()
    outer_transaction = connection.begin()
    from sqlalchemy.orm import sessionmaker
    test_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection,
        join_transaction_mode="create_savepoint",
    )
    session = test_session_factory()
    try:
        yield session
    finally:
        session.close()
        outer_transaction.rollback()
        connection.close()


@pytest.fixture(autouse=True)
def _patch_session_local(monkeypatch, db_session):
    """Bind helper-created sessions to the fixture's rollback transaction."""
    import database
    from sqlalchemy.orm import sessionmaker

    test_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=db_session.connection(),
        join_transaction_mode="create_savepoint",
    )
    monkeypatch.setattr(database, "SessionLocal", test_session_factory)


@pytest.fixture(scope="session")
def live_database_identity():
    """Sanitized live endpoint identity for the canary test."""
    return LIVE_DATABASE_IDENTITY


@pytest.fixture(scope="session")
def live_database_ref():
    """Sanitized live Supabase project ref for the canary test."""
    return LIVE_DATABASE_REF
