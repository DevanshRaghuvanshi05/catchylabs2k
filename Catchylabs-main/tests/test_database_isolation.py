"""Explicit canary against accidental use of a live database endpoint."""

import os
import re

from sqlalchemy.engine import make_url


def _identity(url):
    parsed = make_url(str(url))
    return parsed.host, parsed.database, parsed.username


def _supabase_project_ref(url):
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


def test_database_engine_and_session_factory_are_test_only(
    test_engine, live_database_identity, live_database_ref
):
    import database

    expected_url = make_url(os.environ["TEST_DATABASE_URL"])
    if database.engine.url != expected_url:
        # A driver prefix (for example postgresql+psycopg2) may differ while
        # still resolving to the same endpoint and account.
        assert _identity(database.engine.url) == _identity(expected_url)

    bound = database.SessionLocal.kw["bind"]
    bound_engine = getattr(bound, "engine", bound)
    if bound_engine.url != expected_url:
        assert _identity(bound_engine.url) == _identity(expected_url)

    if live_database_identity[0] is not None:
        assert _identity(expected_url) != live_database_identity

    expected_ref = _supabase_project_ref(expected_url)
    if expected_ref and live_database_ref:
        assert expected_ref != live_database_ref


def test_all_application_tables_have_row_level_security(test_engine):
    import database
    from sqlalchemy import text

    with test_engine.connect() as connection:
        rls_status = {
            table_name: connection.execute(
                text("SELECT relrowsecurity FROM pg_class WHERE oid = to_regclass(:table_name)"),
                {"table_name": f"public.{table_name}"},
            ).scalar_one_or_none()
            for table_name in database.RLS_TABLES
        }

    assert all(enabled is True for enabled in rls_status.values()), rls_status

