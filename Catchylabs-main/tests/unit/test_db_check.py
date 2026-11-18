"""
tests/unit/test_db_check.py -- Unit tests for db_check.py.

All tests are fully mocked: no database connection, no DNS query,
no Gemini or SerpAPI calls are made.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test without touching any real resources.
# ---------------------------------------------------------------------------

DB_CHECK_PATH = str(Path(__file__).resolve().parents[2] / "db_check.py")


def _load_db_check():
    spec = importlib.util.spec_from_file_location("db_check", DB_CHECK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def db_check():
    return _load_db_check()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeURL:
    """Minimal stand-in for sqlalchemy.engine.URL."""

    def __init__(self, host, port, database, username, password):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password


_VALID_URL = (
    "postgresql://postgres.yfxttjycopwmvwyidvbi:testpassword"
    "@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
)

_IPV4_RESULTS = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", 0))]


def _make_success_engine():
    result_mock = MagicMock()
    result_mock.fetchone.return_value = (1,)
    execute_mock = MagicMock(return_value=result_mock)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(execute=execute_mock))
    cm.__exit__ = MagicMock(return_value=False)
    engine_mock = MagicMock()
    engine_mock.connect.return_value = cm
    return engine_mock


def _make_failing_engine(exc_msg: str):
    cm = MagicMock()
    cm.__enter__ = MagicMock(side_effect=Exception(exc_msg))
    cm.__exit__ = MagicMock(return_value=False)
    engine_mock = MagicMock()
    engine_mock.connect.return_value = cm
    return engine_mock


def _run_main(db_check, url: str, extra_argv=None, dns_results=None, engine=None):
    """Run db_check.main() with fully mocked environment."""
    argv = extra_argv or []
    if engine is None:
        engine = _make_success_engine()

    with (
        patch.dict(os.environ, {"DATABASE_URL": url}, clear=False),
        patch.object(db_check, "_load_dotenv"),
        patch("socket.getaddrinfo", return_value=dns_results or _IPV4_RESULTS),
        patch("sqlalchemy.create_engine", return_value=engine),
        patch("sqlalchemy.text", return_value=MagicMock()),
    ):
        return db_check.main(argv)


# ---------------------------------------------------------------------------
# 1. URL validation checks (no network, no DB)
# ---------------------------------------------------------------------------

class TestCheckUrlProblems:
    def test_clean_url_no_problems(self, db_check):
        parsed = _FakeURL(
            host="aws-0-eu-central-1.pooler.supabase.com",
            port=6543,
            database="postgres",
            username="postgres.yfxttjycopwmvwyidvbi",
            password="safepassword",
        )
        problems = db_check.check_url_problems(
            "postgresql://postgres.yfxttjycopwmvwyidvbi:safepassword"
            "@aws-0-eu-central-1.pooler.supabase.com:6543/postgres",
            parsed,
        )
        assert problems == []

    def test_password_with_at_sign(self, db_check):
        parsed = _FakeURL(
            host="db.ref.supabase.co",
            port=5432,
            database="postgres",
            username="postgres",
            password="bad@pass",
        )
        problems = db_check.check_url_problems("", parsed)
        assert any("percent-encoding" in p for p in problems)
        assert any("@" in p for p in problems)

    def test_password_with_colon(self, db_check):
        parsed = _FakeURL(
            host="db.ref.supabase.co",
            port=5432,
            database="postgres",
            username="postgres",
            password="bad:pass",
        )
        problems = db_check.check_url_problems("", parsed)
        assert any("percent-encoding" in p for p in problems)

    def test_password_with_hash(self, db_check):
        parsed = _FakeURL(
            host="db.ref.supabase.co",
            port=5432,
            database="postgres",
            username="postgres",
            password="pass#word",
        )
        problems = db_check.check_url_problems("", parsed)
        assert any("percent-encoding" in p for p in problems)
        assert any("#" in p for p in problems)

    def test_postgres_scheme(self, db_check):
        parsed = _FakeURL(
            host="db.ref.supabase.co",
            port=5432,
            database="postgres",
            username="postgres",
            password="ok",
        )
        problems = db_check.check_url_problems(
            "postgres://postgres:ok@db.ref.supabase.co:5432/postgres",
            parsed,
        )
        assert any("postgres://" in p for p in problems)

    def test_direct_host_flagged(self, db_check):
        parsed = _FakeURL(
            host="db.yfxttjycopwmvwyidvbi.supabase.co",
            port=5432,
            database="postgres",
            username="postgres",
            password="ok",
        )
        problems = db_check.check_url_problems("", parsed)
        assert any("direct" in p.lower() for p in problems)

    def test_non_standard_port(self, db_check):
        parsed = _FakeURL(
            host="aws-0-eu-central-1.pooler.supabase.com",
            port=9999,
            database="postgres",
            username="postgres.ref",
            password="ok",
        )
        problems = db_check.check_url_problems("", parsed)
        assert any("9999" in p for p in problems)

    def test_direct_host_wrong_user(self, db_check):
        parsed = _FakeURL(
            host="db.ref.supabase.co",
            port=5432,
            database="postgres",
            username="someotheruser",
            password="ok",
        )
        problems = db_check.check_url_problems("", parsed)
        assert any("someotheruser" in p for p in problems)

    def test_pooler_host_wrong_user_format(self, db_check):
        parsed = _FakeURL(
            host="aws-0-eu-central-1.pooler.supabase.com",
            port=6543,
            database="postgres",
            username="justpostgres",
            password="ok",
        )
        problems = db_check.check_url_problems("", parsed)
        # The warning mentions 'postgres.<project_ref>' and 'justpostgres'
        assert any("justpostgres" in p for p in problems)


# ---------------------------------------------------------------------------
# 2. Error categorisation
# ---------------------------------------------------------------------------

class TestCategoriseError:

    def _exc(self, msg: str) -> Exception:
        return Exception(msg)

    def test_password_wrong(self, db_check):
        cat, _ = db_check.categorise_error(
            self._exc("password authentication failed for user postgres")
        )
        assert cat == "PASSWORD_WRONG"

    def test_tenant_or_user_not_found(self, db_check):
        cat, _ = db_check.categorise_error(self._exc("tenant or user not found"))
        assert cat == "TENANT_OR_USER_NOT_FOUND"

    def test_role_does_not_exist(self, db_check):
        cat, _ = db_check.categorise_error(self._exc('role "wrong_user" does not exist'))
        assert cat == "TENANT_OR_USER_NOT_FOUND"

    def test_host_not_found(self, db_check):
        cat, _ = db_check.categorise_error(
            self._exc("could not translate host name to address: nodename nor servname provided")
        )
        assert cat == "HOST_NOT_FOUND"

    def test_network_unreachable(self, db_check):
        cat, _ = db_check.categorise_error(self._exc("connection refused (errno 111)"))
        assert cat == "NETWORK_UNREACHABLE"

    def test_timeout(self, db_check):
        cat, _ = db_check.categorise_error(self._exc("connection timed out after 10 seconds"))
        assert cat == "TIMEOUT"

    def test_ssl_error(self, db_check):
        cat, _ = db_check.categorise_error(self._exc("SSL handshake failed: certificate verify error"))
        assert cat == "SSL"

    def test_other(self, db_check):
        cat, _ = db_check.categorise_error(self._exc("some completely unknown error"))
        assert cat == "OTHER"


# ---------------------------------------------------------------------------
# 3. DNS resolution helper
# ---------------------------------------------------------------------------

class TestResolveHost:
    def test_ipv4_only(self, db_check):
        fake_results = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.2.3.4", 0))]
        with patch("socket.getaddrinfo", return_value=fake_results):
            has_ipv4, has_ipv6 = db_check.resolve_host("any.host")
        assert has_ipv4 is True
        assert has_ipv6 is False

    def test_ipv6_only(self, db_check):
        fake_results = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))]
        with patch("socket.getaddrinfo", return_value=fake_results):
            has_ipv4, has_ipv6 = db_check.resolve_host("any.host")
        assert has_ipv4 is False
        assert has_ipv6 is True

    def test_dns_failure(self, db_check):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")):
            has_ipv4, has_ipv6 = db_check.resolve_host("bad.host")
        assert has_ipv4 is False
        assert has_ipv6 is False


# ---------------------------------------------------------------------------
# 4. main() end-to-end (fully mocked)
# ---------------------------------------------------------------------------

class TestMainSuccess:
    def test_select_1_passes(self, db_check, capsys):
        rc = _run_main(db_check, _VALID_URL)
        out = capsys.readouterr().out
        assert rc == 0
        assert "OK -- database check passed." in out
        assert "testpassword" not in out
        assert _VALID_URL not in out

    def test_expect_project_match(self, db_check, capsys):
        rc = _run_main(db_check, _VALID_URL, extra_argv=["--expect-project", "yfxttjycopwmvwyidvbi"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "yfxttjycopwmvwyidvbi" in out

    def test_expect_project_mismatch(self, db_check, capsys):
        rc = _run_main(db_check, _VALID_URL, extra_argv=["--expect-project", "differentproject"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "differentproject" in out


class TestMainConnectionErrors:
    def _check(self, db_check, capsys, exc_msg, expected_category):
        rc = _run_main(db_check, _VALID_URL, engine=_make_failing_engine(exc_msg))
        out = capsys.readouterr().out
        assert rc == 1
        assert expected_category in out
        assert "testpassword" not in out
        return out

    def test_password_wrong(self, db_check, capsys):
        self._check(db_check, capsys, "password authentication failed for user postgres", "PASSWORD_WRONG")

    def test_tenant_or_user_not_found(self, db_check, capsys):
        self._check(db_check, capsys, "tenant or user not found", "TENANT_OR_USER_NOT_FOUND")

    def test_host_not_found(self, db_check, capsys):
        self._check(db_check, capsys, "could not translate host name to address: nodename nor servname provided", "HOST_NOT_FOUND")

    def test_network_unreachable(self, db_check, capsys):
        self._check(db_check, capsys, "connection refused (errno 111)", "NETWORK_UNREACHABLE")

    def test_timeout(self, db_check, capsys):
        self._check(db_check, capsys, "connection timed out after 10 seconds", "TIMEOUT")

    def test_ssl_error(self, db_check, capsys):
        self._check(db_check, capsys, "SSL handshake failed", "SSL")

    def test_other_error(self, db_check, capsys):
        self._check(db_check, capsys, "some completely unknown database error", "OTHER")


class TestMainIpv6Warning:
    def test_ipv6_only_prints_warning(self, db_check, capsys):
        ipv6_results = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))]
        _run_main(db_check, _VALID_URL, dns_results=ipv6_results)
        out = capsys.readouterr().out
        assert "IPv6" in out
        assert "Session pooler" in out


class TestPasswordNeverPrinted:
    def test_password_not_in_output_on_success(self, db_check, capsys):
        _run_main(db_check, _VALID_URL)
        out = capsys.readouterr().out
        assert "testpassword" not in out

    def test_password_not_in_output_on_failure(self, db_check, capsys):
        _run_main(db_check, _VALID_URL, engine=_make_failing_engine("password authentication failed"))
        out = capsys.readouterr().out
        assert "testpassword" not in out

    def test_full_url_not_in_output(self, db_check, capsys):
        _run_main(db_check, _VALID_URL)
        out = capsys.readouterr().out
        assert _VALID_URL not in out
