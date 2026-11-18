"""
db_check.py -- Validate DATABASE_URL before starting the demo server.

Usage:
    python db_check.py [--expect-project <id>]

Prints only the host name, port, database name, user name, and password length.
Never prints the password or the full URL.

Exit codes:
    0 -- connection succeeded (SELECT 1 passed)
    1 -- connection failed or configuration error

Rules:
    - Loads .env from the script directory (existing env vars win).
    - Plain ASCII output only.
    - No live Gemini or SerpAPI calls.
    - No mutating SQL -- SELECT 1 only.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# .env loader (same semantics as run_demo.ps1)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path) -> None:
    """Load KEY=VALUE pairs from env_path into os.environ (existing values win)."""
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            eq = line.find("=")
            if eq < 1:
                continue
            key = line[:eq].strip()
            val = line[eq + 1:].strip()
            # Strip surrounding single or double quotes
            if len(val) >= 2 and (
                (val[0] == '"' and val[-1] == '"') or
                (val[0] == "'" and val[-1] == "'")
            ):
                val = val[1:-1]
            if key not in os.environ:
                os.environ[key] = val


# ---------------------------------------------------------------------------
# Sensitive text scrubber (removes URL-like and password-like tokens)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"[a-zA-Z][a-zA-Z0-9+\-.]*://[^\s\"']{4,}",
    re.IGNORECASE,
)
_PASS_RE = re.compile(
    r"(password|passwd|pwd)\s*[=:]\s*\S+",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    text = _URL_RE.sub("[URL]", text)
    text = _PASS_RE.sub(r"\1=[REDACTED]", text)
    return text[:120]


# ---------------------------------------------------------------------------
# URL validation checks (no network, no DB connection)
# ---------------------------------------------------------------------------

SPECIAL_CHARS = set("@:/#?%")


def check_url_problems(raw_url: str, parsed) -> list[str]:
    """Return a list of plain-text problem messages for common URL mistakes."""
    problems: list[str] = []

    # Password special chars that need percent-encoding
    pwd = parsed.password or ""
    bad_chars = sorted(set(c for c in pwd if c in SPECIAL_CHARS))
    if bad_chars:
        problems.append(
            "WARNING: Password contains characters that need percent-encoding: "
            + " ".join(bad_chars)
            + ". Percent-encode them (e.g. @ -> %40) or wrap the password in URL-safe quotes."
        )

    # postgres:// instead of postgresql://
    if raw_url.startswith("postgres://"):
        problems.append(
            "WARNING: URL starts with 'postgres://' -- use 'postgresql://' instead. "
            "SQLAlchemy requires the full driver name."
        )

    host = parsed.host or ""
    port = parsed.port
    user = parsed.username or ""

    # Direct db.<ref>.supabase.co host
    direct_match = re.match(
        r"^db\.([a-zA-Z0-9_-]+)\.supabase\.co$", host, re.IGNORECASE
    )
    if direct_match:
        problems.append(
            "INFO: Host is a direct Supabase connection (db.<ref>.supabase.co). "
            "This requires IPv6 or a direct network path. "
            "Use the Session pooler string if you are behind NAT."
        )

    # Port check
    if port is not None and port not in (5432, 6543):
        problems.append(
            f"WARNING: Port {port} is not the standard Supabase port (5432 for direct, "
            "6543 for Session pooler). Verify the connection string."
        )

    # Username conventions
    if direct_match:
        # Direct host -- username should be 'postgres'
        if user != "postgres":
            problems.append(
                f"WARNING: Direct Supabase host expects username 'postgres', got '{user}'."
            )
    elif re.search(r"\.supabase(?:\.co|\.com)$", host, re.IGNORECASE):
        # Pooler host -- username should look like postgres.<ref>
        if not re.match(r"^postgres\.[a-zA-Z0-9_-]+$", user, re.IGNORECASE):
            problems.append(
                f"WARNING: Supabase Session/Transaction pooler expects username "
                f"'postgres.<project_ref>', got '{user}'."
            )

    return problems


# ---------------------------------------------------------------------------
# DNS resolution
# ---------------------------------------------------------------------------

def resolve_host(host: str) -> tuple[bool, bool]:
    """Return (has_ipv4, has_ipv6) for host."""
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False, False
    has_ipv4 = any(r[0] == socket.AF_INET for r in results)
    has_ipv6 = any(r[0] == socket.AF_INET6 for r in results)
    return has_ipv4, has_ipv6


# ---------------------------------------------------------------------------
# Error categorisation
# ---------------------------------------------------------------------------

_ERROR_CATEGORIES: list[tuple[str, str, str]] = [
    # (category, pattern, one-line fix)
    (
        "PASSWORD_WRONG",
        r"(password authentication failed|authentication failed for user)",
        "Fix: Check CATCHYLAB_API_KEY and DATABASE_URL password. Percent-encode special characters.",
    ),
    (
        "TENANT_OR_USER_NOT_FOUND",
        r"(tenant or user not found|no pg_hba|role .* does not exist)",
        "Fix: Verify the username in DATABASE_URL. For Supabase pooler use 'postgres.<project_ref>'.",
    ),
    (
        "HOST_NOT_FOUND",
        r"(could not translate host name|name or service not known|nodename nor servname|"
        r"getaddrinfo failed|no such host)",
        "Fix: Check the host in DATABASE_URL. Confirm DNS resolves the Supabase host name.",
    ),
    (
        "NETWORK_UNREACHABLE",
        r"(connection refused|network is unreachable|no route to host|errno 111|errno 101)",
        "Fix: The host is reachable by DNS but the TCP connection was refused. "
        "Check firewall rules, VPN, or Supabase connection limits.",
    ),
    (
        "TIMEOUT",
        r"(timeout expired|connection timed out|timed out|errno 110)",
        "Fix: Connection timed out. Check firewall, VPN, or increase the timeout.",
    ),
    (
        "SSL",
        r"(ssl|certificate|handshake|sslv3)",
        "Fix: SSL negotiation failed. Try adding '?sslmode=require' or '?sslmode=disable' "
        "to DATABASE_URL.",
    ),
]


def categorise_error(exc: Exception) -> tuple[str, str]:
    """Return (category, one_line_fix) for a connection exception."""
    msg = str(exc).lower()
    for category, pattern, fix in _ERROR_CATEGORIES:
        if re.search(pattern, msg, re.IGNORECASE):
            return category, fix
    return "OTHER", "Fix: Check DATABASE_URL, network connectivity, and Supabase project status."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate DATABASE_URL before starting the demo server."
    )
    parser.add_argument(
        "--expect-project",
        metavar="ID",
        default="",
        help="Fail unless the host or username contains this Supabase project id.",
    )
    args = parser.parse_args(argv)

    # Load .env (existing env vars win)
    env_path = Path(__file__).resolve().parent / ".env"
    _load_dotenv(env_path)

    raw_url = os.environ.get("DATABASE_URL", "")
    if not raw_url:
        print("ERROR: DATABASE_URL is not set.")
        return 1

    # Parse URL with SQLAlchemy
    try:
        from sqlalchemy.engine import make_url
        parsed = make_url(raw_url)
    except Exception as exc:
        print(f"ERROR: Cannot parse DATABASE_URL: {_scrub(str(exc))}")
        return 1

    host = parsed.host or "unknown"
    port = parsed.port or "unknown"
    database = parsed.database or "unknown"
    user = parsed.username or "unknown"
    pwd_len = len(parsed.password) if parsed.password else 0

    # Print safe fields only -- never the password or full URL
    print(f"Host     : {host}")
    print(f"Port     : {port}")
    print(f"Database : {database}")
    print(f"User     : {user}")
    print(f"Password : {'*' * min(pwd_len, 8)} ({pwd_len} chars)")

    # --expect-project check
    expect_id = args.expect_project.strip()
    if expect_id:
        if expect_id not in host and expect_id not in user:
            print(
                f"ERROR: Neither host '{host}' nor user '{user}' contains "
                f"expected project id '{expect_id}'."
            )
            return 1
        print(f"Project  : OK ('{expect_id}' found in host or user)")

    # URL validation warnings (no network)
    problems = check_url_problems(raw_url, parsed)
    for p in problems:
        print(p)

    # DNS resolution
    print(f"DNS      : resolving '{host}' ...")
    has_ipv4, has_ipv6 = resolve_host(host)
    if has_ipv4:
        print("DNS      : IPv4 address found -- OK")
    elif has_ipv6:
        print("DNS      : IPv6 address found (no IPv4)")
        print(
            "DNS      : Direct connection needs IPv6. "
            "Use the Session pooler string from the Supabase Connect screen."
        )
    else:
        print("DNS      : neither IPv4 nor IPv6 address found -- host resolution failed")

    # Connection test: SELECT 1 only, 10-second timeout
    print("DB       : attempting SELECT 1 (10 s timeout) ...")
    try:
        import sqlalchemy
        engine = sqlalchemy.create_engine(
            raw_url,
            connect_args={"connect_timeout": 10},
            pool_pre_ping=False,
        )
        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT 1"))
            row = result.fetchone()
            if row and row[0] == 1:
                print("DB       : SELECT 1 -- OK")
            else:
                print("DB       : SELECT 1 returned unexpected value")
                return 1
        engine.dispose()
    except Exception as exc:
        category, fix = categorise_error(exc)
        scrubbed = _scrub(str(exc))
        print(f"DB       : FAILED [{category}] {scrubbed}")
        print(f"DB       : {fix}")
        return 1

    print("OK -- database check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
