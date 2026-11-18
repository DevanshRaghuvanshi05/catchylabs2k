"""
Centralized BYOK (Bring Your Own API Key) credential resolution for the CatchyLabs research pipeline.

Security contract:
- Plaintext keys are NEVER stored in this module's variables beyond the scope of a single call.
- Plaintext keys are NEVER logged.
- Plaintext keys are NEVER placed in job queues or Redis payloads.
- Only user_id travels across asynchronous boundaries; keys are resolved inside trusted worker processes.
- Encryption/decryption delegates exclusively to provider_auth.py (AES-256-GCM with AAD).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status

logger = logging.getLogger("CatchyLabCredentials")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_db():
    """Import SessionLocal lazily to avoid circular imports at module load time."""
    from database import SessionLocal
    return SessionLocal()


def _encrypt(user_id: str, provider: str, plaintext: str) -> str:
    from provider_auth import encrypt_provider_secret
    return encrypt_provider_secret(user_id, provider, plaintext)


def _decrypt(user_id: str, provider: str, envelope: str) -> str:
    from provider_auth import decrypt_provider_secret
    return decrypt_provider_secret(user_id, provider, envelope)


def _mask_key(key: str) -> str:
    """Return masked representation like ••••••••••••abcd (last 4 chars visible)."""
    if not key or len(key) < 4:
        return "••••"
    return "••••••••••••" + key[-4:]


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def store_credential(user_id: str, provider: str, plaintext_key: str) -> dict:
    """
    Encrypt and upsert a credential for user+provider.
    Returns masked status dict (never the plaintext key).
    """
    from database import UserAPICredential

    encrypted = _encrypt(user_id, provider, plaintext_key)
    masked = _mask_key(plaintext_key)
    # plaintext_key reference ends here — no further use

    db = _get_db()
    try:
        existing = (
            db.query(UserAPICredential)
            .filter_by(user_id=user_id, provider=provider)
            .first()
        )
        now = datetime.now(timezone.utc)
        if existing:
            existing.encrypted_key = encrypted
            existing.masked_key = masked
            existing.key_version = "v1"
            existing.is_valid = True
            existing.updated_at = now
            db.commit()
            logger.info(f"[CredentialManager] Updated {provider} credential for user {user_id[:8]}...")
        else:
            record = UserAPICredential(
                id=str(uuid.uuid4()),
                user_id=user_id,
                provider=provider,
                encrypted_key=encrypted,
                masked_key=masked,
                key_version="v1",
                is_valid=True,
                created_at=now,
                updated_at=now,
            )
            db.add(record)
            db.commit()
            logger.info(f"[CredentialManager] Stored new {provider} credential for user {user_id[:8]}...")
        return {
            "provider": provider,
            "masked_key": masked,
            "is_valid": True,
            "configured": True,
        }
    finally:
        db.close()


def delete_credential(user_id: str, provider: str) -> bool:
    """
    Delete a credential for user+provider.
    Returns True if a record was found and deleted, False if none existed.
    """
    from database import UserAPICredential

    db = _get_db()
    try:
        record = (
            db.query(UserAPICredential)
            .filter_by(user_id=user_id, provider=provider)
            .first()
        )
        if record:
            db.delete(record)
            db.commit()
            logger.info(f"[CredentialManager] Deleted {provider} credential for user {user_id[:8]}...")
            return True
        return False
    finally:
        db.close()


def get_credential_status(user_id: str) -> dict:
    """
    Return masked status for all providers for the given user.
    Never returns plaintext keys.
    """
    from database import UserAPICredential

    db = _get_db()
    try:
        records = (
            db.query(UserAPICredential)
            .filter_by(user_id=user_id)
            .all()
        )
        status_map: dict = {}
        for rec in records:
            status_map[rec.provider] = {
                "provider": rec.provider,
                "masked_key": rec.masked_key or "••••••••••••????",
                "is_valid": rec.is_valid,
                "configured": True,
                "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
            }
        # Fill in unconfigured providers
        for prov in ("gemini", "serpapi"):
            if prov not in status_map:
                status_map[prov] = {
                    "provider": prov,
                    "masked_key": None,
                    "is_valid": False,
                    "configured": False,
                    "updated_at": None,
                }
        return status_map
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Key resolution (used inside worker processes)
# ---------------------------------------------------------------------------

def get_user_credential(user_id: str, provider: str) -> Optional[str]:
    """
    Decrypt and return the plaintext API key for user+provider.
    Returns None if no credential is configured for this user+provider.
    Logs a warning but never logs the decrypted key itself.
    """
    from database import UserAPICredential

    db = _get_db()
    try:
        record = (
            db.query(UserAPICredential)
            .filter_by(user_id=user_id, provider=provider, is_valid=True)
            .first()
        )
        if not record:
            return None
        return _decrypt(user_id, provider, record.encrypted_key)
    except HTTPException:
        # Re-raise HTTP exceptions (encryption misconfiguration, etc.)
        raise
    except Exception as exc:
        logger.error(
            f"[CredentialManager] Failed to decrypt {provider} credential for user "
            f"{user_id[:8] if user_id else 'unknown'}...: {type(exc).__name__}"
        )
        return None
    finally:
        db.close()


def get_gemini_key(user_id: Optional[str] = None) -> str:
    """
    Resolve the Gemini API key for the given user.

    Priority:
      1. User's own configured key (from user_api_credentials table).
      2. Platform fallback key — ONLY if ALLOW_PLATFORM_KEY_FALLBACK=1.

    Raises HTTP 402 if no key is available, so callers never silently use
    developer credentials when the user hasn't configured their own.
    """
    from config import GEMINI_API_KEY, ALLOW_PLATFORM_KEY_FALLBACK

    if user_id:
        key = get_user_credential(user_id, "gemini")
        if key:
            return key

    if ALLOW_PLATFORM_KEY_FALLBACK and GEMINI_API_KEY:
        logger.debug("[CredentialManager] Using platform Gemini key (ALLOW_PLATFORM_KEY_FALLBACK=1)")
        return GEMINI_API_KEY

    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=(
            "No Gemini API key configured. "
            "Please add your Gemini API key via POST /api/credentials/gemini."
        ),
    )


def get_serpapi_key(user_id: Optional[str] = None) -> str:
    """
    Resolve the SerpAPI key for the given user.

    Priority:
      1. User's own configured key.
      2. Platform fallback — ONLY if ALLOW_PLATFORM_KEY_FALLBACK=1.

    Raises HTTP 402 if no key is available.
    """
    from config import SERPAPI_API_KEY, ALLOW_PLATFORM_KEY_FALLBACK

    if user_id:
        key = get_user_credential(user_id, "serpapi")
        if key:
            return key

    if ALLOW_PLATFORM_KEY_FALLBACK and SERPAPI_API_KEY:
        logger.debug("[CredentialManager] Using platform SerpAPI key (ALLOW_PLATFORM_KEY_FALLBACK=1)")
        return SERPAPI_API_KEY

    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=(
            "No SerpAPI key configured. "
            "Please add your SerpAPI key via POST /api/credentials/serpapi."
        ),
    )


def get_gemini_client(user_id: Optional[str] = None):
    """
    Return an authenticated google.genai.Client using the resolved Gemini key.
    Raises HTTP 402 if neither user key nor platform fallback is available.
    """
    from google import genai
    api_key = get_gemini_key(user_id)
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Key validation (called at submission time, not in workers)
# ---------------------------------------------------------------------------

def validate_gemini_key(api_key: str) -> bool:
    """
    Attempt a minimal Gemini API call to verify the key is valid.
    Returns True if the key is accepted, False otherwise.
    Never logs the key.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Use the models list endpoint — cheapest possible verification call
        models = client.models.list()
        # If we can iterate at all, the key is valid
        _ = next(iter(models), None)
        return True
    except Exception as exc:
        logger.warning(f"[CredentialManager] Gemini key validation failed: {type(exc).__name__}: {exc}")
        return False


def validate_serpapi_key(api_key: str) -> bool:
    """
    Call the SerpAPI /account endpoint to verify the key.
    Returns True if the key is accepted, False otherwise.
    Never logs the key.
    """
    try:
        import requests
        response = requests.get(
            "https://serpapi.com/account",
            params={"api_key": api_key},
            timeout=8,
        )
        return response.status_code == 200
    except Exception as exc:
        logger.warning(f"[CredentialManager] SerpAPI key validation failed: {type(exc).__name__}: {exc}")
        return False
