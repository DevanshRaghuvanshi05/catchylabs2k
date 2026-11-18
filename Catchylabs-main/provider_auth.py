"""Authentication and encryption helpers for user-owned provider connections."""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx
from fastapi import HTTPException, status


def _supabase_settings() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not url or not anon_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CatchyLab account authentication is not configured.",
        )
    return url, anon_key


async def verify_supabase_access_token(access_token: str) -> dict[str, Any]:
    """Ask Supabase Auth to validate the bearer token; never trust a body user ID."""
    url, anon_key = _supabase_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{url}/auth/v1/user",
                headers={
                    "apikey": anon_key,
                    "Authorization": f"Bearer {access_token}",
                },
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CatchyLab account verification timed out.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CatchyLab account verification is unavailable.",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid CatchyLab session.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = response.json()
    if not isinstance(user, dict) or not isinstance(user.get("id"), str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CatchyLab session did not contain a verified user identity.",
        )
    return user


def _encryption_key() -> bytes:
    encoded_key = os.environ.get("PROVIDER_ENCRYPTION_KEY", "")
    if not encoded_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider credential encryption is not configured.",
        )
    try:
        key = base64.urlsafe_b64decode(encoded_key + "=" * (-len(encoded_key) % 4))
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider credential encryption configuration is invalid.",
        ) from exc
    if len(key) != 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider credential encryption key must encode 32 random bytes.",
        )
    return key


def encrypt_provider_secret(user_id: str, provider: str, plaintext: str) -> str:
    """Encrypt a credential with AES-GCM and contextual authenticated data."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _encryption_key()
    nonce = os.urandom(12)
    aad = f"catchylab:provider:v1:{user_id}:{provider}".encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), aad)
    return "v1." + base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")


def decrypt_provider_secret(user_id: str, provider: str, envelope: str) -> str:
    """Decrypt only for the authenticated owner and expected provider."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not envelope.startswith("v1."):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored provider credential uses an unsupported encryption version.",
        )
    key = _encryption_key()
    try:
        payload = base64.urlsafe_b64decode(envelope[3:])
        if len(key) != 32 or len(payload) < 29:
            raise ValueError
        aad = f"catchylab:provider:v1:{user_id}:{provider}".encode("utf-8")
        return AESGCM(key).decrypt(payload[:12], payload[12:], aad).decode("utf-8")
    except (InvalidTag, ValueError, UnicodeDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored provider credential could not be decrypted.",
        ) from exc
