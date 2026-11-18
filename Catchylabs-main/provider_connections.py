"""Per-user AI provider credentials, model verification, and generation routes."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Self
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from database import ProviderConnection, ProviderGenerationUsage, ProviderOAuthState, SessionLocal
from provider_auth import (
    decrypt_provider_secret,
    encrypt_provider_secret,
    verify_supabase_access_token,
)
from intelligence.language import output_language_instruction, resolve_language

router = APIRouter(prefix="/providers", tags=["AI provider connections"])
logger = logging.getLogger("CatchyLabProviders")
ProviderName = Literal["gemini", "claude", "openai"]
PROVIDER_SCOPES = ["https://www.googleapis.com/auth/generative-language.retriever"]
PROVIDER_URLS = {
    "gemini": "https://generativelanguage.googleapis.com",
    "claude": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}
MAX_GENERATIONS_PER_MINUTE = 20


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid CatchyLab account session is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await verify_supabase_access_token(authorization[7:])


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class GeminiOAuthStartRequest(BaseModel):
    project_id: str = Field(min_length=6, max_length=30, pattern=r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class GenerateRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    language_preference: Literal["auto", "english", "hindi", "hinglish"] = "auto"

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("The last chat message must be from the user.")
        if sum(len(message.content) for message in self.messages) > 12000:
            raise ValueError("Chat history exceeds the 12000 character limit.")
        return self


def _provider_headers(provider: str, credential: dict[str, Any] | str) -> dict[str, str]:
    secret = credential if isinstance(credential, str) else credential.get("access_token", "")
    if provider == "gemini":
        if isinstance(credential, str):
            return {"x-goog-api-key": secret}
        project_id = credential.get("quota_project_id")
        if not isinstance(project_id, str) or not project_id:
            raise HTTPException(status_code=500, detail="Gemini OAuth connection has no quota project configured.")
        return {
            "Authorization": f"Bearer {secret}",
            "x-goog-user-project": project_id,
        }
    if provider == "claude":
        return {
            "Authorization": f"Bearer {secret}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    return {"Authorization": f"Bearer {secret}"}


async def _provider_request(
    provider: str,
    credential: dict[str, Any] | str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    headers = _provider_headers(provider, credential)
    if provider == "gemini" and isinstance(credential, str):
        headers["content-type"] = "application/json"
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.request(
                method,
                f"{PROVIDER_URLS[provider]}{path}",
                headers=headers,
                params=params,
                json=body,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"{provider} did not respond before the timeout.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach {provider}.") from exc
    return response


def _raise_provider_error(provider: str, response: httpx.Response) -> None:
    if response.is_success:
        return
    code = response.status_code
    logger.warning("AI provider request failed: provider=%s status=%s", provider, code)
    if code == 401:
        detail = f"{provider} rejected the credential. Check the API key or reconnect OAuth."
        mapped_status = status.HTTP_401_UNAUTHORIZED
    elif code == 403:
        detail = f"{provider} denied API access. Check project, account, model, or billing permissions."
        mapped_status = status.HTTP_403_FORBIDDEN
    elif code == 429:
        detail = f"{provider} rate limit or quota was reached. Wait or check the provider account's usage limits."
        mapped_status = status.HTTP_429_TOO_MANY_REQUESTS
    elif code >= 500:
        detail = f"{provider} is temporarily unavailable. Retry after the provider recovers."
        mapped_status = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        detail = f"{provider} rejected the request with status {code}."
        mapped_status = status.HTTP_502_BAD_GATEWAY
    raise HTTPException(status_code=mapped_status, detail=detail)


def _response_object(provider: str, response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"{provider} returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail=f"{provider} returned an unrecognized response.")
    return payload


async def _list_models(provider: str, credential: dict[str, Any] | str) -> list[str]:
    if provider == "gemini":
        response = await _provider_request(provider, credential, "GET", "/v1beta/models")
        _raise_provider_error(provider, response)
        payload = _response_object(provider, response)
        provider_models = payload.get("models", [])
        if not isinstance(provider_models, list):
            raise HTTPException(status_code=502, detail="Gemini returned an unrecognized model list.")
        return sorted(
            item["name"].removeprefix("models/")
            for item in provider_models
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("supportedGenerationMethods"), list)
            and "generateContent" in item["supportedGenerationMethods"]
        )
    if provider == "claude":
        response = await _provider_request(provider, credential, "GET", "/v1/models", params={"limit": 1000})
    else:
        response = await _provider_request(provider, credential, "GET", "/v1/models")
    _raise_provider_error(provider, response)
    payload = _response_object(provider, response)
    provider_models = payload.get("data", [])
    if not isinstance(provider_models, list):
        raise HTTPException(status_code=502, detail=f"{provider} returned an unrecognized model list.")
    return sorted(
        item["id"]
        for item in provider_models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


async def _execute_model(
    provider: str,
    credential: dict[str, Any] | str,
    model: str,
    messages: list[dict[str, str]],
    language_preference: str | None = None,
) -> str:
    system_instruction = None
    if language_preference is not None:
        latest_user_message = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        resolved = resolve_language(latest_user_message, language_preference)
        system_instruction = (
            output_language_instruction(resolved["language"])
            + " Treat the requested language as the default; follow a clear language change requested in the user's message."
        )
    if provider == "gemini":
        body = {
            "contents": [
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}],
                }
                for message in messages
            ],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.2},
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        response = await _provider_request(
            provider,
            credential,
            "POST",
            f"/v1beta/models/{quote(model, safe='')}:generateContent",
            body=body,
        )
        _raise_provider_error(provider, response)
        data = _response_object(provider, response)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(
                part["text"] for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(status_code=502, detail="Gemini returned an unrecognized response.") from exc
    if provider == "claude":
        response = await _provider_request(
            provider,
            credential,
            "POST",
            "/v1/messages",
            body={
                "model": model,
                "max_tokens": 256,
                "messages": messages,
                **({"system": system_instruction} if system_instruction else {}),
            },
        )
        _raise_provider_error(provider, response)
        data = _response_object(provider, response)
        try:
            return "\n".join(
                block["text"] for block in data["content"]
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        except (KeyError, TypeError) as exc:
            raise HTTPException(status_code=502, detail="Claude returned an unrecognized response.") from exc
    response = await _provider_request(
        provider,
        credential,
        "POST",
        "/v1/responses",
        body={
            "model": model,
            "input": (
                [{"role": "system", "content": system_instruction}, *messages]
                if system_instruction
                else messages
            ),
            "max_output_tokens": 256,
        },
    )
    _raise_provider_error(provider, response)
    data = _response_object(provider, response)
    try:
        return "\n".join(
            block["text"]
            for item in data["output"]
            for block in item["content"]
            if isinstance(item, dict)
            and isinstance(item.get("content"), list)
            and isinstance(block, dict)
            and block.get("type") == "output_text"
            and isinstance(block.get("text"), str)
        )
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="OpenAI returned an unrecognized response.") from exc


def _connection_json(connection: ProviderConnection) -> dict[str, Any]:
    models = connection.available_models or []
    return {
        "provider": connection.provider,
        "auth_method": connection.auth_method,
        "status": connection.status,
        "models": [
            {
                "provider_id": connection.provider,
                "model_id": model,
                "display_name": model,
                "capabilities": ["text_generation"],
                "availability": "available",
                "verification_status": (
                    "verified"
                    if connection.status == "ready" and connection.selected_model == model
                    else "not_tested"
                ),
            }
            for model in models
            if isinstance(model, str)
        ],
        "selected_model": connection.selected_model,
        "is_active": connection.is_active,
        "updated_at": connection.updated_at.isoformat() if connection.updated_at else None,
    }


def _owned_connection(db: Session, user_id: str, provider: str) -> ProviderConnection:
    connection = db.query(ProviderConnection).filter_by(user_id=user_id, provider=provider).one_or_none()
    if connection is None:
        raise HTTPException(status_code=404, detail=f"No {provider} connection was found for this account.")
    return connection


def _reserve_generation(db: Session, user_id: str) -> None:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=1)
    recent_count = (
        db.query(ProviderGenerationUsage)
        .filter(
            ProviderGenerationUsage.user_id == user_id,
            ProviderGenerationUsage.created_at >= window_start,
        )
        .count()
    )
    if recent_count >= MAX_GENERATIONS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"CatchyLab generation limit reached ({MAX_GENERATIONS_PER_MINUTE} requests per minute). Wait before retrying.",
        )
    db.add(ProviderGenerationUsage(user_id=user_id, created_at=now))
    db.commit()


async def _connection_credential(
    db: Session,
    connection: ProviderConnection,
    user_id: str,
) -> dict[str, Any] | str:
    secret_text = decrypt_provider_secret(user_id, connection.provider, connection.credential_ciphertext)
    if connection.auth_method != "oauth":
        return secret_text
    try:
        credential = json.loads(secret_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored provider authorization is invalid.") from exc
    if not isinstance(credential, dict) or not isinstance(credential.get("access_token"), str):
        raise HTTPException(status_code=500, detail="Stored provider authorization is invalid.")

    expiry = connection.token_expires_at
    now = datetime.now(timezone.utc)
    expiry_utc = (
        expiry.replace(tzinfo=timezone.utc)
        if expiry and expiry.tzinfo is None
        else expiry.astimezone(timezone.utc) if expiry else None
    )
    if expiry_utc and expiry_utc > now + timedelta(seconds=60):
        return credential

    connection = (
        db.query(ProviderConnection)
        .filter_by(id=connection.id, user_id=user_id)
        .populate_existing()
        .with_for_update()
        .one_or_none()
    )
    if connection is None:
        raise HTTPException(status_code=404, detail="Provider connection was removed.")
    secret_text = decrypt_provider_secret(user_id, connection.provider, connection.credential_ciphertext)
    try:
        credential = json.loads(secret_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Stored provider authorization is invalid.") from exc
    if not isinstance(credential, dict) or not isinstance(credential.get("access_token"), str):
        raise HTTPException(status_code=500, detail="Stored provider authorization is invalid.")
    expiry = connection.token_expires_at
    expiry_utc = (
        expiry.replace(tzinfo=timezone.utc)
        if expiry and expiry.tzinfo is None
        else expiry.astimezone(timezone.utc) if expiry else None
    )
    if expiry_utc and expiry_utc > datetime.now(timezone.utc) + timedelta(seconds=60):
        return credential
    refresh_token = credential.get("refresh_token")
    if not refresh_token:
        connection.status = "reauthorization_required"
        db.commit()
        raise HTTPException(status_code=401, detail="Google authorization expired. Reconnect Gemini to continue.")

    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise HTTPException(status_code=503, detail="Gemini OAuth refresh is not configured.")
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Could not refresh Gemini authorization.") from exc
    if not response.is_success:
        connection.status = "reauthorization_required"
        db.commit()
        raise HTTPException(status_code=401, detail="Google authorization expired. Reconnect Gemini to continue.")
    token_data = _response_object("Google OAuth", response)
    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in", 3600)
    rotated_refresh_token = token_data.get("refresh_token")
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(expires_in, int)
        or isinstance(expires_in, bool)
        or expires_in <= 0
        or expires_in > 86400
        or (rotated_refresh_token is not None and not isinstance(rotated_refresh_token, str))
    ):
        raise HTTPException(status_code=502, detail="Google returned an invalid refresh response.")
    credential["access_token"] = access_token
    credential["refresh_token"] = rotated_refresh_token or refresh_token
    connection.credential_ciphertext = encrypt_provider_secret(
        user_id, connection.provider, json.dumps(credential)
    )
    connection.token_expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=expires_in
    )
    db.commit()
    return credential


def _save_connection(
    db: Session,
    user_id: str,
    provider: str,
    method: str,
    secret_payload: str,
    models: list[str],
    scopes: list[str] | None = None,
    expires_at: datetime | None = None,
) -> ProviderConnection:
    connection = db.query(ProviderConnection).filter_by(user_id=user_id, provider=provider).one_or_none()
    if connection is None:
        connection = ProviderConnection(user_id=user_id, provider=provider)
        db.add(connection)
    connection.auth_method = method
    connection.credential_ciphertext = encrypt_provider_secret(user_id, provider, secret_payload)
    connection.status = "authorized" if models else "unavailable"
    connection.available_models = models
    connection.selected_model = None
    connection.is_active = False
    connection.granted_scopes = scopes or []
    connection.token_expires_at = expires_at
    db.commit()
    db.refresh(connection)
    return connection


@router.get("/connections")
async def list_connections(
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    connections = db.query(ProviderConnection).filter_by(user_id=user["id"]).all()
    return {"connections": [_connection_json(connection) for connection in connections]}


@router.post("/{provider}/api-key")
async def connect_api_key(
    provider: ProviderName,
    request: Request,
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="A valid JSON provider credential request is required.") from exc
    api_key = body.get("api_key") if isinstance(body, dict) else None
    if not isinstance(api_key, str) or not 12 <= len(api_key) <= 4096:
        raise HTTPException(status_code=422, detail="Provider API key must be between 12 and 4096 characters.")
    models = await _list_models(provider, api_key)
    if not models:
        raise HTTPException(status_code=422, detail=f"{provider} returned no models available for generation.")
    connection = _save_connection(
        db, user["id"], provider, "api_key", api_key, models
    )
    return _connection_json(connection)


@router.get("/{provider}/models")
async def refresh_models(
    provider: ProviderName,
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    connection = _owned_connection(db, user["id"], provider)
    secret = await _connection_credential(db, connection, user["id"])
    models = await _list_models(provider, secret)
    selected_model = connection.selected_model if connection.selected_model in models else None
    connection.available_models = models
    connection.selected_model = selected_model
    connection.status = "authorized" if models else "unavailable"
    connection.is_active = False
    db.commit()
    db.refresh(connection)
    return _connection_json(connection)


@router.put("/{provider}/model")
async def select_model(
    provider: ProviderName,
    request: ModelRequest,
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    connection = _owned_connection(db, user["id"], provider)
    if request.model not in (connection.available_models or []):
        raise HTTPException(status_code=422, detail="Select a model returned by the connected provider.")
    secret = await _connection_credential(db, connection, user["id"])
    # A tiny real generation verifies access; no provider/model fallback is attempted.
    verification = await _execute_model(
        provider, secret, request.model, [{"role": "user", "content": "Reply with OK."}]
    )
    if not verification.strip():
        raise HTTPException(status_code=502, detail=f"{provider} returned an empty model verification response.")
    db.query(ProviderConnection).filter_by(user_id=user["id"]).update({"is_active": False})
    connection.selected_model = request.model
    connection.status = "ready"
    connection.is_active = True
    db.commit()
    return _connection_json(connection)


@router.delete("/{provider}/connection")
async def disconnect_provider(
    provider: ProviderName,
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    connection = _owned_connection(db, user["id"], provider)
    revocation_status = "not_applicable"
    if provider == "gemini" and connection.auth_method == "oauth":
        secret_text = decrypt_provider_secret(user["id"], provider, connection.credential_ciphertext)
        secret = json.loads(secret_text)
        token = secret.get("refresh_token") or secret.get("access_token")
        if token:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    response = await client.post("https://oauth2.googleapis.com/revoke", data={"token": token})
                revocation_status = "revoked" if response.is_success else "could_not_revoke"
            except httpx.HTTPError:
                revocation_status = "could_not_revoke"
        else:
            revocation_status = "could_not_revoke"
    else:
        revocation_status = "local_only"
    db.delete(connection)
    db.commit()
    return {"disconnected": True, "provider_revocation": revocation_status}


@router.post("/gemini/oauth/start")
async def start_gemini_oauth(
    request: GeminiOAuthStartRequest,
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", "")
    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=503, detail="Gemini OAuth is not configured for this deployment.")
    redirect_parts = urlsplit(redirect_uri)
    if redirect_parts.scheme != "https" and not (
        redirect_parts.scheme == "http" and redirect_parts.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise HTTPException(status_code=503, detail="Gemini OAuth requires an HTTPS callback outside local development.")
    state = secrets.token_urlsafe(32)
    state_hash = hashlib.sha256(state.encode()).hexdigest()
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    db.query(ProviderOAuthState).filter(
        ProviderOAuthState.expires_at < datetime.now(timezone.utc)
    ).delete(synchronize_session=False)
    db.add(ProviderOAuthState(
        state_hash=state_hash,
        user_id=user["id"],
        quota_project_id=request.project_id,
        verifier_ciphertext=encrypt_provider_secret(user["id"], "gemini", verifier),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    ))
    db.commit()
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(PROVIDER_SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    return {"authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)}


@router.get("/gemini/oauth/callback")
async def gemini_oauth_callback(
    request: Request,
    db: Session = Depends(get_db),
):
    frontend_url = os.environ.get("FRONTEND_URL", "")
    if not frontend_url:
        raise HTTPException(status_code=503, detail="Gemini OAuth callback destination is not configured.")
    failure_url = frontend_url.rstrip("/") + "/?provider_connection=failed"
    raw_query = getattr(request.state, "provider_oauth_query", b"")
    query = parse_qs(raw_query.decode("utf-8", errors="ignore"))
    state = query.get("state", [""])[0]
    code = query.get("code", [None])[0]
    error = query.get("error", [None])[0]
    if error or not code or not state:
        if state:
            denied_state_hash = hashlib.sha256(state.encode()).hexdigest()
            denied_pending = (
                db.query(ProviderOAuthState)
                .filter_by(state_hash=denied_state_hash)
                .with_for_update()
                .one_or_none()
            )
            if denied_pending is not None:
                db.delete(denied_pending)
                db.commit()
        return RedirectResponse(failure_url, status_code=303)

    state_hash = hashlib.sha256(state.encode()).hexdigest()
    pending = (
        db.query(ProviderOAuthState)
        .filter_by(state_hash=state_hash)
        .with_for_update()
        .one_or_none()
    )
    now = datetime.now(timezone.utc)
    expires_at = pending.expires_at if pending else None
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if pending is None or expires_at is None or expires_at <= now:
        if pending is not None:
            db.delete(pending)
            db.commit()
        return RedirectResponse(failure_url, status_code=303)
    user_id = pending.user_id
    quota_project_id = pending.quota_project_id
    verifier = decrypt_provider_secret(user_id, "gemini", pending.verifier_ciphertext)
    db.delete(pending)
    db.commit()
    if not isinstance(quota_project_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9-]{4,28}[a-z0-9]", quota_project_id
    ):
        return RedirectResponse(failure_url, status_code=303)

    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI", "")
    if not client_id or not client_secret or not redirect_uri:
        return RedirectResponse(failure_url, status_code=303)
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "code_verifier": verifier,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
    except httpx.HTTPError:
        return RedirectResponse(failure_url, status_code=303)
    if not token_response.is_success:
        return RedirectResponse(failure_url, status_code=303)
    try:
        token_data = _response_object("Google OAuth", token_response)
    except HTTPException:
        return RedirectResponse(failure_url, status_code=303)
    access_token = token_data.get("access_token")
    granted_scopes = str(token_data.get("scope", "")).split()
    expires_in = token_data.get("expires_in", 3600)
    refresh_token = token_data.get("refresh_token")
    token_type = token_data.get("token_type")
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(expires_in, int)
        or isinstance(expires_in, bool)
        or expires_in <= 0
        or expires_in > 86400
        or (refresh_token is not None and not isinstance(refresh_token, str))
        or token_type not in (None, "Bearer")
        or not set(PROVIDER_SCOPES).issubset(granted_scopes)
    ):
        return RedirectResponse(failure_url, status_code=303)
    credential = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": token_type,
        "quota_project_id": quota_project_id,
    }
    try:
        models = await _list_models("gemini", credential)
    except HTTPException:
        return RedirectResponse(failure_url, status_code=303)
    if not models:
        return RedirectResponse(failure_url, status_code=303)
    expires_at = now + timedelta(seconds=expires_in)
    _save_connection(
        db, user_id, "gemini", "oauth", json.dumps(credential), models, granted_scopes, expires_at
    )
    return RedirectResponse(frontend_url.rstrip("/") + "/?provider_connection=authorized", status_code=303)


@router.post("/{provider}/generate")
async def generate(
    provider: ProviderName,
    request: GenerateRequest,
    user: dict[str, Any] = Depends(current_user),
    db: Session = Depends(get_db),
):
    connection = _owned_connection(db, user["id"], provider)
    if connection.status != "ready" or not connection.selected_model or not connection.is_active:
        raise HTTPException(status_code=409, detail="Choose and verify a model before generating.")
    if request.model != connection.selected_model:
        raise HTTPException(
            status_code=409,
            detail="Your selected model changed. Refresh connections and explicitly choose a model again.",
        )
    _reserve_generation(db, user["id"])
    secret = await _connection_credential(db, connection, user["id"])
    output = await _execute_model(
        provider,
        secret,
        request.model,
        [message.model_dump() for message in request.messages],
        request.language_preference,
    )
    if not output.strip():
        raise HTTPException(status_code=502, detail=f"{provider} returned an empty response.")
    return {"provider": provider, "model": connection.selected_model, "text": output}
