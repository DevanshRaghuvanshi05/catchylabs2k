import asyncio
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite://")

import database
import provider_connections
from database import Base, ProviderConnection, ProviderGenerationUsage, ProviderOAuthState
from provider_auth import decrypt_provider_secret, encrypt_provider_secret, verify_supabase_access_token


class ProviderConnectionTests(unittest.TestCase):
    def setUp(self):
        os.environ["PROVIDER_ENCRYPTION_KEY"] = base64.urlsafe_b64encode(os.urandom(32)).decode()
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(
            self.engine,
            tables=[
                ProviderConnection.__table__,
                ProviderOAuthState.__table__,
                ProviderGenerationUsage.__table__,
            ],
        )
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.app = FastAPI()

        def db_dependency():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        async def fake_user(authorization: str | None = Header(default=None)):
            if not authorization or not authorization.startswith("Bearer user-"):
                raise HTTPException(status_code=401, detail="Invalid test session.")
            return {"id": authorization.removeprefix("Bearer ")}

        self.app.dependency_overrides[provider_connections.get_db] = db_dependency
        self.app.dependency_overrides[provider_connections.current_user] = fake_user

        @self.app.middleware("http")
        async def hide_oauth_query(request, call_next):
            if request.url.path == "/providers/gemini/oauth/callback":
                request.state.provider_oauth_query = request.scope.get("query_string", b"")
                request.scope["query_string"] = b""
            return await call_next(request)

        self.app.include_router(provider_connections.router)
        self.client = TestClient(self.app, follow_redirects=False)

    def tearDown(self):
        self.client.close()
        self.engine.dispose()

    def db(self):
        return self.session_factory()

    def test_api_key_connection_with_empty_provider_catalog_does_not_create_connection(self):
        with patch("provider_connections._list_models", new=AsyncMock(return_value=[])):
            response = self.client.post(
                "/providers/gemini/api-key",
                headers={"Authorization": "Bearer user-one"},
                json={"api_key": "AIzaTestKeyWithEnoughLength123"},
            )
        self.assertEqual(response.status_code, 422)
        self.assertIn("no models", response.json()["detail"])
        db = self.db()
        try:
            self.assertEqual(db.query(ProviderConnection).count(), 0)
        finally:
            db.close()

    def test_provider_tables_are_registered_for_creation_and_rls(self):
        self.assertIn("provider_connections", database.RLS_TABLES)
        self.assertIn("provider_oauth_states", database.RLS_TABLES)
        self.assertIn("provider_generation_usage", database.RLS_TABLES)
        self.assertIn("user_id", ProviderConnection.__table__.columns)
        self.assertIn("state_hash", ProviderOAuthState.__table__.columns)

    def test_encrypted_secret_is_bound_to_user_and_provider(self):
        ciphertext = encrypt_provider_secret("user-a", "openai", "sk-private-example")
        self.assertNotIn("sk-private-example", ciphertext)
        self.assertEqual(decrypt_provider_secret("user-a", "openai", ciphertext), "sk-private-example")
        with self.assertRaises(HTTPException):
            decrypt_provider_secret("user-b", "openai", ciphertext)
        with self.assertRaises(HTTPException):
            decrypt_provider_secret("user-a", "claude", ciphertext)

    def test_supabase_token_is_validated_server_side_and_identity_is_not_guessed(self):
        response = httpx.Response(200, json={"id": "verified-user"})
        with patch("provider_auth.httpx.AsyncClient") as async_client:
            async_client.return_value.__aenter__ = AsyncMock()
            async_client.return_value.__aexit__ = AsyncMock(return_value=False)
            async_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
            os.environ["SUPABASE_URL"] = "https://sample.supabase.co"
            os.environ["SUPABASE_ANON_KEY"] = "public-test-anon-key"
            user = asyncio.run(verify_supabase_access_token("signed-token"))
        self.assertEqual(user["id"], "verified-user")

    def test_provider_response_parsers_extract_only_text_blocks(self):
        examples = [
            (
                "gemini",
                {"candidates": [{"content": {"parts": [{"text": "Gemini answer"}]}}]},
                "Gemini answer",
            ),
            (
                "claude",
                {"content": [{"type": "thinking", "text": "hidden"}, {"type": "text", "text": "Claude answer"}]},
                "Claude answer",
            ),
            (
                "openai",
                {"output": [{"content": [{"type": "refusal", "text": "hidden"}, {"type": "output_text", "text": "OpenAI answer"}]}]},
                "OpenAI answer",
            ),
        ]
        for provider, body, expected in examples:
            with self.subTest(provider=provider):
                response = httpx.Response(200, json=body)
                with patch("provider_connections._provider_request", new=AsyncMock(return_value=response)):
                    result = asyncio.run(provider_connections._execute_model(
                        provider,
                        "provider-secret",
                        "model-exact",
                        [{"role": "user", "content": "Prompt"}],
                    ))
                self.assertEqual(result, expected)

    def test_provider_errors_preserve_auth_quota_and_outage_semantics(self):
        expected_statuses = {401: 401, 403: 403, 429: 429, 503: 503}
        for provider_status, expected_status in expected_statuses.items():
            with self.subTest(provider_status=provider_status):
                with self.assertRaises(HTTPException) as raised:
                    provider_connections._raise_provider_error(
                        "openai", httpx.Response(provider_status)
                    )
                self.assertEqual(raised.exception.status_code, expected_status)
                self.assertNotIn("api-key", raised.exception.detail.lower())

    def test_oauth_uses_user_quota_project_and_claude_uses_supported_bearer_auth(self):
        oauth_headers = provider_connections._provider_headers(
            "gemini",
            {"access_token": "access-token", "quota_project_id": "user-project-123"},
        )
        self.assertEqual(oauth_headers["x-goog-user-project"], "user-project-123")
        self.assertEqual(oauth_headers["Authorization"], "Bearer access-token")
        claude_headers = provider_connections._provider_headers("claude", "claude-api-key")
        self.assertEqual(claude_headers["Authorization"], "Bearer claude-api-key")
        self.assertEqual(claude_headers["anthropic-version"], "2023-06-01")

    def test_api_key_connection_verifies_models_and_never_returns_or_stores_plaintext(self):
        with patch("provider_connections._list_models", new=AsyncMock(return_value=["gpt-test-model"])):
            response = self.client.post(
                "/providers/openai/api-key",
                headers={"Authorization": "Bearer user-one"},
                json={"api_key": "sk-test-secret-key-123"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("sk-test-secret-key-123", response.text)
        self.assertEqual(response.json()["status"], "authorized")
        self.assertEqual(response.json()["models"][0]["model_id"], "gpt-test-model")
        self.assertEqual(response.json()["models"][0]["verification_status"], "not_tested")
        db = self.db()
        try:
            stored = db.query(ProviderConnection).one()
            self.assertNotIn("sk-test-secret-key-123", stored.credential_ciphertext)
            self.assertEqual(
                decrypt_provider_secret("user-one", "openai", stored.credential_ciphertext),
                "sk-test-secret-key-123",
            )
        finally:
            db.close()

    def test_invalid_api_key_response_never_echoes_submitted_secret(self):
        secret = "x-secret"
        response = self.client.post(
            "/providers/openai/api-key",
            headers={"Authorization": "Bearer user-one"},
            json={"api_key": secret},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(secret, response.text)

    def test_connections_are_owner_scoped(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-owner",
            provider="claude",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-owner", "claude", "secret"),
            status="authorized",
            available_models=["claude-test"],
        ))
        db.commit()
        db.close()

        response = self.client.get(
            "/providers/claude/models",
            headers={"Authorization": "Bearer user-other"},
        )
        self.assertEqual(response.status_code, 404)

    def test_model_selection_makes_a_live_check_and_does_not_mark_failed_model_ready(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-one",
            provider="openai",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-one", "openai", "sk-key"),
            status="authorized",
            available_models=["gpt-exact-model"],
        ))
        db.commit()
        db.close()

        with patch("provider_connections._execute_model", new=AsyncMock(side_effect=HTTPException(502, "Provider failed."))) as execute:
            response = self.client.put(
                "/providers/openai/model",
                headers={"Authorization": "Bearer user-one"},
                json={"model": "gpt-exact-model"},
            )
        self.assertEqual(response.status_code, 502)
        execute.assert_awaited_once_with(
            "openai", "sk-key", "gpt-exact-model", [{"role": "user", "content": "Reply with OK."}]
        )
        db = self.db()
        try:
            connection = db.query(ProviderConnection).one()
            self.assertEqual(connection.status, "authorized")
            self.assertIsNone(connection.selected_model)
        finally:
            db.close()

    def test_selected_model_is_the_only_model_used_for_generation(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-one",
            provider="claude",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-one", "claude", "claude-key"),
            status="authorized",
            available_models=["claude-exact"],
        ))
        db.add(ProviderConnection(
            user_id="user-one",
            provider="openai",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-one", "openai", "sk-key"),
            status="ready",
            selected_model="gpt-prior",
            is_active=True,
            available_models=["gpt-prior"],
        ))
        db.commit()
        db.close()
        with patch("provider_connections._execute_model", new=AsyncMock(return_value="OK")):
            selected = self.client.put(
                "/providers/claude/model",
                headers={"Authorization": "Bearer user-one"},
                json={"model": "claude-exact"},
            )
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.json()["status"], "ready")
        self.assertEqual(selected.json()["selected_model"], "claude-exact")
        self.assertTrue(selected.json()["is_active"])
        db = self.db()
        try:
            openai_connection = db.query(ProviderConnection).filter_by(
                user_id="user-one", provider="openai"
            ).one()
            self.assertFalse(openai_connection.is_active)
        finally:
            db.close()
        with patch("provider_connections._execute_model", new=AsyncMock(return_value="Response")) as execute:
            generated = self.client.post(
                "/providers/claude/generate",
                headers={"Authorization": "Bearer user-one"},
                json={
                    "model": "claude-exact",
                    "messages": [{"role": "user", "content": "Make a concise summary."}],
                    "language_preference": "hindi",
                },
            )
        self.assertEqual(generated.status_code, 200)
        self.assertEqual(generated.json()["text"], "Response")
        execute.assert_awaited_once_with(
            "claude",
            "claude-key",
            "claude-exact",
            [{"role": "user", "content": "Make a concise summary."}],
            "hindi",
        )

    def test_unlisted_model_is_rejected_without_provider_fallback(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-one",
            provider="openai",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-one", "openai", "sk-key"),
            status="authorized",
            available_models=["gpt-listed"],
        ))
        db.commit()
        db.close()
        response = self.client.put(
            "/providers/openai/model",
            headers={"Authorization": "Bearer user-one"},
            json={"model": "not-returned-by-provider"},
        )
        self.assertEqual(response.status_code, 422)

    def test_generate_requires_the_owner_and_selected_verified_model(self):
        response = self.client.post(
            "/providers/openai/generate",
            headers={"Authorization": "Bearer user-one"},
            json={"model": "gpt-exact-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status_code, 404)

    def test_generate_requires_an_explicit_model(self):
        response = self.client.post(
            "/providers/openai/generate",
            headers={"Authorization": "Bearer user-one"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(response.status_code, 422)

    def test_generate_rejects_a_model_that_is_no_longer_the_active_selection(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-one",
            provider="openai",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-one", "openai", "sk-key"),
            status="ready",
            selected_model="gpt-saved",
            is_active=False,
            available_models=["gpt-saved"],
        ))
        db.commit()
        db.close()
        response = self.client.post(
            "/providers/openai/generate",
            headers={"Authorization": "Bearer user-one"},
            json={
                "model": "gpt-saved",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        self.assertEqual(response.status_code, 409)

    def test_api_key_disconnect_is_local_and_warns_that_provider_key_remains_active(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-one",
            provider="openai",
            auth_method="api_key",
            credential_ciphertext=encrypt_provider_secret("user-one", "openai", "sk-key"),
            status="ready",
            selected_model="gpt-test",
            available_models=["gpt-test"],
        ))
        db.commit()
        db.close()
        response = self.client.delete(
            "/providers/openai/connection",
            headers={"Authorization": "Bearer user-one"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider_revocation"], "local_only")
        db = self.db()
        try:
            self.assertEqual(db.query(ProviderConnection).count(), 0)
        finally:
            db.close()

    def test_oauth_disconnect_attempts_revocation_then_deletes_local_secret(self):
        db = self.db()
        db.add(ProviderConnection(
            user_id="user-one",
            provider="gemini",
            auth_method="oauth",
            credential_ciphertext=encrypt_provider_secret(
                "user-one",
                "gemini",
                json.dumps({"access_token": "access-secret", "refresh_token": "refresh-secret"}),
            ),
            status="ready",
            selected_model="gemini-test",
            available_models=["gemini-test"],
        ))
        db.commit()
        db.close()

        class FakeOAuthClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, _type, _value, _traceback):
                return False

            async def post(self, url, data):
                self.url = url
                self.data = data
                return httpx.Response(200)

        fake_client = FakeOAuthClient()
        with patch("provider_connections.httpx.AsyncClient", return_value=fake_client):
            response = self.client.delete(
                "/providers/gemini/connection",
                headers={"Authorization": "Bearer user-one"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provider_revocation"], "revoked")
        self.assertEqual(fake_client.data, {"token": "refresh-secret"})
        db = self.db()
        try:
            self.assertEqual(db.query(ProviderConnection).count(), 0)
        finally:
            db.close()

    def test_expired_oauth_token_refresh_rotates_credentials_without_exposing_them(self):
        db = self.db()
        connection = ProviderConnection(
            user_id="user-one",
            provider="gemini",
            auth_method="oauth",
            credential_ciphertext=encrypt_provider_secret(
                "user-one",
                "gemini",
                json.dumps({"access_token": "old-access", "refresh_token": "old-refresh"}),
            ),
            status="ready",
            available_models=["gemini-test"],
            selected_model="gemini-test",
            token_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        db.add(connection)
        db.commit()

        class FakeOAuthClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, _type, _value, _traceback):
                return False

            async def post(self, url, data):
                self.url = url
                self.data = data
                return httpx.Response(200, json={
                    "access_token": "new-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 1800,
                })

        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "client-id"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "client-secret"
        fake_client = FakeOAuthClient()
        with patch("provider_connections.httpx.AsyncClient", return_value=fake_client):
            credential = asyncio.run(
                provider_connections._connection_credential(db, connection, "user-one")
            )
        self.assertEqual(credential["access_token"], "new-access")
        self.assertEqual(credential["refresh_token"], "rotated-refresh")
        persisted = db.query(ProviderConnection).one()
        self.assertNotIn("new-access", persisted.credential_ciphertext)
        self.assertEqual(
            json.loads(decrypt_provider_secret("user-one", "gemini", persisted.credential_ciphertext))["refresh_token"],
            "rotated-refresh",
        )
        db.close()

    def test_gemini_oauth_start_uses_pkce_and_persists_only_a_state_hash(self):
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "test-client-id"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "test-client-secret"
        os.environ["GOOGLE_OAUTH_REDIRECT_URI"] = "https://api.example.test/providers/gemini/oauth/callback"
        response = self.client.post(
            "/providers/gemini/oauth/start",
            headers={"Authorization": "Bearer user-oauth"},
            json={"project_id": "user-project-123"},
        )
        self.assertEqual(response.status_code, 200)
        params = parse_qs(urlparse(response.json()["authorization_url"]).query)
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(params["scope"], [provider_connections.PROVIDER_SCOPES[0]])
        self.assertRegex(params["code_challenge"][0], re.compile(r"^[A-Za-z0-9_-]+$"))
        state = params["state"][0]
        db = self.db()
        try:
            pending = db.query(ProviderOAuthState).one()
            self.assertEqual(pending.state_hash, hashlib.sha256(state.encode()).hexdigest())
            self.assertNotEqual(pending.state_hash, state)
            self.assertEqual(pending.user_id, "user-oauth")
            self.assertEqual(pending.quota_project_id, "user-project-123")
        finally:
            db.close()

    def test_invalid_oauth_callback_fails_without_exposing_provider_details(self):
        os.environ["FRONTEND_URL"] = "https://app.example.test"
        response = self.client.get(
            "/providers/gemini/oauth/callback?state=invalid&error=access_denied"
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "https://app.example.test/?provider_connection=failed")

    def test_oauth_callback_exchanges_code_and_saves_verified_models_without_returning_tokens(self):
        os.environ["GOOGLE_OAUTH_CLIENT_ID"] = "client-id"
        os.environ["GOOGLE_OAUTH_CLIENT_SECRET"] = "client-secret"
        os.environ["GOOGLE_OAUTH_REDIRECT_URI"] = "https://api.example.test/providers/gemini/oauth/callback"
        os.environ["FRONTEND_URL"] = "https://app.example.test/"
        state = "one-time-state"
        db = self.db()
        db.add(ProviderOAuthState(
            state_hash=hashlib.sha256(state.encode()).hexdigest(),
            user_id="user-oauth",
            quota_project_id="user-project-123",
            verifier_ciphertext=encrypt_provider_secret("user-oauth", "gemini", "pkce-verifier"),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        ))
        db.commit()
        db.close()

        class FakeOAuthClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, _type, _value, _traceback):
                return False

            async def post(self, url, data):
                self.url = url
                self.data = data
                return httpx.Response(200, json={
                    "access_token": "oauth-access-secret",
                    "refresh_token": "oauth-refresh-secret",
                    "token_type": "Bearer",
                    "scope": provider_connections.PROVIDER_SCOPES[0],
                    "expires_in": 3600,
                })

        fake_client = FakeOAuthClient()
        with (
            patch("provider_connections.httpx.AsyncClient", return_value=fake_client),
            patch("provider_connections._list_models", new=AsyncMock(return_value=["gemini-live-model"])),
        ):
            response = self.client.get(
                f"/providers/gemini/oauth/callback?state={state}&code=authorization-code-secret"
            )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            "https://app.example.test/?provider_connection=authorized",
        )
        self.assertNotIn("authorization-code-secret", response.text)
        db = self.db()
        try:
            self.assertEqual(db.query(ProviderOAuthState).count(), 0)
            connection = db.query(ProviderConnection).one()
            self.assertEqual(connection.status, "authorized")
            self.assertNotIn("oauth-access-secret", connection.credential_ciphertext)
            stored = json.loads(decrypt_provider_secret(
                "user-oauth", "gemini", connection.credential_ciphertext
            ))
            self.assertEqual(stored["refresh_token"], "oauth-refresh-secret")
            self.assertEqual(stored["quota_project_id"], "user-project-123")
        finally:
            db.close()

    def test_expired_oauth_state_is_rejected_and_deleted(self):
        state = "expired-random-state"
        db = self.db()
        db.add(ProviderOAuthState(
            state_hash=hashlib.sha256(state.encode()).hexdigest(),
            user_id="user-oauth",
            quota_project_id="user-project-123",
            verifier_ciphertext=encrypt_provider_secret("user-oauth", "gemini", "verifier"),
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        ))
        db.commit()
        db.close()
        os.environ["FRONTEND_URL"] = "https://app.example.test"
        response = self.client.get(f"/providers/gemini/oauth/callback?state={state}&code=unused")
        self.assertEqual(response.status_code, 303)
        db = self.db()
        try:
            self.assertEqual(db.query(ProviderOAuthState).count(), 0)
        finally:
            db.close()

    def test_invalid_supabase_session_is_rejected(self):
        response = self.client.get("/providers/connections")
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
