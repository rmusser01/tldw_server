import os
import json
import tempfile
import pytest
from typing import Any, Optional, Tuple
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from tldw_Server_API.app.api.v1.endpoints.chat import router as chat_router
from tldw_Server_API.app.api.v1.endpoints.health import router as health_router
from tldw_Server_API.app.core.Chat import chat_service as chat_service_mod
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, DEFAULT_CHARACTER_NAME


app = FastAPI()
app.include_router(health_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1/chat")


def _make_test_db():


    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client")
    db.add_character_card({
        "name": DEFAULT_CHARACTER_NAME,
        "description": "Default",
        "personality": "Helpful",
        "scenario": "Testing",
        "system_prompt": "You are helpful",
        "first_message": "Hello",
        "creator_notes": "test"
    })
    return db, db_path


def _add_non_default_character(db: CharactersRAGDB, name: str = "Guardian Character") -> dict[str, Any]:
    character_id = db.add_character_card({
        "name": name,
        "description": "Non-default test character",
        "personality": "Guarded",
        "scenario": "Testing",
        "system_prompt": "You are in character",
        "first_message": "Hello from character",
        "creator_notes": "test"
    })
    character = db.get_character_card_by_id(int(character_id))
    assert character is not None
    return character


def _post_with_csrf(client: TestClient, url: str, **kwargs):
    headers = kwargs.pop("headers", {}) or {}
    csrf = getattr(client, "csrf_token", "")
    return client.post(url, headers={"X-CSRF-Token": csrf, **headers}, **kwargs)


def _auth_headers(client):


    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    settings = get_settings()
    api_key = settings.SINGLE_USER_API_KEY or os.getenv("API_BEARER", "test-api-key-12345")
    return {"X-API-KEY": api_key, "X-CSRF-Token": getattr(client, 'csrf_token', '')}


class _Policy:
    def __init__(self, categories_enabled: Optional[set] = None):
        self.enabled = True
        self.input_enabled = True
        self.output_enabled = True
        self.input_action = 'block'
        self.output_action = 'redact'
        self.redact_replacement = '[REDACTED]'
        self.block_patterns = []
        self.categories_enabled = categories_enabled


class _Svc:
    def __init__(self, policy: _Policy):
        self._p = policy

    def get_effective_policy(self, user_id: str):
        return self._p

    def evaluate_action(self, text: str, policy: _Policy, phase: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        # Simulate a built-in PII email rule tagged {'pii', 'pii_email'}
        if 'user@example.com' in text:
            cats = policy.categories_enabled
            if not cats or ('pii' in cats or 'pii_email' in cats):
                # redact email
                return 'redact', text.replace('user@example.com', '[PII]'), 'email', 'pii_email'
            return 'pass', None, None, None
        return 'pass', None, None, None

    def redact_text(self, text: str, policy: _Policy) -> str:
        if 'user@example.com' in text:
            cats = policy.categories_enabled
            if not cats or ('pii' in cats or 'pii_email' in cats):
                return text.replace('user@example.com', '[PII]')
        return text

    def check_text(self, text: str, policy: _Policy, phase: str | None = None):
        if 'user@example.com' in text:
            cats = policy.categories_enabled
            if not cats or ('pii' in cats or 'pii_email' in cats):
                return True, "email"
        return False, None


@pytest.mark.unit
def test_categories_allow_pii_redaction():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _Policy(categories_enabled={'pii'})
        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_Svc(policy)):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "the email is user@example.com"}],
                    "stream": False
                }
                # Mock provider to echo back user content
                with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "the email is user@example.com"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }):
                    r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                    assert r.status_code == 200
                    data = r.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    assert "[PII]" in content
                    assert "user@example.com" not in content
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"): os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"): os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_categories_disable_pii_redaction():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _Policy(categories_enabled={'confidential'})  # no 'pii'
        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_Svc(policy)):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "the email is user@example.com"}],
                    "stream": False
                }
                with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "the email is user@example.com"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }):
                    r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                    assert r.status_code == 200
                    data = r.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    # No redaction expected
                    assert "user@example.com" in content
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"): os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"): os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_output_guardian_proxy_receives_character_chat_type_for_continued_character_conversation():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        character = _add_non_default_character(db, name="Guardian Character Output")
        conversation_id = db.add_conversation(
            {
                "character_id": character["id"],
                "title": "Guardian Output Character Chat",
                "client_id": "test_client",
            }
        )
        assert conversation_id

        policy = _Policy(categories_enabled={"pii"})
        proxy_chat_types: list[str | None] = []

        class _SpyGuardianModerationProxy:
            def __init__(self, base, engine, dependent_user_id, *, chat_type=None):
                proxy_chat_types.append(chat_type)
                self._base = base

            def get_effective_policy(self, user_id: str | None = None):
                return self._base.get_effective_policy(user_id)

            def __getattr__(self, name: str):
                return getattr(self._base, name)

        def _fake_bootstrap_guardian_moderation_runtime(*, user_id, dependent_user_id, chat_type):
            return SimpleNamespace(
                dependent_user_id=dependent_user_id,
                chat_type=chat_type,
                guardian_db=object(),
                supervised_engine=object(),
            )

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_Svc(policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.bootstrap_guardian_moderation_runtime", side_effect=_fake_bootstrap_guardian_moderation_runtime, create=True),
            patch("tldw_Server_API.app.core.Moderation.supervised_policy.GuardianModerationProxy", _SpyGuardianModerationProxy),
            patch("tldw_Server_API.app.core.feature_flags.is_guardian_enabled", return_value=True),
            patch("tldw_Server_API.app.core.feature_flags.is_self_monitoring_enabled", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value={
                    "id": "chatcmpl-guard-output-1",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "the email is user@example.com"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "conversation_id": conversation_id,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
                response = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))

        assert response.status_code == 200
        assert proxy_chat_types == ["character"]
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        assert "[PII]" in content
        assert "user@example.com" not in content
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"): os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"): os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_output_guardian_proxy_receives_regular_chat_type_for_resumed_default_assistant_conversation():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        default_character = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        assert default_character is not None
        conversation_id = db.add_conversation(
            {
                "character_id": default_character["id"],
                "title": "Guardian Output Ordinary Chat",
                "client_id": "test_client",
            }
        )
        assert conversation_id

        policy = _Policy(categories_enabled={"pii"})
        proxy_chat_types: list[str | None] = []

        class _SpyGuardianModerationProxy:
            def __init__(self, base, engine, dependent_user_id, *, chat_type=None):
                proxy_chat_types.append(chat_type)
                self._base = base

            def get_effective_policy(self, user_id: str | None = None):
                return self._base.get_effective_policy(user_id)

            def __getattr__(self, name: str):
                return getattr(self._base, name)

        def _fake_bootstrap_guardian_moderation_runtime(*, user_id, dependent_user_id, chat_type):
            return SimpleNamespace(
                dependent_user_id=dependent_user_id,
                chat_type=chat_type,
                guardian_db=object(),
                supervised_engine=object(),
            )

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_Svc(policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.bootstrap_guardian_moderation_runtime", side_effect=_fake_bootstrap_guardian_moderation_runtime, create=True),
            patch("tldw_Server_API.app.core.Moderation.supervised_policy.GuardianModerationProxy", _SpyGuardianModerationProxy),
            patch("tldw_Server_API.app.core.feature_flags.is_guardian_enabled", return_value=True),
            patch("tldw_Server_API.app.core.feature_flags.is_self_monitoring_enabled", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value={
                    "id": "chatcmpl-guard-output-ordinary",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "the email is user@example.com"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "conversation_id": conversation_id,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
                response = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))

        assert response.status_code == 200
        assert proxy_chat_types == ["regular"]
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        assert "[PII]" in content
        assert "user@example.com" not in content
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"): os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"): os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_resolve_moderation_chat_type_treats_default_character_request_as_regular():
    request_data = SimpleNamespace(character_id="123")

    assert chat_service_mod.resolve_moderation_chat_type(
        request_data=request_data,
        assistant_context=None,
        default_character_id=123,
    ) == "regular"
