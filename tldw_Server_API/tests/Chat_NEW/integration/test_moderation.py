import os
import json
import re
import tempfile
import pytest
from typing import Any
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from tldw_Server_API.app.api.v1.endpoints.chat import router as chat_router
from tldw_Server_API.app.api.v1.endpoints.health import router as health_router
from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError


# Minimal app with only health and chat routers to avoid unrelated imports
app = FastAPI()
app.include_router(health_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1/chat")


def _assert_mandatory_audit_detail(detail: dict[str, Any]) -> None:
    assert detail == {
        "error_code": "mandatory_audit_unavailable",
        "message": "Mandatory audit persistence is currently unavailable",
    }


def _cleanup_db_artifacts(db_path: str) -> None:
    os.unlink(db_path)
    if os.path.exists(db_path + "-wal"):
        os.unlink(db_path + "-wal")
    if os.path.exists(db_path + "-shm"):
        os.unlink(db_path + "-shm")

def _make_test_db():

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    db = CharactersRAGDB(db_path, "test_client")
    # Minimal default character required by endpoint
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


class _StubPolicy:
    def __init__(self, enabled=True, input_action='block', output_action='redact', redact='[REDACTED]', patterns=None):
        self.enabled = enabled
        self.input_enabled = True
        self.output_enabled = True
        self.input_action = input_action
        self.output_action = output_action
        self.redact_replacement = redact
        self.block_patterns = patterns or [re.compile(r"secret", re.IGNORECASE)]


class _StubModerationService:
    def __init__(self, policy: _StubPolicy):
        self._policy = policy

    def get_effective_policy(self, user_id: str):
        return self._policy

    def check_text(self, text: str, policy: _StubPolicy, phase: str | None = None):
        for pat in policy.block_patterns:
            if pat.search(text or ""):
                return True, pat.pattern
        return False, None

    def redact_text(self, text: str, policy: _StubPolicy):
        red = text
        for pat in policy.block_patterns:
            red = pat.sub(policy.redact_replacement, red)
        return red


class _EvalModerationService:
    def __init__(self, policy: _StubPolicy):
        self._policy = policy

    def get_effective_policy(self, user_id: str):
        return self._policy

    def evaluate_action(self, text: str, policy: _StubPolicy, phase: str):
        for pat in policy.block_patterns:
            if pat.search(text or ""):
                red = pat.sub(policy.redact_replacement, text)
                return "redact", red, pat.pattern, None
        return "pass", None, None, None

    def evaluate_action_with_match(self, text: str, policy: _StubPolicy, phase: str):
        for pat in policy.block_patterns:
            m = pat.search(text or "")
            if m:
                red = pat.sub(policy.redact_replacement, text)
                return "redact", red, pat.pattern, None, m.span()
        return "pass", None, None, None, None

    def redact_text(self, text: str, policy: _StubPolicy):
        red = text
        for pat in policy.block_patterns:
            red = pat.sub(policy.redact_replacement, red)
        return red

    def check_text(self, text: str, policy: _StubPolicy, phase: str | None = None):
        for pat in policy.block_patterns:
            if pat.search(text or ""):
                return True, pat.pattern
        return False, None


class _CharacterScopedGuardianEngine:
    def __init__(self):
        self.overlay_chat_types = []
        self.check_calls = []

    def build_moderation_policy_overlay(self, dependent_user_id: str, base_policy: _StubPolicy, chat_type: str | None = None):
        self.overlay_chat_types.append(chat_type)
        if chat_type == "character":
            return _StubPolicy(
                enabled=True,
                input_action="block",
                output_action=base_policy.output_action,
                redact=base_policy.redact_replacement,
                patterns=[re.compile(r"secret", re.IGNORECASE)],
            )
        return base_policy

    def check_text(self, text: str, dependent_user_id: str, phase: str, chat_type: str | None = None):
        self.check_calls.append((phase, chat_type, text))
        is_character_secret = chat_type == "character" and "secret" in (text or "").lower()
        return SimpleNamespace(
            action="block" if is_character_secret else "pass",
            notify_guardian=False,
            rule_name_visible="guardian-secret" if is_character_secret else None,
        )


class _FailingMandatoryAuditService:
    def __init__(self):
        self.logged_events: list[dict[str, Any]] = []

    async def log_event(self, **kwargs):
        self.logged_events.append(kwargs)

    async def flush(self, raise_on_failure: bool = False):
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")


async def _failing_audit_service_override():
    return _FailingMandatoryAuditService()


async def _missing_audit_service_override():
    return None


@pytest.mark.unit
def test_input_block_returns_400(monkeypatch):
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _StubPolicy(enabled=True, input_action='block', output_action='redact')

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "please say secret"}],
                    "stream": False
                }
                # No need to patch provider; should block before provider call
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 400
                assert "violates" in r.json().get("detail", "").lower()
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_input_block_fails_closed_when_mandatory_audit_fails():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_audit_service_for_user] = _failing_audit_service_override
        policy = _StubPolicy(enabled=True, input_action='block', output_action='redact')

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_provider:
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "please say secret"}],
                    "stream": False,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 503
                _assert_mandatory_audit_detail(r.json().get("detail", {}))
                assert not mock_provider.called
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


@pytest.mark.unit
def test_input_block_fails_closed_when_audit_service_missing():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_audit_service_for_user] = _missing_audit_service_override
        policy = _StubPolicy(enabled=True, input_action='block', output_action='redact')

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_provider:
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "please say secret"}],
                    "stream": False,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 503
                _assert_mandatory_audit_detail(r.json().get("detail", {}))
                assert not mock_provider.called
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


@pytest.mark.unit
def test_output_redaction_non_streaming(monkeypatch):
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='redact', redact='[REDACTED]')

        # Mock provider to return a fixed non-streaming response
        reply = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "this has secret token"}, "finish_reason": "stop"}
            ]
        }

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=reply):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                data = r.json()
                got = data.get("choices", [{}])[0].get("message", {}).get("content")
                assert got == "this has [REDACTED] token"
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_output_redaction_non_streaming_fails_closed_when_mandatory_audit_fails():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_audit_service_for_user] = _failing_audit_service_override
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='redact', redact='[REDACTED]')

        reply = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "this has secret token"}, "finish_reason": "stop"}
            ]
        }

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=reply):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 503
                _assert_mandatory_audit_detail(r.json().get("detail", {}))
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


@pytest.mark.unit
def test_output_block_non_streaming_fails_closed_when_mandatory_audit_fails():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_audit_service_for_user] = _failing_audit_service_override
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='block')

        reply = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "this has secret token"}, "finish_reason": "stop"}
            ]
        }

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=reply):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 503
                _assert_mandatory_audit_detail(r.json().get("detail", {}))
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


@pytest.mark.unit
def test_streaming_redaction_applied():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='redact', redact='[REDACTED]')

        def upstream_stream():

            chunk1 = {"choices": [{"delta": {"content": "leak: secret"}}]}
            chunk2 = {"choices": [{"delta": {"content": " appears"}}]}
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service",
                  return_value=_StubModerationService(policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                  return_value=upstream_stream()),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                content = r.text.splitlines()
                chunks = []
                for line in content:
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("choices"):
                        delta = obj["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            chunks.append(text)
                full = "".join(chunks)
                assert "secret" not in full
                assert "[REDACTED]" in full
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_streaming_cross_chunk_redaction_output(monkeypatch):
    db, db_path = _make_test_db()
    try:
        monkeypatch.setenv("STREAMS_UNIFIED", "0")
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _StubPolicy(
            enabled=True,
            input_action='warn',
            output_action='redact',
            redact='[REDACTED]',
            patterns=[re.compile(r"XQZP1234", re.IGNORECASE)],
        )

        def upstream_stream():
            chunk1 = {"choices": [{"delta": {"content": "XQZP"}}]}
            chunk2 = {"choices": [{"delta": {"content": "1234 data"}}]}
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service",
                  return_value=_EvalModerationService(policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                  return_value=upstream_stream()),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                content = r.text.splitlines()
                chunks = []
                for line in content:
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("choices"):
                        delta = obj["choices"][0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            chunks.append(text)
                full = "".join(chunks)
                assert "xqzp1234" not in full.lower()
                assert "xqzp" not in full.lower()
                assert "[REDACTED]" in full
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_streaming_block_emits_sse_error_and_finishes():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        # Block on output
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='block', redact='[REDACTED]')

        def upstream_stream():

            chunk1 = {"choices": [{"delta": {"content": "this contains secret"}}]}
            chunk2 = {"choices": [{"delta": {"content": " should be blocked"}}]}
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                lines = r.text.splitlines()
                saw_error = any((ln.startswith("data:") and '"error":' in ln) for ln in lines)
                saw_done = any((ln.strip() == 'data: [DONE]') for ln in lines)
                assert saw_error, f"Expected SSE error in stream, got: {r.text[:200]}"
                assert saw_done, "Expected [DONE] marker for graceful finish"
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_streaming_block_emits_audit_failure_when_mandatory_audit_fails(monkeypatch):
    db, db_path = _make_test_db()
    try:
        monkeypatch.setenv("STREAMS_UNIFIED", "0")
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_audit_service_for_user] = _failing_audit_service_override
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='block', redact='[REDACTED]')

        def upstream_stream():
            chunk1 = {"choices": [{"delta": {"content": "this contains secret"}}]}
            chunk2 = {"choices": [{"delta": {"content": " should be blocked"}}]}
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                assert "audit_persistence_failure" in r.text
                assert "[DONE]" in r.text
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


@pytest.mark.unit
def test_streaming_redaction_emits_audit_failure_and_fails_closed(monkeypatch):
    db, db_path = _make_test_db()
    try:
        monkeypatch.setenv("STREAMS_UNIFIED", "0")
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_audit_service_for_user] = _failing_audit_service_override
        policy = _StubPolicy(enabled=True, input_action='warn', output_action='redact', redact='[REDACTED]')

        def upstream_stream():
            chunk1 = {"choices": [{"delta": {"content": "leak: secret"}}]}
            chunk2 = {"choices": [{"delta": {"content": " appears"}}]}
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service",
                  return_value=_StubModerationService(policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                  return_value=upstream_stream()),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
                assert r.status_code == 200
                text = r.text
                assert "[REDACTED]" in text
                assert "audit_persistence_failure" in text
                assert "[DONE]" in text
                stream_end_payload = None
                for block in text.split("\n\n"):
                    if not block.startswith("event: stream_end"):
                        continue
                    data_line = next((line for line in block.splitlines() if line.startswith("data: ")), None)
                    assert data_line is not None
                    stream_end_payload = json.loads(data_line[6:])
                    break
                assert stream_end_payload is not None, text
                assert stream_end_payload["success"] is False
    finally:
        try:
            _cleanup_db_artifacts(db_path)
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


@pytest.mark.unit
def test_streaming_cross_chunk_redaction_persisted(monkeypatch):
    db, db_path = _make_test_db()
    try:
        monkeypatch.setenv("STREAMS_UNIFIED", "0")
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        policy = _StubPolicy(
            enabled=True,
            input_action='warn',
            output_action='redact',
            redact='[REDACTED]',
            patterns=[re.compile(r"XQZP1234", re.IGNORECASE)],
        )

        def upstream_stream():

            chunk1 = {"choices": [{"delta": {"content": "XQZP"}}]}
            chunk2 = {"choices": [{"delta": {"content": "1234 data"}}]}
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        with patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_EvalModerationService(policy)), \
             patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "save_to_db": True,
                }
                headers = _auth_headers(client)
                with client.stream("POST", "/api/v1/chat/completions", json=body, headers=headers) as r:
                    assert r.status_code == 200
                    for line in r.iter_text():
                        if "[DONE]" in line:
                            break

        convs = db.get_conversations_for_user("test_client", limit=1)
        assert convs, "Expected a persisted conversation"
        conv_id = convs[0]["id"]
        msgs = db.get_messages_for_conversation(conv_id, order_by_timestamp="ASC")
        assert msgs, "Expected persisted messages"
        saved = msgs[-1].get("content", "")
        assert "[REDACTED]" in saved
        assert "xqzp1234" not in saved.lower()
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_character_chat_input_guardian_overlay_uses_character_chat_type():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        character = _add_non_default_character(db, name="Guardian Character Input")

        base_policy = _StubPolicy(
            enabled=True,
            input_action="warn",
            output_action="redact",
            patterns=[re.compile(r"never-match-this", re.IGNORECASE)],
        )
        engine = _CharacterScopedGuardianEngine()
        bootstrap_calls: list[tuple[object, str, str]] = []

        def _fake_bootstrap_guardian_moderation_runtime(*, user_id, dependent_user_id, chat_type):
            bootstrap_calls.append((user_id, dependent_user_id, chat_type))
            return SimpleNamespace(
                dependent_user_id=dependent_user_id,
                chat_type=chat_type,
                guardian_db=object(),
                supervised_engine=engine,
            )

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(base_policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.bootstrap_guardian_moderation_runtime", side_effect=_fake_bootstrap_guardian_moderation_runtime, create=True),
            patch("tldw_Server_API.app.core.feature_flags.is_guardian_enabled", return_value=True),
            patch("tldw_Server_API.app.core.feature_flags.is_self_monitoring_enabled", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value={
                    "id": "chatcmpl-guard-input-1",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "safe"}, "finish_reason": "stop"}],
                },
            ),
        ):
            with TestClient(app) as client:
                resp = client.get("/api/v1/health")
                client.csrf_token = resp.cookies.get("csrf_token", "")
                body = {
                    "api_provider": "openai",
                    "model": "gpt-4o-mini",
                    "character_id": str(character["id"]),
                    "messages": [{"role": "user", "content": "please say secret"}],
                    "stream": False,
                }
                response = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))

        assert response.status_code == 400
        assert bootstrap_calls and bootstrap_calls[0][2] == "character"
        assert engine.overlay_chat_types == ["character"]
        assert engine.check_calls == [("input", "character", "please say secret")]
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_continued_character_conversation_input_guardian_overlay_uses_saved_conversation_chat_type():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        character = _add_non_default_character(db, name="Guardian Character Continued")
        conversation_id = db.add_conversation(
            {
                "character_id": character["id"],
                "title": "Guardian Continued Character Chat",
                "client_id": "test_client",
            }
        )
        assert conversation_id

        base_policy = _StubPolicy(
            enabled=True,
            input_action="warn",
            output_action="redact",
            patterns=[re.compile(r"never-match-this", re.IGNORECASE)],
        )
        engine = _CharacterScopedGuardianEngine()
        bootstrap_calls: list[tuple[object, str, str]] = []

        def _fake_bootstrap_guardian_moderation_runtime(*, user_id, dependent_user_id, chat_type):
            bootstrap_calls.append((user_id, dependent_user_id, chat_type))
            return SimpleNamespace(
                dependent_user_id=dependent_user_id,
                chat_type=chat_type,
                guardian_db=object(),
                supervised_engine=engine,
            )

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(base_policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.bootstrap_guardian_moderation_runtime", side_effect=_fake_bootstrap_guardian_moderation_runtime, create=True),
            patch("tldw_Server_API.app.core.feature_flags.is_guardian_enabled", return_value=True),
            patch("tldw_Server_API.app.core.feature_flags.is_self_monitoring_enabled", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value={
                    "id": "chatcmpl-guard-input-2",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "safe"}, "finish_reason": "stop"}],
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
                    "messages": [{"role": "user", "content": "please say secret"}],
                    "stream": False,
                }
                response = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))

        assert response.status_code == 400
        assert bootstrap_calls and bootstrap_calls[0][2] == "character"
        assert engine.overlay_chat_types == ["character"]
        assert engine.check_calls == [("input", "character", "please say secret")]
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
def test_continued_ordinary_conversation_input_guardian_overlay_stays_regular_for_default_assistant():
    db, db_path = _make_test_db()
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        default_character = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        assert default_character is not None
        conversation_id = db.add_conversation(
            {
                "character_id": default_character["id"],
                "title": "Guardian Continued Ordinary Chat",
                "client_id": "test_client",
            }
        )
        assert conversation_id

        base_policy = _StubPolicy(
            enabled=True,
            input_action="warn",
            output_action="redact",
            patterns=[re.compile(r"never-match-this", re.IGNORECASE)],
        )
        engine = _CharacterScopedGuardianEngine()
        bootstrap_calls: list[tuple[object, str, str]] = []

        def _fake_bootstrap_guardian_moderation_runtime(*, user_id, dependent_user_id, chat_type):
            bootstrap_calls.append((user_id, dependent_user_id, chat_type))
            return SimpleNamespace(
                dependent_user_id=dependent_user_id,
                chat_type=chat_type,
                guardian_db=object(),
                supervised_engine=engine,
            )

        with (
            patch("tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service", return_value=_StubModerationService(base_policy)),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.bootstrap_guardian_moderation_runtime", side_effect=_fake_bootstrap_guardian_moderation_runtime, create=True),
            patch("tldw_Server_API.app.core.feature_flags.is_guardian_enabled", return_value=True),
            patch("tldw_Server_API.app.core.feature_flags.is_self_monitoring_enabled", return_value=False),
            patch(
                "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
                return_value={
                    "id": "chatcmpl-guard-input-ordinary",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "safe"}, "finish_reason": "stop"}],
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
                    "messages": [{"role": "user", "content": "please say secret"}],
                    "stream": False,
                }
                response = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))

        assert response.status_code == 200
        assert bootstrap_calls and bootstrap_calls[0][2] == "regular"
        assert engine.overlay_chat_types
        assert set(engine.overlay_chat_types) == {"regular"}
        assert engine.check_calls == [("input", "regular", "please say secret")]
    finally:
        try:
            os.unlink(db_path)
            if os.path.exists(db_path + "-wal"):
                os.unlink(db_path + "-wal")
            if os.path.exists(db_path + "-shm"):
                os.unlink(db_path + "-shm")
        except Exception:
            _ = None
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
