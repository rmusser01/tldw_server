"""A call handed to an executor thread, and a call under a job root, each write one real llm_usage_log row (spec §1)."""

import asyncio
import uuid
from collections.abc import Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.app.core.Usage.llm_roots import ensure_llm_state_executor, llm_call_root
from tldw_Server_API.tests.provider_credential_test_helpers import resolved_request_fields_async
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_cache_columns, _ensure_llm_tables

pytestmark = pytest.mark.integration


class _ExecProvider(ChatProvider):
    """A sync-only provider, like most production adapters."""

    name = "exec-fake"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion with usage."""
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Unused."""
        return iter(())


async def _fresh_pool(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A real SQLite AuthNZ pool with the usage tables, bound to this test's loop."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.sqlite'}")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.tests._plugins.app_services_fixtures import discard_app_services
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await discard_app_services()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)
    await _ensure_llm_cache_columns(pool)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    return pool


async def _rows(pool) -> list[dict[str, Any]]:
    rows = await pool.fetchall("SELECT user_id, operation FROM llm_usage_log WHERE provider = ?", "exec-fake")
    return [dict(r) if not isinstance(r, dict) else r for r in rows]


async def test_a_request_style_executor_hand_off_records_the_scope_user(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The scope is set in the request task; the provider call runs on the default executor; one row for user 41."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    provider = _ExecProvider()
    request = {"messages": [{"role": "user", "content": "hi"}], "model": "exec-model"}

    with scoped_context(user_id=41):
        ensure_llm_state_executor()
        result = await asyncio.get_running_loop().run_in_executor(None, provider.chat, request)
    assert result["choices"][0]["message"]["content"] == "ok"
    await llm_accounting.drain_accounting()

    rows = await _rows(pool)
    assert len(rows) == 1
    assert (rows[0]["user_id"], rows[0]["operation"]) == (41, "llm_call")


async def test_a_job_root_records_the_job_owner(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No scope; a job root for owner "42"; the shared-layer call writes one row for user 42."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("exec-fake", _ExecProvider)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)

    fields = await resolved_request_fields_async("exec-fake", api_key="k", app_config={}, model="exec-model")
    with llm_call_root("42"):
        result = await chat_service.perform_chat_api_call_async(
            api_endpoint="exec-fake", **fields, messages_payload=[{"role": "user", "content": "hi"}],
            model="exec-model", streaming=False,
        )
    assert result["choices"][0]["message"]["content"] == "ok"
    await llm_accounting.drain_accounting()

    rows = await _rows(pool)
    assert len(rows) == 1
    assert (rows[0]["user_id"], rows[0]["operation"]) == (42, "llm_call")
