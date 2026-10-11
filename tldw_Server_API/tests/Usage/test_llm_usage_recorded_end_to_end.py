"""A call through the shared layer writes one real llm_usage_log row for the scope user (spec §1 to §3)."""

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
from tldw_Server_API.tests.provider_credential_test_helpers import resolved_request_fields_async
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_cache_columns, _ensure_llm_tables

pytestmark = pytest.mark.integration


class _E2EProvider(ChatProvider):
    """A sync-only provider, like most production adapters."""

    name = "e2e-fake"

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


async def test_one_row_is_written_for_the_scope_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG-style usage: scope set, no explicit context, call made through perform_chat_api_call_async."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///./Databases/users_test_e2e_{uuid.uuid4().hex}.sqlite")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)
    await _ensure_llm_cache_columns(pool)

    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("e2e-fake", _E2EProvider)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())

    fields = await resolved_request_fields_async("e2e-fake", api_key="k", app_config={}, model="e2e-model")
    with scoped_context(user_id=41):
        result = await chat_service.perform_chat_api_call_async(
            api_endpoint="e2e-fake", **fields, messages_payload=[{"role": "user", "content": "hi"}],
            model="e2e-model", streaming=False,
        )
    assert result["choices"][0]["message"]["content"] == "ok"
    await llm_accounting.drain_accounting()

    rows = await pool.fetchall(
        "SELECT user_id, operation, endpoint, provider, model, prompt_tokens, completion_tokens "
        "FROM llm_usage_log WHERE provider = ?",
        "e2e-fake",
    )
    assert len(rows) == 1
    row = dict(rows[0]) if not isinstance(rows[0], dict) else rows[0]
    assert (row["user_id"], row["operation"], row["endpoint"]) == (41, "llm_call", "internal")
    assert (row["model"], row["prompt_tokens"], row["completion_tokens"]) == ("e2e-model", 30, 12)
