"""With moderation off, every moderated surface makes zero execution requests and changes nothing
(TASK-13950.27 PR C, part a). The /chat/completions variant joins in part b, with the app wiring."""

from __future__ import annotations

import types

import pytest

import tldw_Server_API.app.core.Moderation.gateway_hooks as hooks_mod
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.Moderation.execution_runner import ModerationExecutionRunner
from tldw_Server_API.app.core.Moderation.surfaces import MODERATED_SURFACES
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.tests.Moderation.conftest import DONE, PROVIDER, finish_frame, openai_frame, reply

pytestmark = pytest.mark.unit

SURFACES = sorted(MODERATED_SURFACES)


@pytest.fixture
def execution_requests(monkeypatch):
    calls: list[str] = []
    execute = ModerationExecutionRunner.execute

    def observe(self, request, scope):
        calls.append(request.operation)
        return execute(self, request, scope)

    monkeypatch.setattr(ModerationExecutionRunner, "execute", observe)
    return calls


@pytest.fixture
def guardian_files(monkeypatch, tmp_path):
    """Guardian is on, the user DB root is tmp_path and the real resolver runs: the user has no DB, none may be created."""
    import tldw_Server_API.app.core.DB_Management.guardian_db_resolver as resolver
    import tldw_Server_API.app.core.feature_flags as flags

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(flags, "is_guardian_enabled", lambda: True)
    real = resolver.resolve_existing_guardian_db_for_user_id
    outcomes: list[str] = []

    def spy(user_id):
        try:
            return real(user_id)
        except FileNotFoundError:
            outcomes.append("missing")
            raise

    monkeypatch.setattr(resolver, "resolve_existing_guardian_db_for_user_id", spy)
    return types.SimpleNamespace(root=tmp_path, outcomes=outcomes)


def _call(surface: str, stream: bool):
    return gateway.perform_chat_api_call(
        api_endpoint=PROVIDER, model="m", api_key="k", streaming=stream, llm_surface=surface,
        messages_payload=[{"role": "user", "content": "forbidden secret"}],
    )


@pytest.mark.parametrize("surface_key", [True, None], ids=["key-on", "no-key"])
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("surface", SURFACES)
def test_global_off_makes_zero_execution_requests(
    make_service, stub_llm_registry, execution_requests, surface, stream, surface_key, guardian_files
):
    make_service(enabled=False, **({f"surface.{surface}": True} if surface_key else {}))
    hooks_mod.register_moderation_gateway_hooks()
    stub_llm_registry.reply = reply("forbidden secret")
    stub_llm_registry.items = [openai_frame("forbidden secret"), finish_frame(), DONE]
    with llm_call_context(user_id=7):
        result = _call(surface, stream)
    if stream:
        assert list(result) == stub_llm_registry.items
    else:
        assert result == reply("forbidden secret")
    assert stub_llm_registry.requests[0]["messages"][0]["content"] == "forbidden secret"
    assert execution_requests == []
    assert guardian_files.outcomes == ["missing"]  # the real resolver ran and found no DB
    assert [p for p in guardian_files.root.rglob("*") if p.suffix == ".db" or "guardian" in p.name.lower()] == []


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("surface", SURFACES)
def test_global_on_submits_an_execution_request(make_service, stub_llm_registry, execution_requests, surface, stream):
    make_service(enabled=True)
    hooks_mod.register_moderation_gateway_hooks()
    stub_llm_registry.reply = reply("fine")
    stub_llm_registry.items = [openai_frame("fine"), finish_frame(), DONE]
    with pytest.raises(LLMPolicyRefusal):
        _call(surface, stream)  # the input holds "forbidden"
    assert execution_requests == ["evaluate"]
