"""Tests for MCP orchestration fields on AgentRegistryEntry (Phase B)."""
from __future__ import annotations

import types

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACPAgentRegisterRequest,
    ACPAgentUpdateRequest,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistryEntry

pytestmark = pytest.mark.unit


def test_registry_mcp_fields_defaults():
    """Entry with just type+name should have all 7 MCP defaults."""
    entry = AgentRegistryEntry(type="test_agent", name="Test Agent")

    assert entry.mcp_orchestration == "agent_driven"
    assert entry.mcp_entry_tool == "execute"
    assert entry.mcp_structured_response is False
    assert entry.mcp_llm_provider is None
    assert entry.mcp_llm_model is None
    assert entry.mcp_max_iterations == 20
    assert entry.mcp_refresh_tools is False


def test_registry_mcp_llm_driven_config():
    """Entry with mcp_orchestration='llm_driven' and explicit provider/model/iterations."""
    entry = AgentRegistryEntry(
        type="llm_agent",
        name="LLM Agent",
        mcp_orchestration="llm_driven",
        mcp_llm_provider="openai",
        mcp_llm_model="gpt-4o",
        mcp_max_iterations=50,
    )

    assert entry.mcp_orchestration == "llm_driven"
    assert entry.mcp_llm_provider == "openai"
    assert entry.mcp_llm_model == "gpt-4o"
    assert entry.mcp_max_iterations == 50
    # Other defaults unchanged
    assert entry.mcp_entry_tool == "execute"
    assert entry.mcp_structured_response is False
    assert entry.mcp_refresh_tools is False


def test_agent_register_request_preserves_mcp_fields():
    """ACP agent registration payloads should expose MCP orchestration fields."""
    request = ACPAgentRegisterRequest(
        agent_type="mcp_agent",
        name="MCP Agent",
        command="mcp-cli",
        mcp_orchestration="llm_driven",
        mcp_entry_tool="run",
        mcp_structured_response=True,
        mcp_llm_provider="openai",
        mcp_llm_model="gpt-4o",
        mcp_max_iterations=6,
        mcp_refresh_tools=True,
    )

    assert request.mcp_orchestration == "llm_driven"
    assert request.mcp_entry_tool == "run"
    assert request.mcp_structured_response is True
    assert request.mcp_llm_provider == "openai"
    assert request.mcp_llm_model == "gpt-4o"
    assert request.mcp_max_iterations == 6
    assert request.mcp_refresh_tools is True


def test_agent_update_request_preserves_mcp_fields():
    """ACP agent update payloads should serialize MCP orchestration fields."""
    request = ACPAgentUpdateRequest(
        mcp_orchestration="llm_driven",
        mcp_entry_tool="run",
        mcp_structured_response=True,
        mcp_llm_provider="openai",
        mcp_llm_model="gpt-4o-mini",
        mcp_max_iterations=8,
        mcp_refresh_tools=True,
    )

    payload = request.model_dump(exclude_unset=True, exclude_none=True)

    assert payload == {
        "mcp_orchestration": "llm_driven",
        "mcp_entry_tool": "run",
        "mcp_structured_response": True,
        "mcp_llm_provider": "openai",
        "mcp_llm_model": "gpt-4o-mini",
        "mcp_max_iterations": 8,
        "mcp_refresh_tools": True,
    }


def test_agent_register_request_exposes_entrypoint_strategy_fields():
    request = ACPAgentRegisterRequest(
        agent_type="native",
        name="Native",
        entrypoint_strategy="native_acp",
        acp_command="native-agent",
        acp_args=["acp"],
    )

    assert request.entrypoint_strategy == "native_acp"
    assert request.acp_command == "native-agent"
    assert request.acp_args == ["acp"]


def test_agent_register_request_rejects_invalid_entrypoint_strategy():
    with pytest.raises(ValidationError):
        ACPAgentRegisterRequest(
            agent_type="bad",
            name="Bad",
            entrypoint_strategy="maybe_acp",
        )


def test_agent_update_request_rejects_explicit_null_for_required_scalar():
    with pytest.raises(ValidationError):
        ACPAgentUpdateRequest(name=None)


async def test_update_endpoint_preserves_explicit_nullable_fields(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry as registry_mod

    captured_updates: dict[str, object] = {}

    class _Registry:
        def update_agent(self, agent_type: str, **kwargs):
            captured_updates.update(kwargs)
            return types.SimpleNamespace(type=agent_type, name="Updated")

    monkeypatch.setattr(registry_mod, "get_agent_registry", lambda: _Registry())

    request = ACPAgentUpdateRequest(
        name="Updated",
        adapter_source=None,
        adapter_docs_url=None,
        certification_blocker=None,
        acp_args=None,
    )
    user = types.SimpleNamespace(id=1, is_admin=True)

    response = await acp_endpoints.acp_update_agent("adapter_agent", request, user)

    assert response.status == "updated"
    assert captured_updates == {
        "name": "Updated",
        "acp_args": None,
        "adapter_source": None,
        "adapter_docs_url": None,
        "certification_blocker": None,
    }
