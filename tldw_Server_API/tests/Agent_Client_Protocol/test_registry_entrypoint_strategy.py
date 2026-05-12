from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
    AgentRegistry,
    AgentRegistryEntry,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def acp_db(tmp_path):
    db = ACPSessionsDB(db_path=str(tmp_path / "acp_sessions.db"))
    try:
        yield db
    finally:
        db.close()


def test_entrypoint_strategy_defaults_to_documented_candidate() -> None:
    entry = AgentRegistryEntry(type="legacy", name="Legacy")

    assert entry.entrypoint_strategy == "documented_candidate"
    assert entry.acp_command == ""
    assert entry.acp_args == []
    assert entry.adapter_source is None
    assert entry.adapter_docs_url is None
    assert entry.certification_blocker is None


def test_registry_loads_entrypoint_strategy_fields_from_yaml(tmp_path) -> None:
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(
        """
agents:
  - type: opencode
    name: OpenCode
    command: opencode
    entrypoint_strategy: native_acp
    acp_command: opencode
    acp_args: ["acp"]
"""
    )

    registry = AgentRegistry(yaml_path=str(yaml_file))
    registry.load()

    entry = registry.get_entry("opencode")
    assert entry is not None
    assert entry.entrypoint_strategy == "native_acp"
    assert entry.acp_command == "opencode"
    assert entry.acp_args == ["acp"]


def test_dynamic_registration_preserves_entrypoint_strategy_fields(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)

    entry = registry.register_agent(
        type="adapter_agent",
        name="Adapter Agent",
        command="agent-cli",
        entrypoint_strategy="adapter_acp",
        acp_command="agent-acp",
        acp_args=["--stdio"],
        adapter_source="example/agent-acp",
        adapter_docs_url="https://example.test/agent-acp",
        certification_blocker="adapter_missing",
    )

    assert entry.entrypoint_strategy == "adapter_acp"
    assert entry.acp_command == "agent-acp"
    assert entry.acp_args == ["--stdio"]

    reloaded = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    reloaded._load_api_entries()
    persisted = reloaded.get_entry("adapter_agent")
    assert persisted is not None
    assert persisted.entrypoint_strategy == "adapter_acp"
    assert persisted.acp_command == "agent-acp"
    assert persisted.acp_args == ["--stdio"]
    assert persisted.adapter_source == "example/agent-acp"


def test_update_agent_clears_nullable_entrypoint_metadata(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    registry.register_agent(
        type="adapter_agent",
        name="Adapter Agent",
        adapter_source="example/agent-acp",
        adapter_docs_url="https://example.test/agent-acp",
        certification_blocker="adapter_missing",
    )

    updated = registry.update_agent(
        "adapter_agent",
        adapter_source=None,
        adapter_docs_url=None,
        certification_blocker=None,
    )

    assert updated is not None
    assert updated.adapter_source is None
    assert updated.adapter_docs_url is None
    assert updated.certification_blocker is None

    persisted = acp_db.get_agent_entry("adapter_agent")
    assert persisted is not None
    assert persisted["adapter_source"] is None
    assert persisted["adapter_docs_url"] is None
    assert persisted["certification_blocker"] is None


def test_update_agent_ignores_none_for_required_name_without_db_failure(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    registry.register_agent(type="named_agent", name="Named Agent")

    updated = registry.update_agent("named_agent", name=None)

    assert updated is not None
    assert updated.name == "Named Agent"
    persisted = acp_db.get_agent_entry("named_agent")
    assert persisted is not None
    assert persisted["name"] == "Named Agent"


def test_register_agent_coerces_invalid_entrypoint_strategy(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)

    entry = registry.register_agent(
        type="bad_strategy",
        name="Bad Strategy",
        entrypoint_strategy="maybe_acp",  # type: ignore[arg-type]
    )

    assert entry.entrypoint_strategy == "documented_candidate"

    reloaded = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    reloaded._load_api_entries()
    persisted = reloaded.get_entry("bad_strategy")
    assert persisted is not None
    assert persisted.entrypoint_strategy == "documented_candidate"


def test_update_agent_coerces_invalid_entrypoint_strategy() -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml")
    registry.register_agent(
        type="bad_update_strategy",
        name="Bad Update Strategy",
        entrypoint_strategy="native_acp",
    )

    updated = registry.update_agent(
        "bad_update_strategy",
        entrypoint_strategy="maybe_acp",
    )

    assert updated is not None
    assert updated.entrypoint_strategy == "documented_candidate"


def test_update_agent_collection_defaults_are_fresh_instances() -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml")
    first = registry.register_agent(
        type="first",
        name="First",
        acp_args=["initial"],
    )
    second = registry.register_agent(
        type="second",
        name="Second",
        acp_args=["initial"],
    )

    first = registry.update_agent("first", acp_args=None)
    assert first is not None
    first.acp_args.append("--first-only")
    second = registry.update_agent("second", acp_args=None)

    assert second is not None
    assert second.acp_args == []
    assert second.acp_args is not first.acp_args
