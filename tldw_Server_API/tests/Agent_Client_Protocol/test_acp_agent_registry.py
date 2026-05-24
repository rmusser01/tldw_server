"""Tests for ACP agent registry (Phase 2)."""
from __future__ import annotations

import os
import tempfile

import pytest

pytestmark = pytest.mark.unit


SAMPLE_REGISTRY = """
agents:
  - type: claude_code
    name: Claude Code
    description: Test agent
    command: nonexistent_binary_xyz
    args: []
    env: {}
    requires_api_key: ANTHROPIC_API_KEY
    default: true

  - type: test_agent
    name: Test Agent
    description: A test agent
    command: python3
    args: ["-c", "pass"]
    env: {}
    requires_api_key: null
    default: false
    support_state: supported_with_caveats
    verification_level: stub_smoke_tested
    compatibility_notes: Stub protocol coverage only
    compatibility_docs_url: /docs-static/Development/ACP_Compatibility_Matrix.md
"""


@pytest.fixture
def registry_file():
    """Create a temporary registry YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(SAMPLE_REGISTRY)
        path = f.name
    yield path
    os.unlink(path)


def test_registry_loads_from_yaml(registry_file):
    """Registry loads agent entries from YAML file."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path=registry_file)
    reg.load()

    assert len(reg.entries) == 2
    assert reg.default_type == "claude_code"


def test_registry_entry_fields(registry_file):
    """Registry entries have correct field values."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path=registry_file)
    reg.load()

    entry = reg.get_entry("claude_code")
    assert entry is not None
    assert entry.name == "Claude Code"
    assert entry.requires_api_key == "ANTHROPIC_API_KEY"
    assert entry.default is True


def test_registry_loads_mcp_fields_from_yaml(tmp_path):
    """YAML-defined MCP fields should populate the registry entry."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    yaml_content = """
agents:
  - type: mcp_agent
    name: MCP Agent
    command: python3
    mcp_orchestration: llm_driven
    mcp_entry_tool: run
    mcp_structured_response: true
    mcp_llm_provider: openai
    mcp_llm_model: gpt-4o
    mcp_max_iterations: 7
    mcp_refresh_tools: true
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)

    reg = AgentRegistry(yaml_path=str(yaml_file))
    reg.load()

    entry = reg.get_entry("mcp_agent")
    assert entry is not None
    assert entry.mcp_orchestration == "llm_driven"
    assert entry.mcp_entry_tool == "run"
    assert entry.mcp_structured_response is True
    assert entry.mcp_llm_provider == "openai"
    assert entry.mcp_llm_model == "gpt-4o"
    assert entry.mcp_max_iterations == 7
    assert entry.mcp_refresh_tools is True


def test_registry_loads_compatibility_fields_from_yaml(registry_file):
    """YAML-defined compatibility fields should populate registry entries and availability."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path=registry_file)
    reg.load()

    entry = reg.get_entry("test_agent")
    assert entry is not None
    assert entry.support_state == "supported_with_caveats"
    assert entry.verification_level == "stub_smoke_tested"
    assert entry.compatibility_notes == "Stub protocol coverage only"
    assert entry.compatibility_docs_url == "/docs-static/Development/ACP_Compatibility_Matrix.md"

    availability = entry.check_availability()
    assert availability["support_state"] == "supported_with_caveats"
    assert availability["verification_level"] == "stub_smoke_tested"
    assert availability["compatibility_notes"] == "Stub protocol coverage only"
    assert availability["compatibility_docs_url"] == "/docs-static/Development/ACP_Compatibility_Matrix.md"


def test_registry_get_entry_none(registry_file):
    """get_entry returns None for unknown type."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path=registry_file)
    reg.load()

    assert reg.get_entry("nonexistent") is None


def test_registry_check_availability(registry_file, monkeypatch):
    """check_availability detects missing binary and API key."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    reg = AgentRegistry(yaml_path=registry_file)
    reg.load()

    # claude_code has nonexistent binary and (likely) no API key
    entry = reg.get_entry("claude_code")
    avail = entry.check_availability()
    assert avail["binary_found"] is False
    assert avail["status"] == "unavailable"

    # test_agent uses python3 (should be on PATH) and no API key required
    entry = reg.get_entry("test_agent")
    avail = entry.check_availability()
    assert avail["binary_found"] is True
    assert avail["api_key_set"] is True
    assert avail["status"] == "available"


def test_registry_get_available_agents(registry_file):
    """get_available_agents returns all entries with availability info."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path=registry_file)
    reg.load()

    available = reg.get_available_agents()
    assert len(available) == 2
    for agent in available:
        assert "type" in agent
        assert "name" in agent
        assert "status" in agent
        assert agent["status"] in ("available", "unavailable", "requires_setup")


def test_registry_missing_file():
    """Registry handles missing YAML file gracefully."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path="/nonexistent/agents.yaml")
    reg.load()
    assert reg.entries == []
    assert reg.default_type == "custom"


def test_registry_invalid_yaml():
    """Registry handles invalid YAML content gracefully."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("not: [valid: yaml: content")
        path = f.name

    try:
        reg = AgentRegistry(yaml_path=path)
        reg.load()
        # Should not crash, just produce empty entries
    finally:
        os.unlink(path)


def test_registry_hot_reload(registry_file):
    """Registry detects file changes and reloads."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    reg = AgentRegistry(yaml_path=registry_file)
    reg._reload_interval = 0  # Disable throttling
    reg.load()
    assert len(reg.entries) == 2

    # Modify the file
    with open(registry_file, "w") as f:
        f.write("""
agents:
  - type: new_agent
    name: New Agent
    description: Added dynamically
    command: python3
    args: []
    env: {}
    requires_api_key: null
    default: true
""")

    # Touch file to update mtime
    import time
    os.utime(registry_file, (time.time() + 1, time.time() + 1))

    # Access entries to trigger reload
    entries = reg.entries
    assert len(entries) == 1
    assert entries[0].type == "new_agent"


def test_default_agents_yaml_exists():
    """The shipped agents.yaml file exists and is valid."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "Config_Files", "agents.yaml",
    )
    assert os.path.isfile(config_path), f"agents.yaml not found at {config_path}"

    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry
    reg = AgentRegistry(yaml_path=config_path)
    reg.load()
    assert len(reg.entries) >= 3  # claude_code, codex, opencode, custom


def test_registry_entry_has_install_instructions(tmp_path):
    """Registry entries parse install_instructions and docs_url."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    yaml_content = """
agents:
  - type: claude_code
    name: Claude Code
    command: nonexistent_xyz
    requires_api_key: ANTHROPIC_API_KEY
    default: true
    install_instructions:
      - "npm install -g @anthropic-ai/claude-code"
    docs_url: "https://docs.anthropic.com/claude-code"
  - type: aider
    name: Aider
    command: aider
    requires_api_key: null
    install_instructions:
      - "pip install aider-chat"
    docs_url: "https://aider.chat"
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    registry = AgentRegistry(yaml_path=str(yaml_file))
    entry = registry.get_entry("claude_code")
    assert entry is not None
    assert entry.install_instructions == ["npm install -g @anthropic-ai/claude-code"]
    assert entry.docs_url == "https://docs.anthropic.com/claude-code"


def test_registry_entry_defaults_empty_install(tmp_path):
    """Entries without install_instructions/docs_url get sensible defaults."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    yaml_content = """
agents:
  - type: test_agent
    name: Test
    command: test-bin
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)
    registry = AgentRegistry(yaml_path=str(yaml_file))
    entry = registry.get_entry("test_agent")
    assert entry is not None
    assert entry.install_instructions == []
    assert entry.docs_url is None


def test_registry_loads_new_agent_types():
    """Verify the extended agents.yaml has the new agent types."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    real_yaml = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "Config_Files", "agents.yaml",
    )
    real_yaml = os.path.abspath(real_yaml)
    registry = AgentRegistry(yaml_path=real_yaml)
    entries = registry.entries
    types = {e.type for e in entries}
    assert "aider" in types
    assert "goose" in types
    assert "hermes" in types
    assert "continue_dev" in types
    assert "claude_code" in types


def test_default_agents_yaml_includes_hermes_native_acp_entrypoint():
    """Hermes is shipped as a native ACP profile with its accept-hooks stdio entrypoint."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    real_yaml = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "Config_Files", "agents.yaml",
    )
    real_yaml = os.path.abspath(real_yaml)
    registry = AgentRegistry(yaml_path=real_yaml)

    entry = registry.get_entry("hermes")

    assert entry is not None
    assert entry.name == "Hermes"
    assert entry.entrypoint_strategy == "native_acp"
    assert entry.command == "hermes"
    assert entry.acp_command == "hermes"
    assert entry.acp_args == ["acp", "--accept-hooks"]
    assert entry.support_state == "documented_unverified"
    assert entry.verification_level == "documented_only"


# ---------------------------------------------------------------------------
# Dynamic registration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def db_registry(tmp_path):
    """Registry backed by temp SQLite for dynamic registration tests."""
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry

    db_path = str(tmp_path / "acp_sessions.db")
    db = ACPSessionsDB(db_path=db_path)

    yaml_content = """
agents:
  - type: claude_code
    name: Claude Code
    command: nonexistent_binary_xyz
    requires_api_key: ANTHROPIC_API_KEY
    default: true
"""
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(yaml_content)

    registry = AgentRegistry(yaml_path=str(yaml_file), db=db)
    registry.load()
    yield registry
    db.close()


class TestDynamicRegistration:
    def test_register_agent(self, db_registry):
        entry = db_registry.register_agent(
            type="my_agent", name="My Agent", command="my-agent-cli",
        )
        assert entry.type == "my_agent"
        assert db_registry.get_entry("my_agent") is not None

    def test_deregister_agent(self, db_registry):
        db_registry.register_agent(type="tmp", name="Tmp", command="tmp")
        assert db_registry.deregister_agent("tmp") is True
        assert db_registry.get_entry("tmp") is None

    def test_deregister_nonexistent(self, db_registry):
        assert db_registry.deregister_agent("nonexistent") is False

    def test_yaml_entries_preserved_after_register(self, db_registry):
        assert db_registry.get_entry("claude_code") is not None
        db_registry.register_agent(type="new", name="New", command="new")
        assert db_registry.get_entry("claude_code") is not None
        assert db_registry.get_entry("new") is not None

    def test_api_overrides_yaml_same_type(self, db_registry):
        db_registry.register_agent(type="claude_code", name="Custom Claude", command="my-claude")
        entry = db_registry.get_entry("claude_code")
        assert entry.name == "Custom Claude"
        assert entry.command == "my-claude"

    def test_update_agent(self, db_registry):
        db_registry.register_agent(type="my_agent", name="My Agent", command="cmd")
        updated = db_registry.update_agent("my_agent", name="Updated Agent", description="new desc")
        assert updated is not None
        assert updated.name == "Updated Agent"
        assert updated.description == "new desc"

    def test_update_nonexistent(self, db_registry):
        assert db_registry.update_agent("nonexistent", name="foo") is None

    def test_persistence_across_reload(self, db_registry):
        """Registered agents survive a registry reload."""
        db_registry.register_agent(type="persistent_agent", name="Persistent", command="persist-cmd")
        db_registry._reload_interval = 0
        db_registry.load()
        entry = db_registry.get_entry("persistent_agent")
        assert entry is not None
        assert entry.name == "Persistent"

    def test_cannot_deregister_yaml_only(self, db_registry):
        """Deregistering a YAML-only entry returns False."""
        assert db_registry.deregister_agent("claude_code") is False

    def test_register_agent_persists_mcp_fields_across_reload(self, db_registry):
        """Dynamic registrations should keep MCP config across DB-backed reloads."""
        db_registry.register_agent(
            type="mcp_agent",
            name="MCP Agent",
            command="mcp-cli",
            mcp_orchestration="llm_driven",
            mcp_entry_tool="run",
            mcp_structured_response=True,
            mcp_llm_provider="openai",
            mcp_llm_model="gpt-4o",
            mcp_max_iterations=9,
            mcp_refresh_tools=True,
        )

        db_registry.load()
        entry = db_registry.get_entry("mcp_agent")
        assert entry is not None
        assert entry.mcp_orchestration == "llm_driven"
        assert entry.mcp_entry_tool == "run"
        assert entry.mcp_structured_response is True
        assert entry.mcp_llm_provider == "openai"
        assert entry.mcp_llm_model == "gpt-4o"
        assert entry.mcp_max_iterations == 9
        assert entry.mcp_refresh_tools is True

    def test_update_agent_persists_mcp_fields_across_reload(self, db_registry):
        """Dynamic updates should save MCP config changes back to the DB."""
        db_registry.register_agent(type="mcp_agent", name="MCP Agent", command="mcp-cli")

        updated = db_registry.update_agent(
            "mcp_agent",
            mcp_orchestration="llm_driven",
            mcp_entry_tool="run",
            mcp_structured_response=True,
            mcp_llm_provider="openai",
            mcp_llm_model="gpt-4o-mini",
            mcp_max_iterations=11,
            mcp_refresh_tools=True,
        )

        assert updated is not None

        db_registry.load()
        entry = db_registry.get_entry("mcp_agent")
        assert entry is not None
        assert entry.mcp_orchestration == "llm_driven"
        assert entry.mcp_entry_tool == "run"
        assert entry.mcp_structured_response is True
        assert entry.mcp_llm_provider == "openai"
        assert entry.mcp_llm_model == "gpt-4o-mini"
        assert entry.mcp_max_iterations == 11
        assert entry.mcp_refresh_tools is True
