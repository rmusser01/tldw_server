from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
    AgentEntrypointClassification,
    AgentRegistry,
    AgentRegistryEntry,
    classify_agent_entrypoint,
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


def test_seeded_codex_profile_uses_pinned_external_acp_adapter() -> None:
    registry = AgentRegistry()
    registry.load()

    entry = registry.get_entry("codex")

    assert entry is not None
    assert entry.entrypoint_strategy == "external_acp_adapter"
    assert entry.command == "codex"
    assert entry.acp_command == "codex-acp"
    assert entry.adapter_source == "zed-industries/codex-acp"
    assert entry.adapter_version == "0.15.0"
    assert entry.adapter_version_policy == "exact_pin_required"
    assert entry.adapter_install_source == "github_release_preferred"
    assert entry.credential_policy == "delegated_to_adapter"
    assert entry.support_state == "experimental"
    assert entry.verification_level == "documented_only"


def test_legacy_adapter_acp_input_is_imported_as_external_acp_adapter(tmp_path) -> None:
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(
        """
agents:
  - type: legacy_codex
    name: Legacy Codex
    command: codex
    entrypoint_strategy: adapter_acp
    acp_command: codex-acp
"""
    )

    registry = AgentRegistry(yaml_path=str(yaml_file))
    registry.load()

    entry = registry.get_entry("legacy_codex")
    assert entry is not None
    assert entry.entrypoint_strategy == "external_acp_adapter"
    assert classify_agent_entrypoint(entry).entrypoint_strategy == "external_acp_adapter"


def test_registry_loads_null_yaml_acp_command_as_missing_entrypoint(tmp_path) -> None:
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(
        """
agents:
  - type: null_command_agent
    name: Null Command Agent
    command: null-command-agent
    entrypoint_strategy: native_acp
    acp_command:
"""
    )

    registry = AgentRegistry(yaml_path=str(yaml_file))
    registry.load()

    entry = registry.get_entry("null_command_agent")
    assert entry is not None
    assert entry.acp_command == ""

    result = classify_agent_entrypoint(entry)
    assert result.probe_state == "blocked"
    assert result.primary_blocker == "entrypoint_strategy_missing"
    assert "binary_missing" not in result.blockers


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

    assert entry.entrypoint_strategy == "external_acp_adapter"
    assert entry.acp_command == "agent-acp"
    assert entry.acp_args == ["--stdio"]

    reloaded = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    reloaded._load_api_entries()
    persisted = reloaded.get_entry("adapter_agent")
    assert persisted is not None
    assert persisted.entrypoint_strategy == "external_acp_adapter"
    assert persisted.acp_command == "agent-acp"
    assert persisted.acp_args == ["--stdio"]
    assert persisted.adapter_source == "example/agent-acp"
    assert persisted.adapter_docs_url == "https://example.test/agent-acp"
    assert persisted.certification_blocker == "adapter_missing"


def test_dynamic_registration_preserves_adapter_metadata_fields(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)

    entry = registry.register_agent(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="adapter_acp",
        acp_command="codex-acp",
        adapter_source="zed-industries/codex-acp",
        adapter_package="@zed-industries/codex-acp",
        adapter_version="0.15.0",
        adapter_version_policy="exact_pin_required",
        adapter_install_source="github_release_preferred",
        credential_policy="delegated_to_adapter",
        runtime_backend="acp_downstream",
    )

    assert entry.entrypoint_strategy == "external_acp_adapter"
    assert entry.adapter_version == "0.15.0"
    assert entry.credential_policy == "delegated_to_adapter"
    assert entry.runtime_backend == "acp_downstream"

    reloaded = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    reloaded._load_api_entries()
    persisted = reloaded.get_entry("codex")
    assert persisted is not None
    assert persisted.entrypoint_strategy == "external_acp_adapter"
    assert persisted.adapter_source == "zed-industries/codex-acp"
    assert persisted.adapter_package == "@zed-industries/codex-acp"
    assert persisted.adapter_version == "0.15.0"
    assert persisted.adapter_version_policy == "exact_pin_required"
    assert persisted.adapter_install_source == "github_release_preferred"
    assert persisted.credential_policy == "delegated_to_adapter"
    assert persisted.runtime_backend == "acp_downstream"


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


def test_update_agent_preserves_adapter_metadata_fields(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    registry.register_agent(type="codex", name="Codex")

    updated = registry.update_agent(
        "codex",
        entrypoint_strategy="adapter_acp",
        adapter_package="@zed-industries/codex-acp",
        adapter_version="0.15.0",
        adapter_version_policy="exact_pin_required",
        adapter_install_source="github_release_preferred",
        credential_policy="delegated_to_adapter",
        runtime_backend="acp_downstream",
    )

    assert updated is not None
    assert updated.entrypoint_strategy == "external_acp_adapter"
    assert updated.adapter_version == "0.15.0"
    assert updated.credential_policy == "delegated_to_adapter"
    assert updated.runtime_backend == "acp_downstream"

    persisted = acp_db.get_agent_entry("codex")
    assert persisted is not None
    assert persisted["entrypoint_strategy"] == "external_acp_adapter"
    assert persisted["adapter_version"] == "0.15.0"
    assert persisted["credential_policy"] == "delegated_to_adapter"
    assert persisted["runtime_backend"] == "acp_downstream"


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


def test_update_agent_coerces_invalid_entrypoint_strategy(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
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

    reloaded = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    reloaded._load_api_entries()
    persisted = reloaded.get_entry("bad_update_strategy")
    assert persisted is not None
    assert persisted.entrypoint_strategy == "documented_candidate"


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


def test_classifier_ready_to_probe_native_entrypoint() -> None:
    entry = AgentRegistryEntry(
        type="opencode",
        name="OpenCode",
        entrypoint_strategy="native_acp",
        acp_command="opencode",
        acp_args=["acp"],
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
        env_getter=lambda _name: "present",
    )

    assert result.probe_state == "ready_to_probe"
    assert result.acp_command == "opencode"
    assert result.acp_args == ("acp",)
    assert result.primary_blocker is None
    assert result.blockers == ()
    assert result.as_dict()["acp_args"] == ["acp"]
    assert result.as_dict()["blockers"] == []


def test_external_acp_adapter_is_canonical_strategy() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        adapter_source="zed-industries/codex-acp",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
        env_getter=lambda _name: None,
    )

    assert result.entrypoint_strategy == "external_acp_adapter"
    assert result.probe_state == "ready_to_probe"
    assert result.acp_command == "codex-acp"
    assert result.primary_blocker is None


def test_classification_is_immutable_against_source_and_as_dict_mutation() -> None:
    source_args = ["acp"]
    source_blockers = ["binary_missing"]
    result = AgentEntrypointClassification(
        profile_key="goose",
        entrypoint_strategy="native_acp",
        probe_state="blocked",
        acp_command="goose",
        acp_args=source_args,
        primary_blocker="binary_missing",
        blockers=source_blockers,
        status_message="blocked",
        docs_url=None,
    )

    source_args.append("--mutated")
    source_blockers.append("credentials_missing")

    serialized = result.as_dict()
    serialized["acp_args"].append("--dict-mutated")
    serialized["blockers"].append("dict_mutated")

    assert result.acp_args == ("acp",)
    assert result.blockers == ("binary_missing",)
    assert result.as_dict()["acp_args"] == ["acp"]
    assert result.as_dict()["blockers"] == ["binary_missing"]


def test_classifier_blocks_native_entrypoint_missing_command() -> None:
    entry = AgentRegistryEntry(
        type="goose",
        name="Goose",
        entrypoint_strategy="native_acp",
        acp_command="goose",
        acp_args=["acp"],
    )

    result = classify_agent_entrypoint(entry, command_resolver=lambda _command: None)

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "binary_missing"
    assert "binary_missing" in result.blockers


def test_classifier_blocks_missing_required_credentials() -> None:
    entry = AgentRegistryEntry(
        type="commercial_agent",
        name="Commercial Agent",
        entrypoint_strategy="native_acp",
        acp_command="commercial-agent",
        requires_api_key="COMMERCIAL_AGENT_API_KEY",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
        env_getter=lambda _name: None,
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "credentials_missing"
    assert "credentials_missing" in result.blockers


def test_classifier_reports_multiple_applicable_blockers() -> None:
    entry = AgentRegistryEntry(
        type="commercial_agent",
        name="Commercial Agent",
        entrypoint_strategy="native_acp",
        acp_command="commercial-agent",
        requires_api_key="COMMERCIAL_AGENT_API_KEY",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda _command: None,
        env_getter=lambda _name: None,
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "credentials_missing"
    assert result.blockers == ("credentials_missing", "binary_missing")
    assert "binary_missing" in result.status_message


def test_classifier_does_not_infer_native_acp_command_from_command() -> None:
    entry = AgentRegistryEntry(
        type="opencode",
        name="OpenCode",
        command="opencode",
        entrypoint_strategy="native_acp",
        acp_command="",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "entrypoint_strategy_missing"
    assert result.acp_command == ""


def test_classifier_does_not_infer_adapter_acp_command_from_command() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="adapter_acp",
        acp_command="",
        adapter_source="example/codex-acp",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "entrypoint_strategy_missing"
    assert result.acp_command == ""


def test_classifier_documented_candidate_keeps_command_separate_from_acp_command() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="documented_candidate",
        acp_command="",
        certification_blocker="adapter_required",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "documented_only"
    assert result.primary_blocker == "adapter_required"
    assert result.acp_command == ""


def test_classifier_documented_candidate_is_documented_only() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        entrypoint_strategy="documented_candidate",
        certification_blocker="adapter_required",
    )

    result = classify_agent_entrypoint(entry)

    assert result.probe_state == "documented_only"
    assert result.primary_blocker == "adapter_required"
    assert result.acp_command == ""


def test_classifier_adapter_requires_adapter_command() -> None:
    entry = AgentRegistryEntry(
        type="adapter",
        name="Adapter",
        entrypoint_strategy="adapter_acp",
        acp_command="adapter-acp",
        acp_args=[],
        adapter_source="example/adapter",
    )

    result = classify_agent_entrypoint(entry, command_resolver=lambda _command: None)

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "adapter_missing"


def test_external_adapter_reports_adapter_missing_without_falling_back_to_agent_command() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        adapter_source="zed-industries/codex-acp",
        credential_policy="delegated_to_adapter",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: "/usr/bin/codex" if command == "codex" else None,
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "adapter_missing"
    assert "adapter_missing" in result.blockers
    assert "binary_missing" not in result.blockers


def test_external_adapter_reports_display_agent_binary_missing_separately() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        credential_policy="delegated_to_adapter",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: "/usr/bin/codex-acp" if command == "codex-acp" else None,
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "agent_binary_missing"
    assert "agent_binary_missing" in result.blockers


def test_external_adapter_blocks_mutable_npx_latest_invocation() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="npx",
        acp_args=["@zed-industries/codex-acp@latest"],
        credential_policy="delegated_to_adapter",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "mutable_adapter_invocation"
    assert "mutable_adapter_invocation" in result.blockers


def test_classifier_custom_template_is_never_probe_ready() -> None:
    entry = AgentRegistryEntry(
        type="custom",
        name="Custom Agent",
        entrypoint_strategy="custom_template",
    )

    result = classify_agent_entrypoint(entry)

    assert result.probe_state == "custom_template"
    assert result.acp_command == ""
    assert result.primary_blocker == "custom_template"
    assert result.blockers == ("custom_template",)
    assert "command, args, env, workspace policy, and evidence bundle" in result.status_message


def test_classifier_rejects_shell_builtin_entrypoint() -> None:
    entry = AgentRegistryEntry(
        type="bad",
        name="Bad",
        entrypoint_strategy="native_acp",
        acp_command="cd",
    )

    result = classify_agent_entrypoint(entry, command_resolver=lambda _command: "/bin/cd")

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "shell_builtin_collision"
