import pytest

from tldw_Server_API.app.core.Setup.first_run_mcp_tools import (
    CATALOG_VERSION,
    CONFIRMATION_VERSION,
    build_mcp_tools_catalog,
    compute_first_run_policy_hash,
    generate_first_run_policy,
)


def test_default_catalog_exposes_five_selected_packs():
    catalog = build_mcp_tools_catalog(tool_entries=[])

    assert [pack["pack_id"] for pack in catalog["packs"]] == [
        "research",
        "learning",
        "writing",
        "media_library",
        "personal_knowledge",
    ]
    assert all(pack["default_selected"] is True for pack in catalog["packs"])


def test_default_catalog_disables_risky_addons():
    catalog = build_mcp_tools_catalog(tool_entries=[])

    assert {addon["addon_id"]: addon["default_selected"] for addon in catalog["add_ons"]} == {
        "external_network_read": False,
        "local_file_read": False,
        "workspace_write": False,
        "destructive_actions": False,
        "process_run_command": False,
    }


def test_unknown_saved_pack_ids_are_returned_as_unavailable_legacy_choices():
    catalog = build_mcp_tools_catalog(
        tool_entries=[],
        selected_pack_ids=["research", "legacy_pack"],
    )

    legacy_pack = catalog["packs"][-1]
    assert legacy_pack["pack_id"] == "legacy_pack"
    assert legacy_pack["available"] is False
    assert legacy_pack["legacy"] is True
    assert legacy_pack["default_selected"] is False


def test_catalog_marks_missing_spec_tools_unavailable_when_registry_is_available():
    catalog = build_mcp_tools_catalog(
        tool_entries=[
            {
                "tool_name": "knowledge.search",
                "module": "knowledge",
                "risk_class": "low",
                "mutates_state": False,
            }
        ]
    )

    research = catalog["packs"][0]
    available = {tool["tool_name"] for tool in research["available_tools"]}
    unavailable = {tool["tool_name"] for tool in research["unavailable_tools"]}

    assert "knowledge.search" in available
    assert "knowledge.get" in unavailable


def test_default_policy_keeps_mcp_discovery_tools_with_registry_rows():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "mcp.catalogs.list",
                "module": "mcp_discovery",
                "category": "unclassified",
                "risk_class": "unclassified",
                "mutates_state": False,
                "uses_filesystem": False,
                "uses_network": False,
                "uses_processes": False,
                "catalog_exempt": True,
            },
            {
                "tool_name": "mcp.modules.list",
                "module": "mcp_discovery",
                "category": "unclassified",
                "risk_class": "unclassified",
                "mutates_state": False,
                "uses_filesystem": False,
                "uses_network": False,
                "uses_processes": False,
                "catalog_exempt": True,
            },
            {
                "tool_name": "mcp.tools.list",
                "module": "mcp_discovery",
                "category": "unclassified",
                "risk_class": "unclassified",
                "mutates_state": False,
                "uses_filesystem": False,
                "uses_network": False,
                "uses_processes": False,
                "catalog_exempt": True,
            },
        ],
    )

    assert "mcp.catalogs.list" in policy["allowed_tools"]
    assert "mcp.modules.list" in policy["allowed_tools"]
    assert "mcp.tools.list" in policy["allowed_tools"]


def test_default_policy_uses_explicit_allowed_tools_not_module_patterns():
    policy = generate_first_run_policy(
        selected_pack_ids=["research", "writing"],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.search",
                "module": "notes",
                "risk_class": "low",
                "mutates_state": False,
            },
            {
                "tool_name": "notes.create",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
            },
        ],
    )

    assert "notes.search" in policy["allowed_tools"]
    assert "notes.create" not in policy["allowed_tools"]
    assert "module_patterns" not in policy
    assert policy["capabilities"] == []


def test_default_policy_does_not_enable_broad_capabilities():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[],
    )

    assert policy["capabilities"] == []
    assert "filesystem.read" not in policy["allowed_tools"]
    assert "filesystem.write" not in policy["allowed_tools"]
    assert "filesystem.delete" not in policy["allowed_tools"]
    assert "network.external" not in policy["allowed_tools"]
    assert "process.execute" not in policy["allowed_tools"]


def test_generated_policy_includes_baseline_when_no_packs_selected():
    policy = generate_first_run_policy(
        selected_pack_ids=[],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[],
    )

    assert policy["allowed_tools"] == ["mcp.tools.list"]


def test_generated_policy_includes_baseline_when_pack_tools_are_filtered_out():
    policy = generate_first_run_policy(
        selected_pack_ids=["writing"],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.search",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
            }
        ],
    )

    assert policy["allowed_tools"] == ["mcp.tools.list"]


def test_generated_policy_hash_changes_when_generated_capabilities_change():
    base_policy = generate_first_run_policy(
        selected_pack_ids=["writing"],
        selected_addon_ids=["destructive_actions"],
        confirmed_addon_ids=["destructive_actions"],
        confirmation_version=CONFIRMATION_VERSION,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.delete",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
                "destructive": True,
            }
        ],
    )
    capability_policy = generate_first_run_policy(
        selected_pack_ids=["writing"],
        selected_addon_ids=["destructive_actions"],
        confirmed_addon_ids=["destructive_actions"],
        confirmation_version=CONFIRMATION_VERSION,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.delete",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
                "uses_filesystem": True,
                "destructive": True,
            }
        ],
    )

    assert base_policy["allowed_tools"] == capability_policy["allowed_tools"]
    assert base_policy["capabilities"] == []
    assert capability_policy["capabilities"] == ["filesystem.delete"]
    assert (
        base_policy["first_run_mcp_tools"]["generated_policy_hash"]
        != capability_policy["first_run_mcp_tools"]["generated_policy_hash"]
    )


def test_compute_policy_hash_uses_stored_catalog_version():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[],
    )
    old_policy = {
        **policy,
        "first_run_mcp_tools": {
            **policy["first_run_mcp_tools"],
            "catalog_version": "2026-01-01.v1",
        },
    }
    current_policy = {
        **policy,
        "first_run_mcp_tools": {
            **policy["first_run_mcp_tools"],
            "catalog_version": CATALOG_VERSION,
        },
    }

    assert compute_first_run_policy_hash(old_policy) != compute_first_run_policy_hash(current_policy)


@pytest.mark.parametrize(
    "addon_id",
    ["workspace_write", "destructive_actions", "process_run_command"],
)
def test_strong_addons_require_selected_confirmed_and_current_confirmation(addon_id):
    with pytest.raises(ValueError, match="requires current confirmation"):
        generate_first_run_policy(
            selected_pack_ids=["research"],
            selected_addon_ids=[addon_id],
            confirmed_addon_ids=[],
            confirmation_version=CONFIRMATION_VERSION,
            setup_instance_id="first_run:test",
            tool_entries=[],
        )

    with pytest.raises(ValueError, match="requires current confirmation"):
        generate_first_run_policy(
            selected_pack_ids=["research"],
            selected_addon_ids=[addon_id],
            confirmed_addon_ids=[addon_id],
            confirmation_version="stale",
            setup_instance_id="first_run:test",
            tool_entries=[],
        )

    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=[],
        confirmed_addon_ids=[addon_id],
        confirmation_version=CONFIRMATION_VERSION,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.create",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
            }
        ],
    )
    assert "notes.create" not in policy["allowed_tools"]


def test_local_file_read_addon_adds_only_safe_read_file_tools():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=["local_file_read"],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "fs.read_text",
                "module": "filesystem",
                "risk_class": "low",
                "mutates_state": False,
                "uses_filesystem": True,
                "path_boundable": True,
            },
            {
                "tool_name": "fs.write_text",
                "module": "filesystem",
                "risk_class": "high",
                "mutates_state": True,
                "uses_filesystem": True,
            },
        ],
    )

    assert "filesystem.read" in policy["capabilities"]
    assert "filesystem.read" not in policy["allowed_tools"]
    assert "fs.read_text" in policy["allowed_tools"]
    assert "fs.write_text" not in policy["allowed_tools"]


def test_external_network_read_addon_adds_only_low_risk_external_read_tools():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=["external_network_read"],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "web.search",
                "module": "web",
                "risk_class": "low",
                "mutates_state": False,
                "uses_network": True,
            },
            {
                "tool_name": "web.lookup",
                "module": "web",
                "risk_class": "low",
                "mutates_state": False,
                "capabilities": ["network.external"],
            },
            {
                "tool_name": "web.post",
                "module": "web",
                "risk_class": "high",
                "mutates_state": True,
                "uses_network": True,
            },
        ],
    )

    assert "network.external" in policy["capabilities"]
    assert "network.external" not in policy["allowed_tools"]
    assert "web.search" in policy["allowed_tools"]
    assert "web.lookup" in policy["allowed_tools"]
    assert "web.post" not in policy["allowed_tools"]


def test_confirmed_workspace_write_addon_enumerates_non_destructive_write_tools():
    policy = generate_first_run_policy(
        selected_pack_ids=["writing"],
        selected_addon_ids=["workspace_write"],
        confirmed_addon_ids=["workspace_write"],
        confirmation_version=CONFIRMATION_VERSION,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.create",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
            },
            {
                "tool_name": "notes.delete",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
                "destructive": True,
            },
        ],
    )

    assert "notes.create" in policy["allowed_tools"]
    assert "notes.delete" not in policy["allowed_tools"]
    assert policy["capabilities"] == []
    assert "filesystem.write" not in policy["allowed_tools"]


def test_confirmed_destructive_addon_enumerates_delete_tools():
    policy = generate_first_run_policy(
        selected_pack_ids=["writing"],
        selected_addon_ids=["destructive_actions"],
        confirmed_addon_ids=["destructive_actions"],
        confirmation_version=CONFIRMATION_VERSION,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "notes.delete",
                "module": "notes",
                "risk_class": "high",
                "mutates_state": True,
                "destructive": True,
            },
            {
                "tool_name": "fs.delete",
                "module": "filesystem",
                "risk_class": "high",
                "mutates_state": True,
                "uses_filesystem": True,
                "destructive": True,
            },
        ],
    )

    assert "notes.delete" in policy["allowed_tools"]
    assert "fs.delete" in policy["allowed_tools"]
    assert "filesystem.delete" in policy["capabilities"]
    assert "filesystem.delete" not in policy["allowed_tools"]


def test_confirmed_process_addon_enumerates_process_tools():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=["process_run_command"],
        confirmed_addon_ids=["process_run_command"],
        confirmation_version=CONFIRMATION_VERSION,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "shell.run",
                "module": "shell",
                "risk_class": "high",
                "mutates_state": True,
                "uses_processes": True,
            }
        ],
    )

    assert "shell.run" in policy["allowed_tools"]
    assert "process.execute" in policy["capabilities"]
    assert "process.execute" not in policy["allowed_tools"]
