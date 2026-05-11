from pathlib import Path

from backlog_py.compat.inventory import load_builtin_inventory
from backlog_py.oracle.manifest import load_oracle_manifest


MATRIX_PATH = Path("tools/backlog-py/docs/agent-critical-parity.md")
MANIFEST_PATH = Path(__file__).parent / "fixtures" / "oracle" / "manifest.yml"

EXPECTED_AGENT_CRITICAL = {
    "cli:help",
    "cli:task-list-plain",
    "cli:task-view-plain",
    "cli:search-plain",
    "cli:board",
    "cli:config-list",
    "cli:task-create",
    "cli:task-edit",
    "cli:doc-list",
    "cli:doc-view",
    "cli:doc-create",
    "cli:doc-update",
    "cli:milestone-list",
    "cli:milestone-add",
    "cli:milestone-rename",
    "cli:milestone-remove",
    "cli:milestone-archive",
    "cli:config-dod-defaults-get",
    "cli:config-dod-defaults-upsert",
    "mcp:workflow-overview",
    "mcp:task-workflow-alias",
    "mcp:task-search",
    "mcp:task-view",
    "mcp:task-create",
    "mcp:task-edit",
    "mcp:document-list",
    "mcp:document-search",
    "mcp:document-view",
    "mcp:document-create",
    "mcp:document-update",
    "mcp:milestone-list",
    "mcp:milestone-add",
    "mcp:milestone-rename",
    "mcp:milestone-remove",
    "mcp:milestone-archive",
    "mcp:definition-of-done-defaults-get",
    "mcp:definition-of-done-defaults-upsert",
}

EXPECTED_DEFERRED = {
    "browser:kanban-drag-drop",
    "cli:interactive-board",
    "cli:rich-colored-output",
    "cli:shell-completion-install",
    "core:on-status-change",
    "git:remote-operations",
    "git:auto-commit",
    "git:hook-bypass",
}


def test_agent_critical_inventory_enumerates_cutover_and_deferral_scope():
    inventory = load_builtin_inventory()
    by_name = {item.name: item for item in inventory.items}

    assert sorted(EXPECTED_AGENT_CRITICAL - by_name.keys()) == []
    assert sorted(EXPECTED_DEFERRED - by_name.keys()) == []

    for name in EXPECTED_AGENT_CRITICAL:
        item = by_name[name]
        assert item.classification == "golden-required"
        assert item.status == "implemented"
        assert item.fixture == name
        assert item.expected

    for name in EXPECTED_DEFERRED:
        item = by_name[name]
        assert item.classification != "golden-required"
        assert item.status == "deferred"
        assert item.deferred_reason


def test_agent_critical_inventory_has_fixture_coverage():
    inventory = load_builtin_inventory()
    manifest = load_oracle_manifest(MANIFEST_PATH)
    fixture_names = {fixture.name for fixture in manifest.fixtures}

    missing = [
        item.name
        for item in inventory.items
        if item.classification == "golden-required" and item.name not in fixture_names
    ]

    assert missing == []


def test_agent_critical_manifest_tracks_all_inventory_items():
    inventory = load_builtin_inventory()
    manifest = load_oracle_manifest(MANIFEST_PATH)
    fixture_by_name = {fixture.name: fixture for fixture in manifest.fixtures}

    assert sorted({item.name for item in inventory.items} - fixture_by_name.keys()) == []

    for item in inventory.items:
        fixture = fixture_by_name[item.name]
        assert fixture.classification == item.classification
        assert fixture.agent_critical is (item.classification == "golden-required")


def test_agent_critical_matrix_doc_matches_inventory():
    inventory = load_builtin_inventory()
    matrix = MATRIX_PATH.read_text(encoding="utf-8")

    for item in inventory.items:
        assert item.name in matrix
        assert item.expected in matrix
        assert item.status in matrix
        if item.classification == "golden-required":
            assert item.fixture in matrix
        else:
            assert item.deferred_reason in matrix
