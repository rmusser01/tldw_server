from pathlib import Path

import pytest
import yaml

from backlog_py.oracle.manifest import load_oracle_manifest


MANIFEST_PATH = Path(__file__).parent / "fixtures" / "oracle" / "manifest.yml"


def test_manifest_pins_upstream_version_and_source():
    manifest = load_oracle_manifest(MANIFEST_PATH)

    assert manifest.upstream_version == "1.44.0"
    assert manifest.source_kind == "npm-release"
    assert manifest.source_ref == "backlog.md@1.44.0"
    assert manifest.package_metadata_sha256 == "b890dde4a33480361ff34195192e1c0a23d6c7dc1c47b095933a29c7ccb4eee6"


def test_manifest_marks_agent_critical_fixtures():
    manifest = load_oracle_manifest(MANIFEST_PATH)
    names = {fixture.name for fixture in manifest.fixtures if fixture.agent_critical}

    assert "cli:task-list-plain" in names
    assert "mcp:workflow-overview" in names


def test_manifest_rejects_invalid_fixture_types(tmp_path):
    manifest_path = tmp_path / "manifest.yml"
    payload = {
        "upstream_version": "1.44.0",
        "source_kind": "npm-release",
        "source_ref": "backlog.md@1.44.0",
        "package_metadata_sha256": "sha",
        "generated_at": "2026-05-10T00:00:00Z",
        "fixtures": [
            {
                "name": "cli:task-list-plain",
                "agent_critical": "true",
                "command": ["task", "list", "--plain"],
            }
        ],
    }
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="agent_critical"):
        load_oracle_manifest(manifest_path)


def test_manifest_rejects_non_string_fixture_values(tmp_path):
    manifest_path = tmp_path / "manifest.yml"
    payload = {
        "upstream_version": "1.44.0",
        "source_kind": "npm-release",
        "source_ref": "backlog.md@1.44.0",
        "package_metadata_sha256": "sha",
        "generated_at": "2026-05-10T00:00:00Z",
        "fixtures": [
            {
                "name": "cli:task-list-plain",
                "agent_critical": True,
                "command": ["task", 1, "--plain"],
            }
        ],
    }
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="command"):
        load_oracle_manifest(manifest_path)
