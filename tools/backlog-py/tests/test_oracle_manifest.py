from pathlib import Path

from backlog_py.oracle.manifest import load_oracle_manifest


MANIFEST_PATH = Path(__file__).parent / "fixtures" / "oracle" / "manifest.yml"


def test_manifest_pins_upstream_version_and_source():
    manifest = load_oracle_manifest(MANIFEST_PATH)

    assert manifest.upstream_version == "1.44.0"
    assert manifest.source_kind in {"npm-release", "github-release", "source-commit"}
    assert manifest.source_ref
    assert manifest.package_metadata_sha256


def test_manifest_marks_agent_critical_fixtures():
    manifest = load_oracle_manifest(MANIFEST_PATH)
    names = {fixture.name for fixture in manifest.fixtures if fixture.agent_critical}

    assert "cli:task-list-plain" in names
    assert "mcp:workflow-overview" in names
