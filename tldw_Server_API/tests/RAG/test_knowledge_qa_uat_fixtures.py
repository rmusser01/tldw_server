from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from Helper_Scripts import seed_knowledge_qa_uat
from tldw_Server_API.tests.RAG import knowledge_qa_uat_fixtures as fixtures


@pytest.mark.unit
def test_knowledge_qa_uat_fixture_strings_are_distinct_and_deterministic() -> None:
    values = [
        fixtures.KNOWN_CITED_QUERY,
        fixtures.KNOWN_CITED_ANSWER_PHRASE,
        fixtures.KNOWN_DISTRACTOR_PHRASE,
        fixtures.SCOPED_EXCLUDED_PHRASE,
        fixtures.NO_MATCH_QUERY,
        fixtures.DEGRADED_UNCITED_ANSWER,
    ]

    assert len(set(values)) == len(values)
    assert fixtures.KNOWN_CITED_ANSWER_PHRASE in fixtures.KNOWN_CITED_SOURCE_BODY
    assert fixtures.SCOPED_EXCLUDED_PHRASE in fixtures.KNOWN_DISTRACTOR_SOURCE_BODY
    assert fixtures.KNOWN_CITED_ANSWER_PHRASE not in fixtures.KNOWN_DISTRACTOR_SOURCE_BODY
    assert fixtures.KNOWLEDGE_QA_UAT_MANIFEST_SCHEMA_VERSION == 1


@pytest.mark.unit
def test_seed_helper_dry_run_manifest_uses_fixture_contract() -> None:
    manifest = seed_knowledge_qa_uat.build_dry_run_manifest()

    assert manifest["schemaVersion"] == fixtures.KNOWLEDGE_QA_UAT_MANIFEST_SCHEMA_VERSION
    assert manifest["queries"]["cited"] == fixtures.KNOWN_CITED_QUERY
    assert manifest["expected"]["scopedExcludedPhrase"] == fixtures.SCOPED_EXCLUDED_PHRASE
    assert sorted(manifest["sources"]) == [
        "cited_media",
        "distractor_media",
        "scoped_note",
    ]
    assert all(source["id"] is None for source in manifest["sources"].values())


@pytest.mark.integration
def test_seed_helper_dry_run_cli_writes_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "knowledge-qa-uat.json"
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            "Helper_Scripts/seed_knowledge_qa_uat.py",
            "--dry-run",
            "--manifest",
            str(manifest_path),
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == fixtures.KNOWLEDGE_QA_UAT_MANIFEST_SCHEMA_VERSION
    assert payload["queries"]["cited"] == fixtures.KNOWN_CITED_QUERY
    assert payload["sources"]["cited_media"]["id"] is None
