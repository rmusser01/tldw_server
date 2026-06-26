from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_filesystem_digest_is_stable_for_sorted_paths() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    first = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={
            "SKILL.md": b"hello\r\n",
            "refs/notes.md": b"reference",
        },
        metadata={"context": "inline"},
    )
    second = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={
            "refs/notes.md": b"reference",
            "SKILL.md": b"hello\r\n",
        },
        metadata={"context": "inline"},
    )

    assert first == second
    assert first.startswith("sha256:")


def test_filesystem_digest_normalizes_unicode_paths() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    composed = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={"refs/caf\u00e9.md": b"reference"},
    )
    decomposed = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:user:1/demo",
        files={"refs/cafe\u0301.md": b"reference"},
    )

    assert composed == decomposed


def test_filesystem_digest_rejects_duplicate_canonical_paths() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    with pytest.raises(ValueError, match="Duplicate canonical file path"):
        canonical_filesystem_digest(
            source_type="skill_file",
            asset_id="skill:user:1/demo",
            files={
                "refs/caf\u00e9.md": b"one",
                "refs/cafe\u0301.md": b"two",
            },
        )


def test_filesystem_digest_matches_golden_payload() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    digest = canonical_filesystem_digest(
        source_type="skill_file",
        asset_id="skill:system/context-integrity",
        files={
            "SKILL.md": b"# Context Integrity\nVerify prompt-bearing assets.\n",
            "references/checklist.md": b"- inventory\n- verify\n",
        },
        metadata={"owner_scope": "system", "executable": False},
    )

    assert digest == ("sha256:cdc83bc127d0f95e66b891c1be48326e35694860185cef18f5d8af634939f1a3")


def test_filesystem_digest_detects_formatting_edits() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    original = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id="prompt_file:rag.prompts.yaml",
        files={"rag.prompts.yaml": b"answer: one\n"},
    )
    edited = canonical_filesystem_digest(
        source_type="prompt_file",
        asset_id="prompt_file:rag.prompts.yaml",
        files={"rag.prompts.yaml": b"answer: one\n# changed\n"},
    )

    assert original != edited


def test_filesystem_digest_rejects_non_string_metadata_keys() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_filesystem_digest,
    )

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        canonical_filesystem_digest(
            source_type="skill_file",
            asset_id="skill:user:1/demo",
            files={"SKILL.md": b"hello\n"},
            metadata={1: "integer key"},
        )


def test_db_prompt_digest_normalizes_unicode_and_line_endings() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    composed = canonical_db_prompt_digest(
        {
            "uuid": "prompt-1",
            "version": 3,
            "name": "Cafe",
            "system": "caf\u00e9\r\nline",
            "user": "body",
            "structured": {"b": 2, "a": 1},
        }
    )
    decomposed = canonical_db_prompt_digest(
        {
            "structured": {"a": 1, "b": 2},
            "user": "body",
            "system": "cafe\u0301\nline",
            "name": "Cafe",
            "version": 3,
            "uuid": "prompt-1",
        }
    )

    assert composed == decomposed
    payload = json.loads(composed.canonical_json)
    assert payload["system"] == "caf\u00e9\nline"


def test_db_prompt_digest_matches_golden_payload() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    digest = canonical_db_prompt_digest(
        {
            "uuid": "prompt-golden-1",
            "version": 7,
            "name": "Golden Prompt",
            "system": "Use source context only.\r\nReturn concise answers.",
            "user": "Question: {question}",
            "structured": {
                "temperature": 0.2,
                "tags": ["rag", "verified"],
            },
        }
    )

    assert digest.digest == ("sha256:0b3be0310f6cf7ae29f018ef4a79bbe62e4f6f620b006aedacbb1255d6712234")


@pytest.mark.parametrize("bad_float", [float("nan"), float("inf"), float("-inf")])
def test_db_prompt_digest_rejects_non_finite_floats(bad_float: float) -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    with pytest.raises(ValueError, match="Non-finite floats"):
        canonical_db_prompt_digest({"uuid": "prompt-1", "score": bad_float})


def test_db_prompt_digest_rejects_non_string_mapping_keys() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        canonical_db_prompt_digest({"uuid": "prompt-1", "structured": {1: "integer key"}})


def test_db_prompt_digest_rejects_unsupported_json_values() -> None:
    from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
        canonical_db_prompt_digest,
    )

    with pytest.raises(TypeError, match="Unsupported canonical JSON value"):
        canonical_db_prompt_digest({"uuid": "prompt-1", "unsupported": object()})


def test_context_integrity_models_freeze_mapping_fields() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextAssetDescriptor,
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )

    metadata = {"context": "inline", "nested": {"source": "caller"}}
    details = {"reason": "changed", "nested": {"source": "caller"}}
    approved = {"asset-1": "sha256:abc"}

    descriptor = ContextAssetDescriptor(
        asset_id="skill:user:1/demo",
        source_type="skill_file",
        digest="sha256:123",
        display_name="Demo skill",
        metadata=metadata,
    )
    finding = ContextIntegrityFinding(
        asset_id="skill:user:1/demo",
        state="trusted",
        severity="info",
        summary="Verified",
        remediation="None",
        source_type="skill_file",
        details=details,
    )
    findings = [finding]
    boot_state = ContextIntegrityBootState(
        mode="audit_only",
        degraded=False,
        manifest_sequence=1,
        manifest_digest="sha256:def",
        approved_digests_by_asset_id=approved,
        findings=findings,
    )

    metadata["context"] = "changed"
    metadata["nested"]["source"] = "changed"
    details["reason"] = "changed again"
    details["nested"]["source"] = "changed"
    approved["asset-1"] = "sha256:changed"
    findings.append(
        ContextIntegrityFinding(
            asset_id="skill:user:1/other",
            state="new_unapproved",
            severity="warning",
            summary="New",
            remediation="Review",
            source_type="skill_file",
        )
    )

    assert descriptor.metadata["context"] == "inline"
    assert descriptor.metadata["nested"]["source"] == "caller"
    assert finding.details["reason"] == "changed"
    assert finding.details["nested"]["source"] == "caller"
    assert boot_state.approved_digests_by_asset_id["asset-1"] == "sha256:abc"
    assert boot_state.findings == (finding,)

    with pytest.raises(TypeError):
        descriptor.metadata["new"] = "value"
    with pytest.raises(TypeError):
        finding.details["new"] = "value"
    with pytest.raises(TypeError):
        boot_state.approved_digests_by_asset_id["asset-2"] = "sha256:new"
