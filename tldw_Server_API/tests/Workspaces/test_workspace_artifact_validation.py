from __future__ import annotations

import json

import pytest
from hypothesis import given, settings, strategies as st

from tldw_Server_API.app.core.exceptions import WorkspaceArtifactExportStateError
from tldw_Server_API.app.core.Workspaces.artifact_validation import (
    validate_workspace_artifact_for_export,
)
from tldw_Server_API.app.core.Workspaces.workspace_artifact_exports import (
    export_workspace_artifact_version,
)


GENERATED_ARTIFACT_TYPES = (
    "quiz",
    "flashcards",
    "audio_summary",
    "adaptive_table",
    "slides",
    "mindmap",
)

PLACEHOLDER_CONTENT = (
    "slides go here",
    "invalid",
    "this is a test",
)


@pytest.mark.parametrize("artifact_type", GENERATED_ARTIFACT_TYPES)
@pytest.mark.parametrize("content", PLACEHOLDER_CONTENT)
def test_generated_artifact_placeholders_are_rejected(artifact_type: str, content: str) -> None:
    artifact = {
        "id": f"{artifact_type}-1",
        "workspace_id": "ws-1",
        "artifact_type": artifact_type,
        "title": artifact_type,
        "review_state": "accepted",
        "content": content,
    }

    with pytest.raises(
        WorkspaceArtifactExportStateError,
        match="workspace_artifact_placeholder_content",
    ):
        validate_workspace_artifact_for_export(artifact)


@settings(max_examples=30)
@given(
    artifact_type=st.sampled_from(GENERATED_ARTIFACT_TYPES),
    content=st.sampled_from(PLACEHOLDER_CONTENT),
    prefix=st.sampled_from(("", "  ", "# ", "```markdown\n")),
    suffix=st.sampled_from(("", "  ", "\n", "\n```")),
)
def test_placeholder_rejection_is_case_and_wrapper_insensitive(
    artifact_type: str,
    content: str,
    prefix: str,
    suffix: str,
) -> None:
    artifact = {
        "artifact_type": artifact_type,
        "review_state": "accepted",
        "content": f"{prefix}{content.upper()}{suffix}",
    }

    with pytest.raises(
        WorkspaceArtifactExportStateError,
        match="workspace_artifact_placeholder_content",
    ):
        validate_workspace_artifact_for_export(artifact)


@pytest.mark.parametrize("artifact_type", GENERATED_ARTIFACT_TYPES)
def test_substantive_generated_artifacts_pass_with_claims_metadata(artifact_type: str) -> None:
    artifact = {
        "id": f"{artifact_type}-1",
        "workspace_id": "ws-1",
        "artifact_type": artifact_type,
        "title": artifact_type,
        "review_state": "accepted",
        "content": "# Findings\n- The source pack supports this answer with cited evidence.",
        "producer_metadata": {
            "claims_validation_required": True,
            "claims_validator_model": "llama.cpp/local",
        },
        "review_metadata": {
            "verification_summary": {
                "validator": "claims_source_pack_v1",
                "model": "llama.cpp/local",
                "unsupported_claim_count": 0,
            }
        },
    }

    metadata = validate_workspace_artifact_for_export(artifact)

    assert metadata["status"] == "passed"
    assert metadata["claims_validation_required"] is True
    assert metadata["claims_validation"]["validator"] == "claims_source_pack_v1"
    assert metadata["claims_validation"]["model"] == "llama.cpp/local"
    assert metadata["claims_validation"]["unsupported_claim_count"] == 0


def test_claims_required_artifact_rejects_missing_validation_report() -> None:
    artifact = {
        "artifact_type": "slides",
        "review_state": "accepted",
        "content": "# Findings\n- The source pack supports this answer.",
        "producer_metadata": {"claims_validation_required": True},
    }

    with pytest.raises(
        WorkspaceArtifactExportStateError,
        match="workspace_artifact_claims_validation_missing",
    ):
        validate_workspace_artifact_for_export(artifact)


def test_numeric_claims_required_flag_is_enforced() -> None:
    artifact = {
        "artifact_type": "slides",
        "review_state": "accepted",
        "content": "# Findings\n- The source pack supports this answer.",
        "producer_metadata": {"claims_validation_required": 1},
    }

    with pytest.raises(
        WorkspaceArtifactExportStateError,
        match="workspace_artifact_claims_validation_missing",
    ):
        validate_workspace_artifact_for_export(artifact)


def test_short_real_content_that_mentions_invalid_is_allowed() -> None:
    artifact = {
        "artifact_type": "quiz",
        "review_state": "accepted",
        "content": "# Quiz\nWhich cited source says the claim is invalid?",
    }

    metadata = validate_workspace_artifact_for_export(artifact)

    assert metadata["status"] == "passed"


def test_non_latin_generated_artifact_content_is_not_treated_as_empty() -> None:
    artifact = {
        "artifact_type": "slides",
        "review_state": "accepted",
        "content": "# 調査結果\n- 出典に基づく要約です。",
    }

    metadata = validate_workspace_artifact_for_export(artifact)

    assert metadata["status"] == "passed"


def test_validation_metadata_copies_and_flattens_warning_sequences() -> None:
    warnings = ["needs source review"]
    artifact = {
        "artifact_type": "slides",
        "review_state": "accepted",
        "content": "# Findings\n- The source pack supports this answer.",
        "review_metadata": {
            "verification_summary": {
                "validator": "claims_source_pack_v1",
                "unsupported_claim_count": 0,
                "warnings": warnings,
            }
        },
    }

    metadata = validate_workspace_artifact_for_export(artifact)

    assert metadata["claims_validation"]["warnings"] == ["needs source review"]
    assert metadata["claims_validation"]["warnings"] is not warnings

    artifact["review_metadata"]["verification_summary"]["warnings"] = ("first", "second")
    metadata = validate_workspace_artifact_for_export(artifact)
    assert metadata["claims_validation"]["warnings"] == ["first", "second"]


def test_claims_validation_metadata_preserves_explicit_non_pass_status() -> None:
    artifact = {
        "artifact_type": "slides",
        "review_state": "accepted",
        "content": "# Findings\n- The source pack supports this answer.",
        "review_metadata": {
            "verification_summary": {
                "validator": "claims_source_pack_v1",
                "unsupported_claim_count": 0,
                "status": "skipped",
            }
        },
    }

    metadata = validate_workspace_artifact_for_export(artifact)

    assert metadata["claims_validation"]["status"] == "skipped"


def test_validator_rejects_non_accepted_artifacts_directly() -> None:
    artifact = {
        "artifact_type": "slides",
        "review_state": "needs_revision",
        "content": "# Findings\n- The source pack supports this answer.",
    }

    with pytest.raises(
        WorkspaceArtifactExportStateError,
        match="workspace_artifact_not_accepted",
    ):
        validate_workspace_artifact_for_export(artifact)


def test_claims_required_artifact_rejects_unresolved_unsupported_claims() -> None:
    artifact = {
        "artifact_type": "quiz",
        "review_state": "accepted",
        "content": "# Quiz\n1. Which result was supported by the source pack?",
        "producer_metadata": {"claims_validation_required": True},
        "review_metadata": {
            "verification_summary": {
                "validator": "claims_source_pack_v1",
                "model": "llama.cpp/local",
                "unsupported_claim_count": 1,
            }
        },
    }

    with pytest.raises(
        WorkspaceArtifactExportStateError,
        match="workspace_artifact_claims_validation_failed",
    ):
        validate_workspace_artifact_for_export(artifact)


def test_export_metadata_includes_validation_result() -> None:
    artifact = {
        "id": "slides-1",
        "workspace_id": "ws-1",
        "artifact_type": "slides",
        "title": "Supported Slides",
        "review_state": "accepted",
        "content": "# Findings\n- The source pack supports this answer.",
        "producer_metadata": {"claims_validation_required": True},
        "review_metadata": {
            "verification_summary": {
                "validator": "claims_source_pack_v1",
                "model": "llama.cpp/local",
                "unsupported_claim_count": 0,
            }
        },
    }

    payload = export_workspace_artifact_version(
        artifact,
        export_format="json",
        generated_at="2026-07-05T00:00:00+00:00",
    )
    exported_json = json.loads(payload["content"])

    validation = exported_json["metadata"]["artifact_validation"]
    assert validation["status"] == "passed"
    assert validation["claims_validation_required"] is True
    assert validation["claims_validation"]["unsupported_claim_count"] == 0
