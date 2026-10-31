"""Direct package tests for every public profile-core contract layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tldw_profile_core import (
    PreferencePayload,
    ProfileManifest,
    ProfileSearchRequest,
    ProfileToolResult,
    ToolOperation,
    ToolResultStatus,
    canonical_bytes,
    export_json_schema,
    validate_profile_semantics,
)


def _manifest() -> ProfileManifest:
    now = datetime(2026, 8, 31, tzinfo=UTC)
    return ProfileManifest(
        profile_id="profile-a",
        revision=0,
        purge_generation=0,
        created_at=now,
        updated_at=now,
        current_version_id="manifest-v1",
    )


def test_canonical_models_and_semantics_share_one_portable_contract() -> None:
    manifest = _manifest()
    encoded = canonical_bytes(manifest)

    assert ProfileManifest.model_validate_json(encoded) == manifest
    validate_profile_semantics(manifest.model_dump(mode="json"))
    assert b"\n" not in encoded


def test_payloads_and_model_requests_reject_blank_content() -> None:
    assert PreferencePayload(subject="format", polarity="like", value="brief").kind == "preference"
    assert ProfileSearchRequest(query="format").limit == 5
    with pytest.raises(ValidationError):
        PreferencePayload(subject=" ", polarity="like", value="brief")


def test_schema_export_contains_profile_models_and_semantic_vocabulary(tmp_path) -> None:
    destination = tmp_path / "personal-context-schema.json"

    export_json_schema(destination)

    schema = json.loads(destination.read_text(encoding="utf-8"))
    assert "ProfileManifest" in schema["$defs"]
    assert "x-tldw-profile-semantics" in schema


def test_tool_contract_is_typed_and_frozen() -> None:
    result = ProfileToolResult(
        operation=ToolOperation.SEARCH,
        status=ToolResultStatus.APPLIED,
        message="Found one matching preference.",
    )

    assert result.operation is ToolOperation.SEARCH
    with pytest.raises(ValidationError):
        ProfileToolResult(operation="unknown", status="applied", message="result")
