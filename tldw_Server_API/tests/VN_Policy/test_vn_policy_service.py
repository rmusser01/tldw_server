from __future__ import annotations

from collections.abc import Generator

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.vn_policy_schemas import VNGenerationProfileCreate
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VN_Policy.service import (
    VNPolicyService,
    evaluate_character_safety_definition,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-policy-service-test-client")
    yield database
    database.close_connection()


@pytest.fixture
def service(chacha_db: CharactersRAGDB) -> VNPolicyService:
    return VNPolicyService(chacha_db, owner_user_id=42)


@pytest.mark.parametrize(
    ("content_rating", "metadata_status", "profile_id", "decision", "requires_acknowledgement"),
    [
        ("general", "missing", "local_default", "warn", True),
        ("general", "unknown_or_ambiguous", "local_default", "warn", True),
        ("mature", "missing", "local_default", "block", False),
        ("mature", "unknown_or_ambiguous", "local_default", "block", False),
        ("general", "conflicting", "local_default", "block", False),
        ("general", "imported_untrusted", "local_default", "warn", True),
        ("general", "missing", "strict_hosted", "block", False),
        ("general", "unknown_or_ambiguous", "strict_hosted", "block", False),
        ("general", "conflicting", "strict_hosted", "block", False),
        ("general", "imported_untrusted", "strict_hosted", "block", False),
    ],
)
@pytest.mark.asyncio
async def test_character_safety_metadata_policy_matrix(
    service: VNPolicyService,
    content_rating: str,
    metadata_status: str,
    profile_id: str,
    decision: str,
    requires_acknowledgement: bool,
) -> None:
    result = await service.evaluate_character_safety_metadata(
        content_rating=content_rating,
        metadata_status=metadata_status,
        policy_profile_id=profile_id,
    )

    assert result["decision"] == decision
    assert result["blocked"] is (decision == "block")
    assert result["requires_acknowledgement"] is requires_acknowledgement
    assert result["profile_id"] == profile_id
    assert result["reasons"][0]["code"].startswith("character_safety_")


def test_generation_profile_schema_rejects_unsupported_bounds() -> None:
    with pytest.raises(ValidationError):
        VNGenerationProfileCreate(
            profile_id="bad_choices",
            display_name="Bad Choices",
            provider="local",
            model="gemma-3-12b",
            supports_structured_output=True,
            temperature_default=0.7,
            temperature_min=0,
            temperature_max=1,
            max_output_tokens=1024,
            allowed_content_ratings=["general"],
            max_choices=0,
            max_branch_depth=8,
            max_model_expansion_scope="scene",
            tts_allowed=True,
            output_persistence_max_days=30,
            audit_mode="metadata",
        )


@pytest.mark.asyncio
async def test_minor_metadata_allows_general_content(service: VNPolicyService) -> None:
    result = await service.evaluate_character_safety_metadata(
        content_rating="general",
        metadata_status="minor",
        policy_profile_id="local_default",
    )

    assert result["decision"] == "allow"
    assert result["reasons"] == []


def test_policy_definition_can_disable_warning_acknowledgement() -> None:
    result = evaluate_character_safety_definition(
        profile_definition={
            "character_safety": {
                "missing": {"general": "warn", "mature": "block"},
                "unknown_or_ambiguous": {"general": "warn", "mature": "block"},
                "conflicting": {"default": "block"},
                "imported_untrusted": {"general": "warn", "mature": "block"},
            },
            "acknowledgement_required_for_warnings": False,
        },
        policy_profile_id="no_ack",
        content_rating="general",
        metadata_status="missing",
    )

    assert result["decision"] == "warn"
    assert result["requires_acknowledgement"] is False
    assert result["reasons"][0]["requires_acknowledgement"] is False


@pytest.mark.asyncio
async def test_snapshot_creation_resolves_effective_policy_and_generation_profiles(
    service: VNPolicyService,
) -> None:
    snapshot_pair = await service.create_profile_snapshots(
        resource_type="script_version",
        resource_id=12,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
    )

    policy_snapshot = service.snapshot_repo.get_profile_snapshot(
        snapshot_pair["policy_snapshot_id"],
        owner_user_id=42,
    )
    generation_snapshot = service.snapshot_repo.get_profile_snapshot(
        snapshot_pair["generation_snapshot_id"],
        owner_user_id=42,
    )

    assert policy_snapshot["snapshot_type"] == "policy"
    assert policy_snapshot["profile_id"] == "local_default"
    assert generation_snapshot["snapshot_type"] == "generation"
    assert generation_snapshot["profile_id"] == "story_default"


@pytest.mark.asyncio
async def test_evaluate_treats_omitted_character_safety_as_missing_metadata(service: VNPolicyService) -> None:
    result = await service.evaluate(
        target_type="session_setup",
        target_id=None,
        policy_profile_id="local_default",
        context={"content_rating": "general"},
    )

    assert result["decision"] == "warn"
    assert result["reasons"][0]["code"] == "character_safety_missing"


@pytest.mark.asyncio
async def test_evaluate_strict_hosted_blocks_omitted_character_safety(service: VNPolicyService) -> None:
    result = await service.evaluate(
        target_type="session_setup",
        target_id=None,
        policy_profile_id="strict_hosted",
        context={"content_rating": "general"},
    )

    assert result["decision"] == "block"
    assert result["reasons"][0]["code"] == "character_safety_missing"


@pytest.mark.asyncio
async def test_evaluate_rejects_target_id_without_authoritative_resolver(service: VNPolicyService) -> None:
    with pytest.raises(ValueError, match="target_resolution_unavailable"):
        await service.evaluate(
            target_type="script_draft",
            target_id=17,
            policy_profile_id="local_default",
            context={
                "content_rating": "general",
                "character_safety": {"metadata_status": "adult"},
            },
        )
