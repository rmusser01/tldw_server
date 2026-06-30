from __future__ import annotations

from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import (
    LOCAL_DEFAULT_POLICY_DEFINITION,
    STORY_DEFAULT_GENERATION_DEFINITION,
    VNPolicyRepository,
    VNProfileSnapshotRepository,
)
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import _publish_request_matches
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Scripts.service import VNScriptService


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-scripts-publish-test-client")
    yield database
    database.close_connection()


def _program() -> dict:
    return {
        "schema_version": "vn_script_program.v1",
        "title": "Archive Door",
        "primary_asset_pack_id": 7,
        "entry_label": "start",
        "variables": {},
        "generation_defaults": {"profile_id": "story_default", "persist_model_outputs": True},
        "labels": {
            "start": [
                {"op": "set_background", "slot_key": "background.archive.default"},
                {"op": "narrate", "text": "The archive door hums."},
                {"op": "end"},
            ]
        },
    }


def _generation_program_with_profile_key(profile_key: str = "choice_writer") -> dict:
    program = _program()
    program["labels"]["start"].insert(
        1,
        {
            "op": "generate",
            "profile_key": profile_key,
            "output_schema": "choice_set",
            "scope": "turn",
            "on_generated_choice": "after_choice",
        },
    )
    program["labels"]["after_choice"] = [{"op": "narrate", "text": "The choice is made."}, {"op": "end"}]
    return program


def _audio_program() -> dict:
    program = _program()
    program["media_refs"] = {
        "bgm.archive": {
            "generated_file_id": 7001,
            "mime_type": "audio/mpeg",
        }
    }
    program["labels"]["start"].insert(1, {"op": "play_bgm", "media_ref": "bgm.archive"})
    return program


def _manifest(slot_key: str = "background.archive.default") -> dict:
    return {
        "schema_version": "vn_asset_manifest.v1",
        "pack_id": 7,
        "title": "Starter Pack",
        "primary_character_id": 11,
        "content_rating": "general",
        "assets": {
            "backgrounds": [{"slot_key": slot_key, "item_id": 100, "mime_type": "image/png"}],
            "sprites": [],
            "depth_companions": [],
            "cgs": [],
        },
    }


def _create_ready_pack_with_adult_character(chacha_db: CharactersRAGDB) -> int:
    character_id = chacha_db.add_character_card(
        {
            "name": "Adult Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
            "extensions": {"safety_metadata": {"age_status": "adult"}},
        }
    )
    asset_service = VNAssetPackService(chacha_db, owner_user_id=42)
    pack = asset_service.create_pack(VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id))
    slot = asset_service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="background",
            slot_key="background.archive.default",
            variant_count=1,
        ),
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    item = repo.create_item(
        pack_id=pack.id,
        slot_id=slot.id,
        variant_index=0,
        generated_file_id=1001,
        mime_type="image/png",
    )
    asset_service.review_item(item["id"], VNAssetReviewRequest(review_status="approved"))
    return pack.id


def test_publish_snapshots_manifest_and_effective_profiles(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        description=None,
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
    )
    service.replace_draft(script["id"], if_revision=0, draft=_program())

    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="v1",
        idempotency_key="publish-v1",
        acknowledgements=["character_safety_missing"],
    )
    version = service.get_version(script["id"], published["version_id"])
    manifest_snapshot = service.get_manifest_snapshot(script["id"], published["version_id"])

    assert published["status"] == "published"
    assert published["version_number"] == 1
    assert version["manifest_snapshot_id"] == published["manifest_snapshot_id"]
    assert version["policy_snapshot_id"] == published["policy_snapshot_id"]
    assert version["generation_profile_snapshot_id"] == published["generation_profile_snapshot_id"]
    assert manifest_snapshot["manifest"]["assets"]["backgrounds"][0]["slot_key"] == "background.archive.default"

    policy_snapshot = service.profile_snapshots.get_profile_snapshot(
        published["policy_snapshot_id"],
        owner_user_id=42,
    )
    generation_snapshot = service.profile_snapshots.get_profile_snapshot(
        published["generation_profile_snapshot_id"],
        owner_user_id=42,
    )
    assert policy_snapshot["resource_id"] == published["version_id"]
    assert generation_snapshot["resource_id"] == published["version_id"]


def test_create_version_links_all_generation_profile_snapshots(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        description=None,
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
    )
    snapshot_repo = VNProfileSnapshotRepository.initialized(chacha_db)
    policy_snapshot = snapshot_repo.create_profile_snapshot(
        owner_user_id=42,
        snapshot_type="policy",
        profile_id="local_default",
        profile_version=1,
        resource_type="script_version",
        resource_id=None,
        definition=LOCAL_DEFAULT_POLICY_DEFINITION,
    )
    default_generation_snapshot = snapshot_repo.create_profile_snapshot(
        owner_user_id=42,
        snapshot_type="generation",
        profile_id="story_default",
        profile_version=1,
        resource_type="script_version",
        resource_id=None,
        definition=STORY_DEFAULT_GENERATION_DEFINITION,
    )
    choice_generation_snapshot = snapshot_repo.create_profile_snapshot(
        owner_user_id=42,
        snapshot_type="generation",
        profile_id="choice_profile",
        profile_version=1,
        resource_type="script_version",
        resource_id=None,
        definition={**STORY_DEFAULT_GENERATION_DEFINITION, "supported_output_schemas": ["choice_set"]},
    )

    version = service.repo.create_version(
        script_id=script["id"],
        owner_user_id=42,
        label="manual",
        draft_revision=0,
        program=_generation_program_with_profile_key(),
        asset_pack_id=7,
        manifest=_manifest(),
        manifest_hash="manual-manifest-hash",
        policy_snapshot_id=policy_snapshot["id"],
        generation_profile_snapshot_id=default_generation_snapshot["id"],
        generation_profile_snapshots={
            "default": default_generation_snapshot["id"],
            "choice_writer": choice_generation_snapshot["id"],
        },
        script_defaults={},
        validation={"valid": True, "errors": [], "warnings": []},
    )

    for snapshot in (policy_snapshot, default_generation_snapshot, choice_generation_snapshot):
        refreshed = snapshot_repo.get_profile_snapshot(snapshot["id"], owner_user_id=42)
        assert refreshed["resource_id"] == version["id"]


def test_publish_snapshots_resolved_custom_profile_versions(chacha_db: CharactersRAGDB) -> None:
    profile_repo = VNPolicyRepository.initialized(chacha_db)
    policy = profile_repo.create_policy_profile(
        profile_id="custom_local",
        display_name="Custom Local",
        definition=LOCAL_DEFAULT_POLICY_DEFINITION,
    )
    policy = profile_repo.update_policy_profile(
        "custom_local",
        display_name="Custom Local V2",
        definition=LOCAL_DEFAULT_POLICY_DEFINITION,
    )
    generation_definition = dict(STORY_DEFAULT_GENERATION_DEFINITION)
    generation_definition["max_choices"] = 3
    generation = profile_repo.create_generation_profile(
        profile_id="custom_story",
        display_name="Custom Story",
        definition=generation_definition,
    )
    generation_definition_v2 = dict(generation_definition)
    generation_definition_v2["max_choices"] = 2
    generation = profile_repo.update_generation_profile(
        "custom_story",
        display_name="Custom Story V2",
        definition=generation_definition_v2,
    )
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    program = _program()
    program["generation_defaults"]["profile_id"] = "custom_story"
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="custom_local",
        generation_profile_id="custom_story",
        content_rating="general",
    )
    service.replace_draft(
        script["id"],
        if_revision=0,
        draft=program,
        policy_profile=policy,
        generation_profile=generation,
    )

    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="v2",
        idempotency_key="publish-v2",
        acknowledgements=["character_safety_missing"],
        policy_profile=policy,
        generation_profile=generation,
    )
    policy_snapshot = service.profile_snapshots.get_profile_snapshot(
        published["policy_snapshot_id"],
        owner_user_id=42,
    )
    generation_snapshot = service.profile_snapshots.get_profile_snapshot(
        published["generation_profile_snapshot_id"],
        owner_user_id=42,
    )

    assert policy_snapshot["profile_id"] == "custom_local"
    assert policy_snapshot["profile_version"] == 2
    assert generation_snapshot["profile_id"] == "custom_story"
    assert generation_snapshot["profile_version"] == 2
    assert generation_snapshot["definition"]["max_choices"] == 2


def test_publish_stores_default_and_additional_generation_profile_snapshots(chacha_db: CharactersRAGDB) -> None:
    profile_repo = VNPolicyRepository.initialized(chacha_db)
    choice_definition = dict(STORY_DEFAULT_GENERATION_DEFINITION)
    choice_definition["supported_output_schemas"] = ["choice_set"]
    choice_profile = profile_repo.create_generation_profile(
        profile_id="choice_profile",
        display_name="Choice Writer",
        definition=choice_definition,
    )
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        generation_profiles={"choice_writer": "choice_profile"},
        content_rating="general",
    )
    service.replace_draft(
        script["id"],
        if_revision=0,
        draft=_generation_program_with_profile_key(),
        generation_profiles={"choice_writer": choice_profile},
    )

    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="profile-map",
        idempotency_key="publish-profile-map",
        acknowledgements=["character_safety_missing"],
        generation_profiles={"choice_writer": choice_profile},
    )
    version = service.get_version(script["id"], published["version_id"])

    assert script["generation_profiles"] == {
        "default": "story_default",
        "choice_writer": "choice_profile",
    }
    assert set(published["generation_profile_snapshots"]) == {"default", "choice_writer"}
    assert version["generation_profile_snapshots"] == published["generation_profile_snapshots"]
    assert version["generation_profile_snapshot_id"] == published["generation_profile_snapshots"]["default"]

    choice_snapshot = service.profile_snapshots.get_profile_snapshot(
        published["generation_profile_snapshots"]["choice_writer"],
        owner_user_id=42,
    )
    assert choice_snapshot["profile_id"] == "choice_profile"
    assert choice_snapshot["resource_id"] == published["version_id"]


def test_publish_idempotency_replays_same_key_after_profile_map_changes(chacha_db: CharactersRAGDB) -> None:
    profile_repo = VNPolicyRepository.initialized(chacha_db)
    choice_definition = dict(STORY_DEFAULT_GENERATION_DEFINITION)
    choice_definition["supported_output_schemas"] = ["choice_set"]
    choice_profile = profile_repo.create_generation_profile(
        profile_id="choice_profile",
        display_name="Choice Writer",
        definition=choice_definition,
    )
    alt_choice_profile = profile_repo.create_generation_profile(
        profile_id="choice_profile_alt",
        display_name="Choice Writer Alt",
        definition=choice_definition,
    )
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        generation_profiles={"choice_writer": "choice_profile"},
        content_rating="general",
    )
    service.replace_draft(
        script["id"],
        if_revision=0,
        draft=_generation_program_with_profile_key(),
        generation_profiles={"choice_writer": choice_profile},
    )
    first = service.publish_script(
        script["id"],
        draft_revision=1,
        label="profile-map",
        idempotency_key="publish-profile-map",
        acknowledgements=["character_safety_missing"],
        generation_profiles={"choice_writer": choice_profile},
    )
    service.update_script(script["id"], {"generation_profiles": {"choice_writer": "choice_profile_alt"}})

    replayed = service.publish_script(
        script["id"],
        draft_revision=1,
        label="profile-map",
        idempotency_key="publish-profile-map",
        acknowledgements=["character_safety_missing"],
        generation_profiles={"choice_writer": alt_choice_profile},
    )

    assert replayed == first


def test_publish_request_match_rejects_missing_payload_when_existing_is_structured() -> None:
    existing = {
        "request_payload": {"draft_revision": 1, "label": "v1"},
        "payload_hash": "legacy-hash",
    }

    assert _publish_request_matches(
        existing,
        request_payload=None,
        legacy_payload_hash="legacy-hash",
    ) is False


def test_service_validation_resolves_declared_audio_refs(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
        audio_ref_resolver=lambda program: {"bgm.archive": {"mime_type": "audio/mpeg", "generated_file_id": 7001}},
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
    )

    valid_result = service.validate_draft(script["id"], _audio_program())
    invalid_program = _audio_program()
    invalid_program["media_refs"]["bgm.archive"]["mime_type"] = "image/png"
    untrusted_service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    inaccessible_result = untrusted_service.validate_draft(script["id"], invalid_program)

    assert valid_result["valid"] is True
    assert inaccessible_result["valid"] is False
    assert inaccessible_result["errors"][0]["code"] == "audio_media_ref_inaccessible"


def test_service_validation_rejects_script_metadata_mismatch(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
    )
    program = _program()
    program["primary_asset_pack_id"] = 8
    program["generation_defaults"]["profile_id"] = "other_story_profile"

    result = service.validate_draft(script["id"], program)

    assert result["valid"] is False
    assert {error["code"] for error in result["errors"]} >= {
        "primary_asset_pack_mismatch",
        "generation_profile_mismatch",
    }


def test_service_validation_treats_blocking_policy_as_invalid(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="strict_hosted",
        generation_profile_id="story_default",
        content_rating="general",
    )

    result = service.validate_draft(script["id"], _program())

    assert result["valid"] is False
    assert "policy_character_safety_missing" in {error["code"] for error in result["errors"]}


def test_publish_uses_selected_pack_character_safety_metadata(chacha_db: CharactersRAGDB) -> None:
    pack_id = _create_ready_pack_with_adult_character(chacha_db)
    service = VNScriptService(chacha_db, owner_user_id=42)
    program = _program()
    program["primary_asset_pack_id"] = pack_id
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=pack_id,
        policy_profile_id="strict_hosted",
        generation_profile_id="story_default",
        content_rating="general",
    )
    service.replace_draft(script["id"], if_revision=0, draft=program)

    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="adult",
        idempotency_key="publish-adult",
        acknowledgements=[],
    )

    assert published["status"] == "published"


def test_version_policy_evaluate_uses_published_policy_snapshot(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
    )
    service.replace_draft(script["id"], if_revision=0, draft=_program())
    published = service.publish_script(
        script["id"],
        draft_revision=1,
        label="v1",
        idempotency_key="publish-v1",
        acknowledgements=["character_safety_missing"],
    )
    service.update_script(
        script["id"],
        {"policy_profile_id": "strict_hosted", "content_rating": "mature"},
    )

    result = service.evaluate_version_policy(
        script["id"],
        published["version_id"],
        context={"character_safety": {"metadata_status": "adult"}},
    )

    assert result["decision"] == "allow"
    assert result["profile_id"] == "local_default"


def test_publish_replays_same_idempotency_key_and_rejects_payload_conflict(
    chacha_db: CharactersRAGDB,
) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="general",
    )
    service.replace_draft(script["id"], if_revision=0, draft=_program())

    first = service.publish_script(
        script["id"],
        draft_revision=1,
        label="v1",
        idempotency_key="publish-v1",
        acknowledgements=["character_safety_missing"],
    )
    replayed = service.publish_script(
        script["id"],
        draft_revision=1,
        label="v1",
        idempotency_key="publish-v1",
        acknowledgements=["character_safety_missing"],
    )

    assert replayed == first

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        service.publish_script(
            script["id"],
            draft_revision=1,
            label="different-label",
            idempotency_key="publish-v1",
            acknowledgements=["character_safety_missing"],
        )


def test_publish_repeats_authoritative_policy_evaluation(chacha_db: CharactersRAGDB) -> None:
    service = VNScriptService(
        chacha_db,
        owner_user_id=42,
        manifest_resolver=lambda asset_pack_id: _manifest(),
    )
    script = service.create_script(
        title="Mature Archive",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
        content_rating="mature",
    )
    service.replace_draft(script["id"], if_revision=0, draft=_program())

    with pytest.raises(ValueError, match="script_publish_policy_blocked"):
        service.publish_script(
            script["id"],
            draft_revision=1,
            label="v1",
            idempotency_key="publish-v1",
        )
