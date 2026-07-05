from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Visual_Identities.service import VisualIdentityService


pytestmark = pytest.mark.unit

OWNER_USER_ID = 42


@pytest.fixture()
def chacha_db(tmp_path: Path):
    database = CharactersRAGDB(
        tmp_path / "visual_identity_service.sqlite",
        client_id="visual-identity-service-test-client",
    )
    yield database
    database.close_connection()


@pytest.fixture()
def repo(chacha_db: CharactersRAGDB) -> VisualIdentityRepository:
    return VisualIdentityRepository.initialized(chacha_db)


@pytest.fixture()
def service(chacha_db: CharactersRAGDB) -> VisualIdentityService:
    return VisualIdentityService(chacha_db, owner_user_id=OWNER_USER_ID)


def test_service_constructor_does_not_initialize_repository_schema(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_initialized(cls, db):
        raise AssertionError("schema initialization should be explicit")

    monkeypatch.setattr(
        VisualIdentityRepository,
        "initialized",
        classmethod(fail_initialized),
    )

    service = VisualIdentityService(chacha_db, owner_user_id=OWNER_USER_ID)

    assert isinstance(service.repository, VisualIdentityRepository)


def test_service_create_pack_has_no_active_version_until_activation(
    service: VisualIdentityService,
) -> None:
    pack = service.create_pack(title="Manual Expressions")

    assert pack["status"] == "active"
    assert pack["active_version_id"] is None
    assert pack["default_expression_key"] == "neutral"


def test_service_owns_visual_identity_idempotency_workflow(
    service: VisualIdentityService,
) -> None:
    first = service.claim_or_replay_idempotency(
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
    )

    assert first.replay_response is None
    assert first.claim_token is not None

    with pytest.raises(BadRequestError, match="idempotency_key_in_progress"):
        service.claim_or_replay_idempotency(
            scope="visual_identity_import",
            resource_id="pack:1",
            idempotency_key="import-1",
            payload_hash="hash-a",
        )

    service.record_idempotency_response(
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
        response={"draft_id": 9},
        claim_token=first.claim_token,
    )
    replay = service.claim_or_replay_idempotency(
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
    )

    assert replay.replay_response == {"draft_id": 9}
    assert replay.claim_token is None

    with pytest.raises(BadRequestError, match="idempotency_key_conflict"):
        service.claim_or_replay_idempotency(
            scope="visual_identity_import",
            resource_id="pack:1",
            idempotency_key="import-1",
            payload_hash="hash-b",
        )

    released = service.claim_or_replay_idempotency(
        scope="visual_identity_import",
        resource_id="pack:2",
        idempotency_key="import-2",
        payload_hash="hash-c",
    )
    assert released.claim_token is not None
    service.release_idempotency_claim(
        scope="visual_identity_import",
        resource_id="pack:2",
        idempotency_key="import-2",
        claim_token=released.claim_token,
    )
    reclaimed = service.claim_or_replay_idempotency(
        scope="visual_identity_import",
        resource_id="pack:2",
        idempotency_key="import-2",
        payload_hash="hash-c",
    )

    assert reclaimed.claim_token
    assert reclaimed.claim_token != released.claim_token


def test_resolve_override_version_without_pack_raises_domain_error(
    chacha_db: CharactersRAGDB,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db)

    with pytest.raises(BadRequestError, match="pack_not_found"):
        service.resolve_expression_asset(
            actor_kind="character",
            actor_id=character_id,
            requested_expression_key="happy",
            override_pack_version_id=10,
        )


def test_resolve_prefers_manual_override_over_mood(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db)
    draft = _create_ready_draft(repo, assets=("neutral", "happy", "angry"))
    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    version_assets = {
        asset["expression_key"]: asset
        for asset in repo.list_assets_for_version(
            activation.pack_version_id,
            owner_user_id=OWNER_USER_ID,
        )
    }

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="sad",
        manual_override_expression_key="happy",
        mood_expression_key="angry",
    )

    assert resolved.fallback_reason == "manual_override"
    assert resolved.expression_key == "happy"
    assert resolved.asset_id == version_assets["happy"]["id"]
    assert resolved.storage_relpath == "visual_identities/happy.png"


def test_activation_binds_pack_to_character_by_default(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Bound Character")
    draft = _create_ready_draft(repo, assets=("neutral", "happy"))

    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )

    pack = repo.get_pack(activation.pack_id, owner_user_id=OWNER_USER_ID)
    assert pack is not None
    assert pack["status"] == "active"
    assert pack["active_version_id"] == activation.pack_version_id
    assert repo.get_draft(draft["id"], owner_user_id=OWNER_USER_ID)["status"] == "activated"

    binding = repo.get_binding_for_actor(
        owner_user_id=OWNER_USER_ID,
        actor_kind="character",
        actor_id=character_id,
    )
    assert binding is not None
    assert binding["pack_id"] == activation.pack_id
    assert binding["active_version_id"] == activation.pack_version_id


def test_activation_binds_pack_to_persona_uuid_string(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    persona_id = chacha_db.create_persona_profile(
        {"user_id": str(OWNER_USER_ID), "name": "Persona UUID Target"}
    )
    draft = _create_ready_draft(repo, assets=("neutral", "excited"))

    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="persona",
        actor_id=persona_id,
    )
    binding = repo.get_binding_for_actor(
        owner_user_id=OWNER_USER_ID,
        actor_kind="persona",
        actor_id=persona_id,
    )
    resolved = service.resolve_expression_asset(
        actor_kind="persona",
        actor_id=persona_id,
        requested_expression_key="excited",
    )

    assert binding is not None
    assert binding["pack_id"] == activation.pack_id
    assert str(binding["actor_id"]) == persona_id
    assert resolved.actor_id == persona_id
    assert resolved.fallback_reason == "requested"
    assert resolved.expression_key == "excited"
    assert resolved.asset_id is not None


def test_resolver_explicit_override_resolves_requested_expression(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Override Character")
    binding_draft = _create_ready_draft(repo, assets=("neutral",), title="Binding Pack")
    service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    override_pack, override_version, override_assets = _create_versioned_pack(
        repo,
        title="Override Pack",
        assets=("happy",),
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="happy",
        role_id="hero",
        role_label="Hero",
        override_pack_id=override_pack["id"],
        override_pack_version_id=override_version["id"],
    )

    assert resolved.pack_id == override_pack["id"]
    assert resolved.pack_version_id == override_version["id"]
    assert resolved.asset_id == override_assets["happy"]["id"]
    assert resolved.resolution_source == "override"
    assert resolved.role_id == "hero"
    assert resolved.role_label == "Hero"


def test_resolver_override_missing_expression_is_strict_by_default(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Strict Override Character")
    binding_draft = _create_ready_draft(repo, assets=("sad",), title="Strict Binding Pack")
    service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    override_pack, override_version, _ = _create_versioned_pack(
        repo,
        title="Strict Override Pack",
        assets=("angry",),
        default_expression_key="missing-default",
    )

    with pytest.raises(ValueError, match="^override_expression_missing$"):
        service.resolve_expression_asset(
            actor_kind="character",
            actor_id=character_id,
            requested_expression_key="sad",
            override_pack_id=override_pack["id"],
            override_pack_version_id=override_version["id"],
        )


def test_resolver_rejects_override_pack_version_mismatch(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Mismatch Override Character")
    binding_draft = _create_ready_draft(repo, assets=("happy",), title="Mismatch Binding Pack")
    service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    override_pack, _, _ = _create_versioned_pack(
        repo,
        title="Mismatch Override Pack",
        assets=("sad",),
    )
    _, other_version, _ = _create_versioned_pack(
        repo,
        title="Other Override Pack",
        assets=("happy",),
    )

    with pytest.raises(ValueError, match="^pack_version_mismatch$"):
        service.resolve_expression_asset(
            actor_kind="character",
            actor_id=character_id,
            requested_expression_key="happy",
            override_pack_id=override_pack["id"],
            override_pack_version_id=other_version["id"],
        )


def test_resolver_override_fallback_opt_in_records_reason(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Override Fallback Character")
    binding_draft = _create_ready_draft(repo, assets=("happy",), title="Fallback Binding Pack")
    service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    override_pack, override_version, override_assets = _create_versioned_pack(
        repo,
        title="Fallback Override Pack",
        assets=("neutral",),
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="sad",
        override_pack_id=override_pack["id"],
        override_pack_version_id=override_version["id"],
        allow_override_fallback=True,
    )

    assert resolved.pack_id == override_pack["id"]
    assert resolved.pack_version_id == override_version["id"]
    assert resolved.asset_id == override_assets["neutral"]["id"]
    assert resolved.resolution_source == "override_fallback"
    assert resolved.fallback_reason == "override_expression_missing:pack_default"


def test_resolver_override_fallback_uses_requested_version_default(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Inactive Override Version Character")
    binding_draft = _create_ready_draft(repo, assets=("neutral",), title="Inactive Binding Pack")
    service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    override_pack, version_one, version_one_assets = _create_versioned_pack(
        repo,
        title="Inactive Override Pack",
        assets=("happy",),
        default_expression_key="happy",
    )
    version_two = repo.create_pack_version(
        owner_user_id=OWNER_USER_ID,
        pack_id=override_pack["id"],
        version_number=2,
        manifest={},
        default_expression_key="neutral",
    )
    repo.create_asset(
        owner_user_id=OWNER_USER_ID,
        pack_id=override_pack["id"],
        pack_version_id=version_two["id"],
        expression_key="neutral",
        original_expression_key="neutral",
        display_label="Neutral",
        source_filename="neutral.png",
        storage_relpath="visual_identities/inactive-override-neutral.png",
        content_type="image/png",
        bytes=123,
        sha256="sha256-inactive-override-neutral",
        width=64,
        height=64,
    )
    repo.set_active_version(
        owner_user_id=OWNER_USER_ID,
        pack_id=override_pack["id"],
        pack_version_id=version_two["id"],
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="sad",
        override_pack_id=override_pack["id"],
        override_pack_version_id=version_one["id"],
        allow_override_fallback=True,
    )

    assert resolved.pack_id == override_pack["id"]
    assert resolved.pack_version_id == version_one["id"]
    assert resolved.asset_id == version_one_assets["happy"]["id"]
    assert resolved.expression_key == "happy"
    assert resolved.fallback_reason == "override_expression_missing:pack_default"
    assert resolved.resolution_source == "override_fallback"


def test_resolver_override_fallback_opt_in_can_fall_through_to_normal_binding(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Override Binding Fallback Character")
    binding_draft = _create_ready_draft(
        repo,
        assets=("sad",),
        title="Normal Binding Pack",
        default_expression_key="missing-default",
    )
    activation = service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    binding_assets = {
        asset["expression_key"]: asset
        for asset in repo.list_assets_for_version(
            activation.pack_version_id,
            owner_user_id=OWNER_USER_ID,
        )
    }
    override_pack, override_version, _ = _create_versioned_pack(
        repo,
        title="Empty Override Fallback Pack",
        assets=("angry",),
        default_expression_key="missing-default",
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="sad",
        role_id="rival",
        role_label="Rival",
        override_pack_id=override_pack["id"],
        override_pack_version_id=override_version["id"],
        allow_override_fallback=True,
    )

    assert resolved.pack_id == activation.pack_id
    assert resolved.pack_version_id == activation.pack_version_id
    assert resolved.asset_id == binding_assets["sad"]["id"]
    assert resolved.resolution_source == "override_binding_fallback"
    assert "override_expression_missing" in resolved.fallback_reason
    assert resolved.role_id == "rival"
    assert resolved.role_label == "Rival"


def test_persona_without_pack_does_not_use_character_legacy_mood(
    chacha_db: CharactersRAGDB,
    service: VisualIdentityService,
) -> None:
    _create_character(
        chacha_db,
        name="Legacy Mood Source Character",
        extensions={"tldw": {"mood_images": {"happy": "legacy://happy.png"}}},
    )
    persona_id = chacha_db.create_persona_profile(
        {"user_id": str(OWNER_USER_ID), "name": "Persona Without Pack"}
    )

    resolved = service.resolve_expression_asset(
        actor_kind="persona",
        actor_id=persona_id,
        requested_expression_key="happy",
    )

    assert resolved.fallback_reason == "placeholder"
    assert resolved.resolution_source == "placeholder"
    assert resolved.asset_url is None


@pytest.mark.parametrize(
    ("actor_kind", "actor_id", "expected_error"),
    (
        ("character", 999999, "visual_identity_character_not_found"),
        ("persona", "missing-persona", "visual_identity_persona_not_found"),
    ),
)
def test_resolver_rejects_invalid_actor(
    service: VisualIdentityService,
    actor_kind: str,
    actor_id: int | str,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{expected_error}$"):
        service.resolve_expression_asset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            requested_expression_key="happy",
        )


def test_resolver_rejects_cross_user_override_pack(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Cross User Override Character")
    binding_draft = _create_ready_draft(repo, assets=("happy",), title="Cross User Binding")
    service.activate_draft(
        draft_id=binding_draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    override_pack, override_version, _ = _create_versioned_pack(
        repo,
        owner_user_id=OWNER_USER_ID + 1,
        title="Other User Override Pack",
        assets=("happy",),
    )

    with pytest.raises(ValueError, match="^(pack_not_found|pack_not_owned)$"):
        service.resolve_expression_asset(
            actor_kind="character",
            actor_id=character_id,
            requested_expression_key="happy",
            override_pack_id=override_pack["id"],
            override_pack_version_id=override_version["id"],
        )


def test_activation_copies_draft_assets_into_version_instead_of_mutating_them(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Immutable Version Character")
    draft = _create_ready_draft(repo, assets=("neutral", "thinking"))
    original_draft_assets = repo.list_draft_assets(draft["id"], owner_user_id=OWNER_USER_ID)
    original_draft_asset_ids = {asset["id"] for asset in original_draft_assets}

    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )

    draft_assets_after_activation = repo.list_draft_assets(draft["id"], owner_user_id=OWNER_USER_ID)
    version_assets = repo.list_assets_for_version(
        activation.pack_version_id,
        owner_user_id=OWNER_USER_ID,
    )
    assert {asset["id"] for asset in draft_assets_after_activation} == original_draft_asset_ids
    assert {asset["id"] for asset in version_assets}.isdisjoint(original_draft_asset_ids)
    assert {asset["expression_key"] for asset in version_assets} == {"neutral", "thinking"}
    assert all(asset["draft_id"] == draft["id"] for asset in draft_assets_after_activation)
    assert all(asset["pack_version_id"] is None for asset in draft_assets_after_activation)
    assert all(asset["draft_id"] is None for asset in version_assets)
    assert all(asset["pack_version_id"] == activation.pack_version_id for asset in version_assets)


def test_activation_copies_asset_source_context_to_version(
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    draft = repo.create_draft(
        owner_user_id=OWNER_USER_ID,
        title="Context Draft",
        source_kind="generated",
        status="ready_for_review",
        default_expression_key="neutral",
    )
    repo.create_asset(
        owner_user_id=OWNER_USER_ID,
        draft_id=draft["id"],
        expression_key="neutral",
        source_filename="neutral.webp",
        storage_relpath="visual_identities/neutral.webp",
        content_type="image/webp",
        bytes=12,
        sha256="sha256-neutral-context",
        width=64,
        height=64,
        source_context={"source_feature": "vn_assets", "generated_file_id": 42},
    )

    activation = service.activate_draft(draft_id=draft["id"])
    version_assets = repo.list_assets_for_version(
        activation.pack_version_id,
        owner_user_id=OWNER_USER_ID,
    )

    assert json.loads(version_assets[0]["source_context_json"]) == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
    }
    manifest = json.loads(
        repo.get_pack_version(
            activation.pack_version_id,
            owner_user_id=OWNER_USER_ID,
        )["manifest_json"]
    )
    assert manifest["assets"][0]["source_context"] == {
        "generated_file_id": 42,
        "source_feature": "vn_assets",
    }


def test_activation_uses_slot_map_replacement_asset(
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    draft = repo.create_draft(
        owner_user_id=OWNER_USER_ID,
        title="Replacement Draft",
        source_kind="zip",
        source_filename="replacement.zip",
        status="ready_for_review",
        default_expression_key="happy",
    )
    old_asset = _create_draft_asset(
        repo,
        draft_id=draft["id"],
        expression_key="happy",
        storage_relpath="visual_identities/happy-old.png",
        sha256="sha256-happy-old",
    )
    new_asset = _create_draft_asset(
        repo,
        draft_id=draft["id"],
        expression_key="happy",
        storage_relpath="visual_identities/happy-new.png",
        sha256="sha256-happy-new",
    )
    repo.update_draft_slot_map(
        draft_id=draft["id"],
        owner_user_id=OWNER_USER_ID,
        slot_map={"happy": {"asset_id": new_asset["id"], "expression_key": "happy"}},
    )

    activation = service.activate_draft(draft_id=draft["id"])
    version_assets = repo.list_assets_for_version(
        activation.pack_version_id,
        owner_user_id=OWNER_USER_ID,
    )

    assert [asset["storage_relpath"] for asset in version_assets] == [
        "visual_identities/happy-new.png"
    ]
    assert version_assets[0]["sha256"] == "sha256-happy-new"
    assert version_assets[0]["id"] != old_asset["id"]


def test_activation_excludes_cleared_slot_map_asset(
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    draft = repo.create_draft(
        owner_user_id=OWNER_USER_ID,
        title="Cleared Slot Draft",
        source_kind="zip",
        source_filename="cleared.zip",
        status="ready_for_review",
        default_expression_key="neutral",
    )
    neutral_asset = _create_draft_asset(
        repo,
        draft_id=draft["id"],
        expression_key="neutral",
        storage_relpath="visual_identities/neutral.png",
        sha256="sha256-neutral",
    )
    _create_draft_asset(
        repo,
        draft_id=draft["id"],
        expression_key="happy",
        storage_relpath="visual_identities/happy-cleared.png",
        sha256="sha256-happy-cleared",
    )
    repo.update_draft_slot_map(
        draft_id=draft["id"],
        owner_user_id=OWNER_USER_ID,
        slot_map={
            "neutral": {"asset_id": neutral_asset["id"], "expression_key": "neutral"},
            "happy": {"asset_id": None, "expression_key": "happy"},
        },
    )

    activation = service.activate_draft(draft_id=draft["id"])
    version_assets = repo.list_assets_for_version(
        activation.pack_version_id,
        owner_user_id=OWNER_USER_ID,
    )

    assert [asset["expression_key"] for asset in version_assets] == ["neutral"]
    assert [asset["storage_relpath"] for asset in version_assets] == [
        "visual_identities/neutral.png"
    ]


def test_activation_preserves_existing_pack_metadata_when_draft_targets_pack(
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    pack = repo.create_pack(
        owner_user_id=OWNER_USER_ID,
        title="Curated Expressions",
        description="Keep this user-authored description",
        source_kind="manual",
        source_context={"curator": "owner", "keep": True},
    )
    draft = _create_ready_draft(
        repo,
        assets=("neutral",),
        title="Imported Replacement Title",
        pack_id=pack["id"],
    )

    activation = service.activate_draft(draft_id=draft["id"])

    updated_pack = repo.get_pack(pack["id"], owner_user_id=OWNER_USER_ID)
    assert updated_pack is not None
    assert updated_pack["active_version_id"] == activation.pack_version_id
    assert updated_pack["title"] == "Curated Expressions"
    assert updated_pack["description"] == "Keep this user-authored description"
    assert updated_pack["source_kind"] == "manual"
    assert json.loads(updated_pack["source_context_json"]) == {
        "curator": "owner",
        "keep": True,
    }


def test_deleted_pack_does_not_resolve_for_new_messages(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(chacha_db, name="Deleted Pack Character")
    draft = _create_ready_draft(repo, assets=("neutral", "happy"))
    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )

    repo.mark_pack_deleted(pack_id=activation.pack_id, owner_user_id=OWNER_USER_ID)
    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="happy",
    )

    assert resolved.fallback_reason == "placeholder"
    assert resolved.asset_id is None
    assert resolved.pack_id is None
    assert resolved.pack_version_id is None


def test_resolve_uses_legacy_character_mood_when_no_visual_identity_binding(
    chacha_db: CharactersRAGDB,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(
        chacha_db,
        name="Legacy Mood Character",
        extensions={"tldw": {"mood_images": {"happy": "legacy://happy.png"}}},
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="happy",
    )

    assert resolved.fallback_reason == "legacy_character_mood"
    assert resolved.expression_key == "happy"
    assert resolved.asset_id is None
    assert resolved.storage_relpath is None
    assert resolved.asset_url == "legacy://happy.png"


def test_legacy_character_mood_prefers_manual_override_when_assets_miss(
    chacha_db: CharactersRAGDB,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(
        chacha_db,
        name="Legacy Manual Mood Character",
        extensions={"tldw": {"mood_images": {"angry": "legacy://manual-angry.png"}}},
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="happy",
        manual_override_expression_key="angry",
        mood_expression_key="sad",
    )

    assert resolved.fallback_reason == "legacy_character_mood"
    assert resolved.expression_key == "angry"
    assert resolved.asset_id is None
    assert resolved.asset_url == "legacy://manual-angry.png"


def test_resolve_uses_legacy_character_mood_from_extension_root_alias(
    chacha_db: CharactersRAGDB,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(
        chacha_db,
        name="Legacy Root Mood Character",
        extensions={"moodImages": {"joy": "legacy://joy.png"}},
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="happy",
    )

    assert resolved.fallback_reason == "legacy_character_mood"
    assert resolved.expression_key == "happy"
    assert resolved.asset_url == "legacy://joy.png"


@pytest.mark.parametrize("raw_alias", ("default", "normal"))
def test_resolve_neutral_alias_checks_raw_default_and_normal_version_assets(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
    raw_alias: str,
) -> None:
    character_id = _create_character(chacha_db, name=f"Raw {raw_alias.title()} Character")
    draft = _create_ready_draft(repo, assets=(raw_alias,))
    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )
    version_asset = repo.list_assets_for_version(
        activation.pack_version_id,
        owner_user_id=OWNER_USER_ID,
    )[0]

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="missing",
    )

    assert resolved.fallback_reason == "neutral_alias"
    assert resolved.expression_key == raw_alias
    assert resolved.asset_id == version_asset["id"]
    assert resolved.storage_relpath == f"visual_identities/{raw_alias}.png"


def test_resolve_uses_legacy_character_mood_after_visual_pack_fallbacks_miss(
    chacha_db: CharactersRAGDB,
    repo: VisualIdentityRepository,
    service: VisualIdentityService,
) -> None:
    character_id = _create_character(
        chacha_db,
        name="Legacy Mood After Pack Character",
        extensions={"tldw": {"mood_images": {"happy": "legacy://pack-miss-happy.png"}}},
    )
    draft = _create_ready_draft(repo, assets=("custom:wave",))
    activation = service.activate_draft(
        draft_id=draft["id"],
        actor_kind="character",
        actor_id=character_id,
    )

    resolved = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key="happy",
    )

    assert resolved.fallback_reason == "legacy_character_mood"
    assert resolved.expression_key == "happy"
    assert resolved.pack_id == activation.pack_id
    assert resolved.pack_version_id == activation.pack_version_id
    assert resolved.asset_id is None
    assert resolved.storage_relpath is None
    assert resolved.asset_url == "legacy://pack-miss-happy.png"


def _create_character(
    db: CharactersRAGDB,
    *,
    name: str = "Visual Identity Character",
    extensions: dict[str, Any] | None = None,
) -> int:
    card_data: dict[str, Any] = {"name": name}
    if extensions is not None:
        card_data["extensions"] = extensions
    character_id = db.add_character_card(card_data)
    assert character_id is not None
    return int(character_id)


def _create_ready_draft(
    repo: VisualIdentityRepository,
    *,
    assets: tuple[str, ...],
    title: str = "Imported Expressions",
    default_expression_key: str = "neutral",
    pack_id: int | None = None,
) -> dict[str, Any]:
    draft = repo.create_draft(
        owner_user_id=OWNER_USER_ID,
        pack_id=pack_id,
        title=title,
        source_kind="zip",
        source_filename=f"{title}.zip",
        status="ready_for_review",
        default_expression_key=default_expression_key,
        validation_summary={"errors": [], "warnings": []},
    )
    for expression_key in assets:
        repo.create_asset(
            owner_user_id=OWNER_USER_ID,
            draft_id=draft["id"],
            expression_key=expression_key,
            original_expression_key=expression_key,
            display_label=expression_key.title(),
            source_filename=f"{expression_key}.png",
            storage_relpath=f"visual_identities/{expression_key}.png",
            content_type="image/png",
            bytes=123,
            sha256=f"sha256-{expression_key}",
            width=64,
            height=64,
    )
    return draft


def _create_draft_asset(
    repo: VisualIdentityRepository,
    *,
    draft_id: int,
    expression_key: str,
    storage_relpath: str,
    sha256: str,
) -> dict[str, Any]:
    return repo.create_asset(
        owner_user_id=OWNER_USER_ID,
        draft_id=draft_id,
        expression_key=expression_key,
        original_expression_key=expression_key,
        display_label=expression_key.title(),
        source_filename=f"{expression_key}.png",
        storage_relpath=storage_relpath,
        content_type="image/png",
        bytes=123,
        sha256=sha256,
        width=64,
        height=64,
    )


def _create_versioned_pack(
    repo: VisualIdentityRepository,
    *,
    title: str,
    assets: tuple[str, ...],
    owner_user_id: int = OWNER_USER_ID,
    default_expression_key: str = "neutral",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    pack = repo.create_pack(
        owner_user_id=owner_user_id,
        title=title,
        default_expression_key=default_expression_key,
    )
    version = repo.create_pack_version(
        owner_user_id=owner_user_id,
        pack_id=pack["id"],
        version_number=1,
        manifest={},
        default_expression_key=default_expression_key,
    )
    pack = repo.set_active_version(
        owner_user_id=owner_user_id,
        pack_id=pack["id"],
        pack_version_id=version["id"],
    )
    version_assets = {}
    for expression_key in assets:
        version_assets[expression_key] = repo.create_asset(
            owner_user_id=owner_user_id,
            pack_id=pack["id"],
            pack_version_id=version["id"],
            expression_key=expression_key,
            original_expression_key=expression_key,
            display_label=expression_key.title(),
            source_filename=f"{expression_key}.png",
            storage_relpath=f"visual_identities/{title}-{expression_key}.png",
            content_type="image/png",
            bytes=123,
            sha256=f"sha256-{title}-{expression_key}",
            width=64,
            height=64,
        )
    return pack, version, version_assets
