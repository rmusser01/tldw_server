from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
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


def test_service_create_pack_has_no_active_version_until_activation(
    service: VisualIdentityService,
) -> None:
    pack = service.create_pack(title="Manual Expressions")

    assert pack["status"] == "active"
    assert pack["active_version_id"] is None
    assert pack["default_expression_key"] == "neutral"


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


def _create_character(db: CharactersRAGDB, *, name: str = "Visual Identity Character") -> int:
    character_id = db.add_character_card({"name": name})
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
