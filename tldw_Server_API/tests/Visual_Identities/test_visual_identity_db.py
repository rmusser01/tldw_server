import json
from typing import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import (
    VisualIdentityRepository,
    ensure_visual_identity_tables,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="visual-identity-test-client")
    yield database
    database.close_connection()


def test_visual_identity_tables_are_created(chacha_db: CharactersRAGDB) -> None:
    ensure_visual_identity_tables(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'visual_identity_%'"
    )
    table_names = {row[0] for row in cursor.fetchall()}
    assert {
        "visual_identity_packs",
        "visual_identity_pack_drafts",
        "visual_identity_pack_versions",
        "visual_identity_assets",
        "visual_identity_bindings",
        "visual_identity_idempotency",
    }.issubset(table_names)


def test_ensure_visual_identity_tables_rejects_non_sqlite_before_transaction() -> None:
    class NonSqliteDB:
        backend_type = BackendType.POSTGRESQL

        def transaction(self):
            raise AssertionError("transaction should not be opened for unsupported backends")

    with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
        ensure_visual_identity_tables(NonSqliteDB())  # type: ignore[arg-type]


def test_repository_constructor_rejects_non_sqlite_backend() -> None:
    class NonSqliteDB:
        backend_type = BackendType.POSTGRESQL

    with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
        VisualIdentityRepository(NonSqliteDB())  # type: ignore[arg-type]


def test_repository_constructor_does_not_create_schema(chacha_db: CharactersRAGDB) -> None:
    VisualIdentityRepository(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'visual_identity_%'"
    )
    assert cursor.fetchall() == []


def test_ensure_visual_identity_tables_preserves_outer_transaction_rollback(
    chacha_db: CharactersRAGDB,
) -> None:
    character_name = "Rolled Back Before Visual Identity Schema"

    with pytest.raises(RuntimeError, match="force rollback"):
        with chacha_db.transaction() as conn:
            conn.execute(
                "INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                (character_name, chacha_db.client_id),
            )
            ensure_visual_identity_tables(chacha_db)
            raise RuntimeError("force rollback")

    cursor = chacha_db.execute_query(
        "SELECT id FROM character_cards WHERE name = ?",
        (character_name,),
    )
    assert cursor.fetchone() is None


def test_idempotency_claim_replays_completed_response_and_rejects_payload_conflict(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)

    first_record, claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
    )
    assert claimed is True
    assert first_record["status"] == "in_progress"

    second_record, second_claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
    )
    assert second_claimed is False
    assert second_record["status"] == "in_progress"

    repo.complete_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
        response={"draft_id": 9},
    )

    completed_record, replay_claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
    )
    assert replay_claimed is False
    assert completed_record["status"] == "completed"
    assert json.loads(completed_record["response_json"]) == {"draft_id": 9}

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.claim_idempotency_record(
            owner_user_id=42,
            scope="visual_identity_import",
            resource_id="pack:1",
            idempotency_key="import-1",
            payload_hash="hash-b",
        )


def test_complete_idempotency_record_rejects_payload_hash_conflict(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    repo.claim_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
        payload_hash="hash-a",
    )

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.complete_idempotency_record(
            owner_user_id=42,
            scope="visual_identity_import",
            resource_id="pack:1",
            idempotency_key="import-1",
            payload_hash="hash-b",
            response={"draft_id": 10},
        )

    record = repo.get_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="import-1",
    )
    assert record["status"] == "in_progress"
    assert record["payload_hash"] == "hash-a"


def test_create_idempotency_record_is_conflict_tolerant_for_same_payload(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)

    first = repo.create_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="legacy-import-1",
        payload_hash="hash-a",
        response={"draft_id": 9},
    )
    second = repo.create_idempotency_record(
        owner_user_id=42,
        scope="visual_identity_import",
        resource_id="pack:1",
        idempotency_key="legacy-import-1",
        payload_hash="hash-a",
        response={"draft_id": 9},
    )

    assert second["id"] == first["id"]
    assert second["status"] == "completed"
    assert json.loads(second["response_json"]) == {"draft_id": 9}
    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_idempotency_record(
            owner_user_id=42,
            scope="visual_identity_import",
            resource_id="pack:1",
            idempotency_key="legacy-import-1",
            payload_hash="hash-b",
            response={"draft_id": 10},
        )


def test_pack_creation_list_update_archive_and_delete(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)

    pack = repo.create_pack(
        owner_user_id=1,
        title="Starter Expressions",
        description="Initial shared expression pack",
        source_kind="manual",
        source_context={"source": "test"},
    )
    assert pack["status"] == "active"
    assert pack["default_expression_key"] == "neutral"
    assert json.loads(pack["source_context_json"]) == {"source": "test"}

    updated = repo.update_pack(
        pack_id=pack["id"],
        owner_user_id=1,
        fields={"title": "Updated Expressions", "default_expression_key": "happy"},
    )
    assert updated is not None
    assert updated["title"] == "Updated Expressions"
    assert updated["default_expression_key"] == "happy"
    assert updated["version"] == pack["version"] + 1

    assert [row["id"] for row in repo.list_packs(owner_user_id=1)] == [pack["id"]]
    assert repo.archive_pack(pack_id=pack["id"], owner_user_id=1)["status"] == "archived"
    assert repo.mark_pack_deleted(pack_id=pack["id"], owner_user_id=1)["status"] == "deleted"
    assert repo.get_pack(pack["id"], owner_user_id=1) is None


def test_update_pack_rejects_unsupported_fields_and_active_version_updates(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, title="Protected Pack")
    version = repo.create_pack_version(
        pack_id=pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"renderer": "sprite_frames"},
    )

    with pytest.raises(ValueError, match="unsupported_pack_update_field:active_version_id"):
        repo.update_pack(
            pack_id=pack["id"],
            owner_user_id=1,
            fields={"active_version_id": version["id"]},
        )
    with pytest.raises(ValueError, match="unsupported_pack_update_field:unknown"):
        repo.update_pack(pack_id=pack["id"], owner_user_id=1, fields={"unknown": "value"})

    assert repo.get_pack(pack["id"], owner_user_id=1)["active_version_id"] is None


def test_draft_creation_status_slot_map_and_assets(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    draft = repo.create_draft(
        owner_user_id=1,
        title="Imported Pack",
        source_kind="zip",
        source_filename="sprites.zip",
        validation_summary={"warnings": []},
    )

    updated = repo.update_draft_slot_map(
        draft_id=draft["id"],
        owner_user_id=1,
        slot_map={"neutral": {"asset_id": 1}},
    )
    assert json.loads(updated["slot_map_json"]) == {"neutral": {"asset_id": 1}}
    assert repo.set_draft_status(
        draft_id=draft["id"],
        owner_user_id=1,
        status="ready_for_review",
    )["status"] == "ready_for_review"

    asset = repo.create_asset(
        owner_user_id=1,
        draft_id=draft["id"],
        expression_key="neutral",
        source_filename="neutral.png",
        storage_relpath="visual_identities/neutral.png",
        content_type="image/png",
        bytes=123,
        sha256="abc123",
        width=64,
        height=64,
    )

    assert repo.get_draft(draft["id"], owner_user_id=1)["status"] == "ready_for_review"
    assert [row["id"] for row in repo.list_draft_assets(draft["id"], owner_user_id=1)] == [
        asset["id"]
    ]


def test_update_draft_validation_summary_is_owner_scoped(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    draft = repo.create_draft(owner_user_id=1, title="Import Draft", source_kind="zip")

    updated = repo.update_draft_validation_summary(
        draft_id=draft["id"],
        owner_user_id=1,
        validation_summary={"errors": [{"code": "unsafe_archive_path"}]},
    )

    assert json.loads(updated["validation_summary_json"]) == {
        "errors": [{"code": "unsafe_archive_path"}]
    }
    with pytest.raises(ValueError, match="visual_identity_draft_not_found"):
        repo.update_draft_validation_summary(
            draft_id=draft["id"],
            owner_user_id=2,
            validation_summary={"errors": []},
        )
    assert json.loads(repo.get_draft(draft["id"], owner_user_id=1)["validation_summary_json"]) == {
        "errors": [{"code": "unsafe_archive_path"}]
    }


def test_update_draft_validation_summary_rejects_missing_draft(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)

    with pytest.raises(ValueError, match="visual_identity_draft_not_found"):
        repo.update_draft_validation_summary(
            draft_id=9999,
            owner_user_id=1,
            validation_summary={"errors": []},
        )


def test_version_creation_and_set_active_version(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, title="Versioned Expressions")

    version = repo.create_pack_version(
        pack_id=pack["id"],
        owner_user_id=1,
        version_number=1,
        default_expression_key="neutral",
        manifest={"renderer": "sprite_frames"},
    )
    active_pack = repo.set_active_version(
        pack_id=pack["id"],
        owner_user_id=1,
        pack_version_id=version["id"],
    )

    assert repo.get_pack_version(version["id"], owner_user_id=1)["pack_id"] == pack["id"]
    assert active_pack["active_version_id"] == version["id"]


def test_pack_version_requires_existing_owner_pack(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, title="Owner Pack")

    with pytest.raises(ValueError, match="visual_identity_pack_not_found"):
        repo.create_pack_version(
            pack_id=pack["id"],
            owner_user_id=2,
            version_number=1,
            manifest={"renderer": "sprite_frames"},
        )
    with pytest.raises(ValueError, match="visual_identity_pack_not_found"):
        repo.create_pack_version(
            pack_id=9999,
            owner_user_id=1,
            version_number=1,
            manifest={"renderer": "sprite_frames"},
        )


def test_asset_lists_for_version_and_soft_delete(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, title="Asset Expressions")
    version = repo.create_pack_version(
        pack_id=pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"assets": []},
    )
    asset = repo.create_asset(
        owner_user_id=1,
        pack_id=pack["id"],
        pack_version_id=version["id"],
        expression_key="happy",
        original_expression_key="joy",
        display_label="Happy",
        source_filename="joy.webp",
        storage_relpath="visual_identities/joy.webp",
        content_type="image/webp",
        bytes=456,
        sha256="def456",
        width=80,
        height=96,
        is_animated=True,
        frame_count=2,
        duration_ms=240,
        preview_relpath="visual_identities/joy-preview.png",
    )

    assert [row["id"] for row in repo.list_assets_for_version(version["id"], owner_user_id=1)] == [
        asset["id"]
    ]
    assert repo.get_asset(asset["id"], owner_user_id=1)["is_animated"] == 1
    assert repo.mark_asset_deleted(asset["id"], owner_user_id=1)["deleted"] == 1
    assert repo.get_asset(asset["id"], owner_user_id=1) is None
    assert repo.list_assets_for_version(version["id"], owner_user_id=1) == []


def test_asset_creation_rejects_orphan_and_mismatched_references(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, title="Asset Pack")
    version = repo.create_pack_version(
        pack_id=pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"renderer": "sprite_frames"},
    )
    foreign_draft = repo.create_draft(owner_user_id=2, title="Foreign Draft", source_kind="zip")

    with pytest.raises(ValueError, match="visual_identity_asset_attachment_required"):
        repo.create_asset(
            owner_user_id=1,
            expression_key="happy",
            source_filename="happy.png",
            storage_relpath="visual_identities/happy.png",
            content_type="image/png",
            bytes=123,
            sha256="abc123",
            width=64,
            height=64,
        )
    with pytest.raises(ValueError, match="visual_identity_pack_version_not_found"):
        repo.create_asset(
            owner_user_id=1,
            pack_id=pack["id"],
            pack_version_id=9999,
            expression_key="happy",
            source_filename="happy.png",
            storage_relpath="visual_identities/happy.png",
            content_type="image/png",
            bytes=123,
            sha256="abc123",
            width=64,
            height=64,
        )
    with pytest.raises(ValueError, match="visual_identity_draft_not_found"):
        repo.create_asset(
            owner_user_id=1,
            draft_id=foreign_draft["id"],
            expression_key="happy",
            source_filename="happy.png",
            storage_relpath="visual_identities/happy.png",
            content_type="image/png",
            bytes=123,
            sha256="abc123",
            width=64,
            height=64,
        )
    with pytest.raises(ValueError, match="visual_identity_asset_dimensions_invalid"):
        repo.create_asset(
            owner_user_id=1,
            pack_id=pack["id"],
            pack_version_id=version["id"],
            expression_key="happy",
            source_filename="happy.png",
            storage_relpath="visual_identities/happy.png",
            content_type="image/png",
            bytes=0,
            sha256="abc123",
            width=64,
            height=64,
        )


def test_binding_upsert_keeps_one_active_binding_per_actor(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    first_pack = repo.create_pack(owner_user_id=1, title="First Bound Pack")
    first_version = repo.create_pack_version(
        pack_id=first_pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"pack": "first"},
    )
    second_pack = repo.create_pack(owner_user_id=1, title="Second Bound Pack")
    second_version = repo.create_pack_version(
        pack_id=second_pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"pack": "second"},
    )

    first = repo.upsert_binding(
        owner_user_id=1,
        actor_kind="character",
        actor_id=7,
        pack_id=first_pack["id"],
        active_version_id=first_version["id"],
    )
    second = repo.upsert_binding(
        owner_user_id=1,
        actor_kind="character",
        actor_id=7,
        pack_id=second_pack["id"],
        active_version_id=second_version["id"],
    )

    assert second["id"] == first["id"]
    binding = repo.get_binding_for_actor(owner_user_id=1, actor_kind="character", actor_id=7)
    assert binding["pack_id"] == second_pack["id"]
    assert binding["active_version_id"] == second_version["id"]


def test_binding_upsert_rejects_version_from_other_pack_and_resolve_cannot_mix_manifest(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack_a = repo.create_pack(owner_user_id=1, title="Pack A")
    version_a = repo.create_pack_version(
        pack_id=pack_a["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"pack": "a"},
    )
    pack_b = repo.create_pack(owner_user_id=1, title="Pack B")
    version_b = repo.create_pack_version(
        pack_id=pack_b["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"pack": "b"},
    )

    with pytest.raises(ValueError, match="visual_identity_pack_version_not_found"):
        repo.upsert_binding(
            owner_user_id=1,
            actor_kind="character",
            actor_id=7,
            pack_id=pack_a["id"],
            active_version_id=version_b["id"],
        )

    with chacha_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO visual_identity_bindings (
                owner_user_id,
                actor_kind,
                actor_id,
                pack_id,
                active_version_id,
                status
            )
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            (1, "character", 7, pack_a["id"], version_b["id"]),
        )

    assert repo.resolve_active_binding(owner_user_id=1, actor_kind="character", actor_id=7) is None
    repo.delete_binding(
        repo.get_binding_for_actor(owner_user_id=1, actor_kind="character", actor_id=7)["id"],
        owner_user_id=1,
    )
    binding = repo.upsert_binding(
        owner_user_id=1,
        actor_kind="character",
        actor_id=7,
        pack_id=pack_a["id"],
        active_version_id=version_a["id"],
    )
    resolved = repo.resolve_active_binding(owner_user_id=1, actor_kind="character", actor_id=7)
    assert resolved["id"] == binding["id"]
    assert json.loads(resolved["active_manifest_json"]) == {"pack": "a"}


def test_binding_delete_and_resolve_active_binding(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, title="Bound Expressions")
    version = repo.create_pack_version(
        pack_id=pack["id"],
        owner_user_id=1,
        version_number=1,
        manifest={"assets": []},
    )
    repo.set_active_version(
        pack_id=pack["id"],
        owner_user_id=1,
        pack_version_id=version["id"],
    )
    binding = repo.upsert_binding(
        owner_user_id=1,
        actor_kind="persona",
        actor_id=3,
        pack_id=pack["id"],
        active_version_id=version["id"],
    )

    resolved = repo.resolve_active_binding(owner_user_id=1, actor_kind="persona", actor_id=3)
    assert resolved["pack_id"] == pack["id"]
    assert resolved["pack_status"] == "active"

    repo.delete_binding(binding["id"], owner_user_id=1)
    assert repo.get_binding_for_actor(owner_user_id=1, actor_kind="persona", actor_id=3) is None
    assert repo.resolve_active_binding(owner_user_id=1, actor_kind="persona", actor_id=3) is None
