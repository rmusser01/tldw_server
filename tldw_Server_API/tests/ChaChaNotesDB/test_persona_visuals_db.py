import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


pytestmark = pytest.mark.unit


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "persona_visuals.sqlite"


@pytest.fixture()
def db_instance(db_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(db_path, "persona-visuals-test-client")
    yield db
    db.close_connection()


def test_user_persona_visuals_dir_is_created_under_user_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_base = tmp_path / "user-dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_base))

    visuals_dir = DatabasePaths.get_user_persona_visuals_dir("user-1")

    assert visuals_dir == user_base / "user-1" / "persona_visuals"
    assert visuals_dir.is_dir()


def test_migration_v44_to_latest_creates_persona_visual_tables(db_path: Path) -> None:
    """Verify v44 migration recreates persona visual tables and indexes."""
    seeded = CharactersRAGDB(db_path, "persona-visuals-seed")
    seeded.close_connection()

    CharactersRAGDB._prepare_sqlite_schema_drift_fixture(
        db_path,
        version=44,
        drop_tables=(
            "persona_visual_candidates",
            "persona_visual_assets",
            "persona_visual_packs",
        ),
    )

    migrated = CharactersRAGDB(db_path, "persona-visuals-migration")
    try:
        conn = migrated.get_connection()

        version = migrated._get_db_version(conn)
        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

        tables = migrated._sqlite_table_names(conn)
        assert {
            "persona_visual_packs",
            "persona_visual_assets",
            "persona_visual_candidates",
        }.issubset(tables)

        pack_indexes = migrated._sqlite_index_names(conn, "persona_visual_packs")
        asset_indexes = migrated._sqlite_index_names(conn, "persona_visual_assets")
        assert "idx_persona_visual_packs_one_active" in pack_indexes
        assert "idx_persona_visual_packs_persona" in pack_indexes
        assert "idx_persona_visual_assets_pack" in asset_indexes
    finally:
        migrated.close_connection()


def test_migration_v44_to_v45_creates_persona_visual_tables(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the exact v44 -> v45 update path is registered and runnable."""
    seeded = CharactersRAGDB(db_path, "persona-visuals-v45-seed")
    seeded.close_connection()

    CharactersRAGDB._prepare_sqlite_schema_drift_fixture(
        db_path,
        version=44,
        drop_tables=(
            "persona_visual_candidates",
            "persona_visual_assets",
            "persona_visual_packs",
        ),
    )
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 45)

    migrated = CharactersRAGDB(db_path, "persona-visuals-v45-migration")
    try:
        conn = migrated.get_connection()

        version = migrated._get_db_version(conn)
        assert version == 45

        tables = migrated._sqlite_table_names(conn)
        assert {
            "persona_visual_packs",
            "persona_visual_assets",
            "persona_visual_candidates",
        }.issubset(tables)

        pack_indexes = migrated._sqlite_index_names(conn, "persona_visual_packs")
        asset_indexes = migrated._sqlite_index_names(conn, "persona_visual_assets")
        assert "idx_persona_visual_packs_one_active" in pack_indexes
        assert "idx_persona_visual_packs_persona" in pack_indexes
        assert "idx_persona_visual_assets_pack" in asset_indexes
    finally:
        migrated.close_connection()


def test_sqlite_linear_migration_registry_maps_v44_to_v45(db_path: Path) -> None:
    """Verify the dispatcher advertises the v44 -> v45 migration path."""
    db = CharactersRAGDB(db_path, "persona-visuals-registry")
    try:
        migration_steps = db._sqlite_linear_migration_steps()

        assert 44 in migration_steps
        assert migration_steps[44].__name__ == "_migrate_from_v44_to_v45"
    finally:
        db.close_connection()


def test_new_database_initialization_uses_linear_migration_registry(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fresh databases use the same linear migration registry."""
    original_steps = CharactersRAGDB._sqlite_linear_migration_steps

    def registry_without_v44(
        db: CharactersRAGDB,
    ) -> dict[int, Callable[[sqlite3.Connection], None]]:
        steps = dict(original_steps(db))
        steps.pop(44)
        return steps

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 45)
    monkeypatch.setattr(
        CharactersRAGDB,
        "_sqlite_linear_migration_steps",
        registry_without_v44,
    )

    db = None
    try:
        with pytest.raises(CharactersRAGDBError, match="Migration path undefined.*from version 0 to 45"):
            db = CharactersRAGDB(db_path, "persona-visuals-new-db-registry")
    finally:
        if db is not None:
            db.close_connection()


def test_migration_v44_to_latest_repairs_missing_persona_tables(db_path: Path) -> None:
    """Verify drifted v44 databases repair missing persona schema artifacts."""
    seeded = CharactersRAGDB(db_path, "persona-visuals-missing-persona-seed")
    seeded.close_connection()

    CharactersRAGDB._prepare_sqlite_schema_drift_fixture(
        db_path,
        version=44,
        drop_tables=(
            "persona_visual_candidates",
            "persona_visual_assets",
            "persona_visual_packs",
            "persona_sessions",
            "persona_memory_entries",
            "persona_policy_rules",
            "persona_scope_rules",
            "persona_profiles",
        ),
    )

    migrated = CharactersRAGDB(db_path, "persona-visuals-missing-persona-migration")
    try:
        conn = migrated.get_connection()

        version = migrated._get_db_version(conn)
        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

        tables = migrated._sqlite_table_names(conn)
        assert {
            "persona_profiles",
            "persona_scope_rules",
            "persona_policy_rules",
            "persona_sessions",
            "persona_memory_entries",
            "persona_visual_packs",
            "persona_visual_assets",
            "persona_visual_candidates",
        }.issubset(tables)

        profile_columns = migrated._sqlite_column_names(conn, "persona_profiles")
        assert {
            "use_persona_state_context_default",
            "voice_defaults_json",
            "setup_json",
            "origin_character_id",
            "origin_character_name",
            "origin_character_snapshot_at",
        }.issubset(profile_columns)

        profile_indexes = migrated._sqlite_index_names(conn, "persona_profiles")
        assert {
            "idx_persona_profiles_user",
            "idx_persona_profiles_user_active",
            "idx_persona_profiles_character",
        }.issubset(profile_indexes)

        memory_columns = migrated._sqlite_column_names(conn, "persona_memory_entries")
        assert {"scope_snapshot_id", "session_id"}.issubset(memory_columns)

        memory_indexes = migrated._sqlite_index_names(conn, "persona_memory_entries")
        assert {"idx_persona_memory_scope", "idx_persona_memory_session"}.issubset(memory_indexes)
    finally:
        migrated.close_connection()


def test_postgres_v45_migration_does_not_define_candidate_provenance_column() -> None:
    assert "generation_provenance_json" not in CharactersRAGDB._MIGRATION_SQL_V44_TO_V45_POSTGRES
    assert "generation_provenance_json" in CharactersRAGDB._MIGRATION_SQL_V46_TO_V47_POSTGRES


def test_create_and_list_visual_pack_for_persona(db_instance: CharactersRAGDB) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Visual Persona"}
    )

    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Default Sprite Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )

    assert pack["persona_id"] == persona_id
    assert pack["user_id"] == "user-1"
    assert pack["title"] == "Default Sprite Pack"
    assert pack["manifest"] == {"manifest_version": 1, "renderer_type": "sprite_frames"}

    listed = db_instance.list_persona_visual_packs(
        persona_id=persona_id,
        user_id="user-1",
    )
    assert [item["id"] for item in listed] == [pack["id"]]


def test_only_one_active_pack_per_persona(db_instance: CharactersRAGDB) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Active Visual Persona"}
    )
    first = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="First Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    second = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Second Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )

    db_instance.activate_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=first["id"],
    )
    db_instance.activate_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=second["id"],
    )

    active = db_instance.get_active_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
    )
    packs = db_instance.list_persona_visual_packs(
        persona_id=persona_id,
        user_id="user-1",
    )

    assert active is not None
    assert active["id"] == second["id"]
    statuses = {pack["id"]: pack["status"] for pack in packs}
    assert statuses[first["id"]] == "archived"
    assert statuses[second["id"]] == "active"


def test_deactivate_visual_pack_clears_active_pack(db_instance: CharactersRAGDB) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Deactivate Visual Persona"}
    )
    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )

    db_instance.activate_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
    )
    db_instance.deactivate_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
    )

    assert (
        db_instance.get_active_persona_visual_pack(
            persona_id=persona_id,
            user_id="user-1",
        )
        is None
    )
    archived = db_instance.get_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
    )
    assert archived is not None
    assert archived["status"] == "archived"


def test_assets_are_scoped_to_pack_persona_and_user(db_instance: CharactersRAGDB) -> None:
    persona_a = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Visual Persona A"}
    )
    persona_b = db_instance.create_persona_profile(
        {"user_id": "user-2", "name": "Visual Persona B"}
    )
    pack_a = db_instance.create_persona_visual_pack(
        persona_id=persona_a,
        user_id="user-1",
        title="Pack A",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    pack_b = db_instance.create_persona_visual_pack(
        persona_id=persona_b,
        user_id="user-2",
        title="Pack B",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )

    asset_a = db_instance.create_persona_visual_asset(
        pack_id=pack_a["id"],
        persona_id=persona_a,
        user_id="user-1",
        asset_role="frame",
        storage_key="persona_visuals/persona-a/pack-a/asset-a.png",
        original_filename="asset-a.png",
        mime_type="image/png",
        byte_size=12,
        checksum_sha256="a" * 64,
        width=64,
        height=64,
        provenance="uploaded",
    )
    db_instance.create_persona_visual_asset(
        pack_id=pack_b["id"],
        persona_id=persona_b,
        user_id="user-2",
        asset_role="frame",
        storage_key="persona_visuals/persona-b/pack-b/asset-b.png",
        original_filename="asset-b.png",
        mime_type="image/png",
        byte_size=12,
        checksum_sha256="b" * 64,
        width=64,
        height=64,
        provenance="uploaded",
    )

    assert [
        item["id"]
        for item in db_instance.list_persona_visual_assets(
            pack_id=pack_a["id"],
            persona_id=persona_a,
            user_id="user-1",
        )
    ] == [asset_a["id"]]
    assert (
        db_instance.list_persona_visual_assets(
            pack_id=pack_a["id"],
            persona_id=persona_a,
            user_id="user-2",
        )
        == []
    )


def test_candidate_accept_reject_round_trip(db_instance: CharactersRAGDB) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Candidate Visual Persona"}
    )
    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Generated Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )

    accepted = db_instance.create_persona_visual_candidate(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        job_id="job-1",
        proposed_manifest_patch={"states": {"thinking": {"animation_id": "think"}}},
        generated_asset_ids=["asset-1"],
        prompt="make a thinking pose",
    )
    rejected = db_instance.create_persona_visual_candidate(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        job_id="job-2",
        proposed_manifest_patch={},
        generated_asset_ids=[],
        prompt="make another pose",
    )

    accepted_after = db_instance.update_persona_visual_candidate_status(
        candidate_id=accepted["id"],
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        status="accepted",
    )
    rejected_after = db_instance.update_persona_visual_candidate_status(
        candidate_id=rejected["id"],
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        status="rejected",
        failure_reason="not useful",
    )

    assert accepted_after is not None
    assert accepted_after["status"] == "accepted"
    assert accepted_after["proposed_manifest_patch"]["states"]["thinking"] == {
        "animation_id": "think"
    }
    assert accepted_after["generated_asset_ids"] == ["asset-1"]
    assert rejected_after is not None
    assert rejected_after["status"] == "rejected"
    assert rejected_after["failure_reason"] == "not useful"


def test_candidate_generation_provenance_round_trip(db_instance: CharactersRAGDB) -> None:
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Candidate Provenance Persona"}
    )
    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Generated Provenance Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )

    candidate = db_instance.create_persona_visual_candidate(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        job_id="job-1",
        proposed_manifest_patch={"states": {"thinking": {"animation_id": "think"}}},
        generated_asset_ids=["asset-1"],
        prompt="make a thinking pose",
        generation_provenance={
            "schema_version": 999,
            "generation_mode": "recipe_backed",
            "request_id": "request-1",
            "job_id": "job-1",
            "backend": "fake\nprovider",
            "target_state": "thinking",
            "recipe": {
                "starter_pack_id": "starter-basic",
                "recipe_output": "required_state_loops",
                "correlation_id": "corr-1",
                "identity_brief": "friendly buddy",
                "neutral_anchor": "api_key=secret\n/Users/macbook-dev/private",
                "static_sheet": "x" * 400,
                "review_checks": ["consistent silhouette", "token secret leak"],
                "user_prompt_included": True,
                "user_prompt": "raw user prompt should not be returned",
            },
            "prompt": "raw generation prompt should not be returned",
        },
    )

    listed = db_instance.list_persona_visual_candidates(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
    )
    listed_candidate = next(item for item in listed if item["id"] == candidate["id"])
    provenance = listed_candidate["generation_provenance"]

    assert provenance["schema_version"] == 1
    assert provenance["generation_mode"] == "recipe_backed"
    assert provenance["request_id"] == "request-1"
    assert provenance["job_id"] == "job-1"
    assert provenance["backend"] == "fake provider"
    assert provenance["target_state"] == "thinking"
    assert provenance["recipe"]["starter_pack_id"] == "starter-basic"
    assert provenance["recipe"]["recipe_output"] == "required_state_loops"
    assert provenance["recipe"]["correlation_id"] == "corr-1"
    assert provenance["recipe"]["identity_brief"] == "friendly buddy"
    assert provenance["recipe"]["neutral_anchor"] == "[redacted]"
    assert len(provenance["recipe"]["static_sheet"]) <= 240
    assert provenance["recipe"]["user_prompt_included"] is True
    serialized = json.dumps(provenance)
    assert "api_key" not in serialized
    assert "/Users/macbook-dev/private" not in serialized
    assert "raw user prompt should not be returned" not in serialized
    assert "raw generation prompt should not be returned" not in serialized
