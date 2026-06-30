from __future__ import annotations

from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import (
    VNScriptsRepository,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-scripts-test-client")
    yield database
    database.close_connection()


def _program(asset_pack_id: int = 7) -> dict:
    return {
        "schema_version": "vn_script_program.v1",
        "title": "Archive Door",
        "primary_asset_pack_id": asset_pack_id,
        "entry_label": "start",
        "variables": {"has_key": {"type": "boolean", "default": False, "public": True}},
        "generation_defaults": {"profile_id": "story_default", "persist_model_outputs": True},
        "labels": {
            "start": [
                {"op": "narrate", "text": "The door hums."},
                {"op": "end"},
            ]
        },
    }


def test_initialized_creates_script_tables_and_profile_snapshot_table(chacha_db: CharactersRAGDB) -> None:
    repo = VNScriptsRepository.initialized(chacha_db)
    script = repo.create_script(
        owner_user_id=42,
        title="Archive Door",
        description="A short scripted story.",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
    )

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_script%' OR name = 'vn_profile_snapshots'"
    )
    table_names = {row["name"] for row in cursor.fetchall()}

    assert {
        "vn_scripts",
        "vn_script_drafts",
        "vn_script_versions",
        "vn_script_manifest_snapshots",
        "vn_script_publish_requests",
        "vn_profile_snapshots",
    }.issubset(table_names)
    assert script["title"] == "Archive Door"
    assert script["draft"]["revision"] == 0


def test_draft_replacement_enforces_if_revision(chacha_db: CharactersRAGDB) -> None:
    repo = VNScriptsRepository.initialized(chacha_db)
    script = repo.create_script(
        owner_user_id=42,
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
    )

    draft = repo.replace_draft(
        script["id"],
        owner_user_id=42,
        if_revision=0,
        draft=_program(),
        diagnostics={"valid": True, "errors": [], "warnings": []},
    )

    assert draft["revision"] == 1
    assert draft["draft"]["entry_label"] == "start"

    with pytest.raises(ValueError, match="draft_revision_conflict"):
        repo.replace_draft(
            script["id"],
            owner_user_id=42,
            if_revision=0,
            draft=_program(),
            diagnostics={"valid": True, "errors": [], "warnings": []},
        )


def test_publish_request_replays_same_payload_and_rejects_conflict(chacha_db: CharactersRAGDB) -> None:
    repo = VNScriptsRepository.initialized(chacha_db)
    script = repo.create_script(
        owner_user_id=42,
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
    )

    first = repo.create_publish_request(
        owner_user_id=42,
        script_id=script["id"],
        idempotency_key="publish-v1",
        payload_hash="hash-a",
        response={"status": "published", "version_id": 9},
    )
    replayed = repo.create_publish_request(
        owner_user_id=42,
        script_id=script["id"],
        idempotency_key="publish-v1",
        payload_hash="hash-a",
        response={"status": "published", "version_id": 9},
    )

    assert replayed["id"] == first["id"]
    assert replayed["response"] == {"status": "published", "version_id": 9}

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_publish_request(
            owner_user_id=42,
            script_id=script["id"],
            idempotency_key="publish-v1",
            payload_hash="hash-b",
            response={"status": "published", "version_id": 10},
        )


def test_publish_version_with_request_replays_without_duplicate_versions(chacha_db: CharactersRAGDB) -> None:
    repo = VNScriptsRepository.initialized(chacha_db)
    script = repo.create_script(
        owner_user_id=42,
        title="Archive Door",
        primary_asset_pack_id=7,
        policy_profile_id="local_default",
        generation_profile_id="story_default",
    )

    first = repo.publish_version_with_request(
        owner_user_id=42,
        script_id=script["id"],
        idempotency_key="publish-v1",
        payload_hash="hash-a",
        label="v1",
        draft_revision=1,
        program=_program(),
        asset_pack_id=7,
        manifest={"schema_version": "vn_asset_manifest.v1", "pack_id": 7, "assets": {}},
        manifest_hash="manifest-a",
        policy_profile={
            "profile_id": "local_default",
            "version": 1,
            "definition": {"character_safety": {"missing": {"general": "warn"}}},
        },
        generation_profile={
            "profile_id": "story_default",
            "version": 1,
            "definition": {"max_choices": 4},
        },
        script_defaults={"content_rating": "general"},
        validation={"valid": True, "errors": [], "warnings": []},
    )
    replayed = repo.publish_version_with_request(
        owner_user_id=42,
        script_id=script["id"],
        idempotency_key="publish-v1",
        payload_hash="hash-a",
        label="ignored",
        draft_revision=1,
        program=_program(),
        asset_pack_id=7,
        manifest={"schema_version": "vn_asset_manifest.v1", "pack_id": 7, "assets": {}},
        manifest_hash="manifest-a",
        policy_profile={
            "profile_id": "local_default",
            "version": 1,
            "definition": {"character_safety": {"missing": {"general": "warn"}}},
        },
        generation_profile={
            "profile_id": "story_default",
            "version": 1,
            "definition": {"max_choices": 4},
        },
        script_defaults={"content_rating": "general"},
        validation={"valid": True, "errors": [], "warnings": []},
    )
    versions, total = repo.list_versions(script["id"], owner_user_id=42)

    assert first["replayed"] is False
    assert replayed["replayed"] is True
    assert replayed["response"] == first["response"]
    assert total == 1
    assert versions[0]["id"] == first["version"]["id"]

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.publish_version_with_request(
            owner_user_id=42,
            script_id=script["id"],
            idempotency_key="publish-v1",
            payload_hash="hash-b",
            label="conflict",
            draft_revision=1,
            program=_program(),
            asset_pack_id=7,
            manifest={"schema_version": "vn_asset_manifest.v1", "pack_id": 7, "assets": {}},
            manifest_hash="manifest-b",
            policy_profile={
                "profile_id": "local_default",
                "version": 1,
                "definition": {"character_safety": {"missing": {"general": "warn"}}},
            },
            generation_profile={
                "profile_id": "story_default",
                "version": 1,
                "definition": {"max_choices": 4},
            },
            script_defaults={"content_rating": "general"},
            validation={"valid": True, "errors": [], "warnings": []},
        )

    _, total_after_conflict = repo.list_versions(script["id"], owner_user_id=42)
    assert total_after_conflict == 1
