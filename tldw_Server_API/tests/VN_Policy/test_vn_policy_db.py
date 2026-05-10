from __future__ import annotations

from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import (
    VNPolicyProfileStore,
    VNPolicyRepository,
    ensure_vn_policy_tables,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-policy-test-client")
    yield database
    database.close_connection()


def test_initialized_creates_policy_tables_and_builtin_profiles(chacha_db: CharactersRAGDB) -> None:
    repo = VNPolicyRepository.initialized(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_%profile%'"
    )
    table_names = {row["name"] for row in cursor.fetchall()}

    assert {
        "vn_policy_profiles",
        "vn_generation_profiles",
        "vn_policy_profile_versions",
        "vn_generation_profile_versions",
        "vn_profile_snapshots",
    }.issubset(table_names)

    policy_profiles, policy_total = repo.list_policy_profiles(limit=20, offset=0)
    generation_profiles, generation_total = repo.list_generation_profiles(limit=20, offset=0)

    assert policy_total >= 2
    assert generation_total >= 1
    assert {"local_default", "strict_hosted"}.issubset(
        {profile["profile_id"] for profile in policy_profiles}
    )
    assert "story_default" in {profile["profile_id"] for profile in generation_profiles}


def test_policy_profile_crud_stores_version_history(chacha_db: CharactersRAGDB) -> None:
    repo = VNPolicyRepository.initialized(chacha_db)

    created = repo.create_policy_profile(
        profile_id="custom_warn",
        display_name="Custom Warn",
        description="Warns for incomplete metadata.",
        definition={
            "character_safety": {
                "missing": {"general": "warn", "mature": "block"},
                "unknown_or_ambiguous": {"general": "warn", "mature": "block"},
                "conflicting": {"default": "block"},
                "imported_untrusted": {"general": "warn", "mature": "block"},
            }
        },
        created_by_user_id=99,
    )
    updated = repo.update_policy_profile(
        "custom_warn",
        display_name="Custom Block",
        definition={
            "character_safety": {
                "missing": {"general": "block", "mature": "block"},
                "unknown_or_ambiguous": {"general": "block", "mature": "block"},
                "conflicting": {"default": "block"},
                "imported_untrusted": {"general": "block", "mature": "block"},
            }
        },
        updated_by_user_id=99,
    )

    versions = repo.list_policy_profile_versions("custom_warn")

    assert created["version"] == 1
    assert updated["version"] == 2
    assert [version["version"] for version in versions] == [1, 2]
    assert versions[0]["definition"]["character_safety"]["missing"]["general"] == "warn"
    assert versions[1]["definition"]["character_safety"]["missing"]["general"] == "block"


def test_generation_profile_validation_happens_before_persistence(chacha_db: CharactersRAGDB) -> None:
    repo = VNPolicyRepository.initialized(chacha_db)

    with pytest.raises(ValueError, match="invalid_generation_profile"):
        repo.create_generation_profile(
            profile_id="bad_temperature",
            display_name="Bad Temperature",
            definition={
                "provider": "local",
                "model": "gemma-3-12b",
                "supports_structured_output": True,
                "temperature_default": 2.5,
                "temperature_min": 0,
                "temperature_max": 1,
                "max_output_tokens": 1024,
                "allowed_content_ratings": ["general"],
                "max_choices": 4,
                "max_branch_depth": 8,
                "max_model_expansion_scope": "scene",
                "tts_allowed": True,
                "output_persistence_max_days": 30,
                "audit_mode": "metadata",
            },
            created_by_user_id=99,
        )


def test_profile_snapshot_is_immutable_after_profile_update(chacha_db: CharactersRAGDB) -> None:
    repo = VNPolicyRepository.initialized(chacha_db)
    profile = repo.get_policy_profile("local_default")
    assert profile is not None

    snapshot = repo.create_profile_snapshot(
        owner_user_id=42,
        snapshot_type="policy",
        profile_id="local_default",
        profile_version=int(profile["version"]),
        resource_type="script_version",
        resource_id=7,
        definition=profile["definition"],
    )
    repo.update_policy_profile(
        "local_default",
        display_name="Changed Local Default",
        definition={
            **profile["definition"],
            "description": "Changed after snapshot.",
        },
        updated_by_user_id=99,
    )

    loaded = repo.get_profile_snapshot(snapshot["id"], owner_user_id=42)

    assert loaded is not None
    assert loaded["definition"] == profile["definition"]
    assert loaded["profile_version"] == profile["version"]


@pytest.mark.asyncio
async def test_global_profile_create_rolls_back_when_version_history_insert_fails(monkeypatch, tmp_path) -> None:
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL=f"sqlite:///{tmp_path / 'authnz.db'}"))
    store = VNPolicyProfileStore(pool)
    await store.initialize()

    async def fail_insert_version_row(*args, **kwargs) -> None:
        raise RuntimeError("version insert failed")

    monkeypatch.setattr(store, "_insert_version_row", fail_insert_version_row)

    with pytest.raises(TransactionError, match="version insert failed"):
        await store.create_policy_profile(
            profile_id="rollback_policy",
            display_name="Rollback Policy",
            definition={
                "character_safety": {
                    "missing": {"general": "warn", "mature": "block"},
                    "unknown_or_ambiguous": {"general": "warn", "mature": "block"},
                    "conflicting": {"default": "block"},
                    "imported_untrusted": {"general": "warn", "mature": "block"},
                }
            },
            created_by_user_id=99,
        )

    assert await store.get_policy_profile("rollback_policy", include_disabled=True) is None
    await pool.close()
