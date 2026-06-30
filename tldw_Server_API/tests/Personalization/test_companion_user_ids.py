from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_existing_companion_storage_user_id,
    resolve_companion_storage_user_id,
    resolve_legacy_companion_storage_user_ids,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def test_resolve_companion_storage_user_id_preserves_numeric_ids() -> None:
    assert resolve_companion_storage_user_id("42") == "42"


def test_resolve_companion_storage_user_id_derives_stable_numeric_key_for_text_ids() -> None:
    left = resolve_companion_storage_user_id("user@example.com")
    right = resolve_companion_storage_user_id("user@example.com")

    assert left == right
    assert left.isdigit()
    assert int(left) > 2**32


def test_resolve_legacy_companion_storage_user_ids_returns_older_text_derivations() -> None:
    preferred = resolve_companion_storage_user_id("user@example.com")
    legacy = resolve_legacy_companion_storage_user_ids("user@example.com")

    assert len(legacy) == 2
    assert preferred not in legacy
    assert all(candidate.isdigit() for candidate in legacy)
    assert all(0 < int(candidate) <= 2**32 - 1 for candidate in legacy)


def test_resolve_existing_companion_storage_user_id_finds_legacy_db_without_creating_new_dir(
    monkeypatch,
    tmp_path,
) -> None:
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir()
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    try:
        user_id = "user@example.com"
        preferred = resolve_companion_storage_user_id(user_id)
        legacy = resolve_legacy_companion_storage_user_ids(user_id)[0]
        legacy_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(legacy)))
        legacy_db.update_profile(user_id, enabled=1)

        resolved = resolve_existing_companion_storage_user_id(user_id)

        assert resolved == legacy
        assert not (base_dir / preferred).exists()
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_resolve_existing_companion_storage_user_id_ignores_legacy_db_for_other_user(
    monkeypatch,
    tmp_path,
) -> None:
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir()
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    try:
        user_id = "user@example.com"
        preferred = resolve_companion_storage_user_id(user_id)
        legacy = resolve_legacy_companion_storage_user_ids(user_id)[0]
        legacy_db = PersonalizationDB(str(DatabasePaths.get_personalization_db_path(legacy)))
        legacy_db.update_profile("other-user@example.com", enabled=1)

        resolved = resolve_existing_companion_storage_user_id(user_id)

        assert resolved == preferred
        assert not (base_dir / preferred).exists()
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_resolve_existing_companion_storage_user_id_prefers_existing_new_db(
    monkeypatch,
    tmp_path,
) -> None:
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir()
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    try:
        user_id = "user@example.com"
        preferred = resolve_companion_storage_user_id(user_id)
        legacy = resolve_legacy_companion_storage_user_ids(user_id)[0]
        PersonalizationDB(str(DatabasePaths.get_personalization_db_path(legacy))).update_profile(
            user_id,
            enabled=1,
        )
        PersonalizationDB(str(DatabasePaths.get_personalization_db_path(preferred))).update_profile(
            user_id,
            enabled=1,
        )

        assert resolve_existing_companion_storage_user_id(user_id) == preferred
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass
