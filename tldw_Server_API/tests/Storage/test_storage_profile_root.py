"""Real blob ownership survives package relocation and profile switches."""

import asyncio

import pytest

from tldw_Server_API.app.core import Storage, config


@pytest.fixture(autouse=True)
def private_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "profile-a"))
    monkeypatch.delenv("FILE_STORAGE_PATH", raising=False)
    monkeypatch.setattr(config, "settings", config.LazySettings(lambda: {"storage_backend": "filesystem"}))
    monkeypatch.setattr(Storage, "__file__", str(tmp_path / "initial" / "app" / "core" / "Storage" / "__init__.py"))
    Storage.reset_storage_backend()
    yield
    Storage.reset_storage_backend()


@pytest.mark.unit
def test_original_survives_package_relocation_and_backend_reset(monkeypatch, tmp_path):
    async def scenario():
        first = Storage.get_storage_backend()
        key = await first.store("11", 1, "original.txt", b"Alice private")
        monkeypatch.setattr(
            Storage, "__file__", str(tmp_path / "replacement" / "app" / "core" / "Storage" / "__init__.py")
        )
        Storage.reset_storage_backend()
        restored = Storage.get_storage_backend()
        assert restored.base_path == tmp_path / "profile-a" / "user_files"  # nosec B101 # Factory boundary regression.
        assert (await restored.retrieve(key)).read() == b"Alice private"  # nosec B101 # Retained real bytes.

    asyncio.run(scenario())


@pytest.mark.unit
def test_switching_profile_without_reset_does_not_share_reused_blob_key(monkeypatch, tmp_path):
    async def scenario():
        alice = Storage.get_storage_backend()
        key = await alice.store("11", 1, "original.txt", b"Alice private")
        monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "profile-b"))
        bob = Storage.get_storage_backend()
        assert not await bob.exists(key)  # nosec B101 # Cross-profile read must be absent.
        await bob.store("11", 1, "original.txt", b"Bob private")
        monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "profile-a"))
        assert (
            await Storage.get_storage_backend().retrieve(key)
        ).read() == b"Alice private"  # nosec B101 # Foreign write leaves owner intact.

    asyncio.run(scenario())


@pytest.mark.unit
def test_explicit_setting_preserves_existing_storage_root(monkeypatch, tmp_path):
    root = tmp_path / "confirmed-legacy-root"
    monkeypatch.setattr(config, "settings", config.LazySettings(lambda: {"file_storage_path": root}))
    assert Storage.get_storage_backend().base_path == root  # nosec B101 # Existing explicit override remains supported.


@pytest.mark.unit
def test_environment_override_rebinds_effective_root(monkeypatch, tmp_path):
    first = Storage.get_storage_backend()
    root = tmp_path / "confirmed-owner-root"
    monkeypatch.setenv("FILE_STORAGE_PATH", str(root))
    current = Storage.get_storage_backend()
    assert current.base_path == root and current is not first  # nosec B101 # Explicit migration root takes effect.


@pytest.mark.unit
def test_unsupported_backend_stays_rejected_after_cache_is_warm(monkeypatch):
    Storage.get_storage_backend()
    monkeypatch.setattr(config, "settings", config.LazySettings(lambda: {"storage_backend": "unsupported"}))
    with pytest.raises(ValueError, match="Unknown storage backend"):
        Storage.get_storage_backend()
