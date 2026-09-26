from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError as AuthNZStorageError
from tldw_Server_API.app.core.Storage import generated_file_helpers


@pytest.mark.asyncio
async def test_vn_helper_resolves_replay_before_quota_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs_dir = tmp_path / "outputs"
    winning = outputs_dir / "vn_assets" / "winner.png"
    winning.parent.mkdir(parents=True)
    winning.write_bytes(b"winner")

    class FullStorageService:
        async def get_vn_generated_file(self, *, user_id, source_ref):
            assert (user_id, source_ref) == (5, "vn_asset_item:4")
            return {"id": 17, "storage_path": "vn_assets/winner.png"}

        async def check_combined_quota(self, *_args, **_kwargs):
            raise RuntimeError("quota denied")

    async def get_service():
        return FullStorageService()

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", get_service)
    monkeypatch.setattr(
        generated_file_helpers.DatabasePaths, "get_user_outputs_dir",
        staticmethod(lambda _user_id: outputs_dir),
    )
    record = await generated_file_helpers.save_and_register_vn_asset_image(
        user_id=5, pack_id=1, item_id=4, asset_type="sprite", image_bytes=b"replacement",
    )

    assert record["id"] == 17
    assert list(outputs_dir.rglob("*.png")) == [winning]


@pytest.mark.asyncio
async def test_vn_registration_collision_removes_losing_file_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs_dir = tmp_path / "outputs"
    winning = outputs_dir / "vn_assets" / "winner.png"
    winning.parent.mkdir(parents=True)
    winning.write_bytes(b"winner")

    class ExistingStorageService:
        async def get_vn_generated_file(self, **_kwargs):
            return None

        async def get_generated_files_repo(self):
            return self

        async def get_live_file_by_storage_path(self, **_kwargs):
            return None

        async def check_combined_quota(self, *_args, **_kwargs):
            return True, {}

        async def register_generated_file(self, **_kwargs):
            return {"id": 17, "storage_path": "vn_assets/winner.png"}

    async def get_service():
        return ExistingStorageService()

    monkeypatch.setattr(generated_file_helpers, "get_storage_service", get_service)
    monkeypatch.setattr(
        generated_file_helpers.DatabasePaths, "get_user_outputs_dir",
        staticmethod(lambda _user_id: outputs_dir),
    )
    record = await generated_file_helpers.save_and_register_vn_asset_image(
        user_id=5, pack_id=1, item_id=4, asset_type="sprite", image_bytes=b"loser",
    )

    assert record["id"] == 17
    assert list(outputs_dir.rglob("*.png")) == [winning]


def test_generate_filename_sanitizes_prefix_and_extension():
    filename = generated_file_helpers._generate_filename("voice/evil name..", "mp3../")
    assert "/" not in filename
    assert "\\" not in filename
    assert " " not in filename
    assert filename.endswith(".mp3")


def test_generated_file_size_guard_uses_authnz_storage_error():
    with pytest.raises(AuthNZStorageError, match="exceeds maximum allowed size"):
        generated_file_helpers._validate_generated_file_size(
            generated_file_helpers.MAX_GENERATED_FILE_SIZE_BYTES + 1
        )


@pytest.mark.asyncio
async def test_save_and_register_image_preflights_quota_before_writing(
    tmp_path,
    monkeypatch,
):
    outputs_dir = tmp_path / "outputs"

    class RejectingStorageService:
        def __init__(self):
            self.preflight_called = False
            self.register_called = False

        async def check_combined_quota(self, user_id, new_bytes, **kwargs):
            self.preflight_called = True
            assert user_id == 42
            assert new_bytes == len(b"image-bytes")
            raise RuntimeError("quota denied")

        async def register_generated_file(self, **kwargs):
            self.register_called = True
            raise AssertionError("registration should not run when quota preflight fails")

    service = RejectingStorageService()

    monkeypatch.setattr(
        generated_file_helpers.DatabasePaths,
        "get_user_outputs_dir",
        staticmethod(lambda _user_id: outputs_dir),
    )

    async def fake_get_storage_service():
        return service

    monkeypatch.setattr(
        generated_file_helpers,
        "get_storage_service",
        fake_get_storage_service,
    )

    with pytest.raises(RuntimeError, match="quota denied"):
        await generated_file_helpers.save_and_register_image(
            user_id=42,
            image_bytes=b"image-bytes",
            check_quota=True,
        )

    assert service.preflight_called is True
    assert service.register_called is False
    assert not outputs_dir.exists()


@pytest.mark.asyncio
async def test_save_and_register_image_uses_resolved_outputs_root_for_storage_path(
    tmp_path,
    monkeypatch,
):
    real_outputs_dir = tmp_path / "real_outputs"
    linked_outputs_dir = tmp_path / "linked_outputs"
    real_outputs_dir.mkdir()
    try:
        linked_outputs_dir.symlink_to(real_outputs_dir, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not support directory symlinks")

    class RecordingStorageService:
        def __init__(self):
            self.kwargs = None

        async def check_combined_quota(self, user_id, new_bytes, **kwargs):
            _ = (user_id, new_bytes, kwargs)

        async def register_generated_file(self, **kwargs):
            self.kwargs = dict(kwargs)
            return {"storage_path": kwargs["storage_path"]}

    service = RecordingStorageService()

    monkeypatch.setattr(
        generated_file_helpers.DatabasePaths,
        "get_user_outputs_dir",
        staticmethod(lambda _user_id: linked_outputs_dir),
    )

    async def fake_get_storage_service():
        return service

    monkeypatch.setattr(
        generated_file_helpers,
        "get_storage_service",
        fake_get_storage_service,
    )

    record = await generated_file_helpers.save_and_register_image(
        user_id=42,
        image_bytes=b"image-bytes",
        check_quota=True,
    )

    assert service.kwargs is not None
    assert record["storage_path"] == service.kwargs["storage_path"]
    assert not record["storage_path"].startswith(str(tmp_path))
    assert (real_outputs_dir / record["storage_path"]).is_file()
