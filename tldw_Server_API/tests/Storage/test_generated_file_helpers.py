import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError as AuthNZStorageError
from tldw_Server_API.app.core.Storage import generated_file_helpers


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
