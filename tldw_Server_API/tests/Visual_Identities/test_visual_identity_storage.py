from __future__ import annotations

import hashlib
from types import SimpleNamespace
from pathlib import Path

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Visual_Identities.constraints import (
    MAX_EXPRESSION_IMAGE_DIMENSION,
)


def _storage_module():
    from tldw_Server_API.app.core.Visual_Identities import storage

    return storage


def _write_png(path: Path, *, size: tuple[int, int] = (16, 16), color: str = "purple") -> None:
    Image.new("RGBA", size, color).save(path, format="PNG")


def _patch_user_outputs_dir(
    monkeypatch: pytest.MonkeyPatch,
    outputs_dir: Path,
) -> None:
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_outputs_dir",
        staticmethod(lambda _user_id: outputs_dir),
    )


def test_get_user_visual_identities_dir_creates_expected_user_subdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_db_base = tmp_path / "user_databases"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))

    visuals_dir = DatabasePaths.get_user_visual_identities_dir(7)

    assert visuals_dir == (user_db_base / "7" / "visual_identities").resolve()
    assert visuals_dir.is_dir()


def test_rejects_image_over_max_dimension(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "large.png"
    Image.new("RGBA", (MAX_EXPRESSION_IMAGE_DIMENSION + 1, 32)).save(image_path)

    with pytest.raises(ValueError, match="image_dimensions_exceed_limit"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="happy",
            storage_root=tmp_path / "store",
        )


def test_animated_gif_original_is_stored_and_marked_animated(tmp_path: Path) -> None:
    storage = _storage_module()
    gif_path = tmp_path / "blink.gif"
    frames = [Image.new("RGBA", (8, 8), color) for color in ("red", "blue")]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )

    stored = storage.validate_and_store_visual_identity_asset(
        source_path=gif_path,
        owner_user_id=1,
        expression_key="surprised",
        storage_root=tmp_path / "store",
    )

    stored_path = (tmp_path / "store" / stored.relpath).resolve()
    assert stored.content_type == "image/gif"
    assert stored.bytes == gif_path.stat().st_size
    assert stored.sha256 == hashlib.sha256(gif_path.read_bytes()).hexdigest()
    assert stored.width == 8
    assert stored.height == 8
    assert stored.is_animated is True
    assert stored.frame_count == 2
    assert stored.duration_ms == 240
    assert stored_path.read_bytes() == gif_path.read_bytes()


def test_avif_is_rejected_when_capability_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    avif_path = tmp_path / "avatar.avif"
    avif_path.write_bytes(b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.constraints.supports_avif",
        lambda: False,
    )

    with pytest.raises(ValueError, match="unsupported_mime_type"):
        storage.validate_and_store_visual_identity_asset(
            source_path=avif_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_resolve_visual_identity_asset_path_rejects_traversal(tmp_path: Path) -> None:
    storage = _storage_module()
    storage_root = tmp_path / "store"

    with pytest.raises(ValueError, match="invalid_storage_path"):
        storage.resolve_visual_identity_asset_path(
            owner_user_id=1,
            relpath="../escape.png",
            storage_root=storage_root,
        )
    with pytest.raises(ValueError, match="invalid_storage_path"):
        storage.resolve_visual_identity_asset_path(
            owner_user_id=1,
            relpath="/tmp/escape.png",
            storage_root=storage_root,
        )
    with pytest.raises(ValueError, match="invalid_storage_path"):
        storage.resolve_visual_identity_asset_path(
            owner_user_id=1,
            relpath=r"assets\escape.png",
            storage_root=storage_root,
        )

    resolved = storage.resolve_visual_identity_asset_path(
        owner_user_id=1,
        relpath="assets/avatar.png",
        storage_root=storage_root,
    )
    assert resolved == (storage_root / "assets" / "avatar.png").resolve()


def test_rejects_declared_mime_header_mismatch(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path)

    with pytest.raises(ValueError, match="mime_mismatch"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
            content_type="image/jpeg",
        )


def test_rejects_extension_mismatch_after_mime_detection(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.jpg"
    _write_png(image_path)

    with pytest.raises(ValueError, match="extension_mismatch"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_rejects_unsupported_extension_after_mime_detection(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.txt"
    _write_png(image_path)

    with pytest.raises(ValueError, match="unsupported_extension"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_rechecks_size_after_reading_source_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path)
    original_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        if self == image_path:
            original = original_stat(self)
            return SimpleNamespace(st_mode=original.st_mode, st_size=1)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.constraints.MAX_EXPRESSION_ASSET_BYTES",
        1,
    )

    with pytest.raises(ValueError, match="file_too_large"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_duplicate_hash_target_with_same_bytes_dedupes_successfully(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path, color="green")

    first = storage.validate_and_store_visual_identity_asset(
        source_path=image_path,
        owner_user_id=1,
        expression_key="neutral",
        storage_root=tmp_path / "store",
    )
    second = storage.validate_and_store_visual_identity_asset(
        source_path=image_path,
        owner_user_id=1,
        expression_key="neutral",
        storage_root=tmp_path / "store",
    )

    assert second == first
    assert (tmp_path / "store" / second.relpath).read_bytes() == image_path.read_bytes()


def test_duplicate_hash_target_with_corrupt_existing_file_fails(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path, color="green")

    stored = storage.validate_and_store_visual_identity_asset(
        source_path=image_path,
        owner_user_id=1,
        expression_key="neutral",
        storage_root=tmp_path / "store",
    )
    stored_path = tmp_path / "store" / stored.relpath
    stored_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="stored_asset_hash_mismatch"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_safe_write_fails_closed_when_hardlink_publish_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path, color="green")

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(storage.os, "link", fail_link)

    with pytest.raises(ValueError, match="stored_asset_publish_unavailable"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_same_dir_temp_write_closes_fd_when_fdopen_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    target_path = tmp_path / "asset.png"
    closed_fds: list[int] = []
    real_close = storage.os.close

    def fail_fdopen(fd: int, mode: str):
        raise OSError("fdopen failed")

    def record_close(fd: int) -> None:
        closed_fds.append(fd)
        real_close(fd)

    monkeypatch.setattr(storage.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(storage.os, "close", record_close)

    with pytest.raises(OSError, match="fdopen failed"):
        storage._write_same_dir_temp_file(target_path, b"content")

    assert len(closed_fds) == 1
    assert list(tmp_path.glob(f".{target_path.name}.*.tmp")) == []


def test_duplicate_hash_preview_with_corrupt_existing_file_fails(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path, color="green")

    stored = storage.validate_and_store_visual_identity_asset(
        source_path=image_path,
        owner_user_id=1,
        expression_key="neutral",
        storage_root=tmp_path / "store",
    )
    assert stored.preview_relpath is not None
    preview_path = tmp_path / "store" / stored.preview_relpath
    preview_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="stored_asset_hash_mismatch"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
        )


def test_rejects_animated_frame_count_over_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    gif_path = tmp_path / "too-many.gif"
    frames = [Image.new("RGBA", (2, 2), color) for color in ("red", "blue")]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.constraints.MAX_EXPRESSION_FRAME_COUNT",
        1,
    )

    with pytest.raises(ValueError, match="image_frame_count_exceeds_limit"):
        storage.validate_and_store_visual_identity_asset(
            source_path=gif_path,
            owner_user_id=1,
            expression_key="surprised",
            storage_root=tmp_path / "store",
        )


def test_preview_failure_does_not_reject_valid_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    _write_png(image_path)

    def fail_save(self: Image.Image, fp, format=None, **params) -> None:
        raise OSError("preview unavailable")

    monkeypatch.setattr(Image.Image, "save", fail_save)

    stored = storage.validate_and_store_visual_identity_asset(
        source_path=image_path,
        owner_user_id=1,
        expression_key="neutral",
        storage_root=tmp_path / "store",
    )

    assert stored.preview_relpath is None
    assert (tmp_path / "store" / stored.relpath).is_file()


def test_copy_generated_file_record_to_expression_asset_uses_validated_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage_module()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _patch_user_outputs_dir(monkeypatch, outputs_dir)
    source_path = outputs_dir / "generated.png"
    _write_png(source_path, size=(12, 10), color="green")
    generated_file = {
        "id": 99,
        "user_id": 1,
        "file_category": "image",
        "source_feature": "vn_assets",
        "storage_path": "generated.png",
        "mime_type": "image/png",
        "is_deleted": False,
    }

    stored = storage.copy_generated_file_record_to_expression_asset(
        owner_user_id=1,
        pack_id=5,
        expression_key="happy",
        generated_file_record=generated_file,
        source_feature="vn_assets",
        storage_root=tmp_path / "store",
    )

    assert stored.content_type == "image/png"
    assert stored.width == 12
    assert stored.height == 10
    assert (tmp_path / "store" / stored.relpath).is_file()


@pytest.mark.parametrize("source_feature", [None, "", "   "])
def test_copy_generated_file_record_requires_expected_source_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_feature: str | None,
) -> None:
    storage = _storage_module()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _patch_user_outputs_dir(monkeypatch, outputs_dir)
    source_path = outputs_dir / "generated.png"
    _write_png(source_path)
    generated_file = {
        "id": 99,
        "user_id": 1,
        "file_category": "image",
        "source_feature": "vn_assets",
        "storage_path": "generated.png",
        "mime_type": "image/png",
        "is_deleted": False,
    }

    with pytest.raises(ValueError, match="generated_file_not_found"):
        storage.copy_generated_file_record_to_expression_asset(
            owner_user_id=1,
            pack_id=5,
            expression_key="happy",
            generated_file_record=generated_file,
            source_feature=source_feature,
            storage_root=tmp_path / "store",
        )


def test_copy_generated_file_record_rejects_public_source_path_override(
    tmp_path: Path,
) -> None:
    storage = _storage_module()
    source_path = tmp_path / "outside.png"
    _write_png(source_path)
    generated_file = {
        "id": 99,
        "user_id": 1,
        "file_category": "image",
        "source_feature": "vn_assets",
        "storage_path": "missing.png",
        "mime_type": "image/png",
        "is_deleted": False,
    }

    with pytest.raises(TypeError, match="source_path"):
        storage.copy_generated_file_record_to_expression_asset(
            owner_user_id=1,
            pack_id=5,
            expression_key="happy",
            generated_file_record=generated_file,
            source_path=source_path,
            source_feature="vn_assets",
            storage_root=tmp_path / "store",
        )


@pytest.mark.parametrize(
    ("record_updates", "source_feature", "expected_error"),
    [
        ({"user_id": 2}, "vn_assets", "generated_file_not_found"),
        ({"is_deleted": True}, "vn_assets", "generated_file_not_found"),
        ({"file_category": "tts_audio"}, "vn_assets", "generated_file_not_image"),
        ({}, "vn_assets", "generated_file_not_found"),
        ({"source_feature": ""}, "vn_assets", "generated_file_not_found"),
        ({"source_feature": "   "}, "vn_assets", "generated_file_not_found"),
        ({"source_feature": "image_gen"}, "vn_assets", "generated_file_not_found"),
    ],
)
def test_copy_generated_file_record_rejects_invalid_record_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_updates: dict[str, object],
    source_feature: str,
    expected_error: str,
) -> None:
    storage = _storage_module()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _patch_user_outputs_dir(monkeypatch, outputs_dir)
    source_path = outputs_dir / "generated.png"
    _write_png(source_path)
    generated_file = {
        "id": 99,
        "user_id": 1,
        "file_category": "image",
        "storage_path": "generated.png",
        "mime_type": "image/png",
        "is_deleted": False,
    }
    generated_file.update(record_updates)

    with pytest.raises(ValueError, match=expected_error):
        storage.copy_generated_file_record_to_expression_asset(
            owner_user_id=1,
            pack_id=5,
            expression_key="happy",
            generated_file_record=generated_file,
            source_feature=source_feature,
            storage_root=tmp_path / "store",
        )


@pytest.mark.parametrize(
    "storage_path",
    [
        "../generated.png",
        "/tmp/generated.png",
        r"nested\generated.png",
    ],
)
def test_copy_generated_file_record_rejects_unsafe_storage_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_path: str,
) -> None:
    storage = _storage_module()
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    _patch_user_outputs_dir(monkeypatch, outputs_dir)
    generated_file = {
        "id": 99,
        "user_id": 1,
        "file_category": "image",
        "source_feature": "vn_assets",
        "storage_path": storage_path,
        "mime_type": "image/png",
        "is_deleted": False,
    }

    with pytest.raises(ValueError, match="generated_file_not_found"):
        storage.copy_generated_file_record_to_expression_asset(
            owner_user_id=1,
            pack_id=5,
            expression_key="happy",
            generated_file_record=generated_file,
            source_feature="vn_assets",
            storage_root=tmp_path / "store",
        )
