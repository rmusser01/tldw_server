from __future__ import annotations

import hashlib
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

    resolved = storage.resolve_visual_identity_asset_path(
        owner_user_id=1,
        relpath="assets/avatar.png",
        storage_root=storage_root,
    )
    assert resolved == (storage_root / "assets" / "avatar.png").resolve()


def test_rejects_declared_mime_header_mismatch(tmp_path: Path) -> None:
    storage = _storage_module()
    image_path = tmp_path / "avatar.png"
    Image.new("RGBA", (16, 16), "purple").save(image_path)

    with pytest.raises(ValueError, match="mime_mismatch"):
        storage.validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="neutral",
            storage_root=tmp_path / "store",
            content_type="image/jpeg",
        )


def test_copy_generated_file_record_to_expression_asset_uses_validated_storage(
    tmp_path: Path,
) -> None:
    storage = _storage_module()
    source_path = tmp_path / "generated.png"
    Image.new("RGBA", (12, 10), "green").save(source_path)
    generated_file = {
        "id": 99,
        "user_id": 1,
        "file_category": "image",
        "mime_type": "image/png",
        "is_deleted": False,
    }

    stored = storage.copy_generated_file_record_to_expression_asset(
        owner_user_id=1,
        pack_id=5,
        expression_key="happy",
        generated_file_record=generated_file,
        source_path=source_path,
        source_feature="vn_assets",
        storage_root=tmp_path / "store",
    )

    assert stored.content_type == "image/png"
    assert stored.width == 12
    assert stored.height == 10
    assert (tmp_path / "store" / stored.relpath).is_file()
