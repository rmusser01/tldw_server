from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_service import (
    MAX_VISUAL_UPLOAD_BYTES,
    PersonaVisualService,
    PersonaVisualServiceError,
)

pytestmark = pytest.mark.unit


def _png_bytes(width: int = 1, height: int = 1) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (width, height), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _animated_bytes(image_format: str) -> bytes:
    frames = [
        Image.new("RGBA", (1, 1), (255, 0, 0, 255)),
        Image.new("RGBA", (1, 1), (0, 0, 255, 255)),
    ]
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format=image_format,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


def _valid_manifest(asset_id: str) -> dict[str, object]:
    states = {
        "idle": {"animation_id": "idle"},
        "wake_armed": {"animation_id": "idle"},
        "listening": {"animation_id": "idle"},
        "thinking": {"animation_id": "idle"},
        "speaking": {"animation_id": "idle"},
        "tool_running": {"animation_id": "idle"},
        "approval_needed": {"animation_id": "idle"},
        "error": {"animation_id": "idle"},
        "offline": {"animation_id": "idle"},
    }
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": states,
        "animations": {
            "idle": {
                "frames": [{"asset_id": asset_id, "duration_ms": 100}],
                "frame_rate": 1,
            }
        },
    }


def _patch_visuals_dir(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
    )


@pytest.fixture()
def db_instance(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "persona_visual_service.sqlite", "persona-visual-service-test")
    yield db
    db.close_connection()


@pytest.fixture()
def service(db_instance: CharactersRAGDB) -> PersonaVisualService:
    return PersonaVisualService(db_instance)


def _create_pack(db: CharactersRAGDB, *, user_id: str = "user-1", title: str = "Pack") -> tuple[str, dict]:
    persona_id = db.create_persona_profile({"user_id": user_id, "name": f"{title} Persona"})
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title=title,
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
        },
    )
    return persona_id, pack


def _review_and_activate(
    service: PersonaVisualService,
    *,
    persona_id: str,
    pack: dict,
    user_id: str = "user-1",
) -> dict:
    review = service.review_pack(
        pack_id=str(pack["id"]),
        user_id=user_id,
        reviewer_user_id=user_id,
        expected_version=int(pack["version"]),
    )
    return service.activate_pack(
        persona_id=persona_id,
        user_id=user_id,
        pack_id=str(pack["id"]),
        expected_version=int(pack["version"]),
        reviewed_fingerprint=str(review["fingerprint"]),
    )


def test_db_allows_explicit_cross_persona_parent_for_duplicate_path(db_instance: CharactersRAGDB) -> None:
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
        companion_behavior={
            "schema_version": 1,
            "entries": [{
                "state": "idle",
                "trigger": "ambient",
                "category": "idle_variant",
            }],
        },
    )

    target_pack = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        title="Duplicate",
        parent_pack_id=source_pack["id"],
        parent_persona_id=source_persona_id,
        status="failed",
        provenance="mixed",
    )
    updated = db_instance.update_persona_visual_pack_status(
        pack_id=target_pack["id"],
        persona_id=target_persona_id,
        user_id=user_id,
        status="draft",
        expected_version=target_pack["version"],
    )

    assert updated["parent_pack_id"] == source_pack["id"]
    assert updated["status"] == "draft"


def _manifest_with_all_reference_shapes(asset_a: str, asset_b: str) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {
            "idle": {"animation_id": "idle"},
            "wake_armed": {"animation_id": "idle"},
            "listening": {"animation_id": "idle"},
            "thinking": {"animation_id": "idle"},
            "speaking": {"animation_id": "idle"},
            "tool_running": {"animation_id": "idle"},
            "approval_needed": {"animation_id": "idle"},
            "error": {"animation_id": "idle"},
            "offline": {"animation_id": "idle"},
        },
        "animations": {
            "idle": {
                "frames": [
                    {"asset_id": asset_a, "duration_ms": 100},
                    {"asset_id": asset_b, "duration_ms": 100},
                ],
                "asset_ids": [asset_a],
                "preview_asset_id": asset_b,
                "frame_rate": 2,
            }
        },
    }


def test_duplicate_pack_to_persona_preserves_all_pack_assets_and_remaps_manifest(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )
    asset_a = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=2, height=2),
        mime_type="image/png",
        original_filename="idle-a.png",
        asset_role="frame",
    )
    asset_b = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=3, height=3),
        mime_type="image/png",
        original_filename="idle-b.png",
        asset_role="preview",
    )
    unused = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=4, height=4),
        mime_type="image/png",
        original_filename="unused.png",
        asset_role="generated_candidate",
    )
    neutral_anchor = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=5, height=5),
        mime_type="image/png",
        original_filename="neutral-anchor.png",
        asset_role="still_pose",
    )
    updated_source = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_manifest_with_all_reference_shapes(asset_a["id"], asset_b["id"]),
        expected_version=source_pack["version"],
    )
    target_active = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        title="Existing Target Active",
        status="draft",
    )
    target_asset = service.create_asset_from_upload(
        persona_id=target_persona_id,
        user_id=user_id,
        pack_id=target_active["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="target.png",
        asset_role="frame",
    )
    target_active = db_instance.update_persona_visual_pack_manifest(
        pack_id=target_active["id"],
        persona_id=target_persona_id,
        user_id=user_id,
        manifest=_valid_manifest(target_asset["id"]),
        expected_version=target_active["version"],
    )
    _review_and_activate(service, persona_id=source_persona_id, pack=updated_source)
    _review_and_activate(service, persona_id=target_persona_id, pack=target_active)

    duplicated = service.duplicate_pack_to_persona(
        source_persona_id=source_persona_id,
        user_id=user_id,
        pack_id=updated_source["id"],
        target_persona_id=target_persona_id,
        title="Target Draft",
    )

    assert duplicated["status"] == "draft"
    assert duplicated["persona_id"] == target_persona_id
    assert duplicated["parent_pack_id"] == source_pack["id"]
    assert duplicated["companion_behavior"] == source_pack["companion_behavior"]
    copied_assets = db_instance.list_persona_visual_assets(
        pack_id=duplicated["id"],
        persona_id=target_persona_id,
        user_id=user_id,
    )
    assert len(copied_assets) == 4
    copied_ids = {asset["id"] for asset in copied_assets}
    copied_checksums = {asset["checksum_sha256"] for asset in copied_assets}
    copied_roles = {asset["asset_role"] for asset in copied_assets}
    assert asset_a["id"] not in copied_ids
    assert asset_b["id"] not in copied_ids
    assert unused["checksum_sha256"] in copied_checksums
    assert neutral_anchor["checksum_sha256"] in copied_checksums
    assert {"frame", "preview", "generated_candidate", "still_pose"}.issubset(copied_roles)
    assert all(
        f"persona_visuals/{target_persona_id}/{duplicated['id']}/" in asset["storage_key"]
        for asset in copied_assets
    )
    remapped_animation = duplicated["manifest"]["animations"]["idle"]
    assert {frame["asset_id"] for frame in remapped_animation["frames"]}.issubset(copied_ids)
    assert set(remapped_animation["asset_ids"]).issubset(copied_ids)
    assert remapped_animation["preview_asset_id"] in copied_ids
    assert db_instance.get_active_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
    )["id"] == updated_source["id"]
    assert db_instance.get_active_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
    )["id"] == target_active["id"]


def test_duplicate_pack_rejects_same_persona_target(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
) -> None:
    persona_id, pack = _create_pack(db_instance)

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.duplicate_pack_to_persona(
            source_persona_id=persona_id,
            user_id="user-1",
            pack_id=pack["id"],
            target_persona_id=persona_id,
        )

    assert exc_info.value.code == "same_persona_target_unsupported"


def test_duplicate_pack_rejects_missing_manifest_asset_row(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
) -> None:
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )
    updated_source = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_valid_manifest("does-not-exist"),
        expected_version=source_pack["version"],
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.duplicate_pack_to_persona(
            source_persona_id=source_persona_id,
            user_id=user_id,
            pack_id=updated_source["id"],
            target_persona_id=target_persona_id,
        )

    assert exc_info.value.code == "invalid_manifest"


def test_duplicate_pack_rejects_missing_source_asset_file(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )
    asset = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="idle.png",
        asset_role="frame",
    )
    Path(asset["storage_path"]).unlink()
    updated_source = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_valid_manifest(asset["id"]),
        expected_version=source_pack["version"],
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.duplicate_pack_to_persona(
            source_persona_id=source_persona_id,
            user_id=user_id,
            pack_id=updated_source["id"],
            target_persona_id=target_persona_id,
        )

    assert exc_info.value.code == "source_asset_missing"


def test_duplicate_pack_rejects_source_asset_checksum_mismatch(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )
    asset = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="idle.png",
        asset_role="frame",
    )
    Path(asset["storage_path"]).write_bytes(b"not the original image bytes")
    updated_source = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_valid_manifest(asset["id"]),
        expected_version=source_pack["version"],
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.duplicate_pack_to_persona(
            source_persona_id=source_persona_id,
            user_id=user_id,
            pack_id=updated_source["id"],
            target_persona_id=target_persona_id,
        )

    assert exc_info.value.code == "source_asset_checksum_mismatch"


def test_duplicate_pack_cleans_up_copied_files_after_partial_failure(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visuals_root = tmp_path / "visuals"
    _patch_visuals_dir(monkeypatch, visuals_root)
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )
    asset_a = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=2, height=2),
        mime_type="image/png",
        original_filename="idle-a.png",
        asset_role="frame",
    )
    asset_b = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=3, height=3),
        mime_type="image/png",
        original_filename="idle-b.png",
        asset_role="preview",
    )
    updated_source = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_manifest_with_all_reference_shapes(asset_a["id"], asset_b["id"]),
        expected_version=source_pack["version"],
    )

    original_create_asset = service.create_asset_from_upload
    target_copy_count = 0

    def _fail_after_first_target_copy(**kwargs: object) -> dict[str, object]:
        nonlocal target_copy_count
        if kwargs.get("persona_id") == target_persona_id:
            target_copy_count += 1
            if target_copy_count > 1:
                raise PersonaVisualServiceError("copy_failed", "Injected copy failure.")
        return original_create_asset(**kwargs)

    monkeypatch.setattr(service, "create_asset_from_upload", _fail_after_first_target_copy)

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.duplicate_pack_to_persona(
            source_persona_id=source_persona_id,
            user_id=user_id,
            pack_id=updated_source["id"],
            target_persona_id=target_persona_id,
        )

    assert exc_info.value.code == "copy_failed"
    target_packs = db_instance.list_persona_visual_packs(
        persona_id=target_persona_id,
        user_id=user_id,
    )
    assert target_packs == []
    deleted_target_packs = db_instance.list_persona_visual_packs(
        persona_id=target_persona_id,
        user_id=user_id,
        include_deleted=True,
    )
    assert len(deleted_target_packs) == 1
    assert deleted_target_packs[0]["status"] == "failed"
    assert deleted_target_packs[0]["deleted"] is True
    copied_asset_rows = db_instance.execute_query(
        "SELECT deleted, storage_key FROM persona_visual_assets WHERE user_id = ? AND persona_id = ?",
        (user_id, target_persona_id),
    ).fetchall()
    assert len(copied_asset_rows) == 1
    assert bool(copied_asset_rows[0]["deleted"]) is True
    copied_path = service._asset_storage_path(
        user_id=user_id,
        storage_key=str(copied_asset_rows[0]["storage_key"]),
    )
    assert not copied_path.exists()
    assert db_instance.get_active_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
    ) is None
    assert visuals_root.exists()


def test_merge_candidate_patch_replaces_authored_trigger_by_id() -> None:
    merged = PersonaVisualService._merge_candidate_patch(
        {
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
            "authored_triggers": [
                {
                    "id": "gesture-wave",
                    "source": "keyword",
                    "match": "hello",
                    "state": "idle",
                    "priority": 10,
                }
            ],
        },
        {
            "authored_triggers": [
                {
                    "id": "gesture-wave",
                    "source": "keyword",
                    "match": "hello",
                    "state": "speaking",
                    "priority": 20,
                },
                {
                    "id": "gesture-nod",
                    "source": "keyword",
                    "match": "yes",
                    "state": "listening",
                    "priority": 5,
                },
            ]
        },
    )

    triggers = merged["authored_triggers"]
    assert [trigger["id"] for trigger in triggers] == ["gesture-wave", "gesture-nod"]
    assert triggers[0]["state"] == "speaking"
    assert triggers[0]["priority"] == 20


def test_service_rejects_unsupported_mime_type_without_writing(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visuals_root = tmp_path / "visuals"
    _patch_visuals_dir(monkeypatch, visuals_root)
    persona_id, pack = _create_pack(db_instance)

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.create_asset_from_upload(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=pack["id"],
            content=b"plain text",
            mime_type="text/plain",
            original_filename="pose.txt",
            asset_role="frame",
        )

    assert exc_info.value.code == "unsupported_mime_type"
    assert not visuals_root.exists()


def test_service_rejects_oversized_upload_without_writing(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visuals_root = tmp_path / "visuals"
    _patch_visuals_dir(monkeypatch, visuals_root)
    persona_id, pack = _create_pack(db_instance)

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.create_asset_from_upload(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=pack["id"],
            content=b"x" * (MAX_VISUAL_UPLOAD_BYTES + 1),
            mime_type="image/png",
            original_filename="pose.png",
            asset_role="frame",
        )

    assert exc_info.value.code == "upload_too_large"
    assert not visuals_root.exists()


def test_service_writes_asset_under_user_visuals_dir(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visuals_root = tmp_path / "visuals"
    _patch_visuals_dir(monkeypatch, visuals_root)
    persona_id, pack = _create_pack(db_instance)

    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
        content=_png_bytes(width=2, height=3),
        mime_type="image/png",
        original_filename="pose.png",
        asset_role="frame",
    )

    stored_path = Path(asset["storage_path"])
    assert stored_path.read_bytes() == _png_bytes(width=2, height=3)
    assert stored_path.resolve().is_relative_to(visuals_root.resolve())
    assert asset["storage_key"].startswith(f"persona_visuals/{persona_id}/{pack['id']}/")
    assert asset["width"] == 2
    assert asset["height"] == 3


def test_service_activation_rejects_invalid_manifest(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="idle.png",
        asset_role="frame",
    )
    db_instance.update_persona_visual_pack_manifest(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {"frames": [{"asset_id": asset["id"], "duration_ms": 100}]}
            },
        },
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.review_pack(
            pack_id=pack["id"],
            user_id="user-1",
            reviewer_user_id="user-1",
            expected_version=2,
        )

    assert exc_info.value.code == "invalid_manifest"
    assert db_instance.get_active_persona_visual_pack(persona_id=persona_id, user_id="user-1") is None


def test_live2d_pack_row_cannot_review_or_activate_sprite_manifest(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Mismatched Renderer Persona"}
    )
    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Mismatched Renderer Pack",
        renderer_type="live2d",
    )
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=str(pack["id"]),
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    pack = db_instance.update_persona_visual_pack_payload(
        pack_id=str(pack["id"]),
        user_id="user-1",
        manifest=_valid_manifest(str(asset["id"])),
        companion_behavior=None,
        expected_version=int(pack["version"]),
    )

    with pytest.raises(PersonaVisualServiceError) as review_error:
        service.review_pack(
            pack_id=str(pack["id"]),
            user_id="user-1",
            reviewer_user_id="user-1",
            expected_version=int(pack["version"]),
        )
    with pytest.raises(PersonaVisualServiceError) as activation_error:
        service.activate_pack(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=str(pack["id"]),
            expected_version=int(pack["version"]),
            reviewed_fingerprint="0" * 64,
        )

    assert review_error.value.code == "invalid_renderer_contract"
    assert activation_error.value.code == "invalid_renderer_contract"
    assert db_instance.get_active_persona_visual_pack(
        persona_id=persona_id, user_id="user-1"
    ) is None


def test_service_activation_archives_previous_active_pack(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, first = _create_pack(db_instance, title="First")
    second = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Second",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
        },
    )
    first_asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=first["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="first.png",
        asset_role="frame",
    )
    second_asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=second["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="second.png",
        asset_role="frame",
    )
    db_instance.update_persona_visual_pack_manifest(
        pack_id=first["id"],
        persona_id=persona_id,
        user_id="user-1",
        manifest=_valid_manifest(first_asset["id"]),
    )
    db_instance.update_persona_visual_pack_manifest(
        pack_id=second["id"],
        persona_id=persona_id,
        user_id="user-1",
        manifest=_valid_manifest(second_asset["id"]),
    )

    first = db_instance.get_persona_visual_pack(
        pack_id=first["id"], persona_id=persona_id, user_id="user-1"
    )
    second = db_instance.get_persona_visual_pack(
        pack_id=second["id"], persona_id=persona_id, user_id="user-1"
    )
    _review_and_activate(service, persona_id=persona_id, pack=first)
    _review_and_activate(service, persona_id=persona_id, pack=second)

    active = db_instance.get_active_persona_visual_pack(persona_id=persona_id, user_id="user-1")
    packs = db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")
    assert active is not None
    assert active["id"] == second["id"]
    statuses = {pack["id"]: pack["status"] for pack in packs}
    assert statuses[first["id"]] == "archived"
    assert statuses[second["id"]] == "active"


def test_service_deactivate_reverts_to_derived_buddy(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="idle.png",
        asset_role="frame",
    )
    db_instance.update_persona_visual_pack_manifest(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        manifest=_valid_manifest(asset["id"]),
    )

    pack = db_instance.get_persona_visual_pack(
        pack_id=pack["id"], persona_id=persona_id, user_id="user-1"
    )
    _review_and_activate(service, persona_id=persona_id, pack=pack)
    service.deactivate_pack(persona_id=persona_id, user_id="user-1")

    assert db_instance.get_active_persona_visual_pack(persona_id=persona_id, user_id="user-1") is None
    archived = db_instance.get_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
    )
    assert archived is not None
    assert archived["status"] == "archived"


def test_deactivated_revision_remains_sealed_and_reactivates_unchanged(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An activated revision stays immutable after archive but remains reusable."""
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=str(pack["id"]),
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    pack = db_instance.update_persona_visual_pack_payload(
        pack_id=str(pack["id"]),
        user_id="user-1",
        manifest=_valid_manifest(str(asset["id"])),
        companion_behavior={"schema_version": 1, "entries": []},
        expected_version=int(pack["version"]),
    )
    review = service.review_pack(
        pack_id=str(pack["id"]),
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=int(pack["version"]),
    )
    service.activate_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=str(pack["id"]),
        expected_version=int(pack["version"]),
        reviewed_fingerprint=str(review["fingerprint"]),
    )
    service.deactivate_pack(persona_id=persona_id, user_id="user-1")
    archived = db_instance.get_persona_visual_pack(
        pack_id=str(pack["id"]), persona_id=persona_id, user_id="user-1"
    )
    assert archived is not None

    with pytest.raises(InputError, match="immutable"):
        db_instance.update_persona_visual_pack_payload(
            pack_id=str(pack["id"]),
            user_id="user-1",
            manifest=archived["manifest"],
            companion_behavior=None,
            expected_version=int(archived["version"]),
        )
    with pytest.raises(InputError, match="immutable"):
        service.create_asset_from_upload(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=str(pack["id"]),
            content=_png_bytes(),
            mime_type="image/png",
            original_filename="new.png",
        )
    with pytest.raises(InputError, match="immutable"):
        db_instance.update_persona_visual_pack_status(
            pack_id=str(pack["id"]),
            persona_id=persona_id,
            user_id="user-1",
            status="draft",
            expected_version=int(archived["version"]),
        )
    with pytest.raises(InputError, match="immutable"):
        db_instance.soft_delete_persona_visual_pack_with_assets(
            pack_id=str(pack["id"]),
            persona_id=persona_id,
            user_id="user-1",
            expected_version=int(archived["version"]),
        )

    with pytest.raises(PersonaVisualServiceError) as stale_error:
        service.activate_pack(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=str(pack["id"]),
            expected_version=int(archived["version"]) - 1,
            reviewed_fingerprint=str(review["fingerprint"]),
        )
    reactivated = service.activate_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=str(pack["id"]),
        expected_version=int(archived["version"]),
        reviewed_fingerprint=str(review["fingerprint"]),
    )

    assert stale_error.value.code == "activation_conflict"
    assert reactivated["status"] == "active"
    assert reactivated["manifest"] == archived["manifest"]
    assert reactivated["companion_behavior"] == archived["companion_behavior"]


def test_review_validates_without_mutating_payload_and_activation_writes_no_normalized_payload(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    stored = db_instance.update_persona_visual_pack_payload(
        pack_id=pack["id"],
        user_id="user-1",
        manifest=_valid_manifest(asset["id"]),
        companion_behavior={"schema_version": 1, "entries": []},
        expected_version=pack["version"],
    )
    payload_before = (stored["manifest"], stored["companion_behavior"], stored["version"])

    review = service.review_pack(
        pack_id=pack["id"],
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=stored["version"],
    )
    after_review = db_instance.get_persona_visual_pack(
        pack_id=pack["id"], persona_id=persona_id, user_id="user-1"
    )
    active = service.activate_pack(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        expected_version=stored["version"],
        reviewed_fingerprint=review["fingerprint"],
    )

    assert (after_review["manifest"], after_review["companion_behavior"], after_review["version"]) == payload_before
    assert active["manifest"] == payload_before[0]
    assert active["companion_behavior"] == payload_before[1]
    assert active["version"] == payload_before[2] + 1


def test_behavior_changes_invalidate_review_and_fork_copies_assets(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    pack = db_instance.update_persona_visual_pack_manifest(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
        manifest=_valid_manifest(asset["id"]),
        expected_version=pack["version"],
    )
    old_review = service.review_pack(
        pack_id=pack["id"],
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=pack["version"],
    )

    fork = service.fork_pack_revision(
        pack_id=pack["id"],
        user_id="user-1",
        expected_version=pack["version"],
        manifest=pack["manifest"],
        companion_behavior={"schema_version": 1, "entries": []},
    )
    new_review = service.review_pack(
        pack_id=fork["id"],
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=fork["version"],
    )

    assert fork["status"] == "draft"
    assert fork["parent_pack_id"] == pack["id"]
    assert fork["revision_number"] == pack["revision_number"] + 1
    assert fork["companion_behavior"] == {"schema_version": 1, "entries": []}
    assert new_review["fingerprint"] != old_review["fingerprint"]
    assert {asset["checksum_sha256"] for asset in fork["assets"]} == {asset["checksum_sha256"]}


def test_fork_rejects_stale_source_before_listing_or_copying_assets(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale revision fork must fail before any source asset work begins."""
    _persona_id, pack = _create_pack(db_instance)
    list_assets = Mock(side_effect=AssertionError("assets must not be listed for a stale source"))
    monkeypatch.setattr(db_instance, "list_persona_visual_assets", list_assets)

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.fork_pack_revision(
            pack_id=pack["id"],
            user_id="user-1",
            expected_version=pack["version"] + 1,
            manifest=pack["manifest"],
            companion_behavior=None,
        )

    assert exc_info.value.code == "fork_conflict"
    list_assets.assert_not_called()


def test_activation_rejects_stale_review_fingerprint(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    pack = db_instance.update_persona_visual_pack_manifest(
        pack_id=pack["id"], persona_id=persona_id, user_id="user-1",
        manifest=_valid_manifest(asset["id"]), expected_version=pack["version"],
    )
    review = service.review_pack(
        pack_id=pack["id"], user_id="user-1", reviewer_user_id="user-1",
        expected_version=pack["version"],
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.activate_pack(
            pack_id=pack["id"], persona_id=persona_id, user_id="user-1",
            expected_version=pack["version"], reviewed_fingerprint="0" * 64,
        )

    assert exc_info.value.code == "stale_review"
    assert review["fingerprint"] != "0" * 64


def test_real_review_cannot_activate_after_inactive_payload_version_changes(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=str(pack["id"]),
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    pack = db_instance.update_persona_visual_pack_payload(
        pack_id=str(pack["id"]),
        user_id="user-1",
        manifest=_valid_manifest(str(asset["id"])),
        companion_behavior=None,
        expected_version=int(pack["version"]),
    )
    review = service.review_pack(
        pack_id=str(pack["id"]),
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=int(pack["version"]),
    )
    changed = db_instance.update_persona_visual_pack_payload(
        pack_id=str(pack["id"]),
        user_id="user-1",
        manifest=pack["manifest"],
        companion_behavior={"schema_version": 1, "entries": []},
        expected_version=int(pack["version"]),
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.activate_pack(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=str(pack["id"]),
            expected_version=int(changed["version"]),
            reviewed_fingerprint=str(review["fingerprint"]),
        )

    assert changed["version"] == review["pack_version"] + 1
    assert exc_info.value.code == "stale_review"
    assert db_instance.get_active_persona_visual_pack(
        persona_id=persona_id, user_id="user-1"
    ) is None


def test_activation_reprobes_protected_bytes_after_successful_review(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=str(pack["id"]),
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="still.png",
    )
    pack = db_instance.update_persona_visual_pack_payload(
        pack_id=str(pack["id"]),
        user_id="user-1",
        manifest=_valid_manifest(str(asset["id"])),
        companion_behavior=None,
        expected_version=int(pack["version"]),
    )
    review = service.review_pack(
        pack_id=str(pack["id"]),
        user_id="user-1",
        reviewer_user_id="user-1",
        expected_version=int(pack["version"]),
    )
    Path(str(asset["storage_path"])).write_bytes(_animated_bytes("GIF"))

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.activate_pack(
            persona_id=persona_id,
            user_id="user-1",
            pack_id=str(pack["id"]),
            expected_version=int(review["pack_version"]),
            reviewed_fingerprint=str(review["fingerprint"]),
        )

    assert exc_info.value.code == "asset_checksum_mismatch"
    assert db_instance.get_active_persona_visual_pack(
        persona_id=persona_id, user_id="user-1"
    ) is None


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    [("GIF", "image/gif"), ("WEBP", "image/webp")],
)
def test_review_rejects_animated_raster_bytes_for_static_coverage(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_format: str,
    mime_type: str,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    persona_id, pack = _create_pack(db_instance)
    asset = service.create_asset_from_upload(
        persona_id=persona_id, user_id="user-1", pack_id=pack["id"],
        content=_animated_bytes(image_format), mime_type=mime_type,
        original_filename=f"animated.{image_format.lower()}",
    )
    pack = db_instance.update_persona_visual_pack_manifest(
        pack_id=pack["id"], persona_id=persona_id, user_id="user-1",
        manifest=_valid_manifest(asset["id"]), expected_version=pack["version"],
    )

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.review_pack(
            pack_id=pack["id"], user_id="user-1", reviewer_user_id="user-1",
            expected_version=pack["version"],
        )

    assert exc_info.value.code == "invalid_static_coverage"
