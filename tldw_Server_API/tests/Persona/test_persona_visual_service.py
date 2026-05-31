from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
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


def _valid_manifest(asset_id: str) -> dict[str, object]:
    states = {
        "idle": {"animation_id": "idle"},
        "listening": {"animation_id": "idle"},
        "thinking": {"animation_id": "idle"},
        "speaking": {"animation_id": "idle"},
        "error": {"animation_id": "idle"},
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


def test_db_allows_explicit_cross_persona_parent_for_duplicate_path(db_instance: CharactersRAGDB) -> None:
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
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
            "listening": {"animation_id": "idle"},
            "thinking": {"animation_id": "idle"},
            "speaking": {"animation_id": "idle"},
            "error": {"animation_id": "idle"},
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
    db_instance.activate_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=updated_source["id"],
    )
    target_active = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        title="Existing Target Active",
        status="draft",
    )
    db_instance.activate_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        pack_id=target_active["id"],
    )

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
        service.activate_pack(persona_id=persona_id, user_id="user-1", pack_id=pack["id"])

    assert exc_info.value.code == "invalid_manifest"
    assert db_instance.get_active_persona_visual_pack(persona_id=persona_id, user_id="user-1") is None


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

    service.activate_pack(persona_id=persona_id, user_id="user-1", pack_id=first["id"])
    service.activate_pack(persona_id=persona_id, user_id="user-1", pack_id=second["id"])

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

    service.activate_pack(persona_id=persona_id, user_id="user-1", pack_id=pack["id"])
    service.deactivate_pack(persona_id=persona_id, user_id="user-1")

    assert db_instance.get_active_persona_visual_pack(persona_id=persona_id, user_id="user-1") is None
    archived = db_instance.get_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        pack_id=pack["id"],
    )
    assert archived is not None
    assert archived["status"] == "archived"
