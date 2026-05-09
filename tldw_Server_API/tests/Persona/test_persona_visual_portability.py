from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_portability.archive import (
    validate_archive_members,
)
from tldw_Server_API.app.core.Persona.visual_portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    PERSONA_VISUAL_PACK_SCHEMA_VERSION,
    REQUIRED_MEMBERS,
)
from tldw_Server_API.app.core.Persona.visual_portability.exporter import (
    PersonaVisualPackExporter,
)
from tldw_Server_API.app.core.Persona.visual_portability.fingerprints import (
    sha256_file,
)
from tldw_Server_API.app.core.Persona.visual_portability.models import (
    PersonaVisualPackExportOptions,
)
from tldw_Server_API.app.core.Persona.visual_portability.preview import (
    PersonaVisualPackImportPreviewer,
)
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualService


pytestmark = pytest.mark.unit


def _png_bytes(width: int = 2, height: int = 3) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (width, height), (24, 120, 200, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_manifest(asset_id: str) -> dict[str, object]:
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
                "frames": [{"asset_id": asset_id, "duration_ms": 100}],
                "frame_rate": 1,
                "preview_frame": 0,
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


def _create_pack_with_asset(
    db: CharactersRAGDB,
    *,
    visuals_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_id: str = "user-1",
) -> tuple[str, dict[str, object], dict[str, object]]:
    _patch_visuals_dir(monkeypatch, visuals_root)
    persona_id = db.create_persona_profile({"user_id": user_id, "name": "Portable Persona"})
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title="Portable Visuals",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
        },
    )
    service = PersonaVisualService(db)
    asset = service.create_asset_from_upload(
        persona_id=persona_id,
        user_id=user_id,
        pack_id=str(pack["id"]),
        content=_png_bytes(),
        mime_type="image/png",
        original_filename="idle.png",
        asset_role="frame",
    )
    db.update_persona_visual_pack_manifest(
        pack_id=str(pack["id"]),
        persona_id=persona_id,
        user_id=user_id,
        manifest=_valid_manifest(str(asset["id"])),
    )
    return persona_id, pack, asset


@pytest.fixture()
def db_instance(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "persona_visual_portability.sqlite", "persona-visual-portability-test")
    yield db
    db.close_connection()


def test_validate_archive_members_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tldw-persona-vpack"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for member in REQUIRED_MEMBERS:
            archive.writestr(member, b"{}")
        archive.writestr("../escape.png", b"data")

    with pytest.raises(ValueError, match="unsafe_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_allows_zip_directory_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "with-directories.tldw-persona-vpack"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("metadata/", b"")
        archive.writestr("assets/", b"")
        archive.writestr("assets/persona_visuals/", b"")
        for member in REQUIRED_MEMBERS:
            archive.writestr(member, b"{}")

    members = validate_archive_members(archive_path)

    assert REQUIRED_MEMBERS <= set(members)
    assert "metadata/" not in members


def test_export_pack_writes_manifest_metadata_checksums_and_asset_bytes(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_id, pack, asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    exporter = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    )

    result = exporter.export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )

    assert result.archive_sha256 == sha256_file(result.archive_path)  # nosec B101
    assert result.file_size_bytes == result.archive_path.stat().st_size  # nosec B101
    assert result.warnings == []  # nosec B101

    with zipfile.ZipFile(result.archive_path) as archive:
        names = set(archive.namelist())
        assert REQUIRED_MEMBERS <= names  # nosec B101

        archive_manifest = json.loads(archive.read(MANIFEST_PATH))
        assert archive_manifest["schema_version"] == PERSONA_VISUAL_PACK_SCHEMA_VERSION  # nosec B101
        assert archive_manifest["archive_profile"] == "backup"  # nosec B101
        assert archive_manifest["pack_title"] == "Portable Visuals"  # nosec B101
        assert archive_manifest["renderer_type"] == "sprite_frames"  # nosec B101
        assert archive_manifest["counts"]["assets"] == 1  # nosec B101
        assert archive_manifest["counts"]["assets_with_bytes"] == 1  # nosec B101
        assert "canonical_payload_fingerprint" in archive_manifest  # nosec B101

        pack_payload = json.loads(archive.read("metadata/pack.json"))["pack"]
        assert pack_payload["title"] == "Portable Visuals"  # nosec B101
        assert pack_payload["visual_manifest"]["animations"]["idle"]["frames"][0]["asset_id"] == asset["id"]  # nosec B101

        assets = json.loads(archive.read("metadata/assets.json"))["assets"]
        assert len(assets) == 1  # nosec B101
        exported_asset = assets[0]
        assert exported_asset["source_asset_id"] == asset["id"]  # nosec B101
        assert exported_asset["asset_bytes_status"] == ASSET_BYTES_STATUS_PRESENT  # nosec B101
        assert exported_asset["asset_path"].startswith("assets/persona_visuals/")  # nosec B101
        assert archive.read(exported_asset["asset_path"]) == _png_bytes()  # nosec B101

        checksums = json.loads(archive.read(CHECKSUMS_PATH))
        assert "metadata/pack.json" in checksums  # nosec B101
        assert "metadata/assets.json" in checksums  # nosec B101
        assert exported_asset["asset_path"] in checksums  # nosec B101


def test_export_pack_strict_mode_rejects_missing_asset_bytes(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_id, pack, asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    Path(str(asset["storage_path"])).unlink()
    exporter = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    )

    with pytest.raises(ValueError, match="missing_asset_bytes"):
        exporter.export_pack(
            persona_id=persona_id,
            pack_id=str(pack["id"]),
            options=PersonaVisualPackExportOptions(strict=True),
        )


def test_import_preview_validates_archive_without_mutating_packs(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_id, pack, _asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    exporter = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    )
    result = exporter.export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    packs_before = db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=result.archive_path,
        owner_user_id="user-1",
        target_persona_id=persona_id,
    )

    assert preview["status"] == "completed"  # nosec B101
    assert preview["schema_version"] == PERSONA_VISUAL_PACK_SCHEMA_VERSION  # nosec B101
    assert preview["archive_sha256"] == result.archive_sha256  # nosec B101
    assert preview["canonical_payload_fingerprint"] == result.canonical_payload_fingerprint  # nosec B101
    assert preview["bundle_summary"]["pack_title"] == "Portable Visuals"  # nosec B101
    assert preview["bundle_summary"]["asset_count"] == 1  # nosec B101
    assert preview["bundle_summary"]["assets_with_bytes"] == 1  # nosec B101
    assert preview["quota_estimate"]["asset_bytes"] == len(_png_bytes())  # nosec B101
    assert preview["required_choices"] == []  # nosec B101
    assert preview["proposed_plan"]["target_mode"] == "create_new"  # nosec B101
    assert preview["proposed_plan"]["trust_modes"] == ["trusted_restore", "untrusted_import"]  # nosec B101
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1") == packs_before  # nosec B101


def test_import_preview_reports_missing_asset_bytes_as_warning(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_id, pack, asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    Path(str(asset["storage_path"])).unlink()
    exporter = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    )
    result = exporter.export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=result.archive_path,
        owner_user_id="user-1",
        target_persona_id=persona_id,
    )

    assert preview["bundle_summary"]["missing_asset_items"] == 1  # nosec B101
    assert preview["validation_warnings"] == [  # nosec B101
        f"missing_asset_bytes:frame:{asset['id']}"
    ]
    exported_assets = preview["bundle_summary"]["assets"]
    assert exported_assets[0]["asset_bytes_status"] == ASSET_BYTES_STATUS_MISSING  # nosec B101
