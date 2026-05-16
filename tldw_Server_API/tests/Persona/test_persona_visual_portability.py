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
    sha256_bytes,
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


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _live2d_v2_manifest() -> dict[str, object]:
    return {
        "manifest_version": 2,
        "renderer_type": "live2d",
        "renderer_contract_version": 1,
        "renderer_assets": {
            "fallback_preview_asset_id": "asset-fallback",
            "source_manifest_asset_id": "asset-model",
        },
        "states": {
            "idle": {"animation_id": "idle"},
        },
        "animations": {
            "idle": {
                "renderer_action": {
                    "motion_group": "Idle",
                    "loop": True,
                },
            },
        },
    }


def _renderer_preview_archive(
    tmp_path: Path,
    *,
    title: str,
    visual_manifest: dict[str, object],
    assets: list[dict[str, object]],
    asset_files: dict[str, bytes] | None = None,
) -> Path:
    archive_path = tmp_path / f"{title.lower().replace(' ', '-')}.tldw-persona-vpack"
    asset_files = asset_files or {}
    archive_manifest = {
        "schema_version": PERSONA_VISUAL_PACK_SCHEMA_VERSION,
        "archive_profile": "backup",
        "pack_title": title,
        "renderer_type": visual_manifest.get("renderer_type"),
        "counts": {
            "assets": len(assets),
            "assets_with_bytes": sum(
                1
                for asset in assets
                if asset.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT
            ),
        },
    }
    pack_payload = {
        "pack": {
            "title": title,
            "source_persona_id": "source-persona-v2",
            "renderer_type": visual_manifest.get("renderer_type"),
            "visual_manifest": visual_manifest,
        }
    }
    entries: dict[str, bytes] = {
        MANIFEST_PATH: _json_bytes(archive_manifest),
        "metadata/pack.json": _json_bytes(pack_payload),
        "metadata/assets.json": _json_bytes({"assets": assets}),
        **asset_files,
    }
    checksums = {
        member_path: sha256_bytes(member_bytes)
        for member_path, member_bytes in entries.items()
    }
    entries[CHECKSUMS_PATH] = _json_bytes(dict(sorted(checksums.items())))

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_path, member_bytes in entries.items():
            archive.writestr(member_path, member_bytes)
    return archive_path


def _live2d_preview_assets(
    *,
    include_fallback: bool = True,
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    fallback_bytes = _png_bytes()
    model_bytes = b'{"Version":3,"FileReferences":{}}'
    assets: list[dict[str, object]] = [
        {
            "source_asset_id": "asset-model",
            "asset_role": "live2d_model_manifest",
            "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
            "asset_path": "assets/persona_visuals/model.model3.json",
            "asset_sha256": sha256_bytes(model_bytes),
            "mime_type": "application/json",
            "original_filename": "model.model3.json",
        },
    ]
    asset_files = {
        "assets/persona_visuals/model.model3.json": model_bytes,
    }
    if include_fallback:
        assets.append(
            {
                "source_asset_id": "asset-fallback",
                "asset_role": "fallback_preview",
                "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
                "asset_path": "assets/persona_visuals/fallback.png",
                "asset_sha256": sha256_bytes(fallback_bytes),
                "mime_type": "image/png",
                "width": 2,
                "height": 3,
                "original_filename": "fallback.png",
            }
        )
        asset_files["assets/persona_visuals/fallback.png"] = fallback_bytes
    return assets, asset_files


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


def test_import_preview_reports_target_pack_conflicts_and_choices(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_persona_id, pack, _asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    target_persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Target Persona"}
    )
    active_pack = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title="Portable Visuals",
        status="active",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    draft_pack = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title="Portable Visuals",
        status="draft",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    exporter = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    )
    result = exporter.export_pack(
        persona_id=source_persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=result.archive_path,
        owner_user_id="user-1",
        target_persona_id=target_persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=target_persona_id,
            user_id="user-1",
        ),
    )

    conflict_ids = {conflict["conflict_id"] for conflict in preview["conflicts"]}
    assert conflict_ids == {  # nosec B101
        f"target_pack_title_match:{active_pack['id']}",
        f"target_pack_title_match:{draft_pack['id']}",
    }
    active_conflict = next(
        conflict for conflict in preview["conflicts"] if conflict["pack_id"] == active_pack["id"]
    )
    draft_conflict = next(
        conflict for conflict in preview["conflicts"] if conflict["pack_id"] == draft_pack["id"]
    )
    assert active_conflict["pack_status"] == "active"  # nosec B101
    assert active_conflict["allowed_choices"] == ["create_new"]  # nosec B101
    assert draft_conflict["pack_status"] == "draft"  # nosec B101
    assert draft_conflict["allowed_choices"] == ["create_new", "replace_draft"]  # nosec B101
    assert preview["proposed_plan"]["target_modes"] == ["create_new", "replace_draft"]  # nosec B101
    assert preview["proposed_plan"]["replaceable_pack_ids"] == [draft_pack["id"]]  # nosec B101
    assert preview["required_choices"] == [  # nosec B101
        {
            "choice_id": "import_target_mode",
            "reason": "target_pack_conflicts",
            "default_target_mode": "create_new",
            "allowed_target_modes": ["create_new", "replace_draft"],
            "replaceable_pack_ids": [draft_pack["id"]],
        }
    ]


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


def test_import_preview_reports_v2_live2d_renderer_diagnostics_without_activation(
    tmp_path: Path,
) -> None:
    assets, asset_files = _live2d_preview_assets()
    archive_path = _renderer_preview_archive(
        tmp_path,
        title="Live2D Preview",
        visual_manifest=_live2d_v2_manifest(),
        assets=assets,
        asset_files=asset_files,
    )

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id="user-1",
        target_persona_id="target-persona",
    )

    renderer_preview = preview["proposed_plan"]["renderer_import_preview"]
    assert preview["status"] == "blocked"  # nosec B101
    assert preview["bundle_summary"]["renderer_type"] == "live2d"  # nosec B101
    assert preview["proposed_plan"]["commit_eligible"] is False  # nosec B101
    assert preview["proposed_plan"]["activation_eligible"] is False  # nosec B101
    assert renderer_preview["status"] == "unsupported_renderer"  # nosec B101
    assert renderer_preview["setup_status"] == "unsupported_renderer"  # nosec B101
    assert renderer_preview["can_commit"] is False  # nosec B101
    assert renderer_preview["activation_eligible"] is False  # nosec B101
    assert "runtime_adapter_not_implemented" in renderer_preview["blockers"]  # nosec B101
    assert renderer_preview["normalized_role_categories"]["fallback_preview"] == [  # nosec B101
        "asset-fallback"
    ]
    assert renderer_preview["normalized_role_categories"]["source_manifest"] == [  # nosec B101
        "asset-model"
    ]


@pytest.mark.parametrize("manifest_version", [2.0, "+2"])
def test_import_preview_routes_integer_like_v2_manifest_versions_to_renderer_diagnostics(
    tmp_path: Path,
    manifest_version: object,
) -> None:
    assets, asset_files = _live2d_preview_assets()
    visual_manifest = _live2d_v2_manifest()
    visual_manifest["manifest_version"] = manifest_version
    archive_path = _renderer_preview_archive(
        tmp_path,
        title=f"Live2D Version {manifest_version}",
        visual_manifest=visual_manifest,
        assets=assets,
        asset_files=asset_files,
    )

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id="user-1",
        target_persona_id="target-persona",
    )

    renderer_preview = preview["proposed_plan"]["renderer_import_preview"]
    assert preview["status"] == "blocked"  # nosec B101
    assert renderer_preview["manifest_version"] == 2  # nosec B101
    assert renderer_preview["status"] == "unsupported_renderer"  # nosec B101


def test_import_preview_reports_v2_missing_required_role_category(
    tmp_path: Path,
) -> None:
    assets, asset_files = _live2d_preview_assets(include_fallback=False)
    archive_path = _renderer_preview_archive(
        tmp_path,
        title="Live2D Missing Fallback",
        visual_manifest=_live2d_v2_manifest(),
        assets=assets,
        asset_files=asset_files,
    )

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id="user-1",
        target_persona_id="target-persona",
    )

    renderer_preview = preview["proposed_plan"]["renderer_import_preview"]
    assert preview["status"] == "blocked"  # nosec B101
    assert "missing_required_role_category:fallback_preview" in renderer_preview["blockers"]  # nosec B101
    assert renderer_preview["normalized_role_categories"]["fallback_preview"] == []  # nosec B101
    assert renderer_preview["normalized_role_categories"]["source_manifest"] == [  # nosec B101
        "asset-model"
    ]


def test_import_preview_reports_v2_unknown_renderer_safely(
    tmp_path: Path,
) -> None:
    visual_manifest = {
        "manifest_version": 2,
        "renderer_type": "unknown\nrenderer\\token",
        "renderer_contract_version": 1,
        "states": {},
        "animations": {},
    }
    archive_path = _renderer_preview_archive(
        tmp_path,
        title="Unknown Renderer Preview",
        visual_manifest=visual_manifest,
        assets=[],
    )

    preview = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id="user-1",
        target_persona_id="target-persona",
    )

    renderer_preview = preview["proposed_plan"]["renderer_import_preview"]
    assert preview["status"] == "blocked"  # nosec B101
    assert preview["proposed_plan"]["commit_eligible"] is False  # nosec B101
    assert renderer_preview["status"] == "unsupported_renderer"  # nosec B101
    assert renderer_preview["blockers"] == [  # nosec B101
        "unknown_renderer:unknown\\nrenderer\\\\token"
    ]
    assert "\n" not in renderer_preview["blockers"][0]  # nosec B101


def test_import_preview_rejects_unsupported_renderer_type_in_visual_manifest(
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
    mutated_archive_path = tmp_path / "unsupported-renderer.tldw-persona-vpack"

    with zipfile.ZipFile(result.archive_path, "r") as source_archive:
        pack_payload = json.loads(source_archive.read("metadata/pack.json"))
        pack_payload["pack"]["visual_manifest"]["renderer_type"] = "live2d"
        mutated_pack_bytes = json.dumps(
            pack_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        checksums = json.loads(source_archive.read(CHECKSUMS_PATH))
        checksums["metadata/pack.json"] = sha256_bytes(mutated_pack_bytes)
        mutated_checksums_bytes = json.dumps(
            dict(sorted(checksums.items())),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        with zipfile.ZipFile(mutated_archive_path, "w", compression=zipfile.ZIP_DEFLATED) as target_archive:
            for member in source_archive.infolist():
                if member.filename == "metadata/pack.json":
                    target_archive.writestr(member.filename, mutated_pack_bytes)
                elif member.filename == CHECKSUMS_PATH:
                    target_archive.writestr(member.filename, mutated_checksums_bytes)
                else:
                    target_archive.writestr(member, source_archive.read(member.filename))

    with pytest.raises(ValueError, match="malformed_visual_manifest"):
        PersonaVisualPackImportPreviewer().create_preview(
            archive_path=mutated_archive_path,
            owner_user_id="user-1",
            target_persona_id=persona_id,
        )
