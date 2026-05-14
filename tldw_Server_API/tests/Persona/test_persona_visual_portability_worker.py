from __future__ import annotations

import io
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_jobs import (
    PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE,
    PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
    PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
)
from tldw_Server_API.app.core.Persona.visual_portability.exporter import (
    PersonaVisualPackExporter,
)
from tldw_Server_API.app.core.Persona.visual_portability.models import (
    PersonaVisualPackExportOptions,
)
from tldw_Server_API.app.core.Persona.visual_portability.preview import (
    PersonaVisualPackImportPreviewer,
)
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualService


pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 3), (30, 80, 180, 255)).save(buffer, format="PNG")
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
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    _patch_visuals_dir(monkeypatch, visuals_root)
    persona_id = db.create_persona_profile({"user_id": user_id, "name": "Worker Persona"})
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title="Worker Visuals",
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
def db_instance(tmp_path: Path) -> Generator[CharactersRAGDB, None, None]:
    db = CharactersRAGDB(tmp_path / "persona_visual_portability_worker.sqlite", "persona-visual-worker-test")
    yield db
    db.close_connection()


def test_persona_visual_portability_records_round_trip(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )

    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    persona_id = db_instance.create_persona_profile({"user_id": "user-1", "name": "Portable Worker"})
    pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="user-1",
        title="Portable Worker Visuals",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    created = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-export-1",
        operation="export",
        status="queued",
        stage="queued",
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        progress={"current": 0, "total": 2},
        warnings=["pending-preview"],
    )

    loaded = repo.get_portability_job(str(created["id"]), owner_user_id="user-1")
    assert loaded is not None
    assert loaded["job_id"] == "job-export-1"
    assert loaded["operation"] == "export"
    assert loaded["persona_id"] == persona_id
    assert loaded["pack_id"] == pack["id"]
    assert json.loads(loaded["progress_json"]) == {"current": 0, "total": 2}
    assert json.loads(loaded["warnings_json"]) == ["pending-preview"]
    assert repo.get_portability_job(str(created["id"]), owner_user_id="other-user") is None

    updated = repo.update_portability_job(
        "job-export-1",
        {
            "status": "processing",
            "stage": "assembling_archive",
            "archive_sha256": "a" * 64,
            "canonical_payload_fingerprint": "b" * 64,
            "progress": {"current": 1, "total": 2},
        },
        owner_user_id="user-1",
    )
    assert updated is not None
    assert updated["stage"] == "assembling_archive"
    assert json.loads(updated["progress_json"]) == {"current": 1, "total": 2}
    assert json.loads(updated["warnings_json"]) == ["pending-preview"]
    assert repo.get_portability_job_by_job_id("job-export-1", owner_user_id="user-1")["id"] == created["id"]

    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="queued",
        archive_path=str(tmp_path / "incoming.tldw-persona-vpack"),
        target_persona_id=persona_id,
        bundle_summary={"pack_title": "Worker Visuals"},
        validation_warnings=["requires-review"],
        proposed_plan={"target_mode": "create_new"},
        quota_estimate={"asset_bytes": 123},
        required_choices=["trust_mode"],
        target_warnings=["target-persona-not-found"],
    )
    loaded_preview = repo.get_import_preview(str(preview["id"]), owner_user_id="user-1")
    assert loaded_preview is not None
    assert loaded_preview["target_persona_id"] == persona_id
    assert json.loads(loaded_preview["bundle_summary_json"]) == {"pack_title": "Worker Visuals"}
    assert json.loads(loaded_preview["target_warnings_json"]) == ["target-persona-not-found"]
    assert repo.get_import_preview(str(preview["id"]), owner_user_id="other-user") is None

    updated_preview = repo.update_import_preview(
        str(preview["id"]),
        {
            "status": "completed",
            "schema_version": "tldw.persona_visual_pack.v1",
            "target_warnings": [],
        },
        owner_user_id="user-1",
    )
    assert updated_preview is not None
    assert updated_preview["status"] == "completed"
    assert updated_preview["schema_version"] == "tldw.persona_visual_pack.v1"
    assert json.loads(updated_preview["target_warnings_json"]) == []


@pytest.mark.asyncio
async def test_persona_visual_export_worker_updates_portability_job(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    persona_id, pack, _asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-export-1",
        operation="export",
        status="queued",
        stage="queued",
        persona_id=persona_id,
        pack_id=str(pack["id"]),
    )
    worker = PersonaVisualPortabilityWorker(
        db=db_instance,
        repo=repo,
        export_staging_root=tmp_path / "exports",
    )

    result = await worker.handle_job_async(
        {
            "id": "job-export-1",
            "job_type": PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE,
            "owner_user_id": "user-1",
            "payload": {
                "user_id": "user-1",
                "persona_id": persona_id,
                "pack_id": str(pack["id"]),
                "portability_job_id": str(portability_job["id"]),
                "request_id": "req-worker",
                "options": {"strict": False},
            },
        }
    )

    updated = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert result["status"] == "exported"
    assert Path(result["archive_path"]).is_file()
    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["stage"] == "completed"
    assert updated["archive_sha256"] == result["archive_sha256"]
    assert updated["canonical_payload_fingerprint"] == result["canonical_payload_fingerprint"]
    assert json.loads(updated["progress_json"])["file_size_bytes"] == result["file_size_bytes"]


@pytest.mark.asyncio
async def test_persona_visual_import_preview_worker_updates_preview_without_mutating_packs(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    persona_id, pack, _asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    packs_before = db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="queued",
        archive_path=str(export_result.archive_path),
        target_persona_id=persona_id,
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-preview-1",
        operation="import_preview",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        archive_path=str(export_result.archive_path),
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    result = await worker.handle_job_async(
        {
            "id": "job-preview-1",
            "job_type": PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
            "owner_user_id": "user-1",
            "payload": {
                "user_id": "user-1",
                "preview_id": str(preview["id"]),
                "archive_path": str(export_result.archive_path),
                "request_id": "req-worker",
                "target_persona_id": persona_id,
            },
        }
    )

    updated_preview = repo.get_import_preview(str(preview["id"]), owner_user_id="user-1")
    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert result["status"] == "previewed"
    assert updated_preview is not None
    assert updated_preview["status"] == "completed"
    assert updated_preview["archive_sha256"] == result["archive_sha256"]
    assert updated_preview["canonical_payload_fingerprint"] == result["canonical_payload_fingerprint"]
    assert json.loads(updated_preview["bundle_summary_json"])["pack_title"] == "Worker Visuals"
    assert updated_job is not None
    assert updated_job["status"] == "completed"
    assert updated_job["stage"] == "completed"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1") == packs_before


@pytest.mark.asyncio
async def test_persona_visual_import_preview_worker_persists_blocked_preview_status(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona import visual_jobs_worker as worker_module

    class _BlockedPreviewer:
        def create_preview(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "blocked",
                "archive_sha256": "a" * 64,
                "canonical_payload_fingerprint": "b" * 64,
                "schema_version": "tldw.persona_visual_pack.v1",
                "bundle_summary": {"renderer_type": "live2d"},
                "validation_warnings": [],
                "conflicts": [],
                "proposed_plan": {
                    "renderer_import_preview": {"status": "unsupported_renderer"},
                    "commit_eligible": False,
                },
                "quota_estimate": {
                    "asset_bytes": 0,
                    "present_asset_items": 0,
                    "missing_asset_items": 0,
                },
                "required_choices": [],
                "target_warnings": [],
            }

    archive_path = tmp_path / "incoming.tldw-persona-vpack"
    archive_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        worker_module,
        "PersonaVisualPackImportPreviewer",
        _BlockedPreviewer,
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-blocked",
        status="queued",
        archive_path=str(archive_path),
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-preview-blocked",
        operation="import_preview",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        archive_path=str(archive_path),
    )
    worker = worker_module.PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    result = await worker.handle_job_async(
        {
            "id": "job-preview-blocked",
            "job_type": PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
            "owner_user_id": "user-1",
            "payload": {
                "user_id": "user-1",
                "preview_id": str(preview["id"]),
                "archive_path": str(archive_path),
                "request_id": "req-worker-blocked",
            },
        }
    )

    updated_preview = repo.get_import_preview(str(preview["id"]), owner_user_id="user-1")
    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert result["status"] == "previewed"
    assert updated_preview is not None
    assert updated_preview["status"] == "blocked"
    assert json.loads(updated_preview["proposed_plan_json"])["commit_eligible"] is False
    assert updated_job is not None
    assert updated_job["status"] == "completed"


@pytest.mark.asyncio
async def test_persona_visual_import_preview_worker_persists_target_conflicts(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    source_persona_id, pack, _asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    target_persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Import Target"}
    )
    target_draft = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title="Worker Visuals",
        status="draft",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=source_persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-conflicts",
        status="queued",
        archive_path=str(export_result.archive_path),
        target_persona_id=target_persona_id,
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-preview-conflicts",
        operation="import_preview",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        archive_path=str(export_result.archive_path),
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    await worker.handle_job_async(
        {
            "id": "job-preview-conflicts",
            "job_type": PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
            "owner_user_id": "user-1",
            "payload": {
                "user_id": "user-1",
                "preview_id": str(preview["id"]),
                "archive_path": str(export_result.archive_path),
                "request_id": "req-worker-conflicts",
                "target_persona_id": target_persona_id,
            },
        }
    )

    updated_preview = repo.get_import_preview(str(preview["id"]), owner_user_id="user-1")
    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert updated_preview is not None
    assert json.loads(updated_preview["conflicts_json"]) == [  # nosec B101
        {
            "conflict_id": f"target_pack_title_match:{target_draft['id']}",
            "type": "target_pack_title_match",
            "severity": "warning",
            "message": "Target persona already has a draft visual pack named Worker Visuals.",
            "pack_id": target_draft["id"],
            "pack_title": "Worker Visuals",
            "pack_status": "draft",
            "allowed_choices": ["create_new", "replace_draft"],
        }
    ]
    assert json.loads(updated_preview["required_choices_json"]) == [  # nosec B101
        {
            "choice_id": "import_target_mode",
            "reason": "target_pack_conflicts",
            "default_target_mode": "create_new",
            "allowed_target_modes": ["create_new", "replace_draft"],
            "replaceable_pack_ids": [target_draft["id"]],
        }
    ]
    assert updated_job is not None
    assert json.loads(updated_job["progress_json"])["pack_title"] == "Worker Visuals"


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_creates_pack_and_remaps_assets(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    persona_id, pack, source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    preview_result = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=export_result.archive_path,
        owner_user_id="user-1",
        target_persona_id=persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=persona_id,
            user_id="user-1",
        ),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        target_persona_id=persona_id,
        bundle_summary=preview_result["bundle_summary"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        required_choices=preview_result["required_choices"],
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=persona_id,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    result = await worker.handle_job_async(
        {
            "id": "job-commit-1",
            "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
            "owner_user_id": "user-1",
            "payload": {
                "user_id": "user-1",
                "preview_id": str(preview["id"]),
                "portability_job_id": str(portability_job["id"]),
                "request_id": "req-commit",
                "target_persona_id": persona_id,
                "trust_mode": "untrusted_import",
                "target_mode": "create_new",
                "conflict_choice_explicit": True,
            },
        }
    )

    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    imported_pack = db_instance.get_persona_visual_pack(
        pack_id=result["pack_id"],
        persona_id=persona_id,
        user_id="user-1",
    )
    imported_assets = db_instance.list_persona_visual_assets(
        pack_id=result["pack_id"],
        persona_id=persona_id,
        user_id="user-1",
    )
    assert result["status"] == "imported"
    assert imported_pack is not None
    assert imported_pack["id"] != pack["id"]
    assert imported_pack["status"] == "draft"
    assert len(imported_assets) == 1
    assert imported_assets[0]["id"] != source_asset["id"]
    assert imported_pack["manifest"]["animations"]["idle"]["frames"][0]["asset_id"] == imported_assets[0]["id"]
    assert updated_job is not None
    assert updated_job["status"] == "completed"
    assert updated_job["stage"] == "completed"
    assert updated_job["pack_id"] == imported_pack["id"]


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_rejects_blocked_revalidation_without_pack(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure stale completed previews cannot commit after blocked revalidation."""

    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona import visual_jobs_worker as worker_module
    from tldw_Server_API.app.core.Persona.visual_portability import importer as importer_module

    class _BlockedPreviewer:
        def create_preview(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "blocked",
                "archive_sha256": "a" * 64,
                "canonical_payload_fingerprint": "b" * 64,
                "schema_version": "tldw.persona_visual_pack.v1",
                "bundle_summary": {"renderer_type": "live2d"},
                "validation_warnings": [],
                "conflicts": [],
                "proposed_plan": {
                    "renderer_import_preview": {"status": "unsupported_renderer"},
                    "commit_eligible": False,
                    "commit_blockers": ["runtime_adapter_not_implemented"],
                },
                "quota_estimate": {
                    "asset_bytes": 0,
                    "present_asset_items": 0,
                    "missing_asset_items": 0,
                },
                "required_choices": [],
                "target_warnings": [],
            }

    persona_id, pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-blocked-revalidation",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=export_result.archive_sha256,
        canonical_payload_fingerprint="b" * 64,
        schema_version="tldw.persona_visual_pack.v1",
        target_persona_id=persona_id,
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-blocked-revalidation",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=persona_id,
    )
    packs_before = db_instance.list_persona_visual_packs(
        persona_id=persona_id,
        user_id="user-1",
    )
    monkeypatch.setattr(
        importer_module,
        "PersonaVisualPackImportPreviewer",
        _BlockedPreviewer,
    )
    worker = worker_module.PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    with pytest.raises(ValueError, match="import_preview_not_commit_eligible"):
        await worker.handle_job_async(
            {
                "id": "job-commit-blocked-revalidation",
                "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
                "owner_user_id": "user-1",
                "payload": {
                    "user_id": "user-1",
                    "preview_id": str(preview["id"]),
                    "portability_job_id": str(portability_job["id"]),
                    "request_id": "req-commit-blocked-revalidation",
                    "target_persona_id": persona_id,
                    "trust_mode": "untrusted_import",
                    "target_mode": "create_new",
                    "conflict_choice_explicit": True,
                },
            }
        )

    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert updated_job is not None
    assert updated_job["status"] == "failed"
    assert updated_job["error_code"] == "import_commit_failed"
    assert db_instance.list_persona_visual_packs(
        persona_id=persona_id,
        user_id="user-1",
    ) == packs_before


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_replaces_selected_draft_only(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    source_persona_id, source_pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    target_persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Worker Target Persona"}
    )
    active_pack = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title="Active Target Visuals",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
        status="active",
    )
    target_draft = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title=str(source_pack["title"]),
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=source_persona_id,
        pack_id=str(source_pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    preview_result = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=export_result.archive_path,
        owner_user_id="user-1",
        target_persona_id=target_persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=target_persona_id,
            user_id="user-1",
        ),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        target_persona_id=target_persona_id,
        bundle_summary=preview_result["bundle_summary"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        required_choices=preview_result["required_choices"],
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=target_persona_id,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    result = await worker.handle_job_async(
        {
            "id": "job-commit-1",
            "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
            "owner_user_id": "user-1",
            "payload": {
                "user_id": "user-1",
                "preview_id": str(preview["id"]),
                "portability_job_id": str(portability_job["id"]),
                "request_id": "req-commit",
                "target_persona_id": target_persona_id,
                "trust_mode": "untrusted_import",
                "target_mode": "replace_draft",
                "target_pack_id": str(target_draft["id"]),
                "title": "Imported Replacement",
                "conflict_choice_explicit": True,
            },
        }
    )

    imported_pack = db_instance.get_persona_visual_pack(
        pack_id=result["pack_id"],
        persona_id=target_persona_id,
        user_id="user-1",
    )
    deleted_draft = db_instance.get_persona_visual_pack(
        pack_id=str(target_draft["id"]),
        persona_id=target_persona_id,
        user_id="user-1",
        include_deleted=True,
    )
    active_after = db_instance.get_active_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
    )
    assert result["status"] == "imported"
    assert result["replaced_pack_id"] == target_draft["id"]
    assert imported_pack is not None
    assert imported_pack["id"] != target_draft["id"]
    assert imported_pack["title"] == "Imported Replacement"
    assert imported_pack["status"] == "draft"
    assert deleted_draft is not None
    assert deleted_draft["deleted"] is True
    assert active_after is not None
    assert active_after["id"] == active_pack["id"]


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_requires_choice_for_revalidated_conflicts(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    source_persona_id, source_pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    target_persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Worker Conflict Target"}
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=source_persona_id,
        pack_id=str(source_pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    preview_result = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=export_result.archive_path,
        owner_user_id="user-1",
        target_persona_id=target_persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=target_persona_id,
            user_id="user-1",
        ),
    )
    assert preview_result["conflicts"] == []
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        target_persona_id=target_persona_id,
        bundle_summary=preview_result["bundle_summary"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        required_choices=preview_result["required_choices"],
    )
    target_draft = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title=str(source_pack["title"]),
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=target_persona_id,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    with pytest.raises(ValueError, match="import_conflict_choice_required"):
        await worker.handle_job_async(
            {
                "id": "job-commit-1",
                "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
                "owner_user_id": "user-1",
                "payload": {
                    "user_id": "user-1",
                    "preview_id": str(preview["id"]),
                    "portability_job_id": str(portability_job["id"]),
                    "request_id": "req-commit",
                    "target_persona_id": target_persona_id,
                    "trust_mode": "untrusted_import",
                    "target_mode": "create_new",
                },
            }
        )

    remaining_packs = db_instance.list_persona_visual_packs(
        persona_id=target_persona_id,
        user_id="user-1",
    )
    assert [pack["id"] for pack in remaining_packs] == [target_draft["id"]]


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_cleans_imported_pack_when_replace_fails(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    source_persona_id, source_pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    target_persona_id = db_instance.create_persona_profile(
        {"user_id": "user-1", "name": "Worker Cleanup Target"}
    )
    active_pack = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title="Active Target Visuals",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
        status="active",
    )
    target_draft = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id="user-1",
        title=str(source_pack["title"]),
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames", "states": {}, "animations": {}},
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=source_persona_id,
        pack_id=str(source_pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    preview_result = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=export_result.archive_path,
        owner_user_id="user-1",
        target_persona_id=target_persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=target_persona_id,
            user_id="user-1",
        ),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        target_persona_id=target_persona_id,
        bundle_summary=preview_result["bundle_summary"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        required_choices=preview_result["required_choices"],
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=target_persona_id,
    )
    original_soft_delete = db_instance.soft_delete_persona_visual_pack_with_assets

    def _fail_target_draft_delete(*, pack_id: str, persona_id: str, user_id: str, **kwargs: Any) -> bool:
        if str(pack_id) == str(target_draft["id"]):
            return False
        return original_soft_delete(pack_id=pack_id, persona_id=persona_id, user_id=user_id, **kwargs)

    monkeypatch.setattr(
        db_instance,
        "soft_delete_persona_visual_pack_with_assets",
        _fail_target_draft_delete,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    with pytest.raises(ValueError, match="import_target_pack_not_replaceable"):
        await worker.handle_job_async(
            {
                "id": "job-commit-1",
                "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
                "owner_user_id": "user-1",
                "payload": {
                    "user_id": "user-1",
                    "preview_id": str(preview["id"]),
                    "portability_job_id": str(portability_job["id"]),
                    "request_id": "req-commit",
                    "target_persona_id": target_persona_id,
                    "trust_mode": "untrusted_import",
                    "target_mode": "replace_draft",
                    "target_pack_id": str(target_draft["id"]),
                    "conflict_choice_explicit": True,
                },
            }
        )

    remaining_packs = db_instance.list_persona_visual_packs(
        persona_id=target_persona_id,
        user_id="user-1",
    )
    assert {pack["id"] for pack in remaining_packs} == {active_pack["id"], target_draft["id"]}


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_rejects_incomplete_preview(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )

    persona_id, pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="queued",
        archive_path=str(export_result.archive_path),
        target_persona_id=persona_id,
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=persona_id,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    with pytest.raises(ValueError, match="import_preview_not_completed"):
        await worker.handle_job_async(
            {
                "id": "job-commit-1",
                "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
                "owner_user_id": "user-1",
                "payload": {
                    "user_id": "user-1",
                    "preview_id": str(preview["id"]),
                    "portability_job_id": str(portability_job["id"]),
                    "request_id": "req-commit",
                    "target_persona_id": persona_id,
                    "trust_mode": "untrusted_import",
                    "target_mode": "create_new",
                },
            }
        )

    assert len(db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_plan_json", ["{not-json", "[]"])
async def test_persona_visual_import_commit_worker_rejects_invalid_stored_plan_before_revalidation(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_plan_json: str,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )
    from tldw_Server_API.app.core.Persona.visual_portability import importer as importer_module

    persona_id, pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    preview_result = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=export_result.archive_path,
        owner_user_id="user-1",
        target_persona_id=persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=persona_id,
            user_id="user-1",
        ),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        target_persona_id=persona_id,
        bundle_summary=preview_result["bundle_summary"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        required_choices=preview_result["required_choices"],
    )
    with db_instance.transaction() as conn:
        conn.execute(
            """
            UPDATE persona_visual_pack_import_previews
               SET proposed_plan_json = ?
             WHERE id = ?
            """,
            (stored_plan_json, preview["id"]),
        )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=persona_id,
    )

    class _UnexpectedPreviewer:
        def create_preview(self, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("import commit revalidated a corrupt stored preview")

    monkeypatch.setattr(
        importer_module,
        "PersonaVisualPackImportPreviewer",
        _UnexpectedPreviewer,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    with pytest.raises(ValueError, match="import_preview_not_commit_eligible"):
        await worker.handle_job_async(
            {
                "id": "job-commit-1",
                "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
                "owner_user_id": "user-1",
                "payload": {
                    "user_id": "user-1",
                    "preview_id": str(preview["id"]),
                    "portability_job_id": str(portability_job["id"]),
                    "request_id": "req-commit",
                    "target_persona_id": persona_id,
                    "trust_mode": "untrusted_import",
                    "target_mode": "create_new",
                },
            }
        )

    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert updated_job is not None
    assert updated_job["status"] == "failed"
    assert updated_job["error_message"] == "import_preview_not_commit_eligible"
    assert len(db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")) == 1


@pytest.mark.asyncio
async def test_persona_visual_import_commit_worker_rejects_revalidated_blocked_preview(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualPortabilityWorker,
    )
    from tldw_Server_API.app.core.Persona.visual_portability import importer as importer_module

    persona_id, pack, _source_asset = _create_pack_with_asset(
        db_instance,
        visuals_root=tmp_path / "visuals",
        monkeypatch=monkeypatch,
    )
    export_result = PersonaVisualPackExporter(
        db=db_instance,
        user_id="user-1",
        staging_root=tmp_path / "exports",
    ).export_pack(
        persona_id=persona_id,
        pack_id=str(pack["id"]),
        options=PersonaVisualPackExportOptions(),
    )
    preview_result = PersonaVisualPackImportPreviewer().create_preview(
        archive_path=export_result.archive_path,
        owner_user_id="user-1",
        target_persona_id=persona_id,
        target_packs=db_instance.list_persona_visual_packs(
            persona_id=persona_id,
            user_id="user-1",
        ),
    )
    repo = PersonaVisualPortabilityRepository.initialized(db_instance)
    preview = repo.create_import_preview(
        owner_user_id="user-1",
        job_id="job-preview-1",
        status="completed",
        stage="completed",
        archive_path=str(export_result.archive_path),
        archive_sha256=preview_result["archive_sha256"],
        canonical_payload_fingerprint=preview_result["canonical_payload_fingerprint"],
        schema_version=preview_result["schema_version"],
        target_persona_id=persona_id,
        bundle_summary=preview_result["bundle_summary"],
        conflicts=preview_result["conflicts"],
        proposed_plan=preview_result["proposed_plan"],
        required_choices=preview_result["required_choices"],
    )
    portability_job = repo.create_portability_job(
        owner_user_id="user-1",
        job_id="job-commit-1",
        operation="import_commit",
        status="queued",
        stage="queued",
        preview_id=str(preview["id"]),
        persona_id=persona_id,
    )

    class _BlockedPreviewer:
        def create_preview(self, **_kwargs: Any) -> dict[str, Any]:
            blocked = dict(preview_result)
            blocked["status"] = "blocked"
            blocked["proposed_plan"] = {
                **preview_result["proposed_plan"],
                "commit_eligible": False,
                "commit_blockers": ["unsupported_renderer"],
            }
            return blocked

    monkeypatch.setattr(
        importer_module,
        "PersonaVisualPackImportPreviewer",
        _BlockedPreviewer,
    )
    worker = PersonaVisualPortabilityWorker(db=db_instance, repo=repo)

    with pytest.raises(ValueError, match="import_preview_not_commit_eligible"):
        await worker.handle_job_async(
            {
                "id": "job-commit-1",
                "job_type": PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
                "owner_user_id": "user-1",
                "payload": {
                    "user_id": "user-1",
                    "preview_id": str(preview["id"]),
                    "portability_job_id": str(portability_job["id"]),
                    "request_id": "req-commit",
                    "target_persona_id": persona_id,
                    "trust_mode": "untrusted_import",
                    "target_mode": "create_new",
                    "conflict_choice_explicit": True,
                },
            }
        )

    updated_job = repo.get_portability_job(str(portability_job["id"]), owner_user_id="user-1")
    assert updated_job is not None
    assert updated_job["status"] == "failed"
    assert updated_job["error_message"] == "import_preview_not_commit_eligible"
    assert len(db_instance.list_persona_visual_packs(persona_id=persona_id, user_id="user-1")) == 1
