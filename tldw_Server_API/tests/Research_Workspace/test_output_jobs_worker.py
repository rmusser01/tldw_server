from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@pytest.fixture
def fake_job_manager() -> object:
    return object()


def _worker_module() -> Any:
    return importlib.import_module("tldw_Server_API.app.services.research_workspace_output_jobs_worker")


def _output_jobs_module() -> Any:
    return importlib.import_module("tldw_Server_API.app.core.Research_Workspace.output_jobs")


@pytest.fixture
def fake_workspace_db() -> object:
    class _FakeWorkspaceDB:
        def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
            assert workspace_id == "ws-1"
            return [
                {
                    "id": "src-1",
                    "workspace_id": "ws-1",
                    "media_id": 1,
                    "title": "Source One",
                },
                {
                    "id": "src-2",
                    "workspace_id": "ws-1",
                    "media_id": 2,
                    "title": "Source Two",
                },
            ]

    return _FakeWorkspaceDB()


@pytest.fixture
def fake_media_db() -> object:
    class _FakeMediaDB:
        def get_media_by_id(self, media_id: int, **kwargs: object) -> dict[str, object] | None:
            if media_id == 1:
                return {"id": 1, "content": "selected media content"}
            if media_id == 2:
                return {"id": 2, "content": "unselected media content"}
            return None

    return _FakeMediaDB()


@pytest.fixture
def fake_collections_db() -> object:
    class _FakeCollectionsDB:
        def __init__(self) -> None:
            self.user_id = 42
            self.created: list[dict[str, object]] = []

        def resolve_output_storage_path(self, path_value: str) -> str:
            assert "/" not in path_value
            return path_value

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            self.created.append(kwargs)
            return SimpleNamespace(id=123)

    return _FakeCollectionsDB()


def test_build_source_context_uses_selected_ready_media(
    fake_workspace_db: object,
    fake_media_db: object,
) -> None:
    output_jobs = _output_jobs_module()

    context = output_jobs.build_research_workspace_output_source_context(
        workspace_db=fake_workspace_db,
        media_db=fake_media_db,
        workspace_id="ws-1",
        source_ids=["src-1"],
        max_chars=10_000,
    )

    assert "# Source One" in context.text
    assert "selected media content" in context.text
    assert "unselected media content" not in context.text
    assert context.source_lineage["selected_source_ids"] == ["src-1"]


def test_build_source_context_caps_title_and_content_text() -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
            assert workspace_id == "ws-1"
            return [
                {
                    "id": "src-1",
                    "workspace_id": "ws-1",
                    "media_id": 1,
                    "title": "T" * 200,
                }
            ]

    class _MediaDB:
        def get_media_by_id(self, media_id: int, **kwargs: object) -> dict[str, object]:
            assert media_id == 1
            return {"id": 1, "content": "body text"}

    context = output_jobs.build_research_workspace_output_source_context(
        workspace_db=_WorkspaceDB(),
        media_db=_MediaDB(),
        workspace_id="ws-1",
        source_ids=["src-1"],
        max_chars=40,
    )

    assert len(context.text) <= 40
    assert context.text.startswith("# ")
    assert "\n\n" in context.text
    assert context.text.split("\n\n", 1)[1].strip()
    assert context.source_lineage["context_truncated"] is True


def test_build_source_context_treats_empty_media_content_as_unavailable() -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
            assert workspace_id == "ws-1"
            return [
                {
                    "id": "src-1",
                    "workspace_id": "ws-1",
                    "media_id": 1,
                    "title": "Source One",
                }
            ]

    class _MediaDB:
        def get_media_by_id(self, media_id: int, **kwargs: object) -> dict[str, object]:
            assert media_id == 1
            return {"id": 1, "content": ""}

        def get_document_version(self, **kwargs: object) -> dict[str, object]:
            return {"content": "document fallback must not be used"}

        def get_latest_transcription(self, media_id: int) -> str:
            return "transcript fallback must not be used"

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        output_jobs.build_research_workspace_output_source_context(
            workspace_db=_WorkspaceDB(),
            media_db=_MediaDB(),
            workspace_id="ws-1",
            source_ids=["src-1"],
            max_chars=10_000,
        )

    assert excinfo.value.public_code == "source_context_empty"
    assert excinfo.value.retryable is False


def test_persist_output_bytes_creates_durable_output_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    fake_collections_db: Any,
) -> None:
    output_jobs = _output_jobs_module()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    artifact = output_jobs.persist_research_workspace_output_bytes(
        collections_db=fake_collections_db,
        user_id=42,
        job_id=7,
        artifact_type="infographic",
        title="Infographic",
        content=b"png-bytes",
        format_="png",
        content_type="image/png",
        workspace_id="ws-1",
        workspace_artifact_id="infographic-1",
    )

    assert artifact.download_url == "/api/v1/outputs/123/download"
    assert artifact.byte_size == len(b"png-bytes")
    assert fake_collections_db.created[0]["type_"] == "research_workspace_infographic"
    written_path = tmp_path / str(fake_collections_db.created[0]["storage_path"])
    assert written_path.read_bytes() == b"png-bytes"


def test_persist_output_bytes_rejects_collections_user_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    fake_collections_db: Any,
) -> None:
    output_jobs = _output_jobs_module()
    fake_collections_db.user_id = 43
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        output_jobs.persist_research_workspace_output_bytes(
            collections_db=fake_collections_db,
            user_id=42,
            job_id=7,
            artifact_type="infographic",
            title="Infographic",
            content=b"png-bytes",
            format_="png",
            content_type="image/png",
            workspace_id="ws-1",
            workspace_artifact_id="infographic-1",
        )

    assert excinfo.value.public_code == "output_user_mismatch"
    assert excinfo.value.retryable is False
    assert fake_collections_db.created == []
    assert list(tmp_path.iterdir()) == []


def test_persist_output_bytes_removes_file_when_artifact_row_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    class _FailingCollectionsDB:
        def __init__(self) -> None:
            self.user_id = 42
            self.storage_path: str | None = None

        def resolve_output_storage_path(self, path_value: str) -> str:
            self.storage_path = path_value
            return path_value

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("db unavailable")

    collections_db = _FailingCollectionsDB()

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        output_jobs.persist_research_workspace_output_bytes(
            collections_db=collections_db,
            user_id=42,
            job_id=7,
            artifact_type="infographic",
            title="Infographic",
            content=b"png-bytes",
            format_="png",
            content_type="image/png",
            workspace_id="ws-1",
            workspace_artifact_id="infographic-1",
        )

    assert excinfo.value.public_code == "output_artifact_create_failed"
    assert excinfo.value.retryable is False
    assert collections_db.storage_path is not None
    assert not (tmp_path / collections_db.storage_path).exists()


def test_persist_output_bytes_keeps_required_metadata_and_drops_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    fake_collections_db: Any,
) -> None:
    output_jobs = _output_jobs_module()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    output_jobs.persist_research_workspace_output_bytes(
        collections_db=fake_collections_db,
        user_id=42,
        job_id=7,
        artifact_type="infographic",
        title="Infographic",
        content=b"png-bytes",
        format_="png",
        content_type="image/png",
        workspace_id="ws-1",
        workspace_artifact_id="infographic-1",
        metadata={
            "origin": "caller",
            "workspace_id": "ws-2",
            "workspace_artifact_id": "other",
            "content_type": "text/plain",
            "byte_size": 999,
            "storage_path": "/tmp/secret.png",
            "note": "/private/tmp/secret.png",
            "relative_file": "tmp/render.png",
            "relative_note": "rendered from outputs/foo.png",
            "safe_note": "kept",
        },
    )

    metadata = json.loads(str(fake_collections_db.created[0]["metadata_json"]))
    assert metadata["origin"] == "research_workspace"
    assert metadata["workspace_id"] == "ws-1"
    assert metadata["workspace_artifact_id"] == "infographic-1"
    assert metadata["content_type"] == "image/png"
    assert metadata["byte_size"] == len(b"png-bytes")
    assert metadata["safe_note"] == "kept"
    assert "storage_path" not in metadata
    assert "relative_file" not in metadata
    assert "relative_note" not in metadata
    assert "/private/tmp/secret.png" not in metadata.values()


def test_persist_output_bytes_sanitizes_nested_metadata_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    fake_collections_db: Any,
) -> None:
    output_jobs = _output_jobs_module()
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    output_jobs.persist_research_workspace_output_bytes(
        collections_db=fake_collections_db,
        user_id=42,
        job_id=7,
        artifact_type="infographic",
        title="Infographic",
        content=b"png-bytes",
        format_="png",
        content_type="image/png",
        workspace_id="ws-1",
        workspace_artifact_id="infographic-1",
        metadata={
            "nested": {
                "safe": "kept",
                "safe_detail": "rendered_from=workspace-output",
                "/tmp/key": "drop-key",
                "description": "rendered from /private/tmp/source.png",
                "delimited": "rendered_from=/private/tmp/source.png",
                "jsonish": '{"path":"/private/tmp/source.png"}',
                "relative_file": "tmp/render.png",
                "relative_note": "rendered from outputs/foo.png",
                "windows_note": "loaded from C:\\Users\\secret\\source.png",
                "source_path": "/tmp/secret.png",
                "windows": "C:\\Users\\secret.png",
            },
            "items": [
                "kept",
                "/tmp/list-secret.png",
                "outputs/foo.png",
                {"inner": "safe", "home": "~/secret.png", "asset_path": "relative"},
            ],
        },
    )

    raw_metadata = str(fake_collections_db.created[0]["metadata_json"])
    metadata = json.loads(raw_metadata)
    assert metadata["nested"] == {"safe": "kept", "safe_detail": "rendered_from=workspace-output"}
    assert metadata["items"] == ["kept", {"inner": "safe"}]
    assert "/private/tmp/source.png" not in raw_metadata
    assert "/tmp/" not in raw_metadata
    assert "~/" not in raw_metadata
    assert "C:\\\\Users" not in raw_metadata
    assert "/tmp/key" not in raw_metadata
    assert "source_path" not in raw_metadata
    assert "relative_file" not in raw_metadata
    assert "relative_note" not in raw_metadata
    assert "outputs/foo.png" not in raw_metadata
    assert "asset_path" not in raw_metadata


@pytest.mark.asyncio
async def test_worker_rejects_unrelated_job_type(fake_job_manager: Any) -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await worker.process_research_workspace_output_job(
            {"id": 1, "job_type": "other", "payload": {}},
            job_manager=fake_job_manager,
        )

    assert excinfo.value.retryable is False


def test_worker_rejects_payload_user_id_mismatch() -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        worker.resolve_research_workspace_output_job_user_id(
            {"owner_user_id": "8"},
            {"user_id": "7"},
        )

    assert excinfo.value.public_code == "owner_user_id_mismatch"
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_worker_processes_valid_job_with_open_databases(
    monkeypatch: pytest.MonkeyPatch,
    fake_job_manager: Any,
) -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()
    notes_db = object()
    media_db = object()
    closed_dbs: list[object] = []
    media_calls: list[dict[str, object]] = []
    delegated: dict[str, object] = {}

    async def _open_notes_db(user_id: int) -> object:
        assert user_id == 7
        return notes_db

    @contextmanager
    def _managed_media_database(client_id: str, **kwargs: object):
        media_calls.append({"client_id": client_id, **kwargs})
        yield media_db

    async def _process_payload(**kwargs: object) -> dict[str, object]:
        delegated.update(kwargs)
        return {"artifact_id": "artifact-1"}

    monkeypatch.setattr(worker, "open_research_workspace_output_notes_db", _open_notes_db)
    monkeypatch.setattr(worker, "close_research_workspace_output_notes_db", closed_dbs.append)
    monkeypatch.setattr(worker, "managed_media_database", _managed_media_database)
    monkeypatch.setattr(worker, "process_research_workspace_output_payload", _process_payload)

    result = await worker.process_research_workspace_output_job(
        {
            "id": 10,
            "job_type": output_jobs.RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"user_id": "7", "artifact_id": "artifact-1"},
        },
        job_manager=fake_job_manager,
    )

    assert result == {"artifact_id": "artifact-1"}
    assert closed_dbs == [notes_db]
    assert media_calls == [
        {
            "client_id": "research_workspace_output_worker",
            "db_path": str(worker.DatabasePaths.get_media_db_path(7)),
            "initialize": False,
        }
    ]
    assert delegated["workspace_db"] is notes_db
    assert delegated["media_db"] is media_db
    assert delegated["job_manager"] is fake_job_manager
    assert delegated["user_id"] == 7
    assert delegated["payload"] == {"user_id": "7", "artifact_id": "artifact-1"}


@pytest.mark.asyncio
async def test_worker_runner_filters_worker_sdk_to_research_output_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_module()
    output_jobs = _output_jobs_module()
    captured: dict[str, object] = {}

    class _FakeWorkerSDK:
        def __init__(self, job_manager: object, config: object) -> None:
            captured["job_manager"] = job_manager
            captured["config"] = config

        async def run(self, **kwargs: object) -> None:
            captured["run_kwargs"] = kwargs

        def stop(self) -> None:
            captured["stopped"] = True

    monkeypatch.setenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ID", "rw-output-test")
    monkeypatch.setenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_QUEUE", "high")
    monkeypatch.setattr(worker, "WorkerSDK", _FakeWorkerSDK)

    await worker.run_research_workspace_output_jobs_worker()

    config = captured["config"]
    assert config.worker_id == "rw-output-test"
    assert config.queue == "high"
    assert captured["run_kwargs"]["job_type"] == output_jobs.RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE
    assert callable(captured["run_kwargs"]["handler"])
    assert callable(captured["run_kwargs"]["progress_cb"])
