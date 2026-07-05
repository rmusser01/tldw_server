from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import FileArtifactsError


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


class _VideoWorkspaceDB:
    def __init__(self, *, fail_complete_update: bool = False) -> None:
        self.artifact = {"id": "video_overview-1", "version": 1, "export_refs": []}
        self.fail_complete_update = fail_complete_update
        self.updates: list[dict[str, Any]] = []

    def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any] | None:
        assert workspace_id == "ws-1"
        assert artifact_id == "video_overview-1"
        return dict(self.artifact)

    def update_workspace_artifact(
        self,
        workspace_id: str,
        artifact_id: str,
        updates: dict[str, Any],
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        assert workspace_id == "ws-1"
        assert artifact_id == "video_overview-1"
        assert expected_version == self.artifact["version"]
        if self.fail_complete_update and updates.get("status") == "complete":
            raise RuntimeError("stale workspace artifact version")
        self.updates.append(updates)
        self.artifact.update(updates)
        self.artifact["version"] = expected_version + 1
        return dict(self.artifact)


def _video_slide(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "order": 0,
        "layout": "content",
        "title": "Slide",
        "content": "Point",
        "speaker_notes": "Narration",
        "metadata": metadata or {},
    }


def _video_payload(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "workspace_id": "ws-1",
        "artifact_id": "video_overview-1",
        "artifact_type": "video_overview",
        "source_ids": ["src-1"],
        "settings": settings or {"title_hint": "Video Brief"},
        "user_id": "42",
    }


def _install_video_overview_success_doubles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    output_jobs: Any,
    *,
    slides: list[dict[str, Any]] | None = None,
    tts_impl: Any | None = None,
    render_impl: Any | None = None,
) -> dict[str, Any]:
    generated_slides = slides or [_video_slide()]
    render_calls: list[dict[str, Any]] = []
    tts_calls: list[dict[str, Any]] = []
    generator_calls: list[dict[str, Any]] = []
    presentation_calls: list[dict[str, Any]] = []

    class _CollectionsDB:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.user_id = 42
            self.created: list[dict[str, Any]] = []
            self.deleted: list[dict[str, Any]] = []
            self.__class__.instances.append(self)

        @classmethod
        def for_user(cls, user_id: str) -> Any:
            assert user_id == "42"
            return cls()

        def __enter__(self) -> "_CollectionsDB":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def resolve_output_storage_path(self, path_value: str) -> str:
            return str(path_value)

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            output_id = 200 + len(self.created)
            row = {**kwargs, "id": output_id}
            self.created.append(row)
            return SimpleNamespace(id=output_id)

        def delete_output_artifact(self, output_id: int, *, hard: bool = False) -> bool:
            self.deleted.append({"output_id": output_id, "hard": hard})
            return True

    class _SlidesGenerator:
        def generate_from_text(self, **kwargs: object) -> dict[str, Any]:
            assert kwargs["source_text"] == "source facts"
            generator_calls.append(kwargs)
            return {"title": "Video Brief", "slides": generated_slides}

    class _SlidesDatabase:
        instances: list[Any] = []

        def __init__(self, *, db_path: str, client_id: str) -> None:
            assert db_path == str(tmp_path / "slides.db")
            assert client_id == "42"
            self.closed = False
            self.__class__.instances.append(self)

        def create_presentation(self, **kwargs: object) -> SimpleNamespace:
            presentation_calls.append(kwargs)
            self.slides = json.loads(str(kwargs["slides"]))
            return SimpleNamespace(id="presentation-1", title=kwargs["title"], version=1)

        def close_connection(self) -> None:
            self.closed = True

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )

    async def _generate_tts_audio_bytes(**kwargs: Any) -> bytes:
        tts_calls.append(kwargs)
        if tts_impl is not None:
            return await tts_impl(**kwargs)
        return b"mp3-bytes"

    def _render_presentation_video(**kwargs: Any) -> SimpleNamespace:
        render_calls.append(kwargs)
        if render_impl is not None:
            return render_impl(**kwargs)
        output_path = tmp_path / "video-overview.mp4"
        output_path.write_bytes(b"mp4-bytes")
        return SimpleNamespace(
            output_format="mp4",
            storage_path=output_path.name,
            output_path=output_path,
            byte_size=output_path.stat().st_size,
        )

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "SlidesGenerator", _SlidesGenerator, raising=False)
    monkeypatch.setattr(output_jobs, "SlidesDatabase", _SlidesDatabase, raising=False)
    monkeypatch.setattr(output_jobs, "generate_tts_audio_bytes", _generate_tts_audio_bytes, raising=False)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", _CollectionsDB)
    monkeypatch.setattr(output_jobs, "render_presentation_video", _render_presentation_video, raising=False)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)
    monkeypatch.setattr(DatabasePaths, "get_slides_db_path", lambda _user_id: tmp_path / "slides.db")

    return {
        "collections_cls": _CollectionsDB,
        "slides_db_cls": _SlidesDatabase,
        "render_calls": render_calls,
        "tts_calls": tts_calls,
        "generator_calls": generator_calls,
        "presentation_calls": presentation_calls,
    }


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
            "plain_note": "report.pdf",
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
    assert "plain_note" not in metadata
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
                "encoded_note": '{"path":"report.pdf"}',
                "jsonish": '{"path":"/private/tmp/source.png"}',
                "localPath": "report.pdf",
                "plain": "render.png",
                "relative_file": "tmp/render.png",
                "relative_note": "rendered from outputs/foo.png",
                "sourcePath": "report.pdf",
                "windows_note": "loaded from C:\\Users\\secret\\source.png",
                "source_path": "/tmp/secret.png",
                "windows": "C:\\Users\\secret.png",
            },
            "items": [
                "kept",
                "audio.mp3",
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
    assert "encoded_note" not in raw_metadata
    assert "localPath" not in raw_metadata
    assert "report.pdf" not in raw_metadata
    assert "render.png" not in raw_metadata
    assert "audio.mp3" not in raw_metadata
    assert "relative_file" not in raw_metadata
    assert "relative_note" not in raw_metadata
    assert "sourcePath" not in raw_metadata
    assert "outputs/foo.png" not in raw_metadata
    assert "asset_path" not in raw_metadata


@pytest.mark.asyncio
async def test_infographic_worker_generates_image_and_updates_workspace_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    fake_image_bytes = b"\x89PNG\r\n\x1a\nfake-adapter-png"

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "infographic-1", "version": 3, "export_refs": []}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any] | None:
            assert workspace_id == "ws-1"
            assert artifact_id == "infographic-1"
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            assert workspace_id == "ws-1"
            assert artifact_id == "infographic-1"
            assert expected_version == self.artifact["version"]
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _CollectionsDB:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.user_id = 42
            self.created: list[dict[str, object]] = []
            self.__class__.instances.append(self)

        @classmethod
        def for_user(cls, user_id: str) -> Any:
            assert user_id == "42"
            return cls()

        def __enter__(self) -> "_CollectionsDB":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def resolve_output_storage_path(self, path_value: str) -> str:
            assert "/" not in path_value
            return path_value

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            self.created.append(kwargs)
            return SimpleNamespace(id=123)

    class _ImageAdapter:
        def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload["backend"] == "fake-image"
            assert payload["prompt"] == "fake infographic prompt"
            assert payload["width"] == 768
            assert payload["height"] == 512
            return {"backend": "fake-image", "prompt": payload["prompt"]}

        def validate(self, structured: dict[str, Any]) -> list[object]:
            assert structured["backend"] == "fake-image"
            return []

        def export(self, structured: dict[str, Any], *, format: str) -> SimpleNamespace:
            assert structured["backend"] == "fake-image"
            assert format == "png"
            return SimpleNamespace(content=fake_image_bytes, content_type="image/png", bytes_len=len(fake_image_bytes))

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )
    workspace_db = _WorkspaceDB()

    monkeypatch.setattr(
        output_jobs,
        "build_research_workspace_output_source_context",
        lambda **_: context,
    )
    monkeypatch.setattr(
        output_jobs,
        "generate_infographic_prompt",
        lambda **_: "fake infographic prompt",
        raising=False,
    )
    monkeypatch.setattr(output_jobs, "ImageAdapter", _ImageAdapter, raising=False)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", _CollectionsDB)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    result = await output_jobs.process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload={
            "workspace_id": "ws-1",
            "artifact_id": "infographic-1",
            "artifact_type": "infographic",
            "source_ids": ["src-1"],
            "settings": {"image_backend": "fake-image", "image_width": 768, "image_height": 512},
            "user_id": "42",
        },
        workspace_db=workspace_db,
        media_db=object(),
        user_id=42,
        job_manager=object(),
    )

    assert result["output_id"] == 123
    assert result["download_url"] == "/api/v1/outputs/123/download"
    update = workspace_db.updates[-1]
    assert update["status"] == "complete"
    assert update["content_type"] == "image/png"
    assert update["export_refs"][0]["url"] == "/api/v1/outputs/123/download"
    assert update["export_refs"][0]["content_type"] == "image/png"
    assert update["export_refs"][0]["bytes"] == len(fake_image_bytes)
    assert (tmp_path / str(_CollectionsDB.instances[0].created[0]["storage_path"])).read_bytes() == fake_image_bytes


@pytest.mark.asyncio
async def test_infographic_worker_cleans_output_when_final_workspace_update_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    fake_image_bytes = b"\x89PNG\r\n\x1a\nfake-adapter-png"

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "infographic-1", "version": 3, "export_refs": []}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any] | None:
            assert workspace_id == "ws-1"
            assert artifact_id == "infographic-1"
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            assert workspace_id == "ws-1"
            assert artifact_id == "infographic-1"
            assert expected_version == self.artifact["version"]
            if updates.get("status") == "complete":
                raise RuntimeError("stale workspace artifact version")
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _CollectionsDB:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.user_id = 42
            self.created: list[dict[str, object]] = []
            self.deleted: list[dict[str, object]] = []
            self.__class__.instances.append(self)

        @classmethod
        def for_user(cls, user_id: str) -> Any:
            assert user_id == "42"
            return cls()

        def __enter__(self) -> "_CollectionsDB":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def resolve_output_storage_path(self, path_value: str) -> str:
            return path_value

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            output_id = 123 + len(self.created)
            row = {**kwargs, "id": output_id}
            self.created.append(row)
            return SimpleNamespace(id=output_id)

        def delete_output_artifact(self, output_id: int, *, hard: bool = False) -> bool:
            self.deleted.append({"output_id": output_id, "hard": hard})
            return True

    class _ImageAdapter:
        def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"backend": payload["backend"], "prompt": payload["prompt"]}

        def validate(self, structured: dict[str, Any]) -> list[object]:
            return []

        def export(self, structured: dict[str, Any], *, format: str) -> SimpleNamespace:
            assert format == "png"
            return SimpleNamespace(content=fake_image_bytes, content_type="image/png")

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )
    workspace_db = _WorkspaceDB()

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "generate_infographic_prompt", lambda **_: "fake prompt", raising=False)
    monkeypatch.setattr(output_jobs, "ImageAdapter", _ImageAdapter, raising=False)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", _CollectionsDB)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "workspace_artifact_update_failed"
    created_rows = _CollectionsDB.instances[0].created
    deleted = [
        item
        for instance in _CollectionsDB.instances
        for item in instance.deleted
    ]
    assert {item["output_id"] for item in deleted} == {row["id"] for row in created_rows}
    assert all(item["hard"] is True for item in deleted)
    for row in created_rows:
        assert not (tmp_path / str(row["storage_path"])).exists()
    assert workspace_db.updates[-1]["status"] == "failed"
    assert workspace_db.updates[-1]["producer_metadata"]["error"] == "workspace_artifact_update_failed"


@pytest.mark.asyncio
async def test_video_overview_worker_generates_narrated_slideshow_mp4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "video_overview-1", "version": 4, "export_refs": []}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any] | None:
            assert workspace_id == "ws-1"
            assert artifact_id == "video_overview-1"
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            assert expected_version == self.artifact["version"]
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _CollectionsDB:
        instances: list[Any] = []

        def __init__(self) -> None:
            self.user_id = 42
            self.created: list[dict[str, object]] = []
            self.__class__.instances.append(self)

        @classmethod
        def for_user(cls, user_id: str) -> Any:
            assert user_id == "42"
            return cls()

        def __enter__(self) -> "_CollectionsDB":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def resolve_output_storage_path(self, path_value: str) -> str:
            assert "/" not in path_value
            return path_value

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            output_id = 200 + len(self.created)
            self.created.append(kwargs)
            return SimpleNamespace(id=output_id)

    class _SlidesGenerator:
        def generate_from_text(self, **kwargs: object) -> dict[str, Any]:
            assert kwargs["source_text"] == "source facts"
            assert kwargs["title_hint"] == "Video Brief"
            return {
                "title": "Video Brief",
                "slides": [
                    {
                        "order": index,
                        "layout": "content",
                        "title": f"Slide {index}",
                        "content": f"Point {index}",
                        "speaker_notes": f"Narration {index}",
                        "metadata": {},
                    }
                    for index in range(9)
                ],
            }

    class _SlidesDatabase:
        def __init__(self, *, db_path: str, client_id: str) -> None:
            assert db_path == str(tmp_path / "slides.db")
            assert client_id == "42"

        def create_presentation(self, **kwargs: object) -> SimpleNamespace:
            slides = json.loads(str(kwargs["slides"]))
            assert len(slides) == 8
            return SimpleNamespace(id="presentation-1", title=kwargs["title"], version=1)

    class _JobManager:
        def __init__(self) -> None:
            self.created_jobs: list[dict[str, object]] = []
            self.progress: list[dict[str, object]] = []

        def create_job(self, **kwargs: object) -> object:
            self.created_jobs.append(kwargs)
            raise AssertionError("video overview worker must not enqueue render jobs")

        def update_job_progress(self, *args: object, **kwargs: object) -> None:
            self.progress.append({"args": args, "kwargs": kwargs})

    render_calls: list[dict[str, Any]] = []

    def _render_presentation_video(**kwargs: Any) -> SimpleNamespace:
        render_calls.append(kwargs)
        slides = kwargs["slides"]
        assert all(
            str(slide["metadata"]["studio"]["audio"]["asset_ref"]).startswith("output:")
            for slide in slides
        )
        return SimpleNamespace(
            output_format="mp4",
            storage_path="renders/video-overview.mp4",
            output_path=tmp_path / "renders/video-overview.mp4",
            byte_size=4567,
        )

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )
    workspace_db = _WorkspaceDB()
    job_manager = _JobManager()

    async def _generate_tts_audio_bytes(**kwargs: object) -> bytes:
        return b"mp3-bytes"

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "SlidesGenerator", _SlidesGenerator, raising=False)
    monkeypatch.setattr(output_jobs, "SlidesDatabase", _SlidesDatabase, raising=False)
    monkeypatch.setattr(output_jobs, "generate_tts_audio_bytes", _generate_tts_audio_bytes, raising=False)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", _CollectionsDB)
    monkeypatch.setattr(output_jobs, "render_presentation_video", _render_presentation_video, raising=False)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)
    monkeypatch.setattr(DatabasePaths, "get_slides_db_path", lambda _user_id: tmp_path / "slides.db")

    result = await output_jobs.process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload={
            "workspace_id": "ws-1",
            "artifact_id": "video_overview-1",
            "artifact_type": "video_overview",
            "source_ids": ["src-1"],
            "settings": {
                "provider": "openai",
                "model": "gpt-test",
                "title_hint": "Video Brief",
                "tts_provider": "openai",
                "tts_model": "tts-test",
                "tts_voice": "alloy",
            },
            "user_id": "42",
        },
        workspace_db=workspace_db,
        media_db=object(),
        user_id=42,
        job_manager=job_manager,
    )

    assert result["format"] == "mp4"
    assert result["output_id"] == 208
    assert result["download_url"] == "/api/v1/outputs/208/download"
    assert len(render_calls) == 1
    assert render_calls[0]["output_format"] == "mp4"
    assert render_calls[0]["output_dir"] == tmp_path
    assert job_manager.created_jobs == []

    update = workspace_db.updates[-1]
    assert update["status"] == "complete"
    assert update["content_type"] == "video/mp4"
    assert update["export_refs"][0]["url"] == "/api/v1/outputs/208/download"

    final_row = _CollectionsDB.instances[0].created[-1]
    assert final_row["type_"] == "research_workspace_video_overview"
    assert final_row["format_"] == "mp4"
    assert final_row["storage_path"] == "renders/video-overview.mp4"


@pytest.mark.asyncio
async def test_video_overview_worker_uses_provider_specific_tts_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    workspace_db = _VideoWorkspaceDB()
    captures = _install_video_overview_success_doubles(monkeypatch, tmp_path, output_jobs)

    await output_jobs.process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload=_video_payload(settings={"title_hint": "Video Brief", "tts_provider": "openai"}),
        workspace_db=workspace_db,
        media_db=object(),
        user_id=42,
        job_manager=object(),
    )

    assert captures["tts_calls"][0]["provider"] == "openai"
    assert captures["tts_calls"][0]["model"] == "tts-1"
    assert captures["tts_calls"][0]["voice"] == "alloy"
    final_row = captures["collections_cls"].instances[0].created[-1]
    final_metadata = json.loads(str(final_row["metadata_json"]))
    assert final_metadata["tts_provider"] == "openai"
    assert final_metadata["tts_model"] == "tts-1"
    assert final_metadata["tts_voice"] == "alloy"


@pytest.mark.asyncio
async def test_video_overview_worker_applies_builtin_visual_style(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    workspace_db = _VideoWorkspaceDB()
    captures = _install_video_overview_success_doubles(monkeypatch, tmp_path, output_jobs)

    await output_jobs.process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload=_video_payload(
            settings={
                "title_hint": "Video Brief",
                "slides_visual_style_id": "notebooklm-blueprint",
            }
        ),
        workspace_db=workspace_db,
        media_db=object(),
        user_id=42,
        job_manager=object(),
    )

    generator_snapshot = captures["generator_calls"][0]["visual_style_snapshot"]
    assert generator_snapshot["id"] == "notebooklm-blueprint"
    assert generator_snapshot["scope"] == "builtin"

    presentation_kwargs = captures["presentation_calls"][0]
    assert presentation_kwargs["visual_style_id"] == "notebooklm-blueprint"
    assert presentation_kwargs["visual_style_scope"] == "builtin"
    assert presentation_kwargs["visual_style_name"] == "Blueprint"
    assert presentation_kwargs["visual_style_version"] == 1
    persisted_snapshot = json.loads(str(presentation_kwargs["visual_style_snapshot"]))
    assert persisted_snapshot["id"] == "notebooklm-blueprint"
    assert persisted_snapshot["scope"] == "builtin"


@pytest.mark.asyncio
async def test_video_overview_worker_ignores_progress_update_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    workspace_db = _VideoWorkspaceDB()
    _install_video_overview_success_doubles(monkeypatch, tmp_path, output_jobs)

    class _FailingProgressJobManager:
        def update_job_progress(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("progress database unavailable")

    result = await output_jobs.process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload=_video_payload(),
        workspace_db=workspace_db,
        media_db=object(),
        user_id=42,
        job_manager=_FailingProgressJobManager(),
    )

    assert result["format"] == "mp4"
    assert workspace_db.updates[-1]["status"] == "complete"


@pytest.mark.asyncio
async def test_video_overview_worker_strips_generated_slide_metadata_before_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    workspace_db = _VideoWorkspaceDB()
    captures = _install_video_overview_success_doubles(
        monkeypatch,
        tmp_path,
        output_jobs,
        slides=[
            _video_slide(
                metadata={
                    "images": [{"data_b64": "untrusted-inline-image"}],
                    "path": "/private/tmp/untrusted.png",
                    "studio": {
                        "audio": {"asset_ref": "output:evil"},
                        "presenter_note": "untrusted",
                    },
                }
            )
        ],
    )

    await output_jobs.process_research_workspace_output_payload(
        job={"id": 7, "owner_user_id": "42"},
        payload=_video_payload(),
        workspace_db=workspace_db,
        media_db=object(),
        user_id=42,
        job_manager=object(),
    )

    rendered_slide = captures["render_calls"][0]["slides"][0]
    metadata = rendered_slide["metadata"]
    assert set(metadata) == {"studio"}
    assert set(metadata["studio"]) == {"audio"}
    assert metadata["studio"]["audio"]["asset_ref"].startswith("output:")
    assert metadata["studio"]["audio"]["asset_ref"] != "output:evil"


@pytest.mark.asyncio
async def test_video_overview_worker_maps_tts_failures_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    workspace_db = _VideoWorkspaceDB()

    async def _fail_tts(**kwargs: object) -> bytes:
        raise RuntimeError("/private/tmp/provider-secret.mp3")

    _install_video_overview_success_doubles(
        monkeypatch,
        tmp_path,
        output_jobs,
        tts_impl=_fail_tts,
    )

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload=_video_payload(),
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "tts_generation_failed"
    assert excinfo.value.retryable is True
    update = workspace_db.updates[-1]
    assert update["status"] == "failed"
    assert update["producer_metadata"]["error"] == "tts_generation_failed"
    assert "/private/tmp" not in json.dumps(update)


@pytest.mark.asyncio
async def test_video_overview_worker_cleans_outputs_when_final_workspace_update_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()
    workspace_db = _VideoWorkspaceDB(fail_complete_update=True)
    render_path = tmp_path / "video-overview.mp4"

    def _render_video(**kwargs: object) -> SimpleNamespace:
        render_path.write_bytes(b"mp4-bytes")
        return SimpleNamespace(
            output_format="mp4",
            storage_path=render_path.name,
            output_path=render_path,
            byte_size=render_path.stat().st_size,
        )

    captures = _install_video_overview_success_doubles(
        monkeypatch,
        tmp_path,
        output_jobs,
        render_impl=_render_video,
    )

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload=_video_payload(),
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "workspace_artifact_update_failed"
    collections_db = captures["collections_cls"].instances[0]
    created_ids = [row["id"] for row in collections_db.created]
    deleted = [
        item
        for instance in captures["collections_cls"].instances
        for item in instance.deleted
    ]
    assert {item["output_id"] for item in deleted} == set(created_ids)
    assert all(item["hard"] is True for item in deleted)
    for row in collections_db.created:
        assert not (tmp_path / str(row["storage_path"])).exists()
    assert not render_path.exists()
    assert workspace_db.updates[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_infographic_worker_marks_workspace_artifact_failed_on_generation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "infographic-1", "version": 3}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any] | None:
            assert workspace_id == "ws-1"
            assert artifact_id == "infographic-1"
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            assert expected_version == self.artifact["version"]
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _CollectionsDB:
        @classmethod
        def for_user(cls, user_id: str) -> Any:
            assert user_id == "42"
            return cls()

        def __enter__(self) -> "_CollectionsDB":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    class _ImageAdapter:
        def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"backend": "fake-image", "prompt": payload["prompt"]}

        def validate(self, structured: dict[str, Any]) -> list[object]:
            return []

        def export(self, structured: dict[str, Any], *, format: str) -> SimpleNamespace:
            raise RuntimeError("/private/tmp/provider-secret.png")

    workspace_db = _WorkspaceDB()
    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "generate_infographic_prompt", lambda **_: "fake prompt", raising=False)
    monkeypatch.setattr(output_jobs, "ImageAdapter", _ImageAdapter, raising=False)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", _CollectionsDB)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "infographic_generation_failed"
    update = workspace_db.updates[-1]
    assert update["status"] == "failed"
    assert update["content_type"] == "image/png"
    assert update["producer_metadata"]["error"] == "infographic_generation_failed"
    assert "/private/tmp" not in json.dumps(update)


@pytest.mark.asyncio
async def test_video_overview_worker_marks_workspace_artifact_failed_on_render_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "video_overview-1", "version": 1}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any]:
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _CollectionsDB:
        def __init__(self) -> None:
            self.user_id = 42
            self.created: list[dict[str, object]] = []

        @classmethod
        def for_user(cls, user_id: str) -> Any:
            return cls()

        def __enter__(self) -> "_CollectionsDB":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def resolve_output_storage_path(self, path_value: str) -> str:
            return path_value

        def create_output_artifact(self, **kwargs: object) -> SimpleNamespace:
            self.created.append(kwargs)
            return SimpleNamespace(id=len(self.created))

    class _SlidesGenerator:
        def generate_from_text(self, **kwargs: object) -> dict[str, Any]:
            return {
                "title": "Video Brief",
                "slides": [
                    {
                        "order": 0,
                        "layout": "content",
                        "title": "Slide",
                        "content": "Point",
                        "speaker_notes": "Narration",
                        "metadata": {},
                    }
                ],
            }

    class _SlidesDatabase:
        def __init__(self, **kwargs: object) -> None:
            return None

        def create_presentation(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(id="presentation-1", title="Video Brief", version=1)

    async def _generate_tts_audio_bytes(**kwargs: object) -> bytes:
        return b"mp3-bytes"

    def _render_presentation_video(**kwargs: object) -> object:
        raise output_jobs.PresentationRenderError("presentation_render_failed")

    workspace_db = _WorkspaceDB()
    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "SlidesGenerator", _SlidesGenerator, raising=False)
    monkeypatch.setattr(output_jobs, "SlidesDatabase", _SlidesDatabase, raising=False)
    monkeypatch.setattr(output_jobs, "generate_tts_audio_bytes", _generate_tts_audio_bytes, raising=False)
    monkeypatch.setattr(output_jobs, "CollectionsDatabase", _CollectionsDB)
    monkeypatch.setattr(output_jobs, "render_presentation_video", _render_presentation_video, raising=False)
    monkeypatch.setattr(DatabasePaths, "get_user_outputs_dir", lambda _user_id: tmp_path)
    monkeypatch.setattr(DatabasePaths, "get_slides_db_path", lambda _user_id: tmp_path / "slides.db")

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "video_overview-1",
                "artifact_type": "video_overview",
                "source_ids": ["src-1"],
                "settings": {"title_hint": "Video Brief"},
                "user_id": "42",
            },
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "presentation_render_failed"
    update = workspace_db.updates[-1]
    assert update["status"] == "failed"
    assert update["content_type"] == "video/mp4"
    assert update["producer_metadata"]["error"] == "presentation_render_failed"
    assert "/private/tmp" not in json.dumps(update)


@pytest.mark.asyncio
async def test_infographic_worker_preserves_original_error_when_failure_marker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any]:
            return {"id": artifact_id, "version": 1}

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            raise RuntimeError("conflict while marking failed")

    def _raise_source_error(**kwargs: object) -> None:
        raise output_jobs.ResearchWorkspaceOutputJobError("source_context_empty", retryable=False)

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", _raise_source_error)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=_WorkspaceDB(),
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "source_context_empty"


@pytest.mark.asyncio
async def test_infographic_worker_preserves_image_adapter_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "infographic-1", "version": 1}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any]:
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _ImageAdapter:
        def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
            raise FileArtifactsError("image_backend_unavailable", detail="/private/tmp/secret")

        def validate(self, structured: dict[str, Any]) -> list[object]:
            return []

        def export(self, structured: dict[str, Any], *, format: str) -> SimpleNamespace:
            raise AssertionError("export should not run")

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )
    workspace_db = _WorkspaceDB()

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "generate_infographic_prompt", lambda **_: "fake prompt", raising=False)
    monkeypatch.setattr(output_jobs, "ImageAdapter", _ImageAdapter, raising=False)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "image_backend_unavailable"
    assert excinfo.value.retryable is True
    update = workspace_db.updates[-1]
    assert update["status"] == "failed"
    assert update["producer_metadata"]["error"] == "image_backend_unavailable"
    assert "/private/tmp" not in json.dumps(update)


@pytest.mark.asyncio
async def test_infographic_worker_rejects_malformed_job_id() -> None:
    output_jobs = _output_jobs_module()

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": "not-an-int", "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=object(),
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "invalid_job_id"
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_infographic_worker_rejects_non_png_image_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "infographic-1", "version": 1}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any]:
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _ImageAdapter:
        def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"backend": "fake-image", "prompt": payload["prompt"]}

        def validate(self, structured: dict[str, Any]) -> list[object]:
            return []

        def export(self, structured: dict[str, Any], *, format: str) -> SimpleNamespace:
            return SimpleNamespace(content=b"jpeg-bytes", content_type="image/jpeg", bytes_len=10)

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )
    workspace_db = _WorkspaceDB()

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "generate_infographic_prompt", lambda **_: "fake prompt", raising=False)
    monkeypatch.setattr(output_jobs, "ImageAdapter", _ImageAdapter, raising=False)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "image_content_type_invalid"
    update = workspace_db.updates[-1]
    assert update["status"] == "failed"
    assert update["producer_metadata"]["error"] == "image_content_type_invalid"


@pytest.mark.asyncio
async def test_infographic_worker_rejects_non_png_bytes_when_image_export_omits_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_jobs = _output_jobs_module()

    class _WorkspaceDB:
        def __init__(self) -> None:
            self.artifact = {"id": "infographic-1", "version": 1}
            self.updates: list[dict[str, Any]] = []

        def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any]:
            return dict(self.artifact)

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict[str, Any],
            *,
            expected_version: int,
        ) -> dict[str, Any]:
            self.updates.append(updates)
            self.artifact.update(updates)
            self.artifact["version"] = expected_version + 1
            return dict(self.artifact)

    class _ImageAdapter:
        def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {"backend": "fake-image", "prompt": payload["prompt"]}

        def validate(self, structured: dict[str, Any]) -> list[object]:
            return []

        def export(self, structured: dict[str, Any], *, format: str) -> SimpleNamespace:
            return SimpleNamespace(content=b"\xff\xd8jpeg-bytes", content_type=None, bytes_len=12)

    context = output_jobs.ResearchWorkspaceOutputSourceContext(
        text="source facts",
        preview_text="source preview",
        source_lineage={"selected_source_ids": ["src-1"], "usable_source_ids": ["src-1"]},
    )
    workspace_db = _WorkspaceDB()

    monkeypatch.setattr(output_jobs, "build_research_workspace_output_source_context", lambda **_: context)
    monkeypatch.setattr(output_jobs, "generate_infographic_prompt", lambda **_: "fake prompt", raising=False)
    monkeypatch.setattr(output_jobs, "ImageAdapter", _ImageAdapter, raising=False)

    with pytest.raises(output_jobs.ResearchWorkspaceOutputJobError) as excinfo:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7, "owner_user_id": "42"},
            payload={
                "workspace_id": "ws-1",
                "artifact_id": "infographic-1",
                "artifact_type": "infographic",
                "source_ids": ["src-1"],
                "settings": {"image_backend": "fake-image"},
                "user_id": "42",
            },
            workspace_db=workspace_db,
            media_db=object(),
            user_id=42,
            job_manager=object(),
        )

    assert excinfo.value.public_code == "image_content_type_invalid"
    update = workspace_db.updates[-1]
    assert update["status"] == "failed"
    assert update["producer_metadata"]["error"] == "image_content_type_invalid"


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
