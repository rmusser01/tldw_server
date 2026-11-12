from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import status

import tldw_Server_API.app.core.Embeddings.jobs_adapter as jobs_adapter_module
import tldw_Server_API.app.core.Embeddings.media_embeddings as core_media_embeddings


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "expected_dispatch"),
    [
        ("jobs", "jobs"),
        ("background", "background"),
    ],
)
def test_media_add_embeddings_dispatch_modes(
    mode: str,
    expected_dispatch: str,
    monkeypatch,
    test_client,
    auth_headers,
    test_media_dir: Path,
):
    monkeypatch.setenv("MEDIA_ADD_EMBEDDINGS_MODE", mode)

    calls: dict[str, int | dict] = {"jobs": 0, "background": 0}

    async def fake_get_media_content(media_id: int, db):  # noqa: ARG001 - signature compatibility
        return {
            "media_item": {
                "title": f"Media {media_id}",
                "author": "tester",
                "metadata": {},
            },
            "content": {"content": "Embeddings dispatch integration content."},
        }

    async def fake_generate_embeddings_for_media(**kwargs):
        calls["background"] = int(calls["background"]) + 1
        calls["kwargs_seen"] = dict(kwargs)
        return {
            "status": "success",
            "embedding_count": 1,
            "chunks_processed": 1,
            "kwargs_seen": kwargs,
        }

    def fake_create_job(self, **kwargs):  # noqa: ANN001
        calls["jobs"] = int(calls["jobs"]) + 1
        calls["job_kwargs"] = dict(kwargs)
        return {"uuid": "job-dispatch-mode-test"}

    if mode == "jobs":
        async def fail_if_background_called(**kwargs):  # noqa: ANN001
            raise AssertionError(f"background dispatch should not run in jobs mode: {kwargs}")

        monkeypatch.setattr(
            jobs_adapter_module.EmbeddingsJobsAdapter,
            "create_job",
            fake_create_job,
        )
        monkeypatch.setattr(
            core_media_embeddings,
            "get_media_content",
            fake_get_media_content,
        )
        monkeypatch.setattr(
            core_media_embeddings,
            "generate_embeddings_for_media",
            fail_if_background_called,
        )
    else:
        def fail_if_jobs_called(self, **kwargs):  # noqa: ANN001
            raise AssertionError(f"jobs dispatch should not run in background mode: {kwargs}")

        monkeypatch.setattr(
            jobs_adapter_module.EmbeddingsJobsAdapter,
            "create_job",
            fail_if_jobs_called,
        )
        monkeypatch.setattr(
            core_media_embeddings,
            "get_media_content",
            fake_get_media_content,
        )
        monkeypatch.setattr(
            core_media_embeddings,
            "generate_embeddings_for_media",
            fake_generate_embeddings_for_media,
        )

    title = f"Embeddings Dispatch {mode} {uuid4().hex}"
    source_path = test_media_dir / f"dispatch_{mode}.txt"
    source_path.write_text(
        "Mode dispatch content for media add embeddings.",
        encoding="utf-8",
    )

    with source_path.open("rb") as handle:
        response = test_client.post(
            "/api/v1/media/add",
            data={
                "media_type": "document",
                "title": title,
                "generate_embeddings": "true",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_provider": "huggingface",
                "chunk_method": "words",
                "chunk_size": "128",
                "chunk_overlap": "16",
            },
            files=[("files", (source_path.name, handle, "text/plain"))],
            headers=auth_headers,
        )

    assert response.status_code in (status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS), response.text
    payload = response.json()
    success_row = next(
        (
            row
            for row in payload.get("results", [])
            if row.get("status") in {"Success", "Warning"} and row.get("db_id") is not None
        ),
        None,
    )
    assert success_row is not None, payload
    assert success_row.get("embeddings_scheduled") is True
    assert success_row.get("embeddings_dispatch") == expected_dispatch
    assert isinstance(success_row.get("embeddings_provenance"), dict)
    assert success_row["embeddings_provenance"]["origin"] == "media_add"
    assert success_row["embeddings_provenance"]["source_id"] == int(success_row["db_id"])

    if mode == "jobs":
        assert success_row.get("embeddings_job_id") == "job-dispatch-mode-test"
        assert int(calls["jobs"]) >= 1
        assert int(calls["background"]) == 0
        assert calls["job_kwargs"]["request_source"] == "media_add"
    else:
        assert "embeddings_job_id" not in success_row
        assert int(calls["jobs"]) == 0
        assert int(calls["background"]) >= 1
        assert calls["background"] >= 1
        detail_response = test_client.get(
            f"/api/v1/media/{success_row['db_id']}",
            params={
                "include_content": "true",
                "include_versions": "false",
                "include_version_content": "false",
            },
            headers=auth_headers,
        )
        assert detail_response.status_code == status.HTTP_200_OK, detail_response.text
        detail_payload = detail_response.json()
        assert detail_payload["processing"]["vector_processing_status"] == 1
        background_kwargs = calls.get("kwargs_seen") or {}
        assert background_kwargs.get("user_id") == "1"


@pytest.mark.integration
def test_media_add_embeddings_form_dispatch_override_wins_over_env(
    monkeypatch,
    test_client,
    auth_headers,
    test_media_dir: Path,
):
    monkeypatch.setenv("MEDIA_ADD_EMBEDDINGS_MODE", "jobs")

    calls: dict[str, int | dict] = {"jobs": 0, "background": 0}

    async def fake_get_media_content(media_id: int, db):  # noqa: ARG001 - signature compatibility
        return {
            "media_item": {
                "title": f"Media {media_id}",
                "author": "tester",
                "metadata": {},
            },
            "content": {"content": "Embeddings dispatch integration content."},
        }

    async def fake_generate_embeddings_for_media(**kwargs):
        calls["background"] = int(calls["background"]) + 1
        calls["kwargs_seen"] = dict(kwargs)
        return {
            "status": "success",
            "embedding_count": 1,
            "chunks_processed": 1,
            "kwargs_seen": kwargs,
        }

    def fail_if_jobs_called(self, **kwargs):  # noqa: ANN001
        calls["jobs"] = int(calls["jobs"]) + 1
        raise AssertionError(f"jobs dispatch should not run when form overrides env: {kwargs}")

    monkeypatch.setattr(
        jobs_adapter_module.EmbeddingsJobsAdapter,
        "create_job",
        fail_if_jobs_called,
    )
    monkeypatch.setattr(
        core_media_embeddings,
        "get_media_content",
        fake_get_media_content,
    )
    monkeypatch.setattr(
        core_media_embeddings,
        "generate_embeddings_for_media",
        fake_generate_embeddings_for_media,
    )

    title = f"Embeddings Dispatch Override {uuid4().hex}"
    source_path = test_media_dir / "dispatch_override.txt"
    source_path.write_text(
        "Form override dispatch content for media add embeddings.",
        encoding="utf-8",
    )

    with source_path.open("rb") as handle:
        response = test_client.post(
            "/api/v1/media/add",
            data={
                "media_type": "document",
                "title": title,
                "generate_embeddings": "true",
                "embedding_dispatch_mode": "background",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_provider": "huggingface",
                "chunk_method": "words",
                "chunk_size": "128",
                "chunk_overlap": "16",
            },
            files=[("files", (source_path.name, handle, "text/plain"))],
            headers=auth_headers,
        )

    assert response.status_code in (status.HTTP_200_OK, status.HTTP_207_MULTI_STATUS), response.text
    payload = response.json()
    success_row = next(
        (
            row
            for row in payload.get("results", [])
            if row.get("status") in {"Success", "Warning"} and row.get("db_id") is not None
        ),
        None,
    )
    assert success_row is not None, payload
    assert success_row.get("embeddings_scheduled") is True
    assert success_row.get("embeddings_dispatch") == "background"
    assert "embeddings_job_id" not in success_row
    assert int(calls["jobs"]) == 0
    assert int(calls["background"]) >= 1
    assert isinstance(success_row.get("embeddings_provenance"), dict)
    assert success_row["embeddings_provenance"]["origin"] == "media_add"
    assert success_row["embeddings_provenance"]["source_id"] == int(success_row["db_id"])

    detail_response = test_client.get(
        f"/api/v1/media/{success_row['db_id']}",
        params={
            "include_content": "true",
            "include_versions": "false",
            "include_version_content": "false",
        },
        headers=auth_headers,
    )
    assert detail_response.status_code == status.HTTP_200_OK, detail_response.text
    detail_payload = detail_response.json()
    assert detail_payload["processing"]["vector_processing_status"] == 1
    background_kwargs = calls.get("kwargs_seen") or {}
    assert background_kwargs.get("user_id") == "1"
