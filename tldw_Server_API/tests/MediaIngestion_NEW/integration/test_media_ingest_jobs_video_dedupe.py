from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def _set_jobs_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "false")


def _allow_url_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        raising=True,
    )


@pytest.mark.asyncio
async def test_media_ingest_job_completion_exposes_reserved_video_lite_summary_metadata(
    auth_headers,
    media_database,
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.main import app
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    _set_jobs_db(monkeypatch, tmp_path)
    _allow_url_policy(monkeypatch)
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")

    source_url = "https://www.youtube.com/watch?v=summary123"
    processor_calls: list[str] = []

    async def _fake_process_videos(**kwargs):
        processor_calls.extend(kwargs.get("inputs") or [])
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": source_url,
                    "processing_source": source_url,
                    "media_type": "video",
                    "transcript": "Transcript content for the test video.",
                    "segments": [],
                    "summary": "A reserved summary for the test video.",
                    "analysis": "Generic analysis should not be used by lite.",
                    "analysis_details": {"transcription_language": "en"},
                    "metadata": {
                        "title": "Reserved Summary Video",
                        "author": "Test Author",
                        "source_url": source_url,
                        "url": source_url,
                        "model": "whisper-test",
                        "provider": "whisper",
                        "source": "youtube",
                    },
                    "warnings": None,
                }
            ],
            "errors_count": 0,
            "errors": [],
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_videos",
        _fake_process_videos,
        raising=True,
    )

    def _override_media_db():
        return media_database

    def _worker_db_factory(_user_id: str):
        db = MediaDatabase(db_path=media_database.db_path_str, client_id="1")
        db.initialize_db()
        return db

    with TestClient(app) as test_client:
        app.dependency_overrides[get_media_db_for_user] = _override_media_db
        monkeypatch.setattr(worker, "_create_db", _worker_db_factory, raising=True)
        try:
            submit_resp = test_client.post(
                "/api/v1/media/ingest/jobs",
                data={
                    "media_type": "video",
                    "urls": source_url,
                    "transcription_model": "whisper-test",
                },
                headers=auth_headers,
            )
            assert submit_resp.status_code == 200, submit_resp.text
            job_id = int(submit_resp.json()["jobs"][0]["id"])

            jm = JobManager()
            queued_job = jm.get_job(job_id)
            assert queued_job is not None
            acquired_job = jm.acquire_next_job(
                domain="media_ingest",
                queue=str(queued_job.get("queue") or "default"),
                lease_seconds=120,
                worker_id="test-worker",
            )
            assert acquired_job is not None

            result = await worker._handle_job(acquired_job, jm, worker._ProgressState())
            completed = jm.complete_job(
                job_id,
                result=result,
                worker_id="test-worker",
                lease_id=str(acquired_job.get("lease_id")),
                enforce=False,
            )
            assert completed is True

            status_resp = test_client.get(
                f"/api/v1/media/ingest/jobs/{job_id}",
                headers=auth_headers,
            )
            assert status_resp.status_code == 200, status_resp.text
            status_body = status_resp.json()
            assert status_body["status"] == "completed"
            assert status_body["result"]["media_id"]

            media_id = int(status_body["result"]["media_id"])
            detail_resp = test_client.get(f"/api/v1/media/{media_id}", headers=auth_headers)
            assert detail_resp.status_code == 200, detail_resp.text
            detail = detail_resp.json()
            assert detail["content"]["text"] == "Transcript content for the test video."
            assert detail["processing"]["safe_metadata"]["video_lite"]["summary"] == (
                "A reserved summary for the test video."
            )
            assert detail["processing"]["safe_metadata"]["video_lite"]["summary_status"] == "ready"
            assert processor_calls == [source_url]
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_media_ingest_job_second_submit_reuses_existing_media_id_without_reprocessing(
    auth_headers,
    media_database,
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.main import app
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    _set_jobs_db(monkeypatch, tmp_path)
    _allow_url_policy(monkeypatch)
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")

    source_url = "https://www.youtube.com/watch?v=dedupe123"
    processor_calls: list[str] = []

    async def _fake_process_videos(**kwargs):
        processor_calls.extend(kwargs.get("inputs") or [])
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": source_url,
                    "processing_source": source_url,
                    "media_type": "video",
                    "transcript": "Reusable transcript.",
                    "segments": [],
                    "summary": "Reusable summary.",
                    "analysis_details": {"transcription_language": "en"},
                    "metadata": {
                        "title": "Reusable Video",
                        "author": "Test Author",
                        "source_url": source_url,
                        "url": source_url,
                        "model": "whisper-test",
                        "provider": "whisper",
                    },
                    "warnings": None,
                }
            ],
            "errors_count": 0,
            "errors": [],
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_videos",
        _fake_process_videos,
        raising=True,
    )

    def _override_media_db():
        return media_database

    def _worker_db_factory(_user_id: str):
        db = MediaDatabase(db_path=media_database.db_path_str, client_id="1")
        db.initialize_db()
        return db

    with TestClient(app) as test_client:
        app.dependency_overrides[get_media_db_for_user] = _override_media_db
        monkeypatch.setattr(worker, "_create_db", _worker_db_factory, raising=True)
        try:
            job_manager = JobManager()
            media_ids: list[int] = []

            for _ in range(2):
                submit_resp = test_client.post(
                    "/api/v1/media/ingest/jobs",
                    data={
                        "media_type": "video",
                        "urls": source_url,
                        "transcription_model": "whisper-test",
                    },
                    headers=auth_headers,
                )
                assert submit_resp.status_code == 200, submit_resp.text
                job_id = int(submit_resp.json()["jobs"][0]["id"])

                queued_job = job_manager.get_job(job_id)
                assert queued_job is not None
                acquired_job = job_manager.acquire_next_job(
                    domain="media_ingest",
                    queue=str(queued_job.get("queue") or "default"),
                    lease_seconds=120,
                    worker_id=f"test-worker-{job_id}",
                )
                assert acquired_job is not None
                result = await worker._handle_job(acquired_job, job_manager, worker._ProgressState())
                completed = job_manager.complete_job(
                    job_id,
                    result=result,
                    worker_id=f"test-worker-{job_id}",
                    lease_id=str(acquired_job.get("lease_id")),
                    enforce=False,
                )
                assert completed is True
                media_ids.append(int(result["media_id"]))

            assert media_ids[0] == media_ids[1]
            assert processor_calls == [source_url]
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_media_ingest_cross_user_dedupe_reuses_transcript_without_reprocessing(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        create_media_database,
        get_full_media_details_rich,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    _set_jobs_db(monkeypatch, tmp_path)
    _allow_url_policy(monkeypatch)
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    await reset_db_pool()

    source_url = "https://www.youtube.com/watch?v=crossuser123"
    processor_calls: list[str] = []
    analyze_calls: list[str] = []

    async def _fake_process_videos(**kwargs):
        processor_calls.extend(kwargs.get("inputs") or [])
        call_no = len(processor_calls)
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": source_url,
                    "processing_source": source_url,
                    "media_type": "video",
                    "content": "Reusable transcript content.",
                    "segments": [],
                    "analysis": f"processor summary {call_no}",
                    "analysis_details": {"transcription_language": "en"},
                    "metadata": {
                        "title": "Reusable Cross-User Video",
                        "author": "Test Author",
                        "source_url": source_url,
                        "url": source_url,
                        "model": "whisper-test",
                        "provider": "whisper",
                        "source": "youtube",
                    },
                    "warnings": None,
                }
            ],
            "errors_count": 0,
            "errors": [],
        }

    def _fake_analyze(
        _api_name,
        input_data,
        _custom_prompt=None,
        _api_key=None,
        *,
        system_message=None,
    ):
        analyze_calls.append(f"{input_data}|{system_message}")
        return "generated fresh summary"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_videos",
        _fake_process_videos,
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze",
        _fake_analyze,
        raising=True,
    )

    job_manager = JobManager()
    captured_details: list[dict[str, object]] = []

    try:
        for user_id in ("1", "2"):
            row = job_manager.create_job(
                domain="media_ingest",
                queue="default",
                job_type="media_ingest_item",
                payload={
                    "batch_id": f"batch-cross-{user_id}",
                    "media_type": "video",
                    "source": source_url,
                    "source_kind": "url",
                    "input_ref": source_url,
                    "options": {
                        "media_type": "video",
                        "transcription_model": "whisper-test",
                        "perform_analysis": True,
                        "api_name": "openai",
                        "custom_prompt": "Summarize this transcript.",
                        "perform_chunking": False,
                    },
                },
                owner_user_id=user_id,
            )
            job = job_manager.get_job(int(row.get("id")))
            assert job is not None
            result = await worker._handle_job(job, job_manager, worker._ProgressState())
            assert result.get("media_id")

            db = create_media_database(
                client_id=f"test-user-{user_id}",
                db_path=str(DatabasePaths.get_media_db_path(user_id)),
            )
            try:
                details = get_full_media_details_rich(db, int(result["media_id"]))
                assert details is not None
                captured_details.append(details)
            finally:
                db.close_connection()

        assert processor_calls == [source_url]
        assert captured_details[0]["processing"]["safe_metadata"]["video_lite"]["summary"] == (
            "processor summary 1"
        )
        assert captured_details[1]["content"]["text"] == "Reusable transcript content."
        assert captured_details[1]["processing"]["safe_metadata"]["video_lite"]["summary"] == (
            "generated fresh summary"
        )
        assert analyze_calls == ["Reusable transcript content.|None"]
    finally:
        await reset_db_pool()


@pytest.mark.asyncio
async def test_media_ingest_cross_user_dedupe_prefers_transcript_artifact_over_media_content(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        create_media_database,
        get_full_media_details_rich,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    _set_jobs_db(monkeypatch, tmp_path)
    _allow_url_policy(monkeypatch)
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    await reset_db_pool()

    source_url = "https://www.youtube.com/watch?v=artifact-first"
    processor_calls: list[str] = []

    async def _fake_process_videos(**kwargs):
        processor_calls.extend(kwargs.get("inputs") or [])
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": source_url,
                    "processing_source": source_url,
                    "media_type": "video",
                    "content": "media row content that should not be reused",
                    "segments": [],
                    "summary": "processor summary 1",
                    "metadata": {
                        "title": "Artifact First Video",
                        "author": "Test Author",
                        "source_url": source_url,
                        "url": source_url,
                        "model": "whisper-test",
                        "provider": "whisper",
                        "source": "youtube",
                    },
                    "normalized_stt": {
                        "text": "canonical transcript artifact text",
                        "segments": [],
                        "metadata": {
                            "model": "whisper-test",
                            "provider": "whisper",
                            "language": "en",
                        },
                    },
                    "warnings": None,
                }
            ],
            "errors_count": 0,
            "errors": [],
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_videos",
        _fake_process_videos,
        raising=True,
    )

    job_manager = JobManager()
    captured_details: list[dict[str, object]] = []

    try:
        for user_id in ("1", "2"):
            row = job_manager.create_job(
                domain="media_ingest",
                queue="default",
                job_type="media_ingest_item",
                payload={
                    "batch_id": f"batch-artifact-{user_id}",
                    "media_type": "video",
                    "source": source_url,
                    "source_kind": "url",
                    "input_ref": source_url,
                    "options": {
                        "media_type": "video",
                        "transcription_model": "whisper-test",
                        "perform_analysis": False,
                        "perform_chunking": False,
                    },
                },
                owner_user_id=user_id,
            )
            job = job_manager.get_job(int(row.get("id")))
            assert job is not None
            result = await worker._handle_job(job, job_manager, worker._ProgressState())
            assert result.get("media_id")

            db = create_media_database(
                client_id=f"artifact-user-{user_id}",
                db_path=str(DatabasePaths.get_media_db_path(user_id)),
            )
            try:
                details = get_full_media_details_rich(db, int(result["media_id"]))
                assert details is not None
                captured_details.append(details)
            finally:
                db.close_connection()

        assert processor_calls == [source_url]
        assert captured_details[0]["content"]["text"] == "media row content that should not be reused"
        assert captured_details[1]["content"]["text"] == "canonical transcript artifact text"
    finally:
        await reset_db_pool()


@pytest.mark.asyncio
async def test_media_ingest_cross_user_dedupe_skips_reuse_when_clip_options_differ(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker

    _set_jobs_db(monkeypatch, tmp_path)
    _allow_url_policy(monkeypatch)
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    await reset_db_pool()

    source_url = "https://www.youtube.com/watch?v=clip-diff"
    processor_calls: list[str] = []

    async def _fake_process_videos(**kwargs):
        processor_calls.extend(kwargs.get("inputs") or [])
        call_no = len(processor_calls)
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": source_url,
                    "processing_source": source_url,
                    "media_type": "video",
                    "content": f"transcript {call_no}",
                    "segments": [],
                    "summary": f"summary {call_no}",
                    "metadata": {
                        "title": "Clip Differ Video",
                        "author": "Test Author",
                        "source_url": source_url,
                        "url": source_url,
                        "model": "whisper-test",
                        "provider": "whisper",
                        "source": "youtube",
                    },
                    "warnings": None,
                }
            ],
            "errors_count": 0,
            "errors": [],
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib.process_videos",
        _fake_process_videos,
        raising=True,
    )

    job_manager = JobManager()

    try:
        row1 = job_manager.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={
                "batch_id": "batch-clip-1",
                "media_type": "video",
                "source": source_url,
                "source_kind": "url",
                "input_ref": source_url,
                "options": {
                    "media_type": "video",
                    "transcription_model": "whisper-test",
                    "perform_analysis": False,
                    "perform_chunking": False,
                    "start_time": "0",
                    "end_time": "60",
                    "timestamp_option": False,
                },
            },
            owner_user_id="1",
        )
        job1 = job_manager.get_job(int(row1.get("id")))
        assert job1 is not None
        await worker._handle_job(job1, job_manager, worker._ProgressState())

        row2 = job_manager.create_job(
            domain="media_ingest",
            queue="default",
            job_type="media_ingest_item",
            payload={
                "batch_id": "batch-clip-2",
                "media_type": "video",
                "source": source_url,
                "source_kind": "url",
                "input_ref": source_url,
                "options": {
                    "media_type": "video",
                    "transcription_model": "whisper-test",
                    "perform_analysis": False,
                    "perform_chunking": False,
                    "start_time": "10",
                    "end_time": "60",
                    "timestamp_option": False,
                },
            },
            owner_user_id="2",
        )
        job2 = job_manager.get_job(int(row2.get("id")))
        assert job2 is not None
        await worker._handle_job(job2, job_manager, worker._ProgressState())

        assert processor_calls == [source_url, source_url]
    finally:
        await reset_db_pool()
