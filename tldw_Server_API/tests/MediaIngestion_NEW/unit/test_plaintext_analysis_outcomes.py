"""Requested analysis must not store legacy error strings as useful analysis."""
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext import Plaintext_Files as docs


@pytest.mark.unit
@pytest.mark.parametrize("analysis", ["Error: Model is required for provider 'llama.cpp'", ""])
def test_document_analysis_failure_keeps_source_and_reports_warning(tmp_path, monkeypatch, analysis):
    source = tmp_path / "cedar.md"
    source.write_text("Project Cedar launches on 22 November 2026.", encoding="utf-8")
    monkeypatch.setattr(docs, "analyze", lambda **kwargs: analysis)
    result = docs.process_document_content(
        source, False, None, True, False, "llama", None, None, None,
    )
    assert result["status"] == "Warning"
    assert result["analysis"] is None
    assert result["warnings"]
    assert "Project Cedar" in result["content"]
    assert result["chunks"][0]["metadata"].get("analysis") is None


@pytest.mark.unit
def test_document_analysis_success_is_retained(tmp_path, monkeypatch):
    source = tmp_path / "cedar.md"
    source.write_text("Project Cedar launches in November.", encoding="utf-8")
    monkeypatch.setattr(docs, "analyze", lambda **kwargs: "Cedar launches in November.")
    result = docs.process_document_content(
        source, False, None, True, False, "llama", None, None, None,
    )
    assert result["status"] == "Success"
    assert result["analysis"] == "Cedar launches in November."


@pytest.mark.unit
@pytest.mark.parametrize(
    "content,finish_reason",
    [
        ("", "stop"),
        ("Partial analysis", "length"),
        ("<think>private-provider-sentinel</think>", "stop"),
        ("<think>outer<think>inner</think>private-provider-sentinel</think>", "stop"),
        ("<think>outer<think>inner</think>private-provider-sentinel", "stop"),
        ("Public-looking answer<think>private-provider-sentinel</reason>", "stop"),
    ],
)
def test_real_summarizer_failure_preserves_plaintext_source(tmp_path, monkeypatch, content, finish_reason):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl

    response = {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": "private-provider-sentinel"},
                "finish_reason": finish_reason,
            }
        ]
    }
    adapter = SimpleNamespace(chat=lambda *_args, **_kwargs: response)
    monkeypatch.setattr(sgl, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _: adapter))
    monkeypatch.setattr(sgl, "load_and_log_configs", lambda: {"llama_api": {"model": "test-model"}})
    monkeypatch.setattr(sgl, "resolve_provider_api_key_from_config", lambda *_args: None)
    source = tmp_path / "cedar.md"
    source.write_text("Project Cedar opens on 24 January 2027.", encoding="utf-8")
    result = docs.process_document_content(source, False, None, True, False, "llama", None, None, None)
    assert result["status"] == "Warning"
    assert result["analysis"] is None
    assert result["warnings"]
    assert "Project Cedar opens on 24 January 2027." in result["content"]
    assert all(chunk["metadata"].get("analysis") is None for chunk in result["chunks"])
    assert "private-provider-sentinel" not in str(result)
    assert "choices" not in str(result)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason,expected_status", [("length", "Warning"), ("stop", "Success")])
async def test_real_analysis_job_persists_source_and_truthful_outcome(
    tmp_path, monkeypatch, finish_reason, expected_status
):
    """Exercise the job and persistence pipeline with isolated provider/config seams."""
    import asyncio
    import json
    from types import SimpleNamespace

    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.core.DB_Management.media_db.api import get_document_version
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setattr(worker.DatabasePaths, "get_media_db_path", lambda _: tmp_path / "media.db")
    response = {
        "choices": [
            {
                "message": {"content": "Cedar opens in January.", "reasoning_content": "private-provider-sentinel"},
                "finish_reason": finish_reason,
            }
        ]
    }
    adapter = SimpleNamespace(chat=lambda *_args, **_kwargs: response)
    monkeypatch.setattr(sgl, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _: adapter))
    monkeypatch.setattr(sgl, "load_and_log_configs", lambda: {"llama_api": {"model": "test-model"}})
    monkeypatch.setattr(sgl, "resolve_provider_api_key_from_config", lambda *_args: None)
    source = tmp_path / "cedar.md"
    source.write_text("Project Cedar opens on 24 January 2027.", encoding="utf-8")
    manager = JobManager()
    row = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        owner_user_id="1",
        payload={
            "batch_id": "analysis-outcome",
            "temp_dir": str(tmp_path),
            "media_type": "document",
            "source": str(source),
            "source_kind": "file",
            "input_ref": "cedar.md",
            "options": {
                "media_type": "document",
                "perform_analysis": True,
                "api_name": "llama",
                "perform_chunking": False,
                "generate_embeddings": False,
            },
        },
    )
    sdk = WorkerSDK(manager, WorkerConfig(domain="media_ingest", queue="default", worker_id="analysis-outcome"))

    async def handle(job):
        try:
            return await worker._handle_job(job, manager, worker._ProgressState())
        finally:
            sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handle), timeout=10)
    completed = manager.get_job(int(row["id"]))
    assert completed["status"] == "completed"
    result = completed["result"]
    assert result["status"] == expected_status
    assert result["media_id"] is not None
    assert bool(result["warnings"]) == (expected_status == "Warning")
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="analysis-outcome-verifier")
    try:
        media = db.get_media_by_id(result["media_id"])
        version = get_document_version(db, result["media_id"])
        assert "Project Cedar opens on 24 January 2027." in media["content"]
        assert version.get("analysis_content") == ("Cedar opens in January." if expected_status == "Success" else None)
        assert "private-provider-sentinel" not in json.dumps([result, media, version], default=str)
        assert "choices" not in json.dumps([result, media, version], default=str)
    finally:
        db.close_connection()
