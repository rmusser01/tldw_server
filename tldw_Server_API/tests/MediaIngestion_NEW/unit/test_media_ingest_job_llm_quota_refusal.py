"""A media ingest job whose owner is over an LLM quota keeps its content and records the refusal (TASK-13864)."""

import asyncio
import wave
from pathlib import Path
from typing import Any

import pytest

import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary
from tldw_Server_API.app.core.Usage import llm_accounting, quota_resolver
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context
from tldw_Server_API.app.core.Usage.llm_quota_gate import user_to_check
from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

pytestmark = pytest.mark.integration

_REFUSAL_MESSAGE = "Analysis skipped: LLM quota exceeded (llm_tokens_month)."


def _write_document(tmp_path: Path, name: str) -> Path:
    source = tmp_path / f"{name}.md"
    source.write_text(f"Project {name} opens on 24 January 2027.", encoding="utf-8")
    return source


def _write_wav(tmp_path: Path) -> Path:
    source = tmp_path / "standup.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    return source


def _create_job(manager: JobManager, tmp_path: Path, source: Path, media_type: str) -> dict[str, Any]:
    return manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        owner_user_id="1",
        payload={
            "batch_id": "quota-refusal",
            "temp_dir": str(tmp_path),
            "media_type": media_type,
            "source": str(source),
            "source_kind": "file",
            "input_ref": source.name,
            "options": {
                "media_type": media_type,
                "perform_analysis": True,
                "api_name": "openai",
                "perform_chunking": False,
                "generate_embeddings": False,
            },
        },
    )


async def _run_jobs(manager: JobManager, count: int) -> None:
    sdk = WorkerSDK(manager, WorkerConfig(domain="media_ingest", queue="default", worker_id="quota-refusal"))
    handled: list[int] = []

    async def handle(job: dict[str, Any]) -> dict[str, Any]:
        try:
            return await worker._handle_job(job, manager, worker._ProgressState())
        finally:
            handled.append(int(job["id"]))
            if len(handled) == count:
                sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handle), timeout=30)


@pytest.fixture
def job_dbs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The jobs DB and the owner's media DB under tmp_path."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setattr(worker.DatabasePaths, "get_media_db_path", lambda _: tmp_path / "media.db")
    return tmp_path


@pytest.fixture
def refused(job_dbs: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every LLM quota check refuses, and so does the provider call; returns the provider calls made."""
    provider_calls: list[dict[str, Any]] = []

    def _refused_call(**kwargs: Any) -> str:
        """What the check at the call does on dev: refuse before the provider is reached."""
        provider_calls.append(kwargs)
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
        )

    monkeypatch.setattr(summary, "_summarize_via_adapter", _refused_call)
    refuse_llm_calls(monkeypatch)
    return provider_calls


@pytest.fixture
async def under_the_limit(job_dbs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, with a monthly token limit the owner is under: every check admits."""

    async def _user_quota(_uid: object, key: str) -> int | None:
        return 1000 if key == "limits.llm_tokens_per_month" else None

    async def _no_tokens(_uid: object) -> float:
        return 0.0

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.user_quota", _user_quota)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.quota_checks.llm_tokens_this_month", _no_tokens)
    monkeypatch.setattr("tldw_Server_API.app.core.Usage.llm_quota_gate.usage_quotas_enabled", lambda: True)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    quota_resolver.invalidate_all()


@pytest.fixture
def transcribed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speech recognition returns a fixed transcript."""
    monkeypatch.setattr(
        audio,
        "speech_to_text",
        lambda **_kwargs: [{"start_seconds": 0, "end_seconds": 1, "Text": "Send the report on Friday."}],
    )
    monkeypatch.setattr(audio, "check_transcription_model_status", lambda _: {"available": True, "usable": True})


def _assert_kept_with_refusal(job: dict[str, Any], tmp_path: Path, expected_text: str) -> None:
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["media_id"] is not None, (result["status"], result["error"])
    assert result["status"] == "Warning"
    assert result["analysis_error"] == {"code": "llm_quota_exceeded", "message": _REFUSAL_MESSAGE}
    assert _REFUSAL_MESSAGE in result["warnings"]
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="quota-refusal-verifier")
    try:
        assert expected_text in db.get_media_by_id(result["media_id"])["content"]
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_a_refused_document_batch_keeps_every_document(job_dbs: Path, refused: list[dict[str, Any]]) -> None:
    """Each job of a batch is refused at its start: every document is stored, and no LLM call is made."""
    manager = JobManager()
    names = ("Cedar", "Birch")
    rows = [_create_job(manager, job_dbs, _write_document(job_dbs, name), "document") for name in names]

    await _run_jobs(manager, len(rows))

    for row, name in zip(rows, names):
        _assert_kept_with_refusal(manager.get_job(int(row["id"])), job_dbs, f"Project {name} opens")
    assert refused == []


@pytest.mark.asyncio
async def test_a_refused_audio_job_keeps_its_transcript(
    job_dbs: Path, refused: list[dict[str, Any]], transcribed: None
) -> None:
    """An audio job is transcribed and stored without analysis."""
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_wav(job_dbs), "audio")

    await _run_jobs(manager, 1)

    _assert_kept_with_refusal(manager.get_job(int(row["id"])), job_dbs, "Send the report on Friday.")
    assert refused == []


@pytest.mark.asyncio
async def test_an_admitted_job_is_not_checked_again_part_way(
    job_dbs: Path, under_the_limit: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job admitted at its start runs as an admitted request does: its LLM calls are not checked again."""
    checked_at_the_call: list[int | None] = []

    def _analysis(**_kwargs: Any) -> str:
        checked_at_the_call.append(user_to_check())
        return "Cedar opens in January."

    monkeypatch.setattr(summary, "_summarize_via_adapter", _analysis)
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document")

    await _run_jobs(manager, 1)

    result = manager.get_job(int(row["id"]))["result"]
    assert result["status"] == "Success", result
    assert "analysis_error" not in result
    assert checked_at_the_call == [None]


@pytest.mark.asyncio
async def test_the_admission_ends_with_the_job(
    job_dbs: Path, under_the_limit: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job admitted at its start leaves no admission behind for whatever its caller runs next."""
    monkeypatch.setattr(summary, "_summarize_via_adapter", lambda **_kwargs: "Cedar opens in January.")
    manager = JobManager()
    row = _create_job(manager, job_dbs, _write_document(job_dbs, "Cedar"), "document")

    result = await worker._handle_job(manager.get_job(int(row["id"])), manager, worker._ProgressState())

    assert result["status"] == "Success", result
    assert current_llm_call_context().prechecked is False
