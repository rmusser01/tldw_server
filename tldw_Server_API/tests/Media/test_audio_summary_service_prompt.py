"""Audio prompt contracts through real multipart, storage, batch and analysis code."""

import threading
import wave
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import get_process_audios_form
from tldw_Server_API.app.api.v1.endpoints.media import process_audios as endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessAudiosForm
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase, ServicePromptOverrideRow
from tldw_Server_API.app.core.exceptions import ServicePromptCorruptOverride
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary
from tldw_Server_API.app.core.Prompt_Management.service_prompts import resolve_service_prompt
from tldw_Server_API.app.core.Utils import prompt_loader

pytestmark = pytest.mark.integration
PROMPT_ID = "media.audio.analysis"


@pytest.mark.parametrize("value", [None, "", "Literal {instructions}"])
def test_audio_form_preserves_each_prompt_presence(value: str | None) -> None:
    """Missing and explicitly empty multipart parts must not collapse together."""
    app = FastAPI()

    @app.post("/parse")
    async def parse(form: ProcessAudiosForm = Depends(get_process_audios_form)) -> dict[str, str | None]:
        """Expose the validated multipart prompt values without endpoint repair."""
        return {"system": form.system_prompt, "user": form.custom_prompt}

    fields = {"api_name": (None, "openai")}
    if value is not None:
        fields.update(system_prompt=(None, value), custom_prompt=(None, value))
    with TestClient(app) as client:
        response = client.post("/parse", files=fields)
    assert response.status_code == 200
    assert response.json() == {"system": value, "user": value}


@pytest.fixture
def context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_with_single_user: tuple[TestClient, object],
) -> Iterator[SimpleNamespace]:
    """Retain real prompt storage, audio processing and analyzer assembly."""
    client, _ = client_with_single_user
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    state = SimpleNamespace(
        client=client,
        owner=1,
        calls=[],
        reads=[],
        transcriptions=[],
        transcript="Send the report on Friday.",
        wav=buffer.getvalue(),
        databases={owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "audio-prompt-test") for owner in (1, 2)},
    )

    async def current_user() -> User:
        """Select the authenticated owner for each test request."""
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def get_db(request: Request, user: User) -> PromptsDatabase:
        """Record owner lookups while returning real prompt databases."""
        state.reads.append(user.id)
        return state.databases[user.id]

    def transcribe(**kwargs: Any) -> list[dict[str, Any]]:
        """Replace external speech recognition with a deterministic transcript."""
        state.transcriptions.append(kwargs)
        return [{"start_seconds": 0, "end_seconds": 1, "Text": state.transcript}]

    def adapter(**kwargs: Any) -> str:
        """Capture the real analyzer's model request without network access."""
        state.calls.append(kwargs)
        return "Report due Friday."

    monkeypatch.setitem(client.app.dependency_overrides, get_request_user, current_user)
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", get_db, raising=False)
    monkeypatch.setattr(audio, "speech_to_text", transcribe)
    monkeypatch.setattr(audio, "check_transcription_model_status", lambda _: {"available": True, "usable": True})
    monkeypatch.setattr(summary, "_summarize_via_adapter", adapter)
    yield state
    for database in state.databases.values():
        database.close_connection()


def save(
    context: SimpleNamespace,
    system: str = "Saved system {literal}",
    user: str = "Saved user {literal}",
    *,
    owner: int = 1,
) -> ServicePromptOverrideRow:
    """Store a full pair without bypassing real database serialization."""
    return context.databases[owner].save_service_prompt_override(PROMPT_ID, {"system": system, "user": user}, None)


def process(context: SimpleNamespace, *, count: int = 1, **options: str) -> dict[str, Any]:
    """Submit real WAV uploads through the authenticated route."""
    response = context.client.post(
        "/api/v1/media/process-audios",
        files=[("files", (f"source-{index}.wav", context.wav, "audio/wav")) for index in range(count)],
        data={"perform_analysis": "true", "perform_chunking": "false", "api_name": "openai", **options},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["errors"], body
    return body


def test_audio_pair_is_owner_scoped(context: SimpleNamespace) -> None:
    """Neither another owner's pair nor email guidance may reach the audio model."""
    save(context)
    save(context, "Owner two system", "Owner two user", owner=2)
    context.databases[1].save_service_prompt_override("media.email.summarization", {"system": "Not audio"}, None)
    result = process(context)
    context.owner = 2
    process(context)
    assert result["results"][0]["analysis"] == "Report due Friday."
    assert [(c["system_message"], c["custom_prompt_arg"]) for c in context.calls] == [
        ("Saved system {literal}", "Saved user {literal}"),
        ("Owner two system", "Owner two user"),
    ]


@pytest.mark.parametrize("field,other", [("system_prompt", "custom_prompt_arg"), ("custom_prompt", "system_message")])
@pytest.mark.parametrize("value", ["Explicit {literal}", ""])
def test_explicit_part_wins_independently(context: SimpleNamespace, field: str, other: str, value: str) -> None:
    """A request can replace one part without discarding the saved companion part."""
    save(context)
    process(context, **{field: value})
    key = "system_message" if field == "system_prompt" else "custom_prompt_arg"
    assert context.calls[0][key] == value
    assert context.calls[0][other] == ("Saved user {literal}" if field == "system_prompt" else "Saved system {literal}")


@pytest.mark.parametrize("value", ["Explicit", ""])
def test_both_explicit_parts_bypass_owner_storage(context: SimpleNamespace, value: str) -> None:
    """Even a corrupt stored pair must not interfere with a fully explicit request."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"wrong": "corrupt"}, None)
    process(context, system_prompt=value, custom_prompt=value)
    assert [(c["system_message"], c["custom_prompt_arg"]) for c in context.calls] == [(value, value)]
    assert context.reads == []


@pytest.mark.parametrize("options", [{"perform_analysis": "false"}, {"api_name": ""}, {"api_name": "none"}])
def test_inactive_analysis_does_not_read_prompts(context: SimpleNamespace, options: dict[str, str]) -> None:
    """Transcription-only requests cannot fail on irrelevant corrupt overrides."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"wrong": "corrupt"}, None)
    process(context, **options)
    assert context.calls == []
    assert context.reads == []
    assert len(context.transcriptions) == 1


@pytest.mark.parametrize("legacy", ["", "anthropic"])
def test_canonical_audio_provider_takes_precedence(context: SimpleNamespace, legacy: str) -> None:
    """Canonical provider selection wins over an absent or conflicting legacy alias."""
    save(context)
    process(context, api_provider="openai", api_name=legacy)
    assert context.calls[0]["api_name"] == "openai"
    assert context.calls[0]["system_message"] == "Saved system {literal}"


def test_pair_is_frozen_across_files_and_recursive_passes(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing both saved parts mid-request must affect only the next request."""
    context.transcript = "The report describes astronomy and biology in detail. " * 400
    row = save(context)
    original = summary._summarize_via_adapter

    def edit_during_analysis(**kwargs: Any) -> str:
        """Change future settings and owner scope during the first model call."""
        if not context.calls:
            database = context.databases[1]
            try:
                database.save_service_prompt_override(
                    PROMPT_ID, {"system": "Future system", "user": "Future user"}, row.revision
                )
            finally:
                database.close_connection()
            context.owner = 2
        return original(**kwargs)

    monkeypatch.setattr(summary, "_summarize_via_adapter", edit_during_analysis)
    process(context, count=2, summarize_recursively="true")
    assert len(context.calls) > 2
    assert {(c["system_message"], c["custom_prompt_arg"]) for c in context.calls} == {
        ("Saved system {literal}", "Saved user {literal}")
    }
    assert context.reads == [1]
    context.owner = 1
    process(context)
    assert (context.calls[-1]["system_message"], context.calls[-1]["custom_prompt_arg"]) == (
        "Future system",
        "Future user",
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_lookup_closes_its_worker_connection_before_processing(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    """Successful and corrupt lookups both close their actual worker connection."""
    database = context.databases[1]
    database.save_service_prompt_override(
        PROMPT_ID, {"wrong": "corrupt"} if corrupt else {"system": "S", "user": "U"}, None
    )
    reads, closes = [], []
    original_read, original_close = database.get_service_prompt_override, database.close_connection

    def read(definition_id: str) -> ServicePromptOverrideRow | None:
        """Observe the worker that opens the real lookup connection."""
        reads.append(threading.get_ident())
        return original_read(definition_id)

    def close() -> None:
        """Release the real connection and record its cleanup worker."""
        original_close()
        closes.append(threading.get_ident())

    monkeypatch.setattr(database, "get_service_prompt_override", read)
    monkeypatch.setattr(database, "close_connection", close)
    if corrupt:
        with pytest.raises(ServicePromptCorruptOverride):
            process(context)
        assert context.transcriptions == []
    else:
        process(context)
    assert len(reads) == 1
    assert closes == reads
    assert reads[0] != threading.get_ident()


def test_deployment_pair_is_shown_and_used_after_reset(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve actual deployment files for both Settings and runtime, not registry constants."""
    path = tmp_path / "audio.prompts.yaml"
    path.write_text("system_prompt: Deployment system\ntranscription_analysis_summary: Deployment user\n")
    monkeypatch.setattr(prompt_loader, "_prompts_dir", lambda: str(tmp_path))
    # This fixture file has no deployment approval ledger; exercise file parsing,
    # not the separate Context Integrity approval workflow.
    monkeypatch.setattr(prompt_loader, "get_global_context_integrity_resolver", lambda: None)
    row = save(context)
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)
    assert dict(resolve_service_prompt(context.databases[1], PROMPT_ID).parts) == {
        "system": "Deployment system",
        "user": "Deployment user",
    }
    process(context, count=2)
    assert [(c["system_message"], c["custom_prompt_arg"]) for c in context.calls] == [
        ("Deployment system", "Deployment user")
    ] * 2


def test_missing_audio_files_keep_legacy_analyzer_fallback(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent audio assets must not create a new default user instruction."""
    monkeypatch.setattr(prompt_loader, "_prompts_dir", lambda: str(tmp_path))
    monkeypatch.setattr(summary, "_resolve_default_system_prompt", lambda: "Shared analyzer fallback")
    process(context)
    assert (context.calls[0]["system_message"], context.calls[0]["custom_prompt_arg"]) == (
        "Shared analyzer fallback",
        "",
    )


def test_direct_core_call_retains_legacy_empty_prompt_defaults(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The opt-in resolved marker must not change existing direct/background callers."""
    (tmp_path / "audio.prompts.yaml").write_text(
        "system_prompt: Legacy system\ntranscription_analysis_summary: Legacy user\n"
    )
    monkeypatch.setattr(prompt_loader, "_prompts_dir", lambda: str(tmp_path))
    monkeypatch.setattr(prompt_loader, "get_global_context_integrity_resolver", lambda: None)
    wav_path = tmp_path / "direct.wav"
    wav_path.write_bytes(context.wav)
    result = audio.process_audio_files(
        inputs=[str(wav_path)],
        transcription_model="base",
        perform_chunking=False,
        api_name="openai",
        custom_prompt_input="",
        system_prompt_input="",
    )
    assert result["results"][0]["analysis"] == "Report due Friday."
    assert (context.calls[0]["system_message"], context.calls[0]["custom_prompt_arg"]) == (
        "Legacy system",
        "Legacy user",
    )
    assert context.reads == []


def test_model_download_message_keeps_resolved_pair(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing model-status analysis path must not bypass owner instructions."""
    save(context)
    monkeypatch.setattr(
        audio, "speech_to_text", lambda **_: [{"status": "model_downloading", "message": "Model is downloading."}]
    )
    process(context)
    assert context.calls
    assert {(c["system_message"], c["custom_prompt_arg"]) for c in context.calls} == {
        ("Saved system {literal}", "Saved user {literal}")
    }
