"""Video prompt boundaries through real multipart, storage, batch and analysis."""

import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import get_process_videos_form
from tldw_Server_API.app.api.v1.endpoints.media import process_videos as endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessVideosForm
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase, ServicePromptOverrideRow
from tldw_Server_API.app.core.exceptions import ServicePromptCorruptOverride
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import Video_DL_Ingestion_Lib as video
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary
from tldw_Server_API.app.core.Prompt_Management.service_prompts import resolve_service_prompt

pytestmark = pytest.mark.integration
PROMPT_ID = "media.video.summarization"


@pytest.mark.parametrize("value", [None, "", "Literal {instructions}"])
def test_video_form_preserves_prompt_presence(value: str | None) -> None:
    """Explicit empty prompt parts must remain distinct from omitted fields."""
    app = FastAPI()

    @app.post("/parse")
    async def parse(form: ProcessVideosForm = Depends(get_process_videos_form)) -> dict[str, str | None]:
        """Return the actual parsed prompts for the multipart contract."""
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_with_single_user: tuple[TestClient, object]
) -> Iterator[SimpleNamespace]:
    """Keep real uploaded-video processing while replacing external STT and LLM."""
    client, _ = client_with_single_user
    state = SimpleNamespace(
        client=client,
        owner=1,
        calls=[],
        reads=[],
        transcriptions=[],
        databases={owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "video-test") for owner in (1, 2)},
    )

    async def current_user() -> User:
        """Select the authenticated owner at request start."""
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def get_db(request: Request, user: User) -> PromptsDatabase:
        """Return real owner storage while recording lookup scope."""
        state.reads.append(user.id)
        return state.databases[user.id]

    def transcribe(**kwargs: Any) -> tuple[None, list[dict[str, Any]]]:
        """Provide a deterministic transcript instead of running speech recognition."""
        state.transcriptions.append(kwargs)
        return None, [{"Text": "Alpha report. Beta report.", "start": 0, "end": 1}]

    def adapter(**kwargs: Any) -> str:
        """Record model-facing requests produced by the real analyzer."""
        state.calls.append(kwargs)
        return "Report summary."

    monkeypatch.setitem(client.app.dependency_overrides, get_request_user, current_user)
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", get_db, raising=False)
    monkeypatch.setattr(video, "perform_transcription", transcribe)
    monkeypatch.setattr(summary, "_summarize_via_adapter", adapter)
    yield state
    for database in state.databases.values():
        database.close_connection()


def save(context: SimpleNamespace, *, owner: int = 1) -> ServicePromptOverrideRow:
    """Store a full literal pair through real database serialization."""
    return context.databases[owner].save_service_prompt_override(
        PROMPT_ID,
        {"system": f"Owner {owner} system {{literal}}", "final_summary": f"Owner {owner} final {{literal}}"},
        None,
    )


def process(context: SimpleNamespace, *, count: int = 1, **options: str) -> dict[str, Any]:
    """Send an authenticated multipart request through the actual route."""
    mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 2048
    fields = [("files", (f"source-{i}.mp4", mp4, "video/mp4")) for i in range(count)]
    fields += [
        (key, (None, value))
        for key, value in {
            "api_name": "openai",
            "perform_chunking": "true",
            "chunk_method": "words",
            "chunk_size": "2",
            "chunk_overlap": "0",
            "summarize_recursively": "true",
            "timestamp_option": "false",
            **options,
        }.items()
    ]
    response = context.client.post("/api/v1/media/process-videos", files=fields)
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["errors"], body
    return body


def test_final_instruction_is_stage_specific_and_owner_scoped(context: SimpleNamespace) -> None:
    """Saved synthesis text must not leak into chunk calls or across owners."""
    save(context)
    save(context, owner=2)
    for owner in (1, 2):
        context.owner = owner
        context.calls.clear()
        process(context)
        assert len(context.calls) == 3
        assert [call["custom_prompt_arg"] for call in context.calls] == [None, None, f"Owner {owner} final {{literal}}"]
        assert {call["system_message"] for call in context.calls} == {f"Owner {owner} system {{literal}}"}


@pytest.mark.parametrize("legacy", ["", "anthropic"])
def test_canonical_provider_uses_owner_video_prompts(context: SimpleNamespace, legacy: str) -> None:
    """The canonical provider must not skip saved guidance or use the legacy provider."""
    save(context)
    process(context, api_provider="openai", api_name=legacy)
    assert len(context.calls) == 3
    assert {call["api_name"] for call in context.calls} == {"openai"}
    assert context.calls[-1]["custom_prompt_arg"] == "Owner 1 final {literal}"


@pytest.mark.parametrize("value", ["Explicit {literal}", ""])
def test_explicit_user_fans_out_but_keeps_saved_system(context: SimpleNamespace, value: str) -> None:
    """An explicit user prompt replaces chunk and final instructions independently."""
    save(context)
    process(context, custom_prompt=value)
    assert [call["custom_prompt_arg"] for call in context.calls] == [value] * 3
    assert {call["system_message"] for call in context.calls} == {"Owner 1 system {literal}"}


@pytest.mark.parametrize("value", ["Explicit", ""])
def test_explicit_system_keeps_saved_final(context: SimpleNamespace, value: str) -> None:
    """Replacing system instructions must not discard saved synthesis guidance."""
    save(context)
    process(context, system_prompt=value)
    assert {call["system_message"] for call in context.calls} == {value}
    assert context.calls[-1]["custom_prompt_arg"] == "Owner 1 final {literal}"


@pytest.mark.parametrize(
    "options",
    [
        {"perform_analysis": "false"},
        {"api_name": "none"},
        {"api_name": ""},
        {"system_prompt": "", "custom_prompt": ""},
        {"system_prompt": "Explicit", "perform_chunking": "false"},
        {"system_prompt": "Explicit", "summarize_recursively": "false"},
    ],
)
def test_irrelevant_storage_is_not_read(context: SimpleNamespace, options: dict[str, str]) -> None:
    """Corrupt saved prompts cannot affect requests that do not need their parts."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"bad": "corrupt"}, None)
    process(context, **options)
    assert context.reads == []


def test_snapshot_survives_edits_and_owner_switch(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """All files and passes must use the configuration captured before processing."""
    row = save(context)
    original = summary._summarize_via_adapter

    def edit(**kwargs: Any) -> str:
        """Change saved settings and future request identity during the first call."""
        if not context.calls:
            database = context.databases[1]
            try:
                database.save_service_prompt_override(
                    PROMPT_ID, {"system": "Future", "final_summary": "Future"}, row.revision
                )
            finally:
                database.close_connection()
            context.owner = 2
        return original(**kwargs)

    monkeypatch.setattr(summary, "_summarize_via_adapter", edit)
    process(context, count=2)
    assert context.reads == [1]
    assert [call["custom_prompt_arg"] for call in context.calls] == [None, None, "Owner 1 final {literal}"] * 2
    assert {call["system_message"] for call in context.calls} == {"Owner 1 system {literal}"}


@pytest.mark.parametrize("corrupt", [False, True])
def test_lookup_closes_worker_connection(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    """Cleanup must run on the lookup thread on success and corrupt-data failure."""
    database = context.databases[1]
    if corrupt:
        database.save_service_prompt_override(PROMPT_ID, {"bad": "corrupt"}, None)
    else:
        save(context)
    reads, closes = [], []
    original_read, original_close = database.get_service_prompt_override, database.close_connection

    def read(definition_id: str) -> ServicePromptOverrideRow | None:
        """Record the worker opening the real read connection."""
        reads.append(threading.get_ident())
        return original_read(definition_id)

    def close() -> None:
        """Close the real connection and record its worker."""
        original_close()
        closes.append(threading.get_ident())

    monkeypatch.setattr(database, "get_service_prompt_override", read)
    monkeypatch.setattr(database, "close_connection", close)
    if corrupt:
        with pytest.raises(ServicePromptCorruptOverride):
            process(context)
        assert not context.transcriptions
    else:
        process(context)
    assert len(reads) == 1
    assert closes == reads


def test_reset_uses_deployed_system_and_legacy_final(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings defaults and model calls agree without adding initial user guidance."""
    monkeypatch.setattr(summary, "_resolve_default_system_prompt", lambda: "Deployment system")
    row = save(context)
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)
    assert dict(resolve_service_prompt(context.databases[1], PROMPT_ID).parts) == {
        "system": "Deployment system",
        "final_summary": "Summarize the key points from the preceding text sections.",
    }
    process(context)
    assert {call["system_message"] for call in context.calls} == {"Deployment system"}
    assert [call["custom_prompt_arg"] for call in context.calls] == [
        None,
        None,
        "Summarize the key points from the preceding text sections.",
    ]


@pytest.mark.parametrize("mode", ["unchunked", "nonrecursive", "empty_chunks", "failed_chunks", "single_chunk"])
def test_final_guidance_never_becomes_initial_guidance(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Nonrecursive and chunking fallback paths use only the saved system part."""
    save(context)
    options = {}
    if mode == "unchunked":
        options["perform_chunking"] = "false"
    elif mode == "nonrecursive":
        options["summarize_recursively"] = "false"
    elif mode == "failed_chunks":

        def fail(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
            """Exercise the existing full-text fallback after a chunking failure."""
            raise ValueError("chunking unavailable")

        monkeypatch.setattr(video, "improved_chunking_process", fail)
    else:
        chunks = [] if mode == "empty_chunks" else [{"text": "One complete report."}]
        monkeypatch.setattr(video, "improved_chunking_process", lambda *_: chunks)
    process(context, **options)
    assert context.calls
    assert {call["custom_prompt_arg"] for call in context.calls} == {None}
    assert {call["system_message"] for call in context.calls} == {"Owner 1 system {literal}"}


@pytest.mark.parametrize("custom", [None, "", "Direct custom"])
def test_direct_video_call_preserves_legacy_synthesis_defaults(
    context: SimpleNamespace, tmp_path: Path, custom: str | None
) -> None:
    """Callers outside the synchronous Settings adapter retain old fallback semantics."""
    path = tmp_path / "direct.mp4"
    path.write_bytes(b"local media fixture")
    result = video.process_single_video(
        video_input=str(path),
        start_seconds=0,
        end_seconds=None,
        diarize=False,
        vad_use=False,
        transcription_model="base",
        transcription_language="en",
        perform_analysis=True,
        custom_prompt=custom,
        system_prompt="Direct system",
        perform_chunking=True,
        chunk_method="words",
        max_chunk_size=2,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language="en",
        summarize_recursively=True,
        api_name="openai",
        use_cookies=False,
        cookies=None,
        timestamp_option=False,
        temp_dir=str(tmp_path),
    )
    assert result["analysis"] == "Report summary."
    assert [call["custom_prompt_arg"] for call in context.calls] == [
        custom,
        custom,
        custom or "Summarize the key points from the preceding text sections.",
    ]
    assert context.reads == []
