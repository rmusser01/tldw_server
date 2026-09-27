"""Exercise document analysis through the live synchronous endpoint."""

import io
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
import yaml
from fastapi import Request, Response, UploadFile
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.media import process_documents as endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessDocumentsForm
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase, ServicePromptOverrideRow
from tldw_Server_API.app.core.Prompt_Management.service_prompts import get_service_prompt_definition

pytestmark = pytest.mark.unit
PROMPT_ID = "media.document.summarization"


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Provide real owner databases and capture requests at the model boundary."""
    databases = {owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "document-summary-test") for owner in (1, 2)}
    reads = []
    calls = []

    async def get_db(request: Request, user: User) -> PromptsDatabase:
        """Return the isolated database for the authenticated owner."""
        reads.append(user.id)
        return databases[user.id]

    def analyze(**kwargs: Any) -> str:
        """Record model inputs without invoking an external provider."""
        calls.append(kwargs)
        return "A summary of this section."

    # The database factory and model are the external boundaries; conversion,
    # batching, chunking, registry validation and SQLite storage remain real.
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", get_db, raising=False)
    monkeypatch.setattr(endpoint.docs, "analyze", analyze)
    yield SimpleNamespace(databases=databases, reads=reads, calls=calls)
    for database in databases.values():
        database.close_connection()


def save(context: SimpleNamespace, owner: int, system: str) -> ServicePromptOverrideRow:
    """Save a document system override in one owner's database."""
    return context.databases[owner].save_service_prompt_override(PROMPT_ID, {"system": system}, expected_revision=None)


async def process(*, owner: int = 1, file_count: int = 1, **options: Any) -> dict[str, Any]:
    """Run the live endpoint on deterministic uploaded text documents."""
    form = ProcessDocumentsForm(
        **{
            "perform_analysis": True,
            "perform_chunking": False,
            "api_name": "openai",
            **options,
        }
    )
    files = [
        UploadFile(
            filename=f"document-{index}.txt",
            file=io.BytesIO(
                b"First section explains astronomy. Second section explains biology. "
                b"Third section describes chemistry. Fourth section covers geology."
            ),
        )
        for index in range(file_count)
    ]

    def log_event(*args: Any, **kwargs: Any) -> None:
        """Ignore usage telemetry unrelated to prompt selection."""

    result = await endpoint.process_documents_endpoint(
        request=Request({"type": "http", "headers": []}),
        injected_response=Response(),
        current_user=User(id=owner, username=f"owner-{owner}"),
        db=None,
        form_data=form,
        files=files,
        usage_log=SimpleNamespace(log_event=log_event),
    )
    body = json.loads(result.body)
    assert result.status_code == 200, body
    assert len(body["results"]) == file_count
    return body


def test_document_prompt_exposes_literal_system_guidance() -> None:
    """Document instructions must preserve authored text without templating."""
    definition = get_service_prompt_definition(PROMPT_ID)
    assert [(part.key, part.mode) for part in definition.parts] == [("system", "literal")]


def test_settings_default_matches_packaged_analyzer_prompt() -> None:
    """The Settings preview must match the packaged analyzer instructions."""
    packaged_path = Path(__file__).resolve().parents[2] / "Config_Files/Prompts/summarization.prompts.yaml"
    packaged = yaml.safe_load(packaged_path.read_text())["summarization_system_prompt"].strip()
    assert get_service_prompt_definition(PROMPT_ID).default_parts["system"] == packaged


async def test_saved_prompt_is_owned_by_authenticated_user(context: SimpleNamespace) -> None:
    """Different authenticated owners must not share their saved guidance."""
    save(context, 1, "Use short sentences for owner one.")
    save(context, 2, "Write detailed French notes for owner two.")
    await process(owner=1)
    await process(owner=2)
    assert [call["system_message"] for call in context.calls] == [
        "Use short sentences for owner one.",
        "Write detailed French notes for owner two.",
    ]
    assert context.reads == [1, 2]


@pytest.mark.parametrize("system", ["Explicit request guidance", ""])
async def test_explicit_system_prompt_bypasses_storage(context: SimpleNamespace, system: str) -> None:
    """Explicit system text, including empty text, takes precedence over storage."""
    await process(system_prompt=system, custom_prompt="Focus on experiments.")
    assert context.calls[0]["system_message"] == system
    assert context.calls[0]["custom_prompt_arg"] == "Focus on experiments."
    assert context.reads == []


async def test_custom_user_instruction_supplements_saved_system(context: SimpleNamespace) -> None:
    """Request-specific user instructions supplement the saved system prompt."""
    save(context, 1, "Write accessible notes.")
    await process(custom_prompt="Focus on experiments.")
    assert context.calls[0]["system_message"] == "Write accessible notes."
    assert context.calls[0]["custom_prompt_arg"] == "Focus on experiments."


@pytest.mark.parametrize("options", [{"perform_analysis": False}, {"api_name": None}])
async def test_disabled_analysis_does_not_access_prompts(context: SimpleNamespace, options: dict[str, Any]) -> None:
    """Disabled analysis and missing providers must not touch prompt storage."""
    await process(**options)
    assert context.calls == []
    assert context.reads == []


async def test_one_snapshot_covers_all_documents_chunks_and_recursive_passes(context: SimpleNamespace) -> None:
    """Every model call in one batch must use the same resolved instructions."""
    save(context, 1, "Keep this request in Spanish.")
    await process(
        file_count=2,
        perform_chunking=True,
        chunk_method="words",
        chunk_size=8,
        chunk_overlap=0,
        summarize_recursively=True,
    )
    assert len(context.calls) > 4
    assert {call["system_message"] for call in context.calls} == {"Keep this request in Spanish."}
    recursive = [call for call in context.calls if "\n\n---\n\n" in call["input_data"]]
    assert len(recursive) == 2
    assert {call["custom_prompt_arg"] for call in recursive} == {
        "Provide a concise overall summary of the following text sections."
    }
    assert context.reads == [1]


async def test_no_override_snapshots_existing_server_default(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncustomized request must resolve the deployment default once."""
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary

    defaults_read = []

    def server_default() -> str:
        """Observe reads of a deployment-specific default."""
        defaults_read.append(True)
        return "Deployment-specific summarization instructions."

    monkeypatch.setattr(summary, "_resolve_default_system_prompt", server_default)
    await process(file_count=2)
    assert {call["system_message"] for call in context.calls} == {"Deployment-specific summarization instructions."}
    assert defaults_read == [True]


async def test_reset_restores_server_default(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resetting an override must restore deployment-configured instructions."""
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary

    row = save(context, 1, "My custom instructions")
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)

    def server_default() -> str:
        """Supply a server default distinct from the saved override."""
        return "Server instructions"

    monkeypatch.setattr(summary, "_resolve_default_system_prompt", server_default)
    await process()
    assert context.calls[0]["system_message"] == "Server instructions"


async def test_corrupt_override_stops_before_input_processing(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid stored parts must fail before uploads or model calls start."""
    from tldw_Server_API.app.core.exceptions import ServicePromptCorruptOverride

    # A prior version or damaged row can contain invalid parts; storage accepts
    # them, while the resolver must enforce the current definition contract.
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"unknown": "bad"}, None)

    async def unexpected_upload(*args: Any, **kwargs: Any) -> NoReturn:
        """Fail if input processing starts before prompt validation."""
        pytest.fail("Input processing started before prompt validation")

    monkeypatch.setattr(endpoint, "core_save_uploaded_files", unexpected_upload)
    with pytest.raises(ServicePromptCorruptOverride):
        await process()
    assert context.calls == []


async def test_saving_new_prompt_mid_analysis_only_affects_next_request(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save during analysis must not alter the current request's snapshot."""
    row = save(context, 1, "Instructions captured at request start")
    original_analyze = endpoint.docs.analyze

    def edit_during_first_analysis(**kwargs: Any) -> str:
        """Save future guidance during the first model invocation."""
        if not context.calls:
            database = context.databases[1]
            try:
                database.save_service_prompt_override(
                    PROMPT_ID, {"system": "Instructions for future requests"}, row.revision
                )
            finally:
                database.close_connection()
        return original_analyze(**kwargs)

    monkeypatch.setattr(endpoint.docs, "analyze", edit_during_first_analysis)
    await process(
        perform_chunking=True, chunk_method="words", chunk_size=8, chunk_overlap=0, summarize_recursively=True
    )
    assert len(context.calls) > 2
    assert {call["system_message"] for call in context.calls} == {"Instructions captured at request start"}
    await process()
    assert context.calls[-1]["system_message"] == "Instructions for future requests"


def test_authenticated_http_route_uses_saved_document_prompt(
    context: SimpleNamespace, client_with_single_user: tuple[TestClient, object]
) -> None:
    """Multipart processing must apply the actual authenticated owner's prompt."""
    client, _ = client_with_single_user
    save(context, 1, "The authenticated account style")
    response = client.post(
        "/api/v1/media/process-documents",
        files=[("files", ("example.txt", b"A document about astronomy.", "text/plain"))],
        data={"perform_analysis": "true", "perform_chunking": "false", "api_name": "openai"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["analysis"] == "A summary of this section."
    assert context.calls[0]["system_message"] == "The authenticated account style"
    assert context.reads == [1]


def test_http_explicit_empty_system_prompt_bypasses_saved_override(
    context: SimpleNamespace, client_with_single_user: tuple[TestClient, object]
) -> None:
    """Multipart empty fields must remain explicit rather than select an override."""
    client, _ = client_with_single_user
    save(context, 1, "This saved prompt must not replace an explicit empty field")
    response = client.post(
        "/api/v1/media/process-documents",
        files=[("files", ("example.txt", b"A document about astronomy.", "text/plain"))],
        data={"perform_chunking": "false", "api_name": "openai", "system_prompt": ""},
    )
    assert response.status_code == 200, response.text
    assert context.calls[0]["system_message"] == ""
    assert context.reads == []
