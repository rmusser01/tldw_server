"""Exercise PDF Service Prompts through real multipart processing and extraction."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pymupdf
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.media import process_pdfs as endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase, ServicePromptOverrideRow
from tldw_Server_API.app.core.exceptions import ServicePromptCorruptOverride
from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary

pytestmark = pytest.mark.integration
PROMPT_ID = "media.pdf.summarization"


@pytest.fixture
def context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_with_single_user: tuple[TestClient, object]
) -> Iterator[SimpleNamespace]:
    """Use real owner databases and PDFs, replacing only auth and model boundaries."""
    client, _ = client_with_single_user
    state = SimpleNamespace(
        client=client,
        owner=1,
        databases={owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "pdf-prompt-test") for owner in (1, 2)},
        reads=[],
        calls=[],
    )
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "First section explains astronomy. Second section explains biology.\n"
            "Third section describes chemistry. Fourth section covers geology.\n"
            "Fifth section explains geometry and compares different shapes.",
        )
        state.pdf_bytes = document.tobytes()

    async def current_user() -> User:
        """Represent the authenticated owner selected by the test."""
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def get_db(request: Request, user: User) -> PromptsDatabase:
        """Return one real database per owner and observe lazy access."""
        state.reads.append(user.id)
        return state.databases[user.id]

    def analyze(**kwargs: Any) -> str:
        """Capture actual processor inputs without contacting an LLM."""
        state.calls.append(kwargs)
        return "A PDF section summary."

    monkeypatch.setitem(client.app.dependency_overrides, get_request_user, current_user)
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", get_db, raising=False)
    monkeypatch.setattr(pdf, "analyze", analyze)
    yield state
    for database in state.databases.values():
        database.close_connection()


def save(context: SimpleNamespace, system: str, *, owner: int = 1) -> ServicePromptOverrideRow:
    """Persist PDF guidance in the selected owner's real prompt database."""
    return context.databases[owner].save_service_prompt_override(PROMPT_ID, {"system": system}, None)


def process(context: SimpleNamespace, *, file_count: int = 1, **options: str) -> dict[str, Any]:
    """Submit valid PDFs to the authenticated HTTP route and return its result."""
    response = context.client.post(
        "/api/v1/media/process-pdfs",
        files=[("files", (f"source-{index}.pdf", context.pdf_bytes, "application/pdf")) for index in range(file_count)],
        data={
            "perform_analysis": "true",
            "perform_chunking": "false",
            "pdf_parsing_engine": "pymupdf",
            "api_name": "openai",
            **options,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["errors"], body
    assert len(body["results"]) == file_count
    return body


def test_pdf_prompt_is_owner_scoped_and_independent_of_document_guidance(context: SimpleNamespace) -> None:
    """The PDF route must use its authenticated owner's PDF override only."""
    save(context, "Write PDF notes in French. Preserve {literal braces}.")
    save(context, "Write PDF notes in Spanish.", owner=2)
    context.databases[1].save_service_prompt_override(
        "media.document.summarization", {"system": "This is not PDF guidance."}, None
    )
    process(context)
    context.owner = 2
    process(context)
    assert [call["system_message"] for call in context.calls] == [
        "Write PDF notes in French. Preserve {literal braces}.",
        "Write PDF notes in Spanish.",
    ]
    assert context.reads == [1, 2]


@pytest.mark.parametrize("system", ["Explicit PDF guidance", ""])
def test_explicit_multipart_prompt_bypasses_saved_override(context: SimpleNamespace, system: str) -> None:
    """Explicit text and empty multipart fields must take precedence over storage."""
    save(context, "Saved guidance must not win.")
    process(context, system_prompt=system, custom_prompt="Focus on experiments.")
    assert len(context.calls) == 1
    assert context.calls[0]["api_name"] == "openai"
    assert context.calls[0]["api_key"] is None
    assert context.calls[0]["system_message"] == system
    assert context.calls[0]["custom_prompt_arg"] == "Focus on experiments."
    assert context.reads == []


@pytest.mark.parametrize("options", [{"perform_analysis": "false"}, {"api_name": ""}])
def test_disabled_analysis_or_missing_provider_skips_prompt_storage(
    context: SimpleNamespace, options: dict[str, str]
) -> None:
    """Requests without model work must not read even a corrupt saved override."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"unknown": "bad"}, None)
    process(context, **options)
    assert context.calls == []
    assert context.reads == []


def test_batch_chunks_and_recursive_passes_keep_one_snapshot(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing during analysis must affect the next request, not later batch calls."""
    row = save(context, "Initial PDF instructions")
    original_analyze = pdf.analyze

    def edit_during_analysis(**kwargs: Any) -> str:
        """Save future guidance during the first model call on its worker."""
        if not context.calls:
            database = context.databases[1]
            try:
                database.save_service_prompt_override(PROMPT_ID, {"system": "Future PDF instructions"}, row.revision)
            finally:
                database.close_connection()
        return original_analyze(**kwargs)

    monkeypatch.setattr(pdf, "analyze", edit_during_analysis)
    process(
        context,
        file_count=2,
        perform_chunking="true",
        chunk_method="words",
        chunk_size="8",
        chunk_overlap="0",
        summarize_recursively="true",
    )
    assert len(context.calls) > 4
    assert {call["system_message"] for call in context.calls} == {"Initial PDF instructions"}
    recursive = [call for call in context.calls if "\n\n---\n\n" in call["input_data"]]
    assert len(recursive) == 2
    assert {call["custom_prompt_arg"] for call in recursive} == {
        "Provide a concise overall analysis of the preceding text sections."
    }
    assert context.reads == [1]
    process(context)
    assert context.calls[-1]["system_message"] == "Future PDF instructions"


def test_unset_and_reset_prompt_snapshot_the_server_default(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset/reset PDF settings must retain deployment-specific defaults per request."""
    defaults_read = []

    def server_default() -> str:
        """Observe resolution of a deployment-specific analyzer default."""
        defaults_read.append(True)
        return "Deployment PDF instructions"

    monkeypatch.setattr(summary, "_resolve_default_system_prompt", server_default)
    process(context, file_count=2)
    assert [call["system_message"] for call in context.calls] == ["Deployment PDF instructions"] * 2
    assert defaults_read == [True]
    row = save(context, "Custom PDF instructions")
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)
    process(context)
    assert context.calls[-1]["system_message"] == "Deployment PDF instructions"
    assert defaults_read == [True, True]


def test_corrupt_override_fails_before_input_processing(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid saved parts must fail closed without reading uploads or calling models."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"unknown": "bad"}, None)

    async def unexpected_upload(*args: Any, **kwargs: Any) -> NoReturn:
        """Detect input processing that starts before prompt validation."""
        raise AssertionError("Uploads started before validating the prompt")

    monkeypatch.setattr(endpoint, "save_uploaded_files", unexpected_upload)
    with pytest.raises(ServicePromptCorruptOverride):
        context.client.post(
            "/api/v1/media/process-pdfs",
            files=[("files", ("source.pdf", context.pdf_bytes, "application/pdf"))],
            data={"perform_analysis": "true", "api_name": "openai"},
        )
    assert context.calls == []
