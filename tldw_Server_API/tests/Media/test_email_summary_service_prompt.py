"""Exercise email prompts through real multipart ingestion and recursive analysis."""

import sys
import threading
from collections.abc import Iterator
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn
from zipfile import ZipFile

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.media_processing_deps import get_process_emails_form
from tldw_Server_API.app.api.v1.endpoints.media import process_emails as endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import ProcessEmailsForm
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase, ServicePromptOverrideRow
from tldw_Server_API.app.core.exceptions import ServicePromptCorruptOverride
from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary

pytestmark = pytest.mark.integration
PROMPT_ID = "media.email.summarization"


def message(body: str = "Please send the revised report on Friday.") -> EmailMessage:
    """Build a real, self-contained message without external fixtures."""
    result = EmailMessage()
    result["From"] = "alice@example.com"
    result["To"] = "bob@example.com"
    result["Subject"] = "Report deadline"
    result.set_content(body)
    return result


@pytest.mark.parametrize("system", [None, "", "Explicit {literal} guidance"])
def test_email_form_preserves_prompt_presence(system: str | None) -> None:
    """The validated model distinguishes an omitted system prompt from empty text."""
    app = FastAPI()

    @app.post("/parse")
    async def parse(form: ProcessEmailsForm = Depends(get_process_emails_form)) -> dict[str, str | None]:
        """Expose the validated field without endpoint repair."""
        return {"system_prompt": form.system_prompt}

    fields = {"api_name": (None, "openai")}
    if system is not None:
        fields["system_prompt"] = (None, system)
    with TestClient(app) as client:
        response = client.post("/parse", files=fields)
    assert response.status_code == 200
    assert response.json() == {"system_prompt": system}


@pytest.fixture
def context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client_with_single_user: tuple[TestClient, object]
) -> Iterator[SimpleNamespace]:
    """Keep real storage, extraction and analyzer logic; replace auth and the LLM boundary."""
    client, _ = client_with_single_user
    state = SimpleNamespace(
        client=client,
        owner=1,
        databases={owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "email-prompt-test") for owner in (1, 2)},
        reads=[],
        calls=[],
        original_adapter=summary._summarize_via_adapter,
        email_bytes=message().as_bytes(),
    )

    async def current_user() -> User:
        """Select an authenticated owner for each request."""
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def get_db(request: Request, user: User) -> PromptsDatabase:
        """Observe the no-read and single-snapshot storage contracts."""
        state.reads.append(user.id)
        return state.databases[user.id]

    def adapter(**kwargs: Any) -> str:
        """Return a deterministic model response, capturing real analyzer output."""
        state.calls.append(kwargs)
        return "Send the revised report on Friday."

    monkeypatch.setitem(client.app.dependency_overrides, get_request_user, current_user)
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", get_db, raising=False)
    monkeypatch.setattr(summary, "_summarize_via_adapter", adapter)
    yield state
    for database in state.databases.values():
        database.close_connection()


def save(context: SimpleNamespace, system: str, *, owner: int = 1) -> ServicePromptOverrideRow:
    """Persist an independent owner's email instructions."""
    return context.databases[owner].save_service_prompt_override(PROMPT_ID, {"system": system}, None)


def process(context: SimpleNamespace, *, file_count: int = 1, **options: str) -> dict[str, Any]:
    """Submit EMLs to the authenticated endpoint with analysis explicitly enabled."""
    response = context.client.post(
        "/api/v1/media/process-emails",
        files=[
            ("files", (f"source-{index}.eml", context.email_bytes, "message/rfc822")) for index in range(file_count)
        ],
        data={"perform_analysis": "true", "perform_chunking": "false", "api_name": "openai", **options},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["errors"], body
    assert len(body["results"]) == file_count
    return body


def test_email_prompt_is_owner_scoped_and_independent_of_other_media(context: SimpleNamespace) -> None:
    """The model sees only the authenticated owner's email instructions."""
    save(context, "French email notes. Preserve {literal braces}.")
    save(context, "Spanish email notes.", owner=2)
    for other_id in ("media.document.summarization", "media.pdf.summarization", "media.ebook.summarization"):
        context.databases[1].save_service_prompt_override(other_id, {"system": "Not email guidance."}, None)
    first = process(context)
    context.owner = 2
    process(context)
    assert first["results"][0]["analysis"] == "Send the revised report on Friday."
    assert [call["system_message"] for call in context.calls] == [
        "French email notes. Preserve {literal braces}.",
        "Spanish email notes.",
    ]


@pytest.mark.parametrize("legacy_provider", ["", "anthropic"])
def test_canonical_provider_drives_email_analysis(context: SimpleNamespace, legacy_provider: str) -> None:
    """Canonical provider works alone and wins over the legacy alias."""
    save(context, "Saved email instructions")
    process(context, api_name=legacy_provider, api_provider="openai")
    assert len(context.calls) == 1
    assert context.calls[0]["api_name"] == "openai"
    assert context.calls[0]["system_message"] == "Saved email instructions"


@pytest.mark.parametrize("provider,configured_key", [("openai", "test-configured-key"), ("ollama", None)])
def test_email_core_resolves_configured_and_keyless_credentials(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, provider: str, configured_key: str | None
) -> None:
    """Keep real credential resolution, replacing only the external adapter call."""
    requests = []

    def chat(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Record the fully resolved provider request without network access."""
        requests.append(request)
        return {"choices": [{"message": {"content": "Send the revised report on Friday."}}]}

    monkeypatch.setattr(summary, "_summarize_via_adapter", context.original_adapter)
    monkeypatch.setattr(
        summary, "get_registry", lambda: SimpleNamespace(get_adapter=lambda name: SimpleNamespace(chat=chat))
    )
    monkeypatch.setattr(
        summary, "loaded_config_data", {f"{provider}_api": {"api_key": configured_key, "model": "test-summary-model"}}
    )
    result = email_lib.process_email_task(
        file_bytes=context.email_bytes,
        filename="source.eml",
        perform_analysis=True,
        api_name=provider,
        system_prompt="Direct caller instructions",
        perform_chunking=False,
    )
    assert result["analysis"] == "Send the revised report on Friday."
    assert requests[0]["api_key"] == configured_key
    assert requests[0]["model"] == "test-summary-model"
    assert requests[0]["system_message"] == "Direct caller instructions"
    assert context.reads == []


@pytest.mark.parametrize("system", ["Explicit email guidance", ""])
def test_explicit_prompt_bypasses_storage(context: SimpleNamespace, system: str) -> None:
    """Explicit text and empty multipart fields win without accessing owner storage."""
    save(context, "Saved guidance must not win.")
    process(context, system_prompt=system, custom_prompt="Focus on deadlines.")
    assert len(context.calls) == 1
    assert context.calls[0]["system_message"] == system
    assert context.calls[0]["custom_prompt_arg"] == "Focus on deadlines."
    assert context.reads == []


@pytest.mark.parametrize("options", [{"perform_analysis": "false"}, {"api_name": ""}])
def test_disabled_analysis_or_missing_provider_skips_storage(context: SimpleNamespace, options: dict[str, str]) -> None:
    """Non-analysis requests remain usable even with an invalid saved override."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"unknown": "bad"}, None)
    body = process(context, **options)
    assert body["results"][0]["analysis"] is None
    assert context.calls == []
    assert context.reads == []


def test_batch_and_recursive_passes_keep_one_snapshot(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit or account change during analysis affects only later requests."""
    context.email_bytes = message("The report describes astronomy and biology in detail. " * 400).as_bytes()
    row = save(context, "Initial email instructions")
    original_adapter = summary._summarize_via_adapter

    def edit_during_analysis(**kwargs: Any) -> str:
        """Change future settings during the first external model call."""
        if not context.calls:
            database = context.databases[1]
            try:
                database.save_service_prompt_override(PROMPT_ID, {"system": "Future instructions"}, row.revision)
            finally:
                database.close_connection()
            context.owner = 2
        return original_adapter(**kwargs)

    monkeypatch.setattr(summary, "_summarize_via_adapter", edit_during_analysis)
    process(context, file_count=2, summarize_recursively="true", custom_prompt="Focus on deadlines.")
    assert len(context.calls) > 2
    assert {call["system_message"] for call in context.calls} == {"Initial email instructions"}
    assert {call["custom_prompt_arg"] for call in context.calls} == {"Focus on deadlines."}
    assert context.reads == [1]  # Exactly one owner snapshot per request is required.
    context.owner = 1
    process(context)
    assert context.calls[-1]["system_message"] == "Future instructions"


@pytest.mark.parametrize("container", ["zip", "mbox"])
def test_mail_containers_share_the_request_prompt(context: SimpleNamespace, container: str) -> None:
    """Archive expansion must not drop email guidance or resolve it again per member."""
    save(context, "Summarize each email in French.")
    if container == "zip":
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            for index in range(2):
                archive.writestr(f"{index}.eml", context.email_bytes)
        payload = buffer.getvalue()
        flag = "accept_archives"
    else:
        payload = (b"From alice@example.com Sat Sep 5 00:00:00 2026\n" + context.email_bytes + b"\n") * 2
        flag = "accept_mbox"
    response = context.client.post(
        "/api/v1/media/process-emails",
        files=[("files", (f"mail.{container}", payload, "application/octet-stream"))],
        data={"perform_analysis": "true", "perform_chunking": "false", "api_provider": "openai", flag: "true"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"]) == 2
    assert [call["system_message"] for call in context.calls] == ["Summarize each email in French."] * 2
    assert context.reads == [1]


@pytest.mark.parametrize("body", ['{"deadline": "Friday"}', '["Friday", "report"]', '"Friday"'])
def test_json_shaped_email_body_is_analyzed_as_literal_text(context: SimpleNamespace, body: str) -> None:
    """Email content is not an analyzer JSON envelope and must reach the model intact."""
    context.email_bytes = message(body).as_bytes()
    result = process(context, system_prompt="Summarize the email.")
    assert result["results"][0]["analysis"] == "Send the revised report on Friday."
    assert context.calls[0]["text_to_summarize"] == body


@pytest.mark.parametrize("extension", ["pst", "ost"])
def test_enabled_pst_ost_messages_keep_request_prompt(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, extension: str
) -> None:
    """Exercise container traversal/conversion with only optional libpff substituted."""
    save(context, "PST and OST email instructions")
    libpff_message = SimpleNamespace(
        subject="Report",
        sender_name="Alice",
        sender_email_address="alice@example.com",
        plain_text_body="Send the revised report on Friday.",
        number_of_recipients=0,
        number_of_attachments=0,
    )
    folder = SimpleNamespace(
        number_of_sub_folders=0, number_of_sub_messages=2, get_sub_message=lambda index: libpff_message
    )
    libpff_file = SimpleNamespace(open=lambda path: None, close=lambda: None, get_root_folder=lambda: folder)
    monkeypatch.setitem(sys.modules, "pypff", SimpleNamespace(file=lambda: libpff_file))
    response = context.client.post(
        "/api/v1/media/process-emails",
        files=[("files", (f"mail.{extension}", b"libpff fixture boundary", "application/octet-stream"))],
        data={"perform_analysis": "true", "perform_chunking": "false", "api_provider": "openai", "accept_pst": "true"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"]) == 2
    assert [call["system_message"] for call in context.calls] == ["PST and OST email instructions"] * 2
    assert context.reads == [1]


@pytest.mark.parametrize("corrupt", [False, True])
def test_prompt_connection_is_closed_on_lookup_worker(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    """Both successful and rejected prompt lookups release their thread-local connection."""
    database = context.databases[1]
    database.save_service_prompt_override(PROMPT_ID, {"unknown" if corrupt else "system": "guidance"}, None)
    lookup_threads = []
    close_threads = []
    original_read = database.get_service_prompt_override
    original_close = database.close_connection

    def read(definition_id: str) -> ServicePromptOverrideRow | None:
        """Observe the real lookup worker."""
        lookup_threads.append(threading.get_ident())
        return original_read(definition_id)

    def close() -> None:
        """Release the actual connection and record which worker released it."""
        original_close()
        close_threads.append(threading.get_ident())

    monkeypatch.setattr(database, "get_service_prompt_override", read)
    monkeypatch.setattr(database, "close_connection", close)
    if corrupt:
        with pytest.raises(ServicePromptCorruptOverride):
            process(context)
    else:
        process(context)
    assert len(lookup_threads) == 1
    assert close_threads == lookup_threads
    assert lookup_threads[0] != threading.get_ident()


def test_nested_attachments_are_not_analyzed(context: SimpleNamespace) -> None:
    """Enabling parent analysis must not start model calls for nested attachments."""
    outer = message()
    outer.add_attachment(message("Private nested content."), subtype="rfc822", filename="nested.eml")
    context.email_bytes = outer.as_bytes()
    save(context, "Summarize the parent email.")
    body = process(context, ingest_attachments="true", max_depth="2")
    assert len(context.calls) == 1
    assert body["results"][0]["children"][0]["analysis"] is None


def test_unset_and_reset_use_frozen_deployment_defaults(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request resolves deployment defaults once, including after reset."""
    defaults_read = []

    def server_default() -> str:
        """Supply a deployment override independently of packaged defaults."""
        defaults_read.append(True)
        return "Deployment email instructions"

    monkeypatch.setattr(summary, "_resolve_default_system_prompt", server_default)
    process(context, file_count=2)
    assert [call["system_message"] for call in context.calls] == ["Deployment email instructions"] * 2
    assert defaults_read == [True]
    row = save(context, "Custom instructions")
    context.databases[1].reset_service_prompt_override(PROMPT_ID, row.revision)
    process(context)
    assert context.calls[-1]["system_message"] == "Deployment email instructions"
    assert defaults_read == [True, True]


def test_corrupt_override_fails_before_input_processing(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid stored parts must fail closed before uploads and LLM calls."""
    context.databases[1].save_service_prompt_override(PROMPT_ID, {"unknown": "bad"}, None)

    async def unexpected_upload(*args: Any, **kwargs: Any) -> NoReturn:
        """Detect processing before prompt validation."""
        raise AssertionError("Uploads started before validating the prompt")

    monkeypatch.setattr(endpoint, "save_uploaded_files", unexpected_upload)
    with pytest.raises(ServicePromptCorruptOverride):
        context.client.post(
            "/api/v1/media/process-emails",
            files=[("files", ("source.eml", context.email_bytes, "message/rfc822"))],
            data={"perform_analysis": "true", "api_name": "openai"},
        )
    assert context.calls == []
