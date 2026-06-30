import datetime
import pytest
from fastapi import status

from tldw_Server_API.app.api.v1.API_Deps.chat_documents_deps import get_document_generator_service
from tldw_Server_API.app.api.v1.endpoints import chat as chat_router
from tldw_Server_API.app.core.Chat.document_generator import DocumentType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError, InputError
from tldw_Server_API.tests._plugins.chat_fixtures import get_auth_headers

pytestmark = pytest.mark.usefixtures("setup_dependencies")


@pytest.fixture(autouse=True)
def _ensure_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield


def _make_payload(**overrides):


    base = {
        "conversation_id": "chat-42",
        "document_type": "summary",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "stream": False,
        "async_generation": False,
    }
    base.update(overrides)
    return base


def test_document_generate_streams_as_sse(authenticated_client, auth_token):


    calls = {}

    class StreamingStubService:
        stored_docs: list[dict] = []
        next_id: int = 1

        def __init__(self, db):

            self._db = db

        def generate_document(self, *, stream, **kwargs):

            calls["stream"] = stream

            async def _generator():
                yield "first chunk"
                yield b"second chunk"

            return _generator()

        def record_streamed_document(
            self,
            *,
            conversation_id,
            document_type,
            content,
            provider,
            model,
            generation_time_ms,
            token_count=None,
        ):

            doc_id = StreamingStubService.next_id
            StreamingStubService.next_id += 1
            StreamingStubService.stored_docs.append(
                {
                    "id": doc_id,
                    "conversation_id": conversation_id,
                    "document_type": document_type.value if hasattr(document_type, "value") else document_type,
                    "title": "Streamed Document",
                    "content": content,
                    "provider": provider,
                    "model": model,
                    "generation_time_ms": generation_time_ms,
                    "token_count": token_count,
                    "created_at": datetime.datetime.utcnow(),
                    "metadata": {},
                }
            )
            return doc_id

        def get_generated_documents(self, conversation_id=None, document_type=None, limit=50, offset=0):

            docs = list(StreamingStubService.stored_docs)
            if conversation_id is not None:
                docs = [doc for doc in docs if doc["conversation_id"] == conversation_id]
            if document_type is not None:
                dtype = document_type.value if hasattr(document_type, "value") else document_type
                docs = [doc for doc in docs if doc["document_type"] == dtype]
            docs.sort(key=lambda item: item["id"], reverse=True)
            return docs[offset:offset + limit]

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: StreamingStubService
    StreamingStubService.stored_docs = []
    StreamingStubService.next_id = 1

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(stream=True),
    )

    assert response.status_code == 200
    assert calls["stream"] is True
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "data: first chunk\n\n" in body
    assert "data: second chunk\n\n" in body
    assert body.strip().endswith("data: [DONE]")
    response.close()

    headers = get_auth_headers(auth_token, getattr(authenticated_client, "csrf_token", ""))
    list_response = authenticated_client.get(
        "/api/v1/chat/documents",
        params={"conversation_id": "chat-42"},
        headers=headers,
    )
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total"] == 1
    assert payload["documents"][0]["content"] == "first chunksecond chunk"
    assert StreamingStubService.stored_docs, "Streamed document was not persisted"


def test_document_generate_bubbles_service_error(authenticated_client):


    class FailingStubService:
        record_calls = 0

        def __init__(self, db):

            self._db = db

        def generate_document(self, *args, **kwargs):

            return {"success": False, "error": "No messages found for conversation chat-42"}

        def get_generated_documents(self, *args, **kwargs):

            return []

        def record_streamed_document(self, *args, **kwargs):

            FailingStubService.record_calls += 1
            return None

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: FailingStubService
    FailingStubService.record_calls = 0

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(),
    )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "No messages found for conversation chat-42"}
    response.close()
    assert FailingStubService.record_calls == 0


def test_document_generate_maps_input_error_from_service(authenticated_client):


    class InputErrorStubService:
        def __init__(self, db):

            self._db = db

        def generate_document(self, *args, **kwargs):

            raise InputError("No messages found for conversation chat-42")

        def get_generated_documents(self, *args, **kwargs):

            return []

        def record_streamed_document(self, *args, **kwargs):

            return None

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: InputErrorStubService

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=_make_payload(),
    )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "No messages found for conversation chat-42"}
    response.close()


def test_document_list_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_documents(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite list exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.get("/api/v1/chat/documents")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to list generated documents"}
    response.close()


def test_document_list_includes_canonical_pagination(authenticated_client) -> None:
    """Generated document listing preserves total while exposing canonical offset pagination."""
    calls = {"list": [], "count": []}

    class PaginatedStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_documents(self, conversation_id=None, document_type=None, limit=50, offset=0):
            calls["list"].append(
                {
                    "conversation_id": conversation_id,
                    "document_type": document_type,
                    "limit": limit,
                    "offset": offset,
                }
            )
            docs = [
                {
                    "id": 12,
                    "conversation_id": conversation_id,
                    "document_type": "summary",
                    "title": "Newest Doc",
                    "content": "newer",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 10,
                    "token_count": 3,
                    "created_at": datetime.datetime.utcnow(),
                    "metadata": {},
                },
                {
                    "id": 11,
                    "conversation_id": conversation_id,
                    "document_type": "summary",
                    "title": "Older Doc",
                    "content": "older",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 12,
                    "token_count": 4,
                    "created_at": datetime.datetime.utcnow(),
                    "metadata": {},
                },
            ]
            return docs[offset:offset + limit]

        def count_generated_documents(self, conversation_id=None, document_type=None):
            calls["count"].append(
                {
                    "conversation_id": conversation_id,
                    "document_type": document_type,
                }
            )
            return 4

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: PaginatedStubService

    response = authenticated_client.get(
        "/api/v1/chat/documents",
        params={"conversation_id": "chat-42", "document_type": "summary", "limit": 1, "offset": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [doc["id"] for doc in payload["documents"]] == [11]
    assert payload["total"] == 4
    assert payload["conversation_id"] == "chat-42"
    assert payload["document_type"] == "summary"
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 1,
        "total": 4,
        "has_more": True,
        "next_offset": 2,
    }
    assert calls["list"] == [
        {
            "conversation_id": "chat-42",
            "document_type": DocumentType.SUMMARY,
            "limit": 1,
            "offset": 1,
        }
    ]
    assert calls["count"] == [
        {
            "conversation_id": "chat-42",
            "document_type": DocumentType.SUMMARY,
        }
    ]
    response.close()


def test_document_get_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_document_by_id(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite get exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.get("/api/v1/chat/documents/123")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to get generated document"}
    response.close()


def test_document_job_status_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_job_status(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite job exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.get("/api/v1/chat/documents/jobs/job-1")

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to get generation job status"}
    response.close()


def test_document_cancel_maps_database_error_from_service(authenticated_client, auth_token):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_job_status(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite cancel exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.delete(
        "/api/v1/chat/documents/jobs/job-1",
        headers=get_auth_headers(auth_token, getattr(authenticated_client, "csrf_token", "")),
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to cancel generation job"}
    response.close()


def test_document_delete_maps_database_error_from_service(authenticated_client, auth_token):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def delete_generated_document(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite delete exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.delete(
        "/api/v1/chat/documents/123",
        headers=get_auth_headers(auth_token, getattr(authenticated_client, "csrf_token", "")),
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to delete generated document"}
    response.close()


def test_document_save_prompt_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def save_user_prompt_config(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite prompt exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.post(
        "/api/v1/chat/documents/prompts",
        json={
            "document_type": "summary",
            "system_prompt": "Summarize.",
            "user_prompt": "Content: {content}",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to save prompt configuration"}
    response.close()


def test_document_bulk_maps_database_error_from_service(authenticated_client):
    class DatabaseErrorStubService:
        def __init__(self, db):
            self._db = db

        def create_generation_job(self, *args, **kwargs):
            raise CharactersRAGDBError("sqlite bulk exploded")

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: DatabaseErrorStubService

    response = authenticated_client.post(
        "/api/v1/chat/documents/bulk",
        json={
            "conversation_ids": ["chat-42"],
            "document_types": ["summary"],
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "async_generation": True,
        },
    )

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": "Failed to create bulk generation jobs"}
    response.close()


@pytest.mark.parametrize(
    ("case_name", "service_method", "request_factory", "expected_detail"),
    [
        (
            "job_status",
            "get_job_status",
            lambda client, token: client.get("/api/v1/chat/documents/jobs/job-1"),
            "Failed to get generation job status",
        ),
        (
            "cancel_job",
            "get_job_status",
            lambda client, token: client.delete(
                "/api/v1/chat/documents/jobs/job-1",
                headers=get_auth_headers(token, getattr(client, "csrf_token", "")),
            ),
            "Failed to cancel generation job",
        ),
        (
            "list_documents",
            "get_generated_documents",
            lambda client, token: client.get("/api/v1/chat/documents"),
            "Failed to list generated documents",
        ),
        (
            "get_document",
            "get_generated_document_by_id",
            lambda client, token: client.get("/api/v1/chat/documents/123"),
            "Failed to get generated document",
        ),
        (
            "delete_document",
            "delete_generated_document",
            lambda client, token: client.delete(
                "/api/v1/chat/documents/123",
                headers=get_auth_headers(token, getattr(client, "csrf_token", "")),
            ),
            "Failed to delete generated document",
        ),
        (
            "save_prompt",
            "save_user_prompt_config",
            lambda client, token: client.post(
                "/api/v1/chat/documents/prompts",
                json={
                    "document_type": "summary",
                    "system_prompt": "Summarize.",
                    "user_prompt": "Content: {content}",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            ),
            "Failed to save prompt configuration",
        ),
        (
            "get_prompt",
            "get_user_prompt_config",
            lambda client, token: client.get("/api/v1/chat/documents/prompts/summary"),
            "Failed to get prompt configuration",
        ),
        (
            "bulk_generate",
            "create_generation_job",
            lambda client, token: client.post(
                "/api/v1/chat/documents/bulk",
                json={
                    "conversation_ids": ["chat-42"],
                    "document_types": ["summary"],
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-test",
                    "async_generation": True,
                },
            ),
            "Failed to create bulk generation jobs",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_document_handlers_sanitize_unexpected_service_errors(
    authenticated_client,
    auth_token,
    case_name,
    service_method,
    request_factory,
    expected_detail,
):
    def _raise_unexpected_error(self, *args, **kwargs):
        _ = (self, args, kwargs)
        raise RuntimeError(f"{case_name} backend unavailable")

    RuntimeErrorStubService = type(
        "RuntimeErrorStubService",
        (),
        {
            "__init__": lambda self, db: setattr(self, "_db", db),
            service_method: _raise_unexpected_error,
        },
    )

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: RuntimeErrorStubService

    response = request_factory(authenticated_client, auth_token)

    assert response.status_code == 500, response.text
    assert response.json() == {"detail": expected_detail}
    response.close()


def test_document_generate_uses_configured_api_key(monkeypatch, authenticated_client):


    captured = {}

    class KeyCaptureService:
        def __init__(self, db):
            self._db = db

        def generate_document(self, *, stream, **kwargs):

            captured["api_key"] = kwargs.get("api_key")
            captured["provider"] = kwargs.get("provider")
            return "Generated content"

        def get_generated_documents(self, conversation_id=None, document_type=None, limit=50, offset=0):

            return [
                {
                    "id": 101,
                    "conversation_id": conversation_id,
                    "document_type": document_type.value if hasattr(document_type, "value") else document_type,
                    "title": "Doc",
                    "content": "Generated content",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "generation_time_ms": 123,
                    "created_at": datetime.datetime.utcnow(),
                }
            ]

    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas as chat_schemas

    authenticated_client.app.dependency_overrides[get_document_generator_service] = lambda: KeyCaptureService
    monkeypatch.setitem(chat_router.API_KEYS, "openai", "sk-configured")
    monkeypatch.setitem(chat_schemas.API_KEYS, "openai", "sk-configured")

    payload = _make_payload()
    payload.pop("api_key", None)

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=payload,
    )

    assert response.status_code == 200, response.text
    assert captured["api_key"] == "sk-configured"
    assert captured["provider"] == "openai"


def test_document_generate_missing_provider_credentials_returns_503(monkeypatch, authenticated_client):


    from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_docs
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

    async def _missing(provider, *_args, **_kwargs):
        return ResolvedByokCredentials(
            provider=provider,
            api_key=None,
            app_config=None,
            credential_fields={},
            source="server",
            allowlisted=True,
        )

    monkeypatch.setattr(chat_docs, "resolve_byok_credentials", _missing)

    payload = _make_payload()
    payload.pop("api_key", None)

    response = authenticated_client.post(
        "/api/v1/chat/documents/generate",
        json=payload,
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = response.json().get("detail", {})
    assert detail.get("error_code") == "missing_provider_credentials"
