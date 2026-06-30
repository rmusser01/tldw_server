from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import prompts as prompts_endpoint
from tldw_Server_API.app.api.v1.endpoints.prompts import (
    create_prompt,
    create_collection,
    create_keyword,
    delete_keyword,
    delete_prompt,
    export_prompts_api,
    export_keywords_api,
    get_collection,
    get_prompt,
    import_prompts_api,
    legacy_create_prompt,
    list_all_keywords,
    list_all_prompts,
    list_collections,
    list_prompt_versions,
    preview_prompt_api,
    record_prompt_usage,
    restore_prompt_version,
    search_all_prompts,
    update_collection,
    update_prompt,
)
from tldw_Server_API.app.api.v1.schemas import prompt_schemas as schemas
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    ConflictError,
    DatabaseError,
    InputError,
)


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_prompts_health_sanitizes_settings_failure(monkeypatch):
    from tldw_Server_API.app.core import config

    class _ExplodingSettings:
        def get(self, _key):
            raise RuntimeError("prompts settings exploded at /private/prompts.db")

    monkeypatch.setattr(config, "settings", _ExplodingSettings())

    response = await prompts_endpoint.prompts_health()

    assert response["status"] == "unhealthy"
    assert response["error"] == "Prompts health check failed"


def _structured_prompt_definition_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "input",
                "label": "Input",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Evaluate {{input}}",
                "enabled": True,
                "order": 10,
                "is_template": True,
            }
        ],
    }


class _BrokenSearchPromptsDb:
    def search_prompts(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _SearchPromptsDb:
    def search_prompts(self, **kwargs):
        assert kwargs["page"] == 2
        assert kwargs["results_per_page"] == 1
        return (
            [
                {
                    "id": 42,
                    "uuid": "00000000-0000-0000-0000-000000000042",
                    "name": "Prompt 42",
                    "author": "tester",
                    "details": None,
                    "system_prompt": "system",
                    "user_prompt": "user",
                    "prompt_format": "legacy",
                    "prompt_schema_version": None,
                    "prompt_definition": None,
                    "last_modified": datetime(2026, 5, 2, tzinfo=timezone.utc),
                    "version": 1,
                    "usage_count": 0,
                    "last_used_at": None,
                    "keywords": [],
                    "deleted": False,
                    "relevance_score": 0.5,
                }
            ],
            3,
        )


class _CreateKeywordDb:
    def __init__(
        self,
        *,
        existing_active_keyword=None,
        add_keyword_exc: Exception | None = None,
    ):
        self._existing_active_keyword = existing_active_keyword
        self._add_keyword_exc = add_keyword_exc

    def get_active_keyword_by_text(self, *_args, **_kwargs):
        return self._existing_active_keyword

    def add_keyword(self, *_args, **_kwargs):
        if self._add_keyword_exc is not None:
            raise self._add_keyword_exc
        return (1, "kw-1")

    def _normalize_keyword(self, keyword_text: str):
        return keyword_text.strip().lower()


class _BrokenListKeywordsDb:
    def fetch_all_keywords(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _DeleteKeywordDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def soft_delete_keyword(self, *_args, **_kwargs):
        raise self._exc


class _CreateCollectionDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def create_prompt_collection(self, *_args, **_kwargs):
        raise self._exc


class _ListCollectionsDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def list_prompt_collections(self, *_args, **_kwargs):
        raise self._exc


class _UpdateCollectionDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def update_prompt_collection(self, *_args, **_kwargs):
        raise self._exc


class _GetCollectionDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def get_prompt_collection_by_id(self, *_args, **_kwargs):
        raise self._exc


class _BrokenListPromptsDb:
    def list_prompts(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _ListPromptsDb:
    def list_prompts(self, **kwargs):
        assert kwargs["page"] == 2
        assert kwargs["per_page"] == 1
        return (
            [
                {
                    "id": 42,
                    "uuid": "00000000-0000-0000-0000-000000000042",
                    "name": "Prompt 42",
                    "author": "tester",
                    "last_modified": datetime(2026, 5, 2, tzinfo=timezone.utc),
                    "usage_count": 0,
                    "last_used_at": None,
                }
            ],
            3,
            2,
            3,
        )


class _BrokenGetPromptDb:
    def fetch_prompt_details(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _BrokenListPromptVersionsDb:
    def fetch_prompt_details(self, *_args, **_kwargs):
        return {"id": 1}

    def get_prompt_versions(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _LegacyCreatePromptDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def add_prompt(self, *_args, **_kwargs):
        raise self._exc


class _CreatePromptDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def add_prompt(self, *_args, **_kwargs):
        raise self._exc


class _CreatePromptNoIdDb:
    def add_prompt(self, *_args, **_kwargs):
        return None, None, "prompt backend exploded at /private/prompts.db"


class _UpdatePromptDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def fetch_prompt_details(self, *_args, **_kwargs):
        return {
            "id": 1,
            "name": "alpha",
            "author": "tester",
            "details": "existing",
            "system_prompt": None,
            "user_prompt": None,
            "keywords": [],
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        }

    def update_prompt_by_id(self, *_args, **_kwargs):
        raise self._exc


class _UpdatePromptNoUuidDb:
    def fetch_prompt_details(self, prompt_identifier, *_args, **_kwargs):
        if prompt_identifier == "updated-uuid":
            return None
        return {
            "id": 1,
            "name": "alpha",
            "author": "tester",
            "details": "existing",
            "system_prompt": None,
            "user_prompt": None,
            "keywords": [],
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        }

    def update_prompt_by_id(self, *_args, **_kwargs):
        return None, "backend exploded"


class _RecordPromptUsageDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def record_prompt_usage(self, *_args, **_kwargs):
        raise self._exc


class _DeletePromptDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def soft_delete_prompt(self, *_args, **_kwargs):
        raise self._exc


class _RestorePromptVersionDb:
    def __init__(self, exc: Exception):
        self._exc = exc

    def fetch_prompt_details(self, *_args, **_kwargs):
        return {"id": 1}

    def restore_prompt_version(self, *_args, **_kwargs):
        raise self._exc


@pytest.mark.asyncio
async def test_search_all_prompts_includes_canonical_page_pagination():
    response = await search_all_prompts(
        search_query="alpha",
        search_fields=None,
        page=2,
        results_per_page=1,
        include_deleted=False,
        db=_SearchPromptsDb(),
    )

    assert response.pagination.model_dump(mode="json") == {
        "mode": "page",
        "page": 2,
        "per_page": 1,
        "total": 3,
        "total_pages": 3,
        "has_more": True,
    }


@pytest.mark.asyncio
async def test_search_all_prompts_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await search_all_prompts(
            search_query="alpha",
            search_fields=None,
            page=1,
            results_per_page=20,
            include_deleted=False,
            db=_BrokenSearchPromptsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error during search."


@pytest.mark.asyncio
async def test_create_keyword_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_keyword(
            keyword_data=schemas.KeywordCreate(keyword_text="alpha"),
            db=_CreateKeywordDb(add_keyword_exc=InputError("invalid keyword")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid keyword"


@pytest.mark.asyncio
async def test_create_keyword_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_keyword(
            keyword_data=schemas.KeywordCreate(keyword_text=" Alpha "),
            db=_CreateKeywordDb(existing_active_keyword={"id": 1}),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Keyword 'alpha' already exists and is active."


@pytest.mark.asyncio
async def test_create_keyword_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_keyword(
            keyword_data=schemas.KeywordCreate(keyword_text="alpha"),
            db=_CreateKeywordDb(add_keyword_exc=DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_list_all_keywords_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await list_all_keywords(db=_BrokenListKeywordsDb())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_create_keyword_sanitizes_unexpected_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_keyword(
            keyword_data=schemas.KeywordCreate(keyword_text="alpha"),
            db=_CreateKeywordDb(add_keyword_exc=OSError("keyword backend exploded")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An unexpected error occurred"


@pytest.mark.asyncio
async def test_delete_keyword_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await delete_keyword(
            keyword_text="alpha",
            db=_DeleteKeywordDb(InputError("invalid keyword")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid keyword"


@pytest.mark.asyncio
async def test_delete_keyword_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await delete_keyword(
            keyword_text="alpha",
            db=_DeleteKeywordDb(ConflictError("keyword in use")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "keyword in use"


@pytest.mark.asyncio
async def test_delete_keyword_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await delete_keyword(
            keyword_text="alpha",
            db=_DeleteKeywordDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_export_prompts_api_maps_database_error(monkeypatch):
    def _raise_db_error(**_kwargs):
        raise DatabaseError("driver failed")

    monkeypatch.setattr(
        prompts_endpoint,
        "db_export_prompts_formatted",
        _raise_db_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await export_prompts_api(
            export_format="csv",
            filter_keywords=None,
            include_system=True,
            include_user=True,
            include_details=True,
            include_author=True,
            include_associated_keywords=True,
            markdown_template_name="Basic Template",
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error during export."


@pytest.mark.asyncio
async def test_export_prompts_api_sanitizes_failed_status_message(monkeypatch):
    monkeypatch.setattr(
        prompts_endpoint,
        "db_export_prompts_formatted",
        lambda **_kwargs: ("export backend exploded", "/tmp/missing-prompts-export.csv"),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await export_prompts_api(
            export_format="csv",
            filter_keywords=None,
            include_system=True,
            include_user=True,
            include_details=True,
            include_author=True,
            include_associated_keywords=True,
            markdown_template_name="Basic Template",
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Export failed"


@pytest.mark.asyncio
async def test_export_prompts_api_sanitizes_unexpected_error(monkeypatch):
    def _raise_unexpected_error(**_kwargs):
        raise OSError("export backend exploded")

    monkeypatch.setattr(
        prompts_endpoint,
        "db_export_prompts_formatted",
        _raise_unexpected_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await export_prompts_api(
            export_format="csv",
            filter_keywords=None,
            include_system=True,
            include_user=True,
            include_details=True,
            include_author=True,
            include_associated_keywords=True,
            markdown_template_name="Basic Template",
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected error during export"


@pytest.mark.asyncio
async def test_export_keywords_api_sanitizes_unexpected_error(monkeypatch):
    def _raise_unexpected_error(**_kwargs):
        raise OSError("keyword export backend exploded")

    monkeypatch.setattr(
        prompts_endpoint,
        "db_export_prompt_keywords_to_csv",
        _raise_unexpected_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await export_keywords_api(db=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected error during keyword export"


@pytest.mark.asyncio
async def test_export_keywords_api_sanitizes_failed_status_message(monkeypatch):
    monkeypatch.setattr(
        prompts_endpoint,
        "db_export_prompt_keywords_to_csv",
        lambda **_kwargs: ("keyword export backend exploded", "/tmp/missing-keyword-export.csv"),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await export_keywords_api(db=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Keyword export failed"


@pytest.mark.asyncio
async def test_import_prompts_api_sanitizes_unexpected_error():
    class _BrokenImportDb:
        def fetch_all_prompt_names(self, *_args, **_kwargs):
            return []

        def add_prompt(self, *_args, **_kwargs):
            raise OSError("import backend exploded")

    payload = schemas.PromptImportRequest(
        prompts=[
            schemas.PromptImportItem(
                name="alpha",
                content="Prompt body",
            )
        ],
        skip_duplicates=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await import_prompts_api(payload=payload, db=_BrokenImportDb())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Unexpected error during import"


@pytest.mark.asyncio
async def test_create_collection_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_collection(
            payload=schemas.PromptCollectionCreateRequest(name="alpha"),
            db=_CreateCollectionDb(InputError("invalid collection")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid collection"


@pytest.mark.asyncio
async def test_create_collection_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_collection(
            payload=schemas.PromptCollectionCreateRequest(name="alpha"),
            db=_CreateCollectionDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_list_collections_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await list_collections(
            limit=10,
            offset=0,
            db=_ListCollectionsDb(InputError("invalid paging")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid paging"


@pytest.mark.asyncio
async def test_list_collections_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await list_collections(
            limit=10,
            offset=0,
            db=_ListCollectionsDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_update_collection_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await update_collection(
            collection_id=1,
            payload=schemas.PromptCollectionUpdateRequest(name="alpha"),
            db=_UpdateCollectionDb(InputError("invalid collection")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid collection"


@pytest.mark.asyncio
async def test_update_collection_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await update_collection(
            collection_id=1,
            payload=schemas.PromptCollectionUpdateRequest(name="alpha"),
            db=_UpdateCollectionDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_get_collection_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await get_collection(
            collection_id=1,
            db=_GetCollectionDb(InputError("invalid collection")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid collection"


@pytest.mark.asyncio
async def test_get_collection_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await get_collection(
            collection_id=1,
            db=_GetCollectionDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_list_all_prompts_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await list_all_prompts(
            page=1,
            per_page=10,
            include_deleted=False,
            sort_by="last_modified",
            sort_order="desc",
            db=_BrokenListPromptsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error listing prompts."


@pytest.mark.asyncio
async def test_list_all_prompts_includes_canonical_page_pagination() -> None:
    response = await list_all_prompts(
        page=2,
        per_page=1,
        include_deleted=False,
        sort_by="last_modified",
        sort_order="desc",
        db=_ListPromptsDb(),
    )

    assert response.total_items == 3
    assert response.current_page == 2
    assert response.total_pages == 3
    assert response.pagination.model_dump(mode="json") == {
        "mode": "page",
        "page": 2,
        "per_page": 1,
        "total": 3,
        "total_pages": 3,
        "has_more": True,
    }


@pytest.mark.asyncio
async def test_get_prompt_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await get_prompt(
            prompt_identifier="alpha",
            include_deleted=False,
            db=_BrokenGetPromptDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_list_prompt_versions_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await list_prompt_versions(
            prompt_identifier="alpha",
            db=_BrokenListPromptVersionsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_legacy_create_prompt_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await legacy_create_prompt(
            payload=schemas.LegacyPromptCreateRequest(name="alpha", content="body"),
            db=_LegacyCreatePromptDb(InputError("invalid prompt")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid prompt"


@pytest.mark.asyncio
async def test_legacy_create_prompt_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await legacy_create_prompt(
            payload=schemas.LegacyPromptCreateRequest(name="alpha", content="body"),
            db=_LegacyCreatePromptDb(ConflictError("prompt already exists")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prompt already exists"


@pytest.mark.asyncio
async def test_legacy_create_prompt_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await legacy_create_prompt(
            payload=schemas.LegacyPromptCreateRequest(name="alpha", content="body"),
            db=_LegacyCreatePromptDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_create_prompt_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_CreatePromptDb(InputError("invalid prompt")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid prompt"


@pytest.mark.asyncio
async def test_create_prompt_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_CreatePromptDb(ConflictError("prompt already exists")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prompt already exists"


@pytest.mark.asyncio
async def test_create_prompt_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_CreatePromptDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error during prompt creation."


@pytest.mark.asyncio
async def test_create_prompt_sanitizes_failed_create_message():
    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_CreatePromptNoIdDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create prompt"


@pytest.mark.asyncio
async def test_update_prompt_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await update_prompt(
            prompt_identifier="alpha",
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_UpdatePromptDb(InputError("invalid prompt")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid prompt"


@pytest.mark.asyncio
async def test_update_prompt_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await update_prompt(
            prompt_identifier="alpha",
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_UpdatePromptDb(ConflictError("prompt already exists")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prompt already exists"


@pytest.mark.asyncio
async def test_update_prompt_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await update_prompt(
            prompt_identifier="alpha",
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_UpdatePromptDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error during prompt update."


@pytest.mark.asyncio
async def test_update_prompt_sanitizes_failed_update_message():
    with pytest.raises(HTTPException) as exc_info:
        await update_prompt(
            prompt_identifier="alpha",
            prompt_data=schemas.PromptCreate(name="alpha", details="body"),
            db=_UpdatePromptNoUuidDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Prompt update failed"


@pytest.mark.asyncio
async def test_record_prompt_usage_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await record_prompt_usage(
            prompt_identifier="alpha",
            db=_RecordPromptUsageDb(InputError("invalid prompt")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid prompt"


@pytest.mark.asyncio
async def test_record_prompt_usage_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await record_prompt_usage(
            prompt_identifier="alpha",
            db=_RecordPromptUsageDb(ConflictError("prompt locked")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prompt locked"


@pytest.mark.asyncio
async def test_record_prompt_usage_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await record_prompt_usage(
            prompt_identifier="alpha",
            db=_RecordPromptUsageDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_delete_prompt_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await delete_prompt(
            prompt_identifier="alpha",
            db=_DeletePromptDb(ConflictError("stale version")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "stale version"


@pytest.mark.asyncio
async def test_delete_prompt_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await delete_prompt(
            prompt_identifier="alpha",
            db=_DeletePromptDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_restore_prompt_version_maps_not_found_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await restore_prompt_version(
            prompt_identifier="alpha",
            version=2,
            db=_RestorePromptVersionDb(InputError("prompt not found")),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "prompt not found"


@pytest.mark.asyncio
async def test_restore_prompt_version_maps_input_error():
    with pytest.raises(HTTPException) as exc_info:
        await restore_prompt_version(
            prompt_identifier="alpha",
            version=2,
            db=_RestorePromptVersionDb(InputError("invalid version")),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid version"


@pytest.mark.asyncio
async def test_restore_prompt_version_maps_conflict_error():
    with pytest.raises(HTTPException) as exc_info:
        await restore_prompt_version(
            prompt_identifier="alpha",
            version=2,
            db=_RestorePromptVersionDb(ConflictError("stale version")),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "stale version"


@pytest.mark.asyncio
async def test_restore_prompt_version_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await restore_prompt_version(
            prompt_identifier="alpha",
            version=2,
            db=_RestorePromptVersionDb(DatabaseError("driver failed")),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database error."


@pytest.mark.asyncio
async def test_preview_prompt_api_maps_input_error(monkeypatch):
    monkeypatch.setattr(
        prompts_endpoint,
        "assemble_prompt_definition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(InputError("invalid preview")),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await preview_prompt_api(
            payload=schemas.StructuredPromptPreviewRequest(
                prompt_format="structured",
                prompt_schema_version=1,
                prompt_definition=_structured_prompt_definition_payload(),
                variables={"input": "SQLite FTS"},
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid preview"
