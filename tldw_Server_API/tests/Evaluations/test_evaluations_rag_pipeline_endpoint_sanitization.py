import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_rag_pipeline
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import PipelinePresetCreate
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


_PIPELINE_SENSITIVE_MARKERS = (
    "pipeline backend leaked",
    "adapter delete leaked",
    "/private/rag-pipeline.db",
    "/private/rag-vector.db",
    "expired_collection_secret",
)


class _LoggerStub:
    def __init__(self):
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.warnings: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append((message, args, kwargs))


class _ExplodingPipelineDB:
    def _raise(self) -> None:
        raise RuntimeError("pipeline backend leaked /private/rag-pipeline.db")

    def upsert_pipeline_preset(self, *_args: object, **_kwargs: object) -> None:
        self._raise()

    def list_pipeline_presets(self, **_kwargs: object) -> None:
        self._raise()

    def get_pipeline_preset(self, *_args: object, **_kwargs: object) -> None:
        self._raise()

    def delete_pipeline_preset(self, *_args: object, **_kwargs: object) -> None:
        self._raise()

    def list_expired_ephemeral_collections(self) -> None:
        self._raise()


class _ExplodingPipelineService:
    def __init__(self):
        self.db = _ExplodingPipelineDB()


class _CleanupItemDB:
    def list_expired_ephemeral_collections(self) -> list[str]:
        return ["expired_collection_secret"]

    def mark_ephemeral_deleted(self, _name: str) -> None:
        raise AssertionError("failed delete should not be marked")


class _CleanupItemService:
    def __init__(self):
        self.db = _CleanupItemDB()


class _PresetListDB:
    def list_pipeline_presets(self, **kwargs: object) -> tuple[list[dict[str, object]], int]:
        assert kwargs == {"limit": 2, "offset": 0, "user_id": "tenant-user"}
        return (
            [
                {"name": "preset_one", "config": {"retriever": {"top_k": 4}}},
                {"name": "preset_two", "config": {"retriever": {"top_k": 8}}},
            ],
            3,
        )


class _PresetListService:
    def __init__(self):
        self.db = _PresetListDB()


class _CleanupItemAdapter:
    async def initialize(self) -> None:
        return None

    async def delete_collection(self, _name: str) -> None:
        raise RuntimeError("adapter delete leaked /private/rag-vector.db")


def _user() -> User:
    return User(id="tenant-user", username="tenant", email=None, is_active=True)


def _patch_failing_service(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_rag_pipeline, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_rag_pipeline,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _ExplodingPipelineService(),
    )
    return logger_stub


def _preset() -> PipelinePresetCreate:
    return PipelinePresetCreate(
        name="sanitizer_preset",
        config={"retriever": {"top_k": 4}},
    )


def _assert_sanitized_detail(exc_info: pytest.ExceptionInfo[HTTPException], expected_message: str) -> None:
    assert exc_info.value.status_code == 500
    detail = exc_info.value.detail
    assert detail == {
        "error": {
            "message": expected_message,
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }
    rendered = str(detail)
    for marker in _PIPELINE_SENSITIVE_MARKERS:
        assert marker not in rendered


def _assert_sanitized_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [(expected_message, (), {})]
    rendered = " ".join(
        [
            logger_stub.errors[0][0],
            *(str(arg) for arg in logger_stub.errors[0][1]),
        ]
    )
    for marker in _PIPELINE_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_create_or_update_pipeline_preset_sanitizes_backend_fallback_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_rag_pipeline.create_or_update_pipeline_preset(
            preset=_preset(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to save preset: An error occurred during save_preset",
    )
    _assert_sanitized_log(logger_stub, "Failed to save preset")


async def test_list_pipeline_presets_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_rag_pipeline.list_pipeline_presets(current_user=_user())

    _assert_sanitized_detail(
        exc_info,
        "Failed to list presets: An error occurred during list_presets",
    )
    _assert_sanitized_log(logger_stub, "Failed to list presets")


async def test_list_pipeline_presets_includes_canonical_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluations_rag_pipeline,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _PresetListService(),
    )

    response = await evaluations_rag_pipeline.list_pipeline_presets(
        limit=2,
        offset=0,
        current_user=_user(),
    )

    assert response.total == 3
    assert response.limit == 2
    assert response.offset == 0
    assert response.pagination.model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": 3,
        "has_more": True,
        "next_offset": 2,
    }
    assert response.has_more is True
    assert response.next_offset == 2


async def test_get_pipeline_preset_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_rag_pipeline.get_pipeline_preset(
            name="sanitizer_preset",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to get preset: An error occurred during get_preset",
    )
    _assert_sanitized_log(logger_stub, "Failed to get preset")


async def test_delete_pipeline_preset_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_rag_pipeline.delete_pipeline_preset(
            name="sanitizer_preset",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to delete preset: An error occurred during delete_preset",
    )
    _assert_sanitized_log(logger_stub, "Failed to delete preset")


async def test_cleanup_ephemeral_collections_sanitizes_backend_fallback_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_rag_pipeline.cleanup_ephemeral_collections(current_user=_user())

    _assert_sanitized_detail(
        exc_info,
        "Cleanup failed: An error occurred during cleanup",
    )
    _assert_sanitized_log(logger_stub, "Cleanup failed")


async def test_cleanup_ephemeral_collections_sanitizes_per_item_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_rag_pipeline, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_rag_pipeline,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _CleanupItemService(),
    )
    monkeypatch.setattr(
        evaluations_rag_pipeline,
        "create_from_settings_for_user",
        lambda *_args, **_kwargs: _CleanupItemAdapter(),
    )

    response = await evaluations_rag_pipeline.cleanup_ephemeral_collections(current_user=_user())

    assert response.expired_count == 1
    assert response.deleted_count == 0
    assert response.errors == ["Collection cleanup failed"]
    assert logger_stub.warnings == [("Failed to delete expired collection", (), {})]
    rendered = repr(logger_stub.warnings)
    for marker in _PIPELINE_SENSITIVE_MARKERS:
        assert marker not in rendered
