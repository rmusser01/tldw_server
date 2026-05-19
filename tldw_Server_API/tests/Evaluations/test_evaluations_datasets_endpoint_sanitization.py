import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_datasets
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    CreateDatasetRequest,
    DatasetSample,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


_DATASET_SENSITIVE_MARKERS = (
    "dataset backend leaked",
    "header assignment leaked",
    "idempotency lookup leaked",
    "idempotency record leaked",
    "existing-dataset-secret",
    "idem-key-secret",
    "/private/evals-datasets.db",
    "sensitive-dataset-id",
)


class _LoggerStub:
    def __init__(self):
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.warnings: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append((message, args, kwargs))


class _ExplodingDatasetDB:
    def _raise(self) -> None:
        raise RuntimeError("dataset backend leaked /private/evals-datasets.db")

    def list_datasets(self, **_kwargs: object) -> None:
        self._raise()

    def get_dataset(self, *_args: object, **_kwargs: object) -> None:
        self._raise()


class _ExplodingDatasetService:
    def __init__(self):
        self.db = _ExplodingDatasetDB()

    async def create_dataset(self, **_kwargs: object) -> None:
        raise RuntimeError("dataset backend leaked /private/evals-datasets.db")

    async def delete_dataset(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("dataset backend leaked /private/evals-datasets.db")


class _IdempotencyRecordWarningDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> None:
        return None

    def record_idempotency(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("idempotency record leaked /private/evals-datasets.db")


class _IdempotencyRecordWarningService:
    def __init__(self):
        self.db = _IdempotencyRecordWarningDB()

    async def create_dataset(self, **_kwargs: object) -> str:
        return "sensitive-dataset-id"

    async def get_dataset(self, dataset_id: str, **_kwargs: object) -> dict[str, object]:
        return {
            "id": dataset_id,
            "name": "dataset",
            "description": "",
            "sample_count": 1,
            "samples": [{"input": {"question": "What changed?"}, "expected": "sanitized", "metadata": {}}],
            "created_by": "tenant-user",
            "metadata": {},
        }


class _IdempotencyLookupDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("idempotency lookup leaked /private/evals-datasets.db")

    def record_idempotency(self, *_args: object, **_kwargs: object) -> None:
        return None


class _IdempotencyLookupDebugService:
    def __init__(self):
        self.db = _IdempotencyLookupDebugDB()

    async def create_dataset(self, **_kwargs: object) -> str:
        return "sensitive-dataset-id"

    async def get_dataset(self, dataset_id: str, **_kwargs: object) -> dict[str, object]:
        return {
            "id": dataset_id,
            "name": "dataset",
            "description": "",
            "sample_count": 1,
            "samples": [{"input": {"question": "What changed?"}, "expected": "sanitized", "metadata": {}}],
            "created_by": "tenant-user",
            "metadata": {},
        }


class _IdempotencyReplayDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> str:
        return "existing-dataset-secret"


class _IdempotencyReplayDebugService:
    def __init__(self):
        self.db = _IdempotencyReplayDebugDB()

    async def get_dataset(self, dataset_id: str, **_kwargs: object) -> dict[str, object]:
        return {
            "id": dataset_id,
            "name": "dataset",
            "description": "",
            "sample_count": 1,
            "samples": [{"input": {"question": "What changed?"}, "expected": "sanitized", "metadata": {}}],
            "created_by": "tenant-user",
            "metadata": {},
        }


class _RaisingHeaders:
    def __setitem__(self, _key: str, _value: str) -> None:
        raise ValueError("header assignment leaked /private/evals-datasets.db")


class _ResponseWithRaisingHeaders:
    headers = _RaisingHeaders()


def _user() -> User:
    return User(id="tenant-user", username="tenant", email=None, is_active=True)


def _patch_failing_service(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_datasets, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_datasets,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _ExplodingDatasetService(),
    )
    return logger_stub


def _dataset_request() -> CreateDatasetRequest:
    return CreateDatasetRequest(
        name="dataset",
        samples=[DatasetSample(input={"question": "What changed?"}, expected="sanitized")],
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
    for marker in _DATASET_SENSITIVE_MARKERS:
        assert marker not in rendered


def _assert_sanitized_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [(expected_message, (), {})]
    rendered = " ".join(
        [
            logger_stub.errors[0][0],
            *(str(arg) for arg in logger_stub.errors[0][1]),
        ]
    )
    for marker in _DATASET_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_create_dataset_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_datasets.create_dataset(
            dataset_request=_dataset_request(),
            user_id=object(),
            current_user=_user(),
            idempotency_key=None,
            response=Response(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to create dataset: An error occurred during creating dataset",
    )
    _assert_sanitized_log(logger_stub, "Failed to create dataset")


async def test_create_dataset_sanitizes_idempotency_record_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_datasets, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_datasets,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _IdempotencyRecordWarningService(),
    )

    response = await evaluations_datasets.create_dataset(
        dataset_request=_dataset_request(),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key",
        response=Response(),
    )

    assert response.id == "sensitive-dataset-id"
    assert logger_stub.warnings == [("Failed to record dataset idempotency key", (), {})]
    rendered = repr(logger_stub.warnings)
    for marker in _DATASET_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_create_dataset_sanitizes_idempotency_lookup_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_datasets, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_datasets,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _IdempotencyLookupDebugService(),
    )

    response = await evaluations_datasets.create_dataset(
        dataset_request=_dataset_request(),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=Response(),
    )

    assert response.id == "sensitive-dataset-id"
    assert logger_stub.debugs == [("Dataset idempotency lookup failed", (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _DATASET_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_create_dataset_sanitizes_idempotent_replay_header_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_datasets, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_datasets,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _IdempotencyReplayDebugService(),
    )

    response = await evaluations_datasets.create_dataset(
        dataset_request=_dataset_request(),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=_ResponseWithRaisingHeaders(),
    )

    assert response.id == "existing-dataset-secret"
    assert logger_stub.debugs == [("Failed to set dataset idempotency replay headers", (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _DATASET_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_list_datasets_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_datasets.list_datasets(
            user_id=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to list datasets: An error occurred during listing datasets",
    )
    _assert_sanitized_log(logger_stub, "Failed to list datasets")


async def test_get_dataset_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_datasets.get_dataset(
            dataset_id="dataset-1",
            user_id=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to get dataset: An error occurred during retrieving dataset",
    )
    _assert_sanitized_log(logger_stub, "Failed to get dataset")


async def test_delete_dataset_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_datasets.delete_dataset(
            dataset_id="dataset-1",
            user_id=object(),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to delete dataset: An error occurred during deleting dataset",
    )
    _assert_sanitized_log(logger_stub, "Failed to delete dataset")
