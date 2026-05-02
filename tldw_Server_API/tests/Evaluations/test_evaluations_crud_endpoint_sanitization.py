import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_crud
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    CreateEvaluationRequest,
    EvaluationSpec,
    EvaluationType,
    UpdateEvaluationRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


_CRUD_SENSITIVE_MARKERS = (
    "crud header leaked",
    "crud lookup leaked",
    "crud record leaked",
    "evaluation backend leaked",
    "/private/evals-crud.db",
    "existing-eval-secret",
    "existing-run-secret",
    "idem-key-secret",
    "sensitive-eval-id",
    "sensitive-run-id",
)


class _LoggerStub:
    def __init__(self):
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


class _ExplodingEvaluationService:
    async def create_evaluation(self, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def list_evaluations(self, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def get_evaluation(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def update_evaluation(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def delete_evaluation(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def create_run(self, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def list_runs(self, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def get_run(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")

    async def cancel_run(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("evaluation backend leaked /private/evals-crud.db")


def _evaluation_row(eval_id: str = "sensitive-eval-id") -> dict[str, object]:
    return {
        "id": eval_id,
        "name": "evaluation_sanitizer",
        "description": None,
        "eval_type": "model_graded",
        "eval_spec": {"metrics": ["faithfulness"]},
        "dataset_id": None,
        "created": 1,
        "created_by": "tenant-user",
        "metadata": {},
    }


def _run_row(run_id: str = "sensitive-run-id") -> dict[str, object]:
    return {
        "id": run_id,
        "eval_id": "eval-1",
        "status": "pending",
        "target_model": "gpt-4o-mini",
        "created": 1,
    }


class _RaisingHeaders:
    def __setitem__(self, _key: str, _value: str) -> None:
        raise ValueError("crud header leaked /private/evals-crud.db")


class _ResponseWithRaisingHeaders:
    headers = _RaisingHeaders()


class _EvaluationLookupDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("crud lookup leaked /private/evals-crud.db")

    def record_idempotency(self, *_args: object, **_kwargs: object) -> None:
        return None


class _EvaluationLookupDebugService:
    def __init__(self):
        self.db = _EvaluationLookupDebugDB()

    async def create_evaluation(self, **_kwargs: object) -> dict[str, object]:
        return _evaluation_row()


class _EvaluationReplayDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> str:
        return "existing-eval-secret"


class _EvaluationReplayDebugService:
    def __init__(self):
        self.db = _EvaluationReplayDebugDB()

    async def get_evaluation(self, eval_id: str, **_kwargs: object) -> dict[str, object]:
        return _evaluation_row(eval_id)


class _EvaluationRecordDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> None:
        return None

    def record_idempotency(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("crud record leaked /private/evals-crud.db")


class _EvaluationRecordDebugService:
    def __init__(self):
        self.db = _EvaluationRecordDebugDB()

    async def create_evaluation(self, **_kwargs: object) -> dict[str, object]:
        return _evaluation_row()


class _RunLookupDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("crud lookup leaked /private/evals-crud.db")

    def record_idempotency(self, *_args: object, **_kwargs: object) -> None:
        return None


class _RunLookupDebugService:
    def __init__(self):
        self.db = _RunLookupDebugDB()

    async def create_run(self, **_kwargs: object) -> dict[str, object]:
        return _run_row()


class _RunReplayDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> str:
        return "existing-run-secret"


class _RunReplayDebugService:
    def __init__(self):
        self.db = _RunReplayDebugDB()

    async def get_run(self, run_id: str, **_kwargs: object) -> dict[str, object]:
        return _run_row(run_id)


class _RunRecordDebugDB:
    def lookup_idempotency(self, *_args: object, **_kwargs: object) -> None:
        return None

    def record_idempotency(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("crud record leaked /private/evals-crud.db")


class _RunRecordDebugService:
    def __init__(self):
        self.db = _RunRecordDebugDB()

    async def create_run(self, **_kwargs: object) -> dict[str, object]:
        return _run_row()


class _ListingEvaluationService:
    async def list_evaluations(self, **kwargs: object) -> tuple[list[dict[str, object]], bool]:
        assert kwargs["limit"] == 2
        assert kwargs["after"] == "eval-cursor"
        assert kwargs["eval_type"] == "model_graded"
        assert kwargs["created_by"] == "tenant-user"
        return (
            [
                _evaluation_row("eval-2"),
                _evaluation_row("eval-1"),
            ],
            True,
        )


class _ListingRunService:
    async def list_runs(self, **kwargs: object) -> tuple[list[dict[str, object]], bool]:
        assert kwargs["eval_id"] == "eval-1"
        assert kwargs["status"] == "pending"
        assert kwargs["limit"] == 2
        assert kwargs["after"] == "run-cursor"
        assert kwargs["created_by"] == "tenant-user"
        return (
            [
                _run_row("run-2"),
                _run_row("run-1"),
            ],
            True,
        )


def _user() -> User:
    return User(id="tenant-user", username="tenant", email=None, is_active=True)


def _patch_failing_service(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_crud, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_crud,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: _ExplodingEvaluationService(),
    )
    return logger_stub


def _create_request() -> CreateEvaluationRequest:
    return CreateEvaluationRequest(
        name="evaluation_sanitizer",
        eval_type=EvaluationType.MODEL_GRADED,
        eval_spec=EvaluationSpec(metrics=["faithfulness"]),
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
    for marker in _CRUD_SENSITIVE_MARKERS:
        assert marker not in rendered


def _assert_sanitized_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [(expected_message, (), {})]
    rendered = " ".join(
        [
            logger_stub.errors[0][0],
            *(str(arg) for arg in logger_stub.errors[0][1]),
        ]
    )
    for marker in _CRUD_SENSITIVE_MARKERS:
        assert marker not in rendered


def _assert_sanitized_debug(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debugs == [(expected_message, (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _CRUD_SENSITIVE_MARKERS:
        assert marker not in rendered


def _patch_service(monkeypatch: pytest.MonkeyPatch, service: object) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evaluations_crud, "logger", logger_stub)
    monkeypatch.setattr(
        evaluations_crud,
        "get_unified_evaluation_service_for_user",
        lambda _user_id: service,
    )
    return logger_stub


async def test_create_evaluation_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.create_evaluation(
            eval_request=_create_request(),
            user_id=object(),
            current_user=_user(),
            idempotency_key=None,
            response=Response(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to create evaluation: An error occurred during evaluation creation",
    )
    _assert_sanitized_log(logger_stub, "Failed to create evaluation")


async def test_create_evaluation_sanitizes_idempotency_lookup_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _patch_service(monkeypatch, _EvaluationLookupDebugService())

    response = await evaluations_crud.create_evaluation(
        eval_request=_create_request(),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=Response(),
    )

    assert response.id == "sensitive-eval-id"
    _assert_sanitized_debug(logger_stub, "evaluations_crud: evaluation idempotency lookup failed")


async def test_create_evaluation_sanitizes_idempotent_replay_header_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _patch_service(monkeypatch, _EvaluationReplayDebugService())

    response = await evaluations_crud.create_evaluation(
        eval_request=_create_request(),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=_ResponseWithRaisingHeaders(),
    )

    assert response.id == "existing-eval-secret"
    _assert_sanitized_debug(logger_stub, "evaluations_crud: failed to set evaluation idempotency replay headers")


async def test_create_evaluation_sanitizes_idempotency_record_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _patch_service(monkeypatch, _EvaluationRecordDebugService())

    response = await evaluations_crud.create_evaluation(
        eval_request=_create_request(),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=Response(),
    )

    assert response.id == "sensitive-eval-id"
    _assert_sanitized_debug(logger_stub, "evaluations_crud: failed to record evaluation idempotency")


async def test_list_evaluations_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.list_evaluations(current_user=_user())

    _assert_sanitized_detail(
        exc_info,
        "Failed to list evaluations: An error occurred during listing evaluations",
    )
    _assert_sanitized_log(logger_stub, "Failed to list evaluations")


async def test_list_evaluations_includes_canonical_cursor_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_service(monkeypatch, _ListingEvaluationService())

    response = await evaluations_crud.list_evaluations(
        limit=2,
        after="eval-cursor",
        eval_type="model_graded",
        current_user=_user(),
    )

    payload = response.model_dump(mode="json")
    assert payload["last_id"] == "eval-1"
    assert payload["next_cursor"] == "eval-1"
    assert payload["pagination"] == {
        "mode": "cursor",
        "limit": 2,
        "cursor": "eval-cursor",
        "next_cursor": "eval-1",
        "has_more": True,
    }


async def test_get_evaluation_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.get_evaluation(
            eval_id="eval-1",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to get evaluation: An error occurred during retrieving evaluation",
    )
    _assert_sanitized_log(logger_stub, "Failed to get evaluation")


async def test_update_evaluation_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.update_evaluation(
            eval_id="eval-1",
            update_request=UpdateEvaluationRequest(description="updated"),
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to update evaluation: An error occurred during updating evaluation",
    )
    _assert_sanitized_log(logger_stub, "Failed to update evaluation")


async def test_delete_evaluation_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.delete_evaluation(
            eval_id="eval-1",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to delete evaluation: An error occurred during deleting evaluation",
    )
    _assert_sanitized_log(logger_stub, "Failed to delete evaluation")


async def test_create_run_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.create_run(
            eval_id="eval-1",
            request=evaluations_crud.CreateRunSimpleRequest(target_model="gpt-4o-mini"),
            user_id=object(),
            current_user=_user(),
            idempotency_key=None,
            response=Response(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to create run: An error occurred during creating run",
    )
    _assert_sanitized_log(logger_stub, "Failed to create run")


async def test_create_run_sanitizes_idempotency_lookup_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_service(monkeypatch, _RunLookupDebugService())

    response = await evaluations_crud.create_run(
        eval_id="eval-1",
        request=evaluations_crud.CreateRunSimpleRequest(target_model="gpt-4o-mini"),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=Response(),
    )

    assert response.id == "sensitive-run-id"
    _assert_sanitized_debug(logger_stub, "evaluations_crud: run idempotency lookup failed")


async def test_create_run_sanitizes_idempotent_replay_header_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _patch_service(monkeypatch, _RunReplayDebugService())

    response = await evaluations_crud.create_run(
        eval_id="eval-1",
        request=evaluations_crud.CreateRunSimpleRequest(target_model="gpt-4o-mini"),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=_ResponseWithRaisingHeaders(),
    )

    assert response.id == "existing-run-secret"
    _assert_sanitized_debug(logger_stub, "evaluations_crud: failed to set run idempotency replay headers")


async def test_create_run_sanitizes_idempotency_record_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_service(monkeypatch, _RunRecordDebugService())

    response = await evaluations_crud.create_run(
        eval_id="eval-1",
        request=evaluations_crud.CreateRunSimpleRequest(target_model="gpt-4o-mini"),
        user_id=object(),
        current_user=_user(),
        idempotency_key="idem-key-secret",
        response=Response(),
    )

    assert response.id == "sensitive-run-id"
    _assert_sanitized_debug(logger_stub, "evaluations_crud: failed to record run idempotency")


async def test_list_runs_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.list_runs(
            eval_id="eval-1",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to list runs: An error occurred during listing runs",
    )
    _assert_sanitized_log(logger_stub, "Failed to list runs")


async def test_list_runs_includes_canonical_cursor_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_service(monkeypatch, _ListingRunService())

    response = await evaluations_crud.list_runs(
        eval_id="eval-1",
        limit=2,
        after="run-cursor",
        run_status="pending",
        current_user=_user(),
    )

    payload = response.model_dump(mode="json")
    assert payload["last_id"] == "run-1"
    assert payload["next_cursor"] == "run-1"
    assert payload["pagination"] == {
        "mode": "cursor",
        "limit": 2,
        "cursor": "run-cursor",
        "next_cursor": "run-1",
        "has_more": True,
    }


async def test_get_run_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.get_run(
            run_id="run-1",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to get run: An error occurred during retrieving run",
    )
    _assert_sanitized_log(logger_stub, "Failed to get run")


async def test_cancel_run_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _patch_failing_service(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_crud.cancel_run(
            run_id="run-1",
            current_user=_user(),
        )

    _assert_sanitized_detail(
        exc_info,
        "Failed to cancel run: An error occurred during cancelling run",
    )
    _assert_sanitized_log(logger_stub, "Failed to cancel run")
