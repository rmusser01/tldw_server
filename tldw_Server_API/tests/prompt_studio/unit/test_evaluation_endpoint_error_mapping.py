import pytest
from fastapi import BackgroundTasks
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_evaluations as evaluations_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations import (
    create_evaluation,
    delete_evaluation,
    get_evaluation,
    list_evaluations,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import EvaluationCreate
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError


pytestmark = pytest.mark.unit


class _FakePsLogger:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "driver failed",
    "driver exploded",
    "/private/tmp/prompt-studio.db",
)


def _assert_sanitized_endpoint_error_log(
    ps_logger: _FakePsLogger,
    expected_message: str,
) -> None:
    assert ps_logger.exception_calls == []
    assert ps_logger.error_calls

    matching_messages = [args[0] for args, _kwargs in ps_logger.error_calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(ps_logger.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_debug_log(
    logger_stub: _FakePsLogger,
    expected_message: str,
) -> None:
    assert logger_stub.debug_calls
    assert ((expected_message,), {}) in logger_stub.debug_calls
    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _patch_prompt_studio_request_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakePsLogger:
    ps_logger = _FakePsLogger()
    monkeypatch.setattr(
        evaluations_endpoint,
        "ensure_request_id",
        lambda _request: "req-1",
        raising=True,
    )
    monkeypatch.setattr(
        evaluations_endpoint,
        "ensure_traceparent",
        lambda _request: "tp-1",
        raising=True,
    )
    monkeypatch.setattr(
        evaluations_endpoint,
        "get_ps_logger",
        lambda **_kwargs: ps_logger,
        raising=True,
    )
    return ps_logger


class _BrokenConnectionDb:
    def get_connection(self):
        raise DatabaseError("driver failed /private/tmp/prompt-studio.db")


class _BrokenEvaluationManager:
    def __init__(self, *_args, **_kwargs):
        pass

    def list_evaluations(self, *_args, **_kwargs):
        raise DatabaseError("driver failed /private/tmp/prompt-studio.db")


class _BrokenCreateEvaluationManager:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run_evaluation_async(self, *_args, **_kwargs):
        raise DatabaseError("driver failed /private/tmp/prompt-studio.db")


class _UnexpectedConnectionDb:
    def get_connection(self):
        raise RuntimeError("driver exploded /private/tmp/prompt-studio.db")


class _ColumnCheckFailureCursor:
    def __init__(self):
        self.rowcount = 0
        self._execute_count = 0

    def execute(self, *_args, **_kwargs):
        self._execute_count += 1
        if self._execute_count == 1:
            raise RuntimeError("driver exploded /private/tmp/prompt-studio.db")
        self.rowcount = 1

    def fetchall(self):
        return []


class _ColumnCheckFailureConnection:
    def __init__(self):
        self.cursor_instance = _ColumnCheckFailureCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True


class _ColumnCheckFailureDb:
    backend_type = "sqlite"
    backend = None

    def __init__(self):
        self.connection = _ColumnCheckFailureConnection()

    def get_connection(self):
        return self.connection


class _UnexpectedEvaluationManager:
    def __init__(self, *_args, **_kwargs):
        pass

    def list_evaluations(self, *_args, **_kwargs):
        raise RuntimeError("driver exploded /private/tmp/prompt-studio.db")


class _UnexpectedCreateEvaluationManager:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run_evaluation_async(self, *_args, **_kwargs):
        raise RuntimeError("driver exploded /private/tmp/prompt-studio.db")


class _NoByokResolution:
    api_key = None
    app_config = None
    uses_byok = False

    async def touch_last_used(self):
        return None


@pytest.mark.asyncio
async def test_create_evaluation_maps_database_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)
    monkeypatch.setattr(
        evaluations_endpoint,
        "EvaluationManager",
        _BrokenCreateEvaluationManager,
        raising=True,
    )
    monkeypatch.setattr(
        evaluations_endpoint,
        "_is_prompt_studio_test_mode",
        lambda: True,
        raising=True,
    )

    async def _fake_resolve_byok_credentials(*_args, **_kwargs):
        return _NoByokResolution()

    monkeypatch.setattr(
        evaluations_endpoint,
        "resolve_byok_credentials",
        _fake_resolve_byok_credentials,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_evaluation(
            evaluation=EvaluationCreate(
                project_id=42,
                prompt_id=7,
                name="Broken Eval",
                test_case_ids=[],
            ),
            background_tasks=BackgroundTasks(),
            request=object(),
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create evaluation"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to create evaluation")


@pytest.mark.asyncio
async def test_list_evaluations_maps_database_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)
    monkeypatch.setattr(
        evaluations_endpoint,
        "EvaluationManager",
        _BrokenEvaluationManager,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_evaluations(
            request=object(),
            project_id=42,
            prompt_id=None,
            limit=100,
            offset=0,
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list evaluations"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to list evaluations")


@pytest.mark.asyncio
async def test_get_evaluation_maps_database_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await get_evaluation(
            evaluation_id=42,
            request=object(),
            db=_BrokenConnectionDb(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get evaluation"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to get evaluation")


@pytest.mark.asyncio
async def test_delete_evaluation_maps_database_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await delete_evaluation(
            evaluation_id=42,
            request=object(),
            db=_BrokenConnectionDb(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete evaluation"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to delete evaluation")


@pytest.mark.asyncio
async def test_delete_evaluation_column_check_failure_log_is_sanitized(monkeypatch):
    logger_stub = _FakePsLogger()
    monkeypatch.setattr(evaluations_endpoint, "logger", logger_stub, raising=True)
    db = _ColumnCheckFailureDb()

    result = await delete_evaluation(
        evaluation_id=42,
        request=object(),
        db=db,
        user_context={"user_id": "tester"},
    )

    assert result == {"message": "Evaluation 42 deleted successfully"}
    assert db.connection.committed is True
    _assert_sanitized_debug_log(
        logger_stub,
        "Failed to check prompt_studio_evaluations columns",
    )


@pytest.mark.asyncio
async def test_create_evaluation_sanitizes_unexpected_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)
    monkeypatch.setattr(
        evaluations_endpoint,
        "EvaluationManager",
        _UnexpectedCreateEvaluationManager,
        raising=True,
    )
    monkeypatch.setattr(
        evaluations_endpoint,
        "_is_prompt_studio_test_mode",
        lambda: True,
        raising=True,
    )

    async def _fake_resolve_byok_credentials(*_args, **_kwargs):
        return _NoByokResolution()

    monkeypatch.setattr(
        evaluations_endpoint,
        "resolve_byok_credentials",
        _fake_resolve_byok_credentials,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_evaluation(
            evaluation=EvaluationCreate(
                project_id=42,
                prompt_id=7,
                name="Broken Eval",
                test_case_ids=[],
            ),
            background_tasks=BackgroundTasks(),
            request=object(),
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create evaluation"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to create evaluation")


@pytest.mark.asyncio
async def test_list_evaluations_sanitizes_unexpected_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)
    monkeypatch.setattr(
        evaluations_endpoint,
        "EvaluationManager",
        _UnexpectedEvaluationManager,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_evaluations(
            request=object(),
            project_id=42,
            prompt_id=None,
            limit=100,
            offset=0,
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list evaluations"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to list evaluations")


@pytest.mark.asyncio
async def test_get_evaluation_sanitizes_unexpected_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await get_evaluation(
            evaluation_id=42,
            request=object(),
            db=_UnexpectedConnectionDb(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get evaluation"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to get evaluation")


@pytest.mark.asyncio
async def test_delete_evaluation_sanitizes_unexpected_error(monkeypatch):
    ps_logger = _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await delete_evaluation(
            evaluation_id=42,
            request=object(),
            db=_UnexpectedConnectionDb(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete evaluation"
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to delete evaluation")
