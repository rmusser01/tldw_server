import pytest
from fastapi import BackgroundTasks, HTTPException
from loguru import logger

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
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError

pytestmark = pytest.mark.unit


def test_prompt_studio_maps_revoked_credential_scope_to_forbidden() -> None:
    mapped = evaluations_endpoint._prompt_studio_credential_http_exception(
        ByokResolutionError("credential_scope_revoked", "openai")
    )

    assert mapped.status_code == 403
    assert mapped.detail["error_code"] == "credential_scope_revoked"


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
    "byok-secret-sentinel",
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


def _assert_detached_http_exception(exc: HTTPException) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None


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
        try:
            raise RuntimeError("driver failed /private/tmp/prompt-studio.db")
        except RuntimeError as private_error:
            raise DatabaseError("driver failed /private/tmp/prompt-studio.db") from private_error


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
        try:
            raise ValueError("driver exploded /private/tmp/prompt-studio.db")
        except ValueError as private_error:
            raise RuntimeError("driver exploded /private/tmp/prompt-studio.db") from private_error


class _NoProviderCredentials:
    api_key = None
    app_config = None
    credentials_resolved = True


class _NoCredentialRuntime:
    async def resolve(self, *_args, **_kwargs):
        return _NoProviderCredentials()

    async def mark_used(self, *_args, **_kwargs):
        return None

    async def close(self):
        return None


def _patch_no_credentials_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluations_endpoint,
        "ProviderCredentialRuntime",
        lambda **_kwargs: _NoCredentialRuntime(),
        raising=True,
    )
    monkeypatch.setattr(
        evaluations_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (None, [], [], False),
        raising=True,
    )


@pytest.mark.asyncio
async def test_create_evaluation_detaches_byok_error_graph(monkeypatch):
    logs: list[str] = []

    class _BoundEvaluationDb:
        @staticmethod
        def get_prompt_with_project(
            prompt_id: int,
            include_deleted: bool = False,
        ) -> dict[str, int]:
            return {"id": prompt_id, "project_id": 42}

        @staticmethod
        def get_project(project_id: int) -> dict[str, str | int]:
            return {"id": project_id, "user_id": "1"}

    class _ChainedByokRuntime:
        async def resolve(self, *_args, **_kwargs):
            try:
                raise RuntimeError(
                    "byok-secret-sentinel /private/tmp/prompt-studio.db"
                )
            except RuntimeError as private_error:
                raise ByokResolutionError(
                    "credential_store_unavailable",
                    "openai",
                ) from private_error

        async def close(self):
            return None

    monkeypatch.setattr(
        evaluations_endpoint,
        "ProviderCredentialRuntime",
        lambda **_kwargs: _ChainedByokRuntime(),
        raising=True,
    )
    monkeypatch.setattr(
        evaluations_endpoint,
        "derive_trusted_credential_scope",
        lambda *_args: (1, [], [], False),
        raising=True,
    )

    sink_id = logger.add(logs.append, format="{message}")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await create_evaluation(
                evaluation=EvaluationCreate(
                    project_id=42,
                    prompt_id=7,
                    config={"provider": "openai", "model_name": "model-a"},
                ),
                background_tasks=BackgroundTasks(),
                request=object(),
                db=_BoundEvaluationDb(),
                user_context={"user_id": "1", "is_admin": False},
            )
    finally:
        logger.remove(sink_id)

    assert exc_info.value.status_code == 503
    assert "byok-secret-sentinel" not in repr(exc_info.value.detail)
    assert "byok-secret-sentinel" not in "".join(logs)
    _assert_detached_http_exception(exc_info.value)


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

    _patch_no_credentials_runtime(monkeypatch)

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    try:
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
    finally:
        logger.remove(sink_id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create evaluation"
    _assert_detached_http_exception(exc_info.value)
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to create evaluation")
    assert all(marker not in "".join(logs) for marker in _SENSITIVE_MARKERS)


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
    _assert_detached_http_exception(exc_info.value)
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
    _assert_detached_http_exception(exc_info.value)
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
    _assert_detached_http_exception(exc_info.value)
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

    _patch_no_credentials_runtime(monkeypatch)

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
    _assert_detached_http_exception(exc_info.value)
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
    _assert_detached_http_exception(exc_info.value)
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
    _assert_detached_http_exception(exc_info.value)
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
    _assert_detached_http_exception(exc_info.value)
    _assert_sanitized_endpoint_error_log(ps_logger, "Failed to delete evaluation")
