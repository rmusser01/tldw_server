import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as optimization_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization import (
    create_optimization,
    OptimizationIterationCreate,
    add_optimization_iteration,
    cancel_optimization,
    compare_strategies,
    get_optimization,
    get_optimization_history,
    list_optimization_iterations,
    list_optimizations,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationCreate,
    OptimizationConfig,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest,
)
from tldw_Server_API.app.core.Logging import log_context as log_context_module
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError


pytestmark = pytest.mark.unit


class _BrokenListOptimizationsDb:
    def list_optimizations(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _BrokenListOptimizationsUnexpectedDb:
    def list_optimizations(self, *_args, **_kwargs):
        raise ValueError("prompt studio backend exploded at /private/prompt-studio.db")


class _BrokenOptimizationLookupDb:
    def __init__(self, message: str = "driver failed") -> None:
        self.message = message

    def get_optimization(self, *_args, **_kwargs):
        raise DatabaseError(self.message)


class _BrokenOptimizationLookupUnexpectedDb:
    def get_optimization(self, *_args, **_kwargs):
        raise ValueError("prompt studio backend exploded at /private/prompt-studio.db")


class _BrokenCancelOptimizationUnexpectedDb:
    def get_optimization(self, *_args, **_kwargs):
        raise ValueError("boom")


class _BrokenCompareStrategiesDb:
    client_id = "client-1"

    def get_prompt_with_project(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _BrokenCompareStrategiesUnexpectedDb:
    client_id = "client-1"

    def get_prompt_with_project(self, *_args, **_kwargs):
        raise ValueError("prompt studio backend exploded at /private/prompt-studio.db")


class _BrokenCreateOptimizationDb:
    client_id = "client-1"

    def get_prompt_with_project(self, *_args, **_kwargs):
        return {"id": 12, "project_id": 7}

    def create_optimization(self, *_args, **_kwargs):
        raise DatabaseError("driver failed")


class _BrokenCreateOptimizationUnexpectedDb:
    client_id = "client-1"

    def get_prompt_with_project(self, *_args, **_kwargs):
        raise ValueError("prompt studio backend exploded at /private/prompt-studio.db")


class _FakePsLogger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class _CapturingPsLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.error_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))


class _EndpointLoggerStub:
    def __init__(self) -> None:
        self.errors: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))


def _assert_sanitized_endpoint_error_log(
    logger_stub: _EndpointLoggerStub,
    expected_message: str,
    raw_marker: str,
) -> None:
    assert logger_stub.errors
    args, kwargs = logger_stub.errors[-1]
    rendered = " ".join(str(arg) for arg in args)

    assert args == (expected_message,)
    assert raw_marker not in rendered
    assert "/private/" not in rendered
    assert raw_marker not in str(kwargs)
    assert "/private/" not in str(kwargs)


def _patch_prompt_studio_request_logging(
    monkeypatch: pytest.MonkeyPatch,
    *,
    logger_instance: object | None = None,
) -> dict[str, object]:
    captured_logger_kwargs: dict[str, object] = {}
    logger_instance = logger_instance or _FakePsLogger()
    for module in (log_context_module, optimization_endpoint):
        monkeypatch.setattr(
            module,
            "ensure_request_id",
            lambda _request: "req-1",
            raising=True,
        )
        monkeypatch.setattr(
            module,
            "ensure_traceparent",
            lambda _request: "tp-1",
            raising=True,
        )
        monkeypatch.setattr(
            module,
            "get_ps_logger",
            lambda **kwargs: captured_logger_kwargs.update(kwargs) or logger_instance,
            raising=True,
        )
    return captured_logger_kwargs


def _build_optimization_create_payload() -> OptimizationCreate:
    return OptimizationCreate(
        project_id=7,
        initial_prompt_id=12,
        optimization_config=OptimizationConfig(
            optimizer_type="iterative",
            target_metric="accuracy",
        ),
        name="Refine",
    )


@pytest.mark.asyncio
async def test_list_optimizations_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await list_optimizations(
            project_id=7,
            page=1,
            per_page=20,
            status_filter=None,
            _=True,
            db=_BrokenListOptimizationsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list optimizations"


@pytest.mark.asyncio
async def test_list_optimizations_database_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await list_optimizations(
            project_id=7,
            page=1,
            per_page=20,
            status_filter=None,
            _=True,
            db=_BrokenListOptimizationsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list optimizations"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error listing optimizations",
        "driver failed",
    )


@pytest.mark.asyncio
async def test_list_optimizations_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await list_optimizations(
            project_id=7,
            page=1,
            per_page=20,
            status_filter=None,
            _=True,
            db=_BrokenListOptimizationsUnexpectedDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list optimizations"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error listing optimizations",
        "prompt studio backend exploded",
    )


@pytest.mark.asyncio
async def test_create_optimization_maps_database_error(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_optimization(
            optimization_data=_build_optimization_create_payload(),
            request=object(),
            _=True,
            db=_BrokenCreateOptimizationDb(),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create optimization"


@pytest.mark.asyncio
async def test_create_optimization_sanitizes_database_error(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_optimization(
            optimization_data=_build_optimization_create_payload(),
            request=object(),
            _=True,
            db=_BrokenCreateOptimizationDb(),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
    )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create optimization"


@pytest.mark.asyncio
async def test_create_optimization_database_error_log_is_sanitized(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await create_optimization(
            optimization_data=_build_optimization_create_payload(),
            request=object(),
            _=True,
            db=_BrokenCreateOptimizationDb(),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create optimization"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error creating optimization",
        "driver failed",
    )


@pytest.mark.asyncio
async def test_create_optimization_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await create_optimization(
            optimization_data=_build_optimization_create_payload(),
            request=object(),
            _=True,
            db=_BrokenCreateOptimizationUnexpectedDb(),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create optimization"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error creating optimization",
        "prompt studio backend exploded",
    )


@pytest.mark.asyncio
async def test_get_optimization_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await get_optimization(
            optimization_id=42,
            db=_BrokenOptimizationLookupDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get optimization"


@pytest.mark.asyncio
async def test_get_optimization_database_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await get_optimization(
            optimization_id=42,
            db=_BrokenOptimizationLookupDb("driver failed at /private/prompt-studio.db"),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get optimization"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error fetching optimization",
        "driver failed",
    )


@pytest.mark.asyncio
async def test_get_optimization_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await get_optimization(
            optimization_id=42,
            db=_BrokenOptimizationLookupUnexpectedDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get optimization"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error getting optimization",
        "prompt studio backend exploded",
    )


@pytest.mark.asyncio
async def test_cancel_optimization_maps_database_error(monkeypatch):
    _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await cancel_optimization(
            request=object(),
            optimization_id=42,
            reason=None,
            db=_BrokenOptimizationLookupDb(),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to cancel optimization"


@pytest.mark.asyncio
async def test_cancel_optimization_logs_database_error_with_request_context(monkeypatch):
    capturing_logger = _CapturingPsLogger()
    logger_kwargs = _patch_prompt_studio_request_logging(
        monkeypatch,
        logger_instance=capturing_logger,
    )

    with pytest.raises(HTTPException):
        await cancel_optimization(
            request=object(),
            optimization_id=42,
            reason=None,
            db=_BrokenOptimizationLookupDb(),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert logger_kwargs == {
        "ps_component": "endpoint",
        "ps_job_kind": "optimization",
        "optimization_id": 42,
        "request_id": "req-1",
        "traceparent": "tp-1",
    }
    assert len(capturing_logger.error_calls) == 1
    log_args, log_kwargs = capturing_logger.error_calls[0]
    assert log_kwargs == {}
    assert log_args[0] == "Database error cancelling optimization %s: %s"
    assert log_args[1] == 42
    assert str(log_args[2]) == "driver failed"


@pytest.mark.asyncio
async def test_cancel_optimization_maps_unexpected_error(monkeypatch):
    capturing_logger = _CapturingPsLogger()
    logger_kwargs = _patch_prompt_studio_request_logging(
        monkeypatch,
        logger_instance=capturing_logger,
    )

    with pytest.raises(HTTPException) as exc_info:
        await cancel_optimization(
            request=object(),
            optimization_id=42,
            reason=None,
            db=_BrokenCancelOptimizationUnexpectedDb(),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to cancel optimization"
    assert logger_kwargs == {
        "ps_component": "endpoint",
        "ps_job_kind": "optimization",
        "optimization_id": 42,
        "request_id": "req-1",
        "traceparent": "tp-1",
    }
    assert len(capturing_logger.error_calls) == 1
    log_args, log_kwargs = capturing_logger.error_calls[0]
    assert log_kwargs == {}
    assert log_args[0] == "Unexpected error cancelling optimization %s: %s"
    assert log_args[1] == 42
    assert str(log_args[2]) == "boom"


@pytest.mark.asyncio
async def test_get_optimization_history_maps_database_error(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await get_optimization_history(
            optimization_id=42,
            db=_BrokenOptimizationLookupDb(
                "prompt studio driver exploded at /private/prompt-studio.db"
            ),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to fetch optimization history"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error fetching optimization history",
        "prompt studio driver exploded",
    )


@pytest.mark.asyncio
async def test_get_optimization_history_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await get_optimization_history(
            optimization_id=42,
            db=_BrokenOptimizationLookupUnexpectedDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to fetch optimization history"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error fetching optimization history",
        "prompt studio backend exploded",
    )


@pytest.mark.asyncio
async def test_add_optimization_iteration_maps_database_error(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await add_optimization_iteration(
            optimization_id=42,
            payload=OptimizationIterationCreate(iteration_number=1),
            db=_BrokenOptimizationLookupDb(
                "prompt studio driver exploded at /private/prompt-studio.db"
            ),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add iteration"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error recording optimization iteration",
        "prompt studio driver exploded",
    )


@pytest.mark.asyncio
async def test_add_optimization_iteration_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await add_optimization_iteration(
            optimization_id=42,
            payload=OptimizationIterationCreate(iteration_number=1),
            db=_BrokenOptimizationLookupUnexpectedDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to add iteration"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error adding optimization iteration",
        "prompt studio backend exploded",
    )


@pytest.mark.asyncio
async def test_list_optimization_iterations_maps_database_error(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await list_optimization_iterations(
            optimization_id=42,
            page=1,
            per_page=50,
            db=_BrokenOptimizationLookupDb(
                "prompt studio driver exploded at /private/prompt-studio.db"
            ),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list iterations"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error listing optimization iterations",
        "prompt studio driver exploded",
    )


@pytest.mark.asyncio
async def test_list_optimization_iterations_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await list_optimization_iterations(
            optimization_id=42,
            page=1,
            per_page=50,
            db=_BrokenOptimizationLookupUnexpectedDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list iterations"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error listing optimization iterations",
        "prompt studio backend exploded",
    )


@pytest.mark.asyncio
async def test_compare_strategies_maps_database_error():
    with pytest.raises(HTTPException) as exc_info:
        await compare_strategies(
            request=CompareStrategiesRequest(
                prompt_id=12,
                test_case_ids=[1],
                strategies=["iterative"],
                model_configuration={"model_name": "gpt-4o-mini"},
            ),
            http_request=object(),
            _=True,
            db=_BrokenCompareStrategiesDb(),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to compare strategies"


@pytest.mark.asyncio
async def test_compare_strategies_database_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await compare_strategies(
            request=CompareStrategiesRequest(
                prompt_id=12,
                test_case_ids=[1],
                strategies=["iterative"],
                model_configuration={"model_name": "gpt-4o-mini"},
            ),
            http_request=object(),
            _=True,
            db=_BrokenCompareStrategiesDb(),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to compare strategies"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Database error comparing strategies",
        "driver failed",
    )


@pytest.mark.asyncio
async def test_compare_strategies_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _EndpointLoggerStub()
    monkeypatch.setattr(optimization_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await compare_strategies(
            request=CompareStrategiesRequest(
                prompt_id=12,
                test_case_ids=[1],
                strategies=["iterative"],
                model_configuration={"model_name": "gpt-4o-mini"},
            ),
            http_request=object(),
            _=True,
            db=_BrokenCompareStrategiesUnexpectedDb(),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to compare strategies"
    _assert_sanitized_endpoint_error_log(
        logger_stub,
        "Unexpected error comparing strategies",
        "prompt studio backend exploded",
    )
