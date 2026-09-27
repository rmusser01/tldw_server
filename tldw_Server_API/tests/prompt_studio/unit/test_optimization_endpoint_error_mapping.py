import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as optimization_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization import (
    OptimizationIterationCreate,
    add_optimization_iteration,
    cancel_optimization,
    compare_strategies,
    create_optimization,
    get_optimization,
    get_optimization_history,
    list_optimization_iterations,
    list_optimizations,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationConfig,
    OptimizationCreate,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError
from tldw_Server_API.app.core.Logging import log_context as log_context_module

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


class _CancellationDb:
    def __init__(
        self,
        events: list[str],
        *,
        prompt_status: str = "running",
        project_owner_user_id: str = "tester",
    ) -> None:
        self.events = events
        self.prompt_status = prompt_status
        self.project_owner_user_id = project_owner_user_id
        self.update_kwargs: dict[str, object] | None = None

    def get_optimization(self, optimization_id):
        return {
            "id": optimization_id,
            "project_id": 7,
            "status": self.prompt_status,
            "deleted": False,
        }

    def set_optimization_status(self, *_args, **_kwargs):
        self.events.append("prompt_cancelled")
        return {"id": 42, "status": "cancelled"}

    def get_project(self, project_id):
        return {"id": project_id, "user_id": self.project_owner_user_id}

    def update_optimization(self, *_args, **kwargs):
        self.update_kwargs = dict(kwargs)
        self.events.append("prompt_cancelled")
        return {"id": 42, "status": "cancelled"}


class _CancellationCoreJobs:
    def __init__(
        self,
        events: list[str],
        *,
        include_job: bool = True,
        status_after_cancel: str | None = None,
    ) -> None:
        self.events = events
        self.include_job = include_job
        self.status_after_cancel = status_after_cancel
        self.job = {
            "id": 1,
            "uuid": "job-1",
            "domain": "prompt_studio",
            "job_type": "optimization",
            "owner_user_id": "tester",
            "status": "processing",
            "payload": {"optimization_id": 42},
            "created_at": "2026-07-16T00:00:00+00:00",
        }

    def list_jobs(self, **_kwargs):
        return [self.job] if self.include_job else []

    def get_job_by_uuid(self, _job_id):
        return self.job

    def cancel_job(self, *_args, **_kwargs):
        self.events.append("job_cancel_attempted")
        if self.status_after_cancel:
            self.job["status"] = self.status_after_cancel
        return False


class _HiddenCancellationCoreJobs(_CancellationCoreJobs):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.job["id"] = 1
        self.job["uuid"] = "job-1"
        self.jobs = [
            {
                **self.job,
                "id": job_id,
                "uuid": f"job-{job_id}",
                "status": job_status,
                "payload": {"optimization_id": 100 + job_id},
                "created_at": f"2026-07-16T00:00:0{job_id}+00:00",
            }
            for job_id, job_status in (
                (4, "completed"),
                (3, "failed"),
                (2, "cancelled"),
            )
        ]
        self.jobs.append(self.job)

    def list_jobs(self, **kwargs):
        return self.jobs[: int(kwargs["limit"])]

    def get_job_by_uuid(self, job_id):
        return next((job for job in self.jobs if job["uuid"] == job_id), None)

    def cancel_job(self, job_id, **_kwargs):
        self.events.append("job_cancel_attempted")
        assert job_id == 1
        self.job["status"] = "cancelled"
        return True


class _SuccessfulCancellationCoreJobs(_CancellationCoreJobs):
    def __init__(self, events: list[str], *, owner_user_id: str) -> None:
        super().__init__(events)
        self.job["owner_user_id"] = owner_user_id
        self.lookup_owner_user_ids: list[str | None] = []

    def list_jobs(self, **kwargs):
        self.lookup_owner_user_ids.append(kwargs.get("owner_user_id"))
        return [self.job]

    def cancel_job(self, job_id, **_kwargs):
        self.events.append("job_cancel_attempted")
        assert job_id == self.job["id"]
        self.job["status"] = "cancelled"
        return True


async def _allow_cancel_write_access(*_args, **_kwargs):
    return True


def _install_cancel_adapter(monkeypatch, core_jobs: _CancellationCoreJobs) -> None:
    adapter_class = optimization_endpoint.PromptStudioJobsAdapter
    adapter = object.__new__(adapter_class)
    adapter._backend = "core"
    adapter._jm = core_jobs
    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_cancel_write_access,
        raising=True,
    )
    monkeypatch.setattr(
        optimization_endpoint,
        "PromptStudioJobsAdapter",
        lambda: adapter,
        raising=True,
    )


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
        return {"id": 12, "project_id": 7, "project_user_id": "tester"}

    def get_test_cases_by_ids(self, test_case_ids):
        return [
            {"id": test_case_id, "project_id": 7, "deleted": False}
            for test_case_id in test_case_ids
        ]

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
        test_case_ids=[1],
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
async def test_cancel_optimization_keeps_prompt_active_when_job_cancel_is_rejected(monkeypatch):
    events: list[str] = []
    _install_cancel_adapter(monkeypatch, _CancellationCoreJobs(events))

    with pytest.raises(HTTPException) as exc_info:
        await cancel_optimization(
            request=object(),
            optimization_id=42,
            reason="stop",
            db=_CancellationDb(events),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Optimization job could not be cancelled (status: processing)"
    assert events == ["job_cancel_attempted"]


@pytest.mark.asyncio
async def test_cancel_optimization_accepts_job_cancelled_by_a_concurrent_request(monkeypatch):
    events: list[str] = []
    db = _CancellationDb(events)
    _install_cancel_adapter(
        monkeypatch,
        _CancellationCoreJobs(events, status_after_cancel="cancelled"),
    )

    response = await cancel_optimization(
        request=object(),
        optimization_id=42,
        reason="stop",
        db=db,
        user_context={"user_id": "tester", "client_id": "client-1"},
    )

    assert response.success is True
    assert events == [
        "job_cancel_attempted",
        "prompt_cancelled",
    ]
    assert "completed" in (db.update_kwargs or {})["expected_statuses"]


@pytest.mark.asyncio
async def test_cancel_optimization_without_a_job_still_cancels_prompt(monkeypatch):
    events: list[str] = []
    _install_cancel_adapter(monkeypatch, _CancellationCoreJobs(events, include_job=False))

    response = await cancel_optimization(
        request=object(),
        optimization_id=42,
        reason=None,
        db=_CancellationDb(events, prompt_status="pending"),
        user_context={"user_id": "tester", "client_id": "client-1"},
    )

    assert response.success is True
    assert events == ["prompt_cancelled"]


@pytest.mark.asyncio
async def test_cancel_optimization_finds_processing_job_behind_newer_owner_jobs(monkeypatch):
    events: list[str] = []
    core_jobs = _HiddenCancellationCoreJobs(events)
    _install_cancel_adapter(monkeypatch, core_jobs)

    response = await cancel_optimization(
        request=object(),
        optimization_id=42,
        reason="stop",
        db=_CancellationDb(events),
        user_context={"user_id": "tester", "client_id": "client-1"},
    )

    assert response.success is True
    assert core_jobs.job["status"] == "cancelled"
    assert [job["status"] for job in core_jobs.jobs[:3]] == [
        "completed",
        "failed",
        "cancelled",
    ]
    assert events == ["job_cancel_attempted", "prompt_cancelled"]


@pytest.mark.asyncio
async def test_admin_cancel_uses_authorized_project_owner_for_core_job(monkeypatch):
    events: list[str] = []
    core_jobs = _SuccessfulCancellationCoreJobs(events, owner_user_id="7")
    _install_cancel_adapter(monkeypatch, core_jobs)

    response = await cancel_optimization(
        request=object(),
        optimization_id=42,
        reason="stop",
        db=_CancellationDb(events, project_owner_user_id="7"),
        user_context={"user_id": "99", "client_id": "client-1", "is_admin": True},
    )

    assert response.success is True
    assert core_jobs.lookup_owner_user_ids == ["7"]
    assert core_jobs.job["status"] == "cancelled"
    assert events == ["job_cancel_attempted", "prompt_cancelled"]


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
                strategies=["iterative", "mipro"],
                model_configuration={
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                },
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
                strategies=["iterative", "mipro"],
                model_configuration={
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                },
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
                strategies=["iterative", "mipro"],
                model_configuration={
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                },
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
