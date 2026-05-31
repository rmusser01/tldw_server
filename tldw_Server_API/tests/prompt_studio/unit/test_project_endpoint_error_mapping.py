import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_projects as projects_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_projects import (
    create_project,
    get_project,
    archive_project,
    get_project_stats,
    list_projects,
    delete_project,
    unarchive_project,
    update_project,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import ProjectCreate
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import ProjectUpdate
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import ConflictError, DatabaseError, InputError


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.error_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "driver failed",
    "idem-sensitive-key",
    "unexpected list exploded",
    "/private/tmp/prompt-studio-projects.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/prompt-studio-projects.db")


def _assert_sanitized_error_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_warning_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.warning_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.warning_calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(logger_stub.warning_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _patch_endpoint_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(projects_endpoint, "logger", logger_stub, raising=True)
    return logger_stub


class _BrokenProjectsDb:
    def __init__(self, update_exc: Exception):
        self._update_exc = update_exc

    def update_project(self, *_args, **_kwargs):
        raise self._update_exc


class _BrokenCreateProjectDb:
    def __init__(
        self,
        create_exc: Exception | None = None,
        fallback_exc: Exception | None = None,
    ):
        self._create_exc = create_exc or _database_failure()
        self._fallback_exc = fallback_exc or DatabaseError("fallback failed")

    def lookup_idempotency(self, *_args, **_kwargs):
        return None

    def create_project(self, *_args, **_kwargs):
        raise self._create_exc

    def _execute(self, *_args, **_kwargs):
        raise self._fallback_exc


class _IdempotencyLookupFailingCreateProjectDb:
    def __init__(self):
        self.recorded_idempotency = None

    def lookup_idempotency(self, *_args, **_kwargs):
        raise _database_failure()

    def create_project(self, **kwargs):
        return {
            "id": 7,
            "uuid": "12345678-1234-5678-1234-567812345678",
            "name": kwargs["name"],
            "description": kwargs.get("description"),
            "status": kwargs["status"],
            "metadata": kwargs.get("metadata"),
            "user_id": kwargs["user_id"],
            "version": 1,
            "created_at": None,
            "updated_at": None,
        }

    def record_idempotency(self, *args, **kwargs):
        self.recorded_idempotency = (args, kwargs)


class _BrokenListProjectsDb:
    def list_projects(self, *_args, **_kwargs):
        raise _database_failure()


class _UnexpectedListProjectsDb:
    def list_projects(self, *_args, **_kwargs):
        raise RuntimeError("unexpected list exploded /private/tmp/prompt-studio-projects.db")


class _DeleteProjectFallbackDb:
    def delete_project(self, *_args, **_kwargs):
        raise _database_failure()

    def update_project(self, *_args, **_kwargs):
        return {"id": 42, "status": "archived"}


class _BrokenGetProjectDb:
    def get_project(self, *_args, **_kwargs):
        raise _database_failure()


class _BrokenProjectStatsDb:
    def get_connection(self):
        raise _database_failure()


class _FakePsLogger:
    def info(self, *_args, **_kwargs):
        return None


def _patch_prompt_studio_request_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        projects_endpoint,
        "ensure_request_id",
        lambda _request: "req-1",
        raising=True,
    )
    monkeypatch.setattr(
        projects_endpoint,
        "ensure_traceparent",
        lambda _request: "tp-1",
        raising=True,
    )
    monkeypatch.setattr(
        projects_endpoint,
        "get_ps_logger",
        lambda **_kwargs: _FakePsLogger(),
        raising=True,
    )


@pytest.mark.asyncio
async def test_update_project_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await update_project(
            project_id=42,
            updates=ProjectUpdate(description="updated"),
            _=True,
            db=_BrokenProjectsDb(_database_failure()),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update project"
    _assert_sanitized_error_log(logger_stub, "Database error updating project")


@pytest.mark.asyncio
async def test_update_project_maps_input_error_to_not_found():
    with pytest.raises(HTTPException) as exc_info:
        await update_project(
            project_id=42,
            updates=ProjectUpdate(description="updated"),
            _=True,
            db=_BrokenProjectsDb(InputError("project not found")),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "project not found"


@pytest.mark.asyncio
async def test_create_project_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await create_project(
            project_data=ProjectCreate(name="Prompt Studio Project"),
            request=object(),
            user_context={"user_id": "tester"},
            db=_BrokenCreateProjectDb(),
            _=True,
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create project"
    _assert_sanitized_error_log(logger_stub, "Database error creating project")


@pytest.mark.asyncio
async def test_create_project_idempotency_lookup_failure_log_is_sanitized(monkeypatch):
    _patch_prompt_studio_request_logging(monkeypatch)
    logger_stub = _patch_endpoint_logger(monkeypatch)
    db = _IdempotencyLookupFailingCreateProjectDb()

    response = await create_project(
        project_data=ProjectCreate(name="Prompt Studio Project"),
        request=object(),
        user_context={"user_id": "tester"},
        db=db,
        _=True,
        idempotency_key="idem-sensitive-key",
    )

    assert response.success is True
    assert db.recorded_idempotency is not None
    _assert_sanitized_warning_log(logger_stub, "Idempotency lookup failed")


@pytest.mark.asyncio
async def test_create_project_maps_conflict_error_after_existing_lookup_fails():
    with pytest.raises(HTTPException) as exc_info:
        await create_project(
            project_data=ProjectCreate(name="Prompt Studio Project"),
            request=object(),
            user_context={"user_id": "tester"},
            db=_BrokenCreateProjectDb(
                create_exc=ConflictError("project name already exists"),
            ),
            _=True,
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "project name already exists"


@pytest.mark.asyncio
async def test_create_project_conflict_lookup_database_error_log_is_sanitized(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await create_project(
            project_data=ProjectCreate(name="Prompt Studio Project"),
            request=object(),
            user_context={"user_id": "tester"},
            db=_BrokenCreateProjectDb(
                create_exc=ConflictError("project name already exists"),
                fallback_exc=_database_failure(),
            ),
            _=True,
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "project name already exists"
    _assert_sanitized_error_log(
        logger_stub,
        "Failed to retrieve existing project after conflict",
    )


@pytest.mark.asyncio
async def test_create_project_conflict_lookup_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(RuntimeError):
        await create_project(
            project_data=ProjectCreate(name="Prompt Studio Project"),
            request=object(),
            user_context={"user_id": "tester"},
            db=_BrokenCreateProjectDb(
                create_exc=ConflictError("project name already exists"),
                fallback_exc=RuntimeError("fallback lookup exploded /private/tmp/prompt-studio-projects.db"),
            ),
            _=True,
            idempotency_key=None,
        )

    _assert_sanitized_error_log(
        logger_stub,
        "Unexpected error retrieving existing project after conflict",
    )


@pytest.mark.asyncio
async def test_list_projects_sanitizes_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await list_projects(
            page=1,
            per_page=20,
            status_filter=None,
            include_deleted=False,
            search=None,
            user_context={"user_id": "tester", "is_admin": False},
            db=_BrokenListProjectsDb(),
    )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list projects"
    _assert_sanitized_error_log(logger_stub, "Database error listing projects")


@pytest.mark.asyncio
async def test_list_projects_sanitizes_unexpected_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await list_projects(
            page=1,
            per_page=20,
            status_filter=None,
            include_deleted=False,
            search=None,
            user_context={"user_id": "tester", "is_admin": False},
            db=_UnexpectedListProjectsDb(),
    )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list projects"
    _assert_sanitized_error_log(logger_stub, "Unexpected error listing projects")


@pytest.mark.asyncio
async def test_get_project_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await get_project(
            project_id=42,
            _=True,
            db=_BrokenGetProjectDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get project"
    _assert_sanitized_error_log(logger_stub, "Database error getting project")


@pytest.mark.asyncio
async def test_get_project_stats_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await get_project_stats(
            project_id=42,
            _=True,
            db=_BrokenProjectStatsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get project statistics"
    _assert_sanitized_error_log(logger_stub, "Database error getting project stats")


@pytest.mark.asyncio
async def test_archive_project_maps_database_error(monkeypatch):
    _patch_prompt_studio_request_logging(monkeypatch)
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await archive_project(
            request=object(),
            project_id=42,
            _=True,
            db=_BrokenProjectsDb(_database_failure()),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to archive project"
    _assert_sanitized_error_log(logger_stub, "Database error archiving project")


@pytest.mark.asyncio
async def test_archive_project_maps_input_error_to_not_found(monkeypatch):
    _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await archive_project(
            request=object(),
            project_id=42,
            _=True,
            db=_BrokenProjectsDb(InputError("project not found")),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "project not found"


@pytest.mark.asyncio
async def test_unarchive_project_maps_database_error(monkeypatch):
    _patch_prompt_studio_request_logging(monkeypatch)
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await unarchive_project(
            request=object(),
            project_id=42,
            _=True,
            db=_BrokenProjectsDb(_database_failure()),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to unarchive project"
    _assert_sanitized_error_log(logger_stub, "Database error unarchiving project")


@pytest.mark.asyncio
async def test_unarchive_project_maps_input_error_to_not_found(monkeypatch):
    _patch_prompt_studio_request_logging(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await unarchive_project(
            request=object(),
            project_id=42,
            _=True,
            db=_BrokenProjectsDb(InputError("project not found")),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "project not found"


@pytest.mark.asyncio
async def test_delete_project_fallback_archive_logs_are_sanitized(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    response = await delete_project(
        project_id=42,
        permanent=False,
        _=True,
        db=_DeleteProjectFallbackDb(),
        user_context={"user_id": "tester"},
    )

    assert response.success is True
    assert response.data == {"message": "Project soft deleted (fallback archive applied)"}
    _assert_sanitized_error_log(logger_stub, "Database error deleting project")
    _assert_sanitized_warning_log(logger_stub, "Fallback archive applied after project delete failure")

    rendered_warnings = repr(logger_stub.warning_calls)
    assert "42" not in rendered_warnings
