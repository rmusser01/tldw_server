import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_test_cases as test_cases_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_test_cases import (
    create_bulk_test_cases,
    create_test_case,
    delete_test_case,
    export_test_cases,
    get_test_case,
    get_csv_import_template,
    generate_test_cases,
    import_test_cases,
    import_test_cases_csv_upload,
    list_test_cases,
    update_test_case,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_test import (
    TestCaseBulkCreate,
    TestCaseCreate,
    TestCaseExportRequest,
    TestCaseGenerateRequest,
    TestCaseImportRequest,
    TestCaseUpdate,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    ConflictError,
    DatabaseError,
    InputError,
)


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

    def info(self, *_args, **_kwargs):
        return None


_SENSITIVE_MARKERS = (
    "duplicate case",
    "duplicate generated cases",
    "duplicate upload case",
    "generic import exploded",
    "generic export exploded",
    "generic generation exploded",
    "generic template exploded",
    "generic upload exploded",
    "driver failed",
    "invalid export filter",
    "invalid generation payload",
    "invalid import payload",
    "invalid signature",
    "unexpected list exploded",
    "project 7",
    "/private/tmp/prompt-studio-test-cases.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/prompt-studio-test-cases.db")


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


class _BrokenTestCaseManager:
    def __init__(self, _db, manager_exc: Exception, *, get_result=None):
        self._manager_exc = manager_exc
        self._get_result = (
            {"id": 42, "project_id": 7}
            if get_result is None
            else get_result
        )

    def get_test_case(self, _test_case_id: int):
        if isinstance(self._get_result, Exception):
            raise self._get_result
        return self._get_result

    def get_test_case_stats(self, _project_id: int):
        return {"total": 0}

    def create_test_case(self, **_kwargs):
        raise self._manager_exc

    def create_bulk_test_cases(self, **_kwargs):
        raise self._manager_exc

    def list_test_cases(self, **_kwargs):
        raise self._manager_exc

    def update_test_case(self, _test_case_id: int, _update_data: dict):
        raise self._manager_exc

    def delete_test_case(self, _test_case_id: int, *, hard_delete: bool):
        _ = hard_delete
        raise self._manager_exc


class _BrokenTestCaseIO:
    def __init__(self, _manager, io_exc: Exception):
        self._io_exc = io_exc

    def import_from_csv(self, **_kwargs):
        raise self._io_exc

    def import_from_json(self, **_kwargs):
        raise self._io_exc

    def generate_csv_template(self, **_kwargs):
        raise self._io_exc

    def export_to_csv(self, **_kwargs):
        raise self._io_exc

    def export_to_json(self, **_kwargs):
        raise self._io_exc


class _BrokenTestCaseGenerator:
    def __init__(self, _manager, generator_exc: Exception):
        self._generator_exc = generator_exc

    def generate_diverse_cases(self, **_kwargs):
        raise self._generator_exc

    def generate_from_description(self, **_kwargs):
        raise self._generator_exc


class _FakeUploadFile:
    def __init__(self, content: bytes):
        self._content = content

    async def read(self):
        return self._content


class _LimitImportTestCaseManager:
    def __init__(self, _db):
        pass

    def get_test_case_stats(self, _project_id: int):
        return {"total": 9}


class _SuccessfulImportTestCaseIO:
    def __init__(self, _manager):
        pass

    def import_from_csv(self, **_kwargs):
        return 2, []

    def import_from_json(self, **_kwargs):
        return 2, []


def _patch_test_case_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    manager_exc: Exception,
    *,
    get_result=None,
) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(test_cases_endpoint, "logger", logger_stub, raising=True)
    monkeypatch.setattr(
        test_cases_endpoint,
        "TestCaseManager",
        lambda db: _BrokenTestCaseManager(db, manager_exc, get_result=get_result),
        raising=True,
    )

    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        test_cases_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )

    async def _allow_read_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        test_cases_endpoint,
        "require_project_access",
        _allow_read_access,
        raising=True,
    )
    return logger_stub


def _patch_successful_import_dependencies(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(test_cases_endpoint, "logger", logger_stub, raising=True)
    monkeypatch.setattr(
        test_cases_endpoint,
        "TestCaseManager",
        _LimitImportTestCaseManager,
        raising=True,
    )
    monkeypatch.setattr(
        test_cases_endpoint,
        "TestCaseIO",
        _SuccessfulImportTestCaseIO,
        raising=True,
    )

    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        test_cases_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )
    return logger_stub


def _patch_test_case_io(
    monkeypatch: pytest.MonkeyPatch,
    io_exc: Exception,
    *,
    manager_exc: Exception | None = None,
) -> _LoggerStub:
    effective_manager_exc = manager_exc or DatabaseError("manager should not be called")
    logger_stub = _patch_test_case_dependencies(monkeypatch, effective_manager_exc)
    monkeypatch.setattr(
        test_cases_endpoint,
        "TestCaseIO",
        lambda manager: _BrokenTestCaseIO(manager, io_exc),
        raising=True,
    )
    return logger_stub


def _patch_test_case_generator(
    monkeypatch: pytest.MonkeyPatch,
    generator_exc: Exception,
    *,
    manager_exc: Exception | None = None,
) -> _LoggerStub:
    effective_manager_exc = manager_exc or DatabaseError("manager should not be called")
    logger_stub = _patch_test_case_dependencies(monkeypatch, effective_manager_exc)
    monkeypatch.setattr(
        test_cases_endpoint,
        "TestCaseGenerator",
        lambda manager: _BrokenTestCaseGenerator(manager, generator_exc),
        raising=True,
    )
    return logger_stub


class _SecurityConfig:
    max_test_cases = 10


@pytest.mark.asyncio
async def test_create_test_case_maps_database_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(monkeypatch, _database_failure())

    with pytest.raises(HTTPException) as exc_info:
        await create_test_case(
            test_case_data=TestCaseCreate(
                project_id=7,
                name="TC1",
                inputs={"question": "hi"},
                expected_outputs={"answer": "hello"},
            ),
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create test case"
    _assert_sanitized_error_log(logger_stub, "Database error creating test case")


@pytest.mark.asyncio
async def test_create_bulk_test_cases_maps_database_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(monkeypatch, _database_failure())

    with pytest.raises(HTTPException) as exc_info:
        await create_bulk_test_cases(
            bulk_data=TestCaseBulkCreate(
                project_id=7,
                test_cases=[
                    {
                        "name": "TC1",
                        "inputs": {"question": "hi"},
                        "expected_outputs": {"answer": "hello"},
                    }
                ],
            ),
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create test cases"
    _assert_sanitized_error_log(logger_stub, "Database error creating bulk test cases")


@pytest.mark.asyncio
async def test_create_test_case_maps_conflict_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(monkeypatch, ConflictError("duplicate case"))

    with pytest.raises(HTTPException) as exc_info:
        await create_test_case(
            test_case_data=TestCaseCreate(
                project_id=7,
                name="TC1",
                inputs={"question": "hi"},
                expected_outputs={"answer": "hello"},
            ),
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Test case already exists"
    _assert_sanitized_warning_log(logger_stub, "Conflict creating test case")


@pytest.mark.asyncio
async def test_create_bulk_test_cases_maps_conflict_error(monkeypatch):
    _patch_test_case_dependencies(monkeypatch, ConflictError("bulk duplicate"))

    with pytest.raises(HTTPException) as exc_info:
        await create_bulk_test_cases(
            bulk_data=TestCaseBulkCreate(
                project_id=7,
                test_cases=[
                    {
                        "name": "TC1",
                        "inputs": {"question": "hi"},
                        "expected_outputs": {"answer": "hello"},
                    }
                ],
            ),
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "bulk duplicate"


@pytest.mark.asyncio
async def test_list_test_cases_sanitizes_database_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(monkeypatch, _database_failure())

    with pytest.raises(HTTPException) as exc_info:
        await list_test_cases(
            project_id=7,
            page=1,
            per_page=20,
            is_golden=None,
            tags=None,
            search=None,
            signature_id=None,
            _=True,
            db=object(),
    )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list test cases"
    _assert_sanitized_error_log(logger_stub, "Database error listing test cases")


@pytest.mark.asyncio
async def test_list_test_cases_sanitizes_unexpected_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(
        monkeypatch,
        RuntimeError("unexpected list exploded /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await list_test_cases(
            project_id=7,
            page=1,
            per_page=20,
            is_golden=None,
            tags=None,
            search=None,
            signature_id=None,
            _=True,
            db=object(),
    )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list test cases"
    _assert_sanitized_error_log(logger_stub, "Unexpected error listing test cases")


@pytest.mark.asyncio
async def test_get_test_case_maps_database_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(
        monkeypatch,
        _database_failure(),
        get_result=_database_failure(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_test_case(
            test_case_id=42,
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get test case"
    _assert_sanitized_error_log(logger_stub, "Database error getting test case")


@pytest.mark.asyncio
async def test_update_test_case_maps_database_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(monkeypatch, _database_failure())

    with pytest.raises(HTTPException) as exc_info:
        await update_test_case(
            test_case_id=42,
            updates=TestCaseUpdate(description="updated"),
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update test case"
    _assert_sanitized_error_log(logger_stub, "Database error updating test case")


@pytest.mark.asyncio
async def test_update_test_case_maps_input_error_to_not_found(monkeypatch):
    _patch_test_case_dependencies(monkeypatch, InputError("test case not found"))

    with pytest.raises(HTTPException) as exc_info:
        await update_test_case(
            test_case_id=42,
            updates=TestCaseUpdate(description="updated"),
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "test case not found"


@pytest.mark.asyncio
async def test_delete_test_case_maps_database_error(monkeypatch):
    logger_stub = _patch_test_case_dependencies(monkeypatch, _database_failure())

    with pytest.raises(HTTPException) as exc_info:
        await delete_test_case(
            test_case_id=42,
            permanent=False,
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete test case"
    _assert_sanitized_error_log(logger_stub, "Database error deleting test case")


@pytest.mark.asyncio
async def test_import_test_cases_maps_input_error(monkeypatch):
    logger_stub = _patch_test_case_io(monkeypatch, InputError("invalid import payload"))

    with pytest.raises(HTTPException) as exc_info:
        await import_test_cases(
            import_data=TestCaseImportRequest(
                project_id=7,
                format="json",
                data='{"test_cases":[]}',
            ),
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid import payload"
    _assert_sanitized_error_log(logger_stub, "Error importing test cases")


@pytest.mark.asyncio
async def test_import_test_cases_sanitizes_generic_error(monkeypatch):
    logger_stub = _patch_test_case_io(
        monkeypatch,
        RuntimeError("generic import exploded /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await import_test_cases(
            import_data=TestCaseImportRequest(
                project_id=7,
                format="json",
                data='{"test_cases":[]}',
            ),
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to import test cases"
    _assert_sanitized_error_log(logger_stub, "Error importing test cases")


@pytest.mark.asyncio
async def test_import_test_cases_limit_warning_log_is_sanitized(monkeypatch):
    logger_stub = _patch_successful_import_dependencies(monkeypatch)

    response = await import_test_cases(
        import_data=TestCaseImportRequest(
            project_id=7,
            format="json",
            data='{"test_cases":[]}',
        ),
        db=object(),
        security_config=_SecurityConfig(),
        user_context={"user_id": "tester"},
    )

    assert response.success is True
    assert response.data["imported"] == 2
    assert response.data["total_test_cases"] == 11
    _assert_sanitized_warning_log(logger_stub, "Import would exceed test case limit")


@pytest.mark.asyncio
async def test_import_test_cases_csv_upload_maps_conflict_error(monkeypatch):
    logger_stub = _patch_test_case_io(monkeypatch, ConflictError("duplicate upload case"))

    with pytest.raises(HTTPException) as exc_info:
        await import_test_cases_csv_upload(
            project_id=7,
            file=_FakeUploadFile(b"name,input.q\ncase,hi\n"),
            signature_id=None,
            auto_generate_names=True,
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "duplicate upload case"
    _assert_sanitized_error_log(logger_stub, "Error importing test cases via upload")


@pytest.mark.asyncio
async def test_import_test_cases_csv_upload_sanitizes_generic_error(monkeypatch):
    logger_stub = _patch_test_case_io(
        monkeypatch,
        RuntimeError("generic upload exploded /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await import_test_cases_csv_upload(
            project_id=7,
            file=_FakeUploadFile(b"name,input.q\ncase,hi\n"),
            signature_id=None,
            auto_generate_names=True,
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to import CSV test cases"
    _assert_sanitized_error_log(logger_stub, "Error importing test cases via upload")


@pytest.mark.asyncio
async def test_get_csv_import_template_maps_input_error(monkeypatch):
    logger_stub = _patch_test_case_io(monkeypatch, InputError("invalid signature"))

    with pytest.raises(HTTPException) as exc_info:
        await get_csv_import_template(
            signature_id=3,
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid signature"
    _assert_sanitized_error_log(logger_stub, "Failed to generate CSV template")


@pytest.mark.asyncio
async def test_get_csv_import_template_sanitizes_generic_error(monkeypatch):
    logger_stub = _patch_test_case_io(
        monkeypatch,
        RuntimeError("generic template exploded /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_csv_import_template(
            signature_id=3,
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate CSV template"
    _assert_sanitized_error_log(logger_stub, "Failed to generate CSV template")


@pytest.mark.asyncio
async def test_export_test_cases_maps_input_error(monkeypatch):
    logger_stub = _patch_test_case_io(monkeypatch, InputError("invalid export filter"))

    with pytest.raises(HTTPException) as exc_info:
        await export_test_cases(
            project_id=7,
            export_request=TestCaseExportRequest(format="json"),
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid export filter"
    _assert_sanitized_error_log(logger_stub, "Error exporting test cases")


@pytest.mark.asyncio
async def test_export_test_cases_sanitizes_generic_error(monkeypatch):
    logger_stub = _patch_test_case_io(
        monkeypatch,
        RuntimeError("generic export exploded /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await export_test_cases(
            project_id=7,
            export_request=TestCaseExportRequest(format="json"),
            db=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to export test cases"
    _assert_sanitized_error_log(logger_stub, "Error exporting test cases")


@pytest.mark.asyncio
async def test_generate_test_cases_maps_conflict_error(monkeypatch):
    logger_stub = _patch_test_case_generator(monkeypatch, ConflictError("duplicate generated cases"))

    with pytest.raises(HTTPException) as exc_info:
        await generate_test_cases(
            generate_request=TestCaseGenerateRequest(
                project_id=7,
                signature_id=2,
                num_cases=2,
                generation_strategy="diverse",
            ),
            _rate=True,
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "duplicate generated cases"
    _assert_sanitized_error_log(logger_stub, "Error generating test cases")


@pytest.mark.asyncio
async def test_generate_test_cases_sanitizes_value_error(monkeypatch):
    logger_stub = _patch_test_case_generator(
        monkeypatch,
        ValueError("invalid generation payload /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await generate_test_cases(
            generate_request=TestCaseGenerateRequest(
                project_id=7,
                signature_id=2,
                num_cases=2,
                generation_strategy="diverse",
            ),
            _rate=True,
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid test case generation request"
    _assert_sanitized_warning_log(logger_stub, "Invalid test case generation request")


@pytest.mark.asyncio
async def test_generate_test_cases_sanitizes_generic_error(monkeypatch):
    logger_stub = _patch_test_case_generator(
        monkeypatch,
        RuntimeError("generic generation exploded /private/tmp/prompt-studio-test-cases.db"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await generate_test_cases(
            generate_request=TestCaseGenerateRequest(
                project_id=7,
                signature_id=2,
                num_cases=2,
                generation_strategy="diverse",
            ),
            _rate=True,
            db=object(),
            security_config=_SecurityConfig(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate test cases"
    _assert_sanitized_error_log(logger_stub, "Error generating test cases")
