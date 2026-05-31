import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_prompts as prompts_endpoint,
)
from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts import (
    create_prompt,
    get_prompt,
    get_prompt_history,
    list_prompts,
    preview_prompt,
    revert_prompt,
    update_prompt,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import (
    PromptCreate,
    PromptUpdate,
    StructuredPromptPreviewRequest,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    ConflictError,
    DatabaseError,
    InputError,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "driver failed",
    "/private/tmp/prompt-studio-prompts.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/prompt-studio-prompts.db")


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


def _patch_endpoint_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(prompts_endpoint, "logger", logger_stub, raising=True)
    return logger_stub


class _BrokenCreatePromptDb:
    def __init__(self, create_exc: Exception | None = None):
        self._create_exc = create_exc or _database_failure()

    def lookup_idempotency(self, *_args, **_kwargs):
        return None

    def create_prompt(self, **_kwargs):
        raise self._create_exc


class _CheckpointFailureCreatePromptDb:
    def lookup_idempotency(self, *_args, **_kwargs):
        raise _database_failure()

    def create_prompt(self, **_kwargs):
        return {
            "id": 42,
            "project_id": 7,
            "name": "Prompt Name",
            "system_prompt": "system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
            "signature_id": None,
            "version_number": 1,
            "parent_version_id": None,
            "change_description": "initial version",
        }

    def record_idempotency(self, *_args, **_kwargs):
        return None


class _BrokenListPromptsDb:
    def __init__(self, list_exc: Exception | None = None):
        self._list_exc = list_exc or _database_failure()

    def list_prompts(self, *_args, **_kwargs):
        raise self._list_exc


class _BrokenGetPromptDb:
    def __init__(self, get_exc: Exception | None = None):
        self._get_exc = get_exc or _database_failure()

    def get_prompt_with_project(self, *_args, **_kwargs):
        raise self._get_exc


class _BrokenUpdatePromptDb:
    def __init__(self, update_exc: Exception | None = None):
        self._update_exc = update_exc or _database_failure()

    def get_prompt_with_project(self, *_args, **_kwargs):
        return {
            "id": 42,
            "project_id": 7,
            "project_user_id": "tester",
            "name": "Prompt Name",
            "system_prompt": "system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
            "signature_id": None,
        }

    def create_prompt_version(self, *_args, **_kwargs):
        raise self._update_exc


class _BrokenRevertPromptDb:
    def __init__(self, revert_exc: Exception):
        self._revert_exc = revert_exc

    def get_prompt(self, *_args, **_kwargs):
        return {"id": 42, "project_id": 7, "name": "Prompt Name"}

    def revert_prompt_to_version(self, *_args, **_kwargs):
        raise self._revert_exc


class _BrokenPromptHistoryDb:
    def __init__(self, history_exc: Exception | None = None):
        self._history_exc = history_exc or _database_failure()

    def get_prompt(self, *_args, **_kwargs):
        return {"id": 42, "project_id": 7, "name": "Prompt Name"}

    def list_prompt_versions(self, *_args, **_kwargs):
        raise self._history_exc


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


@pytest.mark.asyncio
async def test_create_prompt_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_prepare_prompt_record_fields",
        lambda **_kwargs: {
            "system_prompt": "system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_lengths",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_get_signature_for_project",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "convert_legacy_prompt_to_definition",
        lambda **_kwargs: object(),
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_content",
        lambda **_kwargs: None,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=PromptCreate(
                project_id=7,
                name="Prompt Name",
                system_prompt="system",
                user_prompt="{task}",
                change_description="initial version",
            ),
            db=_BrokenCreatePromptDb(),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create prompt"
    _assert_sanitized_error_log(
        logger_stub,
        "Prompt studio storage error creating prompt",
    )


@pytest.mark.asyncio
async def test_create_prompt_idempotency_lookup_failure_log_is_sanitized(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_prepare_prompt_record_fields",
        lambda **_kwargs: {
            "system_prompt": "system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_lengths",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_get_signature_for_project",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "convert_legacy_prompt_to_definition",
        lambda **_kwargs: object(),
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_content",
        lambda **_kwargs: None,
        raising=True,
    )

    response = await create_prompt(
        prompt_data=PromptCreate(
            project_id=7,
            name="Prompt Name",
            system_prompt="system",
            user_prompt="{task}",
            change_description="initial version",
        ),
        db=_CheckpointFailureCreatePromptDb(),
        security_config=object(),
        user_context={"user_id": "tester", "client_id": "client-1"},
        idempotency_key="idempotency-key",
    )

    assert response.success is True
    assert response.data.id == 42
    assert logger_stub.debug_calls
    assert (
        ("Prompt Studio checkpoint sync failed after prompt create",),
        {},
    ) in logger_stub.debug_calls
    rendered_calls = repr(logger_stub.debug_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


@pytest.mark.asyncio
async def test_create_prompt_maps_input_error_safe_message(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_prepare_prompt_record_fields",
        lambda **_kwargs: {
            "system_prompt": "system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_lengths",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_get_signature_for_project",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "convert_legacy_prompt_to_definition",
        lambda **_kwargs: object(),
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_content",
        lambda **_kwargs: None,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=PromptCreate(
                project_id=7,
                name="Prompt Name",
                system_prompt="system",
                user_prompt="{task}",
                change_description="initial version",
            ),
            db=_BrokenCreatePromptDb(
                InputError("raw create failure", safe_message="sanitized create failure")
            ),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized create failure"


@pytest.mark.asyncio
async def test_create_prompt_maps_conflict_error(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_prepare_prompt_record_fields",
        lambda **_kwargs: {
            "system_prompt": "system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_lengths",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_get_signature_for_project",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "convert_legacy_prompt_to_definition",
        lambda **_kwargs: object(),
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_content",
        lambda **_kwargs: None,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_prompt(
            prompt_data=PromptCreate(
                project_id=7,
                name="Prompt Name",
                system_prompt="system",
                user_prompt="{task}",
                change_description="initial version",
            ),
            db=_BrokenCreatePromptDb(
                ConflictError("raw prompt-name conflict")
            ),
            security_config=object(),
            user_context={"user_id": "tester", "client_id": "client-1"},
            idempotency_key=None,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Prompt with this name already exists in the project"


@pytest.mark.asyncio
async def test_list_prompts_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await list_prompts(
            project_id=7,
            page=1,
            per_page=20,
            include_deleted=False,
            _=True,
            db=_BrokenListPromptsDb(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list prompts"
    _assert_sanitized_error_log(
        logger_stub,
        "Prompt studio storage error listing prompts",
    )


@pytest.mark.asyncio
async def test_list_prompts_maps_input_error_safe_message():
    with pytest.raises(HTTPException) as exc_info:
        await list_prompts(
            project_id=7,
            page=1,
            per_page=20,
            include_deleted=False,
            _=True,
            db=_BrokenListPromptsDb(
                InputError("raw list failure", safe_message="sanitized list failure")
            ),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized list failure"


@pytest.mark.asyncio
async def test_get_prompt_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await get_prompt(
            prompt_id=42,
            db=_BrokenGetPromptDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get prompt"
    _assert_sanitized_error_log(
        logger_stub,
        "Prompt studio storage error getting prompt",
    )


@pytest.mark.asyncio
async def test_get_prompt_maps_input_error_safe_message():
    with pytest.raises(HTTPException) as exc_info:
        await get_prompt(
            prompt_id=42,
            db=_BrokenGetPromptDb(
                InputError("raw get failure", safe_message="sanitized get failure")
            ),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized get failure"


@pytest.mark.asyncio
async def test_update_prompt_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    monkeypatch.setattr(
        prompts_endpoint,
        "_get_signature_for_project",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_prepare_prompt_record_fields",
        lambda **_kwargs: {
            "system_prompt": "updated system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_lengths",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "convert_legacy_prompt_to_definition",
        lambda **_kwargs: object(),
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_content",
        lambda **_kwargs: None,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await update_prompt(
            prompt_id=42,
            updates=PromptUpdate(system_prompt="updated system", change_description="revise"),
            db=_BrokenUpdatePromptDb(),
            security_config=object(),
            user_context={"user_id": "tester", "is_admin": False, "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update prompt"
    _assert_sanitized_error_log(
        logger_stub,
        "Prompt studio storage error updating prompt",
    )


@pytest.mark.asyncio
async def test_update_prompt_maps_input_error_safe_message(monkeypatch):
    monkeypatch.setattr(
        prompts_endpoint,
        "_get_signature_for_project",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_prepare_prompt_record_fields",
        lambda **_kwargs: {
            "system_prompt": "updated system",
            "user_prompt": "{task}",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        },
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_lengths",
        lambda **_kwargs: None,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "convert_legacy_prompt_to_definition",
        lambda **_kwargs: object(),
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "_validate_prompt_content",
        lambda **_kwargs: None,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await update_prompt(
            prompt_id=42,
            updates=PromptUpdate(system_prompt="updated system", change_description="revise"),
            db=_BrokenUpdatePromptDb(
                InputError("raw update failure", safe_message="sanitized update failure")
            ),
            security_config=object(),
            user_context={"user_id": "tester", "is_admin": False, "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized update failure"


@pytest.mark.asyncio
async def test_revert_prompt_maps_conflict_error(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await revert_prompt(
            prompt_id=42,
            version=1,
            db=_BrokenRevertPromptDb(ConflictError("prompt version conflict")),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "prompt version conflict"


@pytest.mark.asyncio
async def test_revert_prompt_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await revert_prompt(
            prompt_id=42,
            version=1,
            db=_BrokenRevertPromptDb(_database_failure()),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to revert prompt"
    _assert_sanitized_error_log(
        logger_stub,
        "Prompt studio storage error reverting prompt",
    )


@pytest.mark.asyncio
async def test_revert_prompt_maps_input_error_safe_message(monkeypatch):
    async def _allow_write_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_write_access",
        _allow_write_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await revert_prompt(
            prompt_id=42,
            version=1,
            db=_BrokenRevertPromptDb(
                InputError("raw revert failure", safe_message="sanitized revert failure")
            ),
            user_context={"user_id": "tester", "client_id": "client-1"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized revert failure"


@pytest.mark.asyncio
async def test_get_prompt_history_maps_database_error(monkeypatch):
    logger_stub = _patch_endpoint_logger(monkeypatch)

    async def _allow_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_access",
        _allow_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_prompt_history(
            prompt_id=42,
            db=_BrokenPromptHistoryDb(),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to get prompt history"
    _assert_sanitized_error_log(
        logger_stub,
        "Prompt studio storage error getting prompt history",
    )


@pytest.mark.asyncio
async def test_get_prompt_history_maps_input_error_safe_message(monkeypatch):
    async def _allow_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_access",
        _allow_access,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_prompt_history(
            prompt_id=42,
            db=_BrokenPromptHistoryDb(
                InputError("raw history failure", safe_message="sanitized history failure")
            ),
            user_context={"user_id": "tester", "is_admin": False},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized history failure"


@pytest.mark.asyncio
async def test_preview_prompt_maps_input_error_safe_message(monkeypatch):
    async def _allow_access(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        prompts_endpoint,
        "require_project_access",
        _allow_access,
        raising=True,
    )
    monkeypatch.setattr(
        prompts_endpoint,
        "assemble_prompt_definition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InputError("raw preview failure", safe_message="sanitized preview failure")
        ),
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await preview_prompt(
            payload=StructuredPromptPreviewRequest(
                project_id=7,
                prompt_format="structured",
                prompt_schema_version=1,
                prompt_definition=_structured_prompt_definition_payload(),
                variables={"input": "SQLite FTS"},
            ),
            db=object(),
            security_config=object(),
            user_context={"user_id": "tester"},
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "sanitized preview failure"
