from collections import OrderedDict
from pathlib import Path

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import Slides_DB_Deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Slides.slides_db import SchemaError, SlidesDatabaseError


_PRIVATE_PATH = "/Users/alice/private/slides/Slides.db"
_SECRET_TOKEN = "slides-secret-token-123"
_RAW_DETAIL = f"backend exploded at {_PRIVATE_PATH} using token={_SECRET_TOKEN}"
_SENSITIVE_USER_ID = 424242
_SENSITIVE_CACHE_KEY = "raw-slides-cache-user-777"


def _user(user_id: int = 42) -> User:
    return User(id=user_id, username="slides-user")


def _capture_slides_dep_logs(
    level: str = "ERROR",
    log_format: str = "{message}",
) -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = deps.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level=level,
        format=log_format,
    )
    return messages, sink_id


def _capture_formatted_slides_dep_logs(
    level: str = "ERROR",
    log_format: str = "{message}\n{exception}",
) -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = deps.logger.add(
        lambda message: messages.append(str(message)),
        level=level,
        format=log_format,
    )
    return messages, sink_id


def _assert_sensitive_text_not_logged(rendered_logs: str, *sensitive_values: str) -> None:
    for value in sensitive_values:
        assert value not in rendered_logs


def test_get_slides_db_maps_schema_error(monkeypatch):
    deps.cleanup_slides_db_cache()

    def fail_create(*args, **kwargs):
        raise SchemaError("schema exploded")

    monkeypatch.setattr(deps, "SlidesDatabase", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        deps.get_slides_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


def test_get_slides_db_maps_base_database_error(monkeypatch):
    deps.cleanup_slides_db_cache()

    def fail_create(*args, **kwargs):
        raise SlidesDatabaseError("backend exploded")

    monkeypatch.setattr(deps, "SlidesDatabase", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        deps.get_slides_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Slides DB unavailable"


@pytest.mark.parametrize(
    ("exc", "expected_detail", "expected_error_type"),
    [
        (SchemaError(_RAW_DETAIL), "Database schema error", "SchemaError"),
        (SlidesDatabaseError(_RAW_DETAIL), "Slides DB unavailable", "SlidesDatabaseError"),
        (RuntimeError(_RAW_DETAIL), "Slides DB unavailable", "RuntimeError"),
    ],
)
def test_get_slides_db_sanitizes_initialization_failure_logs(
    monkeypatch,
    exc: Exception,
    expected_detail: str,
    expected_error_type: str,
):
    deps.cleanup_slides_db_cache()

    def fail_create(*args, **kwargs):
        raise exc

    monkeypatch.setattr(deps, "SlidesDatabase", fail_create)
    messages, sink_id = _capture_slides_dep_logs()
    try:
        with pytest.raises(HTTPException) as exc_info:
            deps.get_slides_db_for_user(current_user=_user(_SENSITIVE_USER_ID))
    finally:
        deps.logger.remove(sink_id)
        deps.cleanup_slides_db_cache()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail

    rendered_logs = "\n".join(messages)
    assert f"error_type={expected_error_type}" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        str(_SENSITIVE_USER_ID),
        _PRIVATE_PATH,
        _SECRET_TOKEN,
        _RAW_DETAIL,
        "backend exploded",
    )


def test_cleanup_slides_db_cache_sanitizes_close_failure_logs():
    class FailingCloseDb:
        def close_connection(self):
            raise RuntimeError(_RAW_DETAIL)

    with deps._slides_db_lock:
        deps._slides_db_instances = OrderedDict(
            [(_SENSITIVE_CACHE_KEY, FailingCloseDb())]
        )

    messages, sink_id = _capture_slides_dep_logs(level="WARNING")
    try:
        deps.cleanup_slides_db_cache()
    finally:
        deps.logger.remove(sink_id)

    assert deps._slides_db_instances == OrderedDict()

    rendered_logs = "\n".join(messages)
    assert "error_type=RuntimeError" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        _SENSITIVE_CACHE_KEY,
        _PRIVATE_PATH,
        _SECRET_TOKEN,
        _RAW_DETAIL,
        "backend exploded",
    )


def test_get_slides_db_sanitizes_evicted_close_failure_logs(monkeypatch):
    class FailingCloseDb:
        def close_connection(self):
            raise RuntimeError(_RAW_DETAIL)

    class NewDb:
        def close_connection(self):
            return None

    new_db = NewDb()

    def create_db(*args, **kwargs):
        return new_db

    monkeypatch.setattr(deps, "_MAX_CACHED_SLIDES_DB", 1)
    monkeypatch.setattr(
        deps,
        "_get_slides_db_path_for_user",
        lambda user_id: Path("/tmp/slides.db"),
    )
    monkeypatch.setattr(deps, "SlidesDatabase", create_db)
    with deps._slides_db_lock:
        deps._slides_db_instances = OrderedDict(
            [(_SENSITIVE_CACHE_KEY, FailingCloseDb())]
        )

    messages, sink_id = _capture_slides_dep_logs(level="WARNING")
    try:
        result = deps.get_slides_db_for_user(current_user=_user())
    finally:
        deps.logger.remove(sink_id)
        deps.cleanup_slides_db_cache()

    assert result is new_db

    rendered_logs = "\n".join(messages)
    assert "error_type=RuntimeError" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        _SENSITIVE_CACHE_KEY,
        _PRIVATE_PATH,
        _SECRET_TOKEN,
        _RAW_DETAIL,
        "backend exploded",
    )


def test_try_get_slides_db_sanitizes_http_exception_fallback_logs(monkeypatch):
    def fail_resolve(*args, **kwargs):
        raise HTTPException(status_code=503, detail=_RAW_DETAIL)

    monkeypatch.setattr(deps, "get_slides_db_for_user", fail_resolve)
    messages, sink_id = _capture_slides_dep_logs(level="DEBUG")
    try:
        result = deps.try_get_slides_db_for_user(current_user=_user(_SENSITIVE_USER_ID))
    finally:
        deps.logger.remove(sink_id)

    assert result is None

    rendered_logs = "\n".join(messages)
    _assert_sensitive_text_not_logged(
        rendered_logs,
        str(_SENSITIVE_USER_ID),
        _PRIVATE_PATH,
        _SECRET_TOKEN,
        _RAW_DETAIL,
        "backend exploded",
    )
    assert "status_code=503" in rendered_logs


def test_try_get_slides_db_sanitizes_unexpected_exception_fallback_logs(monkeypatch):
    def fail_resolve(*args, **kwargs):
        raise RuntimeError(_RAW_DETAIL)

    monkeypatch.setattr(deps, "get_slides_db_for_user", fail_resolve)
    messages, sink_id = _capture_formatted_slides_dep_logs(level="ERROR")
    try:
        result = deps.try_get_slides_db_for_user(current_user=_user(_SENSITIVE_USER_ID))
    finally:
        deps.logger.remove(sink_id)

    assert result is None

    rendered_logs = "\n".join(messages)
    _assert_sensitive_text_not_logged(
        rendered_logs,
        str(_SENSITIVE_USER_ID),
        _PRIVATE_PATH,
        _SECRET_TOKEN,
        _RAW_DETAIL,
        "backend exploded",
    )
    assert "error_type=RuntimeError" in rendered_logs
