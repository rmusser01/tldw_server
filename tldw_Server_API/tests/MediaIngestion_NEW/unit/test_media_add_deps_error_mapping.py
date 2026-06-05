import inspect

import pytest
from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import media_add_deps

pytestmark = pytest.mark.unit


_SENSITIVE_MARKERS = (
    "/private/tmp",
    "secret-token",
    "raw-secret-value",
    "form backend exploded",
    "Traceback",
    "JSONDecodeError",
    "ValueError",
)


class _LoguruCapture:
    def __init__(self):
        self.records = []
        self._handler_id = None

    def __enter__(self):
        self._handler_id = logger.add(
            self.records.append,
            level="DEBUG",
            format="{level} {message} {extra} {exception}",
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        logger.remove(self._handler_id)

    @property
    def debug_records(self):
        return [
            message.record
            for message in self.records
            if message.record["level"].name == "DEBUG"
        ]

    @property
    def error_records(self):
        return [
            message.record
            for message in self.records
            if message.record["level"].name == "ERROR"
        ]


def _render_log_record(record) -> str:
    return " ".join(
        [
            record["message"],
            repr(record["extra"]),
            repr(record["exception"]),
        ]
    )


def _assert_debug_log_is_sanitized(capture: _LoguruCapture, expected_message: str):
    matching_records = [
        record
        for record in capture.debug_records
        if record["message"] == expected_message
    ]
    assert len(matching_records) == 1

    record = matching_records[0]
    assert "exc_info" not in record["extra"]
    assert record["exception"] is None

    rendered_record = _render_log_record(record)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_record


def _assert_error_log_is_sanitized(capture: _LoguruCapture, expected_message: str):
    matching_records = [
        record
        for record in capture.error_records
        if record["message"] == expected_message
    ]
    assert len(matching_records) == 1

    record = matching_records[0]
    assert "exc_info" not in record["extra"]
    assert record["exception"] is None

    rendered_record = _render_log_record(record)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_record


def _form_kwargs(**overrides):
    kwargs = {}
    for name, parameter in inspect.signature(
        media_add_deps.get_add_media_form
    ).parameters.items():
        default = parameter.default
        if hasattr(default, "default"):
            kwargs[name] = default.default
        else:
            kwargs[name] = default
    kwargs["media_type"] = "video"
    kwargs["transcription_model"] = None
    kwargs.update(overrides)
    return kwargs


@pytest.mark.asyncio
async def test_get_add_media_form_sanitizes_unexpected_form_error(monkeypatch):
    def _raise_form_error(**_kwargs):
        raise RuntimeError("form backend exploded /private/tmp/raw-secret-value?token=secret-token")

    monkeypatch.setattr(media_add_deps, "AddMediaForm", _raise_form_error)

    with _LoguruCapture() as logs:
        with pytest.raises(HTTPException) as excinfo:
            await media_add_deps.get_add_media_form(media_type="video", transcription_model=None)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal server error during form processing"
    _assert_error_log_is_sanitized(
        logs,
        "Unexpected error creating AddMediaForm: RuntimeError",
    )


@pytest.mark.asyncio
async def test_get_add_media_form_sanitizes_urls_json_fallback_debug_log():
    raw_urls_value = (
        "['https://example.test/watch?token=secret-token', "
        "'/private/tmp/raw-secret-value.mp4']"
    )

    with _LoguruCapture() as logs:
        form = await media_add_deps.get_add_media_form(
            **_form_kwargs(urls=[raw_urls_value])
        )

    assert form.urls == [raw_urls_value]
    _assert_debug_log_is_sanitized(
        logs,
        "Failed to parse JSON list for 'urls' form field; using raw fallback",
    )


@pytest.mark.asyncio
async def test_get_add_media_form_carries_collection_fallback_binding_fields():
    form = await media_add_deps.get_add_media_form(
        **_form_kwargs(
            urls=["https://example.test/watch"],
            media_collection_id="42",
            media_collection_item_id="88",
            media_ingest_job_id="1234",
        )
    )

    assert form.media_collection_id == 42
    assert form.media_collection_item_id == 88
    assert form.media_ingest_job_id == "1234"


@pytest.mark.asyncio
async def test_get_add_media_form_sanitizes_context_window_fallback_debug_log():
    raw_context_window_size = "/private/tmp/raw-secret-value?token=secret-token"

    with _LoguruCapture() as logs:
        with pytest.raises(HTTPException) as excinfo:
            await media_add_deps.get_add_media_form(
                **_form_kwargs(
                    urls=["https://example.test/watch"],
                    context_window_size=raw_context_window_size,
                )
            )

    assert excinfo.value.status_code == media_add_deps.HTTP_422_UNPROCESSABLE
    _assert_debug_log_is_sanitized(
        logs,
        "Failed to coerce context_window_size from form data",
    )
