from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import translate as translate_module
from tldw_Server_API.app.api.v1.schemas.translate_schemas import TranslateRequest


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.error_kwargs: list[dict[str, object]] = []

    def debug(self, *_args, **_kwargs) -> None:
        return None

    def error(self, message: str, *args, **kwargs) -> None:
        if args:
            self.errors.append(str(message).format(*args))
        else:
            self.errors.append(str(message))
        self.error_kwargs.append(dict(kwargs))


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [expected_message]
    rendered = " ".join(logger_stub.errors)
    assert "/private/" not in rendered
    assert "exploded" not in rendered
    assert all(not kwargs for kwargs in logger_stub.error_kwargs)


@pytest.mark.asyncio
async def test_translate_text_sanitizes_error_string_result(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(translate_module, "logger", logger_stub)
    monkeypatch.setattr(
        translate_module,
        "analyze",
        lambda **kwargs: "Error: translation backend exploded at /private/translate-cache",
    )

    request = TranslateRequest(
        text="Hello world",
        target_language="French",
        provider="openai",
    )

    with pytest.raises(HTTPException) as exc_info:
        await translate_module.translate_text(
            request,
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Translation failed"
    _assert_sanitized_error_log(logger_stub, "Translation failed")


@pytest.mark.asyncio
async def test_translate_text_sanitizes_unexpected_exception(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(translate_module, "logger", logger_stub)

    def _raise_unexpected_error(**kwargs):
        raise RuntimeError("translation backend exploded at /private/translate-cache")

    monkeypatch.setattr(translate_module, "analyze", _raise_unexpected_error)

    request = TranslateRequest(
        text="Hello world",
        target_language="French",
        provider="openai",
    )

    with pytest.raises(HTTPException) as exc_info:
        await translate_module.translate_text(
            request,
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Translation failed"
    _assert_sanitized_error_log(logger_stub, "Unexpected translation error")
