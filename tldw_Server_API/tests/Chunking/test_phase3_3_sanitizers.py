from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chunking
from tldw_Server_API.app.api.v1.schemas.chunking_schema import (
    ChunkingOptionsRequest,
    ChunkingTextRequest,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.messages = []

    def debug(self, message, *args, **kwargs):
        self.messages.append(("debug", str(message), args, kwargs))

    def info(self, message, *args, **kwargs):
        self.messages.append(("info", str(message), args, kwargs))

    def warning(self, message, *args, **kwargs):
        self.messages.append(("warning", str(message), args, kwargs))

    def error(self, message, *args, **kwargs):
        self.messages.append(("error", str(message), args, kwargs))


class _BadRsplitFilename(str):
    def rsplit(self, *_args, **_kwargs):
        raise RuntimeError("backend exploded at /private/chunking/path SECRET_TOKEN")


class _BadStrFilename(str):
    def __new__(cls):
        obj = str.__new__(cls, "safe.py")
        obj.calls = 0
        return obj

    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("backend exploded at /private/chunking/path SECRET_TOKEN")
        return "safe.py"


class _FakeUploadFile:
    def __init__(self, filename: str):
        self.filename = filename
        self.closed = False

    async def read(self) -> bytes:
        return b"print('hello')"

    async def close(self) -> None:
        self.closed = True


def _render_logs(logger_stub: _LoggerStub) -> str:
    return "\n".join(
        " ".join([level, message, repr(args), repr(kwargs)])
        for level, message, args, kwargs in logger_stub.messages
    )


def _assert_logs_are_sanitized(logger_stub: _LoggerStub) -> None:
    rendered = _render_logs(logger_stub)
    assert "backend exploded" not in rendered
    assert "/private/" not in rendered
    assert "SECRET_TOKEN" not in rendered
    assert "exc_info" not in rendered


def _successful_chunker(text: str, options: dict[str, Any], *_args) -> list[dict[str, Any]]:
    return [{"text": text, "metadata": {"method": options["method"]}}]


def _failing_chunker(*_args) -> list[dict[str, Any]]:
    raise RuntimeError("backend exploded at /private/chunking/path SECRET_TOKEN")


def _text_request(file_name: str) -> ChunkingTextRequest:
    request = ChunkingTextRequest(
        text_content="print('hello')",
        file_name="safe.py",
        options=ChunkingOptionsRequest(
            method="words",
            max_size=20,
            overlap=0,
            language="",
        ),
    )
    request.file_name = file_name
    return request


def _file_endpoint_kwargs(file: _FakeUploadFile) -> dict[str, Any]:
    return {
        "http_request": object(),
        "file": file,
        "method": "words",
        "max_size": 20,
        "overlap": 0,
        "language": None,
        "tokenizer_name_or_path": "gpt2",
        "code_mode": None,
        "adaptive": False,
        "multi_level": False,
        "custom_chapter_pattern": None,
        "semantic_similarity_threshold": 0.7,
        "semantic_overlap_sentences": 2,
        "json_chunkable_data_key": "data",
        "summarization_detail": 0.5,
        "llm_step_temperature": None,
        "llm_step_system_prompt": None,
        "llm_step_max_tokens": None,
        "current_user": object(),
    }


@pytest.mark.asyncio
async def test_chunk_text_language_inference_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(chunking, "logger", logger_stub)
    monkeypatch.setattr(chunking, "improved_chunking_process", _successful_chunker)

    response = await chunking.process_text_for_chunking_json(
        _text_request(_BadStrFilename()),
        http_request=object(),
        current_user=object(),
        media_db=None,
    )

    assert response.chunks[0].text == "print('hello')"
    assert any(
        level == "debug" and "Failed to infer code language from file extension" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "RuntimeError" in rendered
    _assert_logs_are_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_chunk_text_unexpected_error_log_and_detail_are_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(chunking, "logger", logger_stub)
    monkeypatch.setattr(chunking, "improved_chunking_process", _failing_chunker)

    with pytest.raises(HTTPException) as exc_info:
        await chunking.process_text_for_chunking_json(
            _text_request("safe.py"),
            http_request=object(),
            current_user=object(),
            media_db=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An internal error occurred during text chunking"
    assert any(
        level == "error" and "Unexpected error during chunking process" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "RuntimeError" in rendered
    _assert_logs_are_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_chunk_file_language_inference_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    file = _FakeUploadFile(_BadRsplitFilename("safe.py"))
    monkeypatch.setattr(chunking, "logger", logger_stub)
    monkeypatch.setattr(chunking, "improved_chunking_process", _successful_chunker)

    response = await chunking.process_file_for_chunking(**_file_endpoint_kwargs(file))

    assert file.closed is True
    assert response.chunks[0].text == "print('hello')"
    assert any(
        level == "debug" and "Failed to infer cleaned form language from file extension" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "RuntimeError" in rendered
    _assert_logs_are_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_chunk_file_unexpected_error_log_and_detail_are_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    file = _FakeUploadFile("safe.py")
    monkeypatch.setattr(chunking, "logger", logger_stub)
    monkeypatch.setattr(chunking, "improved_chunking_process", _failing_chunker)

    with pytest.raises(HTTPException) as exc_info:
        await chunking.process_file_for_chunking(**_file_endpoint_kwargs(file))

    assert file.closed is True
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal error during file chunking"
    assert any(
        level == "error" and "Unexpected error during chunking file" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "RuntimeError" in rendered
    _assert_logs_are_sanitized(logger_stub)
