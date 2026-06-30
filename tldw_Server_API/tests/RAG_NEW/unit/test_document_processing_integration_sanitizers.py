import pytest

from tldw_Server_API.app.core.RAG.rag_service import document_processing_integration as dpi
from tldw_Server_API.app.core.RAG.rag_service.document_processing_integration import (
    DocumentProcessor,
    ProcessingConfig,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.debugs: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    def debug(self, message: str) -> None:
        self.debugs.append(str(message))


def _processor() -> DocumentProcessor:
    return DocumentProcessor(
        ProcessingConfig(
            clean_artifacts=False,
            fix_encoding=False,
            detect_structure=False,
            enrich_metadata=False,
            optimize_boundaries=False,
            merge_small_chunks=False,
        )
    )


@pytest.mark.asyncio
async def test_chunking_error_fallback_log_omits_raw_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(dpi, "logger", logger_stub)
    monkeypatch.setattr(dpi, "CHUNK_LIB_AVAILABLE", True)

    def fail_chunking(*_args, **_kwargs):
        raise dpi.ChunkingError(
            "chunk failed for /private/rag/documents/source.txt?token=secret-token"
        )

    monkeypatch.setattr(dpi, "improved_chunking_process", fail_chunking)

    chunks = await _processor().process_document(
        "sensitive source content",
        source="/private/rag/documents/source.txt?token=secret-token",
    )

    assert chunks == []
    assert logger_stub.errors == ["Chunking error during document processing: ChunkingError"]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "chunk failed" not in joined


@pytest.mark.asyncio
async def test_unexpected_processing_fallback_log_omits_raw_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(dpi, "logger", logger_stub)
    monkeypatch.setattr(dpi, "CHUNK_LIB_AVAILABLE", True)

    def fail_processing(*_args, **_kwargs):
        raise RuntimeError(
            "unexpected failure for /private/rag/documents/source.txt?token=secret-token"
        )

    monkeypatch.setattr(dpi, "improved_chunking_process", fail_processing)

    chunks = await _processor().process_document(
        "sensitive source content",
        source="/private/rag/documents/source.txt?token=secret-token",
    )

    assert chunks == []
    assert logger_stub.errors == [
        "Unexpected error during document processing: RuntimeError"
    ]
    joined = "\n".join(logger_stub.errors)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "unexpected failure" not in joined


def test_mojibake_fix_fallback_debug_omits_raw_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(dpi, "logger", logger_stub)

    class _ExplodingText(str):
        def replace(self, *_args: object, **_kwargs: object) -> "_ExplodingText":
            return self

        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            raise UnicodeError(
                "mojibake repair failed for /private/rag/source.txt token=secret-token"
            )

    content = _ExplodingText("Safe visible content")
    fixed = DocumentProcessor()._fix_encoding(content)

    assert fixed == content
    assert logger_stub.debugs == [
        "Mojibake fix failed; returning replacements-only content"
    ]
    joined = "\n".join(logger_stub.debugs)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "mojibake repair failed" not in joined
