import pytest

from tldw_Server_API.app.core.RAG.rag_service import batch_processing as bp


pytestmark = pytest.mark.unit

_SENSITIVE_MESSAGE = "batch backend failed for /private/rag/batch.db token=secret-token"


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.debugs: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    def info(self, message: str) -> None:
        self.infos.append(str(message))

    def debug(self, message: str) -> None:
        self.debugs.append(str(message))


def _assert_no_sensitive_fragments(messages: list[str]) -> None:
    joined = "\n".join(messages)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "batch backend failed" not in joined


@pytest.mark.asyncio
async def test_process_batch_generic_failure_metadata_and_log_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(bp, "logger", logger_stub)
    processor = bp.BatchProcessor(max_concurrent=1, max_retries=1)

    async def _fail_job(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(_SENSITIVE_MESSAGE)

    monkeypatch.setattr(processor, "_process_job", _fail_job)

    job = await processor.process_batch(
        ["private query"],
        lambda _query, _config: "unused",
    )

    assert job.status is bp.BatchStatus.FAILED
    assert job.metadata["error"] == "Batch processing failed"
    assert any(message == "Batch job failed" for message in logger_stub.errors)
    _assert_no_sensitive_fragments(logger_stub.errors)
    _assert_no_sensitive_fragments([str(job.metadata)])


@pytest.mark.asyncio
async def test_query_retry_failure_error_and_log_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(bp, "logger", logger_stub)
    processor = bp.BatchProcessor(max_concurrent=1, max_retries=1)

    async def _fail_query(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError(_SENSITIVE_MESSAGE)

    job = await processor.process_batch(
        ["private query"],
        _fail_query,
    )

    assert job.status is bp.BatchStatus.FAILED
    assert job.queries[0].status is bp.BatchStatus.FAILED
    assert job.queries[0].error == "Query processing failed"
    assert logger_stub.errors == ["Query failed after retry attempts"]
    _assert_no_sensitive_fragments(logger_stub.errors)
    _assert_no_sensitive_fragments([job.queries[0].error or ""])
