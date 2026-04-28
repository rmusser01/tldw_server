import pytest

from tldw_Server_API.app.core.RAG.rag_service import citations
from tldw_Server_API.app.core.RAG.rag_service.citations import (
    AcademicCitationFormatter,
)

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(str(message))


class _MetricsStub:
    def increment(self, *_args, **_kwargs) -> None:
        return None


def test_citation_date_parse_log_omits_raw_value_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(citations, "logger", logger_stub)
    monkeypatch.setattr(citations, "get_metrics_registry", lambda: _MetricsStub())

    raw_date = "not-a-date /private/rag/citation-source.json?token=secret-token"
    formatter = AcademicCitationFormatter()

    formatted = formatter._format_date(raw_date, style="mla")

    assert formatted == raw_date
    assert logger_stub.debugs == ["Citation date parse failed; returning raw"]
    joined = "\n".join(logger_stub.debugs)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "Invalid isoformat" not in joined
