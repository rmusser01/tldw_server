"""Sanitizer coverage for RAG table serialization fallback logs."""

from tldw_Server_API.app.core.RAG.rag_service import table_serialization as ts


def _capture_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = ts.logger.add(lambda message: messages.append(str(message)), level=level)
    return messages, sink_id


def test_json_detection_fallback_log_omits_parser_exception_details(monkeypatch):
    """JSON detection fallback should not expose raw parser exception details."""

    def broken_json_loads(_text):
        raise ValueError("parse failed for /private/rag-table-detect.db?token=secret")

    monkeypatch.setattr(ts.json, "loads", broken_json_loads)

    messages, sink_id = _capture_logs()
    try:
        assert ts.TableParser.detect_format("not a table") is None
    finally:
        ts.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Table JSON detection failed" in joined
    assert "/private/" not in joined
    assert "rag-table-detect.db" not in joined
    assert "secret" not in joined


def test_process_table_warning_omits_parser_exception_details(monkeypatch):
    """Single-table fallback logs should not expose raw parser exception details."""

    class BrokenParser:
        def parse(self, _table_text, _format):
            raise ValueError("table parse failed for /private/rag-process.db?token=secret")

    processor = ts.TableProcessor()
    processor.parser = BrokenParser()

    messages, sink_id = _capture_logs(level="WARNING")
    try:
        result = processor.process_table("| a |\n|---|\n| b |", ts.TableFormat.MARKDOWN)
    finally:
        ts.logger.remove(sink_id)

    assert result == {
        "error": "table parse failed for /private/rag-process.db?token=secret",
        "original": "| a |\n|---|\n| b |",
        "search_text": "| a |\n|---|\n| b |",
    }
    joined = "\n".join(messages)
    assert "Failed to process table" in joined
    assert "/private/" not in joined
    assert "rag-process.db" not in joined
    assert "secret" not in joined


def test_process_document_tables_warning_omits_parser_exception_details(monkeypatch):
    """Document table fallback logs should not expose raw parser exception details."""

    class BrokenParser:
        def parse(self, _table_text, _format):
            raise ValueError("table parse failed for /private/rag-document.db?token=secret")

    document = "before\n| a |\n|---|\n| b |\nafter"
    processor = ts.TableProcessor()
    processor.parser = BrokenParser()

    messages, sink_id = _capture_logs(level="WARNING")
    try:
        processed_text, metadata = processor.process_document_tables(document)
    finally:
        ts.logger.remove(sink_id)

    assert processed_text == document
    assert metadata == []
    joined = "\n".join(messages)
    assert "Failed to process table" in joined
    assert "/private/" not in joined
    assert "rag-document.db" not in joined
    assert "secret" not in joined
