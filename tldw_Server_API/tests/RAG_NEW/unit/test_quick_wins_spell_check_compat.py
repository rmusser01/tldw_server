from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.RAG.rag_service import quick_wins as qw
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up
from tldw_Server_API.app.core.RAG.rag_service.quick_wins import QuerySpellChecker


class _LoggerStub:
    def __init__(self):
        self.debugs = []
        self.errors = []

    def debug(self, message):
        self.debugs.append(message)

    def error(self, message):
        self.errors.append(message)


class _FakeChecker:
    def check_query(self, query: str):
        corrected = query.replace("teh", "the")
        return {
            "original": query,
            "corrected": corrected,
            "has_errors": corrected != query,
            "corrections": {"teh": {"correction": "the", "suggestions": ["the"]}},
        }


class _NoopDebug:
    def log(self, *_args, **_kwargs):
        return None


class _RecordingLogger:
    def __init__(self):
        self.debugs = []
        self.infos = []

    def debug(self, message, *args, **kwargs):
        self.debugs.append({"message": message, "args": args, "kwargs": kwargs})

    def info(self, message, *args, **kwargs):
        self.infos.append({"message": message, "args": args, "kwargs": kwargs})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spell_check_query_accepts_raw_string(monkeypatch):
    monkeypatch.setattr(qw, "get_spell_checker", lambda: _FakeChecker())

    corrected = await qw.spell_check_query("teh query")

    assert corrected == "the query"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spell_check_query_context_still_supported(monkeypatch):
    monkeypatch.setattr(qw, "get_spell_checker", lambda: _FakeChecker())
    monkeypatch.setattr(qw, "get_debug_mode", lambda: _NoopDebug())

    context = SimpleNamespace(
        config={"spell_check": {"enabled": True, "auto_correct": True}},
        query="teh query",
        metadata={},
    )

    out = await qw.spell_check_query(context)

    assert out is context
    assert context.query == "the query"
    assert context.metadata.get("original_query_before_correction") == "teh query"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_spell_check_query_metadata_attach_failure_log_is_sanitized(monkeypatch):
    class _ContextWithRejectingMetadata:
        def __init__(self):
            self.config = {"spell_check": {"enabled": True, "auto_correct": True}}
            self.query = "teh query"
            self._metadata = "not-a-dict"
            self.metadata_set_attempts = 0

        @property
        def metadata(self):
            return self._metadata

        @metadata.setter
        def metadata(self, _value):
            self.metadata_set_attempts += 1
            raise RuntimeError("cannot attach metadata for /tmp/source token=secret")

    logger = _RecordingLogger()
    monkeypatch.setattr(qw, "get_spell_checker", lambda: _FakeChecker())
    monkeypatch.setattr(qw, "get_debug_mode", lambda: _NoopDebug())
    monkeypatch.setattr(qw, "logger", logger)

    context = _ContextWithRejectingMetadata()

    out = await qw.spell_check_query(context)

    assert out is context
    assert context.query == "the query"
    assert context.metadata == "not-a-dict"
    assert context.metadata_set_attempts == 1
    assert logger.debugs == [
        {
            "message": "Quick wins failed to attach metadata mapping to context",
            "args": (),
            "kwargs": {},
        }
    ]
    rendered_logs = repr(logger.debugs)
    assert "/tmp/source" not in rendered_logs
    assert "token=secret" not in rendered_logs
    assert "cannot attach metadata" not in rendered_logs
    assert "exc_info" not in rendered_logs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_spell_check_no_config_attr_error(monkeypatch):
    monkeypatch.setattr(qw, "get_spell_checker", lambda: _FakeChecker())

    class _FakeMultiRetriever:
        def __init__(self, *_args, **_kwargs):
            self.retrievers = {}

        async def retrieve(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(up, "MultiDatabaseRetriever", _FakeMultiRetriever)

    result = await up.unified_rag_pipeline(
        query="teh query",
        spell_check=True,
        enable_generation=False,
        enable_reranking=False,
        enable_cache=False,
        sources=["media_db"],
        fallback_on_error=False,
    )

    assert isinstance(result, UnifiedRAGResponse)
    assert not any("no attribute 'config'" in err for err in (result.errors or []))
    assert result.metadata.get("original_query") == "teh query"
    assert result.metadata.get("corrected_query") == "the query"


@pytest.mark.unit
def test_query_spell_checker_preserves_ambiguous_media_entity_names():
    checker = QuerySpellChecker()

    frieza_result = checker.check_query("frieza new form")
    goku_result = checker.check_query("goku one inch punch on frieza")

    assert frieza_result["corrected"] == "frieza new form"
    assert frieza_result["has_errors"] is False
    assert goku_result["corrected"] == "goku one inch punch on frieza"
    assert goku_result["has_errors"] is False


@pytest.mark.unit
def test_cost_tracker_tokenizer_fallback_log_omits_backend_exception(monkeypatch):
    logger_stub = _LoggerStub()

    def raise_tokenizer_error(_model):
        raise RuntimeError("tokenizer exploded with /private/user/token")

    monkeypatch.setattr(qw, "logger", logger_stub)
    monkeypatch.setattr(qw.tiktoken, "encoding_for_model", raise_tokenizer_error)
    monkeypatch.setattr(
        qw.tiktoken,
        "get_encoding",
        lambda _name: SimpleNamespace(encode=lambda text: text.split()),
    )

    tracker = qw.CostTracker()

    assert tracker.count_tokens("one two") == 2
    assert logger_stub.debugs == ["Falling back to base tokenizer"]
    assert "exploded" not in repr(logger_stub.debugs)
    assert "/private/user/token" not in repr(logger_stub.debugs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_webhook_exception_fallback_log_and_history_omit_backend_exception(monkeypatch):
    logger_stub = _LoggerStub()

    async def raise_webhook_error(**_kwargs):
        raise RuntimeError("webhook secret leaked from /private/user/hook")

    monkeypatch.setattr(qw, "logger", logger_stub)
    monkeypatch.setattr(qw, "afetch", raise_webhook_error)

    notifier = qw.WebhookNotifier()
    sent = await notifier.send_notification(
        "https://example.invalid/webhook",
        "batch_complete",
        {"job_id": "job-1"},
    )

    assert sent is False
    assert logger_stub.errors == ["Webhook error"]
    assert notifier.webhook_history[0]["error"] == "webhook_request_failed"
    rendered = repr(logger_stub.errors) + repr(notifier.webhook_history)
    assert "secret leaked" not in rendered
    assert "/private/user/hook" not in rendered
