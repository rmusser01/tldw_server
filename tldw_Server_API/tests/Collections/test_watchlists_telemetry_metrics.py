import importlib

from loguru import logger as loguru_logger

from tldw_Server_API.app.core import Metrics as metrics_module
from tldw_Server_API.app.core.Watchlists import watchlists_telemetry_metrics as metrics


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        del kwargs
        self.debugs.append(message.format(*args) if args else message)


class _FailingMetricsRegistry:
    def increment(self, *_args, **_kwargs) -> None:
        raise RuntimeError("metrics backend exploded /private/metrics.db")

    def observe(self, *_args, **_kwargs) -> None:
        raise RuntimeError("metrics backend exploded /private/metrics.db")


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debugs == [expected_message]
    assert "metrics backend exploded" not in str(logger_stub.debugs)
    assert "/private/metrics.db" not in str(logger_stub.debugs)


def test_record_onboarding_ingest_result_sanitizes_metrics_emit_failure(monkeypatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(metrics, "logger", logger_stub)
    monkeypatch.setattr(metrics, "register_watchlists_telemetry_metrics", lambda: None)
    monkeypatch.setattr(metrics, "get_metrics_registry", lambda: _FailingMetricsRegistry())

    metrics.record_onboarding_ingest_result("ok")

    _assert_sanitized_debug_log(
        logger_stub,
        "watchlists telemetry metrics ingest emit skipped",
    )


def test_record_summary_request_sanitizes_metrics_emit_failure(monkeypatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(metrics, "logger", logger_stub)
    monkeypatch.setattr(metrics, "register_watchlists_telemetry_metrics", lambda: None)
    monkeypatch.setattr(metrics, "get_metrics_registry", lambda: _FailingMetricsRegistry())

    metrics.record_summary_request("summary", "ok", 0.1)

    _assert_sanitized_debug_log(
        logger_stub,
        "watchlists telemetry metrics summary emit skipped",
    )


def test_import_time_registration_sanitizes_metrics_registration_failure(monkeypatch) -> None:
    logger_stub = _LoggerStub()

    def failing_get_metrics_registry():
        raise RuntimeError("metrics backend exploded /private/metrics.db")

    with monkeypatch.context() as patch_context:
        patch_context.setattr(metrics_module, "get_metrics_registry", failing_get_metrics_registry)
        patch_context.setattr(loguru_logger, "debug", logger_stub.debug)

        importlib.reload(metrics)

    importlib.reload(metrics)

    _assert_sanitized_debug_log(
        logger_stub,
        "watchlists telemetry metrics registration skipped",
    )
