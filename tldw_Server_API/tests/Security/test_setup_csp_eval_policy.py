import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Security import setup_csp as setup_csp_module
from tldw_Server_API.app.core.Security.setup_csp import SetupCSPMiddleware


class _CapturingLogger:
    def __init__(self):
        self.records = []

    def debug(self, message, *args, **kwargs):
        self.records.append(("debug", message, args, dict(kwargs)))


def _joined_records(logger: _CapturingLogger) -> str:
    return "\n".join(f"{level} {message} {args!r} {kwargs!r}" for level, message, args, kwargs in logger.records)


def _capture_setup_csp_logs():
    messages: list[str] = []
    sink_id = setup_csp_module.logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )
    return messages, sink_id


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SetupCSPMiddleware)

    @app.get("/setup/ping")
    async def setup_ping():
        return {"ok": True}

    return app


def _script_src(header_val: str) -> str:
    parts = [p.strip() for p in (header_val or "").split(";")]
    for part in parts:
        if part.startswith("script-src"):
            return part
    return ""


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "Yes", "on", "Y"])
def test_setup_csp_no_eval_env_truthy_disables_eval(monkeypatch, truthy):
    monkeypatch.setenv("TLDW_SETUP_NO_EVAL", truthy)

    app = _make_app()
    client = TestClient(app)
    response = client.get("/setup/ping")
    script_src = _script_src(response.headers.get("Content-Security-Policy", ""))

    assert "'unsafe-inline'" in script_src
    assert "'unsafe-eval'" not in script_src


@pytest.mark.parametrize("falsy", ["0", "false", "False", "off", "n", "no"])
def test_setup_csp_no_eval_env_falsy_enables_eval(monkeypatch, falsy):
    monkeypatch.setenv("TLDW_SETUP_NO_EVAL", falsy)

    app = _make_app()
    client = TestClient(app)
    response = client.get("/setup/ping")
    script_src = _script_src(response.headers.get("Content-Security-Policy", ""))

    assert "'unsafe-inline'" in script_src
    assert "'unsafe-eval'" in script_src


def test_nonce_injection_regex_failure_log_is_sanitized(monkeypatch):
    class FailingScriptTagRegex:
        def sub(self, _repl, _html):
            raise RuntimeError("nonce regex failed at /private/setup-csp.html")

    monkeypatch.setattr(setup_csp_module, "_SCRIPT_TAG_RE", FailingScriptTagRegex())
    messages, sink_id = _capture_setup_csp_logs()

    try:
        html = setup_csp_module._inject_nonce_into_html(b"<script></script>", "nonce")
    finally:
        setup_csp_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert html == b"<script></script>"
    assert "Nonce injection regex failed" in joined
    assert "nonce regex failed" not in joined
    assert "/private/setup-csp.html" not in joined


def test_setup_csp_header_failure_log_is_sanitized(monkeypatch):
    logger = _CapturingLogger()

    def _raise_build_setup_csp(*_args, **_kwargs):
        raise RuntimeError("CSP header failed at /private/setup-csp.html")

    monkeypatch.setattr(setup_csp_module, "logger", logger)
    monkeypatch.setattr(setup_csp_module, "_build_setup_csp", _raise_build_setup_csp)

    app = _make_app()
    client = TestClient(app)
    response = client.get("/setup/ping")

    assert response.status_code == 200
    assert "Content-Security-Policy" not in response.headers
    joined = _joined_records(logger)
    assert "Setup CSP middleware failed to attach CSP header" in joined
    assert "CSP header failed" not in joined
    assert "/private/setup-csp.html" not in joined
    assert "exc_info" not in joined


def test_setup_csp_default_allows_eval(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_NO_EVAL", raising=False)

    app = _make_app()
    client = TestClient(app)
    response = client.get("/setup/ping")
    script_src = _script_src(response.headers.get("Content-Security-Policy", ""))

    assert "'unsafe-inline'" in script_src
    assert "'unsafe-eval'" in script_src
