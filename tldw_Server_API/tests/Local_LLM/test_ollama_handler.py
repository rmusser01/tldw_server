import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Local_LLM.Ollama_Handler import OllamaHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import OllamaConfig
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import InferenceError, ServerError
from tldw_Server_API.app.core.exceptions import NetworkError


@pytest.mark.asyncio
async def test_ollama_inference_404_pull_then_retry(monkeypatch):
    cfg = OllamaConfig()
    handler = OllamaHandler(cfg, global_app_config={})

    # Pretend ollama is installed
    monkeypatch.setattr(handler, "is_ollama_installed", lambda: asyncio.sleep(0, result=True))
    # Model not available initially
    monkeypatch.setattr(handler, "is_model_available", lambda model_name: asyncio.sleep(0, result=False))
    # pull_model succeeds
    monkeypatch.setattr(handler, "pull_model", lambda model_name: asyncio.sleep(0, result="ok"))

    # First request_json raises 404, second returns success
    import tldw_Server_API.app.core.Local_LLM.Ollama_Handler as ol_mod

    calls = {"n": 0}

    async def fake_request_json(client, method, url, json=None, headers=None, retries=2, backoff=0.0):
        if calls["n"] == 0:
            calls["n"] += 1
            exc = NetworkError("model not found")
            setattr(exc, "status_code", 404)
            raise exc
        return {"response": "ok"}

    monkeypatch.setattr(ol_mod, "request_json", fake_request_json)

    result = await handler.inference(model_name="m", prompt="hi")
    assert result["response"] == "ok"


@pytest.mark.asyncio
async def test_ollama_inference_start_then_retry(monkeypatch):
    cfg = OllamaConfig()
    handler = OllamaHandler(cfg, global_app_config={})

    # Model is available
    monkeypatch.setattr(handler, "is_model_available", lambda model_name: asyncio.sleep(0, result=True))

    # First request_json raises connection error; then serve + ready + retry succeed
    import tldw_Server_API.app.core.Local_LLM.Ollama_Handler as ol_mod

    calls = {"n": 0}

    async def fake_request_json(client, method, url, json=None, headers=None, retries=2, backoff=0.0):
        if calls["n"] == 0:
            calls["n"] += 1
            raise NetworkError("connection error")
        return {"response": "ok"}

    monkeypatch.setattr(ol_mod, "request_json", fake_request_json)

    # serve_model called, readiness ok
    monkeypatch.setattr(
        handler,
        "serve_model",
        lambda model_name, port=None, host="127.0.0.1": asyncio.sleep(0, result={"status": "started"}),
    )
    monkeypatch.setattr(
        ol_mod, "wait_for_http_ready", lambda base_url, timeout_total=30.0, interval=0.5: asyncio.sleep(0, result=True)
    )

    result = await handler.inference(model_name="m", prompt="hi")
    assert result["response"] == "ok"


@pytest.mark.asyncio
async def test_ollama_serve_model_not_ready(monkeypatch):
    cfg = OllamaConfig()
    handler = OllamaHandler(cfg, global_app_config={})

    monkeypatch.setattr(handler, "is_ollama_installed", lambda: asyncio.sleep(0, result=True))

    import tldw_Server_API.app.core.Local_LLM.Ollama_Handler as ol_mod

    monkeypatch.setattr(ol_mod.psutil, "net_connections", lambda: [])

    class DummyStderr:
        async def read(self):
            return b""

    class DummyProc:
        def __init__(self):
            self.pid = 123
            self.returncode = None
            self.stderr = DummyStderr()

        async def wait(self):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

    async def fake_cpe(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)
    monkeypatch.setattr(ol_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=False))

    with pytest.raises(ServerError):
        await handler.serve_model("m", port=11435)


@pytest.mark.asyncio
async def test_ollama_stop_server_pid_failure_is_sanitized(monkeypatch):
    cfg = OllamaConfig()
    handler = OllamaHandler(cfg, global_app_config={})

    monkeypatch.setattr(handler, "is_ollama_installed", lambda: asyncio.sleep(0, result=True))

    def fail_terminate(_pid):
        raise RuntimeError("termination failed at /private/ollama.pid")

    monkeypatch.setattr(handler, "_terminate_process", fail_terminate)

    result = await handler.stop_server(pid=123)

    assert result == "Error stopping Ollama server PID 123"
    assert "termination failed" not in result
    assert "/private/ollama.pid" not in result


@pytest.mark.asyncio
async def test_ollama_stop_server_port_failure_is_sanitized(monkeypatch):
    cfg = OllamaConfig()
    handler = OllamaHandler(cfg, global_app_config={})

    import tldw_Server_API.app.core.Local_LLM.Ollama_Handler as ol_mod

    def fail_net_connections():
        raise RuntimeError("port lookup failed at /private/ollama.sock")

    fake_psutil = SimpleNamespace(CONN_LISTEN="LISTEN", net_connections=fail_net_connections)

    monkeypatch.setattr(handler, "is_ollama_installed", lambda: asyncio.sleep(0, result=True))
    monkeypatch.setattr(ol_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(ol_mod, "psutil", fake_psutil)

    result = await handler.stop_server(port=11434)

    assert result == "Error stopping Ollama server on port 11434"
    assert "port lookup failed" not in result
    assert "/private/ollama.sock" not in result
