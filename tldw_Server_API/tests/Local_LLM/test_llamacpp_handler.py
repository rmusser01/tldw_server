import asyncio
import platform
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler import LlamaCppHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError, InferenceError
import httpx


class DummyProcess:
    def __init__(self):
        self.pid = 12345
        self.returncode = None
        self.stdout = None
        self.stderr = None

    async def wait(self):
        return 0

    def terminate(self):

        self.returncode = 0


@pytest.mark.asyncio
async def test_llamacpp_start_server_with_readiness(monkeypatch, tmp_path: Path):
    # Setup dummy executable and model
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "toy.gguf"
    model.write_text("fake")

    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8099,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Stub subprocess and readiness
    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server(model.name, server_args={"port": 8099})
    assert res["status"] == "started"
    assert res["port"] == 8099


@pytest.mark.asyncio
async def test_llamacpp_start_server_wildcard_host_uses_loopback(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "toy.gguf"
    model.write_text("fake")

    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8099,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    captured = {}

    async def fake_ready(base_url, *a, **k):
        captured["base_url"] = base_url
        return True

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", fake_ready)

    res = await handler.start_server(model.name, server_args={"host": "0.0.0.0", "port": 8099})  # nosec B104
    assert res["status"] == "started"
    assert captured["base_url"] == "http://127.0.0.1:8099"


@pytest.mark.asyncio
async def test_llamacpp_start_server_ignores_empty_args(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "toy.gguf"
    model.write_text("fake")

    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8099,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server(model.name, server_args={"port": "", "threads": None})
    assert res["status"] == "started"
    assert res["port"] == 8099


@pytest.mark.asyncio
async def test_llamacpp_start_server_starts_stream_drainers(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "toy.gguf"
    model.write_text("fake")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, default_port=8101)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class DummyStream:
        async def read(self, n: int = -1):
            return b""

    class ProcWithStreams:
        pid = 123
        returncode = None
        stdout = DummyStream()
        stderr = DummyStream()

        async def wait(self):
            return 0

    async def _fake_cpe(*a, **k):
        return ProcWithStreams()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server(model.name, server_args={"port": 8101})
    assert res["status"] == "started"
    assert handler._stream_tasks


@pytest.mark.asyncio
async def test_llamacpp_inference_requires_running_server(tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8099,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    with pytest.raises(ServerError):
        await handler.inference(prompt="hi")


@pytest.mark.asyncio
async def test_llamacpp_inference_wildcard_host_uses_loopback(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8199,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    class RunningProc:
        pid = 1
        returncode = None

    handler._active_server_process = RunningProc()
    handler._active_server_host = "0.0.0.0"  # nosec B104
    handler._active_server_port = 8199

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None, **kwargs):
        captured["url"] = url
        return {"ok": True}

    monkeypatch.setattr(llama_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi")
    assert captured["url"].startswith("http://127.0.0.1:8199")


@pytest.mark.asyncio
async def test_llamacpp_inference_completions_uses_prompt(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, default_port=8199)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class RunningProc:
        pid = 1
        returncode = None

    handler._active_server_process = RunningProc()
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 8199

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None, **kwargs):
        captured["payload"] = json
        return {"ok": True}

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi", api_endpoint="/v1/completions")
    assert "prompt" in captured["payload"]
    assert "messages" not in captured["payload"]


@pytest.mark.asyncio
async def test_llamacpp_inference_drops_timeout(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, default_port=8199)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class RunningProc:
        pid = 1
        returncode = None

    handler._active_server_process = RunningProc()
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 8199

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None, **kwargs):
        captured["payload"] = json
        return {"ok": True}

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi", timeout=1.5)
    assert "timeout" not in captured["payload"]


@pytest.mark.asyncio
async def test_llamacpp_start_server_invalid_arg(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "toy.gguf").write_text("fake")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=tmp_path / "models")
    handler = LlamaCppHandler(cfg, global_app_config={})

    with pytest.raises(ServerError):
        await handler.start_server("toy.gguf", server_args={"not_a_flag": True})


@pytest.mark.asyncio
async def test_llamacpp_start_server_by_path_accepts_allowed_registered_path(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    registered_dir = tmp_path / "registered"
    registered_dir.mkdir()
    registered_model = registered_dir / "registered.gguf"
    registered_model.write_text("fake")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, allowed_paths=[registered_dir], default_port=8199)
    handler = LlamaCppHandler(cfg, global_app_config={})

    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server_by_path(
        registered_model,
        model_label="registered.gguf",
        server_args={"port": 8199},
    )
    assert res["status"] == "started"
    assert res["model"] == "registered.gguf"


@pytest.mark.asyncio
async def test_llamacpp_start_server_by_path_rejects_non_gguf(tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    text_model = model_dir / "not-model.txt"
    text_model.write_text("fake")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    with pytest.raises(ServerError, match="GGUF"):
        await handler.start_server_by_path(text_model)


@pytest.mark.asyncio
async def test_llamacpp_start_server_by_path_rejects_outside_allowlist(tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_model = outside_dir / "outside.gguf"
    outside_model.write_text("fake")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    with pytest.raises(ServerError, match="allowed"):
        await handler.start_server_by_path(outside_model)


@pytest.mark.asyncio
async def test_llamacpp_inference_http_5xx(monkeypatch, tmp_path: Path):
    # Setup handler and running process state
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "toy.gguf").write_text("fake")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, default_port=8199)
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Fake that server is running
    class RunningProc:
        pid = 1
        returncode = None

    handler._active_server_process = RunningProc()
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 8199

    # Make request_json raise HTTPStatusError
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    req = httpx.Request("POST", "http://127.0.0.1:8199/v1/chat/completions")
    resp = httpx.Response(500, request=req, text="server error")
    err = httpx.HTTPStatusError("error", request=req, response=resp)
    monkeypatch.setattr(llama_mod, "request_json", lambda *a, **k: (_ for _ in ()).throw(err))

    with pytest.raises(InferenceError):
        await handler.inference(prompt="hi")


@pytest.mark.asyncio
async def test_llamacpp_stop_timeout(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "toy.gguf").write_text("fake")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class SlowProc:
        def __init__(self):
            self.pid = 88
            self.returncode = None

        async def wait(self):
            return 0

        def terminate(self):
            self.returncode = None

    handler._active_server_process = SlowProc()
    handler._active_server_model = "toy.gguf"
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 8081

    async def fake_wait_for(coro, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    import os as _os

    monkeypatch.setattr(_os, "killpg", lambda *a, **k: None, raising=False)

    msg = await handler.stop_server()
    assert "stopped" in msg.lower() or "terminated" in msg.lower()


@pytest.mark.asyncio
async def test_llamacpp_stop_server_by_pid(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "toy.gguf").write_text("fake")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class Proc:
        def __init__(self):
            self.pid = 77
            self.returncode = None
            self.terminated = False

        async def wait(self):
            return 0

        def terminate(self):
            self.terminated = True
            self.returncode = 0

    proc = Proc()
    handler._active_server_process = proc
    handler._active_server_model = "toy.gguf"
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 8082

    monkeypatch.setattr(platform, "system", lambda: "Windows")

    msg = await handler.stop_server(pid=77)
    assert proc.terminated
    assert "stopped" in msg.lower() or "terminated" in msg.lower()


@pytest.mark.asyncio
async def test_llamacpp_start_server_not_ready(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "toy.gguf").write_text("fake")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, default_port=8099)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class DP:  # dummy process
        pid = 42
        returncode = None
        stdout = None
        stderr = None

        def terminate(self):
            pass

    async def _fake_cpe2(*a, **k):
        return DP()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe2)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=False))

    with pytest.raises(ServerError):
        await handler.start_server("toy.gguf", server_args={"port": 8099})


@pytest.mark.asyncio
async def test_llamacpp_allow_unvalidated_args_unknown_flags_passthrough(monkeypatch, tmp_path: Path):
    # Prepare dummy executable and model
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "toy.gguf").write_text("fake")

    # Enable allow_unvalidated_args to allow passthrough of unknown flags
    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8010,
        allow_unvalidated_args=True,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Capture the command passed to subprocess
    captured_cmd = {"args": ()}

    class DP:
        pid = 99
        returncode = None
        stdout = None
        stderr = None

        async def wait(self):
            return 0

    async def fake_cpe(*args, **kwargs):
        captured_cmd["args"] = args
        return DP()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)

    # Patch the symbol imported into the module (not http_utils) to avoid 30s wait
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    # Unknown flags should be converted to --unknown-flag and --bool-flag (no value)
    res = await handler.start_server(
        "toy.gguf",
        server_args={
            "port": 8010,
            "unknown_flag": 7,
            "another_unknown": "x",
            "bool_unknown": True,
            "none_unknown": None,
        },
    )

    assert res["status"] == "started"
    args = list(captured_cmd["args"])  # tuple of command components
    # Ensure the transformed flags are present
    assert "--unknown-flag" in args and "7" in args
    assert "--another-unknown" in args and "x" in args
    assert "--bool-unknown" in args  # boolean True => flag only
    # None/False should not append a value or flag
    assert "--none-unknown" not in args


@pytest.mark.asyncio
async def test_llamacpp_start_server_accepts_structured_aliases(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "toy.gguf").write_text("fake")

    cfg = LlamaCppConfig(
        executable_path=exe,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8011,
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    captured_cmd = {"args": ()}

    class DP:
        pid = 101
        returncode = None
        stdout = None
        stderr = None

        async def wait(self):
            return 0

    async def fake_cpe(*args, **kwargs):
        captured_cmd["args"] = args
        return DP()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server(
        "toy.gguf",
        server_args={
            "n_ctx": 8192,
            "n_batch": 1024,
            "cache_type": "f16",
            "row_split": True,
            "streaming_llm": True,
            "cpu_moe": True,
            "n_cpu_moe": 7,
            "flash_attn": "on",
            "no_mmproj": True,
            "draft_max": 16,
        },
    )

    assert res["status"] == "started"
    args = list(captured_cmd["args"])

    assert "-c" in args and "8192" in args
    assert "-b" in args and "1024" in args
    assert "--cache-type-k" in args and "f16" in args
    assert "--cache-type-v" in args and "f16" in args
    assert "--split-mode" in args and "row" in args
    assert "--context-shift" in args
    assert "--cpu-moe" in args
    assert "--n-cpu-moe" in args and "7" in args
    assert "--flash-attn" in args and "on" in args
    assert "--no-mmproj" in args
    assert "--draft-max" in args and "16" in args


@pytest.mark.asyncio
async def test_llamacpp_inference_forces_non_streaming(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    class RunningProc:
        pid = 1
        returncode = None

    handler._active_server_process = RunningProc()
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 8199

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None, **kwargs):
        captured["payload"] = json
        return {"ok": True}

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi", stream=True)
    assert captured["payload"]["stream"] is False
