import asyncio
import platform
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.Llamafile_Handler import LlamafileHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamafileConfig
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError, InferenceError, ModelDownloadError
import httpx


class DummyProcess:
    def __init__(self):
        self.pid = 22222
        self.returncode = None
        self.stdout = None
        self.stderr = None

    async def wait(self):
        return 0

    def terminate(self):

        self.returncode = 0


@pytest.mark.asyncio
async def test_llamafile_start_server_redacts_api_key(monkeypatch, tmp_path: Path):
    # Prepare directories and files
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()

    exe = llama_dir / ("llamafile" if not hasattr(asyncio, "WINDOWS") else "llamafile.exe")
    exe.write_text("#!/bin/sh\n")
    model = model_dir / "toy.gguf"
    model.write_text("fake")

    cfg = LlamafileConfig(
        llamafile_dir=llama_dir,
        models_dir=model_dir,
        default_host="127.0.0.1",
        default_port=8077,
        allow_cli_secrets=True,
    )
    handler = LlamafileHandler(cfg, global_app_config={})

    # Monkeypatch binary downloader to return our dummy exe path
    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )

    # Stub process creation and readiness
    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server(model_filename=model.name, server_args={"port": 8077, "api_key": "supersecret"})
    assert res["status"] == "started"
    assert "REDACTED" in res["command"]
    assert "supersecret" not in res["command"]


@pytest.mark.asyncio
async def test_llamafile_start_server_invalid_arg(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()
    exe = llama_dir / "llamafile"
    exe.write_text("#!/bin/sh\n")
    (model_dir / "m.gguf").write_text("fake")

    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=model_dir)
    handler = LlamafileHandler(cfg, global_app_config={})

    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )
    # readiness will succeed but we should fail earlier due to invalid arg
    from tldw_Server_API.app.core.Local_LLM import http_utils

    async def _fake_cpe2(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe2)
    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    with pytest.raises(ServerError):
        await handler.start_server("m.gguf", server_args={"bad_flag": True})


@pytest.mark.asyncio
async def test_llamafile_inference_http_error(monkeypatch, tmp_path: Path):
    # Prepare handler
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=tmp_path)
    handler = LlamafileHandler(cfg, global_app_config={})

    # Fake running server map so it proceeds to HTTP request
    class RP:
        pid = 2
        returncode = None

    handler._active_servers[8080] = RP()

    import tldw_Server_API.app.core.Local_LLM.Llamafile_Handler as lf_mod

    req = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    resp = httpx.Response(500, request=req, text="err")
    err = httpx.HTTPStatusError("error", request=req, response=resp)
    monkeypatch.setattr(lf_mod, "request_json", lambda *a, **k: (_ for _ in ()).throw(err))

    with pytest.raises(InferenceError):
        await handler.inference(prompt="hi", port=8080)


@pytest.mark.asyncio
async def test_llamafile_stop_timeout(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=tmp_path)
    handler = LlamafileHandler(cfg, global_app_config={})

    class SlowProc:
        def __init__(self):
            self.pid = 99
            self.returncode = None

        async def wait(self):
            return 0

        def terminate(self):
            self.returncode = None

        def kill(self):
            self.returncode = -9

    handler._active_servers[5555] = SlowProc()

    async def fake_wait_for(coro, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    # os.killpg will be called; stub it to avoid errors in tests
    import os as _os

    monkeypatch.setattr(_os, "killpg", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    msg = await handler.stop_server(port=5555)
    assert "stopped" in msg.lower() or "terminated" in msg.lower()


@pytest.mark.asyncio
async def test_llamafile_start_server_not_ready(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()
    exe = llama_dir / "llamafile"
    exe.write_text("#!/bin/sh\n")
    (model_dir / "toy.gguf").write_text("fake")

    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=model_dir, default_port=8077)
    handler = LlamafileHandler(cfg, global_app_config={})

    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )

    class DP:
        pid = 11
        returncode = None
        stdout = None
        stderr = None

        async def wait(self):
            return 0

        def terminate(self):
            self.returncode = 0

    async def _fake_cpe3(*a, **k):
        return DP()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe3)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=False))

    with pytest.raises(ServerError):
        await handler.start_server("toy.gguf", server_args={"port": 8077})


@pytest.mark.asyncio
async def test_llamafile_start_server_not_ready_terminates_process(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()
    exe = llama_dir / "llamafile"
    exe.write_text("#!/bin/sh\n")
    (model_dir / "toy.gguf").write_text("fake")

    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=model_dir, default_port=8077)
    handler = LlamafileHandler(cfg, global_app_config={})

    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )

    proc_holder = {}

    class DP:
        pid = 11
        returncode = None
        stdout = None
        stderr = None

        def __init__(self):
            self.terminated = False
            self.killed = False

        async def wait(self):
            self.returncode = 0
            return 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    async def _fake_cpe(*a, **k):
        proc = DP()
        proc_holder["proc"] = proc
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=False))
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    with pytest.raises(ServerError):
        await handler.start_server("toy.gguf", server_args={"port": 8077})

    proc = proc_holder["proc"]
    assert proc.terminated or proc.killed


@pytest.mark.unit
@pytest.mark.local_llm_service
@pytest.mark.asyncio
async def test_llamafile_start_server_wildcard_host_uses_loopback(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()
    exe = llama_dir / "llamafile"
    exe.write_text("#!/bin/sh\n")
    (model_dir / "toy.gguf").write_text("fake")

    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=model_dir, default_port=8077)
    handler = LlamafileHandler(cfg, global_app_config={})

    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )

    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    seen = {}

    async def _fake_ready(base_url, *a, **k):
        seen["base_url"] = base_url
        return True

    monkeypatch.setattr(http_utils, "wait_for_http_ready", _fake_ready)

    # Wildcard host is normalized to loopback in this test; no external exposure.
    res = await handler.start_server("toy.gguf", server_args={"host": "0.0.0.0", "port": 8077})  # nosec B104
    assert res["status"] == "started"
    assert seen["base_url"] == "http://127.0.0.1:8077"


@pytest.mark.asyncio
async def test_llamafile_start_server_ignores_empty_args(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()
    exe = llama_dir / "llamafile"
    exe.write_text("#!/bin/sh\n")
    (model_dir / "toy.gguf").write_text("fake")

    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=model_dir, default_port=8077)
    handler = LlamafileHandler(cfg, global_app_config={})

    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )

    async def _fake_cpe(*a, **k):
        return DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_cpe)
    from tldw_Server_API.app.core.Local_LLM import http_utils

    monkeypatch.setattr(http_utils, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server("toy.gguf", server_args={"port": "", "threads": None})
    assert res["status"] == "started"
    assert res["port"] == 8077


@pytest.mark.asyncio
async def test_llamafile_start_server_starts_stream_drainers(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    model_dir = tmp_path / "models"
    llama_dir.mkdir()
    model_dir.mkdir()
    exe = llama_dir / "llamafile"
    exe.write_text("#!/bin/sh\n")
    (model_dir / "toy.gguf").write_text("fake")

    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=model_dir, default_port=8077)
    handler = LlamafileHandler(cfg, global_app_config={})

    monkeypatch.setattr(
        handler, "download_latest_llamafile_executable", lambda force_download=False: asyncio.sleep(0, result=exe)
    )

    class DummyStream:
        async def read(self, n: int = -1):
            return b""

    class ProcWithStreams:
        pid = 222
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

    res = await handler.start_server("toy.gguf", server_args={"port": 8077})
    assert res["status"] == "started"
    assert handler._stream_tasks.get(8077)


@pytest.mark.unit
@pytest.mark.local_llm_service
@pytest.mark.asyncio
async def test_llamafile_inference_wildcard_host_uses_loopback(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(  # Intentionally tests wildcard bind handling; inference requests are normalized to loopback.
        llamafile_dir=llama_dir,
        models_dir=tmp_path,
        default_host="0.0.0.0",  # nosec B104
    )
    handler = LlamafileHandler(cfg, global_app_config={})

    class RP:
        pid = 2
        returncode = None

    handler._active_servers[8080] = RP()

    import tldw_Server_API.app.core.Local_LLM.Llamafile_Handler as lf_mod

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None):
        captured["url"] = url
        return {"ok": True}

    monkeypatch.setattr(lf_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi", port=8080)
    assert captured["url"].startswith("http://127.0.0.1:8080")


@pytest.mark.asyncio
async def test_llamafile_inference_defaults_port(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=tmp_path, default_port=8123)
    handler = LlamafileHandler(cfg, global_app_config={})

    class RP:
        pid = 2
        returncode = None

    handler._active_servers[8123] = RP()

    import tldw_Server_API.app.core.Local_LLM.Llamafile_Handler as lf_mod

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None):
        captured["url"] = url
        return {"ok": True}

    monkeypatch.setattr(lf_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi")
    assert captured["url"].startswith("http://127.0.0.1:8123")


@pytest.mark.asyncio
async def test_llamafile_inference_drops_timeout(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=tmp_path, default_port=8124)
    handler = LlamafileHandler(cfg, global_app_config={})

    class RP:
        pid = 3
        returncode = None

    handler._active_servers[8124] = RP()

    import tldw_Server_API.app.core.Local_LLM.Llamafile_Handler as lf_mod

    captured = {}

    async def fake_request_json(client, method, url, json=None, headers=None):
        captured["payload"] = json
        return {"ok": True}

    monkeypatch.setattr(lf_mod, "request_json", fake_request_json)

    await handler.inference(prompt="hi", port=8124, timeout=1.5)
    assert "timeout" not in captured["payload"]


@pytest.mark.asyncio
async def test_llamafile_download_selects_exe_asset(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=tmp_path)
    handler = LlamafileHandler(cfg, global_app_config={})
    handler.llamafile_exe_path = llama_dir / "llamafile.exe"

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_request_json(client, method, url):
        return {
            "tag_name": "v1.2.3",
            "assets": [
                {
                    "name": "llamafile-v1.2.3.zip",
                    "browser_download_url": "http://example.com/llamafile.zip",
                    "size": 200,
                },
                {
                    "name": "llamafile-v1.2.3.exe",
                    "browser_download_url": "http://example.com/llamafile.exe",
                    "size": 100,
                },
            ],
        }

    downloads = []

    async def fake_download(url, dest_path):
        downloads.append(url)
        Path(dest_path).write_text("binary")

    import tldw_Server_API.app.core.Local_LLM.Llamafile_Handler as lf_mod
    import tldw_Server_API.app.core.Local_LLM.http_utils as http_utils

    monkeypatch.setattr(lf_mod, "create_async_client", lambda *a, **k: DummyClient())
    monkeypatch.setattr(http_utils, "request_json", fake_request_json)
    monkeypatch.setattr(http_utils, "async_stream_download", fake_download)
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    exe_path = await handler.download_latest_llamafile_executable(force_download=True)

    assert downloads == ["http://example.com/llamafile.exe"]
    assert exe_path == handler.llamafile_exe_path
    assert exe_path.exists()


@pytest.mark.asyncio
async def test_llamafile_download_extracts_zip_asset(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    llama_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=tmp_path)
    handler = LlamafileHandler(cfg, global_app_config={})
    handler.llamafile_exe_path = llama_dir / "llamafile.exe"

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_request_json(client, method, url):
        return {
            "tag_name": "v1.2.3",
            "assets": [
                {
                    "name": "llamafile-v1.2.3.zip",
                    "browser_download_url": "http://example.com/llamafile.zip",
                    "size": 200,
                },
            ],
        }

    downloads = []

    async def fake_download(url, dest_path):
        downloads.append(url)
        with zipfile.ZipFile(dest_path, "w") as zf:
            zf.writestr("llamafile.exe", "binary")

    import tldw_Server_API.app.core.Local_LLM.Llamafile_Handler as lf_mod
    import tldw_Server_API.app.core.Local_LLM.http_utils as http_utils

    monkeypatch.setattr(lf_mod, "create_async_client", lambda *a, **k: DummyClient())
    monkeypatch.setattr(http_utils, "request_json", fake_request_json)
    monkeypatch.setattr(http_utils, "async_stream_download", fake_download)
    monkeypatch.setattr(platform, "system", lambda: "Windows")

    exe_path = await handler.download_latest_llamafile_executable(force_download=True)

    assert downloads == ["http://example.com/llamafile.zip"]
    assert exe_path == handler.llamafile_exe_path
    assert exe_path.exists()


@pytest.mark.asyncio
async def test_llamafile_download_model_rejects_traversal(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    models_dir = tmp_path / "models"
    llama_dir.mkdir()
    models_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=models_dir)
    handler = LlamafileHandler(cfg, global_app_config={})

    import tldw_Server_API.app.core.Local_LLM.http_utils as http_utils

    async def fake_download(url, dest_path):
        raise AssertionError("download should not be called for unsafe paths")

    monkeypatch.setattr(http_utils, "async_stream_download", fake_download)

    with pytest.raises(ModelDownloadError):
        await handler.download_model_file(
            model_name="bad",
            model_url="http://example.com/bad.gguf",
            model_filename="../evil.gguf",
        )

    outside = tmp_path / "outside" / "evil.gguf"
    outside.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ModelDownloadError):
        await handler.download_model_file(
            model_name="bad",
            model_url="http://example.com/bad2.gguf",
            model_filename=str(outside),
        )


@pytest.mark.asyncio
async def test_llamafile_download_model_allows_subdir(monkeypatch, tmp_path: Path):
    llama_dir = tmp_path / "llamafile"
    models_dir = tmp_path / "models"
    llama_dir.mkdir()
    models_dir.mkdir()
    cfg = LlamafileConfig(llamafile_dir=llama_dir, models_dir=models_dir)
    handler = LlamafileHandler(cfg, global_app_config={})

    import tldw_Server_API.app.core.Local_LLM.http_utils as http_utils

    async def fake_download(url, dest_path):
        Path(dest_path).write_text("model")

    monkeypatch.setattr(http_utils, "async_stream_download", fake_download)

    model_path = await handler.download_model_file(
        model_name="ok",
        model_url="http://example.com/ok.gguf",
        model_filename="subdir/ok.gguf",
    )

    assert model_path.exists()
    assert models_dir in model_path.parents
