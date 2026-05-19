import asyncio
import platform
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler import LlamaCppHandler
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError


class DummyProc:
    def __init__(self, pid=777):
        self.pid = pid
        self.returncode = None
        self.stdout = None
        self.stderr = None

    async def wait(self):
        return 0


@pytest.mark.asyncio
async def test_windows_creationflags(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    captured = {}

    async def fake_cpe(*args, **kwargs):
        captured.update(kwargs)
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))

    res = await handler.start_server("m.gguf")
    assert res["status"] == "started"
    assert "creationflags" in captured


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["hf_token", "hf-token", "hfToken"])
async def test_denylist_hf_token_rejected(monkeypatch, tmp_path: Path, key: str):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, allow_unvalidated_args=True)
    handler = LlamaCppHandler(cfg, global_app_config={})
    with pytest.raises(ServerError):
        await handler.start_server("m.gguf", server_args={key: "ABC"})


@pytest.mark.asyncio
async def test_allow_cli_secrets_allows_hf_token(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, allow_unvalidated_args=True, allow_cli_secrets=True)
    handler = LlamaCppHandler(cfg, global_app_config={})

    async def fake_cpe(*a, **k):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))
    res = await handler.start_server("m.gguf", server_args={"hf_token": "ABC"})
    assert res["status"] == "started"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["api_key", "api-key", "apiKey"])
async def test_denylist_api_key_rejected(tmp_path: Path, key: str):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, allow_unvalidated_args=True)
    handler = LlamaCppHandler(cfg, global_app_config={})
    with pytest.raises(ServerError):
        await handler.start_server("m.gguf", server_args={key: "ABC"})


@pytest.mark.asyncio
async def test_path_safety(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Outside models_dir should be rejected
    with pytest.raises(ServerError):
        await handler.start_server("m.gguf", server_args={"grammar_file": str(tmp_path / "outside.bnf")})

    # Inside models_dir should pass
    inside = model_dir / "g.bnf"
    inside.write_text("rule := 'x'")
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    async def fake_cpe(*a, **k):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)
    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))
    res = await handler.start_server("m.gguf", server_args={"grammar_file": str(inside)})
    assert res["status"] == "started"


@pytest.mark.asyncio
async def test_path_prefix_bypass_rejected(tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Path shares prefix with models_dir but is not a child
    outside_dir = tmp_path / "models2"
    outside_dir.mkdir()
    outside = outside_dir / "g.bnf"
    outside.write_text("rule := 'x'")
    with pytest.raises(ServerError):
        await handler.start_server("m.gguf", server_args={"grammar_file": str(outside)})


@pytest.mark.asyncio
async def test_model_path_traversal_rejected(tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_model = outside_dir / "bad.gguf"
    outside_model.write_text("x")

    with pytest.raises(ServerError):
        await handler.start_server(str(outside_model))


@pytest.mark.asyncio
async def test_start_server_rejects_absolute_model_filename_even_under_allowed_dir(tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "m.gguf"
    model.write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    with pytest.raises(ServerError):
        await handler.start_server(str(model))


@pytest.mark.asyncio
async def test_port_autoselect(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(
        executable_path=exe, models_dir=model_dir, default_port=9000, port_autoselect=True, port_probe_max=5
    )
    handler = LlamaCppHandler(cfg, global_app_config={})

    seq = {"i": 0}

    def fake_is_free(host, port):
        seq["i"] += 1
        return seq["i"] >= 2  # first port busy, second free

    monkeypatch.setattr(handler, "_is_port_free", fake_is_free)

    async def fake_cpe(*a, **k):
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_cpe)
    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    monkeypatch.setattr(llama_mod, "wait_for_http_ready", lambda *a, **k: asyncio.sleep(0, result=True))
    res = await handler.start_server("m.gguf")
    assert res["status"] == "started"
    # Expect port increased
    assert res["port"] == cfg.default_port + 1


@pytest.mark.asyncio
async def test_streaming_inference_yields_sse(monkeypatch, tmp_path: Path):
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})
    # Fake running process
    handler._active_server_process = DummyProc(pid=123)
    handler._active_server_host = "127.0.0.1"
    handler._active_server_port = 9999

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
            yield "data: [DONE]"

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json=None, headers=None):
            return FakeStream()

    import tldw_Server_API.app.core.Local_LLM.http_utils as http_utils

    monkeypatch.setattr(http_utils, "create_async_client", lambda *a, **k: FakeClient())

    chunks = []
    async for line in handler.stream_inference(prompt="hello"):
        chunks.append(line)
    assert any("data: [DONE]" in c for c in chunks)


# --- Tests for symlink path traversal protection ---


@pytest.mark.asyncio
async def test_symlink_path_traversal_rejected(tmp_path: Path):
    """Test that symlinks pointing outside allowed directories are rejected."""
    import os

    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")

    # Create a secret file outside models_dir
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("sensitive data")

    # Create a symlink inside models_dir pointing to the secret file
    symlink_path = model_dir / "evil_link"
    try:
        symlink_path.symlink_to(secret_file)
    except OSError:
        pytest.skip("Cannot create symlinks on this platform")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    # The symlink is inside models_dir but points outside - should be rejected
    with pytest.raises(ServerError):
        await handler.start_server("m.gguf", server_args={"grammar_file": str(symlink_path)})


@pytest.mark.asyncio
async def test_symlink_in_allowed_paths_resolved(tmp_path: Path):
    """Test that symlinks in allowed_paths config are properly resolved."""
    import os

    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.gguf").write_text("x")

    # Create a real directory
    real_dir = tmp_path / "real_files"
    real_dir.mkdir()
    real_file = real_dir / "grammar.bnf"
    real_file.write_text("rule := 'x'")

    # Create a symlink directory pointing to real_dir
    symlink_dir = tmp_path / "link_to_files"
    try:
        symlink_dir.symlink_to(real_dir)
    except OSError:
        pytest.skip("Cannot create symlinks on this platform")

    # Add the symlinked path to allowed_paths
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir, allowed_paths=[symlink_dir])
    handler = LlamaCppHandler(cfg, global_app_config={})

    # File accessed via symlink should work (symlink is properly resolved)
    async def fake_cpe(*a, **k):
        return DummyProc()

    import tldw_Server_API.app.core.Local_LLM.LlamaCpp_Handler as llama_mod

    # We need to mock these for the test
    import asyncio as asyncio_mod

    original_cpe = asyncio_mod.create_subprocess_exec

    async def mock_cpe(*args, **kwargs):
        return DummyProc()

    asyncio_mod.create_subprocess_exec = mock_cpe

    async def mock_ready(*a, **k):
        await asyncio.sleep(0)
        return True

    original_ready = llama_mod.wait_for_http_ready
    llama_mod.wait_for_http_ready = mock_ready

    try:
        # Access via the resolved real path should work
        res = await handler.start_server("m.gguf", server_args={"grammar_file": str(real_file)})
        assert res["status"] == "started"
    finally:
        asyncio_mod.create_subprocess_exec = original_cpe
        llama_mod.wait_for_http_ready = original_ready


# --- Test for cleanup guard ---


def test_cleanup_guard_prevents_double_cleanup(tmp_path: Path):
    """Test that cleanup only runs once even if called multiple times."""
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Call cleanup multiple times
    handler._cleanup_managed_server_sync()
    assert handler._cleanup_done is True

    # Second call should return immediately
    handler._cleanup_managed_server_sync()
    # No exception means the guard worked


# --- Test for model swap rollback ---


@pytest.mark.asyncio
async def test_model_swap_rollback_on_stop_failure(monkeypatch, tmp_path: Path):
    """Test that model swap restores state if stop_server fails."""
    exe = tmp_path / "llama_server"
    exe.write_text("#!/bin/sh\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "model1.gguf").write_text("x")
    (model_dir / "model2.gguf").write_text("y")

    cfg = LlamaCppConfig(executable_path=exe, models_dir=model_dir)
    handler = LlamaCppHandler(cfg, global_app_config={})

    # Set up initial "running" state
    original_proc = DummyProc(pid=111)
    handler._active_server_process = original_proc
    handler._active_server_model = "model1.gguf"
    handler._active_server_port = 8080
    handler._active_server_host = "127.0.0.1"

    # Make stop_server fail
    async def failing_stop():
        raise ServerError("Failed to stop server")

    monkeypatch.setattr(handler, "stop_server", failing_stop)

    # Attempt to swap models - should fail and restore state
    with pytest.raises(ServerError, match="Model swap failed"):
        await handler.start_server("model2.gguf")

    # State should be restored
    assert handler._active_server_process is original_proc
    assert handler._active_server_model == "model1.gguf"
    assert handler._active_server_port == 8080
