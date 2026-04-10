from __future__ import annotations

import types

import pytest


@pytest.mark.unit
def test_managed_mode_refuses_start_when_disabled(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "false")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "18080")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--host", "{host}", "--port", "{port}"]')

    def _fail(*args, **kwargs):
        raise AssertionError("managed startup should be refused before spawning")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.subprocess.Popen",
        _fail,
    )

    with pytest.raises(RuntimeError, match="managed OCR startup is disabled"):
        LlamaCppOCRBackend().ocr_image_structured(b"png-bytes", output_format="text")

    reset_managed_process_registry()


@pytest.mark.unit
def test_managed_mode_reuses_existing_ocr_process(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        register_managed_process,
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")

    process = types.SimpleNamespace(pid=999, poll=lambda: None, returncode=None)
    register_managed_process("llamacpp", process, host="127.0.0.1", port=18081)

    spawned: list[object] = []

    def fake_popen(*args, **kwargs):
        spawned.append((args, kwargs))
        raise AssertionError("existing OCR-managed process should be reused")

    def fake_fetch_json(*, method, url, json, timeout):
        assert url == "http://127.0.0.1:18081/v1/chat/completions"  # nosec B101
        return {"choices": [{"message": {"content": "managed text"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr._wait_for_managed_http_ready",
        lambda host, port, timeout_total: True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        fake_fetch_json,
    )

    result = LlamaCppOCRBackend().ocr_image_structured(b"png-bytes", output_format="text")

    assert spawned == []  # nosec B101
    assert result.text == "managed text"  # nosec B101
    reset_managed_process_registry()


@pytest.mark.unit
def test_managed_mode_uses_private_ocr_port_not_admin_port(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19090")
    monkeypatch.setenv("LLAMACPP_SERVER_PORT", "8080")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        '["llama-ocr", "--host", "{host}", "--port", "{port}", "--model", "{model_path}"]',
    )
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")

    spawned: list[list[str]] = []

    class _Process:
        pid = 1234
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        spawned.append(list(cmd))
        return _Process()

    def fake_wait_for_ready(host, port, timeout_total):
        assert host == "127.0.0.1"  # nosec B101
        assert port == 19090  # nosec B101
        assert timeout_total > 0  # nosec B101
        return True

    def fake_fetch_json(*, method, url, json, timeout):
        assert url == "http://127.0.0.1:19090/v1/chat/completions"  # nosec B101
        return {"choices": [{"message": {"content": "managed text"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr._wait_for_managed_http_ready",
        fake_wait_for_ready,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        fake_fetch_json,
    )

    result = LlamaCppOCRBackend().ocr_image_structured(b"png-bytes", output_format="text")

    assert result.text == "managed text"  # nosec B101
    assert "--port" in spawned[0]  # nosec B101
    assert "19090" in spawned[0]  # nosec B101
    assert "8080" not in spawned[0]  # nosec B101
    reset_managed_process_registry()


@pytest.mark.unit
def test_managed_mode_rejects_admin_port_reuse(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "8080")
    monkeypatch.setenv("LLAMACPP_SERVER_PORT", "8080")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--host", "{host}", "--port", "{port}"]')

    with pytest.raises(RuntimeError, match="OCR-private port"):
        LlamaCppOCRBackend().ocr_image_structured(b"png-bytes", output_format="text")

    reset_managed_process_registry()


@pytest.mark.unit
def test_managed_startup_failure_terminates_process_and_leaves_no_registry_entry(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        get_managed_process_record,
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19091")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--host", "{host}", "--port", "{port}"]')

    class _Process:
        pid = None
        returncode = None

        def __init__(self):
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            if self.returncode is None:
                raise TimeoutError("still starting")
            return self.returncode

    process = _Process()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr._wait_for_managed_http_ready",
        lambda host, port, timeout_total: False,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        LlamaCppOCRBackend().ocr_image_structured(b"png-bytes", output_format="text")

    assert process.terminated is True  # nosec B101
    assert get_managed_process_record("llamacpp") is None  # nosec B101
    reset_managed_process_registry()
