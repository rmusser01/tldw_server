from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import types

import pytest


@pytest.mark.unit
def test_llamacpp_extract_message_content_returns_empty_when_choices_missing_or_empty():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        _extract_message_content,
    )

    assert _extract_message_content({"choices": []}) == ""  # nosec B101
    assert _extract_message_content({"choices": None}) == ""  # nosec B101
    assert _extract_message_content({}) == ""  # nosec B101


@pytest.mark.unit
def test_llamacpp_available_is_local_only_for_remote_mode(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "remote")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "8080")

    def _fail(*args, **kwargs):
        raise AssertionError("availability must stay local-only")

    monkeypatch.setattr(socket, "create_connection", _fail)
    monkeypatch.setattr("subprocess.run", _fail)

    assert LlamaCppOCRBackend.available() is True  # nosec B101


@pytest.mark.unit
def test_llamacpp_available_is_true_for_managed_mode_when_managed_start_is_configured(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19090")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--serve"]')

    assert LlamaCppOCRBackend.available() is True  # nosec B101


@pytest.mark.unit
def test_llamacpp_managed_restarts_unhealthy_existing_runtime(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        _ensure_managed_runtime,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        register_managed_process,
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19090")
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--serve", "--host", "{host}", "--port", "{port}"]),
    )

    class _ExistingProcess:
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
                raise TimeoutError("still running")
            return self.returncode

    class _NewProcess:
        pid = None
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return 0

    existing = _ExistingProcess()
    register_managed_process("llamacpp", existing, host="127.0.0.1", port=19090)

    readiness = iter([False, True])
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr._wait_for_managed_http_ready",
        lambda host, port, timeout_total: next(readiness),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.subprocess.Popen",
        lambda *args, **kwargs: _NewProcess(),
    )

    host, port = _ensure_managed_runtime()

    assert existing.terminated is True  # nosec B101
    assert (host, port) == ("127.0.0.1", 19090)  # nosec B101
    reset_managed_process_registry()


@pytest.mark.unit
def test_llamacpp_auto_mode_prefers_remote_for_invocation(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "auto")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "8080")
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    captured: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        return {"choices": [{"message": {"content": "auto remote text"}}]}

    def _fail(*args, **kwargs):
        raise AssertionError("cli path must not run when remote auto mode is usable")

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch_json", fake_fetch_json)
    monkeypatch.setattr("subprocess.run", _fail)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
    )

    assert LlamaCppOCRBackend.available() is True  # nosec B101
    assert LlamaCppOCRBackend().describe()["mode"] == "remote"  # nosec B101
    assert captured["method"] == "POST"  # nosec B101
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"  # nosec B101
    assert result.text == "auto remote text"  # nosec B101
    reset_managed_process_registry()


@pytest.mark.unit
def test_llamacpp_auto_mode_prefers_managed_when_managed_start_is_configured(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "auto")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19090")
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--serve", "--host", "{host}", "--port", "{port}"]),
    )

    managed_calls = {"count": 0}

    def _fake_ensure_managed_runtime():
        managed_calls["count"] += 1
        return ("127.0.0.1", 19090)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr._ensure_managed_runtime",
        _fake_ensure_managed_runtime,
    )

    captured: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout):
        captured["url"] = url
        return {"choices": [{"message": {"content": "auto managed text"}}]}

    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch_json", fake_fetch_json)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
    )

    assert LlamaCppOCRBackend.available() is True  # nosec B101
    assert LlamaCppOCRBackend().describe()["mode"] == "managed"  # nosec B101
    assert managed_calls["count"] == 1  # nosec B101
    assert captured["url"] == "http://127.0.0.1:19090/v1/chat/completions"  # nosec B101
    assert result.text == "auto managed text"  # nosec B101


@pytest.mark.unit
def test_llamacpp_auto_mode_falls_back_to_cli_when_only_cli_is_configured(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "auto")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(stdout="auto cli text", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="text",
    )

    assert LlamaCppOCRBackend.available() is True  # nosec B101
    assert LlamaCppOCRBackend().describe()["mode"] == "cli"  # nosec B101
    assert result.text == "auto cli text"  # nosec B101


@pytest.mark.unit
def test_llamacpp_available_is_false_for_managed_mode_without_private_port(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.delenv("LLAMACPP_OCR_PORT", raising=False)
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--serve"]')

    assert LlamaCppOCRBackend.available() is False  # nosec B101


@pytest.mark.unit
def test_llamacpp_remote_request_uses_openai_compatible_data_url(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "remote")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "8080")
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")
    monkeypatch.setenv("LLAMACPP_OCR_USE_DATA_URL", "true")

    captured: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout):
        captured.update(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )
        return {"choices": [{"message": {"content": "remote text"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        fake_fetch_json,
    )

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
        prompt_preset="doc",
    )

    payload = captured["json"]
    assert captured["method"] == "POST"  # nosec B101
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"  # nosec B101
    assert payload["model"] == "vision.gguf"  # nosec B101  # type: ignore[index]
    assert payload["messages"][0]["content"][0]["text"].startswith("Parse the document")  # nosec B101  # type: ignore[index]
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]  # type: ignore[index]
    assert image_url == f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')}"  # nosec B101
    assert result.text == "remote text"  # nosec B101
    assert result.format == "markdown"  # nosec B101


@pytest.mark.unit
def test_llamacpp_remote_request_cleans_up_temp_file_when_not_using_data_url(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "remote")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "8080")
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")
    monkeypatch.setenv("LLAMACPP_OCR_USE_DATA_URL", "false")

    created_path = tmp_path / "remote-image.png"

    class _TempFile:
        name = str(created_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data):
            created_path.write_bytes(data)

    captured: dict[str, object] = {}

    def fake_tempfile(*args, **kwargs):
        return _TempFile()

    def fake_fetch_json(*, method, url, json, timeout):
        captured["image_url"] = json["messages"][0]["content"][1]["image_url"]["url"]
        assert created_path.exists()  # nosec B101
        return {"choices": [{"message": {"content": "remote text"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.tempfile.NamedTemporaryFile",
        fake_tempfile,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        fake_fetch_json,
    )

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
        prompt_preset="doc",
    )

    assert captured["image_url"] == str(created_path)  # nosec B101
    assert created_path.exists() is False  # nosec B101
    assert result.text == "remote text"  # nosec B101


@pytest.mark.unit
def test_llamacpp_cli_json_output_prefers_full_json_stdout(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert cmd[0] == "llama-ocr"  # nosec B101
        assert "{image_path}" not in cmd  # nosec B101
        assert any(str(part).endswith(".png") for part in cmd)  # nosec B101
        assert check is False  # nosec B101
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(
            stdout=json.dumps(
                {
                    "text": "json text",
                    "blocks": [{"text": "json text", "bbox": [1, 2, 3, 4]}],
                }
            )
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="json",
    )

    assert result.format == "json"  # nosec B101
    assert result.text == "json text"  # nosec B101
    assert result.raw["blocks"][0]["bbox"] == [1, 2, 3, 4]  # nosec B101  # type: ignore[index]


@pytest.mark.unit
def test_llamacpp_cli_uses_closed_temp_image_path(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    seen: dict[str, str] = {}

    def fake_run(cmd, capture_output, text, check, timeout):
        image_path = cmd[2]
        seen["image_path"] = image_path
        assert image_path.endswith(".png")  # nosec B101
        assert os.path.exists(image_path) is True  # nosec B101
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(stdout="plain text", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="text",
    )

    assert result.text == "plain text"  # nosec B101
    assert os.path.exists(seen["image_path"]) is False  # nosec B101


@pytest.mark.unit
def test_llamacpp_cli_timeout_returns_empty_result(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )
    monkeypatch.setenv("LLAMACPP_OCR_TIMEOUT", "7")

    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, check, timeout=None):
        captured["timeout"] = timeout
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="text",
    )

    assert captured["timeout"] == 7  # nosec B101
    assert result.text == ""  # nosec B101
    assert result.format == "text"  # nosec B101


@pytest.mark.unit
def test_llamacpp_structured_preset_parses_json_without_json_output_format(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(
            stdout=json.dumps(
                {
                    "text": "detected text",
                    "blocks": [{"text": "detected text", "bbox": [10, 11, 12, 13]}],
                }
            ),
            returncode=0,
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    spotting_result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="text",
        prompt_preset="spotting",
    )
    json_result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
        prompt_preset="json",
    )

    assert spotting_result.format == "json"  # nosec B101
    assert spotting_result.raw["blocks"][0]["bbox"] == [10, 11, 12, 13]  # nosec B101  # type: ignore[index]
    assert json_result.format == "json"  # nosec B101
    assert json_result.text == "detected text"  # nosec B101


@pytest.mark.unit
def test_llamacpp_cli_json_output_falls_back_to_trailing_json_line(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(
            stdout='markdown prelude\n{"text":"fallback text","blocks":[{"text":"fallback text"}]}'
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="json",
    )

    assert result.format == "json"  # nosec B101
    assert result.text == "fallback text"  # nosec B101
    assert result.raw["blocks"][0]["text"] == "fallback text"  # nosec B101  # type: ignore[index]


@pytest.mark.unit
def test_llamacpp_cli_parses_stdout_when_process_exits_nonzero(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert check is False  # nosec B101
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(stdout="best effort markdown", returncode=2)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
    )

    assert result.format == "markdown"  # nosec B101
    assert result.text == "best effort markdown"  # nosec B101


@pytest.mark.unit
def test_llamacpp_cli_json_request_degrades_when_stdout_is_not_json(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(stdout="plain extracted text", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="json",
    )

    assert result.format == "text"  # nosec B101
    assert result.text == "plain extracted text"  # nosec B101
    assert result.raw == {"raw_output": "plain extracted text"}  # nosec B101
    assert "could not be parsed" in result.warnings[0]  # nosec B101


@pytest.mark.unit
def test_llamacpp_structured_preset_parse_failure_preserves_raw_output(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "cli")
    monkeypatch.setenv(
        "LLAMACPP_OCR_ARGV",
        json.dumps(["llama-ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(stdout="plain extracted text", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LlamaCppOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
        prompt_preset="json",
    )

    assert result.format == "markdown"  # nosec B101
    assert result.text == "plain extracted text"  # nosec B101
    assert result.raw == {"raw_output": "plain extracted text"}  # nosec B101
    assert "could not be parsed" in result.warnings[0]  # nosec B101


@pytest.mark.unit
def test_llamacpp_describe_reports_runtime_and_auto_metadata(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )

    monkeypatch.setenv("LLAMACPP_OCR_MODE", "remote")
    monkeypatch.setenv("LLAMACPP_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "8080")
    monkeypatch.setenv("LLAMACPP_OCR_MODEL_PATH", "vision.gguf")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr"]')
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "false")
    monkeypatch.setenv("LLAMACPP_OCR_MAX_PAGE_CONCURRENCY", "6")
    monkeypatch.setenv("LLAMACPP_OCR_CONFIGURED_FLAGS", "--ctx-size 4096 --temp 0")
    monkeypatch.setenv("LLAMACPP_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "false")

    description = LlamaCppOCRBackend().describe()

    assert description["mode"] == "remote"  # nosec B101
    assert description["model"] == "vision.gguf"  # nosec B101
    assert description["configured_flags"] == "--ctx-size 4096 --temp 0"  # nosec B101
    assert description["auto_eligible"] is True  # nosec B101
    assert description["auto_high_quality_eligible"] is False  # nosec B101
    assert description["backend_concurrency_cap"] == 6  # nosec B101
    assert description["configured"] is True  # nosec B101
    assert description["supports_structured_output"] is True  # nosec B101
    assert description["supports_json"] is True  # nosec B101
    assert description["configured_mode"] == "remote"  # nosec B101
    assert description["url_configured"] is True  # nosec B101
    assert description["cli_configured"] is True  # nosec B101


@pytest.mark.unit
def test_llamacpp_describe_reports_managed_configuration_state(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19090")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--serve"]')

    description = LlamaCppOCRBackend().describe()

    assert description["mode"] == "managed"  # nosec B101
    assert description["allow_managed_start"] is True  # nosec B101
    assert description["managed_configured"] is True  # nosec B101
    assert description["managed_running"] is False  # nosec B101


@pytest.mark.unit
def test_llamacpp_describe_reports_active_managed_runtime(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr import (
        LlamaCppOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        register_managed_process,
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("LLAMACPP_OCR_MODE", "managed")
    monkeypatch.setenv("LLAMACPP_OCR_ALLOW_MANAGED_START", "false")
    monkeypatch.setenv("LLAMACPP_OCR_PORT", "19090")
    monkeypatch.setenv("LLAMACPP_OCR_ARGV", '["llama-ocr", "--serve"]')
    register_managed_process("llamacpp", types.SimpleNamespace(pid=101, poll=lambda: None, returncode=None))

    description = LlamaCppOCRBackend().describe()

    assert description["managed_configured"] is True  # nosec B101
    assert description["managed_running"] is True  # nosec B101
    assert description["allow_managed_start"] is False  # nosec B101
    reset_managed_process_registry()
