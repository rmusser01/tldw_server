from __future__ import annotations

import base64
import json
import os
import socket
import types

import pytest


@pytest.mark.unit
def test_chatllm_extract_message_content_returns_empty_when_choices_missing_or_empty():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        _extract_message_content,
    )

    assert _extract_message_content({"choices": []}) == ""  # nosec B101
    assert _extract_message_content({"choices": None}) == ""  # nosec B101
    assert _extract_message_content({}) == ""  # nosec B101


@pytest.mark.unit
def test_chatllm_available_is_local_only_for_remote_mode(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )

    monkeypatch.setenv("CHATLLM_OCR_MODE", "remote")
    monkeypatch.setenv("CHATLLM_OCR_URL", "http://127.0.0.1:8081/v1/chat/completions")
    monkeypatch.setenv("CHATLLM_OCR_MODEL", "chatllm-vision")

    def _fail(*args, **kwargs):
        raise AssertionError("availability must stay local-only")

    monkeypatch.setattr(socket, "create_connection", _fail)
    monkeypatch.setattr("subprocess.run", _fail)

    assert ChatLLMOCRBackend.available() is True  # nosec B101


@pytest.mark.unit
def test_chatllm_managed_availability_requires_explicit_healthcheck_url(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("CHATLLM_OCR_MODE", "managed")
    monkeypatch.setenv("CHATLLM_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("CHATLLM_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("CHATLLM_OCR_PORT", "19091")
    monkeypatch.setenv("CHATLLM_OCR_SERVER_BINARY", "chatllm-server")
    monkeypatch.setenv(
        "CHATLLM_OCR_SERVER_ARGS_JSON",
        json.dumps(["--model", "{model_path}", "--port", "{port}"]),
    )
    monkeypatch.delenv("CHATLLM_OCR_HEALTHCHECK_URL", raising=False)

    assert ChatLLMOCRBackend.available() is False  # nosec B101


@pytest.mark.unit
def test_chatllm_wait_for_healthcheck_uses_https_scheme_and_query(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        _wait_for_healthcheck,
    )

    captured: dict[str, object] = {}

    def fake_wait_for_managed_http_ready(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setenv("CHATLLM_OCR_HEALTHCHECK_URL", "https://chatllm.local:9443/ready/status?full=1")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.wait_for_managed_http_ready",
        fake_wait_for_managed_http_ready,
    )

    assert _wait_for_healthcheck(2.5) is True  # nosec B101
    assert captured["host"] == "chatllm.local"  # nosec B101
    assert captured["port"] == 9443  # nosec B101
    assert captured["scheme"] == "https"  # nosec B101
    assert captured["paths"] == ("/ready/status?full=1",)  # nosec B101


@pytest.mark.unit
def test_chatllm_remote_uses_openai_compatible_payload(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )

    monkeypatch.setenv("CHATLLM_OCR_MODE", "remote")
    monkeypatch.setenv("CHATLLM_OCR_URL", "http://127.0.0.1:8081/v1/chat/completions")
    monkeypatch.setenv("CHATLLM_OCR_MODEL", "chatllm-vision")
    monkeypatch.setenv("CHATLLM_OCR_API_KEY", "secret-token")
    monkeypatch.setenv("CHATLLM_OCR_MAX_TOKENS", "1024")
    monkeypatch.setenv("CHATLLM_OCR_TEMPERATURE", "0.25")

    captured: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout, headers=None, **kwargs):
        captured.update(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
                "headers": headers,
            }
        )
        return {"choices": [{"message": {"content": "remote markdown"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        fake_fetch_json,
    )

    result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
        prompt_preset="doc",
    )

    payload = captured["json"]
    assert captured["method"] == "POST"  # nosec B101
    assert captured["url"] == "http://127.0.0.1:8081/v1/chat/completions"  # nosec B101
    assert captured["headers"] == {"Authorization": "Bearer secret-token"}  # nosec B101
    assert payload["model"] == "chatllm-vision"  # nosec B101  # type: ignore[index]
    assert payload["messages"][0]["content"][0]["text"].startswith("Parse the document")  # nosec B101  # type: ignore[index]
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]  # type: ignore[index]
    assert image_url == f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')}"  # nosec B101
    assert result.text == "remote markdown"  # nosec B101
    assert result.format == "markdown"  # nosec B101
    assert result.meta["backend"] == "chatllm"  # nosec B101


@pytest.mark.unit
def test_chatllm_remote_base_url_appends_chat_completions_once(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )

    monkeypatch.setenv("CHATLLM_OCR_MODE", "remote")
    monkeypatch.setenv("CHATLLM_OCR_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setenv("CHATLLM_OCR_MODEL", "chatllm-vision")

    captured: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout, headers=None, **kwargs):
        captured["url"] = url
        return {"choices": [{"message": {"content": "remote markdown"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        fake_fetch_json,
    )

    result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
    )

    assert captured["url"] == "http://127.0.0.1:8081/v1/chat/completions"  # nosec B101
    assert result.text == "remote markdown"  # nosec B101


@pytest.mark.unit
def test_chatllm_cli_json_output_falls_back_to_trailing_json_line(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )

    monkeypatch.setenv("CHATLLM_OCR_MODE", "cli")
    monkeypatch.setenv("CHATLLM_OCR_CLI_BINARY", "chatllm")
    monkeypatch.setenv(
        "CHATLLM_OCR_CLI_ARGS_JSON",
        json.dumps(["ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert cmd[:2] == ["chatllm", "ocr"]  # nosec B101
        assert os.path.exists(cmd[3]) is True  # nosec B101
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(
            stdout='markdown prelude\n{"text":"fallback text","blocks":[{"text":"fallback text"}]}',
            returncode=0,
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="json",
    )

    assert result.format == "json"  # nosec B101
    assert result.text == "fallback text"  # nosec B101
    assert result.raw["blocks"][0]["text"] == "fallback text"  # nosec B101  # type: ignore[index]


@pytest.mark.unit
def test_chatllm_managed_restarts_unhealthy_existing_runtime(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        register_managed_process,
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("CHATLLM_OCR_MODE", "managed")
    monkeypatch.setenv("CHATLLM_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("CHATLLM_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("CHATLLM_OCR_PORT", "19091")
    monkeypatch.setenv("CHATLLM_OCR_MODEL_PATH", "/models/chatllm.gguf")
    monkeypatch.setenv("CHATLLM_OCR_SERVER_BINARY", "chatllm-server")
    monkeypatch.setenv(
        "CHATLLM_OCR_SERVER_ARGS_JSON",
        json.dumps(["serve", "--model", "{model_path}", "--port", "{port}"]),
    )
    monkeypatch.setenv("CHATLLM_OCR_HEALTHCHECK_URL", "http://127.0.0.1:19091/ready")

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
    register_managed_process("chatllm", existing, host="127.0.0.1", port=19091)

    readiness = iter([False, True])

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr._wait_for_healthcheck",
        lambda timeout_total: next(readiness),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.subprocess.Popen",
        lambda *args, **kwargs: _NewProcess(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch_json",
        lambda **kwargs: {"choices": [{"message": {"content": "managed text"}}]},
    )

    result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="text",
    )

    assert existing.terminated is True  # nosec B101
    assert result.text == "managed text"  # nosec B101


@pytest.mark.unit
def test_chatllm_structured_preset_parses_json_without_json_output_format(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )

    monkeypatch.setenv("CHATLLM_OCR_MODE", "cli")
    monkeypatch.setenv("CHATLLM_OCR_CLI_BINARY", "chatllm")
    monkeypatch.setenv(
        "CHATLLM_OCR_CLI_ARGS_JSON",
        json.dumps(["ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
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

    spotting_result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="text",
        prompt_preset="spotting",
    )
    json_result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="markdown",
        prompt_preset="json",
    )

    assert spotting_result.format == "json"  # nosec B101
    assert spotting_result.raw["blocks"][0]["bbox"] == [10, 11, 12, 13]  # nosec B101  # type: ignore[index]
    assert json_result.format == "json"  # nosec B101
    assert json_result.text == "detected text"  # nosec B101


@pytest.mark.unit
def test_chatllm_cli_json_failure_degrades_with_warning(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )

    monkeypatch.setenv("CHATLLM_OCR_MODE", "cli")
    monkeypatch.setenv("CHATLLM_OCR_CLI_BINARY", "chatllm")
    monkeypatch.setenv(
        "CHATLLM_OCR_CLI_ARGS_JSON",
        json.dumps(["ocr", "--image", "{image_path}", "--prompt", "{prompt}"]),
    )

    def fake_run(cmd, capture_output, text, check, timeout):
        assert timeout == 60  # nosec B101
        return types.SimpleNamespace(stdout="plain extracted text", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = ChatLLMOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="json",
    )

    assert result.format == "text"  # nosec B101
    assert result.text == "plain extracted text"  # nosec B101
    assert result.raw == {"raw_output": "plain extracted text"}  # nosec B101
    assert "could not be parsed" in result.warnings[0]  # nosec B101


@pytest.mark.unit
def test_chatllm_describe_reports_managed_metadata(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr import (
        ChatLLMOCRBackend,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        reset_managed_process_registry,
    )

    reset_managed_process_registry()
    monkeypatch.setenv("CHATLLM_OCR_MODE", "managed")
    monkeypatch.setenv("CHATLLM_OCR_ALLOW_MANAGED_START", "true")
    monkeypatch.setenv("CHATLLM_OCR_HOST", "127.0.0.1")
    monkeypatch.setenv("CHATLLM_OCR_PORT", "19091")
    monkeypatch.setenv("CHATLLM_OCR_MODEL_PATH", "/models/chatllm.gguf")
    monkeypatch.setenv("CHATLLM_OCR_SERVER_BINARY", "chatllm-server")
    monkeypatch.setenv("CHATLLM_OCR_SERVER_ARGS_JSON", json.dumps(["--model", "{model_path}", "--port", "{port}"]))
    monkeypatch.setenv("CHATLLM_OCR_HEALTHCHECK_URL", "http://127.0.0.1:19091/ready")
    monkeypatch.setenv("CHATLLM_OCR_MAX_PAGE_CONCURRENCY", "2")
    monkeypatch.setenv("CHATLLM_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "false")

    description = ChatLLMOCRBackend().describe()

    assert description["mode"] == "managed"  # nosec B101
    assert description["model"] == "/models/chatllm.gguf"  # nosec B101
    assert description["managed_configured"] is True  # nosec B101
    assert description["allow_managed_start"] is True  # nosec B101
    assert description["managed_running"] is False  # nosec B101
    assert description["healthcheck_url_configured"] is True  # nosec B101
    assert description["backend_concurrency_cap"] == 2  # nosec B101
    assert description["auto_eligible"] is True  # nosec B101
    assert description["auto_high_quality_eligible"] is False  # nosec B101
