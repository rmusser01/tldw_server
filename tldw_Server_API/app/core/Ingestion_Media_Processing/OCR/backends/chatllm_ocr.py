"""ChatLLM OCR backend with remote, managed, and CLI execution modes."""

from __future__ import annotations

import base64
import json
import os
import subprocess  # nosec B404
import tempfile
from threading import RLock
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
    cleanup_managed_process,
    get_managed_process_record,
    managed_process_running,
    register_managed_process,
    render_argv_template,
    terminate_process,
    wait_for_managed_http_ready,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import (
    OCRResult,
    normalize_ocr_format,
)
from tldw_Server_API.app.core.Utils.Utils import logging

_CHATLLM_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_MANAGED_LIFECYCLE_LOCK = RLock()
_PROMPT_PRESETS: dict[str, str] = {
    "general": "Extract all visible text from the image.",
    "doc": "Parse the document and return all text in Markdown. Preserve document structure.",
    "table": "Extract tables faithfully in Markdown. Return all other text in Markdown.",
    "spotting": "Return JSON only with text spans and bounding boxes for each detected region.",
    "json": "Return JSON only with fields: text and blocks.",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _env_json_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def _startup_timeout_seconds() -> float:
    raw = os.getenv("CHATLLM_OCR_STARTUP_TIMEOUT_SEC") or "30"
    try:
        return max(float(raw), 0.1)
    except (TypeError, ValueError):
        return 30.0


def _resolve_mode() -> str:
    return (os.getenv("CHATLLM_OCR_MODE") or "remote").strip().lower()


def _active_mode() -> str:
    mode = _resolve_mode()
    if mode != "auto":
        return mode
    if _remote_configured():
        return "remote"
    if managed_process_running(ChatLLMOCRBackend.name) or _managed_start_configured():
        return "managed"
    if _cli_configured():
        return "cli"
    return "auto"


def _resolve_prompt(prompt_preset: str | None, output_format: str | None) -> str:
    preset = (
        prompt_preset
        or os.getenv("CHATLLM_OCR_PROMPT_PRESET_DEFAULT")
        or ""
    ).strip().lower()
    if not preset:
        fmt = normalize_ocr_format(output_format)
        if fmt == "json":
            preset = "json"
        elif fmt == "markdown":
            preset = "doc"
        else:
            preset = "general"
    return _PROMPT_PRESETS.get(preset, _PROMPT_PRESETS["general"])


def _structured_preset_requested(prompt_preset: str | None) -> bool:
    return (prompt_preset or "").strip().lower() in {"spotting", "json"}


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return str(first_choice or "")

    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        return str(message or "")

    content = message.get("content", "")
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(content, str):
        return content
    return str(content or "")


def _extract_text_from_json(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("text", "markdown", "content", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            block_text = [
                str(block.get("text", "")).strip()
                for block in blocks
                if isinstance(block, dict) and str(block.get("text", "")).strip()
            ]
            if block_text:
                return "\n".join(block_text)
    if isinstance(payload, list):
        items = [str(item).strip() for item in payload if str(item).strip()]
        return "\n".join(items)
    if isinstance(payload, str):
        return payload
    return ""


def _parse_cli_json(stdout: str) -> Any | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _resolve_remote_url() -> str:
    url = (os.getenv("CHATLLM_OCR_URL") or "").strip()
    if not url:
        raise RuntimeError("CHATLLM_OCR_URL is required for remote mode")
    if url.endswith("/v1/chat/completions"):
        return url
    if url.rstrip("/").endswith("/v1"):
        return url.rstrip("/") + "/chat/completions"
    return url.rstrip("/") + "/v1/chat/completions"


def _remote_configured() -> bool:
    return bool((os.getenv("CHATLLM_OCR_URL") or "").strip()) and bool(
        (os.getenv("CHATLLM_OCR_MODEL") or "").strip()
    )


def _cli_binary() -> str | None:
    raw = (os.getenv("CHATLLM_OCR_CLI_BINARY") or "").strip()
    return raw or None


def _cli_argv() -> tuple[str, ...]:
    return _env_json_list("CHATLLM_OCR_CLI_ARGS_JSON")


def _cli_configured() -> bool:
    return bool(_cli_binary()) and bool(_cli_argv())


def _managed_server_binary() -> str | None:
    raw = (os.getenv("CHATLLM_OCR_SERVER_BINARY") or "").strip()
    return raw or None


def _managed_server_argv() -> tuple[str, ...]:
    return _env_json_list("CHATLLM_OCR_SERVER_ARGS_JSON")


def _resolve_managed_endpoint() -> tuple[str, int]:
    host = (os.getenv("CHATLLM_OCR_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = _env_int("CHATLLM_OCR_PORT", 0)
    if port <= 0:
        raise RuntimeError("CHATLLM managed OCR requires CHATLLM_OCR_PORT")
    return host, port


def _healthcheck_url_configured() -> bool:
    return bool((os.getenv("CHATLLM_OCR_HEALTHCHECK_URL") or "").strip())


def _managed_configured() -> bool:
    if (
        not _managed_server_binary()
        or not _managed_server_argv()
        or not _healthcheck_url_configured()
    ):
        return False
    try:
        _resolve_managed_endpoint()
    except RuntimeError:
        return False
    return True


def _managed_start_configured() -> bool:
    return _env_bool("CHATLLM_OCR_ALLOW_MANAGED_START", False) and _managed_configured()


def _wait_for_healthcheck(timeout_total: float) -> bool:
    configured_url = (os.getenv("CHATLLM_OCR_HEALTHCHECK_URL") or "").strip()
    if configured_url:
        parsed = urlparse(configured_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return wait_for_managed_http_ready(
            host=host,
            port=port,
            scheme=parsed.scheme or "http",
            timeout_total=timeout_total,
            paths=(path,),
        )

    host, port = _resolve_managed_endpoint()
    return wait_for_managed_http_ready(host=host, port=port, timeout_total=timeout_total)


def _build_openai_payload(image_bytes: bytes, prompt: str) -> dict[str, Any]:
    return {
        "model": (
            os.getenv("CHATLLM_OCR_MODEL")
            or os.getenv("CHATLLM_OCR_MODEL_PATH")
            or "chatllm-ocr"
        ),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                f"{base64.b64encode(image_bytes).decode('ascii')}"
                            )
                        },
                    },
                ],
            }
        ],
        "temperature": float(os.getenv("CHATLLM_OCR_TEMPERATURE", "0")),
        "max_tokens": _env_int("CHATLLM_OCR_MAX_TOKENS", 2048),
    }


def _request_headers() -> dict[str, str] | None:
    api_key = (os.getenv("CHATLLM_OCR_API_KEY") or "").strip()
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _ocr_via_remote(image_bytes: bytes, prompt: str) -> str:
    from tldw_Server_API.app.core.http_client import fetch_json

    response = fetch_json(
        method="POST",
        url=_resolve_remote_url(),
        json=_build_openai_payload(image_bytes, prompt),
        timeout=_env_int("CHATLLM_OCR_TIMEOUT_SEC", 60),
        headers=_request_headers(),
        require_json_ct=False,
    )
    return _extract_message_content(response)


def _ensure_managed_runtime() -> tuple[str, int]:
    with _MANAGED_LIFECYCLE_LOCK:
        timeout_total = _startup_timeout_seconds()
        record = get_managed_process_record(ChatLLMOCRBackend.name)
        if record is not None and record.host and record.port is not None:
            if _wait_for_healthcheck(min(timeout_total, 1.0)):
                return record.host, record.port
            cleanup_managed_process(ChatLLMOCRBackend.name)

        host, port = _resolve_managed_endpoint()
        if not _env_bool("CHATLLM_OCR_ALLOW_MANAGED_START", False):
            raise RuntimeError("managed OCR startup is disabled")
        binary = _managed_server_binary()
        argv = _managed_server_argv()
        if not binary or not argv:
            raise RuntimeError(
                "managed OCR startup requires CHATLLM_OCR_SERVER_BINARY and "
                "CHATLLM_OCR_SERVER_ARGS_JSON"
            )

        command = [binary, *render_argv_template(
            argv,
            model_path=os.getenv("CHATLLM_OCR_MODEL_PATH"),
            image_path=None,
            prompt=None,
            host=host,
            port=port,
        )]
        process = subprocess.Popen(  # nosec B603
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        if _wait_for_healthcheck(timeout_total):
            register_managed_process(
                ChatLLMOCRBackend.name,
                process,
                host=host,
                port=port,
                argv=command,
            )
            return host, port

        terminate_process(process)
        raise RuntimeError(f"managed OCR runtime did not become ready at {host}:{port}")


def _ocr_via_managed(image_bytes: bytes, prompt: str) -> str:
    from tldw_Server_API.app.core.http_client import fetch_json

    host, port = _ensure_managed_runtime()
    response = fetch_json(
        method="POST",
        url=f"http://{host}:{port}/v1/chat/completions",
        json=_build_openai_payload(image_bytes, prompt),
        timeout=_env_int("CHATLLM_OCR_TIMEOUT_SEC", 60),
        headers=_request_headers(),
        require_json_ct=False,
    )
    return _extract_message_content(response)


def _ocr_via_cli(image_bytes: bytes, prompt: str) -> str:
    binary = _cli_binary()
    argv = _cli_argv()
    if not binary or not argv:
        raise RuntimeError("CHATLLM CLI OCR requires binary and args")

    with tempfile.TemporaryDirectory(prefix="chatllm_ocr_") as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        with open(image_path, "wb") as handle:
            handle.write(image_bytes)
        command = [binary, *render_argv_template(
            argv,
            model_path=os.getenv("CHATLLM_OCR_MODEL_PATH"),
            image_path=image_path,
            prompt=prompt,
            host=None,
            port=None,
        )]
        completed = subprocess.run(  # nosec B603
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_env_int("CHATLLM_OCR_TIMEOUT_SEC", 60),
        )
    return completed.stdout or ""


class ChatLLMOCRBackend(OCRBackend):
    """OCR backend that can use ChatLLM through an endpoint, managed runtime, or CLI."""

    name = "chatllm"

    @classmethod
    def auto_eligible(cls, high_quality: bool) -> bool:
        if high_quality:
            return _env_bool("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", False)
        return _env_bool("CHATLLM_OCR_AUTO_ELIGIBLE", False)

    @classmethod
    def available(cls) -> bool:
        mode = _active_mode()
        try:
            if mode == "remote":
                return _remote_configured()
            if mode == "managed":
                return managed_process_running(cls.name) or _managed_start_configured()
            if mode == "cli":
                return _cli_configured()
        except _CHATLLM_NONCRITICAL_EXCEPTIONS:
            return False
        return False

    def describe(self) -> dict[str, Any]:
        mode = _active_mode()
        return {
            "mode": mode,
            "model": os.getenv("CHATLLM_OCR_MODEL_PATH") or os.getenv("CHATLLM_OCR_MODEL"),
            "configured": (
                _remote_configured()
                or _cli_configured()
                or _managed_configured()
                or managed_process_running(self.name)
            ),
            "supports_structured_output": True,
            "supports_json": True,
            "auto_eligible": _env_bool("CHATLLM_OCR_AUTO_ELIGIBLE", False),
            "auto_high_quality_eligible": _env_bool(
                "CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE",
                False,
            ),
            "managed_configured": mode == "managed" and _managed_configured(),
            "managed_running": managed_process_running(self.name),
            "allow_managed_start": _env_bool("CHATLLM_OCR_ALLOW_MANAGED_START", False),
            "url_configured": _remote_configured(),
            "healthcheck_url_configured": _healthcheck_url_configured(),
            "cli_configured": _cli_configured(),
            "backend_concurrency_cap": _env_int("CHATLLM_OCR_MAX_PAGE_CONCURRENCY", 1),
        }

    def ocr_image(self, image_bytes: bytes, lang: str | None = None) -> str:
        result = self.ocr_image_structured(image_bytes, lang=lang, output_format="text")
        return result.text or ""

    def ocr_image_structured(
        self,
        image_bytes: bytes,
        lang: str | None = None,
        output_format: str | None = None,
        prompt_preset: str | None = None,
    ) -> OCRResult:
        fmt = normalize_ocr_format(output_format)
        if fmt == "unknown":
            fmt = "text"

        mode = _active_mode()
        prompt = _resolve_prompt(prompt_preset, output_format)
        if not self.available():
            if mode == "managed":
                _ensure_managed_runtime()
            else:
                logging.warning(
                    "ChatLLM OCR backend not available: configure CHATLLM_OCR "
                    "runtime settings."
                )
                return OCRResult(text="", format=fmt, meta={"backend": self.name, "mode": mode})

        try:
            if mode == "remote":
                raw_output = _ocr_via_remote(image_bytes, prompt)
            elif mode == "managed":
                raw_output = _ocr_via_managed(image_bytes, prompt)
            elif mode == "cli":
                raw_output = _ocr_via_cli(image_bytes, prompt)
            else:
                logging.warning("ChatLLM OCR mode is unsupported.")
                return OCRResult(text="", format=fmt, meta={"backend": self.name, "mode": mode})
        except _CHATLLM_NONCRITICAL_EXCEPTIONS as exc:
            if mode == "managed" and isinstance(exc, RuntimeError):
                raise
            logging.error(f"ChatLLM OCR failed: {exc}", exc_info=True)
            return OCRResult(text="", format=fmt, meta={"backend": self.name, "mode": mode})

        meta = {
            "backend": self.name,
            "mode": mode,
            "prompt_preset": prompt_preset,
            "output_format": output_format,
            "model": self.describe().get("model"),
        }
        parse_structured = fmt == "json" or _structured_preset_requested(prompt_preset)
        if parse_structured:
            parsed = _parse_cli_json(raw_output)
            if parsed is not None:
                return OCRResult(
                    text=_extract_text_from_json(parsed),
                    format="json",
                    raw=parsed,
                    meta=meta,
                )
            warning = (
                "JSON output requested but CLI output could not be parsed; "
                "returning plain text."
            )
            if fmt == "json":
                return OCRResult(
                    text=raw_output.strip(),
                    format="text",
                    raw={"raw_output": raw_output.strip()},
                    meta=meta,
                    warnings=[warning],
                )
            return OCRResult(
                text=raw_output.strip(),
                format=fmt,
                raw={"raw_output": raw_output.strip()},
                meta=meta,
                warnings=[warning],
            )

        return OCRResult(
            text=raw_output.strip(),
            format=fmt,
            raw=None,
            meta=meta,
        )
