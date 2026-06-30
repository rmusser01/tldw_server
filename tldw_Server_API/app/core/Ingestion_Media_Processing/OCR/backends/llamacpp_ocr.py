"""Llama.cpp OCR backend with auto, remote, managed, and CLI runtime support."""

from __future__ import annotations

import base64
import contextlib
from dataclasses import replace
import json
import os
import subprocess  # nosec B404
import tempfile
from threading import RLock
from typing import Any

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
    CLIOCRProfile,
    ManagedOCRProfile,
    OCRRuntimeProfiles,
    RemoteOCRProfile,
    cleanup_managed_process,
    get_managed_process_record,
    is_profile_available,
    load_ocr_runtime_profiles,
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

_LlamaCppProfile = RemoteOCRProfile | ManagedOCRProfile | CLIOCRProfile
_LLAMACPP_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
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

_PROMPT_PRESETS: dict[str, str] = {
    "general": "Extract all visible text from the image.",
    "doc": "Parse the document and return all text in Markdown. Preserve document structure.",
    "table": "Extract tables faithfully in Markdown. Return all other text in Markdown.",
    "spotting": "Return JSON only with text spans and bounding boxes for each detected region.",
    "json": "Return JSON only with fields: text and blocks.",
}
_MANAGED_LIFECYCLE_LOCK = RLock()


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


def _resolve_profiles() -> OCRRuntimeProfiles:
    return load_ocr_runtime_profiles("LLAMACPP")


def _configured_mode() -> str:
    mode = (os.getenv("LLAMACPP_OCR_MODE") or "cli").strip().lower() or "cli"
    if mode in {"auto", "remote", "managed", "cli"}:
        return mode
    return "cli"


def _profile_for_mode(mode: str) -> _LlamaCppProfile:
    profiles = _resolve_profiles()
    if mode == "remote":
        return replace(profiles.remote, mode="remote")
    if mode == "managed":
        return replace(profiles.managed, mode="managed")
    return replace(profiles.cli, mode="cli")


def _active_profile() -> _LlamaCppProfile | None:
    mode = _configured_mode()
    if mode == "remote":
        return _profile_for_mode("remote")
    if mode == "managed":
        return _profile_for_mode("managed")
    if mode == "cli":
        return _profile_for_mode("cli")
    candidates = [
        _profile_for_mode("remote"),
        _profile_for_mode("managed"),
        _profile_for_mode("cli"),
    ]
    if managed_process_running(LlamaCppOCRBackend.name) or _managed_start_configured():
        candidates = [
            _profile_for_mode("managed"),
            _profile_for_mode("remote"),
            _profile_for_mode("cli"),
        ]
    for candidate in candidates:
        if is_profile_available(candidate, backend_name=LlamaCppOCRBackend.name):
            return candidate
    return None


def _active_mode() -> str:
    profile = _active_profile()
    if profile is None:
        return _configured_mode()
    return profile.normalized_mode()


def _wait_for_managed_http_ready(host: str, port: int, timeout_total: float) -> bool:
    return wait_for_managed_http_ready(host=host, port=port, timeout_total=timeout_total)


def _startup_timeout_seconds() -> float:
    raw = (
        os.getenv("LLAMACPP_OCR_STARTUP_TIMEOUT_SEC")
        or os.getenv("LLAMACPP_OCR_STARTUP_TIMEOUT")
        or "30"
    )
    try:
        return max(float(raw), 0.1)
    except (TypeError, ValueError):
        return 30.0


def _resolve_prompt(prompt_preset: str | None, output_format: str | None) -> str:
    env_prompt = os.getenv("LLAMACPP_OCR_PROMPT")
    if env_prompt:
        return env_prompt

    preset = (prompt_preset or "").strip().lower()
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
    preset = (prompt_preset or "").strip().lower()
    return preset in {"spotting", "json"}


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


def _ocr_via_remote(image_bytes: bytes, prompt: str) -> str:
    profile = _resolve_profiles().remote
    host = getattr(profile, "host", None) or os.getenv("LLAMACPP_OCR_HOST")
    port = getattr(profile, "port", None)
    if port is None:
        port = _env_int("LLAMACPP_OCR_PORT", 0)
    if not host or not port:
        raise RuntimeError("LLAMACPP remote OCR requires LLAMACPP_OCR_HOST and LLAMACPP_OCR_PORT")

    model = (
        getattr(profile, "model_path", None)
        or os.getenv("LLAMACPP_OCR_MODEL_PATH")
        or "llamacpp-ocr"
    )
    timeout = _env_int("LLAMACPP_OCR_TIMEOUT", 60)
    use_data_url = _env_bool("LLAMACPP_OCR_USE_DATA_URL", True)

    temp_path: str | None = None
    try:
        if use_data_url:
            image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(image_bytes)
                image_url = handle.name
                temp_path = handle.name

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": float(os.getenv("LLAMACPP_OCR_TEMPERATURE", "0")),
            "max_tokens": _env_int("LLAMACPP_OCR_MAX_TOKENS", 2048),
        }

        from tldw_Server_API.app.core.http_client import fetch_json

        response = fetch_json(
            method="POST",
            url=f"http://{host}:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        return _extract_message_content(response)
    finally:
        if temp_path:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)


def _resolve_managed_endpoint() -> tuple[str, int]:
    profile = _resolve_profiles().managed
    host = profile.host or os.getenv("LLAMACPP_OCR_HOST") or "127.0.0.1"
    port = profile.port
    if port is None:
        port = _env_int("LLAMACPP_OCR_PORT", 0)
    if port <= 0:
        raise RuntimeError("LLAMACPP managed OCR requires LLAMACPP_OCR_PORT")
    admin_port = _env_int("LLAMACPP_SERVER_PORT", 0)
    if admin_port <= 0:
        admin_port = _env_int("LLAMACPP_PORT", 0)
    if admin_port > 0 and port == admin_port:
        raise RuntimeError(
            "LLAMACPP managed OCR requires an OCR-private port distinct from "
            "the admin server port"
        )
    return host, port


def _managed_runtime_configured() -> bool:
    profile = _resolve_profiles().managed
    if not profile.argv:
        return False
    try:
        _resolve_managed_endpoint()
    except RuntimeError:
        return False
    return True


def _managed_start_configured() -> bool:
    profile = _resolve_profiles().managed
    return profile.allow_managed_start and _managed_runtime_configured()


def _ensure_managed_runtime() -> tuple[str, int]:
    with _MANAGED_LIFECYCLE_LOCK:
        timeout_total = _startup_timeout_seconds()
        record = get_managed_process_record(LlamaCppOCRBackend.name)
        if record is not None:
            record_host = record.host
            record_port = record.port
            if record_host and record_port is not None:
                if _wait_for_managed_http_ready(record_host, record_port, min(timeout_total, 1.0)):
                    return record_host, record_port
                cleanup_managed_process(LlamaCppOCRBackend.name, timeout=min(timeout_total, 1.0))
        profile = _resolve_profiles().managed
        host, port = _resolve_managed_endpoint()
        if not profile.allow_managed_start:
            raise RuntimeError("managed OCR startup is disabled")
        if not profile.argv:
            raise RuntimeError("managed OCR startup requires LLAMACPP_OCR_ARGV")

        command = render_argv_template(
            profile.argv,
            model_path=profile.model_path,
            image_path=None,
            prompt=profile.prompt,
            host=host,
            port=port,
        )
        process = subprocess.Popen(  # nosec B603
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

        if _wait_for_managed_http_ready(host, port, timeout_total):
            register_managed_process(
                LlamaCppOCRBackend.name,
                process,
                host=host,
                port=port,
                argv=command,
            )
            return host, port

        terminate_process(process)
        raise RuntimeError(f"managed OCR runtime did not become ready at {host}:{port}")


def _ocr_via_managed(image_bytes: bytes, prompt: str) -> str:
    host, port = _ensure_managed_runtime()
    model = (
        _resolve_profiles().managed.model_path
        or os.getenv("LLAMACPP_OCR_MODEL_PATH")
        or "llamacpp-ocr"
    )
    timeout = _env_int("LLAMACPP_OCR_TIMEOUT", 60)
    use_data_url = _env_bool("LLAMACPP_OCR_USE_DATA_URL", True)

    temp_path: str | None = None
    try:
        if use_data_url:
            image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(image_bytes)
                image_url = handle.name
                temp_path = handle.name

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": float(os.getenv("LLAMACPP_OCR_TEMPERATURE", "0")),
            "max_tokens": _env_int("LLAMACPP_OCR_MAX_TOKENS", 2048),
        }

        from tldw_Server_API.app.core.http_client import fetch_json

        response = fetch_json(
            method="POST",
            url=f"http://{host}:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        return _extract_message_content(response)
    finally:
        if temp_path:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)


def _remote_configured() -> bool:
    profile = _resolve_profiles().remote
    host = getattr(profile, "host", None) or os.getenv("LLAMACPP_OCR_HOST")
    port = getattr(profile, "port", None)
    if port is None:
        port = _env_int("LLAMACPP_OCR_PORT", 0)
    return bool(host) and bool(port and port > 0)


def _cli_configured() -> bool:
    return bool(_resolve_profiles().cli.argv)


def _ocr_via_cli(image_bytes: bytes, prompt: str) -> str:
    profile = _resolve_profiles().cli
    timeout_seconds = _env_int("LLAMACPP_OCR_TIMEOUT", 60)
    with tempfile.TemporaryDirectory(prefix="llamacpp_ocr_") as tmpdir:
        image_path = os.path.join(tmpdir, "page.png")
        with open(image_path, "wb") as handle:
            handle.write(image_bytes)
        command = render_argv_template(
            profile.argv,
            model_path=profile.model_path,
            image_path=image_path,
            prompt=prompt,
            host=None,
            port=None,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )  # nosec B603
        except subprocess.TimeoutExpired as exc:
            logging.error(
                "Llama.cpp OCR CLI timed out after %ss for model=%r image_path=%r",
                timeout_seconds,
                profile.model_path,
                image_path,
            )
            raise RuntimeError(f"llama.cpp CLI OCR timed out after {timeout_seconds}s") from exc
    return completed.stdout or ""


class LlamaCppOCRBackend(OCRBackend):
    """OCR backend that can call llama.cpp remotely, via a managed server, or via CLI."""

    name = "llamacpp"

    @classmethod
    def available(cls) -> bool:
        try:
            active = _active_profile()
            if active is None:
                return False
            return is_profile_available(active, backend_name=cls.name)
        except _LLAMACPP_NONCRITICAL_EXCEPTIONS:
            return False

    def describe(self) -> dict[str, Any]:
        profiles = _resolve_profiles()
        active = _active_profile()
        active_mode = _active_mode()
        if active is None:
            mode = _configured_mode()
            if mode == "remote":
                active = _profile_for_mode("remote")
            elif mode == "managed":
                active = _profile_for_mode("managed")
            else:
                active = _profile_for_mode("cli")
        host = getattr(active, "host", None)
        port = getattr(active, "port", None)
        description: dict[str, Any] = {
            "mode": active_mode,
            "configured_mode": _configured_mode(),
            "model": getattr(active, "model_path", None),
            "configured": (
                _remote_configured()
                or _cli_configured()
                or _managed_runtime_configured()
                or managed_process_running(self.name)
            ),
            "supports_structured_output": True,
            "supports_json": True,
            "prompt": getattr(active, "prompt", None) or os.getenv("LLAMACPP_OCR_PROMPT"),
            "configured_flags": os.getenv("LLAMACPP_OCR_CONFIGURED_FLAGS"),
            "auto_eligible": _env_bool("LLAMACPP_OCR_AUTO_ELIGIBLE", False),
            "auto_high_quality_eligible": _env_bool(
                "LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE",
                False,
            ),
            "backend_concurrency_cap": active.max_page_concurrency,
            "allow_managed_start": profiles.managed.allow_managed_start,
            "url_configured": _remote_configured(),
            "managed_configured": _managed_runtime_configured(),
            "managed_running": managed_process_running(self.name),
            "cli_configured": _cli_configured(),
            "argv": list(active.argv),
        }
        if host:
            description["host"] = host
        if port is not None:
            description["port"] = port
            description["url"] = f"http://{host}:{port}/v1/chat/completions" if host else None
        return description

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

        prompt = _resolve_prompt(prompt_preset, output_format)
        mode = _active_mode()
        raw_output = ""
        warnings: list[str] = []

        if not self.available():
            if mode == "managed":
                _ensure_managed_runtime()
            else:
                logging.warning(
                    "LlamaCppOCRBackend not available: configure LLAMACPP_OCR "
                    "runtime settings."
                )
                return OCRResult(text="", format=fmt, meta={"backend": self.name, "mode": mode})

        try:
            if mode == "remote":
                raw_output = _ocr_via_remote(image_bytes, prompt)
            elif mode == "cli":
                raw_output = _ocr_via_cli(image_bytes, prompt)
            elif mode == "managed":
                raw_output = _ocr_via_managed(image_bytes, prompt)
            else:
                logging.warning("LlamaCpp OCR mode is unsupported.")
                return OCRResult(text="", format=fmt, meta={"backend": self.name, "mode": mode})
        except _LLAMACPP_NONCRITICAL_EXCEPTIONS as exc:
            if mode == "managed" and isinstance(exc, RuntimeError):
                raise
            logging.error(f"LlamaCpp OCR failed: {exc}", exc_info=True)
            return OCRResult(text="", format=fmt, meta={"backend": self.name, "mode": mode})

        meta = {
            "backend": self.name,
            "mode": mode,
            "prompt_preset": prompt_preset,
            "output_format": output_format,
            "model": self.describe().get("model"),
            "configured_flags": os.getenv("LLAMACPP_OCR_CONFIGURED_FLAGS"),
            "auto_eligible": _env_bool("LLAMACPP_OCR_AUTO_ELIGIBLE", False),
            "auto_high_quality_eligible": _env_bool(
                "LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE",
                False,
            ),
            "backend_concurrency_cap": (
                _active_profile() or _resolve_profiles().cli
            ).max_page_concurrency,
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
            warnings=warnings,
        )
