"""Hunyuan-specific llama.cpp runtime support for GGUF OCR deployments."""

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

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
    CLIOCRProfile,
    ManagedOCRProfile,
    OCRRuntimeProfiles,
    RemoteOCRProfile,
    cleanup_managed_process,
    get_managed_process_record,
    is_profile_available,
    load_ocr_runtime_profiles_from_keys,
    managed_process_running,
    register_managed_process,
    render_argv_template,
    terminate_process,
    wait_for_managed_http_ready,
)
from tldw_Server_API.app.core.Utils.Utils import logging

_HunyuanLlamaCppProfile = RemoteOCRProfile | ManagedOCRProfile | CLIOCRProfile
_HUNYUAN_LLAMACPP_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
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
_MANAGED_PROCESS_KEY = "hunyuan_llamacpp"
_CLI_OUTPUT_SNIPPET_LIMIT = 2000


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
    return load_ocr_runtime_profiles_from_keys(
        mode_key="HUNYUAN_LLAMACPP_MODE",
        allow_managed_start_key="HUNYUAN_LLAMACPP_ALLOW_MANAGED_START",
        max_page_concurrency_key="HUNYUAN_LLAMACPP_MAX_PAGE_CONCURRENCY",
        host_key="HUNYUAN_LLAMACPP_HOST",
        port_key="HUNYUAN_LLAMACPP_PORT",
        remote_model_key="HUNYUAN_LLAMACPP_MODEL",
        model_path_key="HUNYUAN_LLAMACPP_MODEL_PATH",
        managed_argv_key="HUNYUAN_LLAMACPP_SERVER_ARGV",
        cli_argv_key="HUNYUAN_LLAMACPP_CLI_ARGV",
    )


def _configured_mode() -> str:
    mode = (os.getenv("HUNYUAN_LLAMACPP_MODE") or "auto").strip().lower() or "auto"
    if mode in {"auto", "remote", "managed", "cli"}:
        return mode
    return "auto"


def _profile_for_mode(mode: str) -> _HunyuanLlamaCppProfile:
    profiles = _resolve_profiles()
    if mode == "remote":
        return replace(profiles.remote, mode="remote")
    if mode == "managed":
        return replace(profiles.managed, mode="managed")
    return replace(profiles.cli, mode="cli")


def _active_profile() -> _HunyuanLlamaCppProfile | None:
    mode = _configured_mode()
    if mode == "remote":
        return _profile_for_mode("remote")
    if mode == "managed":
        return _profile_for_mode("managed")
    if mode == "cli":
        return _profile_for_mode("cli")

    for candidate in (
        _profile_for_mode("remote"),
        _profile_for_mode("managed"),
        _profile_for_mode("cli"),
    ):
        if is_profile_available(candidate, backend_name=_MANAGED_PROCESS_KEY):
            return candidate
    return None


def _active_mode() -> str:
    profile = _active_profile()
    if profile is None:
        return _configured_mode()
    return profile.normalized_mode()


def _startup_timeout_seconds() -> float:
    raw = os.getenv("HUNYUAN_LLAMACPP_STARTUP_TIMEOUT_SEC") or "30"
    try:
        return max(float(raw), 0.1)
    except (TypeError, ValueError):
        return 30.0


def _wait_for_managed_http_ready(host: str, port: int, timeout_total: float) -> bool:
    return wait_for_managed_http_ready(host=host, port=port, timeout_total=timeout_total)


def _resolve_managed_endpoint() -> tuple[str, int]:
    profile = _profile_for_mode("managed")
    host = profile.host or os.getenv("HUNYUAN_LLAMACPP_HOST") or "127.0.0.1"
    port = profile.port
    if port is None:
        port = _env_int("HUNYUAN_LLAMACPP_PORT", 0)
    if port <= 0:
        raise RuntimeError("HUNYUAN_LLAMACPP managed OCR requires HUNYUAN_LLAMACPP_PORT")
    return host, port


def _remote_configured() -> bool:
    profile = _profile_for_mode("remote")
    host = profile.host or os.getenv("HUNYUAN_LLAMACPP_HOST")
    port = profile.port
    if port is None:
        port = _env_int("HUNYUAN_LLAMACPP_PORT", 0)
    return bool(host) and bool(port and port > 0)


def _managed_runtime_configured() -> bool:
    profile = _profile_for_mode("managed")
    if not profile.argv:
        return False
    try:
        _resolve_managed_endpoint()
    except RuntimeError:
        return False
    return True


def _managed_start_configured() -> bool:
    profile = _profile_for_mode("managed")
    return profile.allow_managed_start and _managed_runtime_configured()


def _cli_configured() -> bool:
    return bool(_profile_for_mode("cli").argv)


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
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    if isinstance(content, str):
        return content
    return str(content or "")


def _remote_or_managed_model(profile: _HunyuanLlamaCppProfile | None = None) -> str:
    if profile is None:
        profile = _active_profile()
    if getattr(profile, "model", None):
        return str(profile.model)
    if os.getenv("HUNYUAN_LLAMACPP_MODEL"):
        return str(os.getenv("HUNYUAN_LLAMACPP_MODEL"))
    if getattr(profile, "model_path", None):
        return str(profile.model_path)
    if os.getenv("HUNYUAN_LLAMACPP_MODEL_PATH"):
        return str(os.getenv("HUNYUAN_LLAMACPP_MODEL_PATH"))
    return "HunyuanOCR"


def _send_openai_compatible_request(
    *,
    image_bytes: bytes,
    prompt: str,
    host: str,
    port: int,
    model: str,
) -> str:
    timeout = _env_int("HUNYUAN_LLAMACPP_TIMEOUT", 60)
    use_data_url = _env_bool("HUNYUAN_LLAMACPP_USE_DATA_URL", True)
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
            "temperature": float(os.getenv("HUNYUAN_LLAMACPP_TEMPERATURE", "0")),
            "max_tokens": _env_int("HUNYUAN_LLAMACPP_MAX_TOKENS", 2048),
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


def _ocr_via_remote(image_bytes: bytes, prompt: str) -> str:
    profile = _profile_for_mode("remote")
    host = profile.host or os.getenv("HUNYUAN_LLAMACPP_HOST")
    port = profile.port
    if port is None:
        port = _env_int("HUNYUAN_LLAMACPP_PORT", 0)
    if not host or not port:
        raise RuntimeError(
            "HUNYUAN_LLAMACPP remote OCR requires HUNYUAN_LLAMACPP_HOST and HUNYUAN_LLAMACPP_PORT"
        )
    return _send_openai_compatible_request(
        image_bytes=image_bytes,
        prompt=prompt,
        host=str(host),
        port=int(port),
        model=_remote_or_managed_model(profile),
    )


def _ensure_managed_runtime() -> tuple[str, int]:
    with _MANAGED_LIFECYCLE_LOCK:
        timeout_total = _startup_timeout_seconds()
        record = get_managed_process_record(_MANAGED_PROCESS_KEY)
        if record is not None:
            record_host = record.host
            record_port = record.port
            if record_host and record_port is not None:
                if _wait_for_managed_http_ready(record_host, record_port, min(timeout_total, 1.0)):
                    return record_host, record_port
                cleanup_managed_process(_MANAGED_PROCESS_KEY, timeout=min(timeout_total, 1.0))

        profile = _profile_for_mode("managed")
        host, port = _resolve_managed_endpoint()
        if not profile.allow_managed_start:
            raise RuntimeError("managed OCR startup is disabled")
        if not profile.argv:
            raise RuntimeError(
                "managed OCR startup requires HUNYUAN_LLAMACPP_SERVER_ARGV"
            )

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
                _MANAGED_PROCESS_KEY,
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
    profile = _profile_for_mode("managed")
    return _send_openai_compatible_request(
        image_bytes=image_bytes,
        prompt=prompt,
        host=host,
        port=port,
        model=_remote_or_managed_model(profile),
    )


def _trim_process_output(value: str | None, *, limit: int = _CLI_OUTPUT_SNIPPET_LIMIT) -> str:
    output = str(value or "").strip()
    if len(output) <= limit:
        return output
    return f"{output[:limit]}...<truncated>"


def _ocr_via_cli(image_bytes: bytes, prompt: str) -> str:
    profile = _profile_for_mode("cli")
    timeout_seconds = _env_int("HUNYUAN_LLAMACPP_TIMEOUT", 60)
    with tempfile.TemporaryDirectory(prefix="hunyuan_llamacpp_ocr_") as tmpdir:
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
                "Hunyuan llama.cpp OCR CLI timed out after %ss for model=%r image_path=%r",
                timeout_seconds,
                profile.model_path,
                image_path,
            )
            raise RuntimeError(
                f"Hunyuan llama.cpp CLI OCR timed out after {timeout_seconds}s"
            ) from exc
        if completed.returncode != 0:
            stderr = _trim_process_output(completed.stderr)
            stdout = _trim_process_output(completed.stdout)
            logging.error(
                "Hunyuan llama.cpp OCR CLI exited with code %s; stderr=%r",
                completed.returncode,
                stderr,
            )
            details = [
                f"Hunyuan llama.cpp CLI OCR failed with exit code {completed.returncode}"
            ]
            if stderr:
                details.append(f"stderr: {stderr}")
            if stdout:
                details.append(f"stdout: {stdout}")
            raise RuntimeError("; ".join(details))
    return completed.stdout or ""


class HunyuanLlamaCppRuntime:
    """Helper for Hunyuan GGUF OCR runtimes backed by llama.cpp."""

    @classmethod
    def auto_eligible(cls, high_quality: bool) -> bool:
        if high_quality:
            return _env_bool("HUNYUAN_LLAMACPP_AUTO_HIGH_QUALITY_ELIGIBLE", False)
        return _env_bool("HUNYUAN_LLAMACPP_AUTO_ELIGIBLE", False)

    @classmethod
    def available(cls) -> bool:
        try:
            active = _active_profile()
            if active is None:
                return False
            return is_profile_available(active, backend_name=_MANAGED_PROCESS_KEY)
        except _HUNYUAN_LLAMACPP_NONCRITICAL_EXCEPTIONS:
            return False

    @classmethod
    def active_mode(cls) -> str:
        return _active_mode()

    @classmethod
    def describe(cls) -> dict[str, Any]:
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
            "model": _remote_or_managed_model(active),
            "configured": (
                _remote_configured()
                or _cli_configured()
                or _managed_runtime_configured()
                or managed_process_running(_MANAGED_PROCESS_KEY)
            ),
            "supports_structured_output": True,
            "supports_json": True,
            "auto_eligible": cls.auto_eligible(False),
            "auto_high_quality_eligible": cls.auto_eligible(True),
            "backend_concurrency_cap": active.max_page_concurrency,
            "allow_managed_start": profiles.managed.allow_managed_start,
            "url_configured": _remote_configured(),
            "managed_configured": _managed_runtime_configured(),
            "managed_running": managed_process_running(_MANAGED_PROCESS_KEY),
            "cli_configured": _cli_configured(),
            "argv": list(active.argv),
        }
        if host:
            description["host"] = host
        if port is not None:
            description["port"] = port
            description["url"] = f"http://{host}:{port}/v1/chat/completions" if host else None
        return description

    @classmethod
    def ocr_image(cls, image_bytes: bytes, prompt: str) -> str:
        mode = _active_mode()
        if mode == "remote":
            return _ocr_via_remote(image_bytes, prompt)
        if mode == "managed":
            return _ocr_via_managed(image_bytes, prompt)
        if mode == "cli":
            return _ocr_via_cli(image_bytes, prompt)
        raise RuntimeError("Hunyuan llama.cpp OCR runtime is not configured")
