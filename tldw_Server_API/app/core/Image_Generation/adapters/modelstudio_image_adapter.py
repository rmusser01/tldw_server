"""Alibaba Model Studio image-generation backend adapter."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
from tldw_Server_API.app.core.http_client import fetch_json
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.Image_Generation.adapters.image_format_utils import (
    decode_base64_image,
    decode_data_url,
    fetch_image_bytes,
    maybe_decode_base64_image,
    reference_image_data_url,
    validate_and_convert_image_output,
)
from tldw_Server_API.app.core.Image_Generation.capabilities import resolve_reference_image_capability
from tldw_Server_API.app.core.Image_Generation.config import (
    DEFAULT_MODELSTUDIO_IMAGE_BASE_URL,
    DEFAULT_MODELSTUDIO_IMAGE_MODEL,
    DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
    get_image_generation_config,
)
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_Server_API.app.core.Image_Generation.request_validation import effective_inline_max_bytes


class ModelStudioImageAdapter:
    name = "modelstudio"
    supported_formats = {"png", "jpg", "webp"}
    _DONE_STATES = {"success", "succeeded", "done", "finished", "completed"}
    _FAILED_STATES = {"failed", "error", "cancelled", "canceled"}
    _REGION_BASE_URLS = {
        "sg": "https://dashscope-intl.aliyuncs.com/api/v1",
        "us": "https://dashscope-us.aliyuncs.com/api/v1",
        "cn": "https://dashscope.aliyuncs.com/api/v1",
    }
    _ALLOWED_IMAGE_HOST_ALLOWLIST = ("aliyuncs.com",)

    def __init__(self) -> None:
        self._config = get_image_generation_config()

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        has_reference_image = request.reference_image is not None
        mode = self._resolve_mode(request)
        if mode == "sync" or has_reference_image:
            content, content_type = self._generate_sync(request)
            return self._finalize_result(content, content_type, output_format)
        if mode == "async":
            content, content_type = self._generate_async(request)
            return self._finalize_result(content, content_type, output_format)

        try:
            content, content_type = self._generate_sync(request)
        except ImageGenerationError as sync_exc:
            logger.info("Model Studio auto mode sync path failed; falling back to async")
            try:
                content, content_type = self._generate_async(request)
            except ImageGenerationError as async_exc:
                logger.warning(
                    "Model Studio auto mode failed on both paths: sync_error={} async_error={}",
                    sync_exc,
                    async_exc,
                )
                raise ImageGenerationError("Model Studio generation failed in auto mode") from async_exc
        return self._finalize_result(content, content_type, output_format)

    def _finalize_result(self, content: bytes, content_type: str, output_format: str) -> ImageGenResult:
        content, content_type = validate_and_convert_image_output(
            content,
            content_type,
            output_format,
            max_bytes=self._max_output_bytes(),
        )
        return ImageGenResult(content=content, content_type=content_type, bytes_len=len(content))

    def _max_output_bytes(self) -> int:
        return effective_inline_max_bytes(self._config)

    def _generate_sync(self, request: ImageGenRequest) -> tuple[bytes, str]:
        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        url = self._sync_generation_url(base_url)
        payload = self._build_sync_payload(request)

        try:
            data = fetch_json(
                method="POST",
                url=url,
                headers=self._headers(api_key),
                json=payload,
                timeout=self._config.modelstudio_image_timeout_seconds or DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Model Studio sync request failed: {}", exc)
            raise ImageGenerationError("Model Studio sync request failed") from exc
        return self._extract_image_content(data)

    def _generate_async(self, request: ImageGenRequest) -> tuple[bytes, str]:
        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        submit_url = self._async_submit_url(base_url)
        payload = self._build_async_payload(request)

        try:
            submit_data = fetch_json(
                method="POST",
                url=submit_url,
                headers=self._headers(api_key),
                json=payload,
                timeout=self._config.modelstudio_image_timeout_seconds or DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Model Studio async submit failed: {}", exc)
            raise ImageGenerationError("Model Studio async submit failed") from exc

        task_id = self._extract_task_id(submit_data)
        if not task_id:
            raise ImageGenerationError("Model Studio submit response did not include task id")

        timeout_seconds = float(
            self._config.modelstudio_image_timeout_seconds or DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS
        )
        poll_interval = max(0.1, float(self._config.modelstudio_image_poll_interval_seconds or 2))
        deadline = time.monotonic() + timeout_seconds
        poll_url = self._task_status_url(base_url, task_id)
        last_payload: dict[str, Any] = {}

        while time.monotonic() < deadline:
            try:
                poll_data = fetch_json(
                    method="GET",
                    url=poll_url,
                    headers=self._headers(api_key),
                    timeout=self._config.modelstudio_image_timeout_seconds or DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning("Model Studio async polling failed: {}", exc)
                raise ImageGenerationError("Model Studio async polling failed") from exc

            if isinstance(poll_data, dict):
                last_payload = poll_data
            state = self._extract_task_status(poll_data)
            if state in self._DONE_STATES:
                return self._extract_image_content(poll_data)
            if state in self._FAILED_STATES:
                detail = self._extract_error_message(poll_data) or state
                raise ImageGenerationError(f"Model Studio task failed: {detail}")
            time.sleep(poll_interval)

        detail = self._extract_error_message(last_payload) or "timed out waiting for Model Studio image task result"
        raise ImageGenerationError(detail)

    def _resolve_mode(self, request: ImageGenRequest) -> str:
        extra_mode = (request.extra_params or {}).get("mode") if isinstance(request.extra_params, dict) else None
        raw = str(extra_mode or self._config.modelstudio_image_mode or "auto").strip().lower()
        if raw not in {"sync", "async", "auto"}:
            return "auto"
        return raw

    def _resolve_api_key(self) -> str:
        api_key = (self._config.modelstudio_image_api_key or "").strip()
        if not api_key:
            api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            api_key = (os.getenv("QWEN_API_KEY") or "").strip()
        if not api_key:
            raise ImageBackendUnavailableError("modelstudio image api key is not configured")
        return api_key

    def _resolve_region(self) -> str:
        env_region = (os.getenv("MODELSTUDIO_IMAGE_REGION") or "").strip().lower()
        if env_region in self._REGION_BASE_URLS:
            return env_region
        cfg_region = (self._config.modelstudio_image_region or "").strip().lower()
        if cfg_region in self._REGION_BASE_URLS:
            return cfg_region
        return "sg"

    def _resolve_base_url(self) -> str:
        raw = os.getenv("MODELSTUDIO_IMAGE_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
        if not raw:
            raw = self._config.modelstudio_image_base_url
        if not raw:
            raw = self._REGION_BASE_URLS.get(self._resolve_region(), DEFAULT_MODELSTUDIO_IMAGE_BASE_URL)
        cleaned = str(raw).strip()
        if not cleaned:
            raise ImageBackendUnavailableError("modelstudio image base URL is not configured")
        if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = f"https://{cleaned}"
        return cleaned.rstrip("/")

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _sync_generation_url(base_url: str) -> str:
        suffix = "/services/aigc/multimodal-generation/generation"
        if base_url.endswith(suffix):
            return base_url
        return f"{base_url}{suffix}"

    @staticmethod
    def _async_submit_url(base_url: str) -> str:
        suffix = "/services/aigc/text2image/image-synthesis"
        if base_url.endswith(suffix):
            return base_url
        return f"{base_url}{suffix}"

    @staticmethod
    def _task_status_url(base_url: str, task_id: str) -> str:
        base = base_url.rstrip("/")
        return f"{base}/tasks/{task_id}"

    def _resolve_model(self, request: ImageGenRequest) -> str:
        return (
            request.model
            or os.getenv("MODELSTUDIO_IMAGE_MODEL")
            or self._config.modelstudio_image_default_model
            or DEFAULT_MODELSTUDIO_IMAGE_MODEL
        )

    @staticmethod
    def _apply_extra_payload_fields(payload: dict[str, Any], extra_params: Any) -> None:
        if not isinstance(extra_params, dict):
            return
        for key, value in extra_params.items():
            if key in {"prompt", "negative_prompt", "mode"}:
                continue
            payload[key] = value

    def _build_sync_payload(self, request: ImageGenRequest) -> dict[str, Any]:
        prompt = request.prompt.strip()
        if request.negative_prompt:
            prompt = f"{prompt}\n\nNegative prompt: {request.negative_prompt.strip()}"

        model = self._resolve_model(request)
        if request.reference_image is not None:
            capability = resolve_reference_image_capability("modelstudio", model, config=self._config)
            if not capability.supported:
                raise ImageGenerationError(f"Model Studio reference images have unsupported model: {model}")

        content: list[dict[str, Any]] = []
        if request.reference_image is not None:
            content.append({"image": reference_image_data_url(request.reference_image)})
        content.append({"text": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
        }
        parameters: dict[str, Any] = {}
        if request.width and request.height:
            parameters["size"] = f"{request.width}*{request.height}"
        if request.seed is not None:
            parameters["seed"] = request.seed
        if request.steps is not None:
            parameters["steps"] = request.steps
        if request.cfg_scale is not None:
            parameters["guidance_scale"] = request.cfg_scale
        if request.sampler:
            parameters["sampler"] = request.sampler
        if parameters:
            payload["parameters"] = parameters

        self._apply_extra_payload_fields(payload, request.extra_params or {})
        return payload

    def _build_async_payload(self, request: ImageGenRequest) -> dict[str, Any]:
        if request.reference_image is not None:
            raise ImageGenerationError("Model Studio reference images are only supported in sync mode")

        prompt = request.prompt.strip()
        if request.negative_prompt:
            prompt = f"{prompt}\n\nNegative prompt: {request.negative_prompt.strip()}"

        payload: dict[str, Any] = {
            "model": self._resolve_model(request),
            "input": {"prompt": prompt},
        }

        parameters: dict[str, Any] = {}
        if request.width is not None:
            parameters["width"] = request.width
        if request.height is not None:
            parameters["height"] = request.height
        if request.seed is not None:
            parameters["seed"] = request.seed
        if request.steps is not None:
            parameters["steps"] = request.steps
        if request.cfg_scale is not None:
            parameters["guidance_scale"] = request.cfg_scale
        if request.sampler:
            parameters["sampler"] = request.sampler
        if parameters:
            payload["parameters"] = parameters

        self._apply_extra_payload_fields(payload, request.extra_params or {})
        return payload

    def _extract_image_content(self, data: Any) -> tuple[bytes, str]:
        candidate = self._extract_from_node(data)
        if candidate:
            return candidate
        raise ImageGenerationError("Model Studio did not return image content")

    def _extract_from_node(self, node: Any) -> tuple[bytes, str] | None:
        if isinstance(node, dict):
            for key in ("b64_json", "image_base64", "base64", "image_b64"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return decode_base64_image(value.strip(), max_bytes=self._max_output_bytes()), "image/png"

            for key in ("image_url", "url", "image"):
                if key in node:
                    extracted = self._extract_from_link_value(node.get(key))
                    if extracted:
                        return extracted

            for key in ("images", "data", "choices", "message", "content", "output", "result", "results"):
                if key not in node:
                    continue
                extracted = self._extract_from_node(node.get(key))
                if extracted:
                    return extracted
            return None

        if isinstance(node, list):
            for item in node:
                extracted = self._extract_from_node(item)
                if extracted:
                    return extracted
            return None

        return self._extract_from_link_value(node)

    def _extract_from_link_value(self, value: Any) -> tuple[bytes, str] | None:
        if isinstance(value, dict):
            for key in ("url", "image_url", "b64_json", "base64", "image"):
                if key in value:
                    extracted = self._extract_from_link_value(value.get(key))
                    if extracted:
                        return extracted
            return None

        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("data:"):
            return decode_data_url(raw, max_bytes=self._max_output_bytes())
        if raw.startswith("http://") or raw.startswith("https://"):
            if not self._is_allowed_remote_image_url(raw):
                raise ImageGenerationError("Model Studio returned unsupported image URL host")
            return fetch_image_bytes(
                raw,
                timeout=self._config.modelstudio_image_timeout_seconds or DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
                max_bytes=self._max_output_bytes(),
            )
        decoded = maybe_decode_base64_image(raw, max_bytes=self._max_output_bytes())
        if decoded is not None:
            return decoded, "image/png"
        return None

    def _is_allowed_remote_image_url(self, raw_url: str) -> bool:
        base_host = (urlparse(self._resolve_base_url()).hostname or "").strip().lower()
        allowlist = list(self._ALLOWED_IMAGE_HOST_ALLOWLIST)
        if base_host:
            allowlist.append(base_host)
        policy_result = evaluate_url_policy(raw_url, allowlist=allowlist)
        if policy_result.allowed:
            return True
        logger.warning(
            "Model Studio blocked remote image URL fetch url={} reason={}",
            raw_url,
            policy_result.reason or "policy_denied",
        )
        return False

    def _extract_task_id(self, data: Any) -> str | None:
        if isinstance(data, dict):
            for key in ("task_id", "taskId", "id"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("output", "data", "result"):
                nested = data.get(key)
                task_id = self._extract_task_id(nested)
                if task_id:
                    return task_id
        return None

    def _extract_task_status(self, data: Any) -> str:
        if isinstance(data, dict):
            for key in ("task_status", "status", "state", "task_state"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
            for key in ("output", "data", "result"):
                nested = data.get(key)
                nested_state = self._extract_task_status(nested)
                if nested_state:
                    return nested_state
        return ""

    def _extract_error_message(self, data: Any) -> str | None:
        if isinstance(data, dict):
            for key in ("message", "error", "error_message", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("output", "data", "result"):
                nested = data.get(key)
                nested_message = self._extract_error_message(nested)
                if nested_message:
                    return nested_message
        return None
