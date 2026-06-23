"""Novita image-generation backend adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from tldw_Server_API.app.core.http_client import fetch_json
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.Image_Generation.adapters.image_format_utils import (
    decode_base64_image,
    decode_data_url,
    fetch_image_bytes,
    maybe_decode_base64_image,
    validate_and_convert_image_output,
)
from tldw_Server_API.app.core.Image_Generation.config import (
    DEFAULT_NOVITA_IMAGE_BASE_URL,
    DEFAULT_NOVITA_IMAGE_MODEL,
    DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS,
    get_image_generation_config,
)
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_Server_API.app.core.Image_Generation.request_validation import effective_inline_max_bytes


class NovitaImageAdapter:
    name = "novita"
    supported_formats = {"png", "jpg", "webp"}
    _DONE_STATES = {"success", "succeeded", "done", "finished", "completed"}
    _FAILED_STATES = {"failed", "error", "cancelled", "canceled"}
    _PENDING_STATES = {"pending", "queued", "running", "processing", "in_progress"}

    def __init__(self) -> None:
        self._config = get_image_generation_config()

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()

        submit_url = f"{base_url}/v3/async/txt2img"
        task_id = self._submit_task(submit_url, api_key, request)
        result_payload = self._poll_task_result(base_url, api_key, task_id)

        content, content_type = self._extract_image_content(result_payload)
        content, content_type = validate_and_convert_image_output(
            content,
            content_type,
            output_format,
            max_bytes=self._max_output_bytes(),
        )
        return ImageGenResult(content=content, content_type=content_type, bytes_len=len(content))

    def _max_output_bytes(self) -> int:
        return effective_inline_max_bytes(self._config)

    def _resolve_api_key(self) -> str:
        api_key = (self._config.novita_image_api_key or "").strip()
        if not api_key:
            api_key = (os.getenv("NOVITA_API_KEY") or "").strip()
        if not api_key:
            raise ImageBackendUnavailableError("novita image api key is not configured")
        return api_key

    def _resolve_base_url(self) -> str:
        raw = (
            os.getenv("NOVITA_IMAGE_BASE_URL")
            or self._config.novita_image_base_url
            or DEFAULT_NOVITA_IMAGE_BASE_URL
        )
        cleaned = str(raw).strip()
        if not cleaned:
            raise ImageBackendUnavailableError("novita image base URL is not configured")
        if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = f"https://{cleaned}"
        return cleaned.rstrip("/")

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _submit_task(self, submit_url: str, api_key: str, request: ImageGenRequest) -> str:
        payload = self._build_submit_payload(request)
        try:
            data = fetch_json(
                method="POST",
                url=submit_url,
                headers=self._headers(api_key),
                json=payload,
                timeout=self._config.novita_image_timeout_seconds or DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise ImageGenerationError(f"Novita submit request failed: {exc}") from exc

        task_id = self._extract_task_id(data)
        if not task_id:
            raise ImageGenerationError("Novita submit response did not include task id")
        return task_id

    def _poll_task_result(self, base_url: str, api_key: str, task_id: str) -> dict[str, Any]:
        timeout_seconds = float(
            self._config.novita_image_timeout_seconds or DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS
        )
        poll_interval = max(1.0, float(self._config.novita_image_poll_interval_seconds or 2))
        deadline = time.monotonic() + timeout_seconds
        poll_url = f"{base_url}/v3/async/task-result"

        last_payload: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                data = fetch_json(
                    method="GET",
                    url=poll_url,
                    headers=self._headers(api_key),
                    params={"task_id": task_id},
                    timeout=self._config.novita_image_timeout_seconds or DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                raise ImageGenerationError(f"Novita task polling failed: {exc}") from exc

            if isinstance(data, dict):
                last_payload = data
            state = self._extract_state(data)
            if state in self._DONE_STATES:
                if not isinstance(data, dict):
                    raise ImageGenerationError("Novita task-result response was not JSON")
                return data
            if state in self._FAILED_STATES:
                detail = self._extract_error_message(data) or state
                raise ImageGenerationError(f"Novita task failed: {detail}")
            time.sleep(poll_interval)

        detail = self._extract_error_message(last_payload) or "timed out waiting for Novita image task result"
        raise ImageGenerationError(detail)

    def _build_submit_payload(self, request: ImageGenRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_name": (
                request.model
                or os.getenv("NOVITA_IMAGE_MODEL")
                or self._config.novita_image_default_model
                or DEFAULT_NOVITA_IMAGE_MODEL
            ),
            "prompt": request.prompt,
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.width is not None:
            payload["width"] = request.width
        if request.height is not None:
            payload["height"] = request.height
        if request.steps is not None:
            payload["steps"] = request.steps
        if request.cfg_scale is not None:
            payload["guidance_scale"] = request.cfg_scale
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.sampler:
            payload["sampler_name"] = request.sampler

        extra_params = request.extra_params or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if key in {"prompt", "negative_prompt"}:
                    continue
                payload[key] = value
        return payload

    def _extract_task_id(self, data: Any) -> str | None:
        if isinstance(data, dict):
            for key in ("task_id", "taskId", "id"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("data", "result", "output"):
                nested = data.get(key)
                task_id = self._extract_task_id(nested)
                if task_id:
                    return task_id
        return None

    def _extract_state(self, data: Any) -> str:
        if isinstance(data, dict):
            for key in ("status", "task_status", "state", "task_state"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
            for key in ("data", "result", "output"):
                nested = data.get(key)
                nested_state = self._extract_state(nested)
                if nested_state:
                    return nested_state
        return ""

    def _extract_error_message(self, data: Any) -> str | None:
        if isinstance(data, dict):
            for key in ("message", "error", "error_message", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for key in ("data", "result", "output"):
                nested = data.get(key)
                nested_msg = self._extract_error_message(nested)
                if nested_msg:
                    return nested_msg
        return None

    def _extract_image_content(self, data: Any) -> tuple[bytes, str]:
        candidate = self._extract_from_node(data)
        if candidate:
            return candidate
        raise ImageGenerationError("Novita did not return image content")

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
            for key in ("images", "data", "output", "result"):
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
            return fetch_image_bytes(
                raw,
                timeout=self._config.novita_image_timeout_seconds or DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS,
                max_bytes=self._max_output_bytes(),
            )
        decoded = maybe_decode_base64_image(raw, max_bytes=self._max_output_bytes())
        if decoded is not None:
            return decoded, "image/png"
        return None
