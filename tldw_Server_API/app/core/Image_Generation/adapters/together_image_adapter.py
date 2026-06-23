"""Together image-generation backend adapter."""

from __future__ import annotations

import os
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
    DEFAULT_TOGETHER_IMAGE_BASE_URL,
    DEFAULT_TOGETHER_IMAGE_MODEL,
    DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS,
    get_image_generation_config,
)
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_Server_API.app.core.Image_Generation.request_validation import effective_inline_max_bytes


class TogetherImageAdapter:
    name = "together"
    supported_formats = {"png", "jpg", "webp"}

    def __init__(self) -> None:
        self._config = get_image_generation_config()

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        url = self._image_generation_url(base_url)
        payload = self._build_payload(request)

        try:
            data = fetch_json(
                method="POST",
                url=url,
                headers=self._headers(api_key),
                json=payload,
                timeout=self._config.together_image_timeout_seconds or DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise ImageGenerationError(f"Together image request failed: {exc}") from exc

        content, content_type = self._extract_image_content(data)
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
        api_key = (self._config.together_image_api_key or "").strip()
        if not api_key:
            api_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if not api_key:
            raise ImageBackendUnavailableError("together image api key is not configured")
        return api_key

    def _resolve_base_url(self) -> str:
        raw = (
            os.getenv("TOGETHER_BASE_URL")
            or self._config.together_image_base_url
            or DEFAULT_TOGETHER_IMAGE_BASE_URL
        )
        cleaned = str(raw).strip()
        if not cleaned:
            raise ImageBackendUnavailableError("together image base URL is not configured")
        if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = f"https://{cleaned}"
        return cleaned.rstrip("/")

    @staticmethod
    def _image_generation_url(base_url: str) -> str:
        lower = base_url.lower()
        if lower.endswith("/v1"):
            return f"{base_url}/images/generations"
        if lower.endswith("/images/generations"):
            return base_url
        return f"{base_url}/v1/images/generations"

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ImageGenRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": (
                request.model
                or os.getenv("TOGETHER_IMAGE_MODEL")
                or self._config.together_image_default_model
                or DEFAULT_TOGETHER_IMAGE_MODEL
            ),
            "prompt": request.prompt,
            "response_format": "b64_json",
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        if request.width and request.height:
            payload["size"] = f"{request.width}x{request.height}"
            payload["width"] = request.width
            payload["height"] = request.height
        if request.steps is not None:
            payload["steps"] = request.steps
        if request.cfg_scale is not None:
            payload["guidance_scale"] = request.cfg_scale
        if request.seed is not None:
            payload["seed"] = request.seed

        extra_params = request.extra_params or {}
        if isinstance(extra_params, dict):
            for key, value in extra_params.items():
                if key in {"prompt", "negative_prompt"}:
                    continue
                payload[key] = value
        return payload

    def _extract_image_content(self, data: Any) -> tuple[bytes, str]:
        candidate = self._extract_from_node(data)
        if candidate:
            return candidate
        raise ImageGenerationError("Together did not return image content")

    def _extract_from_node(self, node: Any) -> tuple[bytes, str] | None:
        if isinstance(node, dict):
            for key in ("b64_json", "image_base64", "base64", "image_b64"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return decode_base64_image(value.strip(), max_bytes=self._max_output_bytes()), "image/png"
            for key in ("url", "image_url", "image"):
                if key in node:
                    extracted = self._extract_from_link_value(node.get(key))
                    if extracted:
                        return extracted
            for key in ("data", "images", "output", "result"):
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
                timeout=self._config.together_image_timeout_seconds or DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS,
                max_bytes=self._max_output_bytes(),
            )
        decoded = maybe_decode_base64_image(raw, max_bytes=self._max_output_bytes())
        if decoded is not None:
            return decoded, "image/png"
        return None
