"""ACE-Step external HTTP adapter for Audio Studio music generation."""

from __future__ import annotations

import os
from typing import Any, Callable
from urllib.parse import urljoin

import httpx

from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest, AudioGenerationResult
from tldw_Server_API.app.core.Audio_Studio.security import validate_external_audio_endpoint
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int


class AceStepHttpAdapter:
    """Generate music through a configured, allowlisted ACE-Step HTTP service."""

    provider_id = "ace_step"
    supported_kinds = frozenset({"music"})

    def __init__(self, *, client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient) -> None:
        self._client_factory = client_factory

    @classmethod
    def is_configured(cls) -> bool:
        """Return whether ACE-Step has a configured and allowlisted base URL."""

        base_url = (os.getenv("AUDIO_STUDIO_ACE_STEP_BASE_URL") or "").strip()
        if not base_url:
            return False
        validate_external_audio_endpoint(base_url)
        return True

    async def generate(self, request: AudioGenerationRequest, **_: Any) -> AudioGenerationResult:
        """Generate music via the configured ACE-Step HTTP endpoint."""

        if request.kind != "music":
            raise ValueError("unsupported_audio_generation_kind")
        base_url = (os.getenv("AUDIO_STUDIO_ACE_STEP_BASE_URL") or "").strip()
        if not base_url:
            raise ValueError("audio_studio_ace_step_not_configured")
        validate_external_audio_endpoint(base_url)
        endpoint = urljoin(base_url.rstrip("/") + "/", "generate")
        validate_external_audio_endpoint(endpoint)

        api_key = (os.getenv("AUDIO_STUDIO_ACE_STEP_API_KEY") or "").strip()
        timeout = coerce_int(os.getenv("AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS"), 60)
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {
            "prompt": request.prompt,
            "text": request.text,
            "workflow": request.workflow,
            "kind": request.kind,
            "options": dict(request.provider_options or {}),
        }

        async with self._client_factory(timeout=timeout, follow_redirects=False) as client:
            response = await self._post_with_validated_redirects(client, endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return AudioGenerationResult(
            mime_type=response.headers.get("content-type", "audio/wav").split(";", 1)[0],
            content_bytes=response.content,
            provider=self.provider_id,
            metadata={"status_code": response.status_code},
        )

    async def _post_with_validated_redirects(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        current_url = url
        for _ in range(4):
            response = await client.post(current_url, headers=headers, json=json)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            if not location:
                return response
            next_url = str(httpx.URL(current_url).join(location))
            current_origin = validate_external_audio_endpoint(current_url)
            next_origin = validate_external_audio_endpoint(next_url, redirect_from=current_url)
            if headers.get("authorization") and next_origin != current_origin:
                raise ValueError("external_audio_redirect_cross_origin_with_auth")
            current_url = next_url
        raise ValueError("external_audio_redirect_limit_exceeded")
