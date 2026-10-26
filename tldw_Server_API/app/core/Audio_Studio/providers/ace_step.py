"""ACE-Step external HTTP adapter for Audio Studio music generation.

Every request goes through ``core.http_client.afetch`` with no redirects and no retries. Its
origin has passed ``validate_external_audio_endpoint`` (the operator's
AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST), so the request is scoped to exactly that origin as a
configured endpoint: a local ACE-Step server on a private address and any port works, while a
cloud-metadata or link-local DNS answer is refused and the resolved address is pinned.
"""

from __future__ import annotations

import asyncio
import json as json_module
import os
from typing import Any, Callable
from urllib.parse import urljoin

import httpx

from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest, AudioGenerationResult
from tldw_Server_API.app.core.Audio_Studio.security import validate_external_audio_endpoint
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch, create_async_client
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


class AceStepGenerationError(RuntimeError):
    """Do not automatically duplicate an external task whose outcome is uncertain."""

    retryable = False


class AceStepHttpAdapter:
    """Generate music through a configured, allowlisted ACE-Step HTTP service."""

    provider_id = "ace_step"
    supported_kinds = frozenset({"music"})

    def __init__(self, *, client_factory: Callable[..., httpx.AsyncClient] | None = None) -> None:
        self._client_factory = client_factory or create_async_client

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
        version = os.getenv("AUDIO_STUDIO_ACE_STEP_API_VERSION", "legacy").strip()
        if version == "1.5":
            return await self._generate_v15(request, base_url)
        if version != "legacy":
            raise AceStepGenerationError("audio_studio_ace_step_api_version_invalid")
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

        async with self._client_factory(timeout=timeout) as client:
            response = await self._post_with_validated_redirects(client, endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return AudioGenerationResult(
            mime_type=response.headers.get("content-type", "audio/wav").split(";", 1)[0],
            content_bytes=response.content,
            provider=self.provider_id,
            metadata={"status_code": response.status_code},
        )

    async def _generate_v15(self, request: AudioGenerationRequest, base_url: str) -> AudioGenerationResult:
        """Submit once, poll that task, and fetch its same-origin audio within one deadline."""

        timeout = max(1, coerce_int(os.getenv("AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS"), 60))
        api_key = (os.getenv("AUDIO_STUDIO_ACE_STEP_API_KEY") or "").strip()
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        base = base_url.rstrip("/") + "/"
        options = request.provider_options or {}
        prompt = ", ".join(str(value).strip() for value in (request.prompt, options.get("style")) if value)
        payload = {
            "prompt": prompt,
            "lyrics": options.get("lyrics", request.text or ""),
            "audio_duration": options.get("duration", 45),
            "audio_format": "wav",
            "batch_size": 1,
            "thinking": False,
        }
        try:
            async with (
                asyncio.timeout(timeout),
                self._client_factory(timeout=timeout) as client,
            ):
                response = await self._send(client, "POST", urljoin(base, "release_task"), headers, payload, base)
                response.raise_for_status()
                submitted = response.json()
                if submitted.get("code") != 200 or submitted.get("error"):
                    raise ValueError("submission_rejected")
                task_id = submitted["data"]["task_id"]
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("invalid_task_id")
                while True:
                    response = await self._send(
                        client, "POST", urljoin(base, "query_result"), headers, {"task_id_list": [task_id]}, base
                    )
                    response.raise_for_status()
                    queried = response.json()
                    if queried.get("code") != 200 or queried.get("error"):
                        raise ValueError("query_rejected")
                    rows = queried["data"]
                    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("task_id") != task_id:
                        raise ValueError("task_identity_mismatch")
                    status = rows[0]["status"]
                    if status == 1:
                        outputs = json_module.loads(rows[0]["result"])
                        if not isinstance(outputs, list) or len(outputs) != 1:
                            raise ValueError("invalid_outputs")
                        audio_url = str(httpx.URL(urljoin(base, outputs[0]["file"])))
                        parsed = httpx.URL(audio_url)
                        if (
                            validate_external_audio_endpoint(audio_url) != validate_external_audio_endpoint(base)
                            or parsed.path != "/v1/audio"
                            or parsed.username
                            or parsed.password
                        ):
                            raise ValueError("invalid_audio_origin")
                        audio = await self._send(client, "GET", audio_url, headers, None, base)
                        audio.raise_for_status()
                        mime = audio.headers.get("content-type", "").split(";", 1)[0]
                        if not mime.startswith("audio/") or not audio.content:
                            raise ValueError("invalid_audio_content")
                        return AudioGenerationResult(mime, audio.content, self.provider_id, {"task_id": task_id})
                    if status != 0:
                        raise ValueError("generation_failed")
                    await asyncio.sleep(0.5)
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            NetworkError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            TimeoutError,
        ) as exc:
            # No upstream idempotency key: an automatic retry could create a second music task.
            reason = "timeout" if isinstance(exc, TimeoutError) else "failed"
            raise AceStepGenerationError(f"ace_step_v15_generation_{reason}") from exc

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
            response = await self._send(client, "POST", current_url, headers, json, current_url)
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

    @staticmethod
    async def _send(
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any] | None,
        allowlisted_url: str,
    ) -> httpx.Response:
        """Send one request scoped to ``allowlisted_url``'s origin, already allowlisted by the caller."""
        try:
            return await afetch(
                method=method,
                url=url,
                client=client,
                headers=headers,
                json=json,
                timeout=client.timeout,  # afetch's default (None) would disable the client's timeout
                allow_redirects=False,
                retry=RetryPolicy(attempts=1),
                configured_endpoint=ConfiguredEndpointScope.from_url(allowlisted_url),
                sensitive_observability=True,
            )
        except EgressPolicyError:
            raise AceStepGenerationError("ace_step_endpoint_refused_by_egress_policy") from None
