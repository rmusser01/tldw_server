from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config
from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    EMBEDDING_REDIRECT_STATUS_CODES,
    resolve_runtime_embedding_base_url,
)
from tldw_Server_API.app.core.testing import is_truthy

from .base import EmbeddingsProvider


class OpenAIEmbeddingsAdapter(EmbeddingsProvider):
    name = "openai-embeddings"

    def capabilities(self) -> dict[str, Any]:
        return {
            "dimensions_default": None,
            "max_batch_size": 2048,
            "default_timeout_seconds": 60,
        }

    def _use_native_http(self) -> bool:
        import os
        v = os.getenv("LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI")
        # Default to False to preserve current behavior; can be flipped in CI later
        return is_truthy(v)

    def _base_url(self, openai_cfg: dict[str, Any] | None = None) -> str:
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import _resolve_openai_api_base
        return _resolve_openai_api_base(openai_cfg or {})

    def _headers(
        self,
        api_key: str | None,
        app_config: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        from tldw_Server_API.app.core.LLM_Calls.openai_credentials import (
            openai_credential_headers,
        )

        return openai_credential_headers(api_key, app_config, provider=self.name)

    def _normalize_response(self, raw: dict[str, Any], *, multi: bool) -> dict[str, Any]:
        # Pass-through OpenAI shape if present; otherwise synthesize a basic structure
        if isinstance(raw, dict) and "data" in raw:
            return raw
        if not multi:
            vec = raw if isinstance(raw, list) else []
            return {"data": [{"index": 0, "embedding": vec}], "model": None, "object": "list"}
        # multi
        if isinstance(raw, list):
            data = [{"index": i, "embedding": e} for i, e in enumerate(raw)]
            return {"data": data, "model": None, "object": "list"}
        return {"data": [], "model": None, "object": "list"}

    def embed(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        inputs = request.get("input")
        model = request.get("model")
        dimensions = request.get("dimensions")
        if inputs is None:
            raise ValueError("Embeddings: 'input' is required")

        raw_config = request.get("app_config")
        app_config = raw_config if isinstance(raw_config, dict) else None
        if request.get("credentials_resolved") is True:
            app_config = app_config or {}
        else:
            app_config = ensure_app_config(app_config)
        openai_cfg = dict((app_config.get("openai_api") or {}) if app_config else {})
        runtime_base_url = resolve_runtime_embedding_base_url(
            request,
            provider=self.name,
        )
        if runtime_base_url is not None:
            openai_cfg["api_base_url"] = runtime_base_url
            app_config = dict(app_config or {})
            app_config["openai_api"] = openai_cfg
        api_key = request.get("api_key") or openai_cfg.get("api_key")
        if not api_key and app_config:
            try:
                emb_cfg = app_config.get("embedding_config") or {}
                models = emb_cfg.get("models") or {}
                model_spec = models.get(model)
                if model_spec is not None:
                    api_key = getattr(model_spec, "api_key", None) or (
                        model_spec.get("api_key") if isinstance(model_spec, dict) else None
                    )
            except Exception:
                api_key = None
        if api_key:
            openai_cfg["api_key"] = api_key
            app_config = dict(app_config or {})
            app_config["openai_api"] = openai_cfg

        credentials_resolved = request.get("credentials_resolved") is True

        # A server-resolved endpoint must not fall through to the redirecting legacy path.
        if credentials_resolved or self._use_native_http():
            from tldw_Server_API.app.core.http_client import fetch as _fetch
            base_url = runtime_base_url or self._base_url(openai_cfg).rstrip("/")
            url = f"{base_url}/embeddings"
            payload = {"input": inputs, "model": model}
            if dimensions is not None:
                try:
                    dim = int(dimensions)
                except Exception:
                    dim = None
                if dim and dim > 0:
                    payload["dimensions"] = dim
            headers = self._headers(api_key, app_config)
            provider_error: Exception | None = None
            try:
                resp = _fetch(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=timeout or 60.0,
                    allow_redirects=not credentials_resolved,
                )
                if getattr(resp, "status_code", None) in EMBEDDING_REDIRECT_STATUS_CODES:
                    raise RuntimeError("Embedding provider redirected the request")
                if resp.status_code >= 400:
                    resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                from tldw_Server_API.app.core.Chat.Chat_Deps import (
                    ChatAuthenticationError,
                    ChatProviderError,
                )
                from tldw_Server_API.app.core.LLM_Calls.error_utils import (
                    get_http_status_from_exception,
                )

                upstream_status = get_http_status_from_exception(exc)
                if upstream_status in {401, 403}:
                    provider_error = ChatAuthenticationError(
                        provider=self.name,
                        message="Embedding provider authentication failed.",
                        status_code=upstream_status,
                    )
                else:
                    provider_error = ChatProviderError(
                        provider=self.name,
                        message="Embedding provider request failed.",
                    )
            if provider_error is not None:
                raise provider_error

        # Delegate-first fallback using legacy helper(s)
        from tldw_Server_API.app.core.LLM_Calls import chat_calls as legacy
        legacy_error: Exception | None = None
        raw_result: Any = None
        multi = isinstance(inputs, list)
        try:
            if multi:
                raw_result = legacy.get_openai_embeddings_batch(
                    inputs,
                    model,
                    app_config=app_config,
                    dimensions=dimensions,
                )
            else:
                raw_result = legacy.get_openai_embeddings(
                    inputs,
                    model,
                    app_config=app_config,
                    dimensions=dimensions,
                )
        except Exception as exc:
            from tldw_Server_API.app.core.Chat.Chat_Deps import (
                ChatAuthenticationError,
                ChatProviderError,
            )
            from tldw_Server_API.app.core.LLM_Calls.error_utils import (
                get_http_status_from_exception,
            )

            upstream_status = get_http_status_from_exception(exc)
            if upstream_status in {401, 403}:
                legacy_error = ChatAuthenticationError(
                    provider=self.name,
                    message="Embedding provider authentication failed.",
                    status_code=upstream_status,
                )
            else:
                legacy_error = ChatProviderError(
                    provider=self.name,
                    message="Embedding provider request failed.",
                    status_code=(
                        upstream_status
                        if isinstance(upstream_status, int) and upstream_status >= 400
                        else 502
                    ),
                )
        if legacy_error is not None:
            raise legacy_error
        return self._normalize_response(raw_result, multi=multi)
