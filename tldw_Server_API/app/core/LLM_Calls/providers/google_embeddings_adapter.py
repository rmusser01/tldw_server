from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from tldw_Server_API.app.core.http_client import create_client
from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    EMBEDDING_REDIRECT_STATUS_CODES,
    encode_google_model_path,
    resolve_runtime_embedding_base_url,
)
from tldw_Server_API.app.core.testing import is_truthy

from .base import EmbeddingsAdapterUnavailableError, EmbeddingsProvider


class GoogleEmbeddingsAdapter(EmbeddingsProvider):
    name = "google-embeddings"

    def capabilities(self) -> dict[str, Any]:
        return {
            "dimensions_default": None,
            "max_batch_size": 128,
            "default_timeout_seconds": 60,
        }

    def _use_native_http(self) -> bool:
        import os
        v = os.getenv("LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE")
        return is_truthy(v)

    def _base_url(self) -> str:
        return os.getenv("GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1").rstrip("/")

    @staticmethod
    def _allow_query_key_fallback(base_url: str, api_key: object) -> bool:
        """Return whether a custom Google-compatible endpoint may retry query auth."""
        if not (isinstance(api_key, str) and api_key.strip()):
            return False
        if not is_truthy(os.getenv("GOOGLE_EMBEDDINGS_QUERY_KEY_FALLBACK", "")):
            return False
        try:
            hostname = (urlsplit(base_url).hostname or "").rstrip(".").casefold()
            return bool(hostname) and hostname != "generativelanguage.googleapis.com"
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _post_embedding(
        client: Any,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        api_key: object,
        allow_query_key_fallback: bool,
    ) -> Any:
        """Post once with header auth and optionally retry a custom endpoint on 401/403."""
        response = client.post(url, headers=headers, json=payload)
        if (
            allow_query_key_fallback
            and getattr(response, "status_code", None) in {401, 403}
        ):
            response = client.post(
                url,
                params={"key": str(api_key)},
                headers={"Content-Type": "application/json"},
                json=payload,
            )
        return response

    def _normalize(self, raw: dict[str, Any], *, multi: bool) -> dict[str, Any]:
        # Google embedContent returns {embedding: {values: [...]}}
        # batchEmbedContents returns {embeddings: [{values: [...]}, ...]}
        if not multi:
            vec = []
            try:
                vec = raw.get("embedding", {}).get("values", [])
            except Exception:
                vec = []
            return {"data": [{"index": 0, "embedding": vec}], "object": "list", "model": None}
        data: list[dict[str, Any]] = []
        try:
            items = raw.get("embeddings", [])
            for i, it in enumerate(items):
                data.append({"index": i, "embedding": (it.get("values") or [])})
        except Exception as parse_error:
            logger.debug("Google embeddings adapter failed to normalize response payload", exc_info=parse_error)
        return {"data": data, "object": "list", "model": None}

    def embed(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        inputs = request.get("input")
        model = request.get("model")
        api_key = request.get("api_key")
        if inputs is None or not model:
            raise ValueError("Embeddings: 'input' and 'model' are required")

        credentials_resolved = request.get("credentials_resolved") is True
        if credentials_resolved or self._use_native_http():
            # Use single embedContent for 1 input; loop for multiple
            base = resolve_runtime_embedding_base_url(request, provider=self.name) or self._base_url()
            try:
                model_path = encode_google_model_path(model)
            except ValueError:
                from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError

                raise ChatBadRequestError(
                    provider=self.name,
                    message="Invalid provider model identifier.",
                ) from None
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["x-goog-api-key"] = str(api_key)
            allow_query_key_fallback = (
                False
                if credentials_resolved
                else self._allow_query_key_fallback(base, api_key)
            )
            provider_error: Exception | None = None
            try:
                client_options: dict[str, Any] = {"timeout": timeout or 60.0}
                if credentials_resolved:
                    client_options["follow_redirects"] = False
                with create_client(**client_options) as client:
                    if isinstance(inputs, list):
                        if credentials_resolved:
                            model_resource = f"models/{model_path}"
                            url = f"{base}/{model_resource}:batchEmbedContents"
                            payload = {
                                "requests": [
                                    {
                                        "model": model_resource,
                                        "content": {"parts": [{"text": text}]},
                                    }
                                    for text in inputs
                                ]
                            }
                            resp = self._post_embedding(
                                client,
                                url=url,
                                headers=headers,
                                payload=payload,
                                api_key=api_key,
                                allow_query_key_fallback=False,
                            )
                            if (
                                getattr(resp, "status_code", None)
                                in EMBEDDING_REDIRECT_STATUS_CODES
                            ):
                                raise RuntimeError(
                                    "Embedding provider redirected the request"
                                )
                            if hasattr(resp, "raise_for_status"):
                                resp.raise_for_status()
                            normalized = self._normalize(resp.json(), multi=True)
                            normalized["model"] = model
                            return normalized
                        out: list[dict[str, Any]] = []
                        for idx, text in enumerate(inputs):
                            url = f"{base}/models/{model_path}:embedContent"
                            payload = {"content": {"parts": [{"text": text}]}}
                            resp = self._post_embedding(
                                client,
                                url=url,
                                headers=headers,
                                payload=payload,
                                api_key=api_key,
                                allow_query_key_fallback=allow_query_key_fallback,
                            )
                            if getattr(resp, "status_code", None) in EMBEDDING_REDIRECT_STATUS_CODES:
                                raise RuntimeError("Embedding provider redirected the request")
                            if hasattr(resp, "raise_for_status"):
                                resp.raise_for_status()
                            data = resp.json()
                            out.append({"index": idx, "embedding": data.get("embedding", {}).get("values", [])})
                        return {"data": out, "object": "list", "model": model}
                    else:
                        url = f"{base}/models/{model_path}:embedContent"
                        payload = {"content": {"parts": [{"text": inputs}]}}
                        resp = self._post_embedding(
                            client,
                            url=url,
                            headers=headers,
                            payload=payload,
                            api_key=api_key,
                            allow_query_key_fallback=allow_query_key_fallback,
                        )
                        if getattr(resp, "status_code", None) in EMBEDDING_REDIRECT_STATUS_CODES:
                            raise RuntimeError("Embedding provider redirected the request")
                        if hasattr(resp, "raise_for_status"):
                            resp.raise_for_status()
                        data = resp.json()
                        return self._normalize(data, multi=False)
            except Exception:
                from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

                provider_error = ChatProviderError(
                    provider=self.name,
                    message="Embedding provider request failed.",
                )
            if provider_error is not None:
                raise provider_error

        msg = (
            "GoogleEmbeddingsAdapter: native HTTP disabled "
            "(set LLM_EMBEDDINGS_NATIVE_HTTP_GOOGLE=1 to enable)"
        )
        logger.debug(msg)
        raise EmbeddingsAdapterUnavailableError(msg)
