"""
chat_calls
Commercial-provider LLM calling utilities — embeddings and backward-compat re-exports.

All ``chat_with_*`` wrapper functions have been removed (Feb 2026). Use
``chat_service.perform_chat_api_call(provider=...)`` directly instead.

This module retains:
- Legacy OpenAI embeddings functions (``get_openai_embeddings``,
  ``get_openai_embeddings_batch``) used until the embeddings adapter native
  HTTP flags are flipped on.
- ``create_session_with_retries`` / ``_SessionShim`` used by embeddings code.

Notes
- Avoid logging secrets; this module only logs high-level metadata.
- Timeouts are provider-configurable; unsafe provider POSTs are single-attempt.
- Use environment variables to override base URLs for testing/mocking.
"""
#########################################
# Import necessary libraries
from typing import Any, Optional

#
# Import Local libraries
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.deprecations import log_runtime_deprecation

#
# Import 3rd-Party Libraries
from tldw_Server_API.app.core.http_client import RetryPolicy, fetch

# ---------------------------------------------------------------------------
# Utilities from adapter_utils used by the embeddings functions below.
# ---------------------------------------------------------------------------
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    _resolve_openai_api_base,
    _safe_cast,
)
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    get_http_error_text,
    get_http_status_from_exception,
    is_http_status_error,
    is_network_error,
)
from tldw_Server_API.app.core.LLM_Calls.http_helpers import (
    create_session_with_retries as _legacy_create_session_with_retries,
)
from tldw_Server_API.app.core.LLM_Calls.openai_credentials import (
    openai_credential_headers,
)
from tldw_Server_API.app.core.Utils.Utils import logging

# -----------------------------------------------------------------------------
# Session shim for non-streaming POST calls
# - Preserves the public name `create_session_with_retries` so tests can
#   monkeypatch it, while centralizing non-streaming requests via http_client.
# - For streaming (stream=True), falls back to the legacy session facade
#   returned by http_helpers.create_session_with_retries to preserve
#   iter_lines() semantics used in streaming paths.
# -----------------------------------------------------------------------------

class _SessionShim:
    def __init__(
        self,
        *,
        total: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: Optional[list[int]] = None,
        allowed_methods: Optional[list[str]] = None,
    ) -> None:
        # Compatibility arguments are intentionally ignored: this shim only
        # exposes provider POSTs, which cannot be replayed without idempotency.
        _ = total, backoff_factor, status_forcelist, allowed_methods
        self._retry = RetryPolicy(attempts=1)
        self._delegate_session = None

    def post(self, url, *, headers=None, json=None, stream: bool = False, timeout=None, **kwargs):
        if stream:
            log_runtime_deprecation(
                "llm_chat_legacy_session",
                message=(
                    "LLM chat streaming path used legacy requests Session compatibility "
                    "facade for iter_lines behavior."
                ),
            )
            # For streaming, use legacy requests session to preserve iter_lines semantics
            self._delegate_session = _legacy_create_session_with_retries(
                total=1,
            )
            return self._delegate_session.post(url, headers=headers, json=json, stream=True, timeout=timeout)
        # Non-streaming via centralized http client (egress/pinning)
        resp = fetch(
            method="POST",
            url=url,
            headers=headers,
            json=json,
            timeout=timeout,
            retry=self._retry,
        )
        return resp

    def close(self):
        try:
            if self._delegate_session is not None:
                self._delegate_session.close()
        except Exception as close_error:
            _ = close_error  # best-effort close for delegate session


def create_session_with_retries(
    *,
    total: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Optional[list[int]] = None,
    allowed_methods: Optional[list[str]] = None,
):
    """Return a session object.

    Provider POSTs are single-attempt until an idempotency contract exists.
    - Under pytest, return the legacy session facade so tests can patch
      `create_session_with_retries` directly.
    - In production, return a shim that routes non-streaming POSTs through
      the centralized HTTP client (egress policy, TLS pinning) and streaming
      through the legacy session facade for iter_lines semantics.
    """
    import os as _os
    if _os.getenv("PYTEST_CURRENT_TEST"):
        log_runtime_deprecation(
            "llm_chat_legacy_session",
            message=(
                "LLM chat used legacy session compatibility path under pytest runtime."
            ),
        )
        return _legacy_create_session_with_retries(
            total=1,
        )
    return _SessionShim(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )


def _resolve_openai_embeddings_api_key(
    model: Optional[str],
    openai_cfg: Optional[dict[str, Any]],
    app_config: Optional[dict[str, Any]],
) -> Optional[str]:
    api_key = (openai_cfg or {}).get("api_key")
    if api_key:
        return api_key
    if not app_config or not model:
        return None
    try:
        emb_cfg = app_config.get("embedding_config") or {}
        models = emb_cfg.get("models") or {}
        model_spec = models.get(model)
        if model_spec is not None:
            return getattr(model_spec, "api_key", None) or (
                model_spec.get("api_key") if isinstance(model_spec, dict) else None
            )
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# OpenAI Embeddings functions (kept until LLM_EMBEDDINGS_NATIVE_HTTP_OPENAI
# is flipped on and validated)
# ---------------------------------------------------------------------------

def get_openai_embeddings(
    input_data: str,
    model: str,
    app_config: Optional[dict[str, Any]] = None,
    dimensions: Optional[int] = None,
) -> list[float]:
    """
    Get embeddings for a single input text from OpenAI API.
    Args:
        input_data (str): The input text to get embeddings for.
        model (str): The model to use for generating embeddings.
        app_config (Optional[Dict[str, Any]]): Pre-loaded application configuration.
                                               If None, config will be loaded internally.
    Returns:
        List[float]: The embeddings generated by the API.
    """
    api_key = None
    openai_cfg: dict[str, Any] = {}
    if app_config:
        # Preferred: explicit openai_api section
        openai_cfg = (app_config.get('openai_api') or {})
        api_key = _resolve_openai_embeddings_api_key(model, openai_cfg, app_config)
    else:
        loaded_config_data = load_and_log_configs()
        openai_cfg = loaded_config_data.get('openai_api', {})
        api_key = _resolve_openai_embeddings_api_key(model, openai_cfg, loaded_config_data)

    if not api_key:
        logging.error("OpenAI Embeddings (single): API key not found or is empty")
        raise ValueError("OpenAI Embeddings (single): API Key Not Provided/Found or is empty")

    logging.debug("OpenAI Embeddings (single): Using configured API key")
    logging.debug(
        f"OpenAI Embeddings (single): input length={len(str(input_data)) if input_data is not None else 0} chars"
    )
    logging.debug(f"OpenAI Embeddings (single): Using model: {model}")

    headers = openai_credential_headers(api_key, {"openai_api": openai_cfg})
    request_data = {
        "input": input_data,
        "model": model,
    }
    if dimensions is not None:
        try:
            dim = int(dimensions)
        except Exception:
            dim = None
        if dim and dim > 0:
            request_data["dimensions"] = dim
    # Resolve OpenAI API base URL using shared helper
    api_base = _resolve_openai_api_base(openai_cfg)
    api_url = api_base.rstrip('/') + '/embeddings'
    try:
        logging.debug(f"OpenAI Embeddings (single): Posting request to embeddings API at {api_url}")
        session = create_session_with_retries(total=1)
        timeout = _safe_cast(openai_cfg.get('api_timeout'), float, 90.0)
        try:
            response = session.post(api_url, headers=headers, json=request_data, timeout=timeout)
            logging.debug(f"OpenAI Embeddings (single): API response status: {response.status_code}")

            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            if 'data' in response_data and len(response_data['data']) > 0 and 'embedding' in response_data['data'][0]:
                embedding = response_data['data'][0]['embedding']
                logging.debug("OpenAI Embeddings (single): Embedding retrieved successfully")
                return embedding
            else:
                logging.warning(
                    f"OpenAI Embeddings (single): Embedding data not found or malformed in response: {response_data}")
                raise ValueError("OpenAI Embeddings (single): Embedding data not available or malformed in the response")
        finally:
            session.close()
    except Exception as e:
        if is_http_status_error(e):
            logging.error(
                "OpenAI Embeddings (single): HTTP request failed with status %s, Response: %s",
                get_http_status_from_exception(e),
                get_http_error_text(e),
                exc_info=True,
            )
            raise
        if is_network_error(e):
            logging.error(f"OpenAI Embeddings (single): Error making API request: {str(e)}", exc_info=True)
            raise ValueError(
                f"OpenAI Embeddings (single): Error making API request: {str(e)}"
            ) from e
        logging.error(f"OpenAI Embeddings (single): Unexpected error: {str(e)}", exc_info=True)
        raise ValueError(f"OpenAI Embeddings (single): Unexpected error occurred: {str(e)}") from e


# NEW BATCH FUNCTION
def get_openai_embeddings_batch(
    texts: list[str],
    model: str,
    app_config: Optional[dict[str, Any]] = None,
    dimensions: Optional[int] = None,
) -> list[list[float]]:
    """
    Get embeddings for a batch of input texts from OpenAI API in a single call.
    Args:
        texts (List[str]): The list of input texts to get embeddings for.
        model (str): The model to use for generating embeddings.
        app_config (Optional[Dict[str, Any]]): Pre-loaded application configuration.
                                               If None, config will be loaded internally.
    Returns:
        List[List[float]]: A list of embeddings, corresponding to the input texts.
    """
    if not texts:
        return []

    openai_cfg: dict[str, Any] = {}
    if app_config:
        openai_cfg = app_config.get('openai_api', {}) or {}
        api_key = _resolve_openai_embeddings_api_key(model, openai_cfg, app_config)
    else:
        # Fallback to loading config internally if not provided
        loaded_config_data = load_and_log_configs()
        openai_cfg = loaded_config_data.get('openai_api', {})
        api_key = _resolve_openai_embeddings_api_key(model, openai_cfg, loaded_config_data)

    if not api_key:
        logging.error("OpenAI Embeddings (batch): API key not found or is empty")
        raise ValueError("OpenAI Embeddings (batch): API Key Not Provided/Found or is empty")

    logging.debug(f"OpenAI Embeddings (batch): Processing {len(texts)} texts.")
    logging.debug("OpenAI Embeddings (batch): Using configured API key")
    logging.debug(f"OpenAI Embeddings (batch): Using model: {model}")

    headers = openai_credential_headers(api_key, {"openai_api": openai_cfg})
    # OpenAI API expects a list of strings for the "input" field for batching
    request_data = {
        "input": texts,
        "model": model,
    }
    if dimensions is not None:
        try:
            dim = int(dimensions)
        except Exception:
            dim = None
        if dim and dim > 0:
            request_data["dimensions"] = dim
    # Resolve OpenAI API base URL using shared helper
    api_base = _resolve_openai_api_base(openai_cfg)
    api_url = api_base.rstrip('/') + '/embeddings'
    try:
        logging.debug(f"OpenAI Embeddings (batch): Posting batch request of {len(texts)} items to API: {api_url}")
        session = create_session_with_retries(total=1)
        timeout = _safe_cast(openai_cfg.get('api_timeout'), float, 90.0)
        try:
            response = session.post(api_url, headers=headers, json=request_data, timeout=timeout)
            logging.debug(f"OpenAI Embeddings (batch): API response status: {response.status_code}")

            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()

            if 'data' in response_data and isinstance(response_data['data'], list):
                # Ensure the number of embeddings matches the number of input texts
                if len(response_data['data']) != len(texts):
                    logging.error(
                        f"OpenAI Embeddings (batch): Mismatch in count. Input: {len(texts)}, Output: {len(response_data['data'])}")
                    raise ValueError(
                        "OpenAI Embeddings (batch): API returned a different number of embeddings than texts provided.")

                response_items = response_data['data']
                indexed_items = [
                    item.get('index')
                    for item in response_items
                    if isinstance(item, dict) and 'index' in item
                ]
                if indexed_items:
                    expected_indices = set(range(len(texts)))
                    if (
                        len(indexed_items) != len(response_items)
                        or any(isinstance(index, bool) or not isinstance(index, int) for index in indexed_items)
                        or set(indexed_items) != expected_indices
                    ):
                        raise ValueError(
                            "OpenAI Embeddings (batch): API response contained malformed embedding indices."
                        )
                    response_items = sorted(response_items, key=lambda item: item['index'])

                embeddings_list = []
                for item in response_items:
                    if 'embedding' in item and isinstance(item['embedding'], list):
                        embeddings_list.append(item['embedding'])
                    else:
                        logging.error(f"OpenAI Embeddings (batch): Malformed embedding item in response: {item}")
                        raise ValueError("OpenAI Embeddings (batch): API response contained malformed embedding data.")

                logging.debug(f"OpenAI Embeddings (batch): {len(embeddings_list)} embeddings retrieved successfully.")
                return embeddings_list
            else:
                logging.warning(
                    f"OpenAI Embeddings (batch): 'data' field not found or not a list in response: {response_data}")
                raise ValueError("OpenAI Embeddings (batch): 'data' field not available or malformed in the API response.")
        finally:
            session.close()

    except Exception as e:
        if is_http_status_error(e):
            # Log the detailed error including the response text for better debugging
            error_message = (
                f"OpenAI Embeddings (batch): HTTP request failed with status {get_http_status_from_exception(e)}."
            )
            try:
                resp = getattr(e, "response", None)
                error_body = resp.json() if resp is not None else None
                if isinstance(error_body, dict):
                    error_message += f" Error details: {error_body.get('error', {}).get('message', get_http_error_text(e))}"
                else:
                    error_message += f" Response: {get_http_error_text(e)}"
            except Exception:
                error_message += f" Response: {get_http_error_text(e)}"
            logging.error(error_message, exc_info=True)
            raise
        if is_network_error(e):
            # Propagate request exceptions so upstream retry logic can handle transient failures
            logging.error(f"OpenAI Embeddings (batch): RequestException: {str(e)}", exc_info=True)
            raise
        logging.error(f"OpenAI Embeddings (batch): Unexpected error: {str(e)}", exc_info=True)
        raise ValueError(f"OpenAI Embeddings (batch): Unexpected error occurred: {str(e)}") from e
