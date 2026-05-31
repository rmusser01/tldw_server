from __future__ import annotations

import os
from typing import Any, Optional

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.Utils.Utils import logging

_PROVIDER_SECTION_MAP: dict[str, str] = {
    "openai": "openai_api",
    "anthropic": "anthropic_api",
    "cohere": "cohere_api",
    "groq": "groq_api",
    "deepseek": "deepseek_api",
    "mistral": "mistral_api",
    "openrouter": "openrouter_api",
    "novita": "novita_api",
    "poe": "poe_api",
    "together": "together_api",
    "huggingface": "huggingface_api",
    "google": "google_api",
    "qwen": "qwen_api",
    "custom-openai-api": "custom_openai_api",
    "custom-openai-api-2": "custom_openai_api_2",
    "moonshot": "moonshot_api",
    "zai": "zai_api",
    "llama.cpp": "llama_api",
    "kobold": "kobold_api",
    "ooba": "ooba_api",
    "tabbyapi": "tabby_api",
    "vllm": "vllm_api",
    "local-llm": "local_llm",
    "ollama": "ollama_api",
    "aphrodite": "aphrodite_api",
    "bedrock": "bedrock_api",
    "mlx": "mlx",
}


def normalize_provider(provider: str | None) -> str:
    return (provider or "").strip().lower()


def resolve_provider_section(provider: str) -> str:
    normalized = normalize_provider(provider)
    if not normalized:
        return ""
    custom_number = custom_openai_provider_number(normalized)
    if custom_number is not None:
        return custom_openai_section_name(custom_number)
    return _PROVIDER_SECTION_MAP.get(
        normalized,
        f"{normalized.replace('.', '_').replace('-', '_')}_api",
    )


def ensure_app_config(app_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if app_config is not None:
        return app_config
    # Prefer the chat_calls loader when tests monkeypatch it.
    try:
        from tldw_Server_API.app.core.LLM_Calls import chat_calls as _chat_calls

        loader = getattr(_chat_calls, "load_and_log_configs", None)
        if callable(loader):
            cfg = loader()
            if cfg is not None:
                return cfg
    except Exception as load_error:
        _ = load_error  # fallback to standard config loader below
    return load_and_log_configs() or {}


def resolve_provider_model(provider: str, app_config: dict[str, Any]) -> str | None:
    normalized = normalize_provider(provider)
    section = resolve_provider_section(provider)
    if section:
        cfg = app_config.get(section) or {}
        model = cfg.get("model") or cfg.get("model_id")
        if model:
            return model
    if normalized == "mlx":
        env_val = (__import__("os").getenv("MLX_MODEL_PATH") or "").strip()
        if env_val:
            return env_val
        mlx_cfg = app_config.get("mlx") or {}
        model = mlx_cfg.get("model_path") or mlx_cfg.get("mlx_model_path") or mlx_cfg.get("model")
        if model:
            return model
    env_key = f"DEFAULT_MODEL_{normalized.replace('.', '_').replace('-', '_').upper()}"
    env_val = (env_key and __import__("os").getenv(env_key))  # avoid top-level os import
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()
    return None


def resolve_provider_api_key_from_config(
    provider: str,
    app_config: dict[str, Any] | None = None,
    *,
    prefer_module_keys_in_tests: bool = True,
) -> str | None:
    api_key, _debug = resolve_provider_api_key(
        provider,
        prefer_module_keys_in_tests=prefer_module_keys_in_tests,
    )
    if api_key:
        return api_key
    cfg = ensure_app_config(app_config)
    section = resolve_provider_section(provider)
    if section:
        api_key = (cfg.get(section) or {}).get("api_key")
    return api_key


def get_adapter_or_raise(provider: str):
    adapter = get_registry().get_adapter(normalize_provider(provider))
    if adapter is None:
        raise ChatConfigurationError(provider=provider, message="LLM adapter unavailable.")
    return adapter


def split_system_message(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_message = None
    remaining: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system" and system_message is None:
            system_message = msg.get("content")
            continue
        remaining.append(msg)
    return system_message, remaining


# ---------------------------------------------------------------------------
# Shared utilities relocated from chat_calls.py (Feb 2026)
# ---------------------------------------------------------------------------

def _safe_cast(value: Any, cast_to: type, default: Any = None) -> Any:
    """Safely casts value to specified type, returning default on failure."""
    if value is None:
        return default
    try:
        return cast_to(value)
    except (ValueError, TypeError):
        logging.warning(f"Could not cast '{value}' to {cast_to}. Using default: {default}")
        return default


def _resolve_openai_api_base(openai_cfg: dict[str, Any]) -> str:
    """Resolve the OpenAI API base URL.

    Precedence: config keys (api_base_url, api_base, base_url),
    then environment vars (OPENAI_API_BASE_URL, OPENAI_API_BASE, OPENAI_BASE_URL, MOCK_OPENAI_BASE_URL),
    then default 'https://api.openai.com/v1'.
    """
    try:
        cfg_base = (
            openai_cfg.get('api_base_url')
            or openai_cfg.get('api_base')
            or openai_cfg.get('base_url')
        )
    except Exception:
        cfg_base = None

    env_api_base = (
        os.getenv('OPENAI_API_BASE_URL')
        or os.getenv('OPENAI_API_BASE')
        or os.getenv('OPENAI_BASE_URL')
        or os.getenv('MOCK_OPENAI_BASE_URL')
    )
    return (cfg_base or env_api_base or 'https://api.openai.com/v1')


def _parse_data_url_for_multimodal(data_url: str) -> Optional[tuple[str, str]]:
    """Parses a data URL (e.g., data:image/png;base64,xxxx) into (mime_type, base64_data)."""
    if data_url.startswith("data:") and ";base64," in data_url:
        try:
            header, b64_data = data_url.split(";base64,", 1)
            mime_type = header.split("data:", 1)[1]
            return mime_type, b64_data
        except Exception as e:
            logging.warning(f"Could not parse data URL: {data_url[:60]}... Error: {e}")
            return None
    logging.debug(f"Data URL did not match expected format: {data_url[:60]}...")
    return None
