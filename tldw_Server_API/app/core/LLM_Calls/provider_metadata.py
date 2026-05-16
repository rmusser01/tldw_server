from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry

PROVIDER_REQUIRES_KEY: dict[str, bool] = {
    "openai": True,
    "bedrock": True,
    "anthropic": True,
    "cohere": True,
    "groq": True,
    "qwen": True,
    "openrouter": True,
    "novita": True,
    "poe": True,
    "together": True,
    "deepseek": True,
    "mistral": True,
    "google": True,
    "huggingface": True,
    "moonshot": True,
    "zai": True,
    "llama.cpp": False,
    "kobold": False,
    "ooba": False,
    "tabbyapi": False,
    "vllm": False,
    "local-llm": False,
    "ollama": False,
    "aphrodite": False,
    "mlx": False,
    "custom-openai-api": True,
    "custom-openai-api-2": True,
}
PROVIDER_REQUIRES_KEY.update(
    {provider_name: True for provider_name in iter_custom_openai_provider_names(start=3)}
)

DEFAULT_BYOK_ALLOWED_FIELDS: set[str] = {"org_id", "project_id"}

BYOK_CREDENTIAL_FIELDS: dict[str, dict[str, set[str]]] = {
    "openai": {"allowed": {"org_id", "project_id"}, "required": set()},
    "openrouter": {"allowed": {"org_id", "project_id"}, "required": set()},
    "novita": {"allowed": {"org_id", "project_id"}, "required": set()},
    "poe": {"allowed": {"org_id", "project_id"}, "required": set()},
    "together": {"allowed": {"org_id", "project_id"}, "required": set()},
    "custom-openai-api": {"allowed": {"org_id", "project_id"}, "required": set()},
    "custom-openai-api-2": {"allowed": {"org_id", "project_id"}, "required": set()},
}
BYOK_CREDENTIAL_FIELDS.update(
    {
        provider_name: {"allowed": {"org_id", "project_id"}, "required": set()}
        for provider_name in iter_custom_openai_provider_names(start=3)
    }
)


def provider_requires_api_key(provider: str) -> bool:
    provider_norm = (provider or "").strip().lower()
    if not provider_norm:
        return True
    return PROVIDER_REQUIRES_KEY.get(provider_norm, True)


def get_byok_credential_policy(provider: str) -> tuple[set[str], set[str]]:
    provider_norm = (provider or "").strip().lower()
    policy = BYOK_CREDENTIAL_FIELDS.get(provider_norm, {})
    allowed = set(policy.get("allowed", DEFAULT_BYOK_ALLOWED_FIELDS))
    required = set(policy.get("required", set()))
    if required and not required.issubset(allowed):
        allowed |= required
    return allowed or set(DEFAULT_BYOK_ALLOWED_FIELDS), required

PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "openai": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 4096,
    },
    "anthropic": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 60,
        "max_output_tokens_default": 8192,
    },
    "google": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": None,
    },
    "mistral": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 60,
        "max_output_tokens_default": 8192,
    },
    "cohere": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 60,
        "max_output_tokens_default": 4096,
    },
    "groq": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 4096,
    },
    "openrouter": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "novita": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "poe": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "together": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "qwen": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "deepseek": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "huggingface": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "bedrock": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 8192,
    },
    "custom-openai-api": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 60,
        "max_output_tokens_default": 4096,
    },
    "custom-openai-api-2": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 60,
        "max_output_tokens_default": 4096,
    },
    "mlx": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": None,
    },
    "llama.cpp": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "kobold": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "ooba": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "tabbyapi": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "vllm": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 8192,
    },
    "local-llm": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "ollama": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "aphrodite": {
        "supports_streaming": True,
        "supports_tools": False,
        "default_timeout_seconds": 120,
        "max_output_tokens_default": 2048,
    },
    "moonshot": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
    "zai": {
        "supports_streaming": True,
        "supports_tools": True,
        "default_timeout_seconds": 90,
        "max_output_tokens_default": 8192,
    },
}
PROVIDER_CAPABILITIES.update(
    {
        provider_name: dict(PROVIDER_CAPABILITIES["custom-openai-api"])
        for provider_name in iter_custom_openai_provider_names(start=3)
    }
)


def list_registered_providers() -> list[str]:
    return get_registry().list_providers()


def get_managed_vllm_provider_metadata(repository: Any | None = None) -> dict[str, Any]:
    """Summarize managed vLLM instances for provider listings."""

    try:
        from tldw_Server_API.app.core.VLLM_Management import (
            derive_effective_capabilities,
            get_default_vllm_instance_repository,
            resolve_vllm_instance_for_request,
        )
    except Exception:
        return {
            "count": 0,
            "default_instance_id": None,
            "default_model": None,
            "default_base_url": None,
            "models": [],
            "capabilities": {},
            "instances": [],
        }

    repo = repository or get_default_vllm_instance_repository()
    records = list(repo.list_instances())
    default_instance_id = repo.get_default_instance_id()
    default_route = None
    if default_instance_id:
        try:
            default_route = resolve_vllm_instance_for_request(
                provider="vllm",
                provider_instance_id=default_instance_id,
                required_capability=None,
                repository=repo,
            )
        except Exception:
            default_route = None

    aggregate_capabilities = {
        "chat": False,
        "embeddings": False,
        "vision": False,
        "audio": False,
        "multimodal": False,
    }
    models: list[str] = []
    instances: list[dict[str, Any]] = []

    ordered_records = sorted(
        records,
        key=lambda record: (
            0 if record.instance_id == default_instance_id else 1,
            str(record.name or "").lower(),
            record.instance_id,
        ),
    )
    for record in ordered_records:
        effective_capabilities = record.effective_capabilities or derive_effective_capabilities(
            declared_capabilities=record.declared_capabilities,
            probed_capabilities=record.probed_capabilities,
        )
        for capability in aggregate_capabilities:
            aggregate_capabilities[capability] = bool(
                aggregate_capabilities[capability] or effective_capabilities.get(capability, False)
            )
        model_name = record.launch_spec.get("served_model_name") or record.launch_spec.get("model")
        if model_name:
            normalized_model = str(model_name).strip()
            if normalized_model and normalized_model not in models:
                models.append(normalized_model)
        instances.append(
            {
                "instance_id": record.instance_id,
                "name": record.name,
                "execution_mode": record.execution_mode,
                "desired_state": record.desired_state,
                "observed_state": record.observed_state,
                "model": str(model_name).strip() if model_name else None,
                "last_known_base_url": record.last_known_base_url,
                "effective_capabilities": effective_capabilities,
            }
        )

    default_model = None
    default_base_url = None
    if default_route is not None:
        default_model = default_route.model
        default_base_url = default_route.base_url
    elif default_instance_id:
        default_record = next(
            (record for record in ordered_records if record.instance_id == default_instance_id),
            None,
        )
        if default_record is not None:
            default_model = (
                default_record.launch_spec.get("served_model_name")
                or default_record.launch_spec.get("model")
            )
            default_base_url = default_record.last_known_base_url

    return {
        "count": len(records),
        "default_instance_id": default_instance_id,
        "default_model": str(default_model).strip() if default_model else None,
        "default_base_url": str(default_base_url).strip() if default_base_url else None,
        "models": models,
        "capabilities": aggregate_capabilities,
        "instances": instances,
    }
