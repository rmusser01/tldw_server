"""Request-scoped routing helpers for managed vLLM instances."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from tldw_Server_API.app.core.DB_Management.VLLM_Management_DB import (
    SqliteVLLMInstanceRepository,
)

from .capabilities import derive_effective_capabilities
from .repository import VLLMInstanceRepository


class ResolvedVLLMRoute(BaseModel):
    """Resolved managed vLLM route for a single request."""

    instance_id: str
    base_url: str
    model: str | None = None
    api_key: str | None = None
    effective_capabilities: dict[str, bool] = Field(default_factory=dict)


def _default_repository_path() -> Path:
    raw = os.getenv("VLLM_INSTANCES_DB_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[4] / "Databases" / "vllm_instances.db"


@lru_cache(maxsize=1)
def get_default_vllm_instance_repository() -> SqliteVLLMInstanceRepository:
    return SqliteVLLMInstanceRepository(_default_repository_path())


def _normalize_base_url(raw: str) -> str:
    base = str(raw or "").strip().rstrip("/")
    if not base:
        raise ValueError("Managed vLLM instance has no reachable base URL")
    parsed = urlparse(base)
    if not parsed.scheme:
        raise ValueError(f"Managed vLLM base URL must include a scheme: {base}")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _derive_base_url_from_record(instance: Any) -> str:
    explicit = (
        instance.routing_policy.get("base_url")
        or instance.launch_spec.get("base_url")
        or instance.transport_config.get("base_url")
    )
    if explicit:
        return _normalize_base_url(str(explicit))

    host = instance.launch_spec.get("host") or instance.transport_config.get("host") or "127.0.0.1"
    port = instance.launch_spec.get("port") or instance.transport_config.get("port") or 8000
    scheme = instance.launch_spec.get("scheme") or instance.transport_config.get("scheme") or "http"
    if isinstance(host, str) and host.startswith(("http://", "https://")):
        parsed = urlparse(host)
        base = host.rstrip("/")
        if parsed.port is None and port:
            base = f"{base}:{port}"
        return _normalize_base_url(base)
    return _normalize_base_url(f"{scheme}://{host}:{port}")


def resolve_vllm_instance_for_request(
    *,
    provider: str | None,
    provider_instance_id: str | None,
    required_capability: str | Iterable[str] | None,
    repository: VLLMInstanceRepository | None = None,
) -> ResolvedVLLMRoute | None:
    """Resolve the managed vLLM instance for a request.

    Returns ``None`` when the request is not targeting ``vllm`` or when no
    managed instance is selected and no managed default exists, allowing the
    legacy single-endpoint configuration path to continue working.
    """

    provider_key = (provider or "").strip().lower()
    if provider_key != "vllm":
        return None

    repo = repository or get_default_vllm_instance_repository()
    selected_instance_id = provider_instance_id or repo.get_default_instance_id()
    if not selected_instance_id:
        return None

    instance = repo.get_instance(selected_instance_id)
    if instance is None:
        if provider_instance_id:
            raise ValueError(f"Managed vLLM instance '{selected_instance_id}' was not found")
        raise ValueError(f"Default managed vLLM instance '{selected_instance_id}' was not found")

    observed_state = str(getattr(instance, "observed_state", "") or "").strip().lower()
    if observed_state != "healthy":
        raise ValueError(
            f"Managed vLLM instance '{instance.instance_id}' is not healthy "
            f"(observed_state='{instance.observed_state}')"
        )

    effective_capabilities = instance.effective_capabilities or derive_effective_capabilities(
        declared_capabilities=instance.declared_capabilities,
        probed_capabilities=instance.probed_capabilities,
    )
    normalized_required_capabilities: tuple[str, ...]
    if required_capability is None:
        normalized_required_capabilities = ()
    elif isinstance(required_capability, str):
        normalized_required_capabilities = (required_capability,)
    else:
        normalized_required_capabilities = tuple(str(item) for item in required_capability if str(item))
    for capability in normalized_required_capabilities:
        if not effective_capabilities.get(capability, False):
            raise ValueError(
                f"Managed vLLM instance '{instance.instance_id}' does not support required capability "
                f"'{capability}'"
            )

    model = instance.launch_spec.get("served_model_name") or instance.launch_spec.get("model")
    api_key = instance.launch_spec.get("api_key")
    return ResolvedVLLMRoute(
        instance_id=instance.instance_id,
        base_url=_derive_base_url_from_record(instance),
        model=str(model) if model is not None else None,
        api_key=str(api_key) if api_key is not None else None,
        effective_capabilities=effective_capabilities,
    )
