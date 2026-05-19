# mlx.py
# MLX provider lifecycle endpoints (admin-only)
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole, check_rate_limit
from tldw_Server_API.app.api.v1.schemas.mlx import MLXLoadRequest, MLXUnloadRequest
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatProviderError
from tldw_Server_API.app.core.LLM_Calls.providers.mlx_provider import (
    MLXSessionRegistry,
    _default_settings,
    get_mlx_registry,
)

router = APIRouter()


def _resolve_mlx_registry() -> MLXSessionRegistry:
    """Resolve the shared MLX session registry instance."""
    return get_mlx_registry()


def _normalize_mlx_response(payload: Any) -> dict[str, Any]:
    """Normalize provider payloads into a dict that always includes backend metadata."""
    if isinstance(payload, dict):
        normalized = dict(payload)
    else:
        normalized = {"message": payload}
    normalized.setdefault("backend", "mlx")
    return normalized


def _normalize_model_path(value: Any) -> str | None:
    """Normalize optional model path strings by trimming and coercing empties to ``None``."""
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return value if isinstance(value, str) else None


@router.post(
    "/llm/providers/mlx/load",
    summary="Load or swap the active MLX model",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def load_mlx_model(
    payload: MLXLoadRequest = Body(default_factory=MLXLoadRequest),
    registry: MLXSessionRegistry = Depends(_resolve_mlx_registry),
) -> dict[str, Any]:
    """Load or swap the active MLX model and return normalized provider status."""
    overrides = payload.model_dump(exclude_none=True)
    model_path = _normalize_model_path(overrides.pop("model_path", None))
    if model_path is None:
        model_path = _normalize_model_path(_default_settings().get("model_path"))
    try:
        status = registry.load(model_path=model_path, overrides=overrides)
        return _normalize_mlx_response(status)
    except ChatBadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ChatProviderError as e:
        raise HTTPException(status_code=500, detail="Failed to load MLX model") from e
    except Exception as e:
        logger.error("Unexpected MLX load failure")
        raise HTTPException(status_code=500, detail="MLX load failed unexpectedly") from e


@router.post(
    "/llm/providers/mlx/unload",
    summary="Unload the active MLX model",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def unload_mlx_model(
    payload: MLXUnloadRequest = Body(default_factory=MLXUnloadRequest),
    registry: MLXSessionRegistry = Depends(_resolve_mlx_registry),
) -> dict[str, Any]:
    """Unload the currently active MLX model and return normalized provider status."""
    try:
        _ = payload
        return _normalize_mlx_response(registry.unload())
    except Exception as e:
        logger.error("Unexpected MLX unload failure")
        raise HTTPException(status_code=500, detail="MLX unload failed unexpectedly") from e


@router.get(
    "/llm/providers/mlx/status",
    summary="Get MLX provider status",
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_mlx_status(
    registry: MLXSessionRegistry = Depends(_resolve_mlx_registry),
) -> dict[str, Any]:
    """Return normalized runtime status for the MLX provider registry."""
    try:
        return _normalize_mlx_response(registry.status())
    except Exception as e:
        logger.error("Unexpected MLX status failure")
        raise HTTPException(status_code=500, detail="Failed to get MLX status") from e
