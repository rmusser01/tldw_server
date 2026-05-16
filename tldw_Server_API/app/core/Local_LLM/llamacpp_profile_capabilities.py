"""Capability and launch resolution helpers for managed llama.cpp profiles."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppAsset
from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import (
    ModelNotFoundError,
    ServerError,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppProfile,
    LlamaCppProfileMode,
)


class LlamaCppResolvedProfileLaunch(BaseModel):
    """Resolved launch contract for one managed llama.cpp profile."""

    profile: LlamaCppProfile
    model_path: Path
    mmproj_path: Path | None = None
    server_args: dict[str, object]
    capabilities: dict[str, bool]
    modalities: dict[str, list[str]]
    warnings: list[str] = Field(default_factory=list)


def resolve_profile_launch(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None = None,
) -> LlamaCppResolvedProfileLaunch:
    """Resolve a profile's model assets, launch args, and mode capabilities."""
    model_path = _resolve_base_model_path(profile, assets)
    server_args = dict(profile.server_args)
    mmproj_path = _resolve_mmproj_path(profile, server_args, assets)

    if profile.mode == LlamaCppProfileMode.VISION and mmproj_path is None:
        raise ServerError("Vision llama.cpp profiles require a valid mmproj asset.")
    if mmproj_path is not None:
        server_args["mmproj"] = str(mmproj_path)

    return LlamaCppResolvedProfileLaunch(
        profile=profile,
        model_path=model_path,
        mmproj_path=mmproj_path,
        server_args=server_args,
        capabilities=_capabilities_for_mode(profile.mode, mmproj_path=mmproj_path),
        modalities=_modalities_for_mode(profile.mode, mmproj_path=mmproj_path),
        warnings=[],
    )


def profile_capability_metadata(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None = None,
) -> dict[str, object]:
    """Return bounded public capability metadata for a managed profile."""
    try:
        resolved = resolve_profile_launch(profile, assets=assets)
    except (ModelNotFoundError, ServerError) as exc:
        return {
            "capabilities": _capabilities_for_mode(profile.mode, mmproj_path=None, validated=False),
            "modalities": _modalities_for_mode(profile.mode, mmproj_path=None, validated=False),
            "capability_warnings": [str(exc)],
        }
    return {
        "capabilities": resolved.capabilities,
        "modalities": resolved.modalities,
        "capability_warnings": list(resolved.warnings),
    }


def _resolve_base_model_path(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None,
) -> Path:
    if profile.model_id:
        asset = llamacpp_inventory_service.resolve_asset_id(
            profile.model_id,
            expected_kind="gguf",
            assets=assets,
        )
        if not asset.resolved_path:
            raise ModelNotFoundError(
                f"Model asset {profile.model_id} does not reference an "
                "available local asset."
            )
        return Path(asset.resolved_path)
    if profile.model_path:
        return llamacpp_inventory_service.resolve_asset_path(
            profile.model_path,
            expected_kind="gguf",
            label="Model",
        )
    raise ModelNotFoundError("Llama.cpp profile requires a model_id or model_path.")


def _resolve_mmproj_path(
    profile: LlamaCppProfile,
    server_args: dict[str, object],
    assets: list[LlamaCppAsset] | None,
) -> Path | None:
    asset_path: Path | None = None
    if profile.mmproj_model_id:
        asset = llamacpp_inventory_service.resolve_asset_id(
            profile.mmproj_model_id,
            expected_kind="mmproj",
            assets=assets,
        )
        if not asset.resolved_path:
            raise ModelNotFoundError(
                f"mmproj asset {profile.mmproj_model_id} does not reference an "
                "available local asset."
            )
        asset_path = Path(asset.resolved_path)

    manual_value = server_args.get("mmproj")
    if manual_value in (None, ""):
        return asset_path

    manual_path = llamacpp_inventory_service.resolve_asset_path(
        str(manual_value),
        expected_kind="mmproj",
        label="mmproj",
    )
    if asset_path is not None and manual_path != asset_path:
        raise ServerError("Profile mmproj_model_id conflicts with server_args['mmproj'].")
    return asset_path or manual_path


def _capabilities_for_mode(
    mode: LlamaCppProfileMode,
    *,
    mmproj_path: Path | None,
    validated: bool = True,
) -> dict[str, bool]:
    return {
        "chat": validated and mode in {LlamaCppProfileMode.CHAT, LlamaCppProfileMode.VISION},
        "vision": validated and mode == LlamaCppProfileMode.VISION and mmproj_path is not None,
        "embeddings": validated and mode == LlamaCppProfileMode.EMBEDDING,
        "rerank": validated and mode == LlamaCppProfileMode.RERANK,
    }


def _modalities_for_mode(
    mode: LlamaCppProfileMode,
    *,
    mmproj_path: Path | None,
    validated: bool = True,
) -> dict[str, list[str]]:
    if validated and mode == LlamaCppProfileMode.VISION and mmproj_path is not None:
        return {"input": ["text", "image"], "output": ["text"]}
    if validated and mode == LlamaCppProfileMode.EMBEDDING:
        return {"input": ["text"], "output": ["embedding"]}
    if validated and mode == LlamaCppProfileMode.RERANK:
        return {"input": ["text"], "output": ["score"]}
    return {"input": ["text"], "output": ["text"]}


__all__ = [
    "LlamaCppResolvedProfileLaunch",
    "profile_capability_metadata",
    "resolve_profile_launch",
]
