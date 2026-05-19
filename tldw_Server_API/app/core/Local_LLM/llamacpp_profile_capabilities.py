"""Capability and launch resolution helpers for managed llama.cpp profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

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
    path_resolver: Callable[[str | Path, str, str], Path] | None = None,
) -> LlamaCppResolvedProfileLaunch:
    """Resolve a profile's model assets, launch args, and mode capabilities."""
    model_path = _resolve_base_model_path(profile, assets, path_resolver)
    server_args = dict(profile.server_args)
    mmproj_path = _resolve_mmproj_path(profile, server_args, assets, path_resolver)

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


def managed_profile_model_metadata(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None = None,
) -> dict[str, object]:
    """Build public model-catalog metadata for one managed llama.cpp profile."""
    alias = profile.provider_alias or f"llamacpp:{profile.profile_id}"
    capabilities = profile_capability_metadata(profile, assets=assets)
    return {
        "provider": "llama.cpp",
        "model": alias,
        "name": profile.name,
        "type": _type_for_mode(profile.mode),
        "llamacpp_profile_id": profile.profile_id,
        "source": "managed_llamacpp_profile",
        "provider_is_configured": profile.enabled,
        "is_configured": profile.enabled,
        "catalog_only": False,
        **capabilities,
    }


def _resolve_base_model_path(
    profile: LlamaCppProfile,
    assets: list[LlamaCppAsset] | None,
    path_resolver: Callable[[str | Path, str, str], Path] | None,
) -> Path:
    """Resolve and validate the profile's base GGUF path.

    The base model can come from an inventory asset ID or a direct profile path.
    Both forms pass through `_resolve_asset_path` so supervisor-provided launch
    validation can enforce the same file, kind, and allowlist checks. Raises
    `ModelNotFoundError` when the profile has no usable base model.
    """
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
        return _resolve_asset_path(asset.resolved_path, "gguf", "Model", path_resolver)
    if profile.model_path:
        return _resolve_asset_path(profile.model_path, "gguf", "Model", path_resolver)
    raise ModelNotFoundError("Llama.cpp profile requires a model_id or model_path.")


def _resolve_mmproj_path(
    profile: LlamaCppProfile,
    server_args: dict[str, object],
    assets: list[LlamaCppAsset] | None,
    path_resolver: Callable[[str | Path, str, str], Path] | None,
) -> Path | None:
    """Resolve and validate the optional mmproj path for a profile launch.

    Projectors can be selected by inventory ID or supplied manually in
    `server_args["mmproj"]`. When both are present they must resolve to the same
    validated path. Raises `ModelNotFoundError` for missing/wrong-kind projector
    assets and `ServerError` for conflicting projector selections.
    """
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
        asset_path = _resolve_asset_path(asset.resolved_path, "mmproj", "mmproj", path_resolver)

    manual_value = server_args.get("mmproj")
    if manual_value in (None, ""):
        return asset_path

    manual_path = _resolve_asset_path(str(manual_value), "mmproj", "mmproj", path_resolver)
    if asset_path is not None and manual_path != asset_path:
        raise ServerError("Profile mmproj_model_id conflicts with server_args['mmproj'].")
    return asset_path or manual_path


def _resolve_asset_path(
    raw_path: str | Path,
    expected_kind: str,
    label: str,
    path_resolver: Callable[[str | Path, str, str], Path] | None,
) -> Path:
    """Validate an asset path through the launch resolver or inventory service.

    The optional resolver lets the supervisor align launch-time validation with
    its chosen source of configuration while plain metadata calls keep using the
    inventory service's saved-config validation. Errors are intentionally
    propagated from the selected resolver so endpoint mappings remain stable.
    """
    if path_resolver is not None:
        return path_resolver(raw_path, expected_kind, label)
    return llamacpp_inventory_service.resolve_asset_path(raw_path, expected_kind=expected_kind, label=label)


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


def _type_for_mode(mode: LlamaCppProfileMode) -> str:
    """Map managed profile modes onto the public model catalog type vocabulary."""
    if mode in {LlamaCppProfileMode.CHAT, LlamaCppProfileMode.VISION}:
        return "chat"
    if mode == LlamaCppProfileMode.EMBEDDING:
        return "embedding"
    if mode == LlamaCppProfileMode.RERANK:
        return "rerank"
    return "other"


__all__ = [
    "LlamaCppResolvedProfileLaunch",
    "managed_profile_model_metadata",
    "profile_capability_metadata",
    "resolve_profile_launch",
]
