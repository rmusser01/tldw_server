"""Capabilities discovery for the backend-owned VN API namespace."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from tldw_Server_API.app.core.VN_Assets.constants import (
    DEFAULT_VN_ASSET_PACK_ITEM_LIMIT,
    DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
)

VN_BASE_PATH = "/api/v1/vn"
VN_SCRIPTED_GENERATION_OUTPUT_SCHEMAS = ["narrative_dialogue", "choice_set", "scene_update"]
VN_SCRIPTED_GENERATION_AUTOMATIC_BATCH_LIMIT = 1
VN_RESOURCE_PATHS = {
    "assets": f"{VN_BASE_PATH}/vn-assets",
    "scripts": f"{VN_BASE_PATH}/vn-scripts",
    "play": f"{VN_BASE_PATH}/vn-play",
    "policy": f"{VN_BASE_PATH}/vn-policy",
    "audio": f"{VN_BASE_PATH}/vn-audio",
}
VN_SCRIPT_AUTHORING_GRAPH_PATHS = (
    f"{VN_RESOURCE_PATHS['scripts']}/scripts/{{script_id}}/draft/graph",
    f"{VN_RESOURCE_PATHS['scripts']}/scripts/{{script_id}}/draft/graph-preview",
    f"{VN_RESOURCE_PATHS['scripts']}/scripts/{{script_id}}/versions/{{version_id}}/graph",
)
VN_SCRIPT_AUTHORING_GRAPH_ROUTES = (
    (VN_SCRIPT_AUTHORING_GRAPH_PATHS[0], "GET"),
    (VN_SCRIPT_AUTHORING_GRAPH_PATHS[1], "POST"),
    (VN_SCRIPT_AUTHORING_GRAPH_PATHS[2], "GET"),
)
VN_SCRIPT_PLAYTEST_ROUTES = (
    (f"{VN_RESOURCE_PATHS['scripts']}/scripts/{{script_id}}/draft/playtest", "POST"),
    (f"{VN_RESOURCE_PATHS['scripts']}/scripts/{{script_id}}/versions/{{version_id}}/playtest", "POST"),
)
VN_SUPPORTED_IMAGE_MEDIA_TYPES = ["image/png", "image/jpeg", "image/webp"]
VN_SUPPORTED_AUDIO_MEDIA_TYPES = ["audio/mpeg", "audio/wav", "audio/ogg"]


def build_vn_capabilities(routes: Iterable[Any]) -> dict[str, Any]:
    """Build a route-aware capabilities payload for the current FastAPI app."""
    route_paths = _route_paths(routes)
    route_methods = _route_methods(routes)
    enabled_modules = {
        key: _has_registered_resource(route_paths, path)
        for key, path in VN_RESOURCE_PATHS.items()
    }
    scripted_generation_enabled = enabled_modules["scripts"] and enabled_modules["play"]
    script_authoring_catalog_enabled = enabled_modules["scripts"]
    script_authoring_graph_enabled = _has_registered_route_methods(
        route_methods,
        VN_SCRIPT_AUTHORING_GRAPH_ROUTES,
    )
    script_playtest_enabled = _has_registered_route_methods(route_methods, VN_SCRIPT_PLAYTEST_ROUTES)

    return {
        "schema_version": "vn_capabilities.v1",
        "generated_at": datetime.now(UTC),
        "base_path": VN_BASE_PATH,
        "resources": dict(VN_RESOURCE_PATHS),
        "enabled_modules": enabled_modules,
        "features": {
            "asset_generation": enabled_modules["assets"],
            "asset_portability": enabled_modules["assets"],
            "scripted_story": scripted_generation_enabled,
            "scripted_generation": scripted_generation_enabled,
            "scripted_generation_confirmation": scripted_generation_enabled,
            "scripted_generation_revision_activation": enabled_modules["play"],
            "scripted_generation_history": enabled_modules["play"],
            "scripted_generation_debug_detail": enabled_modules["play"],
            "script_authoring_catalog": script_authoring_catalog_enabled,
            "script_authoring_graph": script_authoring_graph_enabled,
            "script_playtest": script_playtest_enabled,
            "story_start": enabled_modules["play"],
            "tts_jobs": enabled_modules["audio"],
            "realtime_image_generation": False,
            "subscriptions": False,
        },
        "limits": {
            "max_pack_items": DEFAULT_VN_ASSET_PACK_ITEM_LIMIT,
            "max_slot_variants": DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
            "max_choices_per_scene": 8,
            "runtime_model_timeout_seconds": 120,
            "max_scripted_generation_output_schemas": len(VN_SCRIPTED_GENERATION_OUTPUT_SCHEMAS),
            "max_automatic_generation_batch_count": VN_SCRIPTED_GENERATION_AUTOMATIC_BATCH_LIMIT,
        },
        "supported_content_ratings": ["general", "teen", "suggestive", "mature"],
        "visible_policy_profiles": [
            {"id": "local_default", "name": "Local Default", "visible": True},
            {"id": "strict_hosted", "name": "Strict Hosted", "visible": True},
        ],
        "visible_generation_profiles": [
            {"id": "story_default", "name": "Story Default", "visible": True},
        ],
        "supported_media_types": {
            "image": list(VN_SUPPORTED_IMAGE_MEDIA_TYPES),
            "audio": list(VN_SUPPORTED_AUDIO_MEDIA_TYPES),
        },
        "scripted_generation": {
            "enabled": scripted_generation_enabled,
            "output_schemas": list(VN_SCRIPTED_GENERATION_OUTPUT_SCHEMAS),
            "confirmation_supported": scripted_generation_enabled,
            "revision_activation_supported": enabled_modules["play"],
            "history_supported": enabled_modules["play"],
            "debug_detail_supported": enabled_modules["play"],
            "dynamic_choice_supported": scripted_generation_enabled,
            "scene_update_supported": scripted_generation_enabled,
            "max_automatic_generation_batch_count": VN_SCRIPTED_GENERATION_AUTOMATIC_BATCH_LIMIT,
            "moderation_blocked_raw_reveal_supported": scripted_generation_enabled,
        },
        "route_migration": {
            "canonical": "/api/v1/vn/vn-*",
            "supersedes": ["/api/v1/vn-assets", "/api/v1/vn-play"],
        },
        "docs": {
            "assets": "/docs#/vn-assets",
            "play": "/docs#/vn-play",
            "platform": "/docs#/vn-capabilities",
        },
        "openapi": "/openapi.json",
    }


def _route_paths(routes: Iterable[Any]) -> set[str]:
    """Return the FastAPI route paths registered in the current app."""
    return {
        path
        for route in routes
        if isinstance(path := getattr(route, "path", None), str)
    }


def _route_methods(routes: Iterable[Any]) -> set[tuple[str, str]]:
    """Return registered (path, HTTP method) pairs for FastAPI routes."""
    pairs: set[tuple[str, str]] = set()
    for route in routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not isinstance(path, str) or not isinstance(methods, Iterable):
            continue
        for method in methods:
            if isinstance(method, str):
                pairs.add((path, method.upper()))
    return pairs


def _has_registered_resource(paths: set[str], resource_path: str) -> bool:
    """Return whether any registered route belongs to a resource prefix."""
    prefix = resource_path.rstrip("/")
    return any(path == prefix or path.startswith(f"{prefix}/") for path in paths)


def _has_registered_route_methods(
    route_methods: set[tuple[str, str]],
    required_routes: Iterable[tuple[str, str]],
) -> bool:
    """Return whether all required route paths exist with their expected methods."""
    return all((path, method.upper()) in route_methods for path, method in required_routes)
