from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import APIRouter


@pytest.mark.unit
def test_media_optional_import_sanitizes_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import media as media_module

    fake_logger = MagicMock()
    monkeypatch.setattr(media_module, "logger", fake_logger)

    def _failing_import_module(_name: str):
        raise RuntimeError("media import leaked /private/media-import.py")

    monkeypatch.setattr(media_module.importlib, "import_module", _failing_import_module)

    assert media_module._optional_import_module("leaky.private.module") is None
    fake_logger.warning.assert_called_once_with("Media import skipped: optional module unavailable")


@pytest.mark.unit
def test_media_router_keeps_core_routes_when_optional_module_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "tldw_Server_API.app.api.v1.endpoints.media"
    parent_package_name = "tldw_Server_API.app.api.v1.endpoints"
    module_prefix = f"{module_name}."
    original_media_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == module_name or name.startswith(module_prefix)
    }
    parent_package = sys.modules.get(parent_package_name)
    had_parent_media_attr = bool(parent_package and hasattr(parent_package, "media"))
    original_parent_media_attr = (
        getattr(parent_package, "media")
        if had_parent_media_attr and parent_package is not None
        else None
    )
    for name in list(sys.modules):
        if name == module_name or name.startswith(module_prefix):
            sys.modules.pop(name, None)
    original_import_module = importlib.import_module

    listing_router = APIRouter()

    @listing_router.get("/")
    async def _fake_list_media():
        return {}

    item_router = APIRouter()

    @item_router.get("/{media_id}")
    async def _fake_get_media_item(media_id: int):
        return {"id": media_id}

    fake_listing_module = types.SimpleNamespace(router=listing_router)
    fake_item_module = types.SimpleNamespace(router=item_router)

    def _patched_import_module(name: str, package: str | None = None):
        if name == module_name:
            return original_import_module(name, package)
        if name == f"{module_name}.listing":
            return fake_listing_module
        if name == f"{module_name}.item":
            return fake_item_module
        if name.startswith(f"{module_name}."):
            raise ImportError(f"simulated endpoint import failure for {name}")
        if "Ingestion_Media_Processing.Audio" in name:
            raise ImportError(f"simulated heavy-import block for {name}")
        if name.endswith("Video.Video_DL_Ingestion_Lib"):
            raise ImportError(f"simulated heavy-import block for {name}")
        return original_import_module(name, package)

    try:
        monkeypatch.setattr(importlib, "import_module", _patched_import_module)
        media_module = importlib.import_module(module_name)
        route_paths = {route.path for route in media_module.router.routes}

        assert "/" in route_paths
        assert "/{media_id}" in route_paths
        assert "/process-audios" not in route_paths
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_prefix):
                sys.modules.pop(name, None)
        sys.modules.update(original_media_modules)
        if parent_package is not None:
            if had_parent_media_attr:
                setattr(parent_package, "media", original_parent_media_attr)
            else:
                try:
                    delattr(parent_package, "media")
                except AttributeError:
                    pass
