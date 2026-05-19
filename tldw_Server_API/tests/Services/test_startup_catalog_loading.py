from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _import_startup_catalog_loading():
    sys.modules.pop("tldw_Server_API.app.services.startup_catalog_loading", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_catalog_loading")


class _FakeLogger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []

    def debug(self, message: str, *args: object) -> None:
        if args:
            self.debug_messages.append(str(message).format(*args))
        else:
            self.debug_messages.append(str(message))


def test_load_startup_catalogs_delegates_to_archetype_and_mcp_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogs = _import_startup_catalog_loading()
    logger = _FakeLogger()
    config_dir = Path("/tmp/test-config")
    observed: dict[str, Path] = {}

    monkeypatch.setattr(catalogs, "_resolve_config_dir", lambda module_file: config_dir)
    monkeypatch.setattr(
        catalogs,
        "_load_archetypes_from_directory",
        lambda path: observed.__setitem__("archetypes", path),
    )
    monkeypatch.setattr(
        catalogs,
        "_load_mcp_catalog",
        lambda path: observed.__setitem__("catalog", path),
    )

    catalogs.load_startup_catalogs(
        module_file="/repo/tldw_Server_API/app/main.py",
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert observed["archetypes"] == config_dir / "persona_archetypes"
    assert observed["catalog"] == config_dir / "mcp_server_catalog.yaml"
    assert logger.debug_messages == []


def test_load_startup_catalogs_logs_debug_on_guard_or_import_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogs = _import_startup_catalog_loading()
    logger = _FakeLogger()

    monkeypatch.setattr(catalogs, "_resolve_config_dir", lambda module_file: Path("/tmp/test-config"))

    def _raise_loader_error(path: Path) -> None:
        del path
        raise RuntimeError("boom")

    monkeypatch.setattr(catalogs, "_load_archetypes_from_directory", _raise_loader_error)

    catalogs.load_startup_catalogs(
        module_file="/repo/tldw_Server_API/app/main.py",
        logger=logger,
        startup_guard_exceptions=(RuntimeError,),
        import_exceptions=(ImportError,),
    )

    assert logger.debug_messages == ["Archetype/catalog loading skipped: boom"]
