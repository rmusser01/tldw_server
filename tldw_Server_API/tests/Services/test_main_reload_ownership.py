"""Finite source-bound checks for application ownership across main reloads."""

from __future__ import annotations

import ast
import gc
import weakref
from contextlib import suppress
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[2] / "app" / "main.py"


def _nodes():
    return ast.parse(SOURCE.read_text(encoding="utf-8")).body


def test_openapi_callable_keeps_its_own_warm_cache_after_module_reload():
    """An older app must not serve a replacement app's cached schema."""
    first = SimpleNamespace(openapi_schema={"owner": "first"})
    second = SimpleNamespace(openapi_schema={"owner": "second"})
    node = next(node for node in _nodes() if isinstance(node, ast.FunctionDef) and node.name == "custom_openapi")
    namespace = {"app": first, "FastAPI": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    first.openapi = namespace["custom_openapi"]
    namespace["app"] = second

    assert first.openapi() is first.openapi_schema
    assert second.openapi_schema == {"owner": "second"}


def test_openapi_cold_generation_uses_owner_routes_after_module_reload():
    from fastapi import FastAPI
    from fastapi.openapi.utils import get_openapi

    first = FastAPI(title="first")
    second = FastAPI(title="second")

    async def endpoint():
        return {"owner": "first"}

    first.add_api_route("/first", endpoint)
    nodes = [
        node
        for node in _nodes()
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in {"custom_openapi", "_ensure_openapi_operation_tags_declared"}
        )
        or (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_OPENAPI_HTTP_METHODS")
    ]
    namespace = {"app": first, "FastAPI": FastAPI, "get_openapi": get_openapi, "OPENAPI_TAGS": []}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    first.openapi = namespace["custom_openapi"]
    namespace["app"] = second

    schema = first.openapi()

    assert schema["info"]["title"] == "first"
    assert "/first" in schema["paths"]
    assert first.openapi_schema is schema
    assert first.openapi() is schema
    assert second.openapi_schema is None


class _Logger:
    def __init__(self):
        self._tldw_original_configure = self.configure

    def add(self, sink, *args, **kwargs):
        return sink, args, kwargs

    def configure(self, *args, **kwargs):
        return args, kwargs


class _App:
    pass


def _install_logging(logger, app):
    """Execute actual wrapper definitions and binding statements, not app imports."""
    functions = {"_unwrap_logger_add", "_unwrap_loguru_wrapper", "_safe_logger_add", "_safe_logger_configure"}
    names = {
        "_ROOT_LOGGER",
        "_original_logger_add",
        "_original_unwrapped_logger_add",
        "_root_configure",
        "_original_logger_configure",
    }
    prefixes = ("_ROOT_LOGGER.add", "_ROOT_LOGGER.configure", "logger.configure")
    selected = []
    for node in _nodes():
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in functions
            or isinstance(node, ast.ClassDef)
            and node.name == "_SafeStreamWrapper"
        ):
            selected.append(node)
        elif isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            if target in names or target.startswith(prefixes):
                selected.append(node)
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "callable(_original_logger_configure)":
            selected.append(node)
    namespace = {
        "logger": logger,
        "app": app,
        "_caller_allowed_for_loguru_config": lambda: True,
        "_LOGGING_SETUP_EXCEPTIONS": (OSError, ValueError, RuntimeError),
        "_safe_debug": lambda message: None,
        "_redirect_external_loggers": lambda: None,
        "_IO_EXCEPTIONS": (OSError, ValueError),
        "suppress": suppress,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)


def test_logging_reinstallation_releases_previous_application_namespaces():
    """Global logging roots must not retain every prior module's application."""
    logger = _Logger()
    applications = []
    for _ in range(8):
        app = _App()
        applications.append(weakref.ref(app))
        _install_logging(logger, app)
    del app
    gc.collect()

    assert sum(reference() is not None for reference in applications) == 1
    assert logger.add("sink", 1, level="INFO") == ("sink", (1,), {"level": "INFO"})
    assert logger.configure(extra={"owner": "current"}) == ((), {"extra": {"owner": "current"}})


def test_logging_reinstallation_preserves_denied_caller_guard():
    logger = _Logger()
    _install_logging(logger, _App())
    logger.add.__globals__["_caller_allowed_for_loguru_config"] = lambda: False

    assert logger.add("sink") is None
    assert logger.configure(extra={"owner": "blocked"}) is None


def test_logging_reinstallation_with_real_loguru_keeps_sink_working():
    from loguru import logger as root_logger

    logger = root_logger.bind()
    logger._tldw_original_configure = logger.configure
    applications = []
    for _ in range(8):
        app = _App()
        applications.append(weakref.ref(app))
        _install_logging(logger, app)
    del app
    gc.collect()
    assert sum(reference() is not None for reference in applications) == 1

    sink = StringIO()
    sink_id = logger.add(sink, format="{message}")
    try:
        logger.info("reload-safe")
        assert sink.getvalue() == "reload-safe\n"
    finally:
        logger.remove(sink_id)
