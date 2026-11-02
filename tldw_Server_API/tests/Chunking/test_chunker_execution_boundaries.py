"""Active facade wiring, one-way dependencies and real cold package imports."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os

# Bounded cold imports use the current interpreter, never a shell.
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking import chunker as facade
from tldw_Server_API.app.core.Chunking.base import ChunkerConfig

pytestmark = pytest.mark.unit
ROOT_NAME = "tldw_Server_API.app.core.Chunking"
PACKAGE_NAME = f"{ROOT_NAME}.execution"
REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "tldw_Server_API/app/core/Chunking/execution"
MODULES = ("execution", "execution.models", "execution.service")


def test_text_facade_delegates_without_extra_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker(ChunkerConfig(enable_cache=False))
    sentinel_result = ["delegated text"]
    received = []
    payload: list[str] = []

    class SentinelService:
        def __init__(self, context: Any, hooks: Any) -> None:
            assert context is chunker

        def chunk_text(self, *args: Any, **kwargs: Any) -> Any:
            received.append((args, kwargs))
            return sentinel_result

    monkeypatch.setattr(facade, "ChunkExecutionService", SentinelService, raising=False)
    result = chunker.chunk_text("raw", "words", 4, 0, "en", path=payload, hooks="user hooks")
    assert result is sentinel_result
    assert received == [(("raw", "words", 4, 0, "en"), {"path": payload, "hooks": "user hooks"})]
    assert received[0][1]["path"] is payload


def test_hooks_resolve_replacement_during_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker(ChunkerConfig(enable_cache=False))
    assert callable(getattr(facade, "_execution_hooks", None)), "missing forwarding hook factory"
    hooks = facade._execution_hooks()
    logs = []
    counters = []
    logger = SimpleNamespace(debug=lambda *args, **kwargs: logs.append((args, kwargs)))
    monkeypatch.setattr(facade, "logger", logger)
    monkeypatch.setattr(facade, "increment_counter", lambda *args, **kwargs: counters.append((args, kwargs)))
    assert hooks.get_logger() is logger
    hooks.increment_counter("chunker_cache_get_total", labels={"result": "skip", "reason": "disabled"})
    result = chunker.chunk_text("alpha beta", method="words", max_size=1, overlap=0)
    assert result == ["alpha", "beta"]
    assert counters == [(("chunker_cache_get_total",), {"labels": {"result": "skip", "reason": "disabled"}})] * 2
    assert logs[-1] == (("Created 2 chunks using words method",), {})


@pytest.mark.parametrize("module_suffix", MODULES)
def test_execution_modules_are_required_and_one_way(module_suffix: str) -> None:
    module_name = f"{ROOT_NAME}.{module_suffix}"
    relative = module_suffix.removeprefix("execution").lstrip(".")
    path = PACKAGE / (relative.replace(".", "/") + ".py") if relative else PACKAGE / "__init__.py"
    assert path.is_file(), f"required module missing: {module_name}"
    module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve() == path.resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    blocked = (f"{ROOT_NAME}.chunker", f"{ROOT_NAME}.process_text", f"{ROOT_NAME}.hierarchical")
    for node in ast.walk(tree):
        assert not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__")
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = (
                importlib.util.resolve_name("." * node.level + (node.module or ""), PACKAGE_NAME)
                if node.level
                else node.module or ""
            )
            assert not (name == ROOT_NAME and any(alias.name in {"Chunker", "*"} for alias in node.names))
            names = [name] + [f"{name}.{alias.name}" for alias in node.names]
        assert not any(name == forbidden or name.startswith(forbidden + ".") for name in names for forbidden in blocked)


@pytest.mark.parametrize("requested_module", [ROOT_NAME, *(f"{ROOT_NAME}.{module}" for module in MODULES)])
def test_execution_cold_import(tmp_path: Path, requested_module: str) -> None:
    config = tmp_path / "config.txt"
    config.write_text("[Chunking]\nenable_cache = false\n", encoding="utf-8")
    env = dict(
        os.environ,
        TLDW_CONFIG_FILE=str(config),
        TLDW_TEST_MODE="1",
        USER_DB_BASE_DIR=str(tmp_path / "users"),
        AUTO_DOWNLOAD_MODELS="false",
        PYTHONPATH=str(REPO),
    )
    child = """
import importlib
import json
import sys
requested = sys.argv[1]
root_name = 'tldw_Server_API.app.core.Chunking'
allowed = {root_name, *(root_name + '.' + part for part in ('execution', 'execution.models', 'execution.service'))}
assert requested in allowed
importlib.import_module(requested)
root = importlib.import_module(root_name)
service = importlib.import_module(root_name + '.execution.service')
assert isinstance(root.Chunker, type) and isinstance(service.ChunkExecutionService, type)
print(json.dumps({'requested': requested, 'chunker': root.Chunker.__name__, 'service': service.ChunkExecutionService.__name__}))
"""
    # Fixed interpreter/code and allowlisted module input; no shell or external command.
    completed = subprocess.run(
        [sys.executable, "-c", child, requested_module],
        cwd=REPO,
        env=env,
        shell=False,
        capture_output=True,
        text=True,
        timeout=30,
    )  # nosec B603
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "requested": requested_module,
        "chunker": "Chunker",
        "service": "ChunkExecutionService",
    }
