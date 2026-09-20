from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Moderation import models, moderation_service

pytestmark = pytest.mark.unit

_MODEL_NAMES = (
    "ModerationPolicy",
    "PatternRule",
    "ModerationEvaluationResult",
)
_FORBIDDEN_MODULES = (
    "tldw_Server_API.app.core.config",
    "tldw_Server_API.app.core.Moderation.moderation_service",
    "tldw_Server_API.app.core.Moderation.policy_compiler",
    "tldw_Server_API.app.core.Moderation.policy_evaluator",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_PATH = Path(__file__).resolve().parents[2] / "app" / "core" / "Moderation" / "models.py"


@pytest.mark.parametrize("name", _MODEL_NAMES)
def test_service_facade_exports_exact_canonical_class(name):
    assert getattr(moderation_service, name) is getattr(models, name)


@pytest.mark.parametrize("name", _MODEL_NAMES)
def test_canonical_class_uses_models_module_metadata(name):
    assert getattr(models, name).__module__ == models.__name__


def test_models_module_owns_exactly_three_dataclass_types():
    owned_dataclasses = {
        name
        for name, value in vars(models).items()
        if isinstance(value, type) and is_dataclass(value) and value.__module__ == models.__name__
    }

    assert owned_dataclasses == set(_MODEL_NAMES)


@pytest.mark.parametrize("name", _MODEL_NAMES)
def test_legacy_qualified_name_resolves_to_canonical_class(name):
    legacy = importlib.import_module("tldw_Server_API.app.core.Moderation.moderation_service")

    assert getattr(legacy, name) is getattr(models, name)


def test_models_source_imports_only_approved_standard_library_modules():
    tree = ast.parse(_MODELS_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])

    assert roots == {"__future__", "dataclasses", "json", "re"}


def test_models_import_adds_no_moderation_or_config_dependencies():
    script = f"""
import importlib
import sys

forbidden = {repr(_FORBIDDEN_MODULES)}
importlib.import_module("tldw_Server_API.app.core.Moderation")
assert not [name for name in forbidden if name in sys.modules]
importlib.import_module("tldw_Server_API.app.core.Moderation.models")
assert not [name for name in forbidden if name in sys.modules]
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
