from __future__ import annotations

import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Moderation import (
    moderation_service,
    policy_compiler,
    policy_evaluator,
)
from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVICE_MODULE = "tldw_Server_API.app.core.Moderation.moderation_service"


@pytest.mark.parametrize(
    "script",
    [
        f"""
import sys
from tldw_Server_API.app.core.Moderation.policy_compiler import PolicyCompiler
assert {_SERVICE_MODULE!r} not in sys.modules
types = PolicyCompiler.policy_types()
assert [item.__name__ for item in types] == ["ModerationPolicy", "PatternRule"]
assert all(item.__module__ == "tldw_Server_API.app.core.Moderation.models" for item in types)
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
""",
        f"""
import sys
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
assert {_SERVICE_MODULE!r} not in sys.modules
types = PolicyEvaluator.policy_types()
assert [item.__name__ for item in types] == ["ModerationPolicy", "PatternRule", "ModerationEvaluationResult"]
assert all(item.__module__ == "tldw_Server_API.app.core.Moderation.models" for item in types)
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
assert moderation_service.ModerationEvaluationResult is models.ModerationEvaluationResult
""",
    ],
)
def test_policy_types_do_not_load_service(script):
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "module_order",
    [
        (
            "tldw_Server_API.app.core.Moderation.models",
            "tldw_Server_API.app.core.Moderation.moderation_service",
            "tldw_Server_API.app.core.Moderation.policy_compiler",
            "tldw_Server_API.app.core.Moderation.policy_evaluator",
        ),
        (
            "tldw_Server_API.app.core.Moderation.moderation_service",
            "tldw_Server_API.app.core.Moderation.policy_compiler",
            "tldw_Server_API.app.core.Moderation.policy_evaluator",
        ),
    ],
)
def test_complete_import_orders_resolve_exact_identity(module_order):
    script = f"""
import importlib

for module_name in {module_order!r}:
    importlib.import_module(module_name)
models = importlib.import_module("tldw_Server_API.app.core.Moderation.models")
service = importlib.import_module("tldw_Server_API.app.core.Moderation.moderation_service")
for name in ("ModerationPolicy", "PatternRule", "ModerationEvaluationResult"):
    assert getattr(service, name) is getattr(models, name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_policy_type_descriptors_and_public_namespaces_remain_literal():
    assert isinstance(
        inspect.getattr_static(policy_compiler.PolicyCompiler, "policy_types"),
        staticmethod,
    )
    assert isinstance(
        inspect.getattr_static(policy_evaluator.PolicyEvaluator, "policy_types"),
        staticmethod,
    )
    assert not hasattr(policy_compiler, "ModerationPolicy")
    assert not hasattr(policy_compiler, "PatternRule")
    assert not hasattr(policy_evaluator, "ModerationPolicy")
    assert not hasattr(policy_evaluator, "PatternRule")
    assert not hasattr(policy_evaluator, "ModerationEvaluationResult")

    with pytest.raises(NameError):
        typing.get_type_hints(policy_compiler.PolicyCompiler.compile_user_policy)
    with pytest.raises(NameError):
        typing.get_type_hints(policy_evaluator.PolicyEvaluator.evaluate_text)


def test_service_export_rebinding_does_not_replace_canonical_policy_types(monkeypatch):
    monkeypatch.setattr(moderation_service, "ModerationPolicy", type("Policy", (), {}))
    monkeypatch.setattr(moderation_service, "PatternRule", type("Rule", (), {}))
    monkeypatch.setattr(
        moderation_service,
        "ModerationEvaluationResult",
        type("Result", (), {}),
    )

    assert policy_compiler.PolicyCompiler.policy_types() == (
        ModerationPolicy,
        PatternRule,
    )
    assert policy_evaluator.PolicyEvaluator.policy_types() == (
        ModerationPolicy,
        PatternRule,
        ModerationEvaluationResult,
    )
