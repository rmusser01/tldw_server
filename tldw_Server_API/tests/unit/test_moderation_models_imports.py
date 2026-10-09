from __future__ import annotations

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
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompiler,
    ResolvedModerationConfig,
)
assert {_SERVICE_MODULE!r} not in sys.modules
result = PolicyCompiler().compile_global(
    PolicyCompilationInput(config=ResolvedModerationConfig())
)
assert type(result.policy) is ModerationPolicy
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
""",
        f"""
import sys
from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult,
    ModerationPolicy,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)
assert {_SERVICE_MODULE!r} not in sys.modules
result = PolicyEvaluator().evaluate_text(
    "",
    ModerationPolicy(enabled=False),
    "input",
    EvaluationLimits(1024, 128, 4096, None),
    include_redacted_text=False,
)
assert type(result) is ModerationEvaluationResult
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
assert moderation_service.ModerationEvaluationResult is models.ModerationEvaluationResult
""",
    ],
)
def test_representative_operations_do_not_load_service(script):
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


@pytest.mark.parametrize("name", ("ModerationPolicy", "PatternRule"))
def test_compiler_public_namespace_excludes_canonical_model(name):
    assert not hasattr(policy_compiler, name)


@pytest.mark.parametrize(
    "name",
    ("ModerationPolicy", "PatternRule", "ModerationEvaluationResult"),
)
def test_evaluator_public_namespace_excludes_canonical_model(name):
    assert not hasattr(policy_evaluator, name)


def test_compiler_runtime_type_hints_remain_unresolved():
    with pytest.raises(NameError):
        typing.get_type_hints(policy_compiler.PolicyCompiler.compile_user_policy)


def test_evaluator_runtime_type_hints_remain_unresolved():
    with pytest.raises(NameError):
        typing.get_type_hints(policy_evaluator.PolicyEvaluator.evaluate_text)


def test_service_export_rebinding_does_not_replace_canonical_operation_types(monkeypatch):
    monkeypatch.setattr(moderation_service, "ModerationPolicy", type("Policy", (), {}))
    monkeypatch.setattr(moderation_service, "PatternRule", type("Rule", (), {}))
    monkeypatch.setattr(
        moderation_service,
        "ModerationEvaluationResult",
        type("Result", (), {}),
    )

    compilation = policy_compiler.PolicyCompiler().compile_global(
        policy_compiler.PolicyCompilationInput(
            config=policy_compiler.ResolvedModerationConfig(enabled=True),
            blocklist_lines=["secret -> block"],
        )
    )
    evaluation = policy_evaluator.PolicyEvaluator().evaluate_text(
        "secret",
        compilation.policy,
        "input",
        policy_evaluator.EvaluationLimits(1024, 128, 4096, None),
        include_redacted_text=False,
    )

    assert type(compilation.policy) is ModerationPolicy
    assert len(compilation.policy.block_patterns) == 1
    assert type(compilation.policy.block_patterns[0]) is PatternRule
    assert type(evaluation) is ModerationEvaluationResult
