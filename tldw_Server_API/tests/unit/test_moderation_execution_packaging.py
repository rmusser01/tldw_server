"""Exercise the installed release graph, not an editable checkout or PYTHONPATH."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement

from tldw_Server_API.app.core.Moderation.execution_models import ExecutionBounds
from tldw_Server_API.app.core.Moderation.execution_protocol import (
    decode_response,
    encode_request,
    snapshot_redaction_batch_request,
    snapshot_request,
)
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits

pytestmark = [pytest.mark.unit, pytest.mark.packaging]
ROOT = Path(__file__).resolve().parents[3]


def test_dev_extra_includes_offline_wheel_build_requirements():
    """The installed-release fixture must work in a fresh CI dev environment."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = {Requirement(spec).name for spec in project["build-system"]["requires"]} | {"build"}
    available = {Requirement(spec).name for spec in project["project"]["optional-dependencies"]["dev"]}

    assert required <= available


@pytest.fixture(scope="module")
def installed_release(tmp_path_factory):
    """Build tracked current sources offline; never copy local config/secrets."""
    directory = tmp_path_factory.mktemp("moderation-release")
    source = directory / "source"
    source.mkdir()
    git = shutil.which("git")
    assert git is not None, "wheel source inventory requires git"
    tracked = subprocess.run(
        [git, "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
    ).stdout.decode().split("\0")
    prefixes = ("tldw_Server_API/", "apps/mcp-unified/src/", "packages/tldw_profile_core/", "LICENSES/")
    for name in tracked:
        if not name or not (name in {"pyproject.toml", "README.md", "LICENSE"} or name.startswith(prefixes)):
            continue
        if name.startswith("tldw_Server_API/tests/"):
            continue
        path = ROOT / name
        if path.is_file():
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    artifacts = directory / "dist"
    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(artifacts)],
        cwd=source, capture_output=True, text=True, timeout=180,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel, = artifacts.glob("*.whl")
    target = directory / "installed"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(target), str(wheel)],
        cwd=directory, capture_output=True, text=True, timeout=60,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    return target, directory


@pytest.mark.parametrize("operation", ["evaluate", "redact_many"])
@pytest.mark.parametrize("flags,pattern,text,action,redacted", [
    (re.IGNORECASE | re.MULTILINE, "^release-probe$", "safe\nRELEASE-PROBE\nsafe", "block", "safe\n[R]\nsafe"),
    (re.ASCII | re.IGNORECASE, r"^\w+$", "cafe\u00e9", "pass", "cafe\u00e9"),
    (re.IGNORECASE, r"^\w+$", "cafe\u00e9", "block", "[R]"),
    (re.DEBUG, "^release-probe$", "safe\nRELEASE-PROBE\nsafe", "pass", "safe\nRELEASE-PROBE\nsafe"),
], ids=["multiline", "ascii", "unicode", "debug"])
def test_installed_worker_outside_checkout_preserves_flags_without_app_imports(
    installed_release, flags, pattern, text, action, redacted, operation,
):
    """Missing graph files, package initializers, debug output or engine swaps fail here."""
    target, cwd = installed_release
    worker = target / "tldw_Server_API/app/core/Moderation/regex_worker.py"
    # Blocking imports observes the actual child graph, including attempted imports.
    code = f"""
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {{'regex', 'fastapi', 'sqlite3', 'loguru', 'pydantic'}}:
        raise AssertionError('nonminimal worker import')
    if name.startswith(('tldw_Server_API.app.main', 'tldw_Server_API.app.core.DB_Management',
                        'tldw_Server_API.app.core.LLM_Calls', 'tldw_Server_API.app.core.config')):
        raise AssertionError('application initialization')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_path({str(worker)!r}, run_name='__main__')
"""
    policy = ModerationPolicy(enabled=True, block_patterns=[re.compile(pattern, flags)], redact_replacement="[R]")
    if operation == "evaluate":
        request = snapshot_request(
            "evaluate", text, policy, "input", EvaluationLimits(100, 10, 1000, None), ExecutionBounds(),
        )
    else:
        request = snapshot_redaction_batch_request(
            [text, ""], policy, "input", EvaluationLimits(100, 10, 1000, None), ExecutionBounds(),
        )
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code], cwd=cwd,
        input=encode_request(request), capture_output=True, timeout=10,
    )
    assert child.returncode == 0
    assert child.stderr == b""
    response = decode_response(child.stdout, request)
    assert response.reason is None
    if operation == "evaluate":
        assert response.result.action == action
    else:
        assert response.result == [redacted, ""]
    assert response.resource_limits["cpu"] is (sys.platform != "win32")


def test_installed_release_includes_operator_notes(installed_release):
    """Wheel-only operators need the same rollout and platform notes as source installs."""
    target, _ = installed_release
    assert (target / "tldw_Server_API/app/core/Moderation/README.md").is_file()


def test_release_neutral_writes_require_compatible_rollback_readers(tmp_path):
    """Legacy readers reject neutral rows; a compatible restart retains identity/history."""
    from typing import Literal

    from pydantic import BaseModel, ValidationError

    from tldw_Server_API.app.api.v1.schemas.moderation_schemas import ModerationReviewItem
    from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore

    class OldPhaseReader(BaseModel):
        phase: Literal["input", "output"]

    path = tmp_path / "release-review.db"
    store = ModerationReviewStore(path)
    item = store.upsert_item({
        "idempotency_key": "release-incident", "phase": "unspecified",
        "source_type": "moderation_engine", "category": "moderation_engine_failure",
        "excerpt": "[Moderation execution failed; content withheld]",
        "severity": "high", "recommended_action": "escalate",
    })
    assert ModerationReviewItem(**item).phase == "unspecified"
    with pytest.raises(ValidationError):
        OldPhaseReader(**item)
    decision = store.record_decision(item["id"], action="escalate", decided_by="release-reviewer")
    restarted = ModerationReviewStore(path)
    retained = restarted.get_item(item["id"], include_history=True)
    assert retained["phase"] == "unspecified"
    assert len(retained["decision_history"]) == 1
    restarted.undo_decision(item["id"], undo_token=decision["undo_token"], actor_id="release-reviewer")
    assert restarted.get_item(item["id"])["status"] == "needs_review"


@pytest.mark.parametrize("fault", ["none", "worker", "interpreter"])
@pytest.mark.parametrize("operation", ["custom_substitute", "redact_many"])
def test_installed_controller_launch_and_missing_artifacts_fail_closed(installed_release, fault, operation):
    """Real controller/child exchange must not use checkout paths or expose payloads."""
    target, cwd = installed_release
    code = f"""
import json, pathlib, subprocess, sys, types
root = pathlib.Path({str(target)!r})
for name in ('tldw_Server_API', 'tldw_Server_API.app', 'tldw_Server_API.app.core',
             'tldw_Server_API.app.core.Moderation', 'tldw_Server_API.app.core.Web_Scraping'):
    module = types.ModuleType(name)
    module.__path__ = [str(root.joinpath(*name.split('.')))]
    sys.modules[name] = module
from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation import execution_runner as runner
from tldw_Server_API.app.core.Moderation.execution_models import ExecutionBounds, execution_scope
from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_substitution_request, snapshot_redaction_batch_request
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits
secret = 'private-release-payload'
launches = []
def spawn(argv, **kwargs):
    assert secret not in repr((argv, kwargs))
    assert argv[1:3] == ['-I', '-S']
    assert pathlib.Path(argv[3]).is_relative_to(root)
    assert kwargs['shell'] is False
    launches.append(argv)
    return subprocess.Popen(argv, **kwargs)
if {fault!r} == 'interpreter':
    sys.executable = str(root / 'absent-interpreter')
elif {fault!r} == 'worker':
    runner.__file__ = str(root / 'absent-package' / 'execution_runner.py')
owner = runner.ModerationExecutionRunner(ExecutionBounds(), process_factory=spawn)
request = (snapshot_redaction_batch_request([secret, ''], ModerationPolicy(), None, EvaluationLimits(100, 10, 1000, None), ExecutionBounds())
           if {operation!r} == 'redact_many' else snapshot_substitution_request(secret, [], None, ExecutionBounds()))
try:
    with execution_scope() as scope:
        try:
            result = owner.execute(request, scope)
        except ModerationExecutionError as error:
            assert secret not in str(error) + repr(error)
            assert {fault!r} != 'none'
            print(json.dumps({{'reason': error.reason}}))
        else:
            assert {fault!r} == 'none'
            assert (result.result if {operation!r} == 'redact_many' else [result.result.text, '']) == [secret, '']
            print(json.dumps({{'reason': None}}))
    assert len(launches) == 1
finally:
    owner.close()
assert not owner._entries
"""
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code], cwd=cwd,
        capture_output=True, text=True, timeout=10,
    )
    assert child.returncode == 0, child.stderr
    assert child.stderr == ""
    assert "private-release-payload" not in child.stdout
    expected = {"none": None, "worker": "worker_crashed", "interpreter": "worker_unavailable"}
    assert json.loads(child.stdout) == {"reason": expected[fault]}
