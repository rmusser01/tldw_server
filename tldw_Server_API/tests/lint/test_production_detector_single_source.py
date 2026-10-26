"""Ratchet: production detection has one source, core/testing.runtime_environment() (TASK-13950.6).

About twenty modules used to carry their own production detector, with eight different
key sets, so APP_ENV=production tripped some guards and skipped others (the AuthNZ
weak-key and JWT-template hard-fails among them). They now call runtime_environment()
or is_production_like_env().

Counted: every string constant in app/ equal to one of PRODUCTION_ENV_KEYS, outside
core/testing.py. AST-derived, so getenv, environ.get, environ[...] and key tuples all
count, and comments and docstrings never do. Test modules shipped inside app/
(``app/**/tests/``) are skipped. A module with a hit must be in ALLOWLIST with a reason
(label-only reads, or a documented exception); an entry whose module has no hit left
is stale and fails too.

Counter: ``python tldw_Server_API/tests/lint/test_production_detector_single_source.py``
prints the modules the matcher finds. Seed: 8 modules, all in ALLOWLIST.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from tldw_Server_API.app.core.testing import PRODUCTION_ENV_KEYS

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-required.yml"
SOURCE = "tldw_Server_API/app/core/testing.py"
_KEYS = frozenset(PRODUCTION_ENV_KEYS)

# module -> why it may name a production key. Entries may only be removed.
ALLOWLIST: dict[str, str] = {
    "tldw_Server_API/app/core/Metrics/telemetry.py": (
        "label only: DEPLOYMENT_ENV becomes the OpenTelemetry deployment.environment attribute"
    ),
    "tldw_Server_API/app/services/startup_sentry.py": "label only: DEPLOYMENT_ENV becomes the Sentry environment tag",
    "tldw_Server_API/app/core/Evaluations/config_manager.py": (
        "label only: ENVIRONMENT names the evaluations YAML `environments:` override block"
    ),
    "tldw_Server_API/app/core/Evaluations/evaluation_manager.py": (
        "staging tier only (ENVIRONMENT=staging); production comes from is_production_like_env()"
    ),
    "tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py": (
        "dev label only: ENVIRONMENT/APP_ENV/ENV = dev|development|test|ci widens the "
        "SINGLE_USER_TEST_API_KEY dev context; production comes from is_production_like_env()"
    ),
    "tldw_Server_API/app/core/MCP_unified/server.py": (
        "standalone package boundary (apps/mcp-unified): host flags arrive through "
        "EnvironmentFlagsProvider, which has no production hook; startup already refuses "
        "SINGLE_USER_TEST_API_KEY and TEST_MODE in a production-like runtime"
    ),
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py": (
        "dev label only: the MCTS canary runs when ENVIRONMENT/APP_ENV/ENV names a dev tier"
    ),
    "tldw_Server_API/app/core/Security/egress.py": (
        "owner decision pending: the default egress profile is strict only for "
        "ENVIRONMENT/APP_ENV/ENV=prod|production; routing it through runtime_environment() "
        "would deny every non-allowlisted outbound host for tldw_production=true deploys "
        "(docker-compose.production.yml)"
    ),
}


def production_key_reads(source: str, filename: str = "<memory>") -> list[str]:
    """Return the production keys ``source`` names as string constants, one per occurrence."""
    tree = ast.parse(source, filename=filename)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _KEYS
    ]


def _modules_reading_production_keys() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(APP_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == SOURCE or "tests" in path.relative_to(APP_ROOT).parts:
            continue
        text = path.read_text(encoding="utf-8")
        if not any(f'"{key}"' in text or f"'{key}'" in text for key in _KEYS):
            continue  # cheap pre-filter: a constant equal to a key appears quoted in the source
        hits = production_key_reads(text, rel)
        if hits:
            found[rel] = sorted(set(hits))
    return found


def ratchet_violations(found: dict[str, list[str]], allowlist: dict[str, str]) -> list[str]:
    """One line per module that reads a production key unallowlisted, or per stale entry."""
    problems = [
        f"{module}: reads {', '.join(keys)}. Call core/testing.runtime_environment() or "
        "is_production_like_env() instead; add an ALLOWLIST entry with a reason only for a label-only read."
        for module, keys in sorted(found.items())
        if module not in allowlist
    ]
    problems += [
        f"{module}: ALLOWLIST entry is stale (no production key read left). Remove it."
        for module in sorted(set(allowlist) - set(found))
    ]
    return problems


def test_production_keys_are_read_only_by_core_testing() -> None:
    problems = ratchet_violations(_modules_reading_production_keys(), ALLOWLIST)
    assert not problems, "\n".join(problems)


def test_every_allowlist_entry_has_a_reason() -> None:
    assert all(reason.strip() for reason in ALLOWLIST.values())


@pytest.mark.parametrize(
    "snippet",
    [
        'import os\nprod = os.getenv("APP_ENV") == "production"\n',
        'from os import environ as e\nprod = e["tldw_production"]\n',
        'KEYS = ("ENV", "TLDW_ENV")\n',
        'def f(read):\n    return read("DEPLOYMENT_ENV")\n',
    ],
)
def test_matcher_sees_seeded_violations(snippet: str) -> None:
    assert production_key_reads(snippet)
    problems = ratchet_violations({"app/new.py": production_key_reads(snippet)}, {})
    assert problems and problems[0].startswith("app/new.py: reads ")


def test_matcher_ignores_docstrings_comments_and_lookalikes() -> None:
    snippet = '"""Set APP_ENV=production."""\n# os.getenv("ENV")\nx = "ENVIRONMENT_NAME"\ny = "app_env"\n'
    assert production_key_reads(snippet) == []


def test_stale_allowlist_entry_fails() -> None:
    assert ratchet_violations({}, {"app/gone.py": "reason"}) == [
        "app/gone.py: ALLOWLIST entry is stale (no production key read left). Remove it."
    ]


def test_backend_required_runs_this_ratchet() -> None:
    steps = [
        step
        for job in yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"].values()
        for step in job.get("steps", [])
        if step.get("name") == "Enforce CI contracts and code ratchets"
    ]
    assert len(steps) == 1
    assert "tldw_Server_API/tests/lint/test_production_detector_single_source.py" in steps[0]["run"]


if __name__ == "__main__":
    for module, keys in _modules_reading_production_keys().items():
        print(f"{module}: {', '.join(keys)}")
