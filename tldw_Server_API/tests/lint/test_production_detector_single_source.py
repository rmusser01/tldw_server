"""Ratchet: production detection has one source, core/testing.runtime_environment() (TASK-13950.6).

About twenty modules used to carry their own production detector, with eight different
key sets, so APP_ENV=production tripped some guards and skipped others (the AuthNZ
weak-key and JWT-template hard-fails among them). They now call runtime_environment()
or is_production_like_env().

Counted: every string constant in app/ equal to one of PRODUCTION_ENV_KEYS, outside
core/testing.py. AST-derived, so getenv, environ.get, environ[...] and key tuples all
count, and comments and docstrings never do. Test modules shipped inside app/
(``app/**/tests/``) are skipped. Each allowlisted module names the exact keys it may
read, so a new key in an allowlisted module fails like a new module does; an allowed
key no longer read is stale and fails too.

Counter: ``python tldw_Server_API/tests/lint/test_production_detector_single_source.py``
prints the modules and keys the matcher finds. Seed: 8 modules, all in ALLOWLIST.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_Server_API.app.core.testing import PRODUCTION_ENV_KEYS

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"
THIS_FILE = "tldw_Server_API/tests/lint/test_production_detector_single_source.py"
SOURCE = "tldw_Server_API/app/core/testing.py"
_KEYS = frozenset(PRODUCTION_ENV_KEYS)
_ENV_LABELS = frozenset({"ENVIRONMENT", "APP_ENV", "ENV"})

# module -> (why it may name production keys, exactly which keys). Adding a module or a
# key is a baseline bump: give the reason here and get it reviewed in the PR. Remove an
# entry, or a key, as soon as the read goes away (the stale check enforces it).
ALLOWLIST: dict[str, tuple[str, frozenset[str]]] = {
    "tldw_Server_API/app/core/Metrics/telemetry.py": (
        "label only: DEPLOYMENT_ENV becomes the OpenTelemetry deployment.environment attribute",
        frozenset({"DEPLOYMENT_ENV"}),
    ),
    "tldw_Server_API/app/services/startup_sentry.py": (
        "label only: DEPLOYMENT_ENV becomes the Sentry environment tag",
        frozenset({"DEPLOYMENT_ENV"}),
    ),
    "tldw_Server_API/app/core/Evaluations/config_manager.py": (
        "label only: ENVIRONMENT names the evaluations YAML `environments:` override block",
        frozenset({"ENVIRONMENT"}),
    ),
    "tldw_Server_API/app/core/Evaluations/evaluation_manager.py": (
        "staging tier only (ENVIRONMENT=staging); production comes from is_production_like_env()",
        frozenset({"ENVIRONMENT"}),
    ),
    "tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py": (
        "dev label only: ENVIRONMENT/APP_ENV/ENV = dev|development|test|ci widens the "
        "SINGLE_USER_TEST_API_KEY dev context; production comes from is_production_like_env()",
        _ENV_LABELS,
    ),
    "tldw_Server_API/app/core/MCP_unified/server.py": (
        "standalone package boundary (apps/mcp-unified): host flags arrive through "
        "EnvironmentFlagsProvider, which has no production hook; startup already refuses "
        "SINGLE_USER_TEST_API_KEY and TEST_MODE in a production-like runtime",
        _ENV_LABELS | {"tldw_production"},
    ),
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py": (
        "dev label only: the MCTS canary runs when ENVIRONMENT/APP_ENV/ENV names a dev tier",
        _ENV_LABELS,
    ),
    "tldw_Server_API/app/core/Utils/egress.py": (
        "owner decision pending: the default egress profile is strict only for "
        "ENVIRONMENT/APP_ENV/ENV=prod|production; routing it through runtime_environment() "
        "would deny every non-allowlisted outbound host for tldw_production=true deploys "
        "(docker-compose.production.yml)",
        _ENV_LABELS,
    ),
}


def production_key_reads(source: str, filename: str = "<memory>") -> set[str]:
    """Return the production keys ``source`` names as string constants."""
    tree = ast.parse(source, filename=filename)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _KEYS
    }


def modules_reading_production_keys(app_root: Path = APP_ROOT) -> dict[str, set[str]]:
    """Repo-relative module path -> production keys it names, for every .py under ``app_root``."""
    repo_root = app_root.parents[1]
    found: dict[str, set[str]] = {}
    for path in sorted(app_root.rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel == SOURCE or "tests" in path.relative_to(app_root).parts:
            continue
        text = path.read_text(encoding="utf-8")
        if not any(f'"{key}"' in text or f"'{key}'" in text for key in _KEYS):
            continue  # cheap pre-filter: a constant equal to a key appears quoted in the source
        hits = production_key_reads(text, rel)
        if hits:
            found[rel] = hits
    return found


def ratchet_violations(
    found: dict[str, set[str]], allowlist: dict[str, tuple[str, frozenset[str]]]
) -> list[str]:
    """One line per production key read that is not allowlisted, and per stale allowlisted key."""
    problems: list[str] = []
    for module, keys in sorted(found.items()):
        extra = keys - allowlist.get(module, ("", frozenset()))[1]
        if extra:
            problems.append(
                f"{module}: reads {', '.join(sorted(extra))}. Call core/testing.runtime_environment() "
                "or is_production_like_env() instead. Only a label-only read may stay: add the "
                f"module and key with a reason to ALLOWLIST in {THIS_FILE} (a reviewed baseline bump)."
            )
    for module, (_reason, allowed) in sorted(allowlist.items()):
        gone = allowed - found.get(module, set())
        if gone:
            problems.append(
                f"{module}: ALLOWLIST lists {', '.join(sorted(gone))}, no longer read there. "
                f"Remove it from ALLOWLIST in {THIS_FILE}."
            )
    return problems


def test_production_keys_are_read_only_by_core_testing() -> None:
    problems = ratchet_violations(modules_reading_production_keys(), ALLOWLIST)
    assert not problems, "\n".join(problems)


def test_every_allowlist_entry_has_a_reason_and_known_keys() -> None:
    for reason, keys in ALLOWLIST.values():
        assert reason.strip()
        assert keys and keys <= _KEYS


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
    assert THIS_FILE in problems[0]


def test_a_new_key_in_an_allowlisted_module_fails() -> None:
    allowlist = {"app/telemetry.py": ("label", frozenset({"DEPLOYMENT_ENV"}))}
    problems = ratchet_violations({"app/telemetry.py": {"DEPLOYMENT_ENV", "tldw_production"}}, allowlist)
    assert len(problems) == 1
    assert problems[0].startswith("app/telemetry.py: reads tldw_production. ")


def test_matcher_ignores_docstrings_comments_and_lookalikes() -> None:
    snippet = '"""Set APP_ENV=production."""\n# os.getenv("ENV")\nx = "ENVIRONMENT_NAME"\ny = "app_env"\n'
    assert production_key_reads(snippet) == set()


def test_stale_allowlist_module_and_key_fail() -> None:
    allowlist = {
        "app/gone.py": ("reason", frozenset({"ENV"})),
        "app/kept.py": ("reason", frozenset({"ENV", "APP_ENV"})),
    }
    assert ratchet_violations({"app/kept.py": {"ENV"}}, allowlist) == [
        f"app/gone.py: ALLOWLIST lists ENV, no longer read there. Remove it from ALLOWLIST in {THIS_FILE}.",
        f"app/kept.py: ALLOWLIST lists APP_ENV, no longer read there. Remove it from ALLOWLIST in {THIS_FILE}.",
    ]


def test_scan_finds_seeded_files_through_the_pre_filter(tmp_path: Path) -> None:
    app = tmp_path / "tldw_Server_API" / "app"
    files = {
        "core/new_detector.py": "import os\nPROD = os.getenv('APP_ENV') == 'prod'\n",  # single quotes
        "core/flagged.py": 'import os\nPROD = os.environ["tldw_production"]\n',
        "core/telemetry.py": 'import os\nA = os.getenv("DEPLOYMENT_ENV")\nB = os.getenv("TLDW_ENV")\n',
        "core/testing.py": 'KEYS = ("ENV", "APP_ENV")\n',  # the one source is exempt
        "core/Thing/tests/test_x.py": 'X = "ENVIRONMENT"\n',  # test code under app/ is skipped
        "core/docs_only.py": '"""Set ENV=prod."""\n# os.getenv("ENV")\n',  # no constant read
    }
    for rel, text in files.items():
        path = app / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    found = modules_reading_production_keys(app)

    assert found == {
        "tldw_Server_API/app/core/flagged.py": {"tldw_production"},
        "tldw_Server_API/app/core/new_detector.py": {"APP_ENV"},
        "tldw_Server_API/app/core/telemetry.py": {"DEPLOYMENT_ENV", "TLDW_ENV"},
    }
    allowlist = {"tldw_Server_API/app/core/telemetry.py": ("label", frozenset({"DEPLOYMENT_ENV"}))}
    assert [problem.split(". ")[0] for problem in ratchet_violations(found, allowlist)] == [
        "tldw_Server_API/app/core/flagged.py: reads tldw_production",
        "tldw_Server_API/app/core/new_detector.py: reads APP_ENV",
        "tldw_Server_API/app/core/telemetry.py: reads TLDW_ENV",
    ]


if __name__ == "__main__":
    for module, keys in modules_reading_production_keys().items():
        print(f"{module}: {', '.join(sorted(keys))}")
