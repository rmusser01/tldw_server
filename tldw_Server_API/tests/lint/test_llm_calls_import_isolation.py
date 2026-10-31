"""Guard: importing a module must not load a package it must stay independent of (TASK-13950.13).

Each case imports one module in a fresh interpreter and lists what loaded. The static
layering count (test_layering_contract.py) misses an edge laundered through a third
package (core.LLM_Calls -> core.X -> core.Chat); a fresh ``sys.modules`` does not.

TASK-13950.17 extends this file with a module-keyed ratchet on the AuthNZ modules
LLM_Calls loads, reusing ``loaded_after_import``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable

import pytest

from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT

pytestmark = pytest.mark.unit

_SCRIPT = "import importlib, json, sys; importlib.import_module(sys.argv[1]); print(json.dumps(sorted(sys.modules)))"
CASES = [
    ("tldw_Server_API.app.core.Chat", "tldw_Server_API.app.api"),
    ("tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter", "tldw_Server_API.app.core.Chat"),
    ("tldw_Server_API.app.core.LLM_Calls.gateway", "tldw_Server_API.app.core.Chat"),
]


def loaded_after_import(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter and return every module name then loaded."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, module],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    # The traceback is at the end; startup logging can make stderr long.
    assert result.returncode == 0, f"importing {module} failed:\n{result.stderr[-4000:]}"
    loaded: list[str] = json.loads(result.stdout.strip().splitlines()[-1])
    return loaded


def forbidden_loads(module: str, forbidden: str) -> list[str]:
    """The modules of package ``forbidden`` that importing ``module`` loads."""
    return [m for m in loaded_after_import(module) if m == forbidden or m.startswith(forbidden + ".")]


@pytest.mark.parametrize(("module", "forbidden"), CASES)
def test_importing_does_not_load(module: str, forbidden: str) -> None:
    loaded = forbidden_loads(module, forbidden)
    assert loaded == [], f"importing {module} loaded {loaded[:5]}"


def test_the_guard_sees_a_forbidden_load() -> None:
    # chat_target_resolution imports api.v1.schemas.chat_request_schemas: a known violator.
    assert forbidden_loads("tldw_Server_API.app.core.Chat.chat_target_resolution", "tldw_Server_API.app.api")


AUTHNZ = "tldw_Server_API.app.core.AuthNZ"
# TASK-13950.17: the AuthNZ modules the LLM gateway still loads through core.Provider_Credentials.
# Module-keyed and exact, so it fails on growth and on a stale entry, and it fails if any path
# loads an old-path alias. Shrink it as the inversion follow-up removes them.
AUTHNZ_LOADED_BY_LLM_CALLS: frozenset[str] = frozenset(
    {
        "tldw_Server_API.app.core.AuthNZ",
        "tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets",
        "tldw_Server_API.app.core.AuthNZ.audit_integrity",
        "tldw_Server_API.app.core.AuthNZ.database",
        "tldw_Server_API.app.core.AuthNZ.exceptions",
        "tldw_Server_API.app.core.AuthNZ.ip_allowlist",
        "tldw_Server_API.app.core.AuthNZ.migrations",
        "tldw_Server_API.app.core.AuthNZ.orgs_teams",
        "tldw_Server_API.app.core.AuthNZ.platform_admin",
        "tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema",
        "tldw_Server_API.app.core.AuthNZ.principal_model",
        "tldw_Server_API.app.core.AuthNZ.profile_candidate_schema",
        "tldw_Server_API.app.core.AuthNZ.profile_user_fields",
        "tldw_Server_API.app.core.AuthNZ.profile_user_write_guard",
        "tldw_Server_API.app.core.AuthNZ.repos",
        "tldw_Server_API.app.core.AuthNZ.repos._dual_backend",
        "tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo",
        "tldw_Server_API.app.core.AuthNZ.repos.datetime_utils",
        "tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo",
        "tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo",
        "tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo",
        "tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo",
        "tldw_Server_API.app.core.AuthNZ.repos.usage_repo",
        "tldw_Server_API.app.core.AuthNZ.repos.workspace_provider_installations_repo",
        "tldw_Server_API.app.core.AuthNZ.settings",
        "tldw_Server_API.app.core.AuthNZ.sqlite_profile_version_schema",
    }
)


def authnz_load_drift(loaded: Iterable[str], allowed: frozenset[str]) -> list[str]:
    now = {m for m in loaded if m == AUTHNZ or m.startswith(AUTHNZ + ".")}
    return [f"new: {m}" for m in sorted(now - allowed)] + [f"stale: {m}" for m in sorted(allowed - now)]


def test_llm_calls_loads_only_the_known_authnz_modules() -> None:
    drift = authnz_load_drift(loaded_after_import("tldw_Server_API.app.core.LLM_Calls.gateway"), AUTHNZ_LOADED_BY_LLM_CALLS)
    assert drift == [], "LLM_Calls -> AuthNZ indirect load changed (TASK-13950.17):\n" + "\n".join(drift)


def test_the_authnz_load_ratchet_sees_growth_and_stale_entries() -> None:
    allowed = frozenset({f"{AUTHNZ}.settings", f"{AUTHNZ}.gone"})
    loaded = [f"{AUTHNZ}.settings", f"{AUTHNZ}.database", "tldw_Server_API.app.core.LLM_Calls"]
    assert authnz_load_drift(loaded, allowed) == [f"new: {AUTHNZ}.database", f"stale: {AUTHNZ}.gone"]
