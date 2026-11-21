from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatch

BACKEND_GLOBS = [
    "apps/packages/api-types/**",
    "Helper_Scripts/export_openapi_schema.py",
    "tldw_Server_API/**",
    "Helper_Scripts/ci/emit_ci_gate_flags.py",
    "Helper_Scripts/ci/merge_queue.py",
    "Helper_Scripts/ci/path_classifier.py",
    "Helper_Scripts/ci/vitest_base_ratchet.py",
    "tools/backlog-py/**",
    "pyproject.toml",
    "uv.lock",
    ".github/actions/**",
    ".github/workflows/**",
]

COVERAGE_GLOBS = [
    "tldw_Server_API/**",
    "pyproject.toml",
    "uv.lock",
]

TLDW_FRONTEND_GLOBS = [
    "apps/packages/api-types/**",
    "apps/tldw-frontend/**",
    "apps/packages/ui/**",
    "apps/extension/**",
    "apps/bun.lock",
    "apps/tldw-frontend/package-lock.json",
    "Helper_Scripts/ci/vitest_base_ratchet.py",
    "Helper_Scripts/ci/vitest_deterministic.config.ts",
    "Helper_Scripts/ci/vitest_execution_order_reporter.mjs",
]

# Paths that run the required `chacha-owner-isolation` job (TASK-13950.23). The marked test files are
# listed explicitly; test_license_first_workflow_contracts.py checks the list equals the marked set.
CHACHA_OWNER_SCOPE_GLOBS = [
    "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
    "tldw_Server_API/app/core/DB_Management/chacha/**",
    "tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py",
    "tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py",
    "tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py",
    "tldw_Server_API/tests/_plugins/postgres.py",
    "tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py",
    "tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_postgres_tenancy.py",
    "tldw_Server_API/tests/DB_Management/test_chacha_study_tables_rls.py",
    "tldw_Server_API/tests/DB_Management/test_core_chat_owner_isolation.py",
    "tldw_Server_API/tests/DB_Management/test_core_chat_rls_postgres.py",
    "tldw_Server_API/tests/DB_Management/test_local_keyword_merge_survivor.py",
    "tldw_Server_API/tests/DB_Management/test_note_shared_owner_contract.py",
    "tldw_Server_API/tests/DB_Management/test_study_suggestion_owner_scope.py",
    ".github/workflows/backend-required.yml",
    "Helper_Scripts/ci/path_classifier.py",
    "pyproject.toml",
    "uv.lock",
]

FAMILY_GUARDRAILS_GLOBS = [
    "apps/tldw-frontend/**/family-guardrails*",
    "apps/tldw-frontend/**/option-family-guardrails-wizard*",
    "apps/packages/ui/**/family-guardrails*",
    "apps/packages/ui/**/FamilyGuardrailsWizard*",
]

ADMIN_UI_GLOBS = [
    "admin-ui/**",
    "Helper_Scripts/ci/vitest_base_ratchet.py",
    "Helper_Scripts/ci/vitest_execution_order_reporter.mjs",
]

FRONTEND_GLOBS = [
    *TLDW_FRONTEND_GLOBS,
    *ADMIN_UI_GLOBS,
]

E2E_BACKEND_GLOBS = [
    "tldw_Server_API/app/api/v1/endpoints/**",
    "tldw_Server_API/app/api/v1/schemas/**",
    "tldw_Server_API/app/api/v1/utils/**",
    "tldw_Server_API/app/core/Sandbox/default_service.py",
    "tldw_Server_API/app/core/LLM_Calls/model_metadata.py",
    "tldw_Server_API/app/core/Embeddings/media_embeddings.py",
    "tldw_Server_API/app/core/Prompt_Management/prompt_studio/ws_connections.py",
    "tldw_Server_API/app/core/AuthNZ/**",
    "tldw_Server_API/app/core/Provider_Credentials/**",
    "tldw_Server_API/app/core/Usage/llm_budget_*.py",
    "tldw_Server_API/app/services/authnz_scheduler.py",
]


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def classify_paths(paths: Iterable[str]) -> dict[str, bool]:
    normalized_paths = list(paths)
    backend_changed = any(_matches_any(path, BACKEND_GLOBS) for path in normalized_paths)
    coverage_required = any(_matches_any(path, COVERAGE_GLOBS) for path in normalized_paths)
    tldw_frontend_changed = any(_matches_any(path, TLDW_FRONTEND_GLOBS) for path in normalized_paths)
    family_guardrails_changed = any(
        _matches_any(path, FAMILY_GUARDRAILS_GLOBS) for path in normalized_paths
    )
    admin_ui_changed = any(_matches_any(path, ADMIN_UI_GLOBS) for path in normalized_paths)
    frontend_changed = any(_matches_any(path, FRONTEND_GLOBS) for path in normalized_paths)
    e2e_changed = tldw_frontend_changed or any(_matches_any(path, E2E_BACKEND_GLOBS) for path in normalized_paths)
    chacha_owner_scope_changed = any(
        _matches_any(path, CHACHA_OWNER_SCOPE_GLOBS) for path in normalized_paths
    )
    security_relevant_changed = backend_changed or any(
        path.endswith(("requirements.txt", "pyproject.toml", "uv.lock")) for path in normalized_paths
    )

    return {
        "backend_changed": backend_changed,
        "frontend_changed": frontend_changed,
        "tldw_frontend_changed": tldw_frontend_changed,
        "family_guardrails_changed": family_guardrails_changed,
        "admin_ui_changed": admin_ui_changed,
        "e2e_changed": e2e_changed,
        "security_relevant_changed": security_relevant_changed,
        "coverage_required": coverage_required,
        "chacha_owner_scope_changed": chacha_owner_scope_changed,
    }
