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
    "Helper_Scripts/rag_split/**",
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
    "tldw_Server_API/app/core/DB_Management/backends/factory.py",
    "tldw_Server_API/app/core/DB_Management/backends/base.py",
    "tldw_Server_API/app/core/DB_Management/scope_context.py",
    "tldw_Server_API/app/core/WebClipper/service.py",
    "tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py",
    "tldw_Server_API/app/core/Character_Chat/chat_dictionary.py",
    "tldw_Server_API/app/core/Chat/history_selection.py",
    "tldw_Server_API/tests/_plugins/postgres.py",
    "tldw_Server_API/tests/ChaChaNotesDB/conftest.py",
    "tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py",
    "tldw_Server_API/tests/ChaChaNotesDB/test_web_clipper_postgres_tenancy.py",
    "tldw_Server_API/tests/DB_Management/test_chacha_owner_scope_required.py",
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

# Paths that run the required `rag-pipeline` job (TASK-13950.24): the golden-output test and the
# pipeline tests below, only when RAG code, its request mapping, a gate test or the job itself changes.
RAG_PIPELINE_GLOBS = [
    "tldw_Server_API/app/core/RAG/**",
    "tldw_Server_API/app/api/v1/endpoints/rag_*.py",
    "tldw_Server_API/app/api/v1/schemas/rag_schemas*.py",
    "tldw_Server_API/tests/RAG_NEW/**",
    "tldw_Server_API/tests/RAG/**",
    "tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py",
    "tldw_Server_API/tests/_plugins/postgres.py",
    ".github/workflows/backend-required.yml",
    "Helper_Scripts/ci/path_classifier.py",
    "Helper_Scripts/rag_split/**",
    "pyproject.toml",
    "uv.lock",
]

_RAG_NEW = "tldw_Server_API/tests/RAG_NEW/"
_RAG_NEW_UNIT = [
    "test_batch_round2_flags",
    "test_evidence_chain_provider_isolation",
    "test_generation_executor",
    "test_hyde",
    "test_knowledge_source_retrieval_coverage",
    "test_learned_fusion_simple",
    "test_pipeline_generation_controls",
    "test_pipeline_post_verification_metadata",
    "test_pipeline_two_tier_gate",
    "test_post_retrieval_coordinator",
    "test_post_verifier",
    "test_pre_retrieval_clarification_metrics",
    "test_pre_retrieval_clarification_pipeline",
    "test_quick_wins_spell_check_compat",
    "test_rag_endpoint_contract_cleanup",
    "test_rag_pipeline_types",
    "test_rag_profiles",
    "test_rag_provider_credentials",
    "test_rag_result_mapping_core",
    "test_rag_transport_helpers",
    "test_rag_unified_response_mapping",
    "test_rag_unified_search_agent_defaults",
    "test_retrieval_plan_usage",
    "test_retrieval",
    "test_run_config_caller_equivalence",
    "test_search_depth_explicit_overrides",
    "test_slides_source_retrieval_hardening",
    "test_source_contract",
    "test_standard_core_contract_threading",
    "test_streaming_executor",
    "test_strict_extractive_and_citations",
    "test_temporal_filters",
    "test_unified_pipeline_decomposition",
    "test_unified_pipeline_focused",
    "test_unified_pipeline_golden",
    "test_unified_pipeline_profile_metadata",
    "test_unified_pipeline_query_expansion",
    "test_unified_pipeline_security_cache_helpers",
    "test_unified_pipeline_structured_writer",
    "test_unified_pipeline",
]
_RAG_NEW_INTEGRATION = [
    "test_rag_ablate_api",
    "test_rag_batch_checkpoint_api",
    "test_rag_batch_resume_api",
    "test_rag_convenience_endpoint_credentials",
    "test_rag_doc_researcher_api",
    "test_rag_integration",
    "test_rag_request_contract_parity",
    "test_rag_stream_parity",
    "test_rag_strict_extractive_nli_api",
]
_RAG_OLD = [
    "test_unified_pipeline_document_serialization",
    "test_unified_schema_and_pipeline",
    "test_doc_researcher_features",
    "test_knowledge_qa_live_regressions",
    "test_restricted_postgres_media_retrieval",
    "test_sql_retriever_integration",
]
# The `rag-pipeline` job runs exactly these files, serially (the workflow lists the same paths literally).
RAG_PIPELINE_GATE_TESTS: tuple[str, ...] = (
    *(f"{_RAG_NEW}integration/{n}.py" for n in _RAG_NEW_INTEGRATION),
    f"{_RAG_NEW}property/test_rag_properties.py",
    f"{_RAG_NEW}test_prf_and_spans.py",
    *(f"{_RAG_NEW}unit/{n}.py" for n in _RAG_NEW_UNIT),
    *(f"tldw_Server_API/tests/RAG/{n}.py" for n in _RAG_OLD),
    "tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py",
)
# Pipeline test files the job does not run, with the reason.
RAG_PIPELINE_GATE_EXCLUDED: dict[str, str] = {
    f"{_RAG_NEW}test_unified_pipeline.py": (
        "4 tests attempt live DNS to api.openai.com (teardown errors) and 3 are live_network_baseline entries"
    ),
    f"{_RAG_NEW}integration/test_rag_agentic_api.py": "it skips on HTTP 500",
    f"{_RAG_NEW}integration/test_rag_benchmarks_ablation.py": "it skips without TestClient lifespan",
}

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
    rag_pipeline_changed = any(_matches_any(path, RAG_PIPELINE_GLOBS) for path in normalized_paths)
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
        "rag_pipeline_changed": rag_pipeline_changed,
    }
