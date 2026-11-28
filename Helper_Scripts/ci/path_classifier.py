from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatch

_OUTBOUND_SHARED_GLOBS = [
    "Dockerfiles/Dockerfile.outbound-check",
    "Helper_Scripts/ci/outbound_*",
    "tldw_Server_API/tests/CI/test_outbound_*.py",
    "tldw_Server_API/tests/helpers/offline_network.py",
    "apps/mcp-unified/pyproject.toml",
    "apps/mcp-unified/src/**",
    "packages/tldw_profile_core/pyproject.toml",
    "packages/tldw_profile_core/src/**",
    "pyproject.toml",
    "uv.lock",
]

BACKEND_GLOBS = [
    *_OUTBOUND_SHARED_GLOBS,
    "apps/extension/entrypoints/background.ts",
    "apps/extension/wxt.config.ts",
    "apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx",
    "apps/packages/api-types/**",
    "Helper_Scripts/export_openapi_schema.py",
    "tldw_Server_API/**",
    "Helper_Scripts/ci/emit_ci_gate_flags.py",
    "Helper_Scripts/ci/merge_queue.py",
    "Helper_Scripts/ci/path_classifier.py",
    "Helper_Scripts/ci/vitest_base_ratchet.py",
    "Helper_Scripts/rag_split/**",
    "Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py",
    "tools/backlog-py/**",
    ".github/actions/**",
    ".github/workflows/**",
]

COVERAGE_GLOBS = [
    "tldw_Server_API/**",
    "pyproject.toml",
    "uv.lock",
]

OUTBOUND_BROWSER_GLOBS = [
    *_OUTBOUND_SHARED_GLOBS,
    ".github/actions/detect-required-gate-changes/action.yml",
    ".github/actions/outbound-privacy/action.yml",
    ".github/workflows/frontend-required.yml",
    "Helper_Scripts/ci/emit_ci_gate_flags.py",
    "Helper_Scripts/ci/path_classifier.py",
    "tldw_Server_API/app/**",
    "tldw_Server_API/Config_Files/**",
    "tldw_Server_API/scripts/run_server_guarded_mcp.py",
    "apps/package.json",
    "apps/bun.lock",
    "apps/patches/**",
    "apps/packages/**",
    "apps/tldw-frontend/**",
    "apps/extension/**",
]

TLDW_FRONTEND_GLOBS = [
    ".github/workflows/frontend-required.yml",
    "apps/package.json",
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
    "tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py",
    "tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py",
    "tldw_Server_API/tests/helpers/run_config_spy.py",
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
    "test_pipeline_state_release",
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
    "test_rag_run_config_caller_equivalence",
    "test_rag_transport_helpers",
    "test_rag_unified_response_mapping",
    "test_rag_unified_search_agent_defaults",
    "test_retrieval_plan_usage",
    "test_retrieval",
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

# Paths that run the required `chat-executors` job (TASK-13950.25): the golden-output test and the chat
# service tests below, only when the chat executors, their destination modules, a gate test or the job changes.
_CHAT_CORE = "tldw_Server_API/app/core/Chat/"
CHAT_EXECUTORS_GLOBS = [
    *(
        f"{_CHAT_CORE}{n}.py"
        for n in (
            "chat_service",
            "call_state",
            "completion_pipeline",
            "streaming_pipeline",
            "moderation_pipeline",
            "persistence_service",
            "response_processor",
            "tool_execution_service",
            "history_context",
            "streaming_utils",
        )
    ),
    "tldw_Server_API/tests/Chat/unit/test_chat_executors_golden.py",
    "tldw_Server_API/tests/Chat/unit/golden/**",
    "tldw_Server_API/tests/Chat/unit/test_chat_service_*.py",
    "tldw_Server_API/tests/Chat/unit/test_streaming_structured_events.py",
    "tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py",
    "tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py",
    "tldw_Server_API/tests/Chat/unit/test_history_context.py",
    "tldw_Server_API/tests/Chat/unit/test_history_selected_images.py",
    "tldw_Server_API/tests/Chat/unit/test_native_history_frozen_worldbooks.py",
    "tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py",
    "tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py",
    "tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py",
    "tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py",
    "tldw_Server_API/tests/Chat/integration/test_chat_run_first_rollout.py",
    "tldw_Server_API/tests/Chat_NEW/unit/test_chat_continuation_controls.py",
    "tldw_Server_API/tests/Chat_NEW/unit/test_chat_history_dedup.py",
    "tldw_Server_API/tests/Chat_NEW/unit/test_new_chat_title.py",
    "tldw_Server_API/tests/Chat_NEW/unit/test_tool_message_persistence.py",
    "tldw_Server_API/tests/Chat_NEW/unit/test_chat_queue_integration.py",
    "tldw_Server_API/tests/Chat_NEW/integration/test_chat_rg_refund.py",
    "tldw_Server_API/tests/Chat_NEW/integration/test_native_character_prompt_delivery.py",
    "tldw_Server_API/tests/Chat_NEW/integration/test_server_settled_generation_metadata.py",
    "tldw_Server_API/tests/DB_Management/test_chat_retry_system_messages.py",
    "tldw_Server_API/tests/DB_Management/test_chat_successful_regeneration.py",
    "tldw_Server_API/tests/DB_Management/test_chat_queued_resume_history.py",
    "tldw_Server_API/tests/Guardian/test_chat_integration.py",
    "tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py",
    "tldw_Server_API/tests/unit/test_moderation_event_capture.py",
    "Helper_Scripts/rag_split/**",
    "tldw_Server_API/tests/_plugins/postgres.py",
    ".github/workflows/backend-required.yml",
    ".github/actions/detect-required-gate-changes/action.yml",
    "Helper_Scripts/ci/path_classifier.py",
]

_T = "tldw_Server_API/tests/"
# The `chat-executors` job runs exactly these files, serially (the workflow lists the same paths literally).
CHAT_EXECUTORS_GATE_TESTS: tuple[str, ...] = (
    f"{_T}Chat/unit/test_chat_executors_golden.py",
    *(
        f"{_T}Chat/unit/{n}.py"
        for n in (
            "test_chat_service_fallback",
            "test_chat_service_fallback_credential_taxonomy",
            "test_chat_service_content",
            "test_chat_service_streaming_tool_autoexec",
            "test_chat_service_tool_autoexec",
            "test_chat_service_queue_estimate",
            "test_chat_service_audit_tasks",
            "test_streaming_structured_events",
            "test_prompt_cost_guardrails",
            "test_chat_history_and_streaming",
            "test_history_context",
            "test_history_selected_images",
            "test_native_history_frozen_worldbooks",
            "test_failed_retry_provider_order",
        )
    ),
    *(
        f"{_T}Chat/integration/{n}.py"
        for n in (
            "test_chat_endpoint_simplified",
            "test_chat_endpoint_auto_routing",
            "test_chat_endpoint_streaming_normalization",
            "test_chat_run_first_rollout",
        )
    ),
    *(
        f"{_T}Chat_NEW/unit/{n}.py"
        for n in (
            "test_chat_continuation_controls",
            "test_chat_history_dedup",
            "test_new_chat_title",
            "test_tool_message_persistence",
            "test_chat_queue_integration",
        )
    ),
    *(
        f"{_T}Chat_NEW/integration/{n}.py"
        for n in (
            "test_chat_rg_refund",
            "test_native_character_prompt_delivery",
            "test_server_settled_generation_metadata",
        )
    ),
    *(
        f"{_T}DB_Management/{n}.py"
        for n in (
            "test_chat_retry_system_messages",
            "test_chat_successful_regeneration",
            "test_chat_queued_resume_history",
        )
    ),
    f"{_T}Guardian/test_chat_integration.py",
    f"{_T}Usage/test_llm_quota_byok_calls.py",
    f"{_T}unit/test_moderation_event_capture.py",
)
# Test ids the job deselects (passed as `--deselect`), with the reason.
CHAT_EXECUTORS_GATE_DESELECT: dict[str, str] = {
    f"{_T}Chat/integration/test_chat_endpoint_simplified.py::test_chat_completion_streaming": (
        'static skip: "Streaming tests hang with TestClient"'
    ),
    f"{_T}DB_Management/test_chat_retry_system_messages.py::test_retry_context_does_not_expose_another_owners_instructions[postgres]": (
        "fails at dev with NotFoundError 'Conversation not found' under postgres:18 (TASK-15643)"
    ),
    f"{_T}DB_Management/test_chat_queued_resume_history.py::test_foreign_conversation_is_not_used_as_history_evidence[postgres]": (
        "fails at dev with NotFoundError 'Conversation not found' under postgres:18 (TASK-15643)"
    ),
}
# Chat test files the job does not run, with the reason.
CHAT_EXECUTORS_GATE_EXCLUDED: dict[str, str] = {
    f"{_T}Character_Chat_NEW/unit/test_roleplay_resume_contract.py": (
        "325 s serial run that touches the executors once, through a patch of _format_selected_history_messages; "
        "the structure pin guards that patch and the character shard runs it nightly"
    ),
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
    chat_executors_changed = any(_matches_any(path, CHAT_EXECUTORS_GLOBS) for path in normalized_paths)
    security_relevant_changed = backend_changed or any(
        path.endswith(("requirements.txt", "pyproject.toml", "uv.lock")) for path in normalized_paths
    )

    return {
        "backend_changed": backend_changed,
        "frontend_changed": frontend_changed,
        "tldw_frontend_changed": tldw_frontend_changed,
        "outbound_browser_changed": any(
            _matches_any(path, OUTBOUND_BROWSER_GLOBS) for path in normalized_paths
        ),
        "family_guardrails_changed": family_guardrails_changed,
        "admin_ui_changed": admin_ui_changed,
        "e2e_changed": e2e_changed,
        "security_relevant_changed": security_relevant_changed,
        "coverage_required": coverage_required,
        "chacha_owner_scope_changed": chacha_owner_scope_changed,
        "rag_pipeline_changed": rag_pipeline_changed,
        "chat_executors_changed": chat_executors_changed,
    }
