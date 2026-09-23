"""Ratchet: core/ must not grow new dependencies on the api/ layer.

Docs/Architecture.md states the direction:

    Clients -> FastAPI endpoints -> Core domain services -> Databases / ...

and "keep endpoints thin, push logic into core modules". A core module importing
from app/api/ inverts that. The 2026-09-21 core-module review measured the current
state and found two distinct shapes, which this test tracks separately because they
are not equally bad:

* Schema-only imports from ``api/v1/schemas/*`` -- the common case. The honest reading
  is that those schemas are in the wrong package, not that core is wrong to need them.
* True inversions: core importing ``api/v1/endpoints/*`` or ``api/v1/API_Deps/*``.
  ``Ingestion_Media_Processing/persistence.py`` is the sharpest instance -- four of its
  five such imports exist only so tests can monkeypatch ``endpoints.media.*``, so
  production resolves collaborators via ``getattr`` on an API module at request time.

This is a RATCHET, not a ban: the existing entries are frozen and may only be removed.
Adding a core module to either set fails. Deleting one requires updating the baseline
below, which is the point -- the number can only go down.

Companion to tests/lint/test_endpoint_auth_deps_import_boundary.py, which guards the
OPPOSITE direction. Note those two currently describe a real cycle: that test bans
endpoints from importing core.AuthNZ.User_DB_Handling, while
core/AuthNZ/User_DB_Handling.py imports oauth2_scheme from api/v1/API_Deps.

Baselines are AST-derived, not grep-derived: a grep over-counts by one here, matching a
usage example inside a docstring in core/DB_Management/db_errors.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core"

CORE_TO_API_BASELINE = frozenset({
    "tldw_Server_API/app/core/Admin_Webhooks/audit.py",
    "tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py",
    "tldw_Server_API/app/core/Audio/Realtime/default_pipeline.py",
    "tldw_Server_API/app/core/Audio_Studio/providers/speech.py",
    "tldw_Server_API/app/core/Audiobooks/alignment_utils.py",
    "tldw_Server_API/app/core/Audiobooks/subtitle_generator.py",
    "tldw_Server_API/app/core/Audiobooks/tag_parser.py",
    "tldw_Server_API/app/core/Audit/unified_audit_service.py",
    "tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py",
    "tldw_Server_API/app/core/AuthNZ/api_key_audit.py",
    "tldw_Server_API/app/core/Buddy/service.py",
    "tldw_Server_API/app/core/Buddy/turns.py",
    "tldw_Server_API/app/core/Chat/chat_helpers.py",
    "tldw_Server_API/app/core/Chat/chat_loop_store.py",
    "tldw_Server_API/app/core/Chat/chat_service.py",
    "tldw_Server_API/app/core/Chat/chat_target_resolution.py",
    "tldw_Server_API/app/core/Chat/command_router.py",
    "tldw_Server_API/app/core/Chat_Macros/jobs.py",
    "tldw_Server_API/app/core/Chunking/auto_boundary_assistant.py",
    "tldw_Server_API/app/core/Claims_Extraction/__init__.py",
    "tldw_Server_API/app/core/Claims_Extraction/claims_service.py",
    "tldw_Server_API/app/core/Collections/reading_service.py",
    "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py",
    "tldw_Server_API/app/core/DB_Management/Evaluations_DB.py",
    "tldw_Server_API/app/core/Data_Tables/jobs_worker.py",
    "tldw_Server_API/app/core/Embeddings/audit_adapter.py",
    "tldw_Server_API/app/core/Embeddings/services/jobs_worker.py",
    "tldw_Server_API/app/core/Evaluations/audit_adapter.py",
    "tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py",
    "tldw_Server_API/app/core/Evaluations/embeddings_abtest_runner.py",
    "tldw_Server_API/app/core/Evaluations/embeddings_abtest_service.py",
    "tldw_Server_API/app/core/Evaluations/recipe_runs_jobs.py",
    "tldw_Server_API/app/core/Evaluations/recipe_runs_jobs_worker.py",
    "tldw_Server_API/app/core/Evaluations/recipe_runs_service.py",
    "tldw_Server_API/app/core/Evaluations/recipes/base.py",
    "tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py",
    "tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py",
    "tldw_Server_API/app/core/Evaluations/recipes/persona_dialogue_tree_robustness.py",
    "tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality.py",
    "tldw_Server_API/app/core/Evaluations/recipes/rag_retrieval_tuning.py",
    "tldw_Server_API/app/core/Evaluations/recipes/rag_retrieval_tuning_execution.py",
    "tldw_Server_API/app/core/Evaluations/recipes/registry.py",
    "tldw_Server_API/app/core/Evaluations/recipes/reporting.py",
    "tldw_Server_API/app/core/Evaluations/recipes/summarization_quality.py",
    "tldw_Server_API/app/core/Evaluations/synthetic_eval_generation.py",
    "tldw_Server_API/app/core/Evaluations/synthetic_eval_repository.py",
    "tldw_Server_API/app/core/Evaluations/synthetic_eval_service.py",
    "tldw_Server_API/app/core/File_Artifacts/file_artifacts_service.py",
    "tldw_Server_API/app/core/File_Artifacts/jobs_worker.py",
    "tldw_Server_API/app/core/Flashcards/generation.py",
    "tldw_Server_API/app/core/Flashcards/source_review.py",
    "tldw_Server_API/app/core/Ingestion_Media_Processing/document_upload_preflight.py",
    "tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py",
    "tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_jobs.py",
    "tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py",
    "tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py",
    "tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py",
    "tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py",
    "tldw_Server_API/app/core/MCP_unified/tests/support.py",
    "tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py",
    "tldw_Server_API/app/core/MCP_unified/tests/test_http_validation_bounds.py",
    "tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py",
    "tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py",
    "tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py",
    "tldw_Server_API/app/core/MCP_unified/tests/test_refresh_token.py",
    "tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py",
    "tldw_Server_API/app/core/Notes_Graph/formatters.py",
    "tldw_Server_API/app/core/Notes_Graph/graph_service.py",
    "tldw_Server_API/app/core/Persona/archetype_loader.py",
    "tldw_Server_API/app/core/Persona/live_tts.py",
    "tldw_Server_API/app/core/PrivilegeMaps/service.py",
    "tldw_Server_API/app/core/PrivilegeMaps/snapshots.py",
    "tldw_Server_API/app/core/Prompt_Management/prompt_improvement_dispatch.py",
    "tldw_Server_API/app/core/Prompt_Management/prompt_studio/mcts_optimizer.py",
    "tldw_Server_API/app/core/RAG/rag_service/response_mapping.py",
    "tldw_Server_API/app/core/RAG/rag_service/source_health.py",
    "tldw_Server_API/app/core/RAG/rag_service/transport.py",
    "tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py",
    "tldw_Server_API/app/core/Research/service.py",
    "tldw_Server_API/app/core/Research/streaming.py",
    "tldw_Server_API/app/core/Research_Workspace/capabilities.py",
    "tldw_Server_API/app/core/Research_Workspace/output_jobs.py",
    "tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_rag_adapter.py",
    "tldw_Server_API/app/core/Setup/install_manager.py",
    "tldw_Server_API/app/core/Setup/provider_catalog.py",
    "tldw_Server_API/app/core/Setup/provider_validation.py",
    "tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py",
    "tldw_Server_API/app/core/Sharing/shared_workspace_clone_operations.py",
    "tldw_Server_API/app/core/Skills/skill_executor.py",
    "tldw_Server_API/app/core/Streaming/speech_chat_service.py",
    "tldw_Server_API/app/core/StudyPacks/generation_service.py",
    "tldw_Server_API/app/core/StudyPacks/jobs.py",
    "tldw_Server_API/app/core/Sync/v2/personal_context_ongoing_contract.py",
    "tldw_Server_API/app/core/TTS/realtime_session.py",
    "tldw_Server_API/app/core/TTS/tts_jobs_worker.py",
    "tldw_Server_API/app/core/TTS/tts_service_v2.py",
    "tldw_Server_API/app/core/VN_Assets/preflight.py",
    "tldw_Server_API/app/core/VN_Assets/service.py",
    "tldw_Server_API/app/core/VN_Play/setup_options.py",
    "tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py",
    "tldw_Server_API/app/core/Workflows/adapters/audio/tts.py",
    "tldw_Server_API/app/core/Workflows/adapters/llm/llm.py",
    "tldw_Server_API/app/core/Workspaces/assistant_defaults.py",
})

TRUE_INVERSION_BASELINE = frozenset({
    "tldw_Server_API/app/core/Admin_Webhooks/audit.py",
    "tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py",
    "tldw_Server_API/app/core/Audit/unified_audit_service.py",
    "tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py",
    "tldw_Server_API/app/core/AuthNZ/api_key_audit.py",
    "tldw_Server_API/app/core/Chat/chat_service.py",
    "tldw_Server_API/app/core/Chat/command_router.py",
    "tldw_Server_API/app/core/Chat_Macros/jobs.py",
    "tldw_Server_API/app/core/Embeddings/audit_adapter.py",
    "tldw_Server_API/app/core/Embeddings/services/jobs_worker.py",
    "tldw_Server_API/app/core/Evaluations/audit_adapter.py",
    "tldw_Server_API/app/core/Evaluations/embeddings_abtest_runner.py",
    "tldw_Server_API/app/core/Evaluations/embeddings_abtest_service.py",
    "tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py",
    "tldw_Server_API/app/core/Prompt_Management/prompt_studio/mcts_optimizer.py",
    "tldw_Server_API/app/core/Research_Workspace/capabilities.py",
    "tldw_Server_API/app/core/Setup/install_manager.py",
    "tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py",
})


def _imported_modules(tree: ast.Module) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _is_api_module(module: str) -> bool:
    return module.startswith("tldw_Server_API.app.api") or ".app.api." in f".{module}."


def _is_true_inversion(module: str) -> bool:
    marker = "app.api.v1."
    if marker not in module:
        return False
    tail = module.split(marker, 1)[1]
    return tail.startswith(("endpoints", "API_Deps"))


def _scan() -> tuple[set[str], set[str]]:
    any_api: set[str] = set()
    inversions: set[str] = set()
    for path in sorted(CORE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for module in _imported_modules(tree):
            if not _is_api_module(module):
                continue
            any_api.add(rel)
            # The in-app MCP test tree is test code, not production core.
            if _is_true_inversion(module) and "/tests/" not in rel:
                inversions.add(rel)
    return any_api, inversions


def test_no_new_core_to_api_imports() -> None:
    found, _ = _scan()
    added = sorted(found - CORE_TO_API_BASELINE)
    assert not added, (
        "New core -> api imports. core/ must not depend on app/api/ "
        "(Docs/Architecture.md). If the import is a schema, move the schema into core "
        "rather than adding it here:\n  " + "\n  ".join(added)
    )


def test_no_new_true_inversions() -> None:
    _, found = _scan()
    added = sorted(found - TRUE_INVERSION_BASELINE)
    assert not added, (
        "New core -> api/v1/{endpoints,API_Deps} import. This is the worst shape of the "
        "layering inversion: core reaching into the transport layer. Inject the "
        "dependency instead:\n  " + "\n  ".join(added)
    )


def test_baselines_only_shrink() -> None:
    """Fixing a violation must update the baseline, so the number cannot drift up."""
    found_any, found_inv = _scan()
    stale_any = sorted(CORE_TO_API_BASELINE - found_any)
    stale_inv = sorted(TRUE_INVERSION_BASELINE - found_inv)
    assert not stale_any and not stale_inv, (
        "Baseline entries no longer import from api/ -- remove them from the frozensets "
        "in this file so the ratchet tightens:\n  "
        + "\n  ".join(stale_any + stale_inv)
    )
