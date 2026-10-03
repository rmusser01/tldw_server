"""Ratchet: private scalar/env coercers may only decrease (TASK-13342 stage 4).

core/Utils/coercion.py is the one truthy/int/env contract (TASK-13322). The design
(Docs/Design/2026-09-21-scalar-and-env-coercion-consolidation-design.md) adopts it by
ratchet, not by mass refactor: 200+ private copies across security-relevant switches
cannot be rewritten in one reviewable change, and a sweeping rewrite is the change most
likely to introduce the next fail-open. So today's count per module is frozen here and
may only go down as modules migrate on their own schedule.

Counted: function definitions named exactly like the three clusters the design
measured -- BOOL_NAMES, INT_NAMES and ENV_NAMES below. Excluded as justified divergence
per the design:
Chunking option parsing, per-provider payload coercion, the Sync_DB row mappers, and
the AuthNZ repos' driver-output coercers. api/v1/** is counted but migrated by the
owner only.

AST-derived, like the sibling ratchets, so docstrings and comments never count.
Seed: 212 definitions in 175 modules.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"

BOOL_NAMES = frozenset({"_truthy", "_is_truthy", "_coerce_bool", "_as_bool", "_to_bool"})
INT_NAMES = frozenset({"_coerce_int", "_safe_int", "_as_int", "_to_int"})
ENV_NAMES = frozenset({"_env_bool", "_env_int", "_env_flag", "_env_str"})
COERCER_NAMES = BOOL_NAMES | INT_NAMES | ENV_NAMES

EXCLUDED = (
    "tldw_Server_API/app/core/Chunking/",
    "tldw_Server_API/app/core/LLM_Calls/providers/",
    "tldw_Server_API/app/core/AuthNZ/repos/",
    "tldw_Server_API/app/core/DB_Management/Sync_DB.py",
)

# Per-module counts; entries may only be lowered or removed.
PRIVATE_COERCER_BASELINE: dict[str, int] = {
    "tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py": 3,
    "tldw_Server_API/app/api/v1/endpoints/audit.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/chat.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/config_info.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/discord_support.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/integrations_control_plane.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/jobs_admin.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/llm_providers.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/media/navigation.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/media_embeddings.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/persona.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py": 2,
    "tldw_Server_API/app/api/v1/endpoints/resource_governor.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/sharing.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/slack_support.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/telegram_support.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/watchlists.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/workflows.py": 1,
    "tldw_Server_API/app/api/v1/endpoints/writing.py": 1,
    "tldw_Server_API/app/core/Agent_Client_Protocol/hardening.py": 1,
    "tldw_Server_API/app/core/Audio/tokenizer_service.py": 1,
    "tldw_Server_API/app/core/Audit/audit_shared_migration.py": 1,
    "tldw_Server_API/app/core/Audit/unified_audit_service.py": 3,
    "tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py": 1,
    "tldw_Server_API/app/core/AuthNZ/settings.py": 1,
    "tldw_Server_API/app/core/Character_Chat/character_limits.py": 1,
    "tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py": 1,
    "tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py": 2,
    "tldw_Server_API/app/core/Chat/chat_loop_engine.py": 1,
    "tldw_Server_API/app/core/Chat/chat_service.py": 2,
    "tldw_Server_API/app/core/Chat_Macros/parser.py": 1,
    "tldw_Server_API/app/core/Chatbooks/jobs_adapter.py": 1,
    "tldw_Server_API/app/core/Chatbooks/quota_manager.py": 2,
    "tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py": 1,
    "tldw_Server_API/app/core/Claims_Extraction/budget_guard.py": 1,
    "tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py": 1,
    "tldw_Server_API/app/core/Collections/embedding_queue.py": 1,
    "tldw_Server_API/app/core/Collections/reading_digest_jobs.py": 2,
    "tldw_Server_API/app/core/Collections/reading_import_jobs.py": 2,
    "tldw_Server_API/app/core/Collections/reading_importers.py": 1,
    "tldw_Server_API/app/core/Collections/reading_service.py": 1,
    "tldw_Server_API/app/core/DB_Management/Workflows_DB.py": 1,
    "tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py": 1,
    "tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py": 1,
    "tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py": 1,
    "tldw_Server_API/app/core/Embeddings/chunk_metadata_backfill.py": 1,
    "tldw_Server_API/app/core/Embeddings/jobs_adapter.py": 1,
    "tldw_Server_API/app/core/Embeddings/redis_pipeline.py": 2,
    "tldw_Server_API/app/core/Embeddings/services/jobs_worker.py": 2,
    "tldw_Server_API/app/core/Embeddings/services/redis_worker.py": 1,
    "tldw_Server_API/app/core/Embeddings/simplified_config.py": 1,
    "tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py": 1,
    "tldw_Server_API/app/core/External_Sources/connectors_service.py": 1,
    "tldw_Server_API/app/core/Flashcards/apkg_importer.py": 1,
    "tldw_Server_API/app/core/Governance/service.py": 1,
    "tldw_Server_API/app/core/Image_Generation/config.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Buffered_Transcription.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Custom_Vocabulary.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_VibeVoice.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/chatllm_ocr.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/deepseek_ocr.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/llamacpp_ocr.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/nemotron_parse.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/registry.py": 1,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/mineru_adapter.py": 2,
    "tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py": 1,
    "tldw_Server_API/app/core/Ingestion_Sources/access_policy.py": 1,
    "tldw_Server_API/app/core/Jobs/manager.py": 1,
    "tldw_Server_API/app/core/Jobs/queue_stats.py": 1,
    "tldw_Server_API/app/core/Jobs/settings.py": 2,
    "tldw_Server_API/app/core/LLM_Calls/cache_intents.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/extra_body_compat_catalog.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/llamacpp_request_extensions.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/local_cache_diagnostics.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/provider_readiness.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/routing/candidate_pool.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/routing/runtime.py": 1,
    "tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py": 2,
    "tldw_Server_API/app/core/Logging/system_log_buffer.py": 1,
    "tldw_Server_API/app/core/MCP_unified/protocol.py": 1,
    "tldw_Server_API/app/core/MCP_unified/server.py": 1,
    "tldw_Server_API/app/core/MCP_unified/tool_execution/security.py": 1,
    "tldw_Server_API/app/core/Metrics/telemetry.py": 1,
    "tldw_Server_API/app/core/Monitoring/notification_service.py": 1,
    "tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py": 1,
    "tldw_Server_API/app/core/Notes_Graph/graph_cache.py": 1,
    "tldw_Server_API/app/core/Notes_Graph/graph_service.py": 2,
    "tldw_Server_API/app/core/Persona/session_materialization.py": 1,
    "tldw_Server_API/app/core/Persona/visual_import_preview_validators.py": 1,
    "tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py": 1,
    "tldw_Server_API/app/core/PrivilegeMaps/cache.py": 1,
    "tldw_Server_API/app/core/PrivilegeMaps/service.py": 1,
    "tldw_Server_API/app/core/Prompt_Management/prompt_studio/mcts_optimizer.py": 1,
    "tldw_Server_API/app/core/Prompt_Management/prompt_studio/services/jobs_worker.py": 1,
    "tldw_Server_API/app/core/Prototype_Workspaces/access.py": 1,
    "tldw_Server_API/app/core/Prototype_Workspaces/preview_broker.py": 1,
    "tldw_Server_API/app/core/RAG/block_to_chunks.py": 1,
    "tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py": 1,
    "tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py": 1,
    "tldw_Server_API/app/core/Research/providers/config.py": 1,
    "tldw_Server_API/app/core/Sandbox/macos_diagnostics.py": 1,
    "tldw_Server_API/app/core/Sandbox/network_policy.py": 1,
    "tldw_Server_API/app/core/Sandbox/operator_evidence.py": 1,
    "tldw_Server_API/app/core/Sandbox/operator_status.py": 1,
    "tldw_Server_API/app/core/Sandbox/runners/docker_runner.py": 1,
    "tldw_Server_API/app/core/Sandbox/runners/firecracker_runner.py": 1,
    "tldw_Server_API/app/core/Sandbox/runners/lima_enforcer.py": 1,
    "tldw_Server_API/app/core/Sandbox/runners/lima_runner.py": 1,
    "tldw_Server_API/app/core/Sandbox/runners/seatbelt_runner.py": 1,
    "tldw_Server_API/app/core/Sandbox/runners/vz_common.py": 1,
    "tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_rag_adapter.py": 1,
    "tldw_Server_API/app/core/Security/middleware.py": 1,
    "tldw_Server_API/app/core/Setup/first_run_mcp_tools.py": 1,
    "tldw_Server_API/app/core/Setup/install_manager.py": 1,
    "tldw_Server_API/app/core/Setup/readiness_service.py": 1,
    "tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py": 1,
    "tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py": 1,
    "tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py": 1,
    "tldw_Server_API/app/core/Sharing/unified_share_audit.py": 1,
    "tldw_Server_API/app/core/Slides/slides_export.py": 3,
    "tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py": 1,
    "tldw_Server_API/app/core/TTS/adapters/echo_tts_adapter.py": 2,
    "tldw_Server_API/app/core/TTS/adapters/luxtts_adapter.py": 1,
    "tldw_Server_API/app/core/TTS/adapters/pocket_tts_adapter.py": 1,
    "tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_remote.py": 1,
    "tldw_Server_API/app/core/TTS/adapters/qwen3_tts_adapter.py": 1,
    "tldw_Server_API/app/core/TTS/tts_resource_manager.py": 1,
    "tldw_Server_API/app/core/TTS/tts_service_v2.py": 1,
    "tldw_Server_API/app/core/TTS/tts_validation.py": 1,
    "tldw_Server_API/app/core/Templating/template_renderer.py": 1,
    "tldw_Server_API/app/core/Utils/Utils.py": 1,
    "tldw_Server_API/app/core/Utils/torch_import_guard.py": 1,
    "tldw_Server_API/app/core/VN_Assets/concurrency.py": 1,
    "tldw_Server_API/app/core/Watchlists/briefing_contract.py": 1,
    "tldw_Server_API/app/core/Watchlists/fetchers.py": 1,
    "tldw_Server_API/app/core/Watchlists/report_evidence.py": 2,
    "tldw_Server_API/app/core/Watchlists/template_store.py": 1,
    "tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py": 1,
    "tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py": 1,
    "tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py": 2,
    "tldw_Server_API/app/core/Web_Scraping/extraction/pipeline.py": 1,
    "tldw_Server_API/app/core/Web_Scraping/extraction/strategies/cluster.py": 1,
    "tldw_Server_API/app/core/Workflows/adapters/integration/messaging.py": 2,
    "tldw_Server_API/app/core/Workflows/adapters/rag/search.py": 1,
    "tldw_Server_API/app/core/Workspaces/status_projection.py": 2,
    "tldw_Server_API/app/core/config.py": 6,
    "tldw_Server_API/app/core/startup_preflight.py": 1,
    "tldw_Server_API/app/services/admin_router_analytics_service.py": 2,
    "tldw_Server_API/app/services/chat_macros_jobs_worker.py": 1,
    "tldw_Server_API/app/services/enhanced_web_scraping_service.py": 2,
    "tldw_Server_API/app/services/jobs_metrics_service.py": 1,
    "tldw_Server_API/app/services/jobs_prune_scheduler.py": 1,
    "tldw_Server_API/app/services/loop_lag_watchdog.py": 2,
    "tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py": 1,
    "tldw_Server_API/app/services/mcp_hub_tool_registry.py": 1,
    "tldw_Server_API/app/services/media_ingest_jobs_worker.py": 1,
    "tldw_Server_API/app/services/meetings_webhook_dlq_service.py": 1,
    "tldw_Server_API/app/services/presentation_render_jobs_worker.py": 1,
    "tldw_Server_API/app/services/scheduled_task_recurring_question_scheduler.py": 1,
    "tldw_Server_API/app/services/setup_mcp_tools_service.py": 1,
    "tldw_Server_API/app/services/startup_recurring_schedulers.py": 1,
    "tldw_Server_API/app/services/telegram_delivery_service.py": 1,
    "tldw_Server_API/app/services/telegram_execution_identity_service.py": 1,
    "tldw_Server_API/app/services/workflows_db_maintenance.py": 1,
    "tldw_Server_API/app/services/workflows_webhook_dlq_service.py": 1,
}


def count_private_coercers(source: str, filename: str = "<memory>") -> int:
    """Return how many function definitions in ``source`` use a counted coercer name."""
    tree = ast.parse(source, filename=filename)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COERCER_NAMES
    )


def ratchet_violations(counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Describe every module whose count differs from its baseline entry."""
    problems: list[str] = []
    for module in sorted(set(counts) | set(baseline)):
        now, seeded = counts.get(module, 0), baseline.get(module, 0)
        if now > seeded:
            problems.append(
                f"{module}: {now} private coercer(s), baseline {seeded}. Use "
                "core/Utils/coercion (parse_bool / env_bool / ...) instead of a new private copy."
            )
        elif now < seeded:
            problems.append(
                f"{module}: {now} private coercer(s), baseline {seeded}. Migrated one? "
                "Lower its entry in PRIVATE_COERCER_BASELINE so the ratchet stays tight."
            )
    return problems


def _current_counts() -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in sorted(APP_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(EXCLUDED):
            continue
        found = count_private_coercers(path.read_text(encoding="utf-8"), rel)
        if found:
            counts[rel] = found
    return dict(counts)


def test_private_coercers_only_decrease() -> None:
    problems = ratchet_violations(_current_counts(), PRIVATE_COERCER_BASELINE)
    assert not problems, "\n".join(problems)


def test_the_ratchet_rejects_a_new_private_coercer() -> None:
    """Self-test: a new _as_bool in an unlisted module must fail, not pass silently."""
    new_module = "tldw_Server_API/app/core/Example/new_feature.py"
    source = "def _as_bool(value):\n    return str(value).lower() in {'1', 'true'}\n"
    counts = {**PRIVATE_COERCER_BASELINE, new_module: count_private_coercers(source)}

    problems = ratchet_violations(counts, PRIVATE_COERCER_BASELINE)

    assert len(problems) == 1 and problems[0].startswith(new_module), problems


def test_the_ratchet_asks_for_a_lower_seed_after_a_migration() -> None:
    """Self-test: removing a coercer without lowering the seed must also fail."""
    module, seeded = next(iter(PRIVATE_COERCER_BASELINE.items()))
    counts = {**PRIVATE_COERCER_BASELINE, module: seeded - 1}

    problems = ratchet_violations(counts, PRIVATE_COERCER_BASELINE)

    assert len(problems) == 1 and "Lower its entry" in problems[0], problems


def test_docstrings_and_comments_are_not_counted() -> None:
    source = '"""def _as_bool(x): ..."""\n# def _to_int(x): ...\nx = "_env_flag"\n'
    assert count_private_coercers(source) == 0
