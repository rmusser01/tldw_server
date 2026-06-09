"""Content router group — media, RAG, embeddings, and content processing endpoints.

These routers handle content ingestion, search, retrieval, and
related operations.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.conditional import (
    ImportedRouterSpec,
    append_imported_router_spec,
)
from tldw_Server_API.app.api.v1.router_groups.factories import evaluations_router_factory
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec
from tldw_Server_API.app.core.testing import audio_imports_enabled_for_runtime
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime as _is_explicit_pytest_runtime

API_V1_PREFIX = "/api/v1"


def iter_content_router_specs() -> Iterable[RouterSpec]:
    """Yield content/media-focused router specs."""
    specs: list[RouterSpec] = []

    # RAG unified endpoints (router has its own /api/v1/rag prefix)
    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.rag_unified",
            log_name="rag_unified",
            tags=("rag-unified",),
            route_key="rag-unified",
        ),
    )

    # RAG health and research discovery endpoints
    for discovery_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.rag_health",
            log_name="rag_health",
            tags=("rag-health",),
            route_key="rag-health",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.research",
            log_name="research",
            prefix=f"{API_V1_PREFIX}/research",
            tags=("research",),
            route_key="research",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.research_runs",
            log_name="research_runs",
            prefix=f"{API_V1_PREFIX}",
            tags=("research-runs",),
            route_key="research-runs",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.research_workspace",
            log_name="research_workspace",
            prefix=f"{API_V1_PREFIX}",
            tags=("research-workspace",),
            route_key="research-workspace",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.paper_search",
            log_name="paper_search",
            prefix=f"{API_V1_PREFIX}/paper-search",
            tags=("paper-search",),
            route_key="paper-search",
        ),
    ):
        append_imported_router_spec(specs, discovery_spec)

    # Embedding routers
    for processing_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced",
            log_name="embeddings",
            prefix=f"{API_V1_PREFIX}",
            tags=("embeddings",),
            route_key="embeddings",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.media_embeddings",
            log_name="media_embeddings",
            prefix=f"{API_V1_PREFIX}",
            tags=("media-embeddings",),
            route_key="media-embeddings",
        ),
    ):
        append_imported_router_spec(specs, processing_spec)

    # Evaluations and OCR are lazy so route policy can disable them before
    # importing modules with heavier optional dependencies.
    specs.append(RouterSpec(
        router=evaluations_router_factory,
        prefix=f"{API_V1_PREFIX}",
        tags=("evaluations",),
        route_key="evaluations",
    ))

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.ocr",
            log_name="ocr",
            prefix=f"{API_V1_PREFIX}",
            tags=("ocr",),
            route_key="ocr",
        ),
    )

    # Media endpoints
    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.media",
            log_name="media",
            prefix=f"{API_V1_PREFIX}/media",
            tags=("media",),
            route_key="media",
        ),
    )
    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs",
            log_name="media_ingest_jobs",
            prefix=f"{API_V1_PREFIX}/media",
            tags=("media",),
            route_key="media-ingest-jobs",
        ),
    )

    # Audio endpoints can import heavyweight optional transcriber dependencies.
    if audio_imports_enabled_for_runtime():
        append_imported_router_spec(
            specs,
            ImportedRouterSpec(
                import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio",
                log_name="audio",
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio",),
                route_key="audio",
            ),
        )
        append_imported_router_spec(
            specs,
            ImportedRouterSpec(
                import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio",
                log_name="audio_websocket",
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-websocket",),
                route_key="audio-websocket",
                attr_name="ws_router",
            ),
        )
        append_imported_router_spec(
            specs,
            ImportedRouterSpec(
                import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs",
                log_name="audio_jobs",
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-jobs",),
                route_key="audio-jobs",
            ),
        )
    else:
        logger.info("Skipping audio router imports in pytest (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)")

    # Chunking, vector stores, and prompt operations
    for processing_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chunking",
            log_name="chunking",
            prefix=f"{API_V1_PREFIX}/chunking",
            tags=("chunking",),
            route_key="chunking",
            attr_name="chunking_router",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vector_stores_openai",
            log_name="vector_stores",
            prefix=f"{API_V1_PREFIX}",
            tags=("vector-stores",),
            route_key="vector-stores",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chunking_templates",
            log_name="chunking_templates",
            prefix=f"{API_V1_PREFIX}",
            tags=("chunking-templates",),
            route_key="chunking-templates",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompts",
            log_name="prompts",
            prefix=f"{API_V1_PREFIX}/prompts",
            tags=("prompts",),
            route_key="prompts",
        ),
    ):
        append_imported_router_spec(specs, processing_spec)

    # Workflow routers are force-included in explicit pytest runtime for unit coverage.
    for workflow_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.workflows",
            log_name="workflows",
            tags=("workflows",),
            route_key="" if _is_explicit_pytest_runtime() else "workflows",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chat_workflows",
            log_name="chat_workflows",
            tags=("chat-workflows",),
            route_key="" if _is_explicit_pytest_runtime() else "chat-workflows",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.scheduler_workflows",
            log_name="scheduler_workflows",
            tags=("scheduler",),
            route_key="" if _is_explicit_pytest_runtime() else "scheduler",
            default_stable=False,
        ),
    ):
        append_imported_router_spec(specs, workflow_spec)

    # Utility/content routers
    for utility_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.claims",
            log_name="claims",
            prefix=f"{API_V1_PREFIX}",
            tags=("claims",),
            route_key="claims",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.text2sql",
            log_name="text2sql",
            prefix=f"{API_V1_PREFIX}",
            tags=("text2sql",),
            route_key="text2sql",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.email",
            log_name="email",
            prefix=f"{API_V1_PREFIX}/email",
            tags=("email",),
            route_key="email",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.outputs_templates",
            log_name="outputs_templates",
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs-templates",),
            route_key="outputs-templates",
        ),
    ):
        append_imported_router_spec(specs, utility_spec)

    # Integration endpoints
    for integration_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.slack",
            log_name="slack",
            prefix=f"{API_V1_PREFIX}",
            tags=("slack",),
            route_key="slack",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.discord",
            log_name="discord",
            prefix=f"{API_V1_PREFIX}",
            tags=("discord",),
            route_key="discord",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.telegram",
            log_name="telegram",
            prefix=f"{API_V1_PREFIX}",
            tags=("telegram",),
            route_key="telegram",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.meetings",
            log_name="meetings",
            prefix=f"{API_V1_PREFIX}",
            tags=("meetings",),
            route_key="meetings",
            default_stable=False,
        ),
    ):
        append_imported_router_spec(specs, integration_spec)

    # Collections and reading endpoints
    for collections_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.collections_feeds",
            log_name="collections_feeds",
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-feeds",),
            route_key="collections-feeds",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.collections_websub",
            log_name="collections_websub",
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-websub",),
            route_key="collections-websub",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.collections_websub",
            log_name="collections_websub_callback",
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-websub",),
            route_key="collections-websub",
            attr_name="callback_router",
            skip_context="(callback_router)",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reading",
            log_name="reading",
            prefix=f"{API_V1_PREFIX}",
            tags=("reading",),
            route_key="reading",
        ),
    ):
        append_imported_router_spec(specs, collections_spec)

    # Prompt Studio endpoints
    for prompt_studio_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_projects",
            log_name="prompt_studio_projects",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts",
            log_name="prompt_studio_prompts",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_test_cases",
            log_name="prompt_studio_test_cases",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization",
            log_name="prompt_studio_optimization",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_status",
            log_name="prompt_studio_status",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations",
            log_name="prompt_studio_evaluations",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket",
            log_name="prompt_studio_websocket",
            tags=("prompt-studio",),
            route_key="prompt-studio",
        ),
    ):
        append_imported_router_spec(specs, prompt_studio_spec)

    # Workspace and character endpoints
    for character_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.workspace_migrations",
            log_name="workspace_migrations",
            prefix=f"{API_V1_PREFIX}/workspaces",
            tags=("workspaces",),
            route_key="workspaces",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.workspaces",
            log_name="workspaces",
            prefix=f"{API_V1_PREFIX}/workspaces",
            tags=("workspaces",),
            route_key="workspaces",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.workspace_memberships",
            log_name="workspace_memberships",
            prefix=f"{API_V1_PREFIX}/workspace-memberships",
            tags=("workspaces",),
            route_key="workspace-memberships",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.character_chat_sessions",
            log_name="character_chat_sessions",
            prefix=f"{API_V1_PREFIX}/chats",
            tags=("character-chat-sessions",),
            route_key="character-chat-sessions",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.character_memory",
            log_name="character_memory",
            prefix=f"{API_V1_PREFIX}/characters",
            tags=("character-memory",),
            route_key="character-memory",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.characters_endpoint",
            log_name="characters",
            prefix=f"{API_V1_PREFIX}/characters",
            tags=("characters",),
            route_key="characters",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.character_messages",
            log_name="character_messages",
            prefix=f"{API_V1_PREFIX}",
            tags=("character-messages",),
            route_key="character-messages",
        ),
    ):
        append_imported_router_spec(specs, character_spec)

    # Audiobooks and Voice Assistant endpoints
    for audio_voice_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.audio.audiobooks",
            log_name="audiobooks",
            prefix=f"{API_V1_PREFIX}",
            tags=("audiobooks",),
            route_key="audiobooks",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.voice_assistant",
            log_name="voice_assistant",
            prefix=f"{API_V1_PREFIX}/voice",
            tags=("voice-assistant",),
            route_key="voice-assistant",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.voice_assistant",
            log_name="voice_assistant_ws",
            prefix=f"{API_V1_PREFIX}/voice",
            tags=("voice-assistant-ws",),
            route_key="voice-assistant-ws",
            attr_name="ws_router",
        ),
    ):
        append_imported_router_spec(specs, audio_voice_spec)

    # Kanban Board endpoints
    for kanban_module in (
        "kanban_boards",
        "kanban_lists",
        "kanban_cards",
        "kanban_labels",
        "kanban_checklists",
        "kanban_comments",
        "kanban_search",
        "kanban_links",
        "kanban_workflow",
    ):
        append_imported_router_spec(
            specs,
            ImportedRouterSpec(
                import_path=f"tldw_Server_API.app.api.v1.endpoints.kanban.{kanban_module}",
                log_name=kanban_module,
                prefix=f"{API_V1_PREFIX}/kanban",
                tags=("kanban",),
                route_key="kanban",
            ),
        )

    # Ingestion and adapter endpoints
    for adapter_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.connectors",
            log_name="connectors",
            prefix=f"{API_V1_PREFIX}",
            tags=("connectors",),
            route_key="connectors",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.ingestion_sources",
            log_name="ingestion_sources",
            prefix=f"{API_V1_PREFIX}",
            tags=("ingestion-sources",),
            route_key="ingestion-sources",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.web_scraping",
            log_name="web_scraping",
            prefix=f"{API_V1_PREFIX}",
            tags=("web-scraping",),
            route_key="web-scraping",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reading_highlights",
            log_name="reading_highlights",
            prefix=f"{API_V1_PREFIX}",
            tags=("reading-highlights",),
            route_key="reading-highlights",
        ),
    ):
        append_imported_router_spec(specs, adapter_spec)

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.outputs",
            log_name="outputs",
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs",),
            route_key="outputs",
        ),
    )

    # Notes graph/task routes must be registered before generic notes routes so
    # static subpaths are not shadowed by /{note_id}.
    for notes_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notes_graph",
            log_name="notes_graph",
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
            route_key="notes",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notes_tasks",
            log_name="notes_tasks",
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
            route_key="notes",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notes",
            log_name="notes",
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
            route_key="notes",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.web_clipper",
            log_name="web_clipper",
            prefix=f"{API_V1_PREFIX}/web-clipper",
            tags=("web-clipper",),
            route_key="web-clipper",
        ),
    ):
        append_imported_router_spec(specs, notes_spec)

    for learning_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.translate",
            log_name="translate",
            prefix=f"{API_V1_PREFIX}",
            tags=("translation",),
            route_key="translation",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.slides",
            log_name="slides",
            prefix=f"{API_V1_PREFIX}",
            tags=("slides",),
            route_key="slides",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.flashcards",
            log_name="flashcards",
            prefix=f"{API_V1_PREFIX}",
            tags=("flashcards",),
            route_key="flashcards",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.quizzes",
            log_name="quizzes",
            prefix=f"{API_V1_PREFIX}",
            tags=("quizzes",),
            route_key="quizzes",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.study_suggestions",
            log_name="study_suggestions",
            prefix=f"{API_V1_PREFIX}",
            tags=("study-suggestions",),
            route_key="study-suggestions",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.writing",
            log_name="writing",
            prefix=f"{API_V1_PREFIX}/writing",
            tags=("writing",),
            route_key="writing",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.writing_manuscripts",
            log_name="writing_manuscripts",
            prefix=f"{API_V1_PREFIX}/writing/manuscripts",
            tags=("manuscripts",),
            route_key="manuscripts",
        ),
    ):
        append_imported_router_spec(specs, learning_spec)

    # Chatbooks and sharing endpoints
    for collaboration_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chatbooks",
            log_name="chatbooks",
            prefix=f"{API_V1_PREFIX}",
            tags=("chatbooks",),
            route_key="chatbooks",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.sharing",
            log_name="sharing",
            prefix=f"{API_V1_PREFIX}",
            tags=("sharing",),
            route_key="sharing",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prototype_workspaces",
            log_name="prototype_workspaces",
            prefix=f"{API_V1_PREFIX}",
            tags=("prototype-workspaces",),
            route_key="prototype-workspaces",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.sandbox_workspace_diagnostics",
            log_name="sandbox_workspace_diagnostics",
            prefix=f"{API_V1_PREFIX}",
            tags=("sandbox",),
            route_key="sandbox-workspace-diagnostics",
        ),
    ):
        append_imported_router_spec(specs, collaboration_spec)

    # Persona endpoints are force-included in explicit pytest runtime for WS/unit coverage.
    for persona_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.persona",
            log_name="persona",
            prefix=f"{API_V1_PREFIX}/persona",
            tags=("persona",),
            route_key="" if _is_explicit_pytest_runtime() else "persona",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.personalization",
            log_name="personalization",
            prefix=f"{API_V1_PREFIX}/personalization",
            tags=("personalization",),
            route_key="personalization",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.companion",
            log_name="companion",
            prefix=f"{API_V1_PREFIX}/companion",
            tags=("companion",),
            route_key="companion",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.archetype_endpoints",
            log_name="archetype_endpoints",
            prefix=f"{API_V1_PREFIX}/persona/archetypes",
            tags=("persona-archetypes",),
        ),
    ):
        append_imported_router_spec(specs, persona_spec)

    for tail_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.files",
            log_name="files",
            prefix=f"{API_V1_PREFIX}",
            tags=("files",),
            route_key="files",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.data_tables",
            log_name="data_tables",
            prefix=f"{API_V1_PREFIX}",
            tags=("data-tables",),
            route_key="data-tables",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.items",
            log_name="items",
            prefix=f"{API_V1_PREFIX}",
            tags=("items",),
            route_key="items",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reminders",
            log_name="reminders",
            prefix=f"{API_V1_PREFIX}",
            tags=("tasks",),
            route_key="tasks",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notifications",
            log_name="notifications",
            prefix=f"{API_V1_PREFIX}",
            tags=("notifications",),
            route_key="notifications",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.watchlists",
            log_name="watchlists",
            prefix=f"{API_V1_PREFIX}",
            tags=("watchlists",),
            route_key="watchlists",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.integrations_control_plane",
            log_name="integrations_control_plane",
            prefix=f"{API_V1_PREFIX}",
            tags=("integrations",),
            route_key="integrations",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane",
            log_name="scheduled_tasks_control_plane",
            prefix=f"{API_V1_PREFIX}",
            tags=("scheduled-tasks",),
            route_key="scheduled-tasks",
            default_stable=False,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vn_capabilities",
            log_name="vn_capabilities",
            prefix=f"{API_V1_PREFIX}/vn",
            tags=("vn-capabilities",),
            route_key="vn-capabilities",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vn_assets",
            log_name="vn_assets",
            prefix=f"{API_V1_PREFIX}/vn",
            tags=("vn-assets",),
            route_key="vn-assets",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vn_scripts",
            log_name="vn_scripts",
            prefix=f"{API_V1_PREFIX}/vn",
            tags=("vn-scripts",),
            route_key="vn-scripts",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vn_policy",
            log_name="vn_policy",
            prefix=f"{API_V1_PREFIX}/vn",
            tags=("vn-policy",),
            route_key="vn-policy",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vn_play",
            log_name="vn_play",
            prefix=f"{API_V1_PREFIX}/vn",
            tags=("vn-play",),
            route_key="vn-play",
        ),
    ):
        append_imported_router_spec(specs, tail_spec)

    return specs
