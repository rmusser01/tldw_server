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
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_unified import (
            router as rag_unified_router,
        )

        specs.append(RouterSpec(
            router=rag_unified_router,
            tags=("rag-unified",),
            route_key="rag-unified",
        ))
    except ImportError as e:
        logger.debug(f"Skipping rag_unified router: {e}")

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
            route_key="research",
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

    def _ocr_router_factory():
        from tldw_Server_API.app.api.v1.endpoints.ocr import router as ocr_router

        return ocr_router

    specs.append(RouterSpec(
        router=_ocr_router_factory,
        prefix=f"{API_V1_PREFIX}",
        tags=("ocr",),
        route_key="ocr",
    ))

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
            import_path="tldw_Server_API.app.api.v1.endpoints.workspaces",
            log_name="workspaces",
            prefix=f"{API_V1_PREFIX}/workspaces",
            tags=("workspaces",),
            route_key="workspaces",
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

    # Notes graph routes must be registered before generic notes routes so
    # /graph is not shadowed by /{note_id}.
    for notes_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notes_graph",
            log_name="notes_graph",
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
    try:
        from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router

        specs.append(RouterSpec(
            router=chatbooks_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("chatbooks",),
            route_key="chatbooks",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chatbooks router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router

        specs.append(RouterSpec(
            router=sharing_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("sharing",),
            route_key="sharing",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping sharing router: {e}")

    # Persona endpoints are force-included in explicit pytest runtime for WS/unit coverage.
    try:
        from tldw_Server_API.app.api.v1.endpoints.persona import router as persona_router

        specs.append(RouterSpec(
            router=persona_router,
            prefix=f"{API_V1_PREFIX}/persona",
            tags=("persona",),
            route_key="" if _is_explicit_pytest_runtime() else "persona",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping persona router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.personalization import (
            router as personalization_router,
        )

        specs.append(RouterSpec(
            router=personalization_router,
            prefix=f"{API_V1_PREFIX}/personalization",
            tags=("personalization",),
            route_key="personalization",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping personalization router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.companion import router as companion_router

        specs.append(RouterSpec(
            router=companion_router,
            prefix=f"{API_V1_PREFIX}/companion",
            tags=("companion",),
            route_key="companion",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping companion router: {e}")

    # Archetype template endpoints are always available read-only catalog data.
    try:
        from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import router as archetype_router

        specs.append(RouterSpec(
            router=archetype_router,
            prefix=f"{API_V1_PREFIX}/persona/archetypes",
            tags=("persona-archetypes",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping archetype router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.files import router as files_router

        specs.append(RouterSpec(
            router=files_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("files",),
            route_key="files",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping files router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router

        specs.append(RouterSpec(
            router=data_tables_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("data-tables",),
            route_key="data-tables",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping data_tables router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.items import router as items_router

        specs.append(RouterSpec(
            router=items_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("items",),
            route_key="items",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping items router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.reminders import router as reminders_router

        specs.append(RouterSpec(
            router=reminders_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("tasks",),
            route_key="tasks",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping reminders router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.notifications import (
            router as notifications_router,
        )

        specs.append(RouterSpec(
            router=notifications_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("notifications",),
            route_key="notifications",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping notifications router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

        specs.append(RouterSpec(
            router=watchlists_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("watchlists",),
            route_key="watchlists",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping watchlists router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.integrations_control_plane import (
            router as integrations_control_plane_router,
        )

        specs.append(RouterSpec(
            router=integrations_control_plane_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("integrations",),
            route_key="integrations",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping integrations_control_plane router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane import (
            router as scheduled_tasks_control_plane_router,
        )

        specs.append(RouterSpec(
            router=scheduled_tasks_control_plane_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("scheduled-tasks",),
            route_key="scheduled-tasks",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping scheduled_tasks_control_plane router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router

        specs.append(RouterSpec(
            router=vn_assets_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("vn-assets",),
            route_key="vn-assets",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping VN assets router: {e}")

    return specs
