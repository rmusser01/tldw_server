"""Content router group — media, RAG, embeddings, and content processing endpoints.

These routers handle content ingestion, search, retrieval, and
related operations.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger

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

    # RAG health endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router

        specs.append(RouterSpec(
            router=rag_health_router,
            tags=("rag-health",),
            route_key="rag-health",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping rag_health router: {e}")

    # Research endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.research import router as research_router

        specs.append(RouterSpec(
            router=research_router,
            prefix=f"{API_V1_PREFIX}/research",
            tags=("research",),
            route_key="research",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping research router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.research_runs import router as research_runs_router

        specs.append(RouterSpec(
            router=research_runs_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("research-runs",),
            route_key="research",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping research_runs router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router

        specs.append(RouterSpec(
            router=paper_search_router,
            prefix=f"{API_V1_PREFIX}/paper-search",
            tags=("paper-search",),
            route_key="paper-search",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping paper_search router: {e}")

    # Embeddings (OpenAI-compatible)
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            router as embeddings_router,
        )

        specs.append(RouterSpec(
            router=embeddings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("embeddings",),
            route_key="embeddings",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping embeddings router: {e}")

    # Media embeddings
    try:
        from tldw_Server_API.app.api.v1.endpoints.media_embeddings import (
            router as media_embeddings_router,
        )

        specs.append(RouterSpec(
            router=media_embeddings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("media-embeddings",),
            route_key="media-embeddings",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping media_embeddings router: {e}")

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
    try:
        from tldw_Server_API.app.api.v1.endpoints.media import router as media_router

        specs.append(RouterSpec(
            router=media_router,
            prefix=f"{API_V1_PREFIX}/media",
            tags=("media",),
            route_key="media",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping media router: {e}")

    # Audio endpoints can import heavyweight optional transcriber dependencies.
    if audio_imports_enabled_for_runtime():
        try:
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import ws_router as audio_ws_router

            specs.append(RouterSpec(
                router=audio_router,
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio",),
                route_key="audio",
            ))
            specs.append(RouterSpec(
                router=audio_ws_router,
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-websocket",),
                route_key="audio-websocket",
            ))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Skipping audio routers: {e}")

        try:
            from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import router as audio_jobs_router

            specs.append(RouterSpec(
                router=audio_jobs_router,
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-jobs",),
                route_key="audio-jobs",
            ))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Skipping audio_jobs router: {e}")
    else:
        logger.info("Skipping audio router imports in pytest (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)")

    # Chunking operations
    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking import chunking_router

        specs.append(RouterSpec(
            router=chunking_router,
            prefix=f"{API_V1_PREFIX}/chunking",
            tags=("chunking",),
            route_key="chunking",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chunking router: {e}")

    # Vector stores (OpenAI-compatible)
    try:
        from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router as vector_stores_router

        specs.append(RouterSpec(
            router=vector_stores_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("vector-stores",),
            route_key="vector-stores",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping vector-stores router: {e}")

    # Chunking templates
    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router

        specs.append(RouterSpec(
            router=chunking_templates_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("chunking-templates",),
            route_key="chunking-templates",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chunking_templates router: {e}")

    # Prompts
    try:
        from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router

        specs.append(RouterSpec(
            router=prompt_router,
            prefix=f"{API_V1_PREFIX}/prompts",
            tags=("prompts",),
            route_key="prompts",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping prompts router: {e}")

    # Workflow routers are force-included in explicit pytest runtime for unit coverage.
    try:
        from tldw_Server_API.app.api.v1.endpoints.chat_workflows import router as chat_workflows_router
        from tldw_Server_API.app.api.v1.endpoints.scheduler_workflows import router as scheduler_workflows_router
        from tldw_Server_API.app.api.v1.endpoints.workflows import router as workflows_router

        specs.append(RouterSpec(
            router=workflows_router,
            tags=("workflows",),
            route_key="" if _is_explicit_pytest_runtime() else "workflows",
            default_stable=False,
        ))
        specs.append(RouterSpec(
            router=chat_workflows_router,
            tags=("chat-workflows",),
            route_key="" if _is_explicit_pytest_runtime() else "chat-workflows",
        ))
        specs.append(RouterSpec(
            router=scheduler_workflows_router,
            tags=("scheduler",),
            route_key="" if _is_explicit_pytest_runtime() else "scheduler",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping workflow routers: {e}")

    # Claims
    try:
        from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router

        specs.append(RouterSpec(
            router=claims_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("claims",),
            route_key="claims",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping claims router: {e}")

    # Text2SQL
    try:
        from tldw_Server_API.app.api.v1.endpoints.text2sql import router as text2sql_router

        specs.append(RouterSpec(
            router=text2sql_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("text2sql",),
            route_key="text2sql",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping text2sql router: {e}")

    # Email search
    try:
        from tldw_Server_API.app.api.v1.endpoints.email import router as email_router

        specs.append(RouterSpec(
            router=email_router,
            prefix=f"{API_V1_PREFIX}/email",
            tags=("email",),
            route_key="email",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping email router: {e}")

    # Outputs and output templates
    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs_templates import router as outputs_templates_router

        specs.append(RouterSpec(
            router=outputs_templates_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs-templates",),
            route_key="outputs-templates",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping outputs_templates router: {e}")

    # Slack integration endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.slack import router as slack_router

        specs.append(RouterSpec(
            router=slack_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("slack",),
            route_key="slack",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping slack router: {e}")

    # Discord integration endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.discord import router as discord_router

        specs.append(RouterSpec(
            router=discord_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("discord",),
            route_key="discord",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping discord router: {e}")

    # Telegram integration endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.telegram import router as telegram_router

        specs.append(RouterSpec(
            router=telegram_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("telegram",),
            route_key="telegram",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping telegram router: {e}")

    # Meetings endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.meetings import router as meetings_router

        specs.append(RouterSpec(
            router=meetings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("meetings",),
            route_key="meetings",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping meetings router: {e}")

    # Collections feeds endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as collections_feeds_router

        specs.append(RouterSpec(
            router=collections_feeds_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-feeds",),
            route_key="collections-feeds",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping collections_feeds router: {e}")

    # Collections WebSub endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import callback_router as websub_callback_router
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import router as collections_websub_router

        specs.append(RouterSpec(
            router=collections_websub_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-websub",),
            route_key="collections-websub",
        ))
        specs.append(RouterSpec(
            router=websub_callback_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-websub",),
            route_key="collections-websub",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping collections_websub router: {e}")

    # Reading endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.reading import router as reading_router

        specs.append(RouterSpec(
            router=reading_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("reading",),
            route_key="reading",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping reading router: {e}")

    # Prompt Studio endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations import (
            router as prompt_studio_evaluations_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization import (
            router as prompt_studio_optimization_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_projects import (
            router as prompt_studio_projects_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts import (
            router as prompt_studio_prompts_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_status import (
            router as prompt_studio_status_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_test_cases import (
            router as prompt_studio_test_cases_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
            router as prompt_studio_websocket_router,
        )

        specs.append(RouterSpec(router=prompt_studio_projects_router, tags=("prompt-studio",), route_key="prompt-studio"))
        specs.append(RouterSpec(router=prompt_studio_prompts_router, tags=("prompt-studio",), route_key="prompt-studio"))
        specs.append(RouterSpec(router=prompt_studio_test_cases_router, tags=("prompt-studio",), route_key="prompt-studio"))
        specs.append(RouterSpec(router=prompt_studio_optimization_router, tags=("prompt-studio",), route_key="prompt-studio"))
        specs.append(RouterSpec(router=prompt_studio_status_router, tags=("prompt-studio",), route_key="prompt-studio"))
        specs.append(RouterSpec(router=prompt_studio_evaluations_router, tags=("prompt-studio",), route_key="prompt-studio"))
        specs.append(RouterSpec(router=prompt_studio_websocket_router, tags=("prompt-studio",), route_key="prompt-studio"))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping prompt_studio routers: {e}")

    # Workspace endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router

        specs.append(RouterSpec(
            router=workspaces_router,
            prefix=f"{API_V1_PREFIX}/workspaces",
            tags=("workspaces",),
            route_key="workspaces",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping workspaces router: {e}")

    # Character chat session endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
            router as character_chat_sessions_router,
        )

        specs.append(RouterSpec(
            router=character_chat_sessions_router,
            prefix=f"{API_V1_PREFIX}/chats",
            tags=("character-chat-sessions",),
            route_key="character-chat-sessions",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping character_chat_sessions router: {e}")

    # Character memory endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            router as character_memory_router,
        )

        specs.append(RouterSpec(
            router=character_memory_router,
            prefix=f"{API_V1_PREFIX}/characters",
            tags=("character-memory",),
            route_key="character-memory",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping character_memory router: {e}")

    # Character endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import (
            router as character_router,
        )

        specs.append(RouterSpec(
            router=character_router,
            prefix=f"{API_V1_PREFIX}/characters",
            tags=("characters",),
            route_key="characters",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping characters router: {e}")

    # Character message endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.character_messages import router as character_messages_router

        specs.append(RouterSpec(
            router=character_messages_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("character-messages",),
            route_key="character-messages",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping character_messages router: {e}")

    # Audiobooks endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router

        specs.append(RouterSpec(
            router=audiobooks_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("audiobooks",),
            route_key="audiobooks",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping audiobooks router: {e}")

    # Voice Assistant endpoints (REST + WebSocket)
    try:
        from tldw_Server_API.app.api.v1.endpoints.voice_assistant import (
            router as voice_assistant_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.voice_assistant import (
            ws_router as voice_assistant_ws_router,
        )

        specs.append(RouterSpec(
            router=voice_assistant_router,
            prefix=f"{API_V1_PREFIX}/voice",
            tags=("voice-assistant",),
            route_key="voice-assistant",
        ))
        specs.append(RouterSpec(
            router=voice_assistant_ws_router,
            prefix=f"{API_V1_PREFIX}/voice",
            tags=("voice-assistant-ws",),
            route_key="voice-assistant-ws",
        ))
    except ImportError as e:
        logger.debug(f"Voice assistant endpoints not available: {e}")

    # Kanban Board endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_boards import router as kanban_boards_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_cards import router as kanban_cards_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_checklists import router as kanban_checklists_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_comments import router as kanban_comments_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_labels import router as kanban_labels_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_links import router as kanban_links_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_lists import router as kanban_lists_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_search import router as kanban_search_router
        from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_workflow import router as kanban_workflow_router

        specs.extend([
            RouterSpec(
                router=kanban_router,
                prefix=f"{API_V1_PREFIX}/kanban",
                tags=("kanban",),
                route_key="kanban",
            )
            for kanban_router in (
                kanban_boards_router,
                kanban_lists_router,
                kanban_cards_router,
                kanban_labels_router,
                kanban_checklists_router,
                kanban_comments_router,
                kanban_search_router,
                kanban_links_router,
                kanban_workflow_router,
            )
        ])
    except ImportError as e:
        logger.debug(f"Kanban endpoints unavailable; skipping import: {e}")

    # Connectors endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.connectors import router as connectors_router

        specs.append(RouterSpec(
            router=connectors_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("connectors",),
            route_key="connectors",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping connectors router: {e}")

    # Ingestion sources endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.ingestion_sources import router as ingestion_sources_router

        specs.append(RouterSpec(
            router=ingestion_sources_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("ingestion-sources",),
            route_key="ingestion-sources",
            default_stable=False,
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping ingestion_sources router: {e}")

    # Web scraping endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.web_scraping import router as web_scraping_router

        specs.append(RouterSpec(
            router=web_scraping_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("web-scraping",),
            route_key="web-scraping",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping web_scraping router: {e}")

    # Reading highlights endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.reading_highlights import router as reading_highlights_router

        specs.append(RouterSpec(
            router=reading_highlights_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("reading-highlights",),
            route_key="reading-highlights",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping reading_highlights router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

        specs.append(RouterSpec(
            router=outputs_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs",),
            route_key="outputs",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping outputs router: {e}")

    # Notes graph routes must be registered before generic notes routes so
    # /graph is not shadowed by /{note_id}.
    try:
        from tldw_Server_API.app.api.v1.endpoints.notes_graph import router as notes_graph_router

        specs.append(RouterSpec(
            router=notes_graph_router,
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
            route_key="notes",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping notes_graph router: {e}")

    # Notes
    try:
        from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router

        specs.append(RouterSpec(
            router=notes_router,
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
            route_key="notes",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping notes router: {e}")

    # Web clipper
    try:
        from tldw_Server_API.app.api.v1.endpoints.web_clipper import router as web_clipper_router

        specs.append(RouterSpec(
            router=web_clipper_router,
            prefix=f"{API_V1_PREFIX}/web-clipper",
            tags=("web-clipper",),
            route_key="web-clipper",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping web_clipper router: {e}")

    # Translation
    try:
        from tldw_Server_API.app.api.v1.endpoints.translate import router as translate_router

        specs.append(RouterSpec(
            router=translate_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("translation",),
            route_key="translation",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping translate router: {e}")

    # Slides
    try:
        from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router

        specs.append(RouterSpec(
            router=slides_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("slides",),
            route_key="slides",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping slides router: {e}")

    # Study content endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

        specs.append(RouterSpec(
            router=flashcards_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("flashcards",),
            route_key="flashcards",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping flashcards router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.quizzes import router as quizzes_router

        specs.append(RouterSpec(
            router=quizzes_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("quizzes",),
            route_key="quizzes",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping quizzes router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.study_suggestions import (
            router as study_suggestions_router,
        )

        specs.append(RouterSpec(
            router=study_suggestions_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("study-suggestions",),
            route_key="study-suggestions",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping study_suggestions router: {e}")

    # Writing playground endpoints
    try:
        from tldw_Server_API.app.api.v1.endpoints.writing import router as writing_router

        specs.append(RouterSpec(
            router=writing_router,
            prefix=f"{API_V1_PREFIX}/writing",
            tags=("writing",),
            route_key="writing",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping writing router: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import (
            router as manuscripts_router,
        )

        specs.append(RouterSpec(
            router=manuscripts_router,
            prefix=f"{API_V1_PREFIX}/writing/manuscripts",
            tags=("manuscripts",),
            route_key="manuscripts",
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping writing_manuscripts router: {e}")

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

    return specs
