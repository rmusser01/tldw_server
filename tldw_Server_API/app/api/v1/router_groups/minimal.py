"""Minimal-test router group.

These routers are force-included under MINIMAL_TEST_APP to keep lightweight
integration tests working without importing the broader endpoint surface.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec

API_V1_PREFIX = "/api/v1"


def iter_minimal_test_router_specs() -> Iterable[RouterSpec]:
    """Yield the always-included minimal-test router specs."""
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
        router as character_chat_sessions_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.character_memory import (
        router as character_memory_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.character_messages import (
        router as character_messages_router,
    )
    from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as character_router
    from tldw_Server_API.app.api.v1.endpoints.chat import conversations_alias_router
    from tldw_Server_API.app.api.v1.endpoints.chat import router as chat_router
    from tldw_Server_API.app.api.v1.endpoints.chat_loop import router as chat_loop_router
    from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router
    from tldw_Server_API.app.api.v1.endpoints.research import router as research_router
    from tldw_Server_API.app.api.v1.endpoints.research_runs import router as research_runs_router
    from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router

    return [
        RouterSpec(router=research_router, prefix=f"{API_V1_PREFIX}/research", tags=("research",)),
        RouterSpec(router=research_runs_router, prefix=f"{API_V1_PREFIX}", tags=("research-runs",)),
        RouterSpec(router=paper_search_router, prefix=f"{API_V1_PREFIX}/paper-search", tags=("paper-search",)),
        RouterSpec(router=chat_router, prefix=f"{API_V1_PREFIX}/chat"),
        RouterSpec(router=chat_loop_router, prefix=f"{API_V1_PREFIX}"),
        RouterSpec(router=conversations_alias_router, prefix=f"{API_V1_PREFIX}/chats", tags=("chat",)),
        RouterSpec(router=character_router, prefix=f"{API_V1_PREFIX}/characters", tags=("characters",)),
        RouterSpec(router=character_memory_router, prefix=f"{API_V1_PREFIX}/characters", tags=("character-memory",)),
        RouterSpec(
            router=character_chat_sessions_router,
            prefix=f"{API_V1_PREFIX}/chats",
            tags=("character-chat-sessions",),
        ),
        RouterSpec(router=character_messages_router, prefix=f"{API_V1_PREFIX}", tags=("character-messages",)),
        RouterSpec(router=workspaces_router, prefix=f"{API_V1_PREFIX}/workspaces", tags=("workspaces",)),
    ]


def iter_minimal_optional_router_specs() -> Iterable[RouterSpec]:
    """Yield optional minimal-test router specs, skipping unavailable imports."""
    specs: list[RouterSpec] = []

    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_providers_router

        specs.append(RouterSpec(
            router=llm_providers_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping llm providers router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.mlx import router as mlx_router

        specs.append(RouterSpec(
            router=mlx_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping mlx router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router as vector_stores_router

        specs.append(RouterSpec(
            router=vector_stores_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("vector-stores",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping vector-stores router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router

        specs.append(RouterSpec(
            router=embeddings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("embeddings",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping embeddings router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.media_embeddings import router as media_embeddings_router

        specs.append(RouterSpec(
            router=media_embeddings_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("media-embeddings",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping media_embeddings router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router

        specs.append(RouterSpec(
            router=chunking_templates_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("chunking-templates",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chunking templates router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router

        specs.append(RouterSpec(
            router=prompt_router,
            prefix=f"{API_V1_PREFIX}/prompts",
            tags=("prompts",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping prompts router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router

        specs.append(RouterSpec(
            router=claims_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("claims",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping claims router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_unified import router as rag_unified_router

        specs.append(RouterSpec(
            router=rag_unified_router,
            tags=("rag-unified",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping rag_unified router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.text2sql import router as text2sql_router

        specs.append(RouterSpec(
            router=text2sql_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("text2sql",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping text2sql router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.feedback import router as feedback_router

        specs.append(RouterSpec(
            router=feedback_router,
            prefix=f"{API_V1_PREFIX}/feedback",
            tags=("feedback",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping feedback router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.vlm import router as vlm_router

        specs.append(RouterSpec(
            router=vlm_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("vlm",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping vlm router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router

        specs.append(RouterSpec(
            router=rag_health_router,
            tags=("rag-health",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping rag_health router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.consent import router as consent_router

        specs.append(RouterSpec(
            router=consent_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("consent",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping consent router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs_templates import router as outputs_templates_router

        specs.append(RouterSpec(
            router=outputs_templates_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs-templates",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping outputs_templates router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router

        specs.append(RouterSpec(
            router=outputs_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping outputs router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as collections_feeds_router

        specs.append(RouterSpec(
            router=collections_feeds_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-feeds",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping collections_feeds router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import (
            callback_router as websub_callback_router,
        )
        from tldw_Server_API.app.api.v1.endpoints.collections_websub import (
            router as collections_websub_router,
        )

        specs.extend([
            RouterSpec(
                router=collections_websub_router,
                prefix=f"{API_V1_PREFIX}",
                tags=("collections-websub",),
            ),
            RouterSpec(
                router=websub_callback_router,
                prefix=f"{API_V1_PREFIX}",
                tags=("collections-websub",),
            ),
        ])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping collections_websub router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.slack import router as slack_router

        specs.append(RouterSpec(
            router=slack_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("slack",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping slack router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.discord import router as discord_router

        specs.append(RouterSpec(
            router=discord_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("discord",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping discord router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.telegram import router as telegram_router

        specs.append(RouterSpec(
            router=telegram_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("telegram",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping telegram router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.files import router as files_router

        specs.append(RouterSpec(
            router=files_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("files",),
        ))
    except ImportError as e:
        logger.debug(f"Skipping files router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.storage import router as storage_router

        specs.append(RouterSpec(
            router=storage_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("storage",),
        ))
    except ImportError as e:
        logger.debug(f"Skipping storage router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router

        specs.append(RouterSpec(
            router=data_tables_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("data-tables",),
        ))
    except ImportError as e:
        logger.debug(f"Skipping data_tables router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.reading_highlights import router as reading_highlights_router

        specs.append(RouterSpec(
            router=reading_highlights_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("reading-highlights",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping reading_highlights router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.items import router as items_router

        specs.append(RouterSpec(
            router=items_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("items",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping items router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.reminders import router as reminders_router

        specs.append(RouterSpec(
            router=reminders_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("tasks",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping reminders router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.integrations_control_plane import (
            router as integrations_control_plane_router,
        )

        specs.append(RouterSpec(
            router=integrations_control_plane_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("integrations",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping integrations control plane router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane import (
            router as scheduled_tasks_control_plane_router,
        )

        specs.append(RouterSpec(
            router=scheduled_tasks_control_plane_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("scheduled-tasks",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping scheduled tasks control plane router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.notifications import router as notifications_router

        specs.append(RouterSpec(
            router=notifications_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("notifications",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug("Skipping notifications router in minimal test app: {}", e)

    return specs
