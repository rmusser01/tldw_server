"""Minimal-test router group.

These routers are force-included under MINIMAL_TEST_APP to keep lightweight
integration tests working without importing the broader endpoint surface.
"""
from __future__ import annotations

import importlib
from typing import Any, Iterable

from loguru import logger

from tldw_Server_API.app.api.v1.router_groups.factories import evaluations_router_factory
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec
from tldw_Server_API.app.core.testing import (
    audio_imports_enabled_for_runtime,
    env_flag_enabled as _env_flag_enabled,
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
)

API_V1_PREFIX = "/api/v1"


def _try_add_spec(
    specs: list[RouterSpec],
    import_path: str,
    *,
    log_name: str,
    attr_name: str = "router",
    **spec_kwargs: Any,
) -> None:
    try:
        module = importlib.import_module(import_path)
        specs.append(RouterSpec(router=getattr(module, attr_name), **spec_kwargs))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping {log_name} router in minimal test app: {e}")


def iter_minimal_test_router_specs() -> Iterable[RouterSpec]:
    """Yield the always-included minimal-test router specs."""
    from tldw_Server_API.app.api.v1.endpoints.auth import router as auth_router
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
    from tldw_Server_API.app.api.v1.endpoints.health import router as health_router
    from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router
    from tldw_Server_API.app.api.v1.endpoints.research import router as research_router
    from tldw_Server_API.app.api.v1.endpoints.research_runs import router as research_runs_router
    from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router

    return [
        RouterSpec(router=health_router, prefix=f"{API_V1_PREFIX}", tags=("health",)),
        RouterSpec(router=auth_router, prefix=f"{API_V1_PREFIX}", tags=("authentication",)),
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


def _audio_jobs_imports_enabled_for_runtime() -> bool:
    return not _is_explicit_pytest_runtime() or _env_flag_enabled("MINIMAL_TEST_INCLUDE_AUDIO_JOBS")


def iter_minimal_optional_router_specs() -> Iterable[RouterSpec]:
    """Yield optional minimal-test router specs, skipping unavailable imports."""
    specs: list[RouterSpec] = []

    _try_add_spec(
        specs,
        "tldw_Server_API.app.api.v1.endpoints.llm_providers",
        log_name="llm providers",
        prefix=f"{API_V1_PREFIX}",
        tags=("llm",),
    )
    _try_add_spec(
        specs,
        "tldw_Server_API.app.api.v1.endpoints.mlx",
        log_name="mlx",
        prefix=f"{API_V1_PREFIX}",
        tags=("llm",),
    )

    try:
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import public_router as llamacpp_public_router
        from tldw_Server_API.app.api.v1.endpoints.llamacpp import router as llamacpp_router
        from tldw_Server_API.app.api.v1.endpoints.messages import public_router as messages_public_router
        from tldw_Server_API.app.api.v1.endpoints.messages import router as messages_router

        specs.extend([
            RouterSpec(
                router=llamacpp_router,
                prefix=f"{API_V1_PREFIX}",
                tags=("llamacpp",),
            ),
            RouterSpec(
                router=llamacpp_public_router,
                prefix="",
                tags=("llamacpp",),
            ),
            RouterSpec(
                router=messages_router,
                prefix=f"{API_V1_PREFIX}",
                tags=("messages",),
            ),
            RouterSpec(
                router=messages_public_router,
                prefix="",
                tags=("messages",),
            ),
        ])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping llamacpp/messages routers in minimal test app: {e}")

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

    if audio_imports_enabled_for_runtime():
        def _audio_router_factory():
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router

            return audio_router

        def _audio_ws_router_factory():
            from tldw_Server_API.app.api.v1.endpoints.audio.audio import ws_router as audio_ws_router

            return audio_ws_router

        specs.extend([
            RouterSpec(
                router=_audio_router_factory,
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio",),
                route_key="audio",
            ),
            RouterSpec(
                router=_audio_ws_router_factory,
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-ws",),
                route_key="audio-websocket",
            ),
        ])
    else:
        logger.info("Skipping audio routers in minimal test app (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)")

    def _media_router_factory():
        from tldw_Server_API.app.api.v1.endpoints.media import router as media_router

        return media_router

    specs.append(RouterSpec(
        router=_media_router_factory,
        prefix=f"{API_V1_PREFIX}/media",
        tags=("media",),
        route_key="media",
    ))

    if _audio_jobs_imports_enabled_for_runtime():
        def _audio_jobs_router_factory():
            from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import router as audio_jobs_router

            return audio_jobs_router

        specs.append(RouterSpec(
            router=_audio_jobs_router_factory,
            prefix=f"{API_V1_PREFIX}/audio",
            tags=("audio-jobs",),
            route_key="audio-jobs",
        ))
    else:
        logger.info("Skipping audio-jobs router in minimal test app (set MINIMAL_TEST_INCLUDE_AUDIO_JOBS=1 to enable)")

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
        logger.debug(f"Skipping notifications router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router

        specs.append(RouterSpec(
            router=chatbooks_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("chatbooks",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping chatbooks router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.chat_workflows import router as chat_workflows_router
        from tldw_Server_API.app.api.v1.endpoints.scheduler_workflows import router as scheduler_workflows_router
        from tldw_Server_API.app.api.v1.endpoints.workflows import router as workflows_router

        specs.extend([
            RouterSpec(
                router=workflows_router,
                prefix="",
                tags=("workflows",),
            ),
            RouterSpec(
                router=chat_workflows_router,
                prefix="",
                tags=("chat-workflows",),
            ),
            RouterSpec(
                router=scheduler_workflows_router,
                prefix="",
                tags=("scheduler",),
            ),
        ])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping workflow routers in minimal test app: {e}")

    specs.append(RouterSpec(
        router=evaluations_router_factory,
        prefix=f"{API_V1_PREFIX}",
        tags=("evaluations",),
        route_key="evaluations",
    ))

    def _monitoring_router_factory():
        from tldw_Server_API.app.api.v1.endpoints.monitoring import router as monitoring_router

        return monitoring_router

    specs.append(RouterSpec(
        router=_monitoring_router_factory,
        prefix=f"{API_V1_PREFIX}",
        tags=("monitoring",),
        route_key="monitoring",
    ))

    try:
        from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router

        specs.append(RouterSpec(
            router=sharing_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("sharing",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping sharing router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.personalization import router as personalization_router

        specs.append(RouterSpec(
            router=personalization_router,
            prefix=f"{API_V1_PREFIX}/personalization",
            tags=("personalization",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping personalization router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.companion import router as companion_router

        specs.append(RouterSpec(
            router=companion_router,
            prefix=f"{API_V1_PREFIX}/companion",
            tags=("companion",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping companion router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.family_wizard import router as family_wizard_router
        from tldw_Server_API.app.api.v1.endpoints.guardian_controls import router as guardian_controls_router

        specs.extend([
            RouterSpec(
                router=guardian_controls_router,
                prefix=f"{API_V1_PREFIX}/guardian",
                tags=("guardian",),
            ),
            RouterSpec(
                router=family_wizard_router,
                prefix=f"{API_V1_PREFIX}/guardian",
                tags=("guardian",),
            ),
        ])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping guardian controls router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.self_monitoring import router as self_monitoring_router

        specs.append(RouterSpec(
            router=self_monitoring_router,
            prefix=f"{API_V1_PREFIX}/self-monitoring",
            tags=("self-monitoring",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping self-monitoring router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.persona import router as persona_router

        specs.append(RouterSpec(
            router=persona_router,
            prefix=f"{API_V1_PREFIX}/persona",
            tags=("persona",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping persona router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import router as archetype_router

        specs.append(RouterSpec(
            router=archetype_router,
            prefix=f"{API_V1_PREFIX}/persona/archetypes",
            tags=("persona-archetypes",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping archetype router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router

        specs.append(RouterSpec(
            router=notes_router,
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping notes router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.web_clipper import router as web_clipper_router

        specs.append(RouterSpec(
            router=web_clipper_router,
            prefix=f"{API_V1_PREFIX}/web-clipper",
            tags=("web-clipper",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping web clipper router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.skills import router as skills_router

        specs.append(RouterSpec(
            router=skills_router,
            prefix=f"{API_V1_PREFIX}/skills",
            tags=("skills",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping skills router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.translate import router as translate_router

        specs.append(RouterSpec(
            router=translate_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("translation",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping translate router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router

        specs.append(RouterSpec(
            router=slides_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("slides",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping slides router in minimal test app: {e}")

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
            RouterSpec(router=kanban_boards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_lists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_cards_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_labels_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_checklists_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_comments_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_search_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_links_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
            RouterSpec(router=kanban_workflow_router, prefix=f"{API_V1_PREFIX}/kanban", tags=("kanban",), route_key="kanban"),
        ])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping kanban router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

        specs.append(RouterSpec(
            router=flashcards_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("flashcards",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping flashcards router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.quizzes import router as quizzes_router

        specs.append(RouterSpec(
            router=quizzes_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("quizzes",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping quizzes router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.study_suggestions import (
            router as study_suggestions_router,
        )

        specs.append(RouterSpec(
            router=study_suggestions_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("study-suggestions",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping study_suggestions router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.writing import router as writing_router

        specs.append(RouterSpec(
            router=writing_router,
            prefix=f"{API_V1_PREFIX}/writing",
            tags=("writing",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping writing router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import router as manuscripts_router

        specs.append(RouterSpec(
            router=manuscripts_router,
            prefix=f"{API_V1_PREFIX}/writing/manuscripts",
            tags=("manuscripts",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping manuscripts router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.email import router as email_router

        specs.append(RouterSpec(
            router=email_router,
            prefix=f"{API_V1_PREFIX}/email",
            tags=("email",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping email router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router

        specs.append(RouterSpec(
            router=jobs_admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("jobs",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping jobs_admin router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.audit import router as audit_router

        specs.append(RouterSpec(
            router=audit_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("audit",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping audit router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router

        specs.append(RouterSpec(
            router=config_info_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("config",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping config_info router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.config_admin import router as config_admin_router

        specs.append(RouterSpec(
            router=config_admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("config", "admin"),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping config_admin router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router

        specs.append(RouterSpec(
            router=admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("admin",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping admin router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.admin.admin_byok import router as admin_byok_router

        specs.append(RouterSpec(
            router=admin_byok_router,
            prefix=f"{API_V1_PREFIX}/admin",
            tags=("admin",),
        ))
    except Exception as admin_byok_error:  # noqa: BLE001
        logger.debug(f"Skipping admin_byok router in minimal test app: {admin_byok_error}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.orgs import router as orgs_router

        specs.append(RouterSpec(
            router=orgs_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping orgs router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.org_invites import router as org_invites_router

        specs.append(RouterSpec(
            router=org_invites_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("invites",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping org_invites router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.resource_governor import router as resource_governor_router

        specs.append(RouterSpec(
            router=resource_governor_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("resource-governor",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping resource_governor router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.users import router as users_router

        specs.append(RouterSpec(
            router=users_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping users router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped import router as shared_keys_scoped_router
        from tldw_Server_API.app.api.v1.endpoints.user_keys import router as user_keys_router

        specs.extend([
            RouterSpec(
                router=user_keys_router,
                prefix=f"{API_V1_PREFIX}",
                tags=("users",),
            ),
            RouterSpec(
                router=shared_keys_scoped_router,
                prefix=f"{API_V1_PREFIX}",
                tags=("organizations",),
            ),
        ])
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping BYOK/shared keys routers in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router

        specs.append(RouterSpec(
            router=mcp_unified_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-unified",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping MCP unified router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage import router as mcp_catalogs_manage_router

        specs.append(RouterSpec(
            router=mcp_catalogs_manage_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-catalogs",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping MCP catalogs router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.mcp_hub_management import router as mcp_hub_management_router

        specs.append(RouterSpec(
            router=mcp_hub_management_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-hub",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping MCP hub router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router

        specs.append(RouterSpec(
            router=privileges_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("privileges",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping privileges router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router

        specs.append(RouterSpec(
            router=tools_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("tools",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping tools router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import router as acp_router

        specs.append(RouterSpec(
            router=acp_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("acp",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping ACP router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_schedules import router as acp_schedules_router

        specs.append(RouterSpec(
            router=acp_schedules_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-schedules",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping ACP schedules router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_triggers import router as acp_triggers_router

        specs.append(RouterSpec(
            router=acp_triggers_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-triggers",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping ACP triggers router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_permissions import router as acp_permissions_router

        specs.append(RouterSpec(
            router=acp_permissions_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-permissions",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping ACP permissions router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.acp_multiplex import router as acp_multiplex_router

        specs.append(RouterSpec(
            router=acp_multiplex_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-multiplex",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping ACP multiplex router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import router as orch_router

        specs.append(RouterSpec(
            router=orch_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("agent-orchestration",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping orchestration router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.setup import router as setup_router

        specs.append(RouterSpec(
            router=setup_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("setup",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping setup router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router

        specs.append(RouterSpec(
            router=metrics_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("metrics",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping metrics router in minimal test app: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.authnz_debug import router as authnz_debug_router

        specs.append(RouterSpec(
            router=authnz_debug_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("authnz-debug",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping authnz_debug router in tests: {e}")

    try:
        from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router

        specs.append(RouterSpec(
            router=sandbox_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("sandbox",),
        ))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping sandbox router in minimal test app: {e}")

    return specs
