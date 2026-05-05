"""Minimal-test router group.

These routers are force-included under MINIMAL_TEST_APP to keep lightweight
integration tests working without importing the broader endpoint surface.
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
from tldw_Server_API.app.core.testing import (
    audio_imports_enabled_for_runtime,
    env_flag_enabled as _env_flag_enabled,
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
)

API_V1_PREFIX = "/api/v1"
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

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.llm_providers",
            log_name="llm providers",
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            skip_context="in minimal test app",
        ),
    )
    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mlx",
            log_name="mlx",
            prefix=f"{API_V1_PREFIX}",
            tags=("llm",),
            skip_context="in minimal test app",
        ),
    )

    for llm_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.llamacpp",
            log_name="llamacpp",
            prefix=f"{API_V1_PREFIX}",
            tags=("llamacpp",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.llamacpp",
            log_name="llamacpp_public",
            tags=("llamacpp",),
            attr_name="public_router",
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.messages",
            log_name="messages",
            prefix=f"{API_V1_PREFIX}",
            tags=("messages",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.messages",
            log_name="messages_public",
            tags=("messages",),
            attr_name="public_router",
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, llm_spec)

    for embedding_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vector_stores_openai",
            log_name="vector-stores",
            prefix=f"{API_V1_PREFIX}",
            tags=("vector-stores",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced",
            log_name="embeddings",
            prefix=f"{API_V1_PREFIX}",
            tags=("embeddings",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.media_embeddings",
            log_name="media_embeddings",
            prefix=f"{API_V1_PREFIX}",
            tags=("media-embeddings",),
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, embedding_spec)

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

    for auxiliary_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chunking_templates",
            log_name="chunking templates",
            prefix=f"{API_V1_PREFIX}",
            tags=("chunking-templates",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prompts",
            log_name="prompts",
            prefix=f"{API_V1_PREFIX}/prompts",
            tags=("prompts",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.claims",
            log_name="claims",
            prefix=f"{API_V1_PREFIX}",
            tags=("claims",),
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, auxiliary_spec)

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.rag_unified",
            log_name="rag_unified",
            tags=("rag-unified",),
            skip_context="in minimal test app",
        ),
    )

    for auxiliary_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.text2sql",
            log_name="text2sql",
            prefix=f"{API_V1_PREFIX}",
            tags=("text2sql",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.feedback",
            log_name="feedback",
            prefix=f"{API_V1_PREFIX}/feedback",
            tags=("feedback",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.vlm",
            log_name="vlm",
            prefix=f"{API_V1_PREFIX}",
            tags=("vlm",),
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, auxiliary_spec)

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.rag_health",
            log_name="rag_health",
            tags=("rag-health",),
            skip_context="in minimal test app",
        ),
    )

    for auxiliary_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.consent",
            log_name="consent",
            prefix=f"{API_V1_PREFIX}",
            tags=("consent",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.outputs_templates",
            log_name="outputs_templates",
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs-templates",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.outputs",
            log_name="outputs",
            prefix=f"{API_V1_PREFIX}",
            tags=("outputs",),
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, auxiliary_spec)

    for collections_social_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.collections_feeds",
            log_name="collections_feeds",
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-feeds",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.collections_websub",
            log_name="collections_websub",
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-websub",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.collections_websub",
            log_name="collections_websub_callback",
            prefix=f"{API_V1_PREFIX}",
            tags=("collections-websub",),
            attr_name="callback_router",
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.slack",
            log_name="slack",
            prefix=f"{API_V1_PREFIX}",
            tags=("slack",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.discord",
            log_name="discord",
            prefix=f"{API_V1_PREFIX}",
            tags=("discord",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.telegram",
            log_name="telegram",
            prefix=f"{API_V1_PREFIX}",
            tags=("telegram",),
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, collections_social_spec)

    data_resource_skip_context = "in minimal test app"
    for data_resource_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.files",
            log_name="files",
            prefix=f"{API_V1_PREFIX}",
            tags=("files",),
            skip_context=data_resource_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.storage",
            log_name="storage",
            prefix=f"{API_V1_PREFIX}",
            tags=("storage",),
            skip_context=data_resource_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.data_tables",
            log_name="data_tables",
            prefix=f"{API_V1_PREFIX}",
            tags=("data-tables",),
            skip_context=data_resource_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reading_highlights",
            log_name="reading_highlights",
            prefix=f"{API_V1_PREFIX}",
            tags=("reading-highlights",),
            skip_context=data_resource_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.items",
            log_name="items",
            prefix=f"{API_V1_PREFIX}",
            tags=("items",),
            skip_context=data_resource_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reminders",
            log_name="reminders",
            prefix=f"{API_V1_PREFIX}",
            tags=("tasks",),
            skip_context=data_resource_skip_context,
        ),
    ):
        append_imported_router_spec(specs, data_resource_spec)

    control_support_skip_context = "in minimal test app"
    for control_support_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.integrations_control_plane",
            log_name="integrations_control_plane",
            prefix=f"{API_V1_PREFIX}",
            tags=("integrations",),
            skip_context=control_support_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane",
            log_name="scheduled_tasks_control_plane",
            prefix=f"{API_V1_PREFIX}",
            tags=("scheduled-tasks",),
            skip_context=control_support_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notifications",
            log_name="notifications",
            prefix=f"{API_V1_PREFIX}",
            tags=("notifications",),
            skip_context=control_support_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chatbooks",
            log_name="chatbooks",
            prefix=f"{API_V1_PREFIX}",
            tags=("chatbooks",),
            skip_context=control_support_skip_context,
        ),
    ):
        append_imported_router_spec(specs, control_support_spec)

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

    admin_router_added = False
    try:
        from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router

        specs.append(RouterSpec(
            router=admin_router,
            prefix=f"{API_V1_PREFIX}",
            tags=("admin",),
        ))
        admin_router_added = True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Skipping admin router in minimal test app: {e}")

    if not admin_router_added:
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

    for acp_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.agent_client_protocol",
            log_name="ACP",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_schedules",
            log_name="ACP schedules",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-schedules",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_triggers",
            log_name="ACP triggers",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-triggers",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_permissions",
            log_name="ACP permissions",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-permissions",),
            skip_context="in minimal test app",
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_multiplex",
            log_name="ACP multiplex",
            prefix=f"{API_V1_PREFIX}",
            tags=("acp-multiplex",),
            skip_context="in minimal test app",
        ),
    ):
        append_imported_router_spec(specs, acp_spec)

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

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.sandbox",
            log_name="sandbox",
            prefix=f"{API_V1_PREFIX}",
            tags=("sandbox",),
            skip_context="in minimal test app",
        ),
    )

    return specs
