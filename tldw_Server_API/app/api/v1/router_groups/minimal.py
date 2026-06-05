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
from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
from tldw_Server_API.app.api.v1.router_groups.factories import evaluations_router_factory
from tldw_Server_API.app.api.v1.router_groups.selection import (
    RouterSpecOverride,
    select_router_specs_by_name,
)
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec
from tldw_Server_API.app.core.testing import (
    audio_imports_enabled_for_runtime,
    env_flag_enabled as _env_flag_enabled,
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
)

API_V1_PREFIX = "/api/v1"
REQUIRED_ROUTER_SKIP_EXCEPTIONS: tuple[type[Exception], ...] = ()
MINIMAL_REQUIRED_ROUTER_NAMES = (
    "health",
    "auth",
    "research",
    "research_runs",
    "paper_search",
    "chat",
    "chat_loop",
    "conversations_alias",
    "characters",
    "character_memory",
    "character_chat_sessions",
    "character_messages",
    "workspace_migrations",
    "workspaces",
)


def iter_minimal_test_router_specs() -> Iterable[RouterSpec]:
    """Yield the always-included minimal-test router specs."""
    canonical_specs = (
        *iter_core_router_specs(),
        *iter_content_router_specs(),
    )
    required_override = RouterSpecOverride(
        skip_exceptions=REQUIRED_ROUTER_SKIP_EXCEPTIONS,
    )

    return select_router_specs_by_name(
        canonical_specs,
        MINIMAL_REQUIRED_ROUTER_NAMES,
        overrides={
            name: required_override
            for name in MINIMAL_REQUIRED_ROUTER_NAMES
        },
    )


def _audio_jobs_imports_enabled_for_runtime() -> bool:
    return not _is_explicit_pytest_runtime() or _env_flag_enabled("MINIMAL_TEST_INCLUDE_AUDIO_JOBS")


def _append_first_available_imported_router_spec(
    specs: list[RouterSpec],
    definitions: tuple[ImportedRouterSpec, ...],
) -> None:
    """Append the first imported router whose optional target is available."""
    for definition in definitions:
        candidate_specs: list[RouterSpec] = []
        append_imported_router_spec(candidate_specs, definition)
        if not candidate_specs:
            continue
        candidate_spec = candidate_specs[0]
        try:
            candidate_spec.resolve_router()
        except candidate_spec.skip_exceptions as e:
            spec_name = candidate_spec.name or candidate_spec.route_key or "unnamed"
            context = f" {candidate_spec.skip_context}" if candidate_spec.skip_context else ""
            logger.debug(f"Skipping {spec_name} router{context}: {e}")
            continue
        specs.append(candidate_spec)
        return


def iter_minimal_optional_router_specs() -> Iterable[RouterSpec]:
    """Yield optional minimal-test router specs, skipping unavailable imports."""
    specs: list[RouterSpec] = []
    minimal_skip_context = "in minimal test app"

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
        audio_module_path = "tldw_Server_API.app.api.v1.endpoints.audio.audio"
        for audio_spec in (
            ImportedRouterSpec(
                import_path=audio_module_path,
                log_name="audio",
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio",),
                route_key="audio",
                skip_context=minimal_skip_context,
            ),
            ImportedRouterSpec(
                import_path=audio_module_path,
                log_name="audio-websocket",
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-ws",),
                route_key="audio-websocket",
                attr_name="ws_router",
                skip_context=minimal_skip_context,
            ),
        ):
            append_imported_router_spec(specs, audio_spec)
    else:
        logger.info("Skipping audio routers in minimal test app (set MINIMAL_TEST_INCLUDE_AUDIO=1 to enable)")

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.media",
            log_name="media",
            prefix=f"{API_V1_PREFIX}/media",
            tags=("media",),
            route_key="media",
            skip_context=minimal_skip_context,
        ),
    )

    if _audio_jobs_imports_enabled_for_runtime():
        append_imported_router_spec(
            specs,
            ImportedRouterSpec(
                import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs",
                log_name="audio-jobs",
                prefix=f"{API_V1_PREFIX}/audio",
                tags=("audio-jobs",),
                route_key="audio-jobs",
                skip_context=minimal_skip_context,
            ),
        )
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

    for data_resource_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.files",
            log_name="files",
            prefix=f"{API_V1_PREFIX}",
            tags=("files",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.storage",
            log_name="storage",
            prefix=f"{API_V1_PREFIX}",
            tags=("storage",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.data_tables",
            log_name="data_tables",
            prefix=f"{API_V1_PREFIX}",
            tags=("data-tables",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reading_highlights",
            log_name="reading_highlights",
            prefix=f"{API_V1_PREFIX}",
            tags=("reading-highlights",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.items",
            log_name="items",
            prefix=f"{API_V1_PREFIX}",
            tags=("items",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.reminders",
            log_name="reminders",
            prefix=f"{API_V1_PREFIX}",
            tags=("tasks",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, data_resource_spec)

    for control_support_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.integrations_control_plane",
            log_name="integrations_control_plane",
            prefix=f"{API_V1_PREFIX}",
            tags=("integrations",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane",
            log_name="scheduled_tasks_control_plane",
            prefix=f"{API_V1_PREFIX}",
            tags=("scheduled-tasks",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notifications",
            log_name="notifications",
            prefix=f"{API_V1_PREFIX}",
            tags=("notifications",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chatbooks",
            log_name="chatbooks",
            prefix=f"{API_V1_PREFIX}",
            tags=("chatbooks",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, control_support_spec)

    for workflow_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.workflows",
            log_name="workflows",
            tags=("workflows",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.chat_workflows",
            log_name="chat_workflows",
            tags=("chat-workflows",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.scheduler_workflows",
            log_name="scheduler_workflows",
            tags=("scheduler",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, workflow_spec)

    specs.append(RouterSpec(
        router=evaluations_router_factory,
        prefix=f"{API_V1_PREFIX}",
        tags=("evaluations",),
        route_key="evaluations",
    ))

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.monitoring",
            log_name="monitoring",
            prefix=f"{API_V1_PREFIX}",
            tags=("monitoring",),
            route_key="monitoring",
            skip_context=minimal_skip_context,
        ),
    )

    for experience_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.sharing",
            log_name="sharing",
            prefix=f"{API_V1_PREFIX}",
            tags=("sharing",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.prototype_workspaces",
            log_name="prototype_workspaces",
            prefix=f"{API_V1_PREFIX}",
            tags=("prototype-workspaces",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.personalization",
            log_name="personalization",
            prefix=f"{API_V1_PREFIX}/personalization",
            tags=("personalization",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.companion",
            log_name="companion",
            prefix=f"{API_V1_PREFIX}/companion",
            tags=("companion",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, experience_spec)

    for guardian_safety_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.guardian_controls",
            log_name="guardian_controls",
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=("guardian",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.family_wizard",
            log_name="family_wizard",
            prefix=f"{API_V1_PREFIX}/guardian",
            tags=("guardian",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.self_monitoring",
            log_name="self_monitoring",
            prefix=f"{API_V1_PREFIX}/self-monitoring",
            tags=("self-monitoring",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, guardian_safety_spec)

    for persona_notes_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.persona",
            log_name="persona",
            prefix=f"{API_V1_PREFIX}/persona",
            tags=("persona",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.archetype_endpoints",
            log_name="archetype",
            prefix=f"{API_V1_PREFIX}/persona/archetypes",
            tags=("persona-archetypes",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.notes",
            log_name="notes",
            prefix=f"{API_V1_PREFIX}/notes",
            tags=("notes",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, persona_notes_spec)

    for utility_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.web_clipper",
            log_name="web clipper",
            prefix=f"{API_V1_PREFIX}/web-clipper",
            tags=("web-clipper",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.skills",
            log_name="skills",
            prefix=f"{API_V1_PREFIX}/skills",
            tags=("skills",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.translate",
            log_name="translate",
            prefix=f"{API_V1_PREFIX}",
            tags=("translation",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.slides",
            log_name="slides",
            prefix=f"{API_V1_PREFIX}",
            tags=("slides",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, utility_spec)

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
                skip_context=minimal_skip_context,
            ),
        )

    for study_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.flashcards",
            log_name="flashcards",
            prefix=f"{API_V1_PREFIX}",
            tags=("flashcards",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.quizzes",
            log_name="quizzes",
            prefix=f"{API_V1_PREFIX}",
            tags=("quizzes",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.study_suggestions",
            log_name="study_suggestions",
            prefix=f"{API_V1_PREFIX}",
            tags=("study-suggestions",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, study_spec)

    for writing_email_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.writing",
            log_name="writing",
            prefix=f"{API_V1_PREFIX}/writing",
            tags=("writing",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.writing_manuscripts",
            log_name="manuscripts",
            prefix=f"{API_V1_PREFIX}/writing/manuscripts",
            tags=("manuscripts",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.email",
            log_name="email",
            prefix=f"{API_V1_PREFIX}/email",
            tags=("email",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, writing_email_spec)

    for ops_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.jobs_admin",
            log_name="jobs_admin",
            prefix=f"{API_V1_PREFIX}",
            tags=("jobs",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.audit",
            log_name="audit",
            prefix=f"{API_V1_PREFIX}",
            tags=("audit",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.config_info",
            log_name="config_info",
            prefix=f"{API_V1_PREFIX}",
            tags=("config",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.config_admin",
            log_name="config_admin",
            prefix=f"{API_V1_PREFIX}",
            tags=("config", "admin"),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, ops_spec)

    _append_first_available_imported_router_spec(
        specs,
        (
            ImportedRouterSpec(
                import_path="tldw_Server_API.app.api.v1.endpoints.admin",
                log_name="admin",
                prefix=f"{API_V1_PREFIX}",
                tags=("admin",),
                skip_context=minimal_skip_context,
            ),
            ImportedRouterSpec(
                import_path="tldw_Server_API.app.api.v1.endpoints.admin.admin_byok",
                log_name="admin_byok",
                prefix=f"{API_V1_PREFIX}/admin",
                tags=("admin",),
                skip_context=minimal_skip_context,
            ),
        ),
    )

    for org_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.orgs",
            log_name="orgs",
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.org_invites",
            log_name="org_invites",
            prefix=f"{API_V1_PREFIX}",
            tags=("invites",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, org_spec)

    for access_resource_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.resource_governor",
            log_name="resource_governor",
            prefix=f"{API_V1_PREFIX}",
            tags=("resource-governor",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.users",
            log_name="users",
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, access_resource_spec)

    for byok_shared_keys_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.user_keys",
            log_name="user_keys",
            prefix=f"{API_V1_PREFIX}",
            tags=("users",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped",
            log_name="shared_keys_scoped",
            prefix=f"{API_V1_PREFIX}",
            tags=("organizations",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, byok_shared_keys_spec)

    for mcp_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint",
            log_name="mcp_unified_endpoint",
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-unified",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage",
            log_name="mcp_catalogs_manage",
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-catalogs",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.mcp_hub_management",
            log_name="mcp_hub_management",
            prefix=f"{API_V1_PREFIX}",
            tags=("mcp-hub",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, mcp_spec)

    for privileges_tools_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.privileges",
            log_name="privileges",
            prefix=f"{API_V1_PREFIX}",
            tags=("privileges",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.tools",
            log_name="tools",
            prefix=f"{API_V1_PREFIX}",
            tags=("tools",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, privileges_tools_spec)

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

    for tail_spec in (
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.agent_orchestration",
            log_name="agent_orchestration",
            prefix=f"{API_V1_PREFIX}",
            tags=("agent-orchestration",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.setup",
            log_name="setup",
            prefix=f"{API_V1_PREFIX}",
            tags=("setup",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.metrics",
            log_name="metrics",
            prefix=f"{API_V1_PREFIX}",
            tags=("metrics",),
            skip_context=minimal_skip_context,
        ),
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.authnz_debug",
            log_name="authnz_debug",
            prefix=f"{API_V1_PREFIX}",
            tags=("authnz-debug",),
            skip_context=minimal_skip_context,
        ),
    ):
        append_imported_router_spec(specs, tail_spec)

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
