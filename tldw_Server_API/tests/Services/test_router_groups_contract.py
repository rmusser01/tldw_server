from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Callable
import sys

import pytest
from fastapi import APIRouter, FastAPI

from tldw_Server_API.app.api.v1.router_groups.admin import iter_admin_router_specs
from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_groups.core import iter_core_router_specs
from tldw_Server_API.app.api.v1.router_groups.spec import RouterSpec
from tldw_Server_API.app.api.v1.router_registry import register_router_specs


pytestmark = pytest.mark.unit


def _main_source_text() -> str:
    main_path = Path(__file__).resolve().parents[2] / "app" / "main.py"
    return main_path.read_text(encoding="utf-8")


def _install_fake_router_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    *,
    path: str,
    attr_name: str = "router",
) -> APIRouter:
    router = APIRouter()

    @router.get(path)
    def _endpoint() -> dict[str, str]:
        return {"path": path}

    existing_module = sys.modules.get(module_name)
    fake_module = existing_module if isinstance(existing_module, ModuleType) else ModuleType(module_name)
    setattr(fake_module, attr_name, router)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    return router


def _first_router_path(router: APIRouter | Callable[[], APIRouter]) -> str:
    if not isinstance(router, APIRouter):
        router = router()
    for route in router.routes:
        route_path = getattr(route, "path", None)
        if route_path is not None:
            return str(route_path)
    raise AssertionError("router had no path-bearing routes")


def test_register_router_specs_respects_route_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    router = APIRouter()

    @router.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda route_key, default_stable=True: route_key != "health",
    )

    count = register_router_specs(
        app,
        [
            RouterSpec(router=router, prefix="/api/v1", tags=("health",), route_key="health"),
        ],
    )

    assert count == 0
    assert "/api/v1/health" not in {route.path for route in app.routes}


def test_register_router_specs_resolves_lazy_router_after_route_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    router = APIRouter()
    calls = 0

    @router.get("/lazy")
    def _lazy() -> dict[str, str]:
        return {"status": "ok"}

    def router_factory() -> APIRouter:
        nonlocal calls
        calls += 1
        return router

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda route_key, default_stable=True: route_key != "disabled-lazy",
    )

    disabled_count = register_router_specs(
        app,
        [
            RouterSpec(router=router_factory, prefix="/api/v1", tags=("lazy",), route_key="disabled-lazy"),
        ],
    )
    enabled_count = register_router_specs(
        app,
        [
            RouterSpec(router=router_factory, prefix="/api/v1", tags=("lazy",), route_key="enabled-lazy"),
        ],
    )

    assert disabled_count == 0
    assert enabled_count == 1
    assert calls == 1
    assert "/api/v1/lazy" in {route.path for route in app.routes}


def test_iter_core_router_specs_populates_expected_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.health",
        path="/health",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.auth",
        path="/auth/login",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.authnz_debug",
        path="/auth/debug",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.users",
        path="/users/me",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.user_keys",
        path="/users/keys",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.feedback",
        path="/submit",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.config_info",
        path="/documentation/info",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.moderation",
        path="/moderation/check",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.monitoring",
        path="/monitoring/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.metrics",
        path="/metrics/text",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audit",
        path="/audit/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.consent",
        path="/consent/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.setup",
        path="/setup/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.sync",
        path="/send",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.tools",
        path="/tools",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat",
        path="/chat/completions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat",
        attr_name="conversations_alias_router",
        path="/conversations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat_loop",
        path="/chat/loop/start",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.agent_client_protocol",
        path="/acp/run",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.acp_schedules",
        path="/acp/schedules",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.acp_triggers",
        path="/acp/triggers",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.acp_permissions",
        path="/acp/permissions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.acp_multiplex",
        path="/acp/multiplex",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.llm_providers",
        path="/llm/providers",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.mlx",
        path="/mlx/health",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.messages",
        path="/messages/send",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.messages",
        attr_name="public_router",
        path="/messages/public/send",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.llamacpp",
        path="/llamacpp/completions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.llamacpp",
        attr_name="public_router",
        path="/llamacpp/public/completions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.vlm",
        path="/vlm/models",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint",
        path="/mcp/status",
    )

    specs = list(iter_core_router_specs())
    by_key = {spec.route_key: spec for spec in specs}
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_key["auth"].prefix == "/api/v1"
    assert by_key["auth"].tags == ("authentication",)
    assert by_key["authnz-debug"].prefix == "/api/v1"
    assert by_key["authnz-debug"].tags == ("authnz-debug",)
    assert by_key["authnz-debug"].default_stable is True
    assert by_key["users"].prefix == "/api/v1"
    assert by_key["users"].tags == ("users",)
    assert by_first_path["/users/keys"].prefix == "/api/v1"
    assert by_first_path["/users/keys"].tags == ("users",)
    assert by_first_path["/users/keys"].route_key == "users"
    assert by_key["health"].prefix == "/api/v1"
    assert by_key["health"].tags == ("health",)
    assert by_key["moderation"].prefix == "/api/v1"
    assert by_key["moderation"].tags == ("moderation",)
    assert by_key["monitoring"].prefix == "/api/v1"
    assert by_key["monitoring"].tags == ("monitoring",)
    assert by_key["metrics"].prefix == "/api/v1"
    assert by_key["metrics"].tags == ("metrics",)
    assert by_key["audit"].prefix == "/api/v1"
    assert by_key["audit"].tags == ("audit",)
    assert by_key["consent"].prefix == "/api/v1"
    assert by_key["consent"].tags == ("consent",)
    assert by_key["setup"].prefix == "/api/v1"
    assert by_key["setup"].tags == ("setup",)
    assert by_key["sync"].prefix == "/api/v1/sync"
    assert by_key["sync"].tags == ("sync",)
    assert by_key["tools"].prefix == "/api/v1"
    assert by_key["tools"].tags == ("tools",)
    assert by_key["tools"].default_stable is False
    assert by_first_path["/chat/completions"].prefix == "/api/v1/chat"
    assert by_first_path["/chat/completions"].tags == ()
    assert by_first_path["/chat/completions"].route_key == "chat"
    assert by_first_path["/chat/loop/start"].prefix == "/api/v1"
    assert by_first_path["/chat/loop/start"].tags == ()
    assert by_first_path["/chat/loop/start"].route_key == "chat"
    assert by_first_path["/conversations"].prefix == "/api/v1/chats"
    assert by_first_path["/conversations"].tags == ("chat",)
    assert by_first_path["/conversations"].route_key == "chat"
    assert by_first_path["/acp/run"].prefix == "/api/v1"
    assert by_first_path["/acp/run"].tags == ("acp",)
    assert by_first_path["/acp/run"].route_key == "acp"
    assert by_first_path["/acp/run"].default_stable is False
    assert by_first_path["/acp/schedules"].prefix == "/api/v1"
    assert by_first_path["/acp/schedules"].tags == ("acp-schedules",)
    assert by_first_path["/acp/schedules"].route_key == "acp"
    assert by_first_path["/acp/schedules"].default_stable is False
    assert by_first_path["/acp/triggers"].prefix == "/api/v1"
    assert by_first_path["/acp/triggers"].tags == ("acp-triggers",)
    assert by_first_path["/acp/triggers"].route_key == "acp"
    assert by_first_path["/acp/triggers"].default_stable is False
    assert by_first_path["/acp/permissions"].prefix == "/api/v1"
    assert by_first_path["/acp/permissions"].tags == ("acp-permissions",)
    assert by_first_path["/acp/permissions"].route_key == "acp"
    assert by_first_path["/acp/permissions"].default_stable is False
    assert by_first_path["/acp/multiplex"].prefix == "/api/v1"
    assert by_first_path["/acp/multiplex"].tags == ("acp-multiplex",)
    assert by_first_path["/acp/multiplex"].route_key == "acp"
    assert by_first_path["/acp/multiplex"].default_stable is False
    assert by_key["llm-providers"].prefix == "/api/v1"
    assert by_key["llm-providers"].tags == ("llm",)
    assert by_first_path["/mlx/health"].prefix == "/api/v1"
    assert by_first_path["/mlx/health"].tags == ("llm",)
    assert by_first_path["/mlx/health"].route_key == "llm"
    assert by_first_path["/messages/send"].prefix == "/api/v1"
    assert by_first_path["/messages/send"].tags == ("messages",)
    assert by_first_path["/messages/send"].route_key == "llm"
    assert by_first_path["/messages/public/send"].prefix == ""
    assert by_first_path["/messages/public/send"].tags == ("messages",)
    assert by_first_path["/messages/public/send"].route_key == "llm"
    assert by_first_path["/llamacpp/completions"].prefix == "/api/v1"
    assert by_first_path["/llamacpp/completions"].tags == ("llamacpp",)
    assert by_first_path["/llamacpp/completions"].route_key == "llamacpp"
    assert by_first_path["/llamacpp/public/completions"].prefix == ""
    assert by_first_path["/llamacpp/public/completions"].tags == ("llamacpp",)
    assert by_first_path["/llamacpp/public/completions"].route_key == "llamacpp"
    assert by_key["vlm"].prefix == "/api/v1"
    assert by_key["vlm"].tags == ("vlm",)
    assert by_key["mcp-unified"].prefix == "/api/v1"
    assert by_key["mcp-unified"].tags == ("mcp-unified",)
    assert by_key["feedback"].prefix == "/api/v1/feedback"
    assert by_key["feedback"].tags == ("feedback",)
    assert by_key["config"].prefix == "/api/v1"
    assert by_key["config"].tags == ("config",)


def test_iter_minimal_test_router_specs_populates_expected_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_test_router_specs

    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.research",
        path="/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.research_runs",
        path="/runs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.paper_search",
        path="/papers",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat",
        path="/completions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat",
        attr_name="conversations_alias_router",
        path="/conversations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat_loop",
        path="/chat/loop/start",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions",
        path="/sessions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.character_memory",
        path="/memory",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.character_messages",
        path="/messages",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.workspaces",
        path="/workspaces",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.characters_endpoint",
        path="/characters",
    )

    specs = list(iter_minimal_test_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/search"].prefix == "/api/v1/research"
    assert by_first_path["/search"].tags == ("research",)
    assert by_first_path["/search"].route_key == ""
    assert by_first_path["/runs"].prefix == "/api/v1"
    assert by_first_path["/runs"].tags == ("research-runs",)
    assert by_first_path["/papers"].prefix == "/api/v1/paper-search"
    assert by_first_path["/papers"].tags == ("paper-search",)
    assert by_first_path["/completions"].prefix == "/api/v1/chat"
    assert by_first_path["/completions"].tags == ()
    assert by_first_path["/chat/loop/start"].prefix == "/api/v1"
    assert by_first_path["/chat/loop/start"].tags == ()
    assert by_first_path["/conversations"].prefix == "/api/v1/chats"
    assert by_first_path["/conversations"].tags == ("chat",)
    assert by_first_path["/characters"].prefix == "/api/v1/characters"
    assert by_first_path["/characters"].tags == ("characters",)
    assert by_first_path["/memory"].prefix == "/api/v1/characters"
    assert by_first_path["/memory"].tags == ("character-memory",)
    assert by_first_path["/sessions"].prefix == "/api/v1/chats"
    assert by_first_path["/sessions"].tags == ("character-chat-sessions",)
    assert by_first_path["/messages"].prefix == "/api/v1"
    assert by_first_path["/messages"].tags == ("character-messages",)
    assert by_first_path["/workspaces"].prefix == "/api/v1/workspaces"
    assert by_first_path["/workspaces"].tags == ("workspaces",)


def test_iter_minimal_optional_router_specs_populates_llm_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.llm_providers",
        path="/llm/providers",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.mlx",
        path="/mlx/health",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.vector_stores_openai",
        path="/vector_stores",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced",
        path="/embeddings",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.media_embeddings",
        path="/media_embeddings",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chunking_templates",
        path="/chunking-templates",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompts",
        path="/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.claims",
        path="/claims",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.rag_unified",
        path="/api/v1/rag/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.text2sql",
        path="/text2sql",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.feedback",
        path="/feedback-event",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.vlm",
        path="/vlm/providers",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.rag_health",
        path="/api/v1/rag/health",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.consent",
        path="/consent",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.outputs_templates",
        path="/outputs/templates",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.outputs",
        path="/outputs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.collections_feeds",
        path="/collections/feeds",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.collections_websub",
        path="/collections/websub",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.collections_websub",
        path="/websub/callback",
        attr_name="callback_router",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.slack",
        path="/slack/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.discord",
        path="/discord/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.telegram",
        path="/telegram/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.files",
        path="/files",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.storage",
        path="/storage",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.data_tables",
        path="/data-tables",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.reading_highlights",
        path="/reading-highlights",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.items",
        path="/items",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.reminders",
        path="/reminders",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.integrations_control_plane",
        path="/integrations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane",
        path="/scheduled-tasks",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.notifications",
        path="/notifications",
    )

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/llm/providers"].prefix == "/api/v1"
    assert by_first_path["/llm/providers"].tags == ("llm",)
    assert by_first_path["/llm/providers"].route_key == ""
    assert by_first_path["/mlx/health"].prefix == "/api/v1"
    assert by_first_path["/mlx/health"].tags == ("llm",)
    assert by_first_path["/mlx/health"].route_key == ""
    assert by_first_path["/vector_stores"].prefix == "/api/v1"
    assert by_first_path["/vector_stores"].tags == ("vector-stores",)
    assert by_first_path["/vector_stores"].route_key == ""
    assert by_first_path["/embeddings"].prefix == "/api/v1"
    assert by_first_path["/embeddings"].tags == ("embeddings",)
    assert by_first_path["/embeddings"].route_key == ""
    assert by_first_path["/media_embeddings"].prefix == "/api/v1"
    assert by_first_path["/media_embeddings"].tags == ("media-embeddings",)
    assert by_first_path["/media_embeddings"].route_key == ""
    assert by_first_path["/chunking-templates"].prefix == "/api/v1"
    assert by_first_path["/chunking-templates"].tags == ("chunking-templates",)
    assert by_first_path["/chunking-templates"].route_key == ""
    assert by_first_path["/list"].prefix == "/api/v1/prompts"
    assert by_first_path["/list"].tags == ("prompts",)
    assert by_first_path["/list"].route_key == ""
    assert by_first_path["/claims"].prefix == "/api/v1"
    assert by_first_path["/claims"].tags == ("claims",)
    assert by_first_path["/claims"].route_key == ""
    assert by_first_path["/api/v1/rag/search"].prefix == ""
    assert by_first_path["/api/v1/rag/search"].tags == ("rag-unified",)
    assert by_first_path["/api/v1/rag/search"].route_key == ""
    assert by_first_path["/text2sql"].prefix == "/api/v1"
    assert by_first_path["/text2sql"].tags == ("text2sql",)
    assert by_first_path["/text2sql"].route_key == ""
    assert by_first_path["/feedback-event"].prefix == "/api/v1/feedback"
    assert by_first_path["/feedback-event"].tags == ("feedback",)
    assert by_first_path["/feedback-event"].route_key == ""
    assert by_first_path["/vlm/providers"].prefix == "/api/v1"
    assert by_first_path["/vlm/providers"].tags == ("vlm",)
    assert by_first_path["/vlm/providers"].route_key == ""
    assert by_first_path["/api/v1/rag/health"].prefix == ""
    assert by_first_path["/api/v1/rag/health"].tags == ("rag-health",)
    assert by_first_path["/api/v1/rag/health"].route_key == ""
    assert by_first_path["/consent"].prefix == "/api/v1"
    assert by_first_path["/consent"].tags == ("consent",)
    assert by_first_path["/consent"].route_key == ""
    assert by_first_path["/outputs/templates"].prefix == "/api/v1"
    assert by_first_path["/outputs/templates"].tags == ("outputs-templates",)
    assert by_first_path["/outputs/templates"].route_key == ""
    assert by_first_path["/outputs"].prefix == "/api/v1"
    assert by_first_path["/outputs"].tags == ("outputs",)
    assert by_first_path["/outputs"].route_key == ""
    assert by_first_path["/collections/feeds"].prefix == "/api/v1"
    assert by_first_path["/collections/feeds"].tags == ("collections-feeds",)
    assert by_first_path["/collections/feeds"].route_key == ""
    assert by_first_path["/collections/websub"].prefix == "/api/v1"
    assert by_first_path["/collections/websub"].tags == ("collections-websub",)
    assert by_first_path["/collections/websub"].route_key == ""
    assert by_first_path["/websub/callback"].prefix == "/api/v1"
    assert by_first_path["/websub/callback"].tags == ("collections-websub",)
    assert by_first_path["/websub/callback"].route_key == ""
    assert by_first_path["/slack/events"].prefix == "/api/v1"
    assert by_first_path["/slack/events"].tags == ("slack",)
    assert by_first_path["/slack/events"].route_key == ""
    assert by_first_path["/discord/events"].prefix == "/api/v1"
    assert by_first_path["/discord/events"].tags == ("discord",)
    assert by_first_path["/discord/events"].route_key == ""
    assert by_first_path["/telegram/events"].prefix == "/api/v1"
    assert by_first_path["/telegram/events"].tags == ("telegram",)
    assert by_first_path["/telegram/events"].route_key == ""
    assert by_first_path["/files"].prefix == "/api/v1"
    assert by_first_path["/files"].tags == ("files",)
    assert by_first_path["/files"].route_key == ""
    assert by_first_path["/storage"].prefix == "/api/v1"
    assert by_first_path["/storage"].tags == ("storage",)
    assert by_first_path["/storage"].route_key == ""
    assert by_first_path["/data-tables"].prefix == "/api/v1"
    assert by_first_path["/data-tables"].tags == ("data-tables",)
    assert by_first_path["/data-tables"].route_key == ""
    assert by_first_path["/reading-highlights"].prefix == "/api/v1"
    assert by_first_path["/reading-highlights"].tags == ("reading-highlights",)
    assert by_first_path["/reading-highlights"].route_key == ""
    assert by_first_path["/items"].prefix == "/api/v1"
    assert by_first_path["/items"].tags == ("items",)
    assert by_first_path["/items"].route_key == ""
    assert by_first_path["/reminders"].prefix == "/api/v1"
    assert by_first_path["/reminders"].tags == ("tasks",)
    assert by_first_path["/reminders"].route_key == ""
    assert by_first_path["/integrations"].prefix == "/api/v1"
    assert by_first_path["/integrations"].tags == ("integrations",)
    assert by_first_path["/integrations"].route_key == ""
    assert by_first_path["/scheduled-tasks"].prefix == "/api/v1"
    assert by_first_path["/scheduled-tasks"].tags == ("scheduled-tasks",)
    assert by_first_path["/scheduled-tasks"].route_key == ""
    assert by_first_path["/notifications"].prefix == "/api/v1"
    assert by_first_path["/notifications"].tags == ("notifications",)
    assert by_first_path["/notifications"].route_key == ""


def test_iter_content_router_specs_populates_expected_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.rag_unified",
        path="/api/v1/rag/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompts",
        path="/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.outputs",
        path="/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.email",
        path="/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced",
        path="/embeddings",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.vector_stores_openai",
        path="/vector_stores",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.media_embeddings",
        path="/media_embeddings",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.media",
        path="/media/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified",
        path="/evaluations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.ocr",
        path="/ocr/process",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audio.audio",
        path="/transcriptions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audio.audio",
        attr_name="ws_router",
        path="/stream/transcribe",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs",
        path="/jobs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chunking_templates",
        path="/chunking_templates",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chunking",
        attr_name="chunking_router",
        path="/chunking/split",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.outputs_templates",
        path="/outputs_templates",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.slack",
        path="/slack/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.discord",
        path="/discord/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.telegram",
        path="/telegram/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.meetings",
        path="/meetings/health",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.collections_feeds",
        path="/collections/feeds",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.collections_websub",
        path="/collections/feeds/{feed_id}/websub/subscribe",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.collections_websub",
        attr_name="callback_router",
        path="/websub/callback/{user_id}/{callback_token}",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.reading_highlights",
        attr_name="router",
        path="/reading/items/1/highlights",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.reading",
        path="/reading/items",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_projects",
        path="/api/v1/prompt-studio/projects",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts",
        path="/api/v1/prompt-studio/prompts",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_test_cases",
        path="/api/v1/prompt-studio/test-cases",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization",
        path="/api/v1/prompt-studio/optimizations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_status",
        path="/api/v1/prompt-studio/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations",
        path="/api/v1/prompt-studio/evaluations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket",
        path="/api/v1/prompt-studio/ws",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.workspaces",
        path="/workspaces/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions",
        path="/character-chat-sessions/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.character_memory",
        path="/character-memory/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.characters_endpoint",
        path="/characters/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.character_messages",
        path="/character-messages/send",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audio.audiobooks",
        path="/audiobooks/create",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.voice_assistant",
        path="/sessions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.voice_assistant",
        attr_name="ws_router",
        path="/ws",
    )
    for module_suffix, path in (
        ("kanban_boards", "/boards"),
        ("kanban_lists", "/lists"),
        ("kanban_cards", "/cards"),
        ("kanban_labels", "/labels"),
        ("kanban_checklists", "/checklists"),
        ("kanban_comments", "/comments"),
        ("kanban_search", "/search"),
        ("kanban_links", "/links"),
        ("kanban_workflow", "/workflow"),
    ):
        _install_fake_router_module(
            monkeypatch,
            f"tldw_Server_API.app.api.v1.endpoints.kanban.{module_suffix}",
            path=path,
        )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.connectors",
        path="/connectors/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.ingestion_sources",
        path="/ingestion-sources/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.web_scraping",
        path="/web-scraping/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.notes_graph",
        path="/graph",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.notes",
        path="/notes/list",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.web_clipper",
        path="/capture",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.translate",
        path="/translate",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.slides",
        path="/render",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.flashcards",
        path="/flashcards",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.quizzes",
        path="/quizzes",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.study_suggestions",
        path="/study-suggestions",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.writing",
        path="/writing",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.writing_manuscripts",
        path="/writing/manuscripts",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chatbooks",
        path="/chatbooks",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.workflows",
        path="/api/v1/workflows",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.chat_workflows",
        path="/api/v1/chat-workflows",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.scheduler_workflows",
        path="/api/v1/scheduler/workflows",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.sharing",
        path="/sharing",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.personalization",
        path="/personalization",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.companion",
        path="/companion",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.persona",
        path="/persona",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.archetype_endpoints",
        path="/archetypes",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.files",
        path="/files",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.data_tables",
        path="/data-tables",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.items",
        path="/items",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.reminders",
        path="/tasks",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.notifications",
        path="/notifications",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.watchlists",
        path="/watchlists",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.integrations_control_plane",
        path="/integrations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane",
        path="/scheduled-tasks",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.research",
        path="/research",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.research_runs",
        path="/research-runs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.paper_search",
        path="/paper-search",
    )

    specs = list(iter_content_router_specs())
    by_key = {spec.route_key: spec for spec in specs}
    by_tags = {spec.tags: spec for spec in specs}
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_key["rag"].prefix == ""
    assert by_key["rag"].tags == ("rag-unified",)
    assert by_key["rag-health"].prefix == ""
    assert by_key["rag-health"].tags == ("rag-health",)
    assert by_key["prompts"].prefix == "/api/v1/prompts"
    assert by_key["prompts"].tags == ("prompts",)
    assert by_key["outputs"].prefix == "/api/v1"
    assert by_key["outputs"].tags == ("outputs",)
    assert by_key["claims"].prefix == "/api/v1"
    assert by_key["claims"].tags == ("claims",)
    assert by_key["text2sql"].prefix == "/api/v1"
    assert by_key["text2sql"].tags == ("text2sql",)
    assert by_key["email"].prefix == "/api/v1/email"
    assert by_key["email"].tags == ("email",)
    assert by_key["embeddings"].prefix == "/api/v1"
    assert by_key["embeddings"].tags == ("embeddings",)
    assert by_key["vector-stores"].prefix == "/api/v1"
    assert by_key["vector-stores"].tags == ("vector-stores",)
    assert by_key["media-embeddings"].prefix == "/api/v1"
    assert by_key["media-embeddings"].tags == ("media-embeddings",)
    assert by_key["media"].prefix == "/api/v1/media"
    assert by_key["media"].tags == ("media",)
    assert by_key["audio"].prefix == "/api/v1/audio"
    assert by_key["audio"].tags == ("audio",)
    assert by_key["audio-websocket"].prefix == "/api/v1/audio"
    assert by_key["audio-websocket"].tags == ("audio-websocket",)
    assert by_key["audio-jobs"].prefix == "/api/v1/audio"
    assert by_key["audio-jobs"].tags == ("audio-jobs",)
    assert by_key["evaluations"].prefix == "/api/v1"
    assert by_key["evaluations"].tags == ("evaluations",)
    assert by_key["ocr"].prefix == "/api/v1"
    assert by_key["ocr"].tags == ("ocr",)
    assert by_key["chunking"].prefix == "/api/v1/chunking"
    assert by_key["chunking"].tags == ("chunking",)
    assert by_key["chunking-templates"].prefix == "/api/v1"
    assert by_key["chunking-templates"].tags == ("chunking-templates",)
    assert by_key["outputs-templates"].prefix == "/api/v1"
    assert by_key["outputs-templates"].tags == ("outputs-templates",)
    assert by_key["slack"].prefix == "/api/v1"
    assert by_key["slack"].tags == ("slack",)
    assert by_key["slack"].default_stable is False
    assert by_key["discord"].prefix == "/api/v1"
    assert by_key["discord"].tags == ("discord",)
    assert by_key["discord"].default_stable is False
    assert by_key["telegram"].prefix == "/api/v1"
    assert by_key["telegram"].tags == ("telegram",)
    assert by_key["telegram"].default_stable is False
    assert by_key["meetings"].prefix == "/api/v1"
    assert by_key["meetings"].tags == ("meetings",)
    assert by_key["meetings"].default_stable is False
    assert by_key["collections-feeds"].prefix == "/api/v1"
    assert by_key["collections-feeds"].tags == ("collections-feeds",)
    assert by_first_path["/collections/feeds/{feed_id}/websub/subscribe"].prefix == "/api/v1"
    assert by_first_path["/collections/feeds/{feed_id}/websub/subscribe"].tags == ("collections-websub",)
    assert by_first_path["/collections/feeds/{feed_id}/websub/subscribe"].route_key == "collections-websub"
    assert by_first_path["/websub/callback/{user_id}/{callback_token}"].prefix == "/api/v1"
    assert by_first_path["/websub/callback/{user_id}/{callback_token}"].tags == ("collections-websub",)
    assert by_first_path["/websub/callback/{user_id}/{callback_token}"].route_key == "collections-websub"
    assert by_key["reading"].prefix == "/api/v1"
    assert by_key["reading"].tags == ("reading",)
    assert by_first_path["/api/v1/prompt-studio/projects"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/projects"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/projects"].route_key == "prompt-studio"
    assert by_first_path["/api/v1/prompt-studio/prompts"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/prompts"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/prompts"].route_key == "prompt-studio"
    assert by_first_path["/api/v1/prompt-studio/test-cases"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/test-cases"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/test-cases"].route_key == "prompt-studio"
    assert by_first_path["/api/v1/prompt-studio/optimizations"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/optimizations"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/optimizations"].route_key == "prompt-studio"
    assert by_first_path["/api/v1/prompt-studio/status"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/status"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/status"].route_key == "prompt-studio"
    assert by_first_path["/api/v1/prompt-studio/evaluations"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/evaluations"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/evaluations"].route_key == "prompt-studio"
    assert by_first_path["/api/v1/prompt-studio/ws"].prefix == ""
    assert by_first_path["/api/v1/prompt-studio/ws"].tags == ("prompt-studio",)
    assert by_first_path["/api/v1/prompt-studio/ws"].route_key == "prompt-studio"
    assert by_key["workspaces"].prefix == "/api/v1/workspaces"
    assert by_key["workspaces"].tags == ("workspaces",)
    assert by_key["character-chat-sessions"].prefix == "/api/v1/chats"
    assert by_key["character-chat-sessions"].tags == ("character-chat-sessions",)
    assert by_key["character-memory"].prefix == "/api/v1/characters"
    assert by_key["character-memory"].tags == ("character-memory",)
    assert by_key["characters"].prefix == "/api/v1/characters"
    assert by_key["characters"].tags == ("characters",)
    assert by_key["character-messages"].prefix == "/api/v1"
    assert by_key["character-messages"].tags == ("character-messages",)
    assert by_key["audiobooks"].prefix == "/api/v1"
    assert by_key["audiobooks"].tags == ("audiobooks",)
    assert by_key["audiobooks"].default_stable is False
    assert by_key["voice-assistant"].prefix == "/api/v1/voice"
    assert by_key["voice-assistant"].tags == ("voice-assistant",)
    assert by_key["voice-assistant-ws"].prefix == "/api/v1/voice"
    assert by_key["voice-assistant-ws"].tags == ("voice-assistant-ws",)
    kanban_specs = [spec for spec in specs if spec.route_key == "kanban"]
    assert len(kanban_specs) == 9
    assert {
        (_first_router_path(spec.router), spec.prefix, spec.tags)
        for spec in kanban_specs
    } == {
        ("/boards", "/api/v1/kanban", ("kanban",)),
        ("/lists", "/api/v1/kanban", ("kanban",)),
        ("/cards", "/api/v1/kanban", ("kanban",)),
        ("/labels", "/api/v1/kanban", ("kanban",)),
        ("/checklists", "/api/v1/kanban", ("kanban",)),
        ("/comments", "/api/v1/kanban", ("kanban",)),
        ("/search", "/api/v1/kanban", ("kanban",)),
        ("/links", "/api/v1/kanban", ("kanban",)),
        ("/workflow", "/api/v1/kanban", ("kanban",)),
    }
    assert by_key["connectors"].prefix == "/api/v1"
    assert by_key["connectors"].tags == ("connectors",)
    assert by_key["connectors"].default_stable is False
    assert by_key["ingestion-sources"].prefix == "/api/v1"
    assert by_key["ingestion-sources"].tags == ("ingestion-sources",)
    assert by_key["ingestion-sources"].default_stable is False
    web_scraping_specs = [spec for spec in specs if spec.route_key == "web-scraping"]
    assert len(web_scraping_specs) == 2
    assert {(spec.prefix, spec.tags) for spec in web_scraping_specs} == {
        ("", ("web-scraping",)),
        ("/api/v1", ("web-scraping",)),
    }
    assert by_key["reading-highlights"].prefix == "/api/v1"
    assert by_key["reading-highlights"].tags == ("reading-highlights",)
    assert by_first_path["/graph"].prefix == "/api/v1/notes"
    assert by_first_path["/graph"].tags == ("notes",)
    assert by_first_path["/graph"].route_key == "notes"
    assert by_key["notes"].prefix == "/api/v1/notes"
    assert by_key["notes"].tags == ("notes",)
    content_paths = [_first_router_path(spec.router) for spec in specs]
    assert content_paths.index("/graph") < content_paths.index("/notes/list")
    assert by_key["web-clipper"].prefix == "/api/v1/web-clipper"
    assert by_key["web-clipper"].tags == ("web-clipper",)
    assert by_key["translation"].prefix == "/api/v1"
    assert by_key["translation"].tags == ("translation",)
    assert by_key["slides"].prefix == "/api/v1"
    assert by_key["slides"].tags == ("slides",)
    assert by_key["flashcards"].prefix == "/api/v1"
    assert by_key["flashcards"].tags == ("flashcards",)
    assert by_key["quizzes"].prefix == "/api/v1"
    assert by_key["quizzes"].tags == ("quizzes",)
    assert by_key["study-suggestions"].prefix == "/api/v1"
    assert by_key["study-suggestions"].tags == ("study-suggestions",)
    assert by_key["writing"].prefix == "/api/v1/writing"
    assert by_key["writing"].tags == ("writing",)
    assert by_key["manuscripts"].prefix == "/api/v1/writing/manuscripts"
    assert by_key["manuscripts"].tags == ("manuscripts",)
    assert by_key["chatbooks"].prefix == "/api/v1"
    assert by_key["chatbooks"].tags == ("chatbooks",)
    assert by_first_path["/api/v1/workflows"].prefix == ""
    assert by_first_path["/api/v1/workflows"].tags == ("workflows",)
    assert by_first_path["/api/v1/workflows"].route_key == ""
    assert by_first_path["/api/v1/workflows"].default_stable is False
    assert by_first_path["/api/v1/chat-workflows"].prefix == ""
    assert by_first_path["/api/v1/chat-workflows"].tags == ("chat-workflows",)
    assert by_first_path["/api/v1/chat-workflows"].route_key == ""
    assert by_first_path["/api/v1/scheduler/workflows"].prefix == ""
    assert by_first_path["/api/v1/scheduler/workflows"].tags == ("scheduler",)
    assert by_first_path["/api/v1/scheduler/workflows"].route_key == ""
    assert by_first_path["/api/v1/scheduler/workflows"].default_stable is False
    assert by_key["sharing"].prefix == "/api/v1"
    assert by_key["sharing"].tags == ("sharing",)
    assert by_key["personalization"].prefix == "/api/v1/personalization"
    assert by_key["personalization"].tags == ("personalization",)
    assert by_key["companion"].prefix == "/api/v1/companion"
    assert by_key["companion"].tags == ("companion",)
    assert by_first_path["/persona"].prefix == "/api/v1/persona"
    assert by_first_path["/persona"].tags == ("persona",)
    assert by_first_path["/persona"].route_key == ""
    assert by_first_path["/archetypes"].prefix == "/api/v1/persona/archetypes"
    assert by_first_path["/archetypes"].tags == ("persona-archetypes",)
    assert by_first_path["/archetypes"].route_key == ""
    assert by_key["files"].prefix == "/api/v1"
    assert by_key["files"].tags == ("files",)
    assert by_key["data-tables"].prefix == "/api/v1"
    assert by_key["data-tables"].tags == ("data-tables",)
    assert by_key["items"].prefix == "/api/v1"
    assert by_key["items"].tags == ("items",)
    assert by_key["tasks"].prefix == "/api/v1"
    assert by_key["tasks"].tags == ("tasks",)
    assert by_key["notifications"].prefix == "/api/v1"
    assert by_key["notifications"].tags == ("notifications",)
    assert by_key["watchlists"].prefix == "/api/v1"
    assert by_key["watchlists"].tags == ("watchlists",)
    assert by_key["integrations"].prefix == "/api/v1"
    assert by_key["integrations"].tags == ("integrations",)
    assert by_key["scheduled-tasks"].prefix == "/api/v1"
    assert by_key["scheduled-tasks"].tags == ("scheduled-tasks",)
    assert by_tags[("research",)].prefix == "/api/v1/research"
    assert by_tags[("research",)].tags == ("research",)
    assert by_tags[("research-runs",)].prefix == "/api/v1"
    assert by_tags[("research-runs",)].tags == ("research-runs",)
    assert by_tags[("paper-search",)].prefix == "/api/v1/paper-search"
    assert by_tags[("paper-search",)].tags == ("paper-search",)


def test_iter_admin_router_specs_populates_expected_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.admin",
        path="/admin/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.guardian_controls",
        path="/guardian/controls",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.family_wizard",
        path="/guardian/family-wizard",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.self_monitoring",
        path="/self-monitoring/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.sandbox",
        path="/sandbox/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.billing",
        path="/billing/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.orgs",
        path="/orgs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped",
        path="/shared-keys/scoped",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage",
        path="/mcp/catalogs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.mcp_hub_management",
        path="/mcp/hub",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.org_invites",
        path="/orgs/invites",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.config_admin",
        path="/admin/config/effective",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.resource_governor",
        path="/resource-governor/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.jobs_admin",
        path="/jobs/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.privileges",
        path="/privileges",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.benchmark_api",
        path="/benchmarks/status",
    )

    specs = list(iter_admin_router_specs())
    by_key = {spec.route_key: spec for spec in specs}
    by_tags = {spec.tags: spec for spec in specs}
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_key["admin"].prefix == "/api/v1"
    assert by_key["admin"].tags == ("admin",)
    assert by_first_path["/guardian/controls"].prefix == "/api/v1/guardian"
    assert by_first_path["/guardian/controls"].tags == ("guardian",)
    assert by_first_path["/guardian/controls"].route_key == "guardian"
    assert by_first_path["/guardian/controls"].default_stable is False
    assert by_first_path["/guardian/family-wizard"].prefix == "/api/v1/guardian"
    assert by_first_path["/guardian/family-wizard"].tags == ("guardian",)
    assert by_first_path["/guardian/family-wizard"].route_key == "guardian"
    assert by_first_path["/guardian/family-wizard"].default_stable is False
    assert by_key["self-monitoring"].prefix == "/api/v1/self-monitoring"
    assert by_key["self-monitoring"].tags == ("self-monitoring",)
    assert by_key["self-monitoring"].default_stable is False
    assert by_first_path["/sandbox/status"].prefix == "/api/v1"
    assert by_first_path["/sandbox/status"].tags == ("sandbox",)
    assert by_first_path["/sandbox/status"].route_key == ""
    assert by_first_path["/sandbox/status"].default_stable is False
    assert by_key["billing"].prefix == "/api/v1"
    assert by_key["billing"].tags == ("billing",)
    assert by_key["mcp-catalogs"].prefix == "/api/v1"
    assert by_key["mcp-catalogs"].tags == ()
    assert by_key["mcp-hub"].prefix == "/api/v1"
    assert by_key["mcp-hub"].tags == ("mcp-hub",)
    assert by_key["orgs"].prefix == "/api/v1"
    assert by_key["orgs"].tags == ("organizations",)
    assert by_first_path["/shared-keys/scoped"].prefix == "/api/v1"
    assert by_first_path["/shared-keys/scoped"].tags == ("organizations",)
    assert by_first_path["/shared-keys/scoped"].route_key == "orgs"
    assert by_key["org-invites"].prefix == "/api/v1"
    assert by_key["org-invites"].tags == ("invites",)
    assert by_key["benchmarks"].prefix == "/api/v1"
    assert by_key["benchmarks"].tags == ("benchmarks",)
    assert by_key["benchmarks"].default_stable is False
    assert by_key["jobs"].prefix == "/api/v1"
    assert by_key["jobs"].tags == ("jobs",)
    assert by_key["jobs"].default_stable is False
    assert by_key["resource-governor"].prefix == "/api/v1"
    assert by_key["resource-governor"].tags == ("resource-governor",)
    assert by_key["privileges"].prefix == "/api/v1"
    assert by_key["privileges"].tags == ("privileges",)
    assert by_tags[("config", "admin")].prefix == "/api/v1"
    assert by_tags[("config", "admin")].tags == ("config", "admin")


def test_main_source_does_not_inline_register_grouped_admin_routers() -> None:
    source = _main_source_text()

    assert '_include_if_enabled("admin", admin_router' not in source
    assert '_include_if_enabled("billing", billing_router' not in source
    assert '_include_if_enabled(\n            "guardian",\n            guardian_controls_router_full,' not in source
    assert '_include_if_enabled(\n            "guardian",\n            family_wizard_router_full,' not in source
    assert '_include_if_enabled(\n            "self-monitoring",\n            self_monitoring_router_full,' not in source
    assert "In tests, force-include sandbox endpoints regardless of route policy" not in source
    assert '_include_if_enabled(\n                "sandbox", sandbox_router' not in source
    assert "_HAS_SANDBOX" not in source
    assert '_include_if_enabled("mcp-catalogs", mcp_catalogs_manage_router' not in source
    assert '_include_if_enabled("mcp-hub", mcp_hub_management_router' not in source
    assert '_include_if_enabled("orgs", orgs_router' not in source
    assert '_include_if_enabled("orgs", shared_keys_scoped_router' not in source
    assert '_include_if_enabled("org-invites", org_invites_router' not in source
    assert '"benchmarks", benchmark_router, prefix=f"{API_V1_PREFIX}", tags=["benchmarks"], default_stable=False' not in source
    assert '_include_if_enabled("config", config_admin_router' not in source
    assert '_include_if_enabled(\n            "jobs",\n            jobs_admin_router' not in source
    assert '_include_if_enabled(\n            "resource-governor", resource_governor_router' not in source
    assert '_include_if_enabled("privileges", privileges_router' not in source


def test_main_source_does_not_keep_stale_router_feature_flags() -> None:
    source = _main_source_text()

    assert "Initialize feature flags up-front" not in source
    assert "_HAS_HEALTH" not in source
    assert "_HAS_UNIFIED_EVALUATIONS" not in source
    assert "_HAS_JOBS_ADMIN" not in source
    assert "_HAS_ADMIN_MIN" not in source
    assert "Admin endpoints are used by several pytest modules; import for minimal app" not in source


def test_main_source_does_not_keep_unused_route_include_helper() -> None:
    source = _main_source_text()

    assert "def _include_if_enabled(" not in source
    assert "Route gating error for {route_key}; including by default" not in source


def test_main_source_delegates_full_app_router_imports_to_groups() -> None:
    source = _main_source_text()
    minimal_only_imports = (
        "from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router",
        "from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import router as acp_router",
        "from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router",
        "from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router",
        "from tldw_Server_API.app.api.v1.endpoints.users import router as users_router",
        "from tldw_Server_API.app.api.v1.endpoints.writing import router as writing_router",
        "from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import router as manuscripts_router",
    )

    for import_line in minimal_only_imports:
        assert source.count(import_line) == 1
    assert "from tldw_Server_API.app.api.v1.endpoints.sync import router as sync_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.web_scraping import router as web_scraping_router" not in source
    assert "Tools endpoints unavailable at import time; deferring" not in source
    assert "ACP endpoints unavailable at import time; deferring" not in source
    assert "Users endpoints unavailable at import time; deferring" not in source


def test_main_source_delegates_minimal_research_chat_character_routers_to_group() -> None:
    source = _main_source_text()

    assert "from tldw_Server_API.app.api.v1.router_groups.minimal import (" in source
    assert "iter_minimal_test_router_specs" in source
    assert "iter_minimal_test_router_specs()" in source
    assert "from tldw_Server_API.app.api.v1.endpoints.paper_search import router as paper_search_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.research import router as research_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.research_runs import router as research_runs_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.character_memory import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.character_messages import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as character_router" not in source
    assert "conversations_alias_router" not in source
    assert "router as chat_router" not in source
    assert "router as chat_loop_router" not in source
    assert "include_router_idempotent(app, research_router" not in source
    assert "include_router_idempotent(app, chat_router" not in source
    assert "include_router_idempotent(app, character_router" not in source
    assert "include_router_idempotent(app, workspaces_router" not in source


def test_main_source_delegates_minimal_optional_llm_routers_to_group() -> None:
    source = _main_source_text()

    assert "iter_minimal_optional_router_specs()" in source
    assert "from tldw_Server_API.app.api.v1.endpoints.llm_providers import router as llm_providers_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.mlx import router as mlx_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router as vector_stores_router" not in source
    assert (
        "from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router"
        not in source
    )
    assert "from tldw_Server_API.app.api.v1.endpoints.media_embeddings import router as media_embeddings_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.rag_unified import router as rag_unified_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.text2sql import router as text2sql_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.feedback import router as feedback_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.vlm import router as vlm_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.rag_health import router as rag_health_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.consent import router as consent_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.outputs_templates import router as outputs_templates_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.outputs import router as outputs_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.collections_feeds import router as collections_feeds_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.collections_websub import (" not in source
    assert "callback_router as websub_callback_router" not in source
    assert "router as collections_websub_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.slack import router as slack_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.discord import router as discord_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.telegram import router as telegram_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.files import router as files_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.storage import router as storage_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.data_tables import router as data_tables_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.reading_highlights import router as reading_highlights_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.items import router as items_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.reminders import router as reminders_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.integrations_control_plane import (" not in source
    assert "router as integrations_control_plane_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.scheduled_tasks_control_plane import (" not in source
    assert "router as scheduled_tasks_control_plane_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.notifications import router as notifications_router" not in source
    assert "app.include_router(llm_providers_router" not in source
    assert "app.include_router(mlx_router" not in source
    assert "app.include_router(vector_stores_router" not in source
    assert "app.include_router(embeddings_router" not in source
    assert "app.include_router(media_embeddings_router" not in source
    assert "app.include_router(chunking_templates_router" not in source
    assert "app.include_router(prompt_router" not in source
    assert "app.include_router(claims_router" not in source
    assert "app.include_router(rag_unified_router" not in source
    assert "app.include_router(text2sql_router" not in source
    assert "app.include_router(feedback_router" not in source
    assert "app.include_router(vlm_router" not in source
    assert "app.include_router(rag_health_router" not in source
    assert "app.include_router(consent_router" not in source
    assert "app.include_router(outputs_templates_router" not in source
    assert "app.include_router(outputs_router" not in source
    assert "app.include_router(collections_feeds_router" not in source
    assert "app.include_router(collections_websub_router" not in source
    assert "app.include_router(websub_callback_router" not in source
    assert "app.include_router(slack_router" not in source
    assert "app.include_router(discord_router" not in source
    assert "app.include_router(telegram_router" not in source
    assert "app.include_router(files_router" not in source
    assert "app.include_router(storage_router" not in source
    assert "app.include_router(data_tables_router" not in source
    assert "app.include_router(reading_highlights_router" not in source
    assert "app.include_router(items_router" not in source
    assert "app.include_router(reminders_router" not in source
    assert "app.include_router(integrations_control_plane_router" not in source
    assert "app.include_router(scheduled_tasks_control_plane_router" not in source
    assert "app.include_router(notifications_router" not in source
    assert "Skipping llm_providers router in minimal test app" not in source
    assert "Skipping mlx router in minimal test app" not in source
    assert "Skipping vector-stores router in minimal test app" not in source
    assert "Skipping embeddings router in minimal test app" not in source
    assert "Skipping media_embeddings router in minimal test app" not in source
    assert "Skipping chunking templates router in minimal test app" not in source
    assert "Skipping prompts router in minimal test app" not in source
    assert "Skipping claims router in minimal test app" not in source
    assert "Skipping rag_unified router in minimal test app" not in source
    assert "Skipping text2sql router in minimal test app" not in source
    assert "Skipping feedback router in minimal test app" not in source
    assert "Skipping vlm router in minimal test app" not in source
    assert "Skipping rag_health router in minimal test app" not in source
    assert "Skipping consent router in minimal test app" not in source
    assert "Skipping outputs_templates router in minimal test app" not in source
    assert "Skipping outputs router in minimal test app" not in source
    assert "Skipping collections_feeds router in minimal test app" not in source
    assert "Skipping collections_websub router in minimal test app" not in source
    assert "Skipping slack router in minimal test app" not in source
    assert "Skipping discord router in minimal test app" not in source
    assert "Skipping telegram router in minimal test app" not in source
    assert "Skipping files router in minimal test app" not in source
    assert "Skipping storage router in minimal test app" not in source
    assert "Skipping data_tables router in minimal test app" not in source
    assert "Skipping reading_highlights router in minimal test app" not in source
    assert "Skipping items router in minimal test app" not in source
    assert "Skipping reminders router in minimal test app" not in source
    assert "Skipping integrations control plane router in minimal test app" not in source
    assert "Skipping scheduled tasks control plane router in minimal test app" not in source
    assert "Skipping notifications router in minimal test app" not in source


def test_main_source_does_not_inline_register_grouped_core_routers() -> None:
    source = _main_source_text()

    assert '_include_if_enabled("auth", auth_router' not in source
    assert '_include_if_enabled(\n            "authnz-debug",\n            authnz_debug_router,' not in source
    assert '_include_if_enabled("users", users_router' not in source
    assert '_include_if_enabled("users", user_keys_router' not in source
    assert '"health", health_router, prefix=f"{API_V1_PREFIX}", tags=["health"]' not in source
    assert '_include_if_enabled("moderation", moderation_router' not in source
    assert '_include_if_enabled("monitoring", monitoring_router' not in source
    assert '_include_if_enabled("metrics", metrics_router' not in source
    assert '_include_if_enabled("audit", audit_router' not in source
    assert '_include_if_enabled("consent", consent_router' not in source
    assert '_include_if_enabled("setup", setup_router' not in source
    assert '_include_if_enabled("sync", sync_router' not in source
    assert '_include_if_enabled("mcp-unified", mcp_unified_router' not in source
    assert '_include_if_enabled("feedback", feedback_router' not in source
    assert '_include_if_enabled("config", config_info_router' not in source
    assert '_include_if_enabled("tools", tools_router, prefix=f"{API_V1_PREFIX}", tags=["tools"], default_stable=False)' not in source
    assert '_include_if_enabled("chat", chat_router, prefix=f"{API_V1_PREFIX}/chat")' not in source
    assert '_include_if_enabled("chat", chat_loop_router, prefix=f"{API_V1_PREFIX}")' not in source
    assert '_include_if_enabled("chat", conversations_alias_router, prefix=f"{API_V1_PREFIX}/chats", tags=["chat"])' not in source
    assert '_include_if_enabled("acp", acp_router, prefix=f"{API_V1_PREFIX}", tags=["acp"], default_stable=False)' not in source
    assert '_include_if_enabled("acp", acp_schedules_router, prefix=f"{API_V1_PREFIX}", tags=["acp-schedules"], default_stable=False)' not in source
    assert '_include_if_enabled("acp", acp_triggers_router, prefix=f"{API_V1_PREFIX}", tags=["acp-triggers"], default_stable=False)' not in source
    assert '_include_if_enabled("acp", acp_permissions_router, prefix=f"{API_V1_PREFIX}", tags=["acp-permissions"], default_stable=False)' not in source
    assert '_include_if_enabled("acp", acp_multiplex_router, prefix=f"{API_V1_PREFIX}", tags=["acp-multiplex"], default_stable=False)' not in source
    assert '_include_if_enabled("llm", llm_providers_router, prefix=f"{API_V1_PREFIX}", tags=["llm"])' not in source
    assert '_include_if_enabled("llm", mlx_router, prefix=f"{API_V1_PREFIX}", tags=["llm"])' not in source
    assert '_include_if_enabled("llm", messages_router, prefix=f"{API_V1_PREFIX}", tags=["messages"])' not in source
    assert '_include_if_enabled("llm", messages_public_router, prefix="", tags=["messages"])' not in source
    assert '_include_if_enabled("llamacpp", llamacpp_router, prefix=f"{API_V1_PREFIX}", tags=["llamacpp"])' not in source
    assert '_include_if_enabled("llamacpp", llamacpp_public_router, prefix="", tags=["llamacpp"])' not in source


def test_main_source_does_not_inline_register_grouped_content_routers() -> None:
    source = _main_source_text()

    assert '_include_if_enabled("rag", rag_unified_router' not in source
    assert '_include_if_enabled("rag-health", rag_health_router' not in source
    assert '_include_if_enabled("chunking", chunking_router' not in source
    assert '_include_if_enabled("slack", slack_router' not in source
    assert '_include_if_enabled("discord", discord_router' not in source
    assert '_include_if_enabled("telegram", telegram_router' not in source
    assert '_include_if_enabled("meetings", meetings_router' not in source
    assert '_include_if_enabled("collections-feeds", collections_feeds_router' not in source
    assert '_include_if_enabled("collections-websub", collections_websub_router' not in source
    assert '_include_if_enabled("collections-websub", websub_callback_router' not in source
    assert '_include_if_enabled("reading", _reading_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_projects_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_prompts_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_test_cases_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_optimization_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_status_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_evaluations_router' not in source
    assert '_include_if_enabled("prompt-studio", prompt_studio_websocket_router' not in source
    assert '_include_if_enabled(\n            "workspaces", workspaces_router, prefix=f"{API_V1_PREFIX}/workspaces", tags=["workspaces"]' not in source
    assert '_include_if_enabled("characters", character_router, prefix=f"{API_V1_PREFIX}/characters", tags=["characters"])' not in source
    assert '_include_if_enabled(\n            "character-chat-sessions",\n            character_chat_sessions_router,' not in source
    assert '_include_if_enabled(\n            "character-memory", character_memory_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character-memory"]' not in source
    assert '_include_if_enabled(\n            "character-messages", character_messages_router, prefix=f"{API_V1_PREFIX}", tags=["character-messages"]' not in source
    assert '_include_if_enabled(\n            "audiobooks",\n            audiobooks_router,' not in source
    assert '_include_if_enabled(\n            "voice-assistant", voice_assistant_router' not in source
    assert '_include_if_enabled(\n            "voice-assistant-ws",\n            voice_assistant_ws_router,' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_boards_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_lists_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_cards_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_labels_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_checklists_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_comments_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_search_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_links_router' not in source
    assert '_include_if_enabled(\n            "kanban", kanban_workflow_router' not in source
    assert '_include_if_enabled(\n            "connectors", connectors_router, prefix=f"{API_V1_PREFIX}", tags=["connectors"], default_stable=False' not in source
    assert '_include_if_enabled(\n        "ingestion-sources",\n        ingestion_sources_router,' not in source
    assert '_include_if_enabled("web-scraping", web_scraping_router, tags=["web-scraping"])' not in source
    assert '_include_if_enabled("web-scraping", web_scraping_router, prefix=f"{API_V1_PREFIX}", tags=["web-scraping"])' not in source
    assert '_include_if_enabled("reading-highlights", reading_highlights_router' not in source
    assert '_include_if_enabled("embeddings", embeddings_router' not in source
    assert '_include_if_enabled("media-embeddings", media_embeddings_router' not in source
    assert '_include_if_enabled("vector-stores", vector_stores_router' not in source
    assert '_include_if_enabled("chunking-templates", chunking_templates_router' not in source
    assert '_include_if_enabled("prompts", prompt_router' not in source
    assert '_include_if_enabled("claims", claims_router' not in source
    assert '_include_if_enabled("text2sql", text2sql_router' not in source
    assert '_include_if_enabled("email", email_router' not in source
    assert '_include_if_enabled("outputs-templates", outputs_templates_router' not in source
    assert '_include_if_enabled("outputs", outputs_router' not in source
    assert '_include_if_enabled(\n            "notes", notes_graph_router' not in source
    assert '_include_if_enabled("notes", notes_router' not in source
    assert '_include_if_enabled("web-clipper", web_clipper_router' not in source
    assert '_include_if_enabled("translation", translate_router' not in source
    assert '_include_if_enabled("slides", slides_router' not in source
    assert '"flashcards", flashcards_router' not in source
    assert '"quizzes", quizzes_router' not in source
    assert '"study-suggestions",\n        study_suggestions_router' not in source
    assert '"writing", writing_router' not in source
    assert '"manuscripts", manuscripts_router' not in source
    assert '"chatbooks", chatbooks_router' not in source
    assert "_HAS_WORKFLOWS" not in source
    assert "_HAS_CHAT_WORKFLOWS" not in source
    assert "_HAS_SCHEDULER_WF" not in source
    assert "_HAS_CHUNKING" not in source
    assert "_HAS_MEETINGS" not in source
    assert "_HAS_COLLECTIONS_FEEDS" not in source
    assert "_HAS_COLLECTIONS_WEBSUB" not in source
    assert "_HAS_SLACK" not in source
    assert "_HAS_DISCORD" not in source
    assert "_HAS_TELEGRAM" not in source
    assert "_HAS_FILES" not in source
    assert "_HAS_DATA_TABLES" not in source
    assert "_HAS_READING_HIGHLIGHTS" not in source
    assert "_HAS_ITEMS" not in source
    assert "_HAS_PROMPT_STUDIO" not in source
    assert "Prompt Studio endpoints unavailable; skipping import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.character_memory import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.character_messages import" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.workspaces import router as workspaces_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as character_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.chat import (" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.chat_loop import (" not in source
    assert source.count("from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router") == 1
    assert "ingestion_sources_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.web_scraping import" not in source
    assert "_reading_router" not in source
    assert "MINIMAL_TEST_INCLUDE_READING" not in source
    assert "_HAS_MEDIA" not in source
    assert '_include_if_enabled("media", media_router' not in source
    assert "_HAS_AUDIO" not in source
    assert "_HAS_AUDIO_JOBS" not in source
    assert "_full_audio_import_enabled" not in source
    assert '_include_if_enabled("audio", audio_router' not in source
    assert '_include_if_enabled("audio-jobs", audio_jobs_router' not in source
    assert '_include_if_enabled(\n            "audio-websocket", audio_ws_router' not in source
    assert source.count("from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import") == 1
    assert source.count('app.include_router(_evaluations_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])') == 1
    assert "Route gating error for evaluations; skipping import" not in source
    assert 'route_enabled("ocr")' not in source
    assert "endpoints.ocr import router as _ocr_router" not in source
    assert "In test contexts, force-include workflows regardless of policy to avoid 404s." not in source
    assert '_include_if_enabled("workflows", workflows_router' not in source
    assert '_include_if_enabled("chat-workflows", chat_workflows_router' not in source
    assert '_include_if_enabled("scheduler", scheduler_workflows_router' not in source
    assert 'route_key in {"workflows", "scheduler"}' not in source
    assert "force-include certain routes" not in source
    assert '"sharing", sharing_router' not in source
    assert '"personalization",\n        personalization_router' not in source
    assert '"companion",\n        companion_router' not in source
    assert "In tests, force-include persona endpoints regardless of route policy for WS/unit coverage" not in source
    assert '_include_if_enabled(\n            "persona", persona_router' not in source
    assert "Archetype template endpoints are always available (read-only catalog data)" not in source
    assert "archetype_router  # noqa: F811" not in source
    assert "Admin config endpoint unavailable; skipping import" not in source
    assert '_include_if_enabled("files", _files_router' not in source
    assert '_include_if_enabled("data-tables", _data_tables_router' not in source
    assert '_include_if_enabled("items", _items_router' not in source
    assert '_include_if_enabled("tasks", _reminders_router' not in source
    assert '_include_if_enabled("notifications", _notifications_router' not in source
    assert '_include_if_enabled("watchlists", _watchlists_router' not in source
    assert '"integrations",\n            _integrations_control_plane_router,' not in source
    assert '"scheduled-tasks",\n            _scheduled_tasks_control_plane_router,' not in source
    assert '_include_if_enabled("research", research_router' not in source
    assert '_include_if_enabled("research", research_runs_router' not in source
    assert '"paper-search", paper_search_router' not in source
