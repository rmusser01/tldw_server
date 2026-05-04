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

    fake_module = sys.modules.get(module_name)
    if (
        not isinstance(fake_module, ModuleType)
        or not getattr(fake_module, "_router_test_fake", False)
    ):
        fake_module = ModuleType(module_name)
        setattr(fake_module, "_router_test_fake", True)
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


def test_install_fake_router_module_does_not_mutate_existing_real_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify test fakes replace real modules instead of mutating them in place."""
    module_name = "tldw_Server_API.app.api.v1.endpoints.fake_existing_real"
    real_module = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, real_module)

    router = _install_fake_router_module(
        monkeypatch,
        module_name,
        path="/fake-existing-real",
    )

    assert not hasattr(real_module, "router")
    assert sys.modules[module_name] is not real_module
    assert getattr(sys.modules[module_name], "router") is router


def test_append_imported_router_spec_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify imported router specs retain the metadata used for registration."""
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    specs: list[RouterSpec] = []
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.acp_schedules",
        path="/acp/schedules",
    )

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.acp_schedules",
            log_name="acp_schedules",
            prefix="/api/v1",
            tags=("acp-schedules",),
            route_key="acp",
            default_stable=False,
        ),
    )

    assert len(specs) == 1
    assert _first_router_path(specs[0].router) == "/acp/schedules"
    assert specs[0].prefix == "/api/v1"
    assert specs[0].tags == ("acp-schedules",)
    assert specs[0].route_key == "acp"
    assert specs[0].default_stable is False


def test_append_imported_router_spec_defers_router_attr_lookup_until_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify router attribute lookup stays lazy until RouterSpec resolution."""
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    module_name = "tldw_Server_API.app.api.v1.endpoints.lazy_router"
    access_count = {"router": 0}
    fake_module = ModuleType(module_name)
    router = APIRouter()

    @router.get("/lazy/router")
    def _lazy_router() -> dict[str, str]:
        return {"status": "ok"}

    def _module_getattr(name: str) -> APIRouter:
        if name != "router":
            raise AttributeError(name)
        access_count["router"] += 1
        return router

    fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs: list[RouterSpec] = []
    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path=module_name,
            log_name="lazy-router",
            prefix="/api/v1",
            tags=("lazy-router",),
        ),
    )

    assert len(specs) == 1
    assert access_count["router"] == 0
    assert _first_router_path(specs[0].router) == "/lazy/router"
    assert access_count["router"] == 1


def test_append_imported_router_spec_defers_module_import_until_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify route policy can disable imported specs before module import."""
    import importlib
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    module_name = "tldw_Server_API.app.api.v1.endpoints.policy_gated_router"
    import_count = 0
    router = APIRouter()
    fake_module = ModuleType(module_name)

    @router.get("/policy-gated")
    def _policy_gated() -> dict[str, str]:
        return {"status": "ok"}

    fake_module.router = router  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    real_import_module = importlib.import_module

    def _import_module(import_path: str) -> ModuleType:
        nonlocal import_count
        if import_path == module_name:
            import_count += 1
        return real_import_module(import_path)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )
    specs: list[RouterSpec] = []

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path=module_name,
            log_name="policy-gated",
            prefix="/api/v1",
            tags=("policy-gated",),
            route_key="policy-gated",
        ),
    )

    assert len(specs) == 1
    assert import_count == 0

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda route_key, default_stable=True: route_key != "policy-gated",
    )

    disabled_app = FastAPI()
    assert register_router_specs(disabled_app, specs) == 0
    assert import_count == 0

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda *_args, **_kwargs: True,
    )

    enabled_app = FastAPI()
    assert register_router_specs(enabled_app, specs) == 1
    assert import_count == 1
    assert "/api/v1/policy-gated" in {route.path for route in enabled_app.routes}


def test_append_imported_router_spec_skips_optional_import_error_at_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unavailable optional router modules are skipped during registration."""
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    def _raise_import_error(_import_path: str) -> ModuleType:
        raise ModuleNotFoundError("optional router is unavailable")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _raise_import_error,
    )
    specs: list[RouterSpec] = []
    debug_messages: list[str] = []

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.optional_missing",
            log_name="optional-missing",
            skip_context="in minimal test app",
        ),
    )

    monkeypatch.setattr("loguru.logger.debug", debug_messages.append)

    assert len(specs) == 1
    assert register_router_specs(FastAPI(), specs) == 0
    assert debug_messages == [
        "Skipping optional-missing router in minimal test app: optional router is unavailable"
    ]


def test_append_imported_router_spec_logs_unexpected_import_error_at_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unexpected import-time failures are logged by registration."""
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    def _raise_runtime_error(_import_path: str) -> ModuleType:
        raise RuntimeError("router module crashed during import")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _raise_runtime_error,
    )
    specs: list[RouterSpec] = []
    debug_messages: list[str] = []

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path="tldw_Server_API.app.api.v1.endpoints.crashing_router",
            log_name="crashing-router",
        ),
    )

    monkeypatch.setattr("loguru.logger.debug", debug_messages.append)

    assert len(specs) == 1
    assert register_router_specs(FastAPI(), specs) == 0
    assert debug_messages == ["Skipping crashing-router router: router module crashed during import"]


def test_append_imported_router_spec_skips_static_missing_attr_at_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify static modules missing router attrs are skipped during registration."""
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    module_name = "tldw_Server_API.app.api.v1.endpoints.missing_router_attr"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    specs: list[RouterSpec] = []
    debug_messages: list[str] = []

    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path=module_name,
            log_name="missing-router-attr",
            prefix="/api/v1",
            tags=("missing-router-attr",),
        ),
    )

    monkeypatch.setattr("loguru.logger.debug", debug_messages.append)

    assert len(specs) == 1
    assert register_router_specs(FastAPI(), specs) == 0
    assert debug_messages == [
        f"Skipping missing-router-attr router: {module_name}.router"
    ]


def test_append_imported_router_spec_logs_missing_lazy_attr_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify lazy attr misses produce one registry-owned skip log."""
    from tldw_Server_API.app.api.v1.router_groups.conditional import (
        ImportedRouterSpec,
        append_imported_router_spec,
    )

    module_name = "tldw_Server_API.app.api.v1.endpoints.missing_lazy_router_attr"
    fake_module = ModuleType(module_name)
    debug_messages: list[str] = []

    def _module_getattr(name: str) -> APIRouter:
        raise AttributeError(name)

    fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs: list[RouterSpec] = []
    append_imported_router_spec(
        specs,
        ImportedRouterSpec(
            import_path=module_name,
            log_name="missing-lazy-router-attr",
            prefix="/api/v1",
            tags=("missing-lazy-router-attr",),
        ),
    )

    monkeypatch.setattr("loguru.logger.debug", debug_messages.append)

    assert len(specs) == 1
    assert register_router_specs(FastAPI(), specs) == 0
    assert debug_messages == [
        f"Skipping missing-lazy-router-attr router: {module_name}.router"
    ]


def test_iter_core_router_specs_defers_chat_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered core chat specs keep router attr lookup lazy."""
    chat_module_name = "tldw_Server_API.app.api.v1.endpoints.chat"
    chat_loop_module_name = "tldw_Server_API.app.api.v1.endpoints.chat_loop"
    access_count = {
        "chat.router": 0,
        "chat.conversations_alias_router": 0,
        "chat_loop.router": 0,
    }

    chat_router = APIRouter()
    conversations_router = APIRouter()
    chat_loop_router = APIRouter()

    @chat_router.get("/chat/completions")
    def _chat_completions() -> dict[str, str]:
        return {"status": "ok"}

    @conversations_router.get("/conversations")
    def _conversations() -> dict[str, str]:
        return {"status": "ok"}

    @chat_loop_router.get("/chat/loop/start")
    def _chat_loop_start() -> dict[str, str]:
        return {"status": "ok"}

    chat_module = ModuleType(chat_module_name)

    def _chat_getattr(name: str) -> APIRouter:
        if name == "router":
            access_count["chat.router"] += 1
            return chat_router
        if name == "conversations_alias_router":
            access_count["chat.conversations_alias_router"] += 1
            return conversations_router
        raise AttributeError(name)

    chat_module.__getattr__ = _chat_getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, chat_module_name, chat_module)

    chat_loop_module = ModuleType(chat_loop_module_name)

    def _chat_loop_getattr(name: str) -> APIRouter:
        if name == "router":
            access_count["chat_loop.router"] += 1
            return chat_loop_router
        raise AttributeError(name)

    chat_loop_module.__getattr__ = _chat_loop_getattr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, chat_loop_module_name, chat_loop_module)

    specs = list(iter_core_router_specs())
    assert access_count == {
        "chat.router": 0,
        "chat.conversations_alias_router": 0,
        "chat_loop.router": 0,
    }

    by_meta = {
        (spec.prefix, spec.tags): spec
        for spec in specs
        if spec.route_key == "chat"
    }

    expected_paths_by_meta = {
        ("/api/v1/chat", ()): "/chat/completions",
        ("/api/v1", ()): "/chat/loop/start",
        ("/api/v1/chats", ("chat",)): "/conversations",
    }
    assert {
        meta: _first_router_path(spec.router)
        for meta, spec in by_meta.items()
        if meta in expected_paths_by_meta
    } == expected_paths_by_meta
    assert access_count == {
        "chat.router": 1,
        "chat.conversations_alias_router": 1,
        "chat_loop.router": 1,
    }


def test_iter_core_router_specs_defers_llm_provider_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered core LLM/provider specs keep router attr lookup lazy."""
    module_paths = {
        "tldw_Server_API.app.api.v1.endpoints.llm_providers": {
            "router": "/llm/providers",
        },
        "tldw_Server_API.app.api.v1.endpoints.mlx": {
            "router": "/mlx/health",
        },
        "tldw_Server_API.app.api.v1.endpoints.messages": {
            "router": "/messages/send",
            "public_router": "/messages/public/send",
        },
        "tldw_Server_API.app.api.v1.endpoints.llamacpp": {
            "router": "/llamacpp/completions",
            "public_router": "/llamacpp/public/completions",
        },
        "tldw_Server_API.app.api.v1.endpoints.vlm": {
            "router": "/vlm/models",
        },
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint": {
            "router": "/mcp/status",
        },
    }
    access_count = {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    for module_name, attr_paths in module_paths.items():
        fake_module = ModuleType(module_name)
        routers: dict[str, APIRouter] = {}
        for attr_name, path in attr_paths.items():
            router = APIRouter()

            @router.get(path)
            def _endpoint() -> dict[str, str]:
                return {"status": "ok"}

            routers[attr_name] = router

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            routers: dict[str, APIRouter] = routers,
        ) -> APIRouter:
            if name not in routers:
                raise AttributeError(name)
            access_count[f"{module_name}.{name}"] += 1
            return routers[name]

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_core_router_specs())
    assert access_count == {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    selected_specs = [
        spec
        for spec in specs
        if spec.route_key in {"llm", "llamacpp", "vlm", "mcp-unified"}
    ]
    by_first_path = {_first_router_path(spec.router): spec for spec in selected_specs}

    assert by_first_path["/llm/providers"].prefix == "/api/v1"
    assert by_first_path["/llm/providers"].tags == ("llm",)
    assert by_first_path["/llm/providers"].route_key == "llm"
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
    assert by_first_path["/vlm/models"].prefix == "/api/v1"
    assert by_first_path["/vlm/models"].tags == ("vlm",)
    assert by_first_path["/vlm/models"].route_key == "vlm"
    assert by_first_path["/mcp/status"].prefix == "/api/v1"
    assert by_first_path["/mcp/status"].tags == ("mcp-unified",)
    assert by_first_path["/mcp/status"].route_key == "mcp-unified"
    assert access_count == {
        f"{module_name}.{attr_name}": 1
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }


def test_iter_admin_router_specs_defers_selected_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered admin specs keep router attr lookup lazy."""
    import importlib

    sandbox_module_name = "tldw_Server_API.app.api.v1.endpoints.sandbox"
    router_definitions = {
        "tldw_Server_API.app.api.v1.endpoints.admin": "/admin/status",
        "tldw_Server_API.app.api.v1.endpoints.family_wizard": "/guardian/family-wizard",
        "tldw_Server_API.app.api.v1.endpoints.guardian_controls": "/guardian/controls",
        "tldw_Server_API.app.api.v1.endpoints.self_monitoring": "/self-monitoring/status",
        sandbox_module_name: "/sandbox/status",
        "tldw_Server_API.app.api.v1.endpoints.billing": "/billing/status",
        "tldw_Server_API.app.api.v1.endpoints.benchmark_api": "/benchmarks/status",
        "tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage": "/mcp/catalogs",
        "tldw_Server_API.app.api.v1.endpoints.mcp_hub_management": "/mcp/hub",
        "tldw_Server_API.app.api.v1.endpoints.orgs": "/orgs",
        "tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped": "/shared-keys/scoped",
        "tldw_Server_API.app.api.v1.endpoints.privileges": "/privileges",
        "tldw_Server_API.app.api.v1.endpoints.config_admin": "/admin/config/effective",
        "tldw_Server_API.app.api.v1.endpoints.resource_governor": "/resource-governor/status",
        "tldw_Server_API.app.api.v1.endpoints.jobs_admin": "/jobs/status",
        "tldw_Server_API.app.api.v1.endpoints.org_invites": "/orgs/invites",
    }
    access_count = {module_name: 0 for module_name in router_definitions}
    real_import_module = importlib.import_module

    def _guarded_import_module(module_name: str, package: str | None = None) -> ModuleType:
        if module_name == sandbox_module_name and module_name not in sys.modules:
            raise AssertionError("sandbox router was imported without a test stub")
        return real_import_module(module_name, package)

    monkeypatch.setattr(importlib, "import_module", _guarded_import_module)

    for module_name, path in router_definitions.items():
        router = APIRouter()

        @router.get(path)
        def _endpoint() -> dict[str, str]:
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_admin_router_specs())
    assert access_count == {module_name: 0 for module_name in router_definitions}

    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    assert set(router_definitions.values()).issubset(by_first_path)
    assert access_count == {module_name: 1 for module_name in router_definitions}


def test_iter_core_router_specs_skips_crashing_chat_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify chat import crashes skip only the affected router at registration."""
    import importlib

    app = FastAPI()
    chat_module_name = "tldw_Server_API.app.api.v1.endpoints.chat"
    chat_loop_module_name = "tldw_Server_API.app.api.v1.endpoints.chat_loop"
    chat_module = ModuleType(chat_module_name)
    chat_router = APIRouter()
    conversations_router = APIRouter()
    debug_messages: list[str] = []

    @chat_router.get("/chat/completions")
    def _chat_completions() -> dict[str, str]:
        return {"status": "ok"}

    @conversations_router.get("/conversations")
    def _conversations() -> dict[str, str]:
        return {"status": "ok"}

    chat_module.router = chat_router  # type: ignore[attr-defined]
    chat_module.conversations_alias_router = conversations_router  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, chat_module_name, chat_module)

    real_import_module = importlib.import_module

    def _import_module(module_name: str) -> ModuleType:
        if module_name == chat_loop_module_name:
            raise RuntimeError("chat loop crashed during import")
        return real_import_module(module_name)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "loguru.logger.debug",
        debug_messages.append,
    )

    specs = list(iter_core_router_specs())
    chat_specs = [
        spec
        for spec in specs
        if spec.route_key == "chat"
    ]
    count = register_router_specs(app, chat_specs)
    chat_paths = {route.path for route in app.routes}

    assert count == 2
    assert "/api/v1/chat/chat/completions" in chat_paths
    assert "/api/v1/chats/conversations" in chat_paths
    assert "/api/v1/chat/loop/start" not in chat_paths
    assert "Skipping chat_loop router: chat loop crashed during import" in debug_messages


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


def test_register_router_specs_logs_spec_name_for_resolution_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    debug_messages: list[str] = []

    def router_factory() -> APIRouter:
        raise RuntimeError("lazy router failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.route_enabled",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr("loguru.logger.debug", debug_messages.append)

    count = register_router_specs(
        app,
        [
            RouterSpec(
                router=router_factory,
                prefix="/api/v1",
                tags=("lazy",),
                route_key="chat",
                name="chat_loop",
            ),
        ],
    )

    assert count == 0
    assert debug_messages == ["Skipping chat_loop router: lazy router failed"]


def test_register_router_specs_fails_closed_when_route_policy_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    router = APIRouter()

    @router.get("/guarded")
    def _guarded() -> dict[str, str]:
        return {"status": "ok"}

    def _raise_policy_error(route_key: str, default_stable: bool = True) -> bool:
        raise RuntimeError(f"policy failed for {route_key}")

    monkeypatch.setattr("tldw_Server_API.app.core.config.route_enabled", _raise_policy_error)

    count = register_router_specs(
        app,
        [
            RouterSpec(router=router, prefix="/api/v1", tags=("guarded",), route_key="guarded"),
        ],
    )

    assert count == 0
    assert "/api/v1/guarded" not in {route.path for route in app.routes}


def test_register_router_specs_deduplicates_factory_routers_by_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    calls = 0

    def router_factory() -> APIRouter:
        nonlocal calls
        calls += 1
        router = APIRouter()

        @router.get("/factory")
        def _factory() -> dict[str, str]:
            return {"status": "ok"}

        return router

    monkeypatch.setattr("tldw_Server_API.app.core.config.route_enabled", lambda *_args, **_kwargs: True)

    spec = RouterSpec(router=router_factory, prefix="/api/v1", tags=("factory",), route_key="factory")

    assert register_router_specs(app, [spec]) == 1
    assert register_router_specs(app, [spec]) == 0
    assert calls == 1
    assert [route.path for route in app.routes].count("/api/v1/factory") == 1


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
    assert by_first_path["/llm/providers"].prefix == "/api/v1"
    assert by_first_path["/llm/providers"].tags == ("llm",)
    assert by_first_path["/llm/providers"].route_key == "llm"
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
    assert by_key["vlm"].route_key == "vlm"
    assert by_key["mcp-unified"].prefix == "/api/v1"
    assert by_key["mcp-unified"].tags == ("mcp-unified",)
    assert by_key["feedback"].prefix == "/api/v1/feedback"
    assert by_key["feedback"].tags == ("feedback",)
    assert by_key["config"].prefix == "/api/v1"
    assert by_key["config"].tags == ("config",)


def test_iter_content_router_specs_uses_canonical_rag_key_and_single_web_scraping_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.rag_unified",
        path="/api/v1/rag/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.web_scraping",
        path="/api/v1/web/scrape",
    )

    specs = list(iter_content_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    web_scraping_specs = [spec for spec in specs if spec.route_key == "web-scraping"]

    assert by_first_path["/api/v1/rag/search"].route_key == "rag-unified"
    assert len(web_scraping_specs) == 1
    assert web_scraping_specs[0].prefix == "/api/v1"


def test_iter_content_router_specs_defers_discovery_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered content discovery specs keep router attr lookup lazy."""
    router_definitions = {
        "tldw_Server_API.app.api.v1.endpoints.rag_health": "/rag/health",
        "tldw_Server_API.app.api.v1.endpoints.research": "/research/search",
        "tldw_Server_API.app.api.v1.endpoints.research_runs": "/research-runs",
        "tldw_Server_API.app.api.v1.endpoints.paper_search": "/paper-search",
    }
    access_count = {module_name: 0 for module_name in router_definitions}

    for module_name, path in router_definitions.items():
        router = APIRouter()

        @router.get(path)
        def _endpoint() -> dict[str, str]:
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_content_router_specs())
    assert access_count == {module_name: 0 for module_name in router_definitions}

    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    assert set(router_definitions.values()).issubset(by_first_path)
    assert access_count == {module_name: 1 for module_name in router_definitions}


def test_iter_content_router_specs_defers_media_audio_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered media/audio specs keep router attr lookup lazy."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_media_audio")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    module_paths = {
        "tldw_Server_API.app.api.v1.endpoints.media": {"router": "/media/list"},
        "tldw_Server_API.app.api.v1.endpoints.audio.audio": {
            "router": "/transcriptions",
            "ws_router": "/stream/transcribe",
        },
        "tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs": {"router": "/audio/jobs"},
    }
    access_count = {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    for module_name, attr_paths in module_paths.items():
        fake_module = ModuleType(module_name)
        routers: dict[str, APIRouter] = {}
        for attr_name, path in attr_paths.items():
            router = APIRouter()

            @router.get(path)
            def _endpoint() -> dict[str, str]:
                return {"status": "ok"}

            routers[attr_name] = router

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            routers: dict[str, APIRouter] = routers,
        ) -> APIRouter:
            if name not in routers:
                raise AttributeError(name)
            access_count[f"{module_name}.{name}"] += 1
            return routers[name]

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_content_router_specs())
    assert access_count == {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    selected_specs = [
        spec
        for spec in specs
        if spec.route_key in {"media", "audio", "audio-websocket", "audio-jobs"}
    ]
    by_first_path = {_first_router_path(spec.router): spec for spec in selected_specs}

    assert by_first_path["/media/list"].prefix == "/api/v1/media"
    assert by_first_path["/media/list"].tags == ("media",)
    assert by_first_path["/transcriptions"].prefix == "/api/v1/audio"
    assert by_first_path["/transcriptions"].tags == ("audio",)
    assert by_first_path["/stream/transcribe"].prefix == "/api/v1/audio"
    assert by_first_path["/stream/transcribe"].tags == ("audio-websocket",)
    assert by_first_path["/audio/jobs"].prefix == "/api/v1/audio"
    assert by_first_path["/audio/jobs"].tags == ("audio-jobs",)
    assert access_count == {
        f"{module_name}.{attr_name}": 1
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }


def test_iter_content_router_specs_defers_workflow_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered workflow specs keep router attr lookup lazy."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_workflow_routers")
    module_paths = {
        "tldw_Server_API.app.api.v1.endpoints.workflows": "/workflows",
        "tldw_Server_API.app.api.v1.endpoints.chat_workflows": "/chat/workflows",
        "tldw_Server_API.app.api.v1.endpoints.scheduler_workflows": "/scheduler/workflows",
    }
    access_count = {module_name: 0 for module_name in module_paths}

    for module_name, path in module_paths.items():
        router = APIRouter()

        @router.get(path)
        def _endpoint() -> dict[str, str]:
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_content_router_specs())
    assert access_count == {module_name: 0 for module_name in module_paths}

    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    assert by_first_path["/workflows"].prefix == ""
    assert by_first_path["/workflows"].tags == ("workflows",)
    assert by_first_path["/workflows"].route_key == ""
    assert by_first_path["/workflows"].default_stable is False
    assert by_first_path["/chat/workflows"].prefix == ""
    assert by_first_path["/chat/workflows"].tags == ("chat-workflows",)
    assert by_first_path["/chat/workflows"].route_key == ""
    assert by_first_path["/scheduler/workflows"].prefix == ""
    assert by_first_path["/scheduler/workflows"].tags == ("scheduler",)
    assert by_first_path["/scheduler/workflows"].route_key == ""
    assert by_first_path["/scheduler/workflows"].default_stable is False
    assert access_count == {module_name: 1 for module_name in module_paths}


def test_iter_content_router_specs_defers_processing_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered processing/prompt specs keep router attr lookup lazy."""
    module_paths = {
        "tldw_Server_API.app.api.v1.endpoints.chunking": {
            "chunking_router": "/chunk",
        },
        "tldw_Server_API.app.api.v1.endpoints.vector_stores_openai": {
            "router": "/vector_stores",
        },
        "tldw_Server_API.app.api.v1.endpoints.chunking_templates": {
            "router": "/chunking/templates",
        },
        "tldw_Server_API.app.api.v1.endpoints.prompts": {
            "router": "/prompts",
        },
    }
    access_count = {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    for module_name, attr_paths in module_paths.items():
        fake_module = ModuleType(module_name)
        routers: dict[str, APIRouter] = {}
        for attr_name, path in attr_paths.items():
            router = APIRouter()

            @router.get(path)
            def _endpoint() -> dict[str, str]:
                return {"status": "ok"}

            routers[attr_name] = router

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            routers: dict[str, APIRouter] = routers,
        ) -> APIRouter:
            if name not in routers:
                raise AttributeError(name)
            access_count[f"{module_name}.{name}"] += 1
            return routers[name]

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_content_router_specs())
    assert access_count == {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    assert by_first_path["/chunk"].prefix == "/api/v1/chunking"
    assert by_first_path["/chunk"].tags == ("chunking",)
    assert by_first_path["/chunk"].route_key == "chunking"
    assert by_first_path["/vector_stores"].prefix == "/api/v1"
    assert by_first_path["/vector_stores"].tags == ("vector-stores",)
    assert by_first_path["/vector_stores"].route_key == "vector-stores"
    assert by_first_path["/chunking/templates"].prefix == "/api/v1"
    assert by_first_path["/chunking/templates"].tags == ("chunking-templates",)
    assert by_first_path["/chunking/templates"].route_key == "chunking-templates"
    assert by_first_path["/prompts"].prefix == "/api/v1/prompts"
    assert by_first_path["/prompts"].tags == ("prompts",)
    assert by_first_path["/prompts"].route_key == "prompts"
    assert access_count == {
        f"{module_name}.{attr_name}": 1
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }


def test_iter_content_router_specs_defers_embedding_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered embedding specs keep router attr lookup lazy."""
    module_paths = {
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced": {
            "router": "/embeddings",
        },
        "tldw_Server_API.app.api.v1.endpoints.media_embeddings": {
            "router": "/media_embeddings",
        },
    }
    access_count = {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    for module_name, attr_paths in module_paths.items():
        fake_module = ModuleType(module_name)
        routers: dict[str, APIRouter] = {}
        for attr_name, path in attr_paths.items():
            router = APIRouter()

            @router.get(path)
            def _endpoint() -> dict[str, str]:
                return {"status": "ok"}

            routers[attr_name] = router

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            routers: dict[str, APIRouter] = routers,
        ) -> APIRouter:
            if name not in routers:
                raise AttributeError(name)
            access_count[f"{module_name}.{name}"] += 1
            return routers[name]

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_content_router_specs())
    assert access_count == {
        f"{module_name}.{attr_name}": 0
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }

    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    assert by_first_path["/embeddings"].prefix == "/api/v1"
    assert by_first_path["/embeddings"].tags == ("embeddings",)
    assert by_first_path["/embeddings"].route_key == "embeddings"
    assert by_first_path["/media_embeddings"].prefix == "/api/v1"
    assert by_first_path["/media_embeddings"].tags == ("media-embeddings",)
    assert by_first_path["/media_embeddings"].route_key == "media-embeddings"
    assert access_count == {
        f"{module_name}.{attr_name}": 1
        for module_name, attrs in module_paths.items()
        for attr_name in attrs
    }


def test_iter_content_router_specs_defers_utility_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered utility content specs keep import and attr lookup lazy."""
    import importlib

    router_definitions = {
        "claims": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.claims",
            "path": "/claims",
            "prefix": "/api/v1",
            "tags": ("claims",),
        },
        "text2sql": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.text2sql",
            "path": "/text2sql/query",
            "prefix": "/api/v1",
            "tags": ("text2sql",),
        },
        "email": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.email",
            "path": "/email/search",
            "prefix": "/api/v1/email",
            "tags": ("email",),
        },
        "outputs-templates": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.outputs_templates",
            "path": "/outputs/templates",
            "prefix": "/api/v1",
            "tags": ("outputs-templates",),
        },
    }
    access_count = {
        str(definition["module_name"]): 0
        for definition in router_definitions.values()
    }
    import_calls: list[str] = []

    for definition in router_definitions.values():
        module_name = str(definition["module_name"])
        router = APIRouter()

        @router.get(str(definition["path"]))
        def _endpoint() -> dict[str, str]:
            """Return a deterministic response for the fake utility router."""
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            """Track lazy router attribute resolution for the fake module."""
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    real_import_module = importlib.import_module

    def _import_module(module_name: str, package: str | None = None) -> ModuleType:
        """Track lazy module imports for selected utility router modules."""
        if module_name in access_count:
            import_calls.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )

    specs = list(iter_content_router_specs())
    assert access_count == {
        str(definition["module_name"]): 0
        for definition in router_definitions.values()
    }
    assert import_calls == []

    utility_specs = {
        spec.route_key: spec
        for spec in specs
        if spec.route_key in router_definitions
    }
    assert set(utility_specs) == set(router_definitions)

    for route_key, spec in utility_specs.items():
        definition = router_definitions[route_key]
        assert spec.prefix == definition["prefix"]
        assert spec.tags == definition["tags"]
        assert _first_router_path(spec.router) == definition["path"]

    assert import_calls == [
        str(router_definitions[route_key]["module_name"])
        for route_key in utility_specs
    ]
    assert access_count == {
        str(definition["module_name"]): 1
        for definition in router_definitions.values()
    }


def test_iter_content_router_specs_defers_output_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered output specs keep router attr lookup lazy."""
    module_paths = {
        "tldw_Server_API.app.api.v1.endpoints.outputs_templates": "/outputs/templates",
        "tldw_Server_API.app.api.v1.endpoints.outputs": "/outputs",
    }
    access_count = {module_name: 0 for module_name in module_paths}

    for module_name, path in module_paths.items():
        router = APIRouter()

        @router.get(path)
        def _endpoint() -> dict[str, str]:
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    specs = list(iter_content_router_specs())
    assert access_count == {module_name: 0 for module_name in module_paths}

    by_first_path = {_first_router_path(spec.router): spec for spec in specs}
    assert by_first_path["/outputs/templates"].prefix == "/api/v1"
    assert by_first_path["/outputs/templates"].tags == ("outputs-templates",)
    assert by_first_path["/outputs/templates"].route_key == "outputs-templates"
    assert by_first_path["/outputs"].prefix == "/api/v1"
    assert by_first_path["/outputs"].tags == ("outputs",)
    assert by_first_path["/outputs"].route_key == "outputs"
    assert access_count == {module_name: 1 for module_name in module_paths}


def test_iter_content_router_specs_defers_integration_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify covered integration specs keep import and attr lookup lazy."""
    import importlib

    router_definitions = {
        "slack": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.slack",
            "path": "/slack/events",
            "prefix": "/api/v1",
            "tags": ("slack",),
        },
        "discord": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.discord",
            "path": "/discord/events",
            "prefix": "/api/v1",
            "tags": ("discord",),
        },
        "telegram": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.telegram",
            "path": "/telegram/events",
            "prefix": "/api/v1",
            "tags": ("telegram",),
        },
        "meetings": {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.meetings",
            "path": "/meetings/health",
            "prefix": "/api/v1",
            "tags": ("meetings",),
        },
    }
    access_count = {
        str(definition["module_name"]): 0
        for definition in router_definitions.values()
    }
    import_calls: list[str] = []

    for definition in router_definitions.values():
        module_name = str(definition["module_name"])
        router = APIRouter()

        @router.get(str(definition["path"]))
        def _endpoint() -> dict[str, str]:
            """Return a deterministic response for the fake integration router."""
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            """Track lazy router attribute resolution for the fake module."""
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    real_import_module = importlib.import_module

    def _import_module(module_name: str, package: str | None = None) -> ModuleType:
        """Track lazy module imports for selected integration router modules."""
        if module_name in access_count:
            import_calls.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )

    specs = list(iter_content_router_specs())
    assert access_count == {
        str(definition["module_name"]): 0
        for definition in router_definitions.values()
    }
    assert import_calls == []

    integration_specs = {
        spec.route_key: spec
        for spec in specs
        if spec.route_key in router_definitions
    }
    assert set(integration_specs) == set(router_definitions)

    for route_key, spec in integration_specs.items():
        definition = router_definitions[route_key]
        assert spec.prefix == definition["prefix"]
        assert spec.tags == definition["tags"]
        assert spec.default_stable is False
        assert _first_router_path(spec.router) == definition["path"]

    assert import_calls == [
        str(router_definitions[route_key]["module_name"])
        for route_key in integration_specs
    ]
    assert access_count == {
        str(definition["module_name"]): 1
        for definition in router_definitions.values()
    }


def test_iter_content_router_specs_defers_collections_reading_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collections and reading specs keep import and attr lookup lazy."""
    import importlib

    router_definitions = [
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.collections_feeds",
            "attr_name": "router",
            "expected_name": "collections_feeds",
            "expected_skip_context": "",
            "path": "/collections/feeds",
            "prefix": "/api/v1",
            "tags": ("collections-feeds",),
            "route_key": "collections-feeds",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.collections_websub",
            "attr_name": "router",
            "expected_name": "collections_websub",
            "expected_skip_context": "",
            "path": "/collections/feeds/{feed_id}/websub/subscribe",
            "prefix": "/api/v1",
            "tags": ("collections-websub",),
            "route_key": "collections-websub",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.collections_websub",
            "attr_name": "callback_router",
            "expected_name": "collections_websub_callback",
            "expected_skip_context": "(callback_router)",
            "path": "/websub/callback/{user_id}/{callback_token}",
            "prefix": "/api/v1",
            "tags": ("collections-websub",),
            "route_key": "collections-websub",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.reading",
            "attr_name": "router",
            "expected_name": "reading",
            "expected_skip_context": "",
            "path": "/reading/items",
            "prefix": "/api/v1",
            "tags": ("reading",),
            "route_key": "reading",
        },
    ]
    access_count = {
        f"{definition['module_name']}.{definition['attr_name']}": 0
        for definition in router_definitions
    }
    routers_by_module: dict[str, dict[str, APIRouter]] = {}
    import_calls: list[str] = []

    for definition in router_definitions:
        module_name = str(definition["module_name"])
        attr_name = str(definition["attr_name"])
        router = APIRouter()

        @router.get(str(definition["path"]))
        def _endpoint() -> dict[str, str]:
            """Return a deterministic response for the fake collections router."""
            return {"status": "ok"}

        routers_by_module.setdefault(module_name, {})[attr_name] = router

    for module_name, routers in routers_by_module.items():
        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            routers: dict[str, APIRouter] = routers,
        ) -> APIRouter:
            """Track lazy router attribute resolution for the fake module."""
            if name not in routers:
                raise AttributeError(name)
            access_count[f"{module_name}.{name}"] += 1
            return routers[name]

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    real_import_module = importlib.import_module

    def _import_module(module_name: str, package: str | None = None) -> ModuleType:
        """Track lazy module imports for selected collections router modules."""
        if module_name in routers_by_module:
            import_calls.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )

    specs = list(iter_content_router_specs())
    assert access_count == {
        f"{definition['module_name']}.{definition['attr_name']}": 0
        for definition in router_definitions
    }
    assert import_calls == []

    selected_specs = [
        spec
        for spec in specs
        if spec.route_key in {"collections-feeds", "collections-websub", "reading"}
    ]
    assert len(selected_specs) == len(router_definitions)

    by_first_path = {_first_router_path(spec.router): spec for spec in selected_specs}
    for definition in router_definitions:
        spec = by_first_path[str(definition["path"])]
        assert spec.prefix == definition["prefix"]
        assert spec.tags == definition["tags"]
        assert spec.route_key == definition["route_key"]
        assert spec.name == definition["expected_name"]
        assert spec.skip_context == definition["expected_skip_context"]

    assert import_calls == [
        str(definition["module_name"])
        for definition in router_definitions
    ]
    assert access_count == {
        f"{definition['module_name']}.{definition['attr_name']}": 1
        for definition in router_definitions
    }


def test_iter_content_router_specs_defers_prompt_studio_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Prompt Studio specs keep import and attr lookup lazy."""
    import importlib

    router_definitions = [
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_projects",
            "expected_name": "prompt_studio_projects",
            "path": "/api/v1/prompt-studio/projects",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts",
            "expected_name": "prompt_studio_prompts",
            "path": "/api/v1/prompt-studio/prompts",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_test_cases",
            "expected_name": "prompt_studio_test_cases",
            "path": "/api/v1/prompt-studio/test-cases",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_optimization",
            "expected_name": "prompt_studio_optimization",
            "path": "/api/v1/prompt-studio/optimizations",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_status",
            "expected_name": "prompt_studio_status",
            "path": "/api/v1/prompt-studio/status",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations",
            "expected_name": "prompt_studio_evaluations",
            "path": "/api/v1/prompt-studio/evaluations",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket",
            "expected_name": "prompt_studio_websocket",
            "path": "/api/v1/prompt-studio/ws",
        },
    ]
    access_count = {str(definition["module_name"]): 0 for definition in router_definitions}
    import_calls: list[str] = []

    for definition in router_definitions:
        module_name = str(definition["module_name"])
        router = APIRouter()

        @router.get(str(definition["path"]))
        def _endpoint() -> dict[str, str]:
            """Return a deterministic response for the fake Prompt Studio router."""
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            """Track lazy router attribute resolution for the fake module."""
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    real_import_module = importlib.import_module

    def _import_module(module_name: str, package: str | None = None) -> ModuleType:
        """Track lazy module imports for selected Prompt Studio router modules."""
        if module_name in access_count:
            import_calls.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )

    specs = list(iter_content_router_specs())
    assert access_count == {
        str(definition["module_name"]): 0 for definition in router_definitions
    }
    assert import_calls == []

    selected_specs = [
        spec
        for spec in specs
        if spec.name and spec.name.startswith("prompt_studio_")
    ]
    assert len(selected_specs) == len(router_definitions)

    by_first_path = {_first_router_path(spec.router): spec for spec in selected_specs}
    for definition in router_definitions:
        spec = by_first_path[str(definition["path"])]
        assert spec.prefix == ""
        assert spec.tags == ("prompt-studio",)
        assert spec.route_key == "prompt-studio"
        assert spec.name == definition["expected_name"]

    assert import_calls == [
        str(definition["module_name"])
        for definition in router_definitions
    ]
    assert access_count == {
        str(definition["module_name"]): 1 for definition in router_definitions
    }


def test_iter_content_router_specs_defers_workspace_character_router_attr_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify workspace and character specs keep import and attr lookup lazy."""
    import importlib

    router_definitions = [
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.workspaces",
            "expected_name": "workspaces",
            "path": "/workspaces/list",
            "prefix": "/api/v1/workspaces",
            "tags": ("workspaces",),
            "route_key": "workspaces",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions",
            "expected_name": "character_chat_sessions",
            "path": "/character-chat-sessions/list",
            "prefix": "/api/v1/chats",
            "tags": ("character-chat-sessions",),
            "route_key": "character-chat-sessions",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.character_memory",
            "expected_name": "character_memory",
            "path": "/character-memory/list",
            "prefix": "/api/v1/characters",
            "tags": ("character-memory",),
            "route_key": "character-memory",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.characters_endpoint",
            "expected_name": "characters",
            "path": "/characters/list",
            "prefix": "/api/v1/characters",
            "tags": ("characters",),
            "route_key": "characters",
        },
        {
            "module_name": "tldw_Server_API.app.api.v1.endpoints.character_messages",
            "expected_name": "character_messages",
            "path": "/character-messages/send",
            "prefix": "/api/v1",
            "tags": ("character-messages",),
            "route_key": "character-messages",
        },
    ]
    access_count = {str(definition["module_name"]): 0 for definition in router_definitions}
    import_calls: list[str] = []

    for definition in router_definitions:
        module_name = str(definition["module_name"])
        router = APIRouter()

        @router.get(str(definition["path"]))
        def _endpoint() -> dict[str, str]:
            """Return a deterministic response for the fake character router."""
            return {"status": "ok"}

        fake_module = ModuleType(module_name)

        def _module_getattr(
            name: str,
            *,
            module_name: str = module_name,
            router: APIRouter = router,
        ) -> APIRouter:
            """Track lazy router attribute resolution for the fake module."""
            if name != "router":
                raise AttributeError(name)
            access_count[module_name] += 1
            return router

        fake_module.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, fake_module)

    real_import_module = importlib.import_module

    def _import_module(module_name: str, package: str | None = None) -> ModuleType:
        """Track lazy module imports for selected workspace/character routers."""
        if module_name in access_count:
            import_calls.append(module_name)
        return real_import_module(module_name, package)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.router_groups.conditional.importlib.import_module",
        _import_module,
    )

    specs = list(iter_content_router_specs())
    assert access_count == {
        str(definition["module_name"]): 0 for definition in router_definitions
    }
    assert import_calls == []

    selected_specs = [
        spec
        for spec in specs
        if spec.route_key
        in {
            "workspaces",
            "character-chat-sessions",
            "character-memory",
            "characters",
            "character-messages",
        }
    ]
    assert len(selected_specs) == len(router_definitions)

    by_first_path = {_first_router_path(spec.router): spec for spec in selected_specs}
    for definition in router_definitions:
        spec = by_first_path[str(definition["path"])]
        assert spec.prefix == definition["prefix"]
        assert spec.tags == definition["tags"]
        assert spec.route_key == definition["route_key"]
        assert spec.name == definition["expected_name"]

    assert import_calls == [
        str(definition["module_name"])
        for definition in router_definitions
    ]
    assert access_count == {
        str(definition["module_name"]): 1 for definition in router_definitions
    }


def test_qodo_reviewed_router_policy_regressions(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.rag_unified",
        path="/api/v1/rag/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.llm_providers",
        path="/llm/providers",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.vlm",
        path="/vlm/backends",
    )

    content_specs = list(iter_content_router_specs())
    core_specs = list(iter_core_router_specs())
    by_content_path = {_first_router_path(spec.router): spec for spec in content_specs}
    by_core_path = {
        _first_router_path(spec.router): spec
        for spec in core_specs
        if spec.tags in {("llm",), ("vlm",)} and spec.prefix == "/api/v1"
    }
    main_source = _main_source_text()

    assert by_content_path["/api/v1/rag/search"].route_key == "rag-unified"
    assert by_core_path["/llm/providers"].route_key == "llm"
    assert by_core_path["/vlm/backends"].route_key == "vlm"
    assert "from tldw_Server_API.app.api.v1.endpoints.vlm import router as vlm_router" not in main_source
    assert "app.include_router(vlm_router" not in main_source


def test_iter_admin_router_specs_keeps_independent_guardian_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.family_wizard",
        path="/family-wizard",
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.api.v1.endpoints.guardian_controls",
        ModuleType("tldw_Server_API.app.api.v1.endpoints.guardian_controls"),
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.self_monitoring",
        path="/self-monitoring/status",
    )

    specs = list(iter_admin_router_specs())
    guardian_specs = {
        spec.name: spec
        for spec in specs
        if spec.name in {"family_wizard", "guardian_controls", "self_monitoring"}
    }
    by_first_path = {
        _first_router_path(spec.router): spec
        for spec in (
            guardian_specs["family_wizard"],
            guardian_specs["self_monitoring"],
        )
    }

    assert by_first_path["/family-wizard"].prefix == "/api/v1/guardian"
    assert by_first_path["/family-wizard"].route_key == "guardian"
    assert by_first_path["/self-monitoring/status"].prefix == "/api/v1/self-monitoring"
    assert by_first_path["/self-monitoring/status"].route_key == "self-monitoring"
    with pytest.raises(AttributeError, match="router"):
        guardian_specs["guardian_controls"].resolve_router()


def test_iter_admin_router_specs_uses_policy_key_for_sandbox_in_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_sandbox_policy")
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.sandbox",
        path="/sandbox/status",
    )

    specs = list(iter_admin_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/sandbox/status"].route_key == "sandbox"
    assert by_first_path["/sandbox/status"].default_stable is False


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


def test_iter_minimal_test_router_specs_includes_health_and_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_test_router_specs

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

    specs = list(iter_minimal_test_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/health"].prefix == "/api/v1"
    assert by_first_path["/health"].tags == ("health",)
    assert by_first_path["/health"].route_key == ""
    assert by_first_path["/auth/login"].prefix == "/api/v1"
    assert by_first_path["/auth/login"].tags == ("authentication",)
    assert by_first_path["/auth/login"].route_key == ""


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
        "tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified",
        path="/evaluations",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.monitoring",
        path="/monitoring/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.sharing",
        path="/sharing",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.personalization",
        path="/personalization/profile",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.companion",
        path="/companion/profile",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.guardian_controls",
        path="/controls",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.family_wizard",
        path="/family-wizard",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.self_monitoring",
        path="/self-monitoring/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.persona",
        path="/persona/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.archetype_endpoints",
        path="/archetypes",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.notes",
        path="/notes",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.web_clipper",
        path="/web-clipper",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.skills",
        path="/skills",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.translate",
        path="/translate",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.slides",
        path="/slides",
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
        "tldw_Server_API.app.api.v1.endpoints.email",
        path="/email/search",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.jobs_admin",
        path="/jobs/health",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audit",
        path="/audit/events",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.config_info",
        path="/documentation/info",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.config_admin",
        path="/config/effective",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.admin",
        path="/admin/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.admin.admin_byok",
        path="/keys/shared",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.orgs",
        path="/orgs",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.org_invites",
        path="/orgs/invites",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.resource_governor",
        path="/resource-governor/status",
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
        "tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped",
        path="/organizations/shared-keys",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint",
        path="/mcp/status",
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
        "tldw_Server_API.app.api.v1.endpoints.privileges",
        path="/privileges",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.tools",
        path="/tools",
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
        "tldw_Server_API.app.api.v1.endpoints.agent_orchestration",
        path="/agent-orchestration",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.setup",
        path="/setup/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.metrics",
        path="/metrics/text",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.authnz_debug",
        path="/authnz-debug/status",
    )
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.sandbox",
        path="/sandbox/status",
    )

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/llm/providers"].prefix == "/api/v1"
    assert by_first_path["/llm/providers"].tags == ("llm",)
    assert by_first_path["/llm/providers"].route_key == ""
    assert by_first_path["/mlx/health"].prefix == "/api/v1"
    assert by_first_path["/mlx/health"].tags == ("llm",)
    assert by_first_path["/mlx/health"].route_key == ""
    assert by_first_path["/messages/send"].prefix == "/api/v1"
    assert by_first_path["/messages/send"].tags == ("messages",)
    assert by_first_path["/messages/send"].route_key == ""
    assert by_first_path["/messages/public/send"].prefix == ""
    assert by_first_path["/messages/public/send"].tags == ("messages",)
    assert by_first_path["/messages/public/send"].route_key == ""
    assert by_first_path["/llamacpp/completions"].prefix == "/api/v1"
    assert by_first_path["/llamacpp/completions"].tags == ("llamacpp",)
    assert by_first_path["/llamacpp/completions"].route_key == ""
    assert by_first_path["/llamacpp/public/completions"].prefix == ""
    assert by_first_path["/llamacpp/public/completions"].tags == ("llamacpp",)
    assert by_first_path["/llamacpp/public/completions"].route_key == ""
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
    assert by_first_path["/chatbooks"].prefix == "/api/v1"
    assert by_first_path["/chatbooks"].tags == ("chatbooks",)
    assert by_first_path["/chatbooks"].route_key == ""
    assert by_first_path["/api/v1/workflows"].prefix == ""
    assert by_first_path["/api/v1/workflows"].tags == ("workflows",)
    assert by_first_path["/api/v1/workflows"].route_key == ""
    assert by_first_path["/api/v1/chat-workflows"].prefix == ""
    assert by_first_path["/api/v1/chat-workflows"].tags == ("chat-workflows",)
    assert by_first_path["/api/v1/chat-workflows"].route_key == ""
    assert by_first_path["/api/v1/scheduler/workflows"].prefix == ""
    assert by_first_path["/api/v1/scheduler/workflows"].tags == ("scheduler",)
    assert by_first_path["/api/v1/scheduler/workflows"].route_key == ""
    assert by_first_path["/evaluations"].prefix == "/api/v1"
    assert by_first_path["/evaluations"].tags == ("evaluations",)
    assert by_first_path["/evaluations"].route_key == "evaluations"
    assert by_first_path["/monitoring/status"].prefix == "/api/v1"
    assert by_first_path["/monitoring/status"].tags == ("monitoring",)
    assert by_first_path["/monitoring/status"].route_key == "monitoring"
    assert by_first_path["/sharing"].prefix == "/api/v1"
    assert by_first_path["/sharing"].tags == ("sharing",)
    assert by_first_path["/sharing"].route_key == ""
    assert by_first_path["/personalization/profile"].prefix == "/api/v1/personalization"
    assert by_first_path["/personalization/profile"].tags == ("personalization",)
    assert by_first_path["/personalization/profile"].route_key == ""
    assert by_first_path["/companion/profile"].prefix == "/api/v1/companion"
    assert by_first_path["/companion/profile"].tags == ("companion",)
    assert by_first_path["/companion/profile"].route_key == ""
    assert by_first_path["/controls"].prefix == "/api/v1/guardian"
    assert by_first_path["/controls"].tags == ("guardian",)
    assert by_first_path["/controls"].route_key == ""
    assert by_first_path["/family-wizard"].prefix == "/api/v1/guardian"
    assert by_first_path["/family-wizard"].tags == ("guardian",)
    assert by_first_path["/family-wizard"].route_key == ""
    assert by_first_path["/self-monitoring/status"].prefix == "/api/v1/self-monitoring"
    assert by_first_path["/self-monitoring/status"].tags == ("self-monitoring",)
    assert by_first_path["/self-monitoring/status"].route_key == ""
    assert by_first_path["/persona/status"].prefix == "/api/v1/persona"
    assert by_first_path["/persona/status"].tags == ("persona",)
    assert by_first_path["/persona/status"].route_key == ""
    assert by_first_path["/archetypes"].prefix == "/api/v1/persona/archetypes"
    assert by_first_path["/archetypes"].tags == ("persona-archetypes",)
    assert by_first_path["/archetypes"].route_key == ""
    assert by_first_path["/notes"].prefix == "/api/v1/notes"
    assert by_first_path["/notes"].tags == ("notes",)
    assert by_first_path["/notes"].route_key == ""
    assert by_first_path["/web-clipper"].prefix == "/api/v1/web-clipper"
    assert by_first_path["/web-clipper"].tags == ("web-clipper",)
    assert by_first_path["/web-clipper"].route_key == ""
    assert by_first_path["/skills"].prefix == "/api/v1/skills"
    assert by_first_path["/skills"].tags == ("skills",)
    assert by_first_path["/skills"].route_key == ""
    assert by_first_path["/translate"].prefix == "/api/v1"
    assert by_first_path["/translate"].tags == ("translation",)
    assert by_first_path["/translate"].route_key == ""
    assert by_first_path["/slides"].prefix == "/api/v1"
    assert by_first_path["/slides"].tags == ("slides",)
    assert by_first_path["/slides"].route_key == ""
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
    assert by_first_path["/flashcards"].prefix == "/api/v1"
    assert by_first_path["/flashcards"].tags == ("flashcards",)
    assert by_first_path["/flashcards"].route_key == ""
    assert by_first_path["/quizzes"].prefix == "/api/v1"
    assert by_first_path["/quizzes"].tags == ("quizzes",)
    assert by_first_path["/quizzes"].route_key == ""
    assert by_first_path["/study-suggestions"].prefix == "/api/v1"
    assert by_first_path["/study-suggestions"].tags == ("study-suggestions",)
    assert by_first_path["/study-suggestions"].route_key == ""
    assert by_first_path["/writing"].prefix == "/api/v1/writing"
    assert by_first_path["/writing"].tags == ("writing",)
    assert by_first_path["/writing"].route_key == ""
    assert by_first_path["/writing/manuscripts"].prefix == "/api/v1/writing/manuscripts"
    assert by_first_path["/writing/manuscripts"].tags == ("manuscripts",)
    assert by_first_path["/writing/manuscripts"].route_key == ""
    assert by_first_path["/email/search"].prefix == "/api/v1/email"
    assert by_first_path["/email/search"].tags == ("email",)
    assert by_first_path["/email/search"].route_key == ""
    assert by_first_path["/jobs/health"].prefix == "/api/v1"
    assert by_first_path["/jobs/health"].tags == ("jobs",)
    assert by_first_path["/jobs/health"].route_key == ""
    assert by_first_path["/audit/events"].prefix == "/api/v1"
    assert by_first_path["/audit/events"].tags == ("audit",)
    assert by_first_path["/audit/events"].route_key == ""
    assert by_first_path["/documentation/info"].prefix == "/api/v1"
    assert by_first_path["/documentation/info"].tags == ("config",)
    assert by_first_path["/documentation/info"].route_key == ""
    assert by_first_path["/config/effective"].prefix == "/api/v1"
    assert by_first_path["/config/effective"].tags == ("config", "admin")
    assert by_first_path["/config/effective"].route_key == ""
    assert by_first_path["/admin/status"].prefix == "/api/v1"
    assert by_first_path["/admin/status"].tags == ("admin",)
    assert by_first_path["/admin/status"].route_key == ""
    assert by_first_path["/orgs"].prefix == "/api/v1"
    assert by_first_path["/orgs"].tags == ("organizations",)
    assert by_first_path["/orgs"].route_key == ""
    assert by_first_path["/orgs/invites"].prefix == "/api/v1"
    assert by_first_path["/orgs/invites"].tags == ("invites",)
    assert by_first_path["/orgs/invites"].route_key == ""
    assert by_first_path["/resource-governor/status"].prefix == "/api/v1"
    assert by_first_path["/resource-governor/status"].tags == ("resource-governor",)
    assert by_first_path["/resource-governor/status"].route_key == ""
    assert by_first_path["/users/me"].prefix == "/api/v1"
    assert by_first_path["/users/me"].tags == ("users",)
    assert by_first_path["/users/me"].route_key == ""
    assert by_first_path["/users/keys"].prefix == "/api/v1"
    assert by_first_path["/users/keys"].tags == ("users",)
    assert by_first_path["/users/keys"].route_key == ""
    assert by_first_path["/organizations/shared-keys"].prefix == "/api/v1"
    assert by_first_path["/organizations/shared-keys"].tags == ("organizations",)
    assert by_first_path["/organizations/shared-keys"].route_key == ""
    assert by_first_path["/mcp/status"].prefix == "/api/v1"
    assert by_first_path["/mcp/status"].tags == ("mcp-unified",)
    assert by_first_path["/mcp/status"].route_key == ""
    assert by_first_path["/mcp/catalogs"].prefix == "/api/v1"
    assert by_first_path["/mcp/catalogs"].tags == ("mcp-catalogs",)
    assert by_first_path["/mcp/catalogs"].route_key == ""
    assert by_first_path["/mcp/hub"].prefix == "/api/v1"
    assert by_first_path["/mcp/hub"].tags == ("mcp-hub",)
    assert by_first_path["/mcp/hub"].route_key == ""
    assert by_first_path["/privileges"].prefix == "/api/v1"
    assert by_first_path["/privileges"].tags == ("privileges",)
    assert by_first_path["/privileges"].route_key == ""
    assert by_first_path["/tools"].prefix == "/api/v1"
    assert by_first_path["/tools"].tags == ("tools",)
    assert by_first_path["/tools"].route_key == ""
    assert by_first_path["/acp/run"].prefix == "/api/v1"
    assert by_first_path["/acp/run"].tags == ("acp",)
    assert by_first_path["/acp/run"].route_key == ""
    assert by_first_path["/acp/schedules"].prefix == "/api/v1"
    assert by_first_path["/acp/schedules"].tags == ("acp-schedules",)
    assert by_first_path["/acp/schedules"].route_key == ""
    assert by_first_path["/acp/triggers"].prefix == "/api/v1"
    assert by_first_path["/acp/triggers"].tags == ("acp-triggers",)
    assert by_first_path["/acp/triggers"].route_key == ""
    assert by_first_path["/acp/permissions"].prefix == "/api/v1"
    assert by_first_path["/acp/permissions"].tags == ("acp-permissions",)
    assert by_first_path["/acp/permissions"].route_key == ""
    assert by_first_path["/acp/multiplex"].prefix == "/api/v1"
    assert by_first_path["/acp/multiplex"].tags == ("acp-multiplex",)
    assert by_first_path["/acp/multiplex"].route_key == ""
    assert by_first_path["/agent-orchestration"].prefix == "/api/v1"
    assert by_first_path["/agent-orchestration"].tags == ("agent-orchestration",)
    assert by_first_path["/agent-orchestration"].route_key == ""
    assert by_first_path["/setup/status"].prefix == "/api/v1"
    assert by_first_path["/setup/status"].tags == ("setup",)
    assert by_first_path["/setup/status"].route_key == ""
    assert by_first_path["/metrics/text"].prefix == "/api/v1"
    assert by_first_path["/metrics/text"].tags == ("metrics",)
    assert by_first_path["/metrics/text"].route_key == ""
    assert by_first_path["/authnz-debug/status"].prefix == "/api/v1"
    assert by_first_path["/authnz-debug/status"].tags == ("authnz-debug",)
    assert by_first_path["/authnz-debug/status"].route_key == ""
    assert by_first_path["/sandbox/status"].prefix == "/api/v1"
    assert by_first_path["/sandbox/status"].tags == ("sandbox",)
    assert by_first_path["/sandbox/status"].route_key == ""


def test_iter_minimal_optional_router_specs_falls_back_to_admin_byok_when_admin_router_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    admin_package = ModuleType("tldw_Server_API.app.api.v1.endpoints.admin")
    admin_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.api.v1.endpoints.admin", admin_package)
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.admin.admin_byok",
        path="/keys/shared",
    )

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert "/admin/status" not in by_first_path
    assert by_first_path["/keys/shared"].prefix == "/api/v1/admin"
    assert by_first_path["/keys/shared"].tags == ("admin",)
    assert by_first_path["/keys/shared"].route_key == ""


def test_iter_minimal_optional_router_specs_includes_audio_jobs_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_audio_jobs")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO_JOBS", "1")
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs",
        path="/jobs",
    )

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/jobs"].prefix == "/api/v1/audio"
    assert by_first_path["/jobs"].tags == ("audio-jobs",)
    assert by_first_path["/jobs"].route_key == "audio-jobs"


def test_iter_minimal_optional_router_specs_includes_media_audio_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_media_audio")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.media",
        path="/media/list",
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

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/media/list"].prefix == "/api/v1/media"
    assert by_first_path["/media/list"].tags == ("media",)
    assert by_first_path["/media/list"].route_key == "media"
    assert by_first_path["/transcriptions"].prefix == "/api/v1/audio"
    assert by_first_path["/transcriptions"].tags == ("audio",)
    assert by_first_path["/transcriptions"].route_key == "audio"
    assert by_first_path["/stream/transcribe"].prefix == "/api/v1/audio"
    assert by_first_path["/stream/transcribe"].tags == ("audio-ws",)
    assert by_first_path["/stream/transcribe"].route_key == "audio-websocket"


def test_iter_minimal_optional_router_specs_skips_audio_during_pytest_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_media_audio")
    monkeypatch.delenv("MINIMAL_TEST_INCLUDE_AUDIO", raising=False)
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.media",
        path="/media/list",
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

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert by_first_path["/media/list"].prefix == "/api/v1/media"
    assert "/transcriptions" not in by_first_path
    assert "/stream/transcribe" not in by_first_path


def test_iter_minimal_optional_router_specs_skips_audio_jobs_during_pytest_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_audio_jobs")
    monkeypatch.delenv("MINIMAL_TEST_INCLUDE_AUDIO_JOBS", raising=False)
    _install_fake_router_module(
        monkeypatch,
        "tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs",
        path="/jobs",
    )

    specs = list(iter_minimal_optional_router_specs())
    by_first_path = {_first_router_path(spec.router): spec for spec in specs}

    assert "/jobs" not in by_first_path


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

    assert by_key["rag-unified"].prefix == ""
    assert by_key["rag-unified"].tags == ("rag-unified",)
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
    assert len(web_scraping_specs) == 1
    assert {(spec.prefix, spec.tags) for spec in web_scraping_specs} == {
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
    assert by_first_path["/sandbox/status"].route_key == "sandbox"
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
    assert "from tldw_Server_API.app.api.v1.endpoints.sync import router as sync_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.web_scraping import router as web_scraping_router" not in source
    assert "Tools endpoints unavailable at import time; deferring" not in source
    assert "ACP endpoints unavailable at import time; deferring" not in source
    assert "Users endpoints unavailable at import time; deferring" not in source


def test_main_source_no_longer_keeps_minimal_users_import_stub() -> None:
    source = _main_source_text()

    assert "from tldw_Server_API.app.api.v1.endpoints.users import router as users_router" not in source
    assert "_ = users_router" not in source
    assert "Skipping users router in minimal test app" not in source


def test_main_source_no_longer_keeps_minimal_tools_acp_import_stubs() -> None:
    source = _main_source_text()

    assert "from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import router as acp_router" not in source
    assert "tools_router = None" not in source
    assert "acp_router = None" not in source


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


def test_main_source_delegates_minimal_health_auth_routers_to_group() -> None:
    source = _main_source_text()

    assert "iter_minimal_test_router_specs()" in source
    assert "from tldw_Server_API.app.api.v1.endpoints.auth import router as auth_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.health import router as health_router" not in source
    assert 'app.include_router(auth_router, prefix=f"{API_V1_PREFIX}", tags=["authentication"])' not in source
    assert 'health_router, prefix=f"{API_V1_PREFIX}", tags=["health"]' not in source
    assert "Auth router consolidated: endpoints/auth.py (minimal test app)" not in source
    assert "Skipping health router in minimal test app" not in source
    assert "Skipping auth router in minimal test app" not in source


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
    assert "from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.sharing import router as sharing_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.personalization import router as personalization_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.companion import router as companion_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.guardian_controls import router as guardian_controls_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.family_wizard import router as family_wizard_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.self_monitoring import router as self_monitoring_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.persona import router as persona_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import router as archetype_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.web_clipper import router as web_clipper_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.skills import router as skills_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.translate import router as translate_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_boards import router as kanban_boards_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.kanban.kanban_workflow import router as kanban_workflow_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.quizzes import router as quizzes_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.study_suggestions import (" not in source
    assert "router as study_suggestions_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.writing import router as writing_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.writing_manuscripts import router as manuscripts_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.email import router as email_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.audit import router as audit_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.config_admin import router as config_admin_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.user_keys import router as user_keys_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.shared_keys_scoped import router as shared_keys_scoped_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.mcp_catalogs_manage import router as mcp_catalogs_manage_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.mcp_hub_management import router as mcp_hub_management_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.privileges import router as privileges_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.acp_schedules import router as acp_schedules_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.acp_triggers import router as acp_triggers_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.acp_permissions import router as acp_permissions_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.acp_multiplex import router as acp_multiplex_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.agent_orchestration import router as orch_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.authnz_debug import router as authnz_debug_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router" not in source
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
    assert "app.include_router(chatbooks_router" not in source
    assert "app.include_router(sharing_router" not in source
    assert "app.include_router(personalization_router" not in source
    assert "app.include_router(companion_router" not in source
    assert "app.include_router(guardian_controls_router" not in source
    assert "app.include_router(family_wizard_router" not in source
    assert "app.include_router(self_monitoring_router" not in source
    assert "app.include_router(persona_router" not in source
    assert "app.include_router(archetype_router" not in source
    assert "app.include_router(notes_router" not in source
    assert "app.include_router(web_clipper_router" not in source
    assert "app.include_router(skills_router" not in source
    assert "app.include_router(translate_router" not in source
    assert "app.include_router(slides_router" not in source
    assert "app.include_router(kanban_boards_router" not in source
    assert "app.include_router(kanban_workflow_router" not in source
    assert "app.include_router(flashcards_router" not in source
    assert "app.include_router(quizzes_router" not in source
    assert "app.include_router(study_suggestions_router" not in source
    assert "app.include_router(writing_router" not in source
    assert "app.include_router(manuscripts_router" not in source
    assert "app.include_router(email_router" not in source
    assert "app.include_router(jobs_admin_router" not in source
    assert "app.include_router(audit_router" not in source
    assert "app.include_router(config_info_router" not in source
    assert "app.include_router(config_admin_router" not in source
    assert "app.include_router(users_router" not in source
    assert "app.include_router(user_keys_router" not in source
    assert "app.include_router(shared_keys_scoped_router" not in source
    assert "app.include_router(mcp_unified_router" not in source
    assert "app.include_router(mcp_catalogs_manage_router" not in source
    assert "app.include_router(mcp_hub_management_router" not in source
    assert "app.include_router(privileges_router" not in source
    assert "app.include_router(tools_router" not in source
    assert "app.include_router(acp_router" not in source
    assert "app.include_router(acp_schedules_router" not in source
    assert "app.include_router(acp_triggers_router" not in source
    assert "app.include_router(acp_permissions_router" not in source
    assert "app.include_router(acp_multiplex_router" not in source
    assert "app.include_router(orch_router" not in source
    assert "app.include_router(setup_router" not in source
    assert "app.include_router(metrics_router" not in source
    assert "app.include_router(authnz_debug_router" not in source
    assert "app.include_router(sandbox_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.workflows import router as _wf_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.chat_workflows import (" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.scheduler_workflows import router as _sch_wf_router" not in source
    assert 'app.include_router(_wf_router, prefix="", tags=["workflows"])' not in source
    assert 'app.include_router(_chat_wf_router, prefix="", tags=["chat-workflows"])' not in source
    assert 'app.include_router(_sch_wf_router, prefix="", tags=["scheduler"])' not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import (" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.monitoring import router as _monitoring_router" not in source
    assert 'app.include_router(_evaluations_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])' not in source
    assert 'app.include_router(_monitoring_router, prefix=f"{API_V1_PREFIX}", tags=["monitoring"])' not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.llamacpp import (" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.messages import (" not in source
    assert "app.include_router(llamacpp_router" not in source
    assert "app.include_router(llamacpp_public_router" not in source
    assert "app.include_router(messages_router" not in source
    assert "app.include_router(messages_public_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.admin.admin_byok import (" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.orgs import router as orgs_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.org_invites import router as org_invites_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.resource_governor import router as resource_governor_router" not in source
    assert "app.include_router(admin_router" not in source
    assert "app.include_router(admin_byok_router" not in source
    assert "app.include_router(orgs_router" not in source
    assert "app.include_router(org_invites_router" not in source
    assert "app.include_router(resource_governor_router" not in source
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
    assert "Skipping chatbooks router in minimal test app" not in source
    assert "Skipping sharing router in minimal test app" not in source
    assert "Skipping personalization router in minimal test app" not in source
    assert "Skipping companion router in minimal test app" not in source
    assert "Skipping guardian controls router in minimal test app" not in source
    assert "Skipping self-monitoring router in minimal test app" not in source
    assert "Skipping persona router in minimal test app" not in source
    assert "Skipping archetype router in minimal test app" not in source
    assert "Skipping notes router in minimal test app" not in source
    assert "Skipping web clipper router in minimal test app" not in source
    assert "Skipping skills router in minimal test app" not in source
    assert "Skipping translate router in minimal test app" not in source
    assert "Skipping slides router in minimal test app" not in source
    assert "Skipping kanban router in minimal test app" not in source
    assert "Skipping flashcards router in minimal test app" not in source
    assert "Skipping quizzes router in minimal test app" not in source
    assert "Skipping study_suggestions router in minimal test app" not in source
    assert "Skipping writing router in minimal test app" not in source
    assert "Skipping manuscripts router in minimal test app" not in source
    assert "Skipping email router in minimal test app" not in source
    assert "Skipping jobs_admin router in minimal test app" not in source
    assert "Skipping audit router in minimal test app" not in source
    assert "Skipping config_info router in minimal test app" not in source
    assert "Skipping config_admin router in minimal test app" not in source
    assert "Skipping BYOK/shared keys routers in minimal test app" not in source
    assert "Skipping MCP unified router in minimal test app" not in source
    assert "Skipping MCP catalogs router in minimal test app" not in source
    assert "Skipping MCP hub router in minimal test app" not in source
    assert "Skipping privileges router in minimal test app" not in source
    assert "Skipping tools router in minimal test app" not in source
    assert "Skipping ACP router in minimal test app" not in source
    assert "Skipping ACP schedules router in minimal test app" not in source
    assert "Skipping ACP triggers router in minimal test app" not in source
    assert "Skipping ACP permissions router in minimal test app" not in source
    assert "Skipping ACP multiplex router in minimal test app" not in source
    assert "Skipping orchestration router in minimal test app" not in source
    assert "Skipping setup router in minimal test app" not in source
    assert "Skipping metrics router in minimal test app" not in source
    assert "Skipping authnz_debug router in tests" not in source
    assert "Skipping sandbox router in minimal test app" not in source
    assert "Skipping workflows router in minimal test app" not in source
    assert "Skipping chat workflows router in minimal test app" not in source
    assert "Skipping scheduler workflows router in minimal test app" not in source
    assert "Skipping evaluations routers in minimal test app" not in source
    assert "Skipping monitoring router in minimal test app" not in source
    assert "Skipping llamacpp router in minimal test app" not in source
    assert "Skipping admin router include in minimal test app" not in source
    assert "Skipping admin BYOK router in minimal test app" not in source
    assert "Skipping orgs router in minimal test app" not in source
    assert "Skipping org_invites router in minimal test app" not in source
    assert "Skipping resource_governor router in minimal test app" not in source


def test_main_source_delegates_minimal_audio_jobs_router_to_group() -> None:
    source = _main_source_text()

    assert "iter_minimal_optional_router_specs()" in source
    assert "from tldw_Server_API.app.api.v1.endpoints.audio.audio_jobs import router as audio_jobs_router" not in source
    assert 'app.include_router(audio_jobs_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio-jobs"])' not in source
    assert 'route_enabled("audio-jobs")' not in source
    assert "MINIMAL_TEST_INCLUDE_AUDIO_JOBS" not in source
    assert "Skipping audio-jobs router in minimal test app" not in source
    assert "Route disabled by policy: audio-jobs (minimal test app)" not in source


def test_main_source_delegates_minimal_media_audio_routers_to_group() -> None:
    source = _main_source_text()

    assert "iter_minimal_optional_router_specs()" in source
    assert "from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.audio.audio import ws_router as audio_ws_router" not in source
    assert "from tldw_Server_API.app.api.v1.endpoints.media import router as media_router" not in source
    assert 'app.include_router(audio_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio"])' not in source
    assert 'app.include_router(audio_ws_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio-ws"])' not in source
    assert 'app.include_router(media_router, prefix=f"{API_V1_PREFIX}/media", tags=["media"])' not in source
    assert 'route_enabled("audio")' not in source
    assert 'route_enabled("audio-websocket")' not in source
    assert 'route_enabled("media")' not in source
    assert "MINIMAL_TEST_INCLUDE_AUDIO" not in source
    assert "Skipping audio routers in minimal test app" not in source
    assert "Route disabled by policy: audio/audio-websocket (minimal test app)" not in source
    assert "Skipping media router in minimal test app" not in source
    assert "Route disabled by policy: media (minimal test app)" not in source


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
    assert "from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import" not in source
    assert 'app.include_router(_evaluations_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])' not in source
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
