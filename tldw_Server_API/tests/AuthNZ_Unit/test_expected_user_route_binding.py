from fastapi.routing import APIRoute

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import (
    character_chat_sessions,
    character_messages,
    chat,
    rag_unified,
    research,
    service_prompts,
)
from tldw_Server_API.app.api.v1.endpoints.media import add as media_add


def test_expected_user_guard_is_bound_only_to_scope_bound_routes() -> None:
    guarded_routes: set[tuple[str, str, str]] = set()

    for router_name, router in (
        ("character-chat-sessions", character_chat_sessions.router),
        ("character-messages", character_messages.router),
        ("chat", chat.router),
        ("rag", rag_unified.router),
        ("research", research.router),
        ("service-prompts", service_prompts.router),
        ("media-add", media_add.router),
    ):
        for route in router.routes:
            if not isinstance(route, APIRoute):
                continue
            dependencies = {dependency.call for dependency in route.dependant.dependencies}
            if auth_deps.require_expected_user not in dependencies:
                continue
            assert route.dependant.dependencies[0].call is auth_deps.require_expected_user
            for method in route.methods:
                guarded_routes.add((router_name, method, route.path))

    assert guarded_routes == {
        ("character-chat-sessions", "POST", "/"),
        ("character-messages", "POST", "/chats/{chat_id}/messages"),
        ("chat", "POST", "/completions"),
        ("rag", "POST", "/api/v1/rag/search"),
        ("research", "POST", "/websearch"),
        ("service-prompts", "DELETE", "/service-prompts/{definition_id}"),
        ("service-prompts", "GET", "/service-prompts"),
        ("service-prompts", "GET", "/service-prompts/{definition_id}"),
        ("service-prompts", "PUT", "/service-prompts/{definition_id}"),
        ("media-add", "POST", "/add"),
    }
