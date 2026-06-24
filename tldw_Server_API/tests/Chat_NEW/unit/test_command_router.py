from typing import Optional

import asyncio
import pytest

from tldw_Server_API.app.core.Chat import command_router


@pytest.fixture(autouse=True)
def _reset_command_router_buckets():
    command_router._buckets.clear()
    command_router._global_buckets.clear()
    yield
    command_router._buckets.clear()
    command_router._global_buckets.clear()


def _authorized_ctx(user_id: str = "u1", *permissions: str) -> command_router.CommandContext:
    return command_router.CommandContext(
        user_id=user_id,
        auth_user_id=1,
        request_meta={
            "permissions": list(permissions or ("chat.commands.*",)),
            "roles": [],
            "is_admin": False,
        },
    )


@pytest.mark.unit
def test_commands_enabled_accepts_single_letter_y(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "y")
    assert command_router.commands_enabled() is True


@pytest.mark.asyncio
async def test_parse_and_dispatch_time(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    parsed = command_router.parse_slash_command("/time")
    assert parsed is not None
    name, args = parsed
    assert name == "time"
    ctx = _authorized_ctx("u1", "chat.commands.time")
    res = await command_router.async_dispatch_command(ctx, name, args)
    assert res.ok
    assert "Current time" in res.content


@pytest.mark.asyncio
async def test_rate_limit_per_user_per_command(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "100")
    # Ensure a fresh process bucket by using a unique user id
    ctx = _authorized_ctx("rl_user", "chat.commands.time")
    # First call allowed
    res1 = await command_router.async_dispatch_command(ctx, "time", None)
    assert res1.ok
    # Second call immediately after should be rate-limited
    res2 = await command_router.async_dispatch_command(ctx, "time", None)
    assert not res2.ok
    assert "rate limited" in res2.content.lower()


@pytest.mark.asyncio
async def test_weather_stub(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    class OkClient:
        def get_current(self, location: Optional[str] = None, lat=None, lon=None):
            class Result:
                ok = True
                summary = f"Sunny at {location or 'somewhere'}"
                metadata = {"provider": "test"}

            return Result()

    # Patch provider factory
    from tldw_Server_API.app.core.Integrations import weather_providers

    monkeypatch.setattr(weather_providers, "get_weather_client", lambda: OkClient())

    ctx = _authorized_ctx("u2", "chat.commands.weather")
    res = await command_router.async_dispatch_command(ctx, "weather", "Boston")
    assert res.ok
    assert "Sunny" in res.content


@pytest.mark.unit
def test_list_commands_includes_rich_metadata(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_USER", "3/min")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "30/min")

    by_name = {entry["name"]: entry for entry in command_router.list_commands()}
    time_cmd = by_name["time"]
    weather_cmd = by_name["weather"]

    assert time_cmd["usage"] == "/time [timezone]"
    assert time_cmd["args"] == ["timezone"]
    assert time_cmd["requires_api_key"] is True
    assert time_cmd["rbac_required"] is True
    assert "per-user 3/min" in time_cmd["rate_limit"]
    assert "global 30/min" in time_cmd["rate_limit"]
    assert weather_cmd["usage"] == "/weather [location]"
    assert weather_cmd["args"] == ["location"]


@pytest.mark.unit
def test_list_commands_includes_skill_commands_metadata():
    by_name = {entry["name"]: entry for entry in command_router.list_commands()}

    assert "skills" in by_name
    assert by_name["skills"]["required_permission"] == "chat.commands.skills"
    assert by_name["skills"]["usage"] == "/skills [filter]"
    assert by_name["skills"]["args"] == ["filter"]
    assert by_name["skills"]["requires_api_key"] is True
    assert by_name["skills"]["rbac_required"] is True

    assert "skill" in by_name
    assert by_name["skill"]["required_permission"] == "chat.commands.skill"
    assert by_name["skill"]["usage"] == "/skill <name> [args]"
    assert by_name["skill"]["args"] == ["name", "args"]
    assert by_name["skill"]["requires_api_key"] is True
    assert by_name["skill"]["rbac_required"] is True


@pytest.mark.unit
def test_filter_skills_for_query_matches_name_description_and_hint():
    skills = [
        {"name": "summarize", "description": "Summarize docs", "argument_hint": "<topic>"},
        {"name": "code-review", "description": "Review code", "argument_hint": None},
        {"name": "research", "description": "Deep analysis", "argument_hint": "<question>"},
    ]

    by_name = command_router._filter_skills_for_query(skills, "sum")
    assert [s["name"] for s in by_name] == ["summarize"]

    by_desc = command_router._filter_skills_for_query(skills, "analysis")
    assert [s["name"] for s in by_desc] == ["research"]

    by_hint = command_router._filter_skills_for_query(skills, "topic")
    assert [s["name"] for s in by_hint] == ["summarize"]


@pytest.mark.asyncio
async def test_skills_command_lists_only_invocable_skills(monkeypatch):
    async def fake_list(ctx, filter_text=None):
        assert filter_text is None
        return [
            {"name": "summarize", "description": "Summarize docs", "argument_hint": "<topic>"},
            {"name": "code-review", "description": "Review code", "argument_hint": None},
        ]

    monkeypatch.setattr(command_router, "_list_invocable_skills", fake_list)
    ctx = _authorized_ctx("u1", "chat.commands.skills")

    res = await command_router.async_dispatch_command(ctx, "skills", None)
    assert res.ok
    assert "summarize" in res.content
    assert "code-review" in res.content
    assert res.metadata.get("count") == 2


@pytest.mark.asyncio
async def test_skills_command_applies_filter(monkeypatch):
    async def fake_list(ctx, filter_text=None):
        assert filter_text == "sum"
        return [{"name": "summarize", "description": "Summarize docs", "argument_hint": None}]

    monkeypatch.setattr(command_router, "_list_invocable_skills", fake_list)
    ctx = _authorized_ctx("u1", "chat.commands.skills")

    res = await command_router.async_dispatch_command(ctx, "skills", "sum")
    assert res.ok
    assert "summarize" in res.content
    assert res.metadata.get("count") == 1


@pytest.mark.asyncio
async def test_skill_command_requires_name():
    ctx = _authorized_ctx("u1", "chat.commands.skill")
    res = await command_router.async_dispatch_command(ctx, "skill", None)
    assert not res.ok
    assert "Usage" in res.content


@pytest.mark.asyncio
async def test_skill_command_executes_inline(monkeypatch):
    async def fake_exec(ctx, skill_name, skill_args):
        assert skill_name == "summarize"
        assert skill_args == "release notes"
        return {
            "success": True,
            "execution_mode": "inline",
            "rendered_prompt": "Summarized",
            "fork_output": None,
        }

    monkeypatch.setattr(command_router, "_execute_skill", fake_exec)
    ctx = _authorized_ctx("u1", "chat.commands.skill")
    res = await command_router.async_dispatch_command(ctx, "skill", "summarize release notes")
    assert res.ok
    assert "Summarized" in res.content
    assert res.metadata.get("execution_mode") == "inline"


@pytest.mark.asyncio
async def test_skill_command_executes_fork(monkeypatch):
    async def fake_exec(ctx, skill_name, skill_args):
        return {
            "success": True,
            "execution_mode": "fork",
            "rendered_prompt": "ignored",
            "fork_output": "Fork result",
        }

    monkeypatch.setattr(command_router, "_execute_skill", fake_exec)
    ctx = _authorized_ctx("u1", "chat.commands.skill")
    res = await command_router.async_dispatch_command(ctx, "skill", "research-plan q1")
    assert res.ok
    assert "Fork result" in res.content
    assert res.metadata.get("execution_mode") == "fork"


@pytest.mark.asyncio
async def test_skill_command_rejects_non_invocable_skill(monkeypatch):
    async def fake_exec(ctx, skill_name, skill_args):
        return {"success": False, "error": "skill_not_invocable"}

    monkeypatch.setattr(command_router, "_execute_skill", fake_exec)
    ctx = _authorized_ctx("u1", "chat.commands.skill")
    res = await command_router.async_dispatch_command(ctx, "skill", "hidden-skill x")
    assert not res.ok
    assert "not invocable" in res.content.lower()


@pytest.mark.asyncio
async def test_skill_command_reports_not_found(monkeypatch):
    async def fake_exec(ctx, skill_name, skill_args):
        return {"success": False, "error": "skill_not_found"}

    monkeypatch.setattr(command_router, "_execute_skill", fake_exec)
    ctx = _authorized_ctx("u1", "chat.commands.skill")
    res = await command_router.async_dispatch_command(ctx, "skill", "missing-skill x")
    assert not res.ok
    assert "not found" in res.content.lower()


@pytest.mark.asyncio
async def test_skill_command_handles_runtime_resolution_exception(monkeypatch):
    async def fake_resolve(ctx):
        raise Exception("resolver exploded")

    monkeypatch.setattr(command_router, "_resolve_skills_runtime", fake_resolve)
    ctx = _authorized_ctx("u1", "chat.commands.skill")

    res = await command_router.async_dispatch_command(ctx, "skill", "summarize release notes")
    assert not res.ok
    assert res.metadata.get("error") == "runtime_error"
    assert "execution failed" in res.content.lower()


@pytest.mark.asyncio
async def test_required_command_permission_is_enforced_without_legacy_flag(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.delenv("CHAT_COMMANDS_REQUIRE_PERMISSIONS", raising=False)

    denied = await command_router.async_dispatch_command(
        command_router.CommandContext(user_id="anon", auth_user_id=None),
        "time",
        None,
    )

    assert not denied.ok
    assert denied.metadata["error"] == "permission_denied"
    assert denied.metadata["required_permission"] == "chat.commands.time"


@pytest.mark.asyncio
async def test_command_permission_allows_claim_without_db_lookup(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    def fail_if_called(_user_id, _permission):
        raise AssertionError("permission DB should not be called for claim hit")

    monkeypatch.setattr(command_router, "_user_has_permission", fail_if_called)

    allowed = await command_router.async_dispatch_command(
        _authorized_ctx("claim-user", "chat.commands.time"),
        "time",
        None,
    )

    assert allowed.ok
    assert allowed.metadata["rbac"]["source"] == "claims"


@pytest.mark.asyncio
async def test_command_permission_does_not_allow_child_wildcard_claim(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    denied = await command_router.async_dispatch_command(
        _authorized_ctx("claim-user", "chat.commands.time.*"),
        "time",
        None,
    )

    assert not denied.ok
    assert denied.metadata["error"] == "permission_denied"
    assert denied.metadata["required_permission"] == "chat.commands.time"


@pytest.mark.asyncio
async def test_command_permission_ignores_mapping_claim_keys(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    denied = await command_router.async_dispatch_command(
        command_router.CommandContext(
            user_id="mapping-user",
            auth_user_id=None,
            request_meta={"permissions": {"chat.commands.time": True}},
        ),
        "time",
        None,
    )

    assert not denied.ok
    assert denied.metadata["error"] == "permission_denied"


@pytest.mark.asyncio
async def test_command_permission_allows_single_user_owner(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    allowed = await command_router.async_dispatch_command(
        command_router.CommandContext(
            user_id="owner",
            auth_user_id=None,
            request_meta={"auth_mode": "single_user", "is_single_user_owner": True},
        ),
        "time",
        None,
    )

    assert allowed.ok
    assert allowed.metadata["rbac"]["source"] == "single_user_owner"


@pytest.mark.asyncio
async def test_rbac_enforcement(monkeypatch):
    # Enable commands and RBAC enforcement
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_REQUIRE_PERMISSIONS", "1")

    # Without auth_user_id, permission should be denied for a command requiring permission
    ctx = command_router.CommandContext(user_id="anon", auth_user_id=None)
    denied = await command_router.async_dispatch_command(ctx, "time", None)
    assert not denied.ok
    assert denied.metadata.get("error") == "permission_denied"

    # With auth_user_id and granted permission, should pass
    monkeypatch.setattr(command_router, "_user_has_permission", lambda uid, perm: True)
    ctx2 = command_router.CommandContext(user_id="u", auth_user_id=42)
    allowed = await command_router.async_dispatch_command(ctx2, "time", None)
    assert allowed.ok


def test_dispatch_command_removed_raises():

    ctx = command_router.CommandContext(user_id="legacy")
    with pytest.raises(RuntimeError):
        command_router.dispatch_command(ctx, "time", None)


@pytest.mark.asyncio
async def test_rate_limit_parses_user_rpm_suffix(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_USER", "2/min")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "100/min")

    ctx = _authorized_ctx("rpm_suffix_user", "chat.commands.time")
    assert (await command_router.async_dispatch_command(ctx, "time", None)).ok
    assert (await command_router.async_dispatch_command(ctx, "time", None)).ok

    denied = await command_router.async_dispatch_command(ctx, "time", None)
    assert not denied.ok
    assert denied.metadata.get("scope") == "user"


@pytest.mark.asyncio
async def test_rate_limit_global_per_command(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_USER", "100/min")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "1/min")

    first = await command_router.async_dispatch_command(_authorized_ctx("u1", "chat.commands.time"), "time", None)
    second = await command_router.async_dispatch_command(_authorized_ctx("u2", "chat.commands.time"), "time", None)

    assert first.ok
    assert not second.ok
    assert second.metadata.get("error") == "rate_limited"
    assert second.metadata.get("scope") == "global"


@pytest.mark.asyncio
async def test_command_output_truncation(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "100")
    monkeypatch.setenv("CHAT_COMMANDS_MAX_CHARS", "20")

    class LongClient:
        def get_current(self, location: Optional[str] = None, lat=None, lon=None):
            class Result:
                ok = True
                summary = "very long weather summary " * 8
                metadata = {"provider": "test"}

            return Result()

    from tldw_Server_API.app.core.Integrations import weather_providers

    monkeypatch.setattr(weather_providers, "get_weather_client", lambda: LongClient())

    res = await command_router.async_dispatch_command(
        _authorized_ctx("truncate-user", "chat.commands.weather"),
        "weather",
        "Boston",
    )
    assert res.ok
    assert len(res.content) <= 20
    assert res.metadata.get("truncated") is True
    assert res.metadata.get("max_chars") == 20
    assert int(res.metadata.get("original_chars", 0)) > 20


@pytest.mark.asyncio
async def test_async_dispatch_command_concurrent_respects_rate_limit(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT", "5")
    monkeypatch.setenv("CHAT_COMMANDS_RATE_LIMIT_GLOBAL", "100")

    ctx = _authorized_ctx("async_rl_user", "chat.commands.time")

    async def call():
        return await command_router.async_dispatch_command(ctx, "time", None)

    results = await asyncio.gather(*[call() for _ in range(10)])
    ok = [r for r in results if r.ok]
    limited = [r for r in results if not r.ok and "rate limited" in r.content.lower()]

    # At most the configured limit should be allowed.
    assert len(ok) <= 5
    # Under concurrent load, we should see some rate-limited responses.
    assert len(limited) >= 1
