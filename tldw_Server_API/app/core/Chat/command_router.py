"""
command_router.py

Lightweight slash-command router for chat pre-processing.

Scope:
- Registry with built-in slash commands (/time, /weather, /skills, /skill)
- Per-user + global per-command rate limits using TokenBucket
- Optional RBAC enforcement hooks
- Command output truncation for safe injection paths
- Metrics logging

Env flags:
- CHAT_COMMANDS_ENABLED: '1' to enable (default: off)
- CHAT_COMMANDS_RATE_LIMIT_USER: per-user RPM per command (default: 10)
- CHAT_COMMANDS_RATE_LIMIT_GLOBAL: global RPM per command (default: 100)
- CHAT_COMMANDS_RATE_LIMIT: backward-compatible alias for USER limit
- CHAT_COMMANDS_MAX_CHARS: max chars in command output (default: 300)
- CHAT_COMMAND_INJECTION_MODE: 'system', 'preface', or 'replace' (default: 'system')
- DEFAULT_LOCATION: fallback location for /weather (default: '')
"""

from __future__ import annotations

import contextlib
import inspect
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.Chat.command_authorization import (
    authorize_command,
    build_command_authorization_context,
)
from tldw_Server_API.app.core.Chat.rate_limiter import TokenBucket
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.Context_Integrity.resolver import ContextIntegrityBlocked
from tldw_Server_API.app.core.Integrations import weather_providers
from tldw_Server_API.app.core.Metrics import increment_counter
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter
from tldw_Server_API.app.core.testing import is_truthy

_COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

try:
    from tldw_Server_API.app.core.AuthNZ.rbac import user_has_permission as _user_has_permission
except ImportError:  # pragma: no cover - fallback if AuthNZ is trimmed in tests

    def _user_has_permission(user_id: int, permission: str) -> bool:  # type: ignore
        return False


SLASH_RE = re.compile(r"^/(\w+)(?:\s+(.*))?$")
RPM_VALUE_RE = re.compile(
    r"^\s*(?P<value>\d+)\s*(?:$|/(?P<unit>m|min|minute|minutes)|\s+per\s+(?P<unit_per>m|min|minute|minutes))\s*$",
    re.IGNORECASE,
)


def _cfg() -> Any | None:
    try:
        return load_comprehensive_config()
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
        return None


def _cfg_bool(env_name: str, cfg_key: str, fallback: bool) -> bool:
    v = os.getenv(env_name)
    if isinstance(v, str) and v.strip():
        return is_truthy(v)
    cp = _cfg()
    if cp and cp.has_section("Chat-Commands"):
        try:
            raw = cp.get("Chat-Commands", cfg_key, fallback=str(fallback))
            return is_truthy(raw)
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            return fallback
    return fallback


def _cfg_int(env_name: str, cfg_key: str, fallback: int) -> int:
    v = os.getenv(env_name)
    if isinstance(v, str) and v.strip():
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            return fallback
    cp = _cfg()
    if cp and cp.has_section("Chat-Commands"):
        try:
            raw = cp.get("Chat-Commands", cfg_key, fallback=str(fallback))
            return max(1, int(str(raw)))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _cfg_str(env_name: str, cfg_key: str, fallback: str) -> str:
    v = os.getenv(env_name)
    if isinstance(v, str) and v.strip():
        return v.strip()
    cp = _cfg()
    if cp and cp.has_section("Chat-Commands"):
        try:
            raw = cp.get("Chat-Commands", cfg_key, fallback=fallback)
            return str(raw).strip()
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            return fallback
    return fallback


def _parse_positive_int(raw: object) -> int | None:
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def _parse_rpm(raw: object) -> int | None:
    if isinstance(raw, int):
        return max(1, raw)
    text = str(raw).strip().lower()
    if not text:
        return None
    m = RPM_VALUE_RE.match(text)
    if not m:
        return None
    try:
        return max(1, int(m.group("value")))
    except (TypeError, ValueError):
        return None


def _cfg_int_alias(
    env_names: tuple[str, ...],
    cfg_keys: tuple[str, ...],
    fallback: int,
    *,
    parse_rpm: bool = False,
) -> int:
    parser = _parse_rpm if parse_rpm else _parse_positive_int
    for env_name in env_names:
        raw = os.getenv(env_name)
        if isinstance(raw, str) and raw.strip():
            parsed = parser(raw)
            if parsed is not None:
                return parsed
    cp = _cfg()
    if cp and cp.has_section("Chat-Commands"):
        for cfg_key in cfg_keys:
            with contextlib.suppress(_COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS):
                raw = cp.get("Chat-Commands", cfg_key, fallback=None)
                if raw is None:
                    continue
                parsed = parser(raw)
                if parsed is not None:
                    return parsed
    return fallback


def commands_enabled() -> bool:
    return _cfg_bool("CHAT_COMMANDS_ENABLED", "commands_enabled", False)


def is_single_user_mode() -> bool:
    """Return True when AuthNZ is configured in single-user mode.

    This helper exists primarily as a test seam for RBAC behavior; production
    code should prefer env/config-driven enforcement.
    """
    try:
        return str(os.getenv("AUTH_MODE", "")).strip().lower() == "single_user"
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
        return False


def get_injection_mode() -> str:
    mode = _cfg_str("CHAT_COMMAND_INJECTION_MODE", "injection_mode", "system").lower().strip()
    # Supported modes:
    # - system: inject result as a separate system message
    # - preface: preface the user's message with the command result
    # - replace: replace the user's message content with the command result
    return mode if mode in {"system", "preface", "replace"} else "system"


def _per_command_user_rpm() -> int:
    return _cfg_int_alias(
        ("CHAT_COMMANDS_RATE_LIMIT_USER", "CHAT_COMMANDS_RATE_LIMIT"),
        ("commands_rate_limit_user", "commands_rate_limit"),
        10,
        parse_rpm=True,
    )


def _per_command_global_rpm() -> int:
    return _cfg_int_alias(
        ("CHAT_COMMANDS_RATE_LIMIT_GLOBAL",),
        ("commands_rate_limit_global",),
        100,
        parse_rpm=True,
    )


def _per_command_rpm() -> int:
    # Backward-compatible seam retained for older tests.
    return _per_command_user_rpm()


def _command_max_chars() -> int:
    return _cfg_int_alias(
        ("CHAT_COMMANDS_MAX_CHARS",),
        ("commands_max_chars",),
        300,
        parse_rpm=False,
    )


def default_rate_limit_display() -> str:
    return f"per-user {_per_command_user_rpm()}/min, global {_per_command_global_rpm()}/min"


def build_injection_text(command: str, content: str) -> str:
    """Build bounded slash-command injection text with a standard prefix."""
    max_chars = _command_max_chars()
    if max_chars <= 0:
        return ""
    prefix = f"[/{str(command).strip().lower()}] "
    if len(prefix) >= max_chars:
        return prefix[:max_chars]

    source = content if isinstance(content, str) else str(content)
    remaining = max_chars - len(prefix)
    if len(source) <= remaining:
        return f"{prefix}{source}"
    if remaining <= 3:
        return f"{prefix}{source[:remaining]}"
    return f"{prefix}{source[:remaining - 3].rstrip()}..."


@dataclass
class CommandContext:
    user_id: str = "anonymous"
    conversation_id: str | None = None
    request_meta: dict | None = None
    auth_user_id: int | None = None  # numeric user id when available for RBAC


@dataclass
class CommandResult:
    ok: bool
    command: str
    content: str
    metadata: dict[str, Any]


Handler = Callable[[CommandContext, Optional[str]], CommandResult]


@dataclass
class CommandSpec:
    name: str
    description: str
    handler: Handler
    allowed_roles: list[str] | None = None  # reserved for future RBAC
    required_permission: str | None = None  # permission string when enforcement is enabled
    usage: str | None = None
    args: list[str] | None = None
    requires_api_key: bool = True
    rate_limit: str | None = None
    rbac_required: bool | None = None


_registry: dict[str, CommandSpec] = {}
_buckets: dict[tuple[str, str], TokenBucket] = {}
_global_buckets: dict[str, TokenBucket] = {}
_bucket_registry_lock = threading.RLock()


def register_command(
    name: str,
    description: str,
    handler: Handler,
    allowed_roles: list[str] | None = None,
    required_permission: str | None = None,
    usage: str | None = None,
    args: list[str] | None = None,
    requires_api_key: bool = True,
    rate_limit: str | None = None,
    rbac_required: bool | None = None,
) -> None:
    computed_rbac_required = bool(required_permission) if rbac_required is None else bool(rbac_required)
    _registry[name.lower()] = CommandSpec(
        name=name.lower(),
        description=description,
        handler=handler,
        allowed_roles=allowed_roles,
        required_permission=required_permission,
        usage=usage,
        args=list(args or []),
        requires_api_key=bool(requires_api_key),
        rate_limit=rate_limit,
        rbac_required=computed_rbac_required,
    )


def list_commands() -> list[dict[str, Any]]:
    default_rate_limit = default_rate_limit_display()
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "required_permission": spec.required_permission,
            "usage": spec.usage,
            "args": list(spec.args or []),
            "requires_api_key": bool(spec.requires_api_key),
            "rate_limit": spec.rate_limit or default_rate_limit,
            "rbac_required": (
                bool(spec.required_permission) if spec.rbac_required is None else bool(spec.rbac_required)
            ),
        }
        for spec in _registry.values()
    ]


def reserved_core_command_names() -> set[str]:
    """Return names reserved by the built-in slash command registry."""
    return set(_registry)


def extract_slash_candidate(message: str) -> tuple[str, str | None] | None:
    """Parse any slash-shaped message without requiring a registered command."""
    if not isinstance(message, str):
        return None
    m = SLASH_RE.match(message.strip())
    if not m:
        return None
    cmd = (m.group(1) or "").lower()
    args = (m.group(2) or "").strip() or None
    return cmd, args


def parse_slash_command(message: str) -> tuple[str, str | None] | None:
    if not isinstance(message, str):
        return None
    m = SLASH_RE.match(message.strip())
    if not m:
        return None
    cmd = (m.group(1) or "").lower()
    args = (m.group(2) or "").strip() or None
    if cmd in _registry:
        return cmd, args
    return None


def _acquire_bucket(user_id: str, command: str) -> TokenBucket:
    key = (user_id or "anonymous", command)
    with _bucket_registry_lock:
        bucket = _buckets.get(key)
        if bucket is not None:
            return bucket
        rpm = _per_command_user_rpm()
        bucket = TokenBucket(capacity=rpm, refill_rate=rpm / 60.0)
        _buckets[key] = bucket
        return bucket


def _acquire_global_bucket(command: str) -> TokenBucket:
    key = command
    with _bucket_registry_lock:
        bucket = _global_buckets.get(key)
        if bucket is not None:
            return bucket
        rpm = _per_command_global_rpm()
        bucket = TokenBucket(capacity=rpm, refill_rate=rpm / 60.0)
        _global_buckets[key] = bucket
        return bucket


def _finalize_result(result: CommandResult) -> CommandResult:
    source = result.content if isinstance(result.content, str) else str(result.content)
    max_chars = _command_max_chars()
    truncated = False
    content = source
    if max_chars > 0 and len(source) > max_chars:
        truncated = True
        if max_chars <= 3:
            content = source[:max_chars]
        else:
            content = f"{source[:max_chars - 3].rstrip()}..."

    metadata = dict(result.metadata or {})
    metadata.setdefault("max_chars", max_chars)
    if truncated:
        metadata["truncated"] = True
        metadata["original_chars"] = len(source)
        with contextlib.suppress(_COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS):
            increment_counter("chat_command_output_truncated_total", labels={"command": result.command})
    return CommandResult(
        ok=bool(result.ok),
        command=result.command,
        content=content,
        metadata=metadata,
    )


def dispatch_command(ctx: CommandContext, command: str, args: str | None) -> CommandResult:
    raise RuntimeError(
        "command_router.dispatch_command has been removed. "
        "Use async_dispatch_command(...) or the chat orchestrator "
        "(achat or its sync wrapper) instead."
    )


async def async_dispatch_command(ctx: CommandContext, command: str, args: str | None) -> CommandResult:
    """Async variant of dispatch_command that uses TokenBucket.consume() safely.

    Mirrors dispatch_command behavior but awaits the bucket's consume method to
    respect its asyncio.Lock and prevent race conditions under concurrency.
    """
    cmd = command.lower()
    spec = _registry.get(cmd)
    if not spec:
        return _finalize_result(
            CommandResult(
                ok=False,
                command=cmd,
                content=f"Unknown command: /{cmd}",
                metadata={"error": "unknown_command"},
            )
        )

    decision = authorize_command(
        spec=spec,
        context=build_command_authorization_context(ctx),
        permission_checker=_user_has_permission,
    )
    if not decision.allowed:
        log_counter("chat_command_error", labels={"command": cmd, "reason": "permission_denied"})
        try:
            increment_counter("chat_command_errors_total", labels={"command": cmd, "reason": "permission_denied"})
            increment_counter("chat_command_invoked_total", labels={"command": cmd, "status": "denied"})
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            pass
        return _finalize_result(
            CommandResult(
                ok=False,
                command=cmd,
                content=f"Permission denied for /{cmd}",
                metadata={"error": "permission_denied", **decision.metadata},
            )
        )

    # Global per-command and per-user per-command rate limiting (safe, lock-respecting)
    global_bucket = _acquire_global_bucket(cmd)
    global_allowed = await global_bucket.consume(1)
    if not global_allowed:
        log_counter("chat_command_error", labels={"command": cmd, "reason": "rate_limited"})
        try:
            increment_counter("chat_command_errors_total", labels={"command": cmd, "reason": "rate_limited"})
            increment_counter("chat_command_invoked_total", labels={"command": cmd, "status": "rate_limited"})
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            pass
        return _finalize_result(
            CommandResult(
                ok=False,
                command=cmd,
                content=f"Command /{cmd} is globally rate limited. Please try again shortly.",
                metadata={"error": "rate_limited", "scope": "global"},
            )
        )

    bucket = _acquire_bucket(ctx.user_id, cmd)
    allowed = await bucket.consume(1)
    if not allowed:
        with contextlib.suppress(_COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS):
            await global_bucket.refund(1)
        log_counter("chat_command_error", labels={"command": cmd, "reason": "rate_limited"})
        try:
            increment_counter("chat_command_errors_total", labels={"command": cmd, "reason": "rate_limited"})
            increment_counter("chat_command_invoked_total", labels={"command": cmd, "status": "rate_limited"})
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            pass
        return _finalize_result(
            CommandResult(
                ok=False,
                command=cmd,
                content=f"Command /{cmd} is rate limited. Please try again shortly.",
                metadata={"error": "rate_limited", "scope": "user"},
            )
        )

    try:
        res = spec.handler(ctx, args)
        if inspect.isawaitable(res):  # future-proof if handlers become async
            res = await res  # type: ignore[assignment]
        if not isinstance(res, CommandResult):
            raise TypeError(f"Command handler for /{cmd} returned {type(res)}")
        # annotate result metadata with RBAC info when applicable
        if decision.metadata.get("checked"):
            with contextlib.suppress(_COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS):
                res.metadata = {
                    **(res.metadata or {}),
                    "rbac": {
                        **decision.metadata,
                        "permitted": True,
                    },
                }
        log_counter("chat_command_invoked", labels={"command": cmd})
        with contextlib.suppress(_COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS):
            increment_counter("chat_command_invoked_total", labels={"command": cmd, "status": "success"})
        return _finalize_result(res)
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error executing /{cmd}: {e}", exc_info=True)
        log_counter("chat_command_error", labels={"command": cmd, "reason": "exception"})
        try:
            increment_counter("chat_command_errors_total", labels={"command": cmd, "reason": "exception"})
            increment_counter("chat_command_invoked_total", labels={"command": cmd, "status": "error"})
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            pass
        return _finalize_result(
            CommandResult(
                ok=False,
                command=cmd,
                content=f"Command /{cmd} failed: {e}",
                metadata={"error": "exception"},
            )
        )


# -----------------------------
# Built-in command handlers
# -----------------------------


def _time_handler(ctx: CommandContext, args: str | None) -> CommandResult:
    from datetime import datetime

    try:
        # Optional timezone support via zoneinfo
        tzlabel = (args or "").strip() if args else None
        dt = None
        tzused = "local"
        if tzlabel:
            try:
                from zoneinfo import ZoneInfo  # Python 3.9+

                dt = datetime.now(ZoneInfo(tzlabel))
                tzused = tzlabel
            except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
                dt = datetime.now()
                tzused = "local"
        else:
            dt = datetime.now()
        text = dt.strftime("%Y-%m-%d %H:%M:%S")
        return CommandResult(
            ok=True,
            command="time",
            content=f"Current time ({tzused}): {text}",
            metadata={"tz": tzused},
        )
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS as e:
        return CommandResult(
            ok=False, command="time", content=f"Time lookup failed: {e}", metadata={"error": "time_error"}
        )


def _weather_handler(ctx: CommandContext, args: str | None) -> CommandResult:
    location = (args or "").strip()
    if not location:
        location = _cfg_str("DEFAULT_LOCATION", "default_location", "").strip()
    # Obtain client via test seam so monkeypatches take effect
    try:
        client = get_weather_client(ctx)
    except TypeError:
        # Backward-compatible: some tests patch a zero-arg seam
        client = get_weather_client()
    try:
        result = client.get_current(location=location or None)
        metadata = dict(getattr(result, "metadata", {}) or {})
        if result.ok:
            return CommandResult(ok=True, command="weather", content=result.summary, metadata=metadata)
        metadata.setdefault("error", "unavailable")
        return CommandResult(ok=False, command="weather", content=result.summary, metadata=metadata)
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Weather provider error: {e}", exc_info=True)
        return CommandResult(
            ok=False, command="weather", content=f"Weather unavailable: {e}", metadata={"error": "exception"}
        )


def _request_meta(ctx: CommandContext) -> dict[str, Any]:
    meta = getattr(ctx, "request_meta", None)
    return meta if isinstance(meta, dict) else {}


def _extract_user_id(ctx: CommandContext, meta: dict[str, Any]) -> int | None:
    candidate = meta.get("auth_user_id")
    if isinstance(candidate, int) and candidate > 0:
        return candidate

    if isinstance(ctx.auth_user_id, int) and ctx.auth_user_id > 0:
        return int(ctx.auth_user_id)

    raw_user_id = str(ctx.user_id or "").strip()
    if raw_user_id.isdigit():
        return int(raw_user_id)

    if is_single_user_mode():
        try:
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

            return int(DatabasePaths.get_single_user_id())
        except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
            return None

    return None


async def _resolve_skills_runtime(ctx: CommandContext) -> tuple[int, Path, Any]:
    meta = _request_meta(ctx)
    user_id = _extract_user_id(ctx, meta)
    if user_id is None:
        raise ValueError("Unable to resolve user identity for skills command execution")

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    raw_base_path = meta.get("user_base_dir")
    if isinstance(raw_base_path, Path):
        base_path = raw_base_path
    elif isinstance(raw_base_path, str) and raw_base_path.strip():
        base_path = Path(raw_base_path)
    else:
        base_path = DatabasePaths.get_user_base_directory(user_id)

    db = meta.get("chat_db")
    if db is None:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id

        db = await get_chacha_db_for_user_id(user_id, client_id=str(ctx.user_id or user_id))

    return user_id, base_path, db


def _filter_skills_for_query(skills: list[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
    if not query:
        return list(skills)
    q = query.strip().lower()
    if not q:
        return list(skills)
    return [
        skill
        for skill in skills
        if (
            q in str(skill.get("name") or "").lower()
            or q in str(skill.get("description") or "").lower()
            or q in str(skill.get("argument_hint") or "").lower()
        )
    ]


async def _list_invocable_skills(ctx: CommandContext, filter_text: str | None = None) -> list[dict[str, Any]]:
    from tldw_Server_API.app.core.Skills.skills_service import SkillsService

    user_id, base_path, db = await _resolve_skills_runtime(ctx)
    service = SkillsService(user_id=user_id, base_path=base_path, db=db)
    payload = await service.get_context_payload_async()
    skills = [
        skill
        for skill in list(payload.get("available_skills") or [])
        if isinstance(skill, dict) and str(skill.get("name") or "").strip()
    ]
    filtered = _filter_skills_for_query(skills, filter_text)
    return sorted(filtered, key=lambda skill: str(skill.get("name") or ""))


def _split_skill_invocation(raw_args: str) -> tuple[str, str]:
    payload = str(raw_args or "").strip()
    if not payload:
        return "", ""
    parts = payload.split(maxsplit=1)
    skill_name = str(parts[0] or "").strip().lower()
    skill_args = str(parts[1] or "").strip() if len(parts) > 1 else ""
    return skill_name, skill_args


def _tool_names_from_definitions(tool_defs: list[Any]) -> list[str]:
    names: list[str] = []
    for tool in tool_defs:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            fn_name = function.get("name")
            if isinstance(fn_name, str) and fn_name.strip():
                names.append(fn_name.strip())
    return names


def _build_skill_request_context(ctx: CommandContext, user_id: int):
    from tldw_Server_API.app.core.Skills.skill_executor import RequestContext

    meta = _request_meta(ctx)
    raw_tool_defs = meta.get("tools")
    tool_defs = raw_tool_defs if isinstance(raw_tool_defs, list) else None
    available_tools = _tool_names_from_definitions(tool_defs or [])
    selected_provider = meta.get("selected_provider")
    selected_model = meta.get("selected_model")
    conversation_id = meta.get("conversation_id")

    return RequestContext(
        user_id=user_id,
        default_provider=str(selected_provider) if isinstance(selected_provider, str) and selected_provider else None,
        default_model=str(selected_model) if isinstance(selected_model, str) and selected_model else None,
        conversation_id=str(conversation_id) if isinstance(conversation_id, str) and conversation_id else None,
        client_id=str(ctx.user_id or user_id),
        available_tools=available_tools,
        tool_definitions=tool_defs,
    )


async def _execute_skill(ctx: CommandContext, skill_name: str, skill_args: str) -> dict[str, Any]:
    from tldw_Server_API.app.core.Skills.exceptions import SkillNotFoundError, SkillsError
    from tldw_Server_API.app.core.Skills.skill_executor import SkillExecutor
    from tldw_Server_API.app.core.Skills.skills_service import SkillsService

    normalized_name = str(skill_name or "").strip().lower()
    if not normalized_name:
        return {"success": False, "error": "missing_name"}

    try:
        user_id, base_path, db = await _resolve_skills_runtime(ctx)
        service = SkillsService(user_id=user_id, base_path=base_path, db=db)
        skill_data = await service.get_skill(normalized_name)
    except ContextIntegrityBlocked:
        return {"success": False, "error": "context_integrity_blocked"}
    except SkillNotFoundError:
        return {"success": False, "error": "skill_not_found"}
    except SkillsError as e:
        return {"success": False, "error": "skills_error", "detail": str(e)}
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS as e:
        return {"success": False, "error": "runtime_error", "detail": str(e)}
    except Exception as e:  # pragma: no cover - defensive guard for uncommon dependency errors
        return {"success": False, "error": "runtime_error", "detail": str(e)}

    user_invocable = bool(skill_data.get("user_invocable", True))
    disable_model_invocation = bool(skill_data.get("disable_model_invocation", False))
    if not user_invocable or disable_model_invocation:
        return {"success": False, "error": "skill_not_invocable"}

    try:
        executor = SkillExecutor()
        execution_ctx = _build_skill_request_context(ctx, user_id)
        result = await executor.execute(
            skill_data=skill_data,
            arguments=skill_args or "",
            context=execution_ctx,
        )
        return {
            "success": True,
            "execution_mode": result.execution_mode,
            "rendered_prompt": result.rendered_prompt,
            "fork_output": result.fork_output,
        }
    except SkillsError as e:
        return {"success": False, "error": "execution_failed", "detail": str(e)}
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS as e:
        return {"success": False, "error": "execution_failed", "detail": str(e)}
    except Exception as e:  # pragma: no cover - defensive guard for uncommon dependency errors
        return {"success": False, "error": "execution_failed", "detail": str(e)}


async def _skills_handler(ctx: CommandContext, args: str | None) -> CommandResult:
    filter_text = str(args or "").strip() or None
    try:
        skills = await _list_invocable_skills(ctx, filter_text=filter_text)
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS as e:
        return CommandResult(
            ok=False,
            command="skills",
            content=f"Unable to list skills: {e}",
            metadata={"error": "skills_unavailable"},
        )
    except Exception as e:  # pragma: no cover - defensive guard for uncommon dependency errors
        return CommandResult(
            ok=False,
            command="skills",
            content=f"Unable to list skills: {e}",
            metadata={"error": "skills_unavailable"},
        )

    if not skills:
        if filter_text:
            message = f"No invocable skills matched '{filter_text}'."
        else:
            message = "No invocable skills are available."
        return CommandResult(
            ok=True,
            command="skills",
            content=message,
            metadata={"count": 0, "filter": filter_text},
        )

    lines = [f"Available skills ({len(skills)}):"]
    for skill in skills:
        name = str(skill.get("name") or "").strip()
        if not name:
            continue
        hint = str(skill.get("argument_hint") or "").strip()
        description = str(skill.get("description") or "No description").strip()
        suffix = f" {hint}" if hint else ""
        lines.append(f"- {name}{suffix}: {description}")

    return CommandResult(
        ok=True,
        command="skills",
        content="\n".join(lines),
        metadata={"count": len(skills), "filter": filter_text},
    )


async def _skill_handler(ctx: CommandContext, args: str | None) -> CommandResult:
    payload = str(args or "").strip()
    if not payload:
        return CommandResult(
            ok=False,
            command="skill",
            content="Usage: /skill <name> [args]",
            metadata={"error": "missing_name"},
        )

    skill_name, skill_args = _split_skill_invocation(payload)
    if not skill_name:
        return CommandResult(
            ok=False,
            command="skill",
            content="Usage: /skill <name> [args]",
            metadata={"error": "missing_name"},
        )

    result = await _execute_skill(ctx, skill_name, skill_args)
    if not result.get("success"):
        error = str(result.get("error") or "execution_failed")
        if error == "missing_name":
            message = "Usage: /skill <name> [args]"
        elif error == "skill_not_found":
            message = f"Skill '{skill_name}' not found."
        elif error == "skill_not_invocable":
            message = f"Skill '{skill_name}' is not invocable."
        elif error == "context_integrity_blocked":
            message = "Skill is quarantined pending admin review."
        else:
            detail = str(result.get("detail") or "Skill execution failed.")
            message = f"Skill execution failed: {detail}"
        return CommandResult(
            ok=False,
            command="skill",
            content=message,
            metadata={"error": error, "skill_name": skill_name},
        )

    execution_mode = str(result.get("execution_mode") or "inline").strip().lower() or "inline"
    if execution_mode == "fork":
        output = str(result.get("fork_output") or "")
    else:
        output = str(result.get("rendered_prompt") or "")
    if not output:
        output = f"Skill '{skill_name}' executed."

    return CommandResult(
        ok=True,
        command="skill",
        content=output,
        metadata={"skill_name": skill_name, "execution_mode": execution_mode},
    )


# Register built-in commands on import
register_command(
    "time",
    "Show the current time (optional TZ).",
    _time_handler,
    required_permission="chat.commands.time",
    usage="/time [timezone]",
    args=["timezone"],
    requires_api_key=True,
    rbac_required=True,
)
register_command(
    "weather",
    "Show current weather for a location.",
    _weather_handler,
    required_permission="chat.commands.weather",
    usage="/weather [location]",
    args=["location"],
    requires_api_key=True,
    rbac_required=True,
)
register_command(
    "skills",
    "List invocable skills for this user.",
    _skills_handler,
    required_permission="chat.commands.skills",
    usage="/skills [filter]",
    args=["filter"],
    requires_api_key=True,
    rbac_required=True,
)
register_command(
    "skill",
    "Execute an invocable skill by name.",
    _skill_handler,
    required_permission="chat.commands.skill",
    usage="/skill <name> [args]",
    args=["name", "args"],
    requires_api_key=True,
    rbac_required=True,
)


# --- Test seam helpers ---
def get_weather_client(ctx: CommandContext | None = None):
    """Thin wrapper to allow tests to monkeypatch the weather client at the router level.

    Tests expect to patch command_router.get_weather_client; delegate to the
    actual provider factory so production code continues to use the unified
    weather providers module.
    """
    return weather_providers.get_weather_client()
