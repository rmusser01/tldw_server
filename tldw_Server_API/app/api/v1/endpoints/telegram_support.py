from __future__ import annotations

import asyncio
import inspect
import secrets
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.telegram_schemas import (
    TELEGRAM_WEBHOOK_SECRET_MIN_LENGTH,
    TelegramBotConfigResponse,
    TelegramBotConfigUpdate,
    TelegramLinkedActorListResponse,
    TelegramLinkedActorRevokeResponse,
    TelegramWebhookUpdate,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.telegram_approvals_repo import (
    get_telegram_approvals_repo,
)
from tldw_Server_API.app.core.AuthNZ.repos.telegram_runtime_repo import (
    get_telegram_runtime_repo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Telegram.session_mapper import (
    build_telegram_session_key,
    derive_telegram_assistant_conversation_id,
)
from tldw_Server_API.app.services.mcp_hub_approval_service import (
    _scope_key_for_tool_call,
    get_mcp_hub_approval_service,
)
from tldw_Server_API.app.services.telegram_delivery_service import TelegramDeliveryService
from tldw_Server_API.app.services.telegram_execution_identity_service import (
    get_telegram_execution_identity_service,
)

_PROVIDER = "telegram"
_DEFAULT_BOT_USERNAME = "example_bot"
_CREDENTIAL_VERSION = 1
_TELEGRAM_APPROVAL_CALLBACK_PREFIX = "tgapp1."
_TELEGRAM_APPROVAL_CALLBACK_MAX_LENGTH = 64
_TELEGRAM_PENDING_APPROVAL_TTL_SECONDS = 300
_WEBHOOK_REPLAY_WINDOW_SECONDS = 3600
_TELEGRAM_PAIRING_CODE_TTL_SECONDS = 900


@dataclass(frozen=True)
class TelegramScope:
    scope_type: str
    scope_id: int


@dataclass(frozen=True)
class TelegramWebhookContext:
    scope: TelegramScope
    bot_username: str


@dataclass(frozen=True)
class TelegramCommand:
    action: str
    input: str
    target_bot_username: str | None = None


@dataclass(frozen=True)
class TelegramMessagePolicy:
    should_process: bool
    reason: str | None = None
    command: TelegramCommand | None = None


_TELEGRAM_REQUEST_NAMESPACE = uuid.UUID("f3b8b11f-6a65-4a2a-a7fd-8f9d9cf3f3d4")


def _coerce_nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_scope_ids(values: list[int] | None) -> list[int]:
    out: set[int] = set()
    for raw in values or []:
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


async def build_telegram_approval_callback_data(
    *,
    approval_policy_id: int | None,
    context_key: str,
    conversation_id: str | None,
    tool_name: str,
    tool_args: Any,
    scope: TelegramScope,
    initiating_auth_user_id: int,
    scope_payload: dict[str, Any] | None = None,
    scope_key: str | None = None,
    expires_at: datetime | None = None,
    ttl_seconds: int = _TELEGRAM_PENDING_APPROVAL_TTL_SECONDS,
    repo: Any | None = None,
) -> str:
    """Build short callback data and persist the pending approval server-side."""
    approval_token = secrets.token_urlsafe(12)
    repo = repo if repo is not None else await get_telegram_approvals_repo()
    expiry = _coerce_datetime(expires_at)
    now = datetime.now(timezone.utc)
    if expiry is None:
        expiry = now + timedelta(seconds=max(1, int(ttl_seconds)))
    scope_key_value = _coerce_nonempty_string(scope_key) or _scope_key_for_tool_call(
        tool_name,
        tool_args,
        scope_payload=scope_payload,
    )
    record = await repo.create_pending_approval(
        approval_token=approval_token,
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        approval_policy_id=approval_policy_id,
        context_key=context_key,
        conversation_id=conversation_id,
        tool_name=tool_name,
        scope_key=scope_key_value,
        initiating_auth_user_id=initiating_auth_user_id,
        expires_at=expiry,
        now=now,
    )
    callback_data = f"{_TELEGRAM_APPROVAL_CALLBACK_PREFIX}{record['approval_token']}"
    if len(callback_data) > _TELEGRAM_APPROVAL_CALLBACK_MAX_LENGTH:
        raise ValueError("Telegram approval callback_data exceeds Telegram limits")
    return callback_data


def _parse_telegram_approval_callback_data(callback_data: Any) -> dict[str, Any] | None:
    text = _coerce_nonempty_string(callback_data)
    if not text or not text.startswith(_TELEGRAM_APPROVAL_CALLBACK_PREFIX):
        return None
    approval_token = text[len(_TELEGRAM_APPROVAL_CALLBACK_PREFIX) :]
    if not approval_token:
        return None
    return {"approval_token": approval_token}


async def _peek_pending_telegram_approval_for_tests(callback_data: Any, *, repo: Any) -> dict[str, Any] | None:
    parsed = _parse_telegram_approval_callback_data(callback_data)
    if parsed is None:
        return None
    return await repo.get_pending_approval_by_token(parsed["approval_token"])


def _generate_telegram_pairing_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _normalize_pairing_code(value: Any) -> str | None:
    text = _coerce_nonempty_string(value)
    if not text:
        return None
    return text.replace(" ", "").upper() or None


async def _store_telegram_pairing_code(
    *,
    auth_user_id: int,
    scope: TelegramScope,
    runtime_repo: Any,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return await runtime_repo.create_pairing_code(
        pairing_code=_generate_telegram_pairing_code(),
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        auth_user_id=auth_user_id,
        expires_at=now + timedelta(seconds=_TELEGRAM_PAIRING_CODE_TTL_SECONDS),
        now=now,
    )


async def _register_telegram_actor_link_for_tests_async(
    *,
    scope_type: str,
    scope_id: int,
    telegram_user_id: int,
    auth_user_id: int,
    telegram_username: str | None = None,
) -> None:
    """Install a deterministic Telegram actor link for tests."""
    runtime_repo = await get_telegram_runtime_repo()
    await runtime_repo.upsert_actor_link(
        scope_type=scope_type,
        scope_id=scope_id,
        telegram_user_id=telegram_user_id,
        auth_user_id=auth_user_id,
        telegram_username=telegram_username,
    )


def _register_telegram_actor_link_for_tests(
    *,
    scope_type: str,
    scope_id: int,
    telegram_user_id: int,
    auth_user_id: int,
    telegram_username: str | None = None,
) -> None:
    asyncio.run(
        _register_telegram_actor_link_for_tests_async(
            scope_type=scope_type,
            scope_id=scope_id,
            telegram_user_id=telegram_user_id,
            auth_user_id=auth_user_id,
            telegram_username=telegram_username,
        )
    )


def _normalize_telegram_bot_username(value: Any) -> str | None:
    username = _coerce_nonempty_string(value)
    if not username:
        return None
    if username.startswith("@"):
        username = username[1:]
    username = username.strip().lower()
    return username or None


def _command_targets_configured_bot(
    command: TelegramCommand | None,
    configured_bot_username: Any,
) -> bool:
    if command is None:
        return False
    if command.target_bot_username is None:
        return True
    normalized_bot_username = _normalize_telegram_bot_username(configured_bot_username)
    if normalized_bot_username is None:
        return False
    return command.target_bot_username == normalized_bot_username


def parse_telegram_command(text: Any) -> TelegramCommand | None:
    normalized = _coerce_nonempty_string(text)
    if not normalized:
        return None
    if not normalized.startswith("/"):
        return None

    parts = normalized.split(maxsplit=1)
    command_token = parts[0][1:]
    if not command_token:
        return None
    action_token, target_username = (command_token.split("@", 1) + [None])[:2]
    action = action_token.strip().lower()
    if not action:
        return None
    argument = _coerce_nonempty_string(parts[1]) if len(parts) > 1 else ""
    return TelegramCommand(
        action=action,
        input=argument or "",
        target_bot_username=_normalize_telegram_bot_username(target_username),
    )


def evaluate_telegram_message_policy(
    *,
    chat_type: Any,
    text: Any,
    bot_username: Any = None,
    reply_to_bot: bool = False,
) -> TelegramMessagePolicy:
    command = parse_telegram_command(text)
    if command is not None and not _command_targets_configured_bot(command, bot_username):
        return TelegramMessagePolicy(
            should_process=False,
            reason="command_not_for_configured_bot",
            command=command,
        )
    if command is not None:
        return TelegramMessagePolicy(should_process=True, command=command)

    normalized_chat_type = _coerce_nonempty_string(chat_type)
    if normalized_chat_type in {"group", "supergroup"} and not reply_to_bot:
        return TelegramMessagePolicy(
            should_process=False,
            reason="group_freeform_requires_command_or_reply",
        )

    return TelegramMessagePolicy(should_process=True)


def _is_privileged_telegram_command(command: TelegramCommand | None) -> bool:
    if command is None:
        return False
    if command.action not in {"persona", "character"}:
        return False
    tokens = command.input.lower().split()
    if not tokens:
        return False
    return tokens[0] == "set"


def _extract_message_actor_and_text(payload: dict[str, Any]) -> tuple[int | None, str | None]:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None, None
    from_block = message.get("from")
    telegram_user_id = _coerce_int(from_block.get("id")) if isinstance(from_block, dict) else None
    text = _coerce_nonempty_string(message.get("text"))
    return telegram_user_id, text


def _extract_message_actor_username(payload: dict[str, Any]) -> str | None:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    from_block = message.get("from")
    if not isinstance(from_block, dict):
        return None
    return _coerce_nonempty_string(from_block.get("username"))


def _extract_message_chat_type(payload: dict[str, Any]) -> str | None:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    return _coerce_nonempty_string(chat.get("type"))


def _extract_message_chat_id(payload: dict[str, Any]) -> int | None:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    return _coerce_int(chat.get("id"))


def _extract_message_thread_id(payload: dict[str, Any]) -> int | None:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    return _coerce_int(message.get("message_thread_id"))


def _extract_message_id(payload: dict[str, Any]) -> int | None:
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    return _coerce_int(message.get("message_id"))


def _message_reply_to_bot(
    payload: dict[str, Any],
    *,
    configured_bot_username: Any = None,
) -> bool:
    message = payload.get("message")
    if not isinstance(message, dict):
        return False
    reply_to_message = message.get("reply_to_message")
    if not isinstance(reply_to_message, dict):
        return False
    reply_from = reply_to_message.get("from")
    if not isinstance(reply_from, dict):
        return False
    if not bool(reply_from.get("is_bot")):
        return False
    reply_username = _normalize_telegram_bot_username(reply_from.get("username"))
    configured_username = _normalize_telegram_bot_username(configured_bot_username)
    if reply_username is None or configured_username is None:
        return False
    return reply_username == configured_username


def _build_telegram_tenant_id(scope: TelegramScope) -> str:
    return f"{scope.scope_type}:{scope.scope_id}"


def _build_telegram_request_id(scope: TelegramScope, update_id: int) -> str:
    return str(uuid.uuid5(_TELEGRAM_REQUEST_NAMESPACE, f"{scope.scope_type}:{scope.scope_id}:{update_id}"))


def _build_telegram_ask_job_payload(
    *,
    scope: TelegramScope,
    update_id: int,
    message_id: int | None,
    telegram_user_id: int,
    text: str,
    chat_type: str | None,
    reply_to_bot: bool,
    configured_bot_username: str,
    command: TelegramCommand,
    linked_actor: dict[str, Any],
    telegram_chat_id: int | None,
    telegram_thread_id: int | None,
    request_id: str,
) -> dict[str, Any]:
    tenant_id = _build_telegram_tenant_id(scope)
    session_key = build_telegram_session_key(
        tenant_id=tenant_id,
        chat_type=chat_type or "private",
        telegram_user_id=telegram_user_id,
        telegram_chat_id=telegram_chat_id,
        topic_or_thread_id=telegram_thread_id,
    )
    return {
        "telegram": {
            "scope_type": scope.scope_type,
            "scope_id": scope.scope_id,
            "update_id": update_id,
            "message_id": message_id,
            "chat_type": chat_type,
            "telegram_user_id": telegram_user_id,
            "telegram_chat_id": telegram_chat_id,
            "topic_or_thread_id": telegram_thread_id,
            "text": text,
            "reply_to_bot": reply_to_bot,
            "bot_username": configured_bot_username,
            "command": {
                "action": command.action,
                "input": command.input,
                "target_bot_username": command.target_bot_username,
            },
            "linked_actor": {
                "auth_user_id": str(linked_actor["auth_user_id"]),
                "telegram_user_id": telegram_user_id,
            },
        },
        "session": {
            "tenant_id": tenant_id,
            "session_key": session_key,
            "assistant_conversation_id": derive_telegram_assistant_conversation_id(session_key),
        },
        "request_id": request_id,
    }


async def _resolve_job_manager_for_request(request: Request) -> JobManager:
    from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager

    resolver = get_job_manager
    app = getattr(request, "app", None)
    dependency_overrides = getattr(app, "dependency_overrides", None)
    if isinstance(dependency_overrides, dict):
        override = dependency_overrides.get(get_job_manager)
        if override is not None:
            resolver = override

    resolved = resolver()
    if inspect.isawaitable(resolved):
        resolved = await resolved
    return resolved


def _normalize_bot_config_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "provider": _PROVIDER,
        "credential_version": _CREDENTIAL_VERSION,
        "bot_username": _DEFAULT_BOT_USERNAME,
        "enabled": False,
    }
    if isinstance(payload, dict):
        merged.update(payload)
    merged["bot_username"] = _normalize_telegram_bot_username(merged.get("bot_username")) or _DEFAULT_BOT_USERNAME
    merged["enabled"] = bool(merged.get("enabled"))
    return merged


def _public_bot_config_record(
    payload: dict[str, Any] | None,
    *,
    scope: TelegramScope,
) -> TelegramBotConfigResponse:
    normalized = _normalize_bot_config_payload(payload)
    return TelegramBotConfigResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        bot_username=normalized["bot_username"],
        enabled=normalized["enabled"],
    )


async def _get_org_secret_repo() -> AuthnzOrgProviderSecretsRepo:
    pool = await get_db_pool()
    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()
    return repo


def _encrypt_telegram_payload(payload: dict[str, Any]) -> str:
    return dumps_envelope(encrypt_byok_payload(payload))


def _decrypt_telegram_payload(encrypted_blob: str | None) -> dict[str, Any] | None:
    if not encrypted_blob:
        return None
    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except Exception:
        logger.warning("Failed to decrypt Telegram bot config payload")
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_shared_scope(
    *,
    principal: AuthPrincipal,
    request: Request | None = None,
) -> TelegramScope:
    request_active_team_id = _coerce_int(getattr(request.state, "active_team_id", None)) if request else None
    request_active_org_id = _coerce_int(getattr(request.state, "active_org_id", None)) if request else None

    active_team_id = _coerce_int(principal.active_team_id)
    if active_team_id is None:
        active_team_id = request_active_team_id
    active_org_id = _coerce_int(principal.active_org_id)
    if active_org_id is None:
        active_org_id = request_active_org_id

    team_ids = _collect_scope_ids(principal.team_ids)
    org_ids = _collect_scope_ids(principal.org_ids)

    if active_team_id is not None:
        if active_team_id not in team_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An active org/team scope is required",
            )
        return TelegramScope(scope_type="team", scope_id=active_team_id)

    if active_org_id is not None:
        if active_org_id not in org_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An active org/team scope is required",
            )
        return TelegramScope(scope_type="org", scope_id=active_org_id)

    visible_scopes: list[TelegramScope] = [TelegramScope("team", team_id) for team_id in team_ids]
    visible_scopes.extend(TelegramScope("org", org_id) for org_id in org_ids)
    if len(visible_scopes) == 1:
        return visible_scopes[0]

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="An active org/team scope is required",
    )


def _extract_callback_query(payload: dict[str, Any]) -> dict[str, Any] | None:
    callback_query = payload.get("callback_query")
    return callback_query if isinstance(callback_query, dict) else None


def _extract_callback_query_actor_id(payload: dict[str, Any]) -> int | None:
    callback_query = _extract_callback_query(payload)
    if callback_query is None:
        return None
    from_block = callback_query.get("from")
    if not isinstance(from_block, dict):
        return None
    return _coerce_int(from_block.get("id"))


def _extract_callback_query_data(payload: dict[str, Any]) -> str | None:
    callback_query = _extract_callback_query(payload)
    if callback_query is None:
        return None
    return _coerce_nonempty_string(callback_query.get("data"))


async def _handle_telegram_approval_callback_query(
    *,
    scope: TelegramScope,
    payload: dict[str, Any],
    runtime_repo: Any,
    get_telegram_approvals_repo: Callable[[], Awaitable[Any]],
    get_approval_service: Callable[[], Awaitable[Any]],
) -> JSONResponse:
    callback_data = _extract_callback_query_data(payload)
    parsed_callback = _parse_telegram_approval_callback_data(callback_data)
    if parsed_callback is None:
        return _telegram_webhook_error(status.HTTP_400_BAD_REQUEST, "invalid_payload")

    now = datetime.now(timezone.utc)
    try:
        approvals_repo = await get_telegram_approvals_repo()
    except Exception:
        logger.error("Failed to resolve Telegram approvals repo")
        return _telegram_webhook_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "approval_service_unavailable",
        )

    pending_approval = await approvals_repo.get_pending_approval_by_token(
        parsed_callback["approval_token"],
        now=now,
    )
    if pending_approval is None:
        return _telegram_webhook_error(status.HTTP_409_CONFLICT, "approval_unavailable")

    telegram_user_id = _extract_callback_query_actor_id(payload)
    if telegram_user_id is None:
        return _telegram_webhook_error(status.HTTP_400_BAD_REQUEST, "invalid_payload")

    try:
        if (
            pending_approval.get("scope_type") != scope.scope_type
            or _coerce_int(pending_approval.get("scope_id")) != scope.scope_id
        ):
            return _telegram_webhook_error(status.HTTP_409_CONFLICT, "approval_unavailable")

        linked_actor = await runtime_repo.get_actor_link(
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
            telegram_user_id=telegram_user_id,
        )
        if linked_actor is None:
            return _telegram_webhook_error(status.HTTP_403_FORBIDDEN, "account_link_required")

        linked_auth_user_id = _coerce_int(linked_actor.get("auth_user_id"))
        initiating_auth_user_id = _coerce_int(pending_approval.get("initiating_auth_user_id"))
        if (
            linked_auth_user_id is None
            or initiating_auth_user_id is None
            or linked_auth_user_id != initiating_auth_user_id
        ):
            return _telegram_webhook_error(status.HTTP_403_FORBIDDEN, "approval_not_authorized")

        consumed_pending = await approvals_repo.consume_pending_approval(
            parsed_callback["approval_token"],
            now=now,
        )
        if consumed_pending is None:
            return _telegram_webhook_error(status.HTTP_409_CONFLICT, "approval_unavailable")

        try:
            approval_service = await get_approval_service()
        except Exception:
            logger.error("Failed to resolve Telegram approval service")
            return _telegram_webhook_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "approval_service_unavailable",
            )

        if not hasattr(approval_service, "record_decision"):
            return _telegram_webhook_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "approval_service_unavailable",
            )

        decision = await approval_service.record_decision(
            approval_policy_id=_coerce_int(consumed_pending.get("approval_policy_id")),
            context_key=str(consumed_pending.get("context_key") or ""),
            conversation_id=_coerce_nonempty_string(consumed_pending.get("conversation_id")),
            tool_name=str(consumed_pending.get("tool_name") or ""),
            scope_key=str(consumed_pending.get("scope_key") or ""),
            decision="approved",
            consume_on_match=True,
            expires_at=None,
            actor_id=linked_auth_user_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "status": "approved",
                "approval_decision_id": decision.get("id") if isinstance(decision, dict) else None,
                "approval_policy_id": consumed_pending.get("approval_policy_id"),
                "scope_key": consumed_pending.get("scope_key"),
            },
        )
    except Exception:
        logger.error("Failed to persist Telegram approval decision")
        return _telegram_webhook_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "approval_service_unavailable",
        )


async def _reset_telegram_runtime_state_for_tests_async() -> None:
    runtime_repo = await get_telegram_runtime_repo()
    await runtime_repo.clear_all_for_tests()


def _reset_telegram_webhook_state_for_tests() -> None:
    """Reset webhook dedupe receipts for deterministic tests."""
    asyncio.run(_reset_telegram_runtime_state_for_tests_async())


def _reset_telegram_link_state_for_tests() -> None:
    """Reset Telegram pairing/link state for deterministic tests."""
    asyncio.run(_reset_telegram_runtime_state_for_tests_async())


def _telegram_webhook_error(
    status_code: int,
    error: str,
    *,
    detail: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": error,
            **({"detail": detail} if detail else {}),
        },
    )


def _coerce_valid_webhook_secret_header(request: Request) -> str | None:
    webhook_secret = _coerce_nonempty_string(request.headers.get("X-Telegram-Bot-Api-Secret-Token"))
    if not webhook_secret:
        return None
    if len(webhook_secret) < TELEGRAM_WEBHOOK_SECRET_MIN_LENGTH:
        return None
    return webhook_secret


async def _resolve_webhook_scope_from_secret(
    *,
    repo: AuthnzOrgProviderSecretsRepo,
    webhook_secret: str,
) -> TelegramWebhookContext | None:
    try:
        rows = await repo.list_secrets(provider=_PROVIDER)
    except Exception as exc:
        logger.error("Failed to list Telegram bot configs for webhook resolution")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram bot configuration is unavailable",
        ) from exc

    matches: list[TelegramWebhookContext] = []
    for row in rows:
        scope_type = str(row.get("scope_type") or "").strip().lower()
        scope_id = _coerce_int(row.get("scope_id"))
        if scope_type not in {"org", "team"} or scope_id is None:
            continue
        try:
            secret_row = await repo.fetch_secret(scope_type, scope_id, _PROVIDER)
        except Exception as exc:
            logger.error("Failed to load Telegram bot config for webhook resolution")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Telegram bot configuration is unavailable",
            ) from exc
        if not secret_row:
            continue
        payload = _decrypt_telegram_payload(secret_row.get("encrypted_blob"))
        if not isinstance(payload, dict):
            continue
        stored_secret = _coerce_nonempty_string(payload.get("webhook_secret"))
        if not stored_secret or not secrets.compare_digest(stored_secret, webhook_secret):
            continue
        if not bool(payload.get("enabled")):
            continue
        matches.append(
            TelegramWebhookContext(
                scope=TelegramScope(scope_type=scope_type, scope_id=scope_id),
                bot_username=_normalize_telegram_bot_username(payload.get("bot_username")) or _DEFAULT_BOT_USERNAME,
            )
        )
        if len(matches) > 1:
            logger.warning("Ambiguous Telegram webhook secret matched multiple scopes; rejecting request")
            return None

    return matches[0] if matches else None


async def _handle_telegram_link_command(
    *,
    scope: TelegramScope,
    payload: dict[str, Any],
    command: TelegramCommand,
    runtime_repo: Any,
) -> JSONResponse:
    telegram_user_id = _extract_message_actor_and_text(payload)[0]
    if telegram_user_id is None:
        return _telegram_webhook_error(status.HTTP_400_BAD_REQUEST, "invalid_payload")
    pairing_code = _normalize_pairing_code(command.input)
    if pairing_code is None:
        return _telegram_webhook_error(status.HTTP_400_BAD_REQUEST, "pairing_code_required")

    consumed_pairing = await runtime_repo.consume_pairing_code(pairing_code)
    if consumed_pairing is None:
        return _telegram_webhook_error(status.HTTP_403_FORBIDDEN, "invalid_pairing_code")

    auth_user_id = _coerce_int(consumed_pairing.get("auth_user_id"))
    if auth_user_id is None:
        return _telegram_webhook_error(status.HTTP_409_CONFLICT, "invalid_pairing_code")

    linked_actor = await runtime_repo.upsert_actor_link(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        telegram_user_id=telegram_user_id,
        auth_user_id=auth_user_id,
        telegram_username=_extract_message_actor_username(payload),
    )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "status": "linked",
            "scope_type": linked_actor.get("scope_type", scope.scope_type),
            "scope_id": linked_actor.get("scope_id", scope.scope_id),
            "telegram_user_id": linked_actor.get("telegram_user_id", telegram_user_id),
        },
    )


async def telegram_webhook_impl(
    *,
    request: Request,
    job_manager: JobManager | None = None,
    get_org_secret_repo: Callable[[], Awaitable[Any]] = _get_org_secret_repo,
    get_telegram_approvals_repo: Callable[[], Awaitable[Any]] = get_telegram_approvals_repo,
    get_telegram_runtime_repo: Callable[[], Awaitable[Any]] = get_telegram_runtime_repo,
    get_approval_service: Callable[[], Awaitable[Any]] = get_mcp_hub_approval_service,
    get_execution_identity_service: Callable[[], Awaitable[Any]] = get_telegram_execution_identity_service,
    dedupe_ttl_seconds: int = _WEBHOOK_REPLAY_WINDOW_SECONDS,
) -> JSONResponse:
    """Validate, dedupe, and route a Telegram webhook update."""
    webhook_secret = _coerce_valid_webhook_secret_header(request)
    if not webhook_secret:
        return _telegram_webhook_error(status.HTTP_401_UNAUTHORIZED, "invalid_secret")

    repo = await get_org_secret_repo()
    webhook_context = await _resolve_webhook_scope_from_secret(
        repo=repo,
        webhook_secret=webhook_secret,
    )
    if webhook_context is None:
        return _telegram_webhook_error(status.HTTP_401_UNAUTHORIZED, "invalid_secret")
    scope = webhook_context.scope
    configured_bot_username = webhook_context.bot_username
    runtime_repo = await get_telegram_runtime_repo()

    raw_body = await request.body()
    try:
        webhook_update = TelegramWebhookUpdate.model_validate_json(raw_body or b"{}")
    except ValidationError as exc:
        if any(error.get("type") == "json_invalid" for error in exc.errors()):
            return _telegram_webhook_error(status.HTTP_400_BAD_REQUEST, "invalid_json")
        return _telegram_webhook_error(status.HTTP_400_BAD_REQUEST, "invalid_payload")
    payload = webhook_update.model_dump(by_alias=True, exclude_none=True)

    update_id_int = int(webhook_update.update_id)

    dedupe_key = f"{scope.scope_type}:{scope.scope_id}:{update_id_int}"
    receipt_stored = await runtime_repo.store_webhook_receipt(
        dedupe_key=dedupe_key,
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        update_id=update_id_int,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=max(1, dedupe_ttl_seconds)),
    )
    if not receipt_stored:
        return JSONResponse(status_code=200, content={"ok": True, "status": "duplicate"})

    callback_query = _extract_callback_query(payload)
    if callback_query is not None:
        return await _handle_telegram_approval_callback_query(
            scope=scope,
            payload=payload,
            runtime_repo=runtime_repo,
            get_telegram_approvals_repo=get_telegram_approvals_repo,
            get_approval_service=get_approval_service,
        )

    telegram_user_id, text = _extract_message_actor_and_text(payload)
    chat_type = _extract_message_chat_type(payload)
    telegram_chat_id = _extract_message_chat_id(payload)
    telegram_thread_id = _extract_message_thread_id(payload)
    message_id = _extract_message_id(payload)
    reply_to_bot = _message_reply_to_bot(payload, configured_bot_username=configured_bot_username)
    message_policy = evaluate_telegram_message_policy(
        chat_type=chat_type,
        text=text,
        bot_username=configured_bot_username,
        reply_to_bot=reply_to_bot,
    )
    if not message_policy.should_process:
        return JSONResponse(status_code=200, content={"ok": True, "status": "ignored"})

    if message_policy.command is not None and message_policy.command.action == "link":
        return await _handle_telegram_link_command(
            scope=scope,
            payload=payload,
            command=message_policy.command,
            runtime_repo=runtime_repo,
        )

    linked_actor = None
    if telegram_user_id is not None:
        linked_actor = await runtime_repo.get_actor_link(
            scope_type=scope.scope_type,
            scope_id=scope.scope_id,
            telegram_user_id=telegram_user_id,
        )

    if _is_privileged_telegram_command(message_policy.command):
        if linked_actor is None:
            return _telegram_webhook_error(status.HTTP_403_FORBIDDEN, "account_link_required")

    if message_policy.command is not None and message_policy.command.action == "ask" and linked_actor is not None:
        if job_manager is None:
            job_manager = await _resolve_job_manager_for_request(request)
        request_id = _build_telegram_request_id(scope, update_id_int)
        payload = _build_telegram_ask_job_payload(
            scope=scope,
            update_id=update_id_int,
            message_id=message_id,
            telegram_user_id=telegram_user_id,
            text=text or "",
            chat_type=chat_type,
            reply_to_bot=reply_to_bot,
            configured_bot_username=configured_bot_username,
            command=message_policy.command,
            linked_actor=linked_actor,
            telegram_chat_id=telegram_chat_id,
            telegram_thread_id=telegram_thread_id,
            request_id=request_id,
        )
        try:
            execution_identity_service = await get_execution_identity_service()
            execution_identity = execution_identity_service.mint_telegram_identity(
                tenant_id=str(payload["session"]["tenant_id"]),
                auth_user_id=str(linked_actor["auth_user_id"]),
                request_id=request_id,
                conversation_id=str(payload["session"]["assistant_conversation_id"]),
                telegram_user_id=telegram_user_id,
                telegram_chat_id=telegram_chat_id,
                telegram_thread_id=telegram_thread_id,
                scope_type=scope.scope_type,
                scope_id=scope.scope_id,
            )
        except Exception:
            logger.error("Failed to mint Telegram execution identity")
            return _telegram_webhook_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "execution_identity_unavailable",
            )
        payload["execution_identity"] = execution_identity.to_payload()
        queued_job = await TelegramDeliveryService(job_manager).queue_inbound_ask(
            owner_user_id=str(linked_actor["auth_user_id"]),
            request_id=request_id,
            payload=payload,
        )
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "status": "queued",
                "request_id": request_id,
                "job_id": queued_job.get("id"),
            },
        )

    return JSONResponse(status_code=200, content={"ok": True, "status": "accepted"})


async def telegram_admin_start_link_impl(
    *,
    principal: AuthPrincipal,
    request: Request | None = None,
    get_telegram_runtime_repo: Callable[[], Awaitable[Any]] = get_telegram_runtime_repo,
) -> dict[str, Any]:
    scope = _resolve_shared_scope(principal=principal, request=request)
    auth_user_id = _coerce_int(principal.user_id)
    if auth_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authenticated user id is required",
        )
    runtime_repo = await get_telegram_runtime_repo()
    record = await _store_telegram_pairing_code(
        scope=scope,
        auth_user_id=auth_user_id,
        runtime_repo=runtime_repo,
    )
    expires_at = _coerce_datetime(record.get("expires_at"))
    return {
        "ok": True,
        "pairing_code": record["pairing_code"],
        "scope_type": record["scope_type"],
        "scope_id": record["scope_id"],
        "expires_at": expires_at.isoformat() if expires_at is not None else str(record.get("expires_at") or ""),
    }


async def telegram_admin_list_linked_actors_impl(
    *,
    principal: AuthPrincipal,
    request: Request | None = None,
    get_telegram_runtime_repo: Callable[[], Awaitable[Any]] = get_telegram_runtime_repo,
) -> TelegramLinkedActorListResponse:
    scope = _resolve_shared_scope(principal=principal, request=request)
    runtime_repo = await get_telegram_runtime_repo()
    items = await runtime_repo.list_actor_links(scope_type=scope.scope_type, scope_id=scope.scope_id)
    return TelegramLinkedActorListResponse(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        items=items,
    )


async def telegram_admin_revoke_linked_actor_impl(
    *,
    link_id: int,
    principal: AuthPrincipal,
    request: Request | None = None,
    get_telegram_runtime_repo: Callable[[], Awaitable[Any]] = get_telegram_runtime_repo,
) -> TelegramLinkedActorRevokeResponse:
    scope = _resolve_shared_scope(principal=principal, request=request)
    runtime_repo = await get_telegram_runtime_repo()
    deleted = await runtime_repo.delete_actor_link(
        link_id=link_id,
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="linked_actor_not_found",
        )
    return TelegramLinkedActorRevokeResponse(
        deleted=True,
        id=int(link_id),
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
    )


async def telegram_admin_put_bot_impl(
    *,
    principal: AuthPrincipal,
    payload: TelegramBotConfigUpdate,
    request: Request | None = None,
    get_org_secret_repo: Callable[[], Awaitable[Any]] = _get_org_secret_repo,
    encrypt_telegram_payload: Callable[[dict[str, Any]], str] = _encrypt_telegram_payload,
) -> TelegramBotConfigResponse:
    bot_token = _coerce_nonempty_string(payload.bot_token)
    webhook_secret = _coerce_nonempty_string(payload.webhook_secret)
    if not bot_token or not webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bot_token and webhook_secret are required",
        )

    scope = _resolve_shared_scope(principal=principal, request=request)
    config_payload = _normalize_bot_config_payload(
        {
            "bot_token": bot_token,
            "webhook_secret": webhook_secret,
            "bot_username": payload.bot_username,
            "enabled": bool(payload.enabled),
        }
    )

    repo = await get_org_secret_repo()
    now = datetime.now(timezone.utc)
    actor_user_id = _coerce_int(principal.user_id)
    await repo.upsert_secret(
        scope_type=scope.scope_type,
        scope_id=scope.scope_id,
        provider=_PROVIDER,
        encrypted_blob=encrypt_telegram_payload(config_payload),
        key_hint=key_hint_for_api_key(bot_token),
        metadata={
            "bot_username": config_payload["bot_username"],
            "enabled": config_payload["enabled"],
            "credential_version": _CREDENTIAL_VERSION,
        },
        updated_at=now,
        created_by=actor_user_id,
        updated_by=actor_user_id,
    )
    return _public_bot_config_record(config_payload, scope=scope)


async def telegram_admin_get_bot_impl(
    *,
    principal: AuthPrincipal,
    request: Request | None = None,
    get_org_secret_repo: Callable[[], Awaitable[Any]] = _get_org_secret_repo,
    decrypt_telegram_payload: Callable[[str | None], dict[str, Any] | None] = _decrypt_telegram_payload,
) -> TelegramBotConfigResponse:
    scope = _resolve_shared_scope(principal=principal, request=request)
    repo = await get_org_secret_repo()
    row = await repo.fetch_secret(scope.scope_type, scope.scope_id, _PROVIDER)
    payload = decrypt_telegram_payload(row.get("encrypted_blob")) if row else None
    return _public_bot_config_record(payload, scope=scope)
