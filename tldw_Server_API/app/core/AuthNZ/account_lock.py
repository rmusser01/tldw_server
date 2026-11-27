"""The account security lock (TASK-15671): the rule, the lock, the unlock and the take-back.

While ``users.security_locked_at`` is set every credential of the account is refused (423). Only the
link mailed to the account's address, or for 7 days after an email change to its previous address,
unlocks it. An unlock sets ``users.credentials_valid_after`` so no token minted before it works again,
and suspends the API keys made in the week before the lock. A link to the previous address also takes
the account back: it restores that address, replaces the password and drops the recent SSO links.

The errors and string constants live in ``account_lock_errors`` (a leaf module) and are re-exported here.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventCategory, AuditEventType
from tldw_Server_API.app.core.AuthNZ import api_key_audit
from tldw_Server_API.app.core.AuthNZ.account_lock_errors import (
    ACCOUNT_LOCK_REVOKE_REASON,
    ACCOUNT_LOCKED_CODE,
    ACCOUNT_RESTORED_REVOKE_REASON,
    AccountSecurityLockedError,
    StaleCredentialError,
    TakeBackUnavailableError,
)
from tldw_Server_API.app.core.AuthNZ.device_label import device_label
from tldw_Server_API.app.core.AuthNZ.email_service import _mask_email, get_email_service
from tldw_Server_API.app.core.AuthNZ.exceptions import RollbackSignal, UserRegistrationException
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.user_profile_writes import restore_previous_email
from tldw_Server_API.app.core.exceptions import ServiceInitializationError
from tldw_Server_API.app.core.Utils.iso_datetime import parse_iso_utc

__all__ = [
    "UNLOCK_PURPOSE",
    "LockAvailability",
    "LockResult",
    "lock_account",
    "lock_availability",
    "lock_availability_for_user",
    "send_lock_mail",
    "ACCOUNT_LOCKED_CODE",
    "ACCOUNT_LOCK_REVOKE_REASON",
    "ACCOUNT_RESTORED_REVOKE_REASON",
    "PREVIOUS_ADDRESS_MAIL_DAYS",
    "RECENT_API_KEY_DAYS",
    "AccountSecurityLockedError",
    "StaleCredentialError",
    "TakeBackUnavailableError",
    "assert_account_not_locked",
    "clear_lock",
    "is_security_locked",
    "is_user_security_locked",
    "security_mail_recipients",
    "set_lock",
    "take_back_address",
    "write_account_audit",
]

RECENT_API_KEY_DAYS = 7
PREVIOUS_ADDRESS_MAIL_DAYS = 7  # the window in which update_user_email keeps the previous address

UNLOCK_PURPOSE = "account_unlock"  # the magic-link ``purpose`` claim of an unlock link

_TAKE_BACK_ATTEMPTS = 3

# Side effects that must never fail a lock or an unlock: session revocation, SSO-link cleanup, audit.
_ACCOUNT_LOCK_NONCRITICAL_EXCEPTIONS = (
    UserRegistrationException,  # SessionError from revoke_all_user_sessions; AuthNZ DatabaseError/TransactionError
    ServiceInitializationError,  # the audit service failed to start, timed out or is shutting down
    sqlite3.Error,  # SQLite statements (aiosqlite re-raises sqlite3 errors)
    asyncpg.PostgresError,  # PostgreSQL statements
    OSError,  # audit and user database files
    RuntimeError,  # FederatedIdentityRepo.ensure_tables without the tables; the session key file checks
    ValueError,  # SessionManager() with a malformed SESSION_ENCRYPTION_KEY; audit tenant resolution
    TimeoutError,  # pool acquisition and statement timeouts
)
# A restore that collides with another account's address (it took the address after step 1).
_UNIQUE_VIOLATIONS = (sqlite3.IntegrityError, asyncpg.UniqueViolationError)


def is_security_locked(user: Mapping[str, Any]) -> bool:
    """Whether the account is locked: any stored lock value counts, readable or not (fail closed)."""
    return user.get("security_locked_at") is not None


def assert_account_not_locked(user: Mapping[str, Any], token_iat: int | float | None = None) -> None:
    """Raise ``AccountSecurityLockedError`` for a locked account, ``StaleCredentialError`` for a dead token.

    A token is dead when it was issued (``token_iat``, whole seconds) before ``credentials_valid_after``.
    """
    if is_security_locked(user):
        raise AccountSecurityLockedError()
    if token_iat is None:
        return
    valid_after = parse_iso_utc(user.get("credentials_valid_after"))
    if valid_after is not None and token_iat < int(valid_after.timestamp()):
        raise StaleCredentialError()


async def is_user_security_locked(user_id: int) -> bool:
    """Read the lock straight from the users row (no cache); False for an unknown id."""
    user = await (await AuthnzUsersRepo.from_pool()).get_user_by_id(int(user_id))
    return user is not None and is_security_locked(user)


def security_mail_recipients(user: Mapping[str, Any], *, now: datetime | None = None) -> list[tuple[str, bool]]:
    """``[(address, is_previous_address)]``: the current email, then the previous one while it is recent."""
    current = str(user.get("email") or "").strip()
    recipients = [(current, False)] if current else []
    previous = str(user.get("previous_email") or "").strip()
    changed_at = parse_iso_utc(user.get("email_changed_at"))
    window_start = (now or datetime.now(timezone.utc)) - timedelta(days=PREVIOUS_ADDRESS_MAIL_DAYS)
    if previous and previous.lower() != current.lower() and changed_at is not None and changed_at > window_start:
        recipients.append((previous, True))
    return recipients


def take_back_address(
    user: Mapping[str, Any], token_payload: Mapping[str, Any], *, now: datetime | None = None
) -> str | None:
    """The token's email when the token restores it and it is the user's recent previous address."""
    if token_payload.get("restore_email") is not True:
        return None
    email = str(token_payload.get("email") or "").strip()
    for address, is_previous in security_mail_recipients(user, now=now):
        if is_previous and address.lower() == email.lower():
            return email
    return None


async def set_lock(user_id: int) -> datetime | None:
    """Lock the account; the lock time if this call locked it, None if it was already locked."""
    now = datetime.now(timezone.utc)
    users = await AuthnzUsersRepo.from_pool()
    if not await users.set_security_lock(int(user_id), now=now):
        return None
    logger.info("Account security lock set for user {}", user_id)
    return now


def _whole_second(value: datetime) -> datetime:
    return value.replace(microsecond=0)


async def _revoke_sessions(
    user_id: int, *, reason: str = ACCOUNT_RESTORED_REVOKE_REASON, revoked_by: int | None = None
) -> None:
    """Sign every session out; a failure is logged and the caller goes on (the lock or the iat rule covers the tokens)."""
    from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

    try:
        manager = await get_session_manager()
        await manager.revoke_all_user_sessions(user_id, reason=reason, revoked_by=revoked_by)
    except _ACCOUNT_LOCK_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Revoking sessions of user {} ({}) failed: {}", user_id, reason, type(exc).__name__)


async def _restore_address(users: AuthnzUsersRepo, user_id: int, address: str, password_hash: str) -> bool:
    """Make ``address`` the email again with ``password_hash``; True when a row changed.

    Raises ``TakeBackUnavailableError`` when another account took the address after step 1.
    """
    backend = "postgres" if users._is_postgres_backend() else "sqlite"
    try:
        async with users.db_pool.transaction() as conn:
            try:
                restored = await restore_previous_email(
                    conn, backend=backend, user_id=user_id, previous_email=address, password_hash=password_hash
                )
            except _UNIQUE_VIOLATIONS:
                raise RollbackSignal() from None  # passes through the transaction wrapper, which rolls back
    except RollbackSignal:
        raise TakeBackUnavailableError() from None
    return bool(restored.affected_user_ids)


async def _take_address_back(users: AuthnzUsersRepo, user_id: int, address: str, since: datetime) -> None:
    """Step 5: restore ``address``, end password sign-in, drop the SSO links made since ``since``.

    Raises ``TakeBackUnavailableError``, with the account still locked, when the address cannot be restored.
    """
    passwords = PasswordService()
    # A password nobody knows; the column is NOT NULL on SQLite, so it is replaced rather than cleared.
    password_hash = passwords.hash_password(passwords.generate_secure_password(64))
    if not await _restore_address(users, user_id, address, password_hash):
        current = str(((await users.get_user_by_id(user_id)) or {}).get("email") or "")
        if current.strip().lower() != address.strip().lower():
            logger.info("Take-back of user {} refused: the previous address is no longer on the account", user_id)
            raise TakeBackUnavailableError()
        # Already restored (a retry, or a second link to the same address): the password is replaced anyway.
        async with users.db_pool.transaction() as conn:
            await passwords.set_password(user_id, password_hash, conn, is_postgres=users._is_postgres_backend())
    links = FederatedIdentityRepo(users.db_pool)
    try:
        await links.ensure_tables()
    except _ACCOUNT_LOCK_NONCRITICAL_EXCEPTIONS as exc:
        logger.info("Take-back of user {} dropped no SSO links: {}", user_id, type(exc).__name__)
        return
    await links.delete_identity_links_created_since(user_id=user_id, since=since)


async def _clear_once(
    users: AuthnzUsersRepo, user_id: int, lock_value: Any, since: datetime, *, take_back: str | None
) -> datetime | None:
    """Steps 3-8 against one stored lock value; None when another request changed that lock first."""
    now = datetime.now(timezone.utc)
    keys = AuthnzApiKeysRepo(db_pool=users.db_pool)
    await keys.suspend_keys_created_since(user_id=user_id, since=since, now=now)  # 3.
    await _revoke_sessions(user_id)  # 4.
    if take_back is not None:  # 5.
        await _take_address_back(users, user_id, take_back, since)
    await keys.suspend_keys_created_since(user_id=user_id, since=since, now=now)  # 6. a key made meanwhile

    # 7. Every token minted up to the lock second is older than valid_after.
    lock_time = parse_iso_utc(lock_value) or now
    valid_after = max(_whole_second(datetime.now(timezone.utc)), _whole_second(lock_time) + timedelta(seconds=1))
    if not await users.clear_security_lock(user_id, locked_at=lock_value, valid_after=valid_after):
        return None

    # 8. Return once valid_after has begun, so the caller's new tokens are not older than it.
    wait = (valid_after - datetime.now(timezone.utc)).total_seconds()
    await asyncio.sleep(min(1.0, max(0.0, wait)))
    logger.info("Account security lock cleared for user {} (take-back: {})", user_id, take_back is not None)
    return valid_after


async def clear_lock(user_id: int, *, take_back: str | None = None) -> datetime | None:
    """Unlock the account; with ``take_back`` (its recent previous address) also take it back.

    Returns the ``credentials_valid_after`` it set, once that second has begun. Returns None when it did
    nothing to unlock: not locked and no take-back, another request cleared the same lock first, or a
    take-back that lost its compare-and-set on every attempt (it leaves the account locked). Raises
    ``TakeBackUnavailableError`` when the address cannot be restored: before any write when another account
    holds it, otherwise from step 5 with the account left locked.
    """
    user_id = int(user_id)
    users = await AuthnzUsersRepo.from_pool()
    # 1. One read; every value below comes from it.
    user = await users.get_user_by_id(user_id)
    if user is None or (not is_security_locked(user) and take_back is None):
        return None
    if take_back is not None and await users.email_in_use_by_another_user(take_back, user_id=user_id):
        raise TakeBackUnavailableError()
    now = datetime.now(timezone.utc)
    week = timedelta(days=RECENT_API_KEY_DAYS)
    starts = []
    if is_security_locked(user):
        starts.append((parse_iso_utc(user["security_locked_at"]) or now) - week)
    changed_at = parse_iso_utc(user.get("email_changed_at"))
    if take_back is not None and changed_at is not None:
        starts.append(changed_at - week)
    since = min(starts, default=now - week)

    if take_back is None:
        valid_after = await _clear_once(users, user_id, user["security_locked_at"], since, take_back=None)
        if valid_after is None:
            logger.info("Unlock of user {} did nothing: another request cleared the lock first", user_id)
        return valid_after

    # 2. A take-back writes a fresh lock first, so nothing can mint a credential while it runs. When its
    #    compare-and-set loses (another unlock or take-back changed the lock meanwhile, and may have let a
    #    credential through), it locks again and redoes steps 3-7. It unlocks only by winning its own
    #    compare-and-set; after the last attempt it leaves the account locked.
    for attempt in range(1, _TAKE_BACK_ATTEMPTS + 1):
        await users.force_security_lock(user_id, now=datetime.now(timezone.utc))
        lock_value = ((await users.get_user_by_id(user_id)) or {}).get("security_locked_at")
        valid_after = await _clear_once(users, user_id, lock_value, since, take_back=take_back)
        if valid_after is not None:
            return valid_after
        logger.info("Take-back of user {} lost its compare-and-set (attempt {})", user_id, attempt)
    await users.force_security_lock(user_id, now=datetime.now(timezone.utc))
    logger.warning("Take-back of user {} gave up after {} attempts; the account stays locked", user_id, _TAKE_BACK_ATTEMPTS)
    return None


async def write_account_audit(
    event_type: AuditEventType,
    *,
    user_id: int,
    session_id: int | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Write one account event to the user's audit log and flush it; failures are logged, never raised."""
    try:
        service = await api_key_audit._get_or_create_audit_service(int(user_id))
        await service.log_event(
            event_type=event_type,
            context=AuditContext(
                user_id=str(user_id),
                session_id=None if session_id is None else str(session_id),
                ip_address=ip_address,
                user_agent=user_agent,
            ),
            resource_type="user",
            resource_id=str(user_id),
            metadata=metadata,
        )
        flushed = await service.flush()
    except _ACCOUNT_LOCK_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Audit event {} not written for user {}: {}", event_type.value, user_id, type(exc).__name__)
        return
    if not flushed:
        logger.warning("Audit event {} not flushed for user {}", event_type.value, user_id)


async def unsuspend_api_key(*, user_id: int, key_id: int) -> dict | None:
    """Restore one suspended key of the user; its metadata row, or None when it is not theirs or not suspended."""
    repo = AuthnzApiKeysRepo(db_pool=(await AuthnzUsersRepo.from_pool()).db_pool)
    # The restore and its mandatory audit commit together: a failed audit raises and rolls the restore back.
    async with repo.db_pool.transaction() as conn:
        if not await repo.unsuspend_key_for_user(key_id=key_id, user_id=user_id, conn=conn):
            return None
        await api_key_audit.emit_mandatory_api_key_management_audit(
            user_id=user_id,
            event_type=AuditEventType.DATA_UPDATE,
            category=AuditEventCategory.DATA_MODIFICATION,
            action="api_key.unsuspend",
            resource_id=str(key_id),
            actor_user_id=user_id,
            actor_kind="user",
        )
    row = await repo.fetch_key_for_user(key_id=key_id, user_id=user_id)
    if row is None:
        return None
    row = dict(row)
    row.pop("key_hash", None)
    return row


@dataclass(frozen=True)
class LockAvailability:
    """Whether the account can be locked, why not, and the masked addresses the unlock link would go to."""

    available: bool
    reason: str | None
    sent_to: list[str]  # masked addresses the unlock link would go to


@dataclass(frozen=True)
class LockResult:
    """What a lock did: when, whether the unlock mail went out, and to which masked addresses."""

    locked_at: datetime
    email_sent: bool
    sent_to: list[str]  # masked addresses that were mailed


def lock_availability(user: Mapping[str, Any]) -> LockAvailability:
    """Whether the user may lock: only when the unlock link can reach a verified address."""
    mail = get_email_service()
    recipients = [_mask_email(address) for address, _ in security_mail_recipients(user)]
    reason = None
    if get_settings().AUTH_MODE == "single_user":
        reason = "single_user_mode"
    elif mail.provider != "smtp":
        reason = "mail_not_configured"
    elif mail.public_web_base_url() is None:
        reason = "public_url_not_configured"
    elif not user.get("is_verified"):
        reason = "email_not_verified"
    return LockAvailability(available=reason is None, reason=reason, sent_to=recipients)


async def lock_availability_for_user(user_id: int) -> LockAvailability:
    """``lock_availability`` of the stored user row (an unknown id is never available)."""
    return lock_availability(await (await AuthnzUsersRepo.from_pool()).get_user_by_id(int(user_id)) or {})


async def send_lock_mail(
    user: Mapping[str, Any],
    *,
    locked_at: datetime | None,
    device: str,
    ip_address: str | None,
    only_address: str | None = None,
) -> list[str]:
    """Mail one unlock link per recipient (or only ``only_address``); the masked addresses whose send succeeded."""
    recipients = security_mail_recipients(user)
    if only_address is not None:
        recipients = [r for r in recipients if r[0].lower() == only_address.lower()] or recipients
    minutes = max(1, int(get_settings().MAGIC_LINK_EXPIRE_MINUTES))
    jwt = get_jwt_service()
    mail = get_email_service()

    async def _send(address: str, is_previous: bool) -> bool:
        try:
            token = jwt.create_magic_link_token(
                email=address,
                user_id=int(user["id"]),
                expires_in_minutes=minutes,
                additional_claims={"purpose": UNLOCK_PURPOSE, "restore_email": is_previous},
            )
            return bool(
                await mail.send_account_locked_email(
                    to_email=address,
                    username=user.get("username"),
                    magic_token=token,
                    expires_in_minutes=minutes,
                    locked_at=locked_at,
                    device=device,
                    ip_address=ip_address,
                    restores_address=is_previous,
                )
            )
        except _ACCOUNT_LOCK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Lock mail not sent for user {}: {}", user.get("id"), type(exc).__name__)
            return False

    results = await asyncio.gather(*(_send(address, is_previous) for address, is_previous in recipients))
    return [_mask_email(address) for (address, _), ok in zip(recipients, results) if ok]


async def lock_account(
    *, user_id: int, acting_session_id: int, ip_address: str | None, user_agent: str | None
) -> LockResult:
    """Lock the caller's own account, sign every session out and mail the unlock link."""
    user = await (await AuthnzUsersRepo.from_pool()).get_user_by_id(int(user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    availability = lock_availability(user)
    if not availability.available:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "lock_unavailable", "reason": availability.reason},
        )
    locked_at = await set_lock(user_id)
    if locked_at is None:
        raise AccountSecurityLockedError()
    await _revoke_sessions(user_id, reason=ACCOUNT_LOCK_REVOKE_REASON, revoked_by=user_id)
    sent_to: list[str] = []
    try:
        sent_to = await send_lock_mail(
            user, locked_at=locked_at, device=device_label(user_agent), ip_address=ip_address
        )
    except _ACCOUNT_LOCK_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Lock mail failed for user {}: {}", user_id, type(exc).__name__)
    await write_account_audit(
        AuditEventType.AUTH_ACCOUNT_LOCKED,
        user_id=user_id,
        session_id=acting_session_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    await write_account_audit(
        AuditEventType.AUTH_ACCOUNT_LOCK_EMAIL_SENT,
        user_id=user_id,
        session_id=acting_session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        metadata={"recipient_count": len(sent_to)},
    )
    return LockResult(locked_at=locked_at, email_sent=bool(sent_to), sent_to=sent_to)
