# api_key_manager.py
# Description: API key management with rotation, expiration, and revocation capabilities
#
# Imports
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union
from weakref import WeakKeyDictionary

#
# 3rd-party imports
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
    format_api_key,
    generate_api_key_id,
    generate_api_key_secret,
    is_kdf_hash,
    kdf_hash_api_key,
    parse_api_key,
    verify_kdf_hash,
)
from tldw_Server_API.app.core.AuthNZ.crypto_utils import (
    derive_hmac_key,
    derive_hmac_key_candidates,
)

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, InvalidTokenError, TransactionError
from tldw_Server_API.app.core.AuthNZ.api_key_audit import (
    emit_mandatory_api_key_management_audit,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo


_DEFAULT_API_KEY_USAGE_TOUCH_INTERVAL_SECONDS = 1.0
_MAX_API_KEY_USAGE_TOUCH_CACHE_SIZE = 4096


def _read_usage_touch_interval_seconds() -> float:
    """Read per-process API key usage write throttle interval."""
    raw = os.getenv("API_KEY_USAGE_TOUCH_INTERVAL_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_API_KEY_USAGE_TOUCH_INTERVAL_SECONDS
    try:
        parsed = float(raw)
    except ValueError:
        logger.warning(
            "Invalid API_KEY_USAGE_TOUCH_INTERVAL_SECONDS={!r}; using default={}",
            raw,
            _DEFAULT_API_KEY_USAGE_TOUCH_INTERVAL_SECONDS,
        )
        return _DEFAULT_API_KEY_USAGE_TOUCH_INTERVAL_SECONDS
    return max(parsed, 0.0)


def _compute_hmac_fingerprint(settings: Settings) -> str:
    """
    Compute a non-reversible fingerprint of HMAC key material for cache invalidation.

    This mirrors the precedence and candidate selection used by
    derive_hmac_key_candidates so that changes to SINGLE_USER_API_KEY,
    API_KEY_PEPPER, or JWT secrets/keys all produce a new fingerprint.
    """
    try:
        candidates = derive_hmac_key_candidates(settings)
        if not candidates:
            return ""
        # Use the current key candidate (first entry) as fingerprint material.
        # The candidate is already a 32-byte SHA256 digest; hash once more and
        # return a hex string to avoid exposing raw key bytes.
        return hashlib.sha256(candidates[0]).hexdigest()
    except (TypeError, ValueError, AttributeError) as exc:
        # Preserve previous behavior: on any settings/derivation issue, return
        # an empty string rather than raising during manager initialization.
        logger.debug("APIKeyManager: HMAC fingerprint derivation failed: {}", exc)
        return ""

#######################################################################################################################
#
# Enums and Constants
#

class APIKeyStatus(Enum):
    """API key status states"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATED = "rotated"

class APIKeyScope(Enum):
    """API key permission scopes"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SERVICE = "service"


# Valid scope values for validation
VALID_SCOPE_VALUES = frozenset({"read", "write", "admin", "service"})


def normalize_scope(scope: Optional[Union[str, list[str]]]) -> set[str]:
    """
    Normalize stored scope value to an explicit set of scope strings.

    Supports both single string scopes and list scopes for backward compatibility.

    Args:
        scope: A scope string, list of scope strings, JSON array string, or None

    Returns:
        Set of normalized scope strings (lowercase, stripped)

    Examples:
        >>> normalize_scope(None)
        set()
        >>> normalize_scope("write")
        {'write'}
        >>> normalize_scope(["read", "write"])
        {'read', 'write'}
        >>> normalize_scope('["read", "admin"]')
        {'read', 'admin'}
    """
    if scope is None:
        return set()

    if isinstance(scope, str):
        scope_str = scope.strip()
        # Handle JSON array stored as string
        if scope_str.startswith("["):
            try:
                parsed = json.loads(scope_str)
                if isinstance(parsed, list):
                    return {s.strip().lower() for s in parsed if isinstance(s, str) and s.strip()}
            except json.JSONDecodeError:
                pass
        # Single scope string
        return {scope_str.lower()}

    if isinstance(scope, (list, tuple)):
        return {s.strip().lower() for s in scope if isinstance(s, str) and s.strip()}

    return set()


def has_scope(key_scopes: set[str], required_scope: str) -> bool:
    """
    Check if key scopes satisfy the required scope using explicit matching.

    Admin and service scopes always satisfy any requirement (superuser bypass).
    Otherwise, the required scope must be explicitly present in key_scopes.

    Args:
        key_scopes: Set of scope strings from the API key
        required_scope: Single scope string to check for

    Returns:
        True if the key has the required scope or has admin/service bypass
    """
    # Admin and service scopes have full access
    if "admin" in key_scopes or "service" in key_scopes:
        return True

    # Explicit scope matching
    return required_scope.lower() in key_scopes


_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_WRITE_ENDPOINT_HINTS = (
    "chat.",
    "rag.",
    "embeddings",
    "audio.",
    "evaluations.",
    "research.",
    "media.",
    "mcp.",
)


def _serialize_scope_value(scope: Optional[Union[str, list[str]]]) -> Optional[str]:
    """Serialize an optional scope value for storage."""
    if scope is None:
        return None
    if isinstance(scope, (list, tuple, set)):
        values = [
            str(s).strip().lower()
            for s in scope
            if s is not None and str(s).strip()
        ]
        return json.dumps(values) if values else None
    scope_str = str(scope).strip().lower()
    return scope_str or None


def _infer_virtual_key_scope(
    *,
    scope: Optional[Union[str, list[str]]],
    allowed_methods: Optional[list[str]],
    allowed_endpoints: Optional[list[str]],
) -> str:
    """
    Infer a safe default scope for virtual keys.

    - Explicit `scope` always wins.
    - Non-read HTTP methods imply write scope.
    - Known write-style endpoint hints imply write scope.
    - Otherwise default to read scope to preserve legacy behavior.
    """
    explicit = _serialize_scope_value(scope)
    if explicit:
        return explicit

    if allowed_methods:
        methods_upper = [str(m).strip().upper() for m in allowed_methods if str(m).strip()]
        if any(m not in _READ_METHODS for m in methods_upper):
            return "write"
        return "read"

    if allowed_endpoints:
        for endpoint_id in allowed_endpoints:
            ep = str(endpoint_id or "").strip().lower()
            if not ep:
                continue
            if any(ep.startswith(hint) for hint in _WRITE_ENDPOINT_HINTS):
                return "write"
        return "read"

    return "read"


#######################################################################################################################
#
# API Key Manager Class
#

class APIKeyManager:
    """Manages API keys with rotation, expiration, and revocation capabilities"""

    def __init__(self, db_pool: Optional[DatabasePool] = None):
        """Initialize API key manager"""
        self._db_pool: Optional[DatabasePool] = None
        self._repo: Optional[AuthnzApiKeysRepo] = None
        # Use the property setter so that any future re-binding of db_pool
        # explicitly clears the cached repository.
        self.db_pool = db_pool
        self._initialized = False
        self.settings = get_settings()
        self.key_prefix = "tldw_"  # Prefix for identifying our API keys
        # Fingerprint the HMAC key material to detect settings changes (e.g., JWT_SECRET_KEY)
        self._hmac_key_fingerprint = _compute_hmac_fingerprint(self.settings)
        self._usage_touch_interval_seconds = _read_usage_touch_interval_seconds()
        self._usage_touch_cache: dict[int, float] = {}
        self._usage_touch_cache_guard = threading.Lock()

    def _should_skip_usage_touch(self, key_id: int) -> bool:
        """Return True when usage write for this key should be throttled."""
        interval_seconds = self._usage_touch_interval_seconds
        if interval_seconds <= 0:
            return False

        now_mono = time.monotonic()
        with self._usage_touch_cache_guard:
            last_touch = self._usage_touch_cache.get(key_id)
            if last_touch is not None and (now_mono - last_touch) < interval_seconds:
                return True

            self._usage_touch_cache[key_id] = now_mono
            if len(self._usage_touch_cache) > _MAX_API_KEY_USAGE_TOUCH_CACHE_SIZE:
                cutoff = now_mono - interval_seconds
                stale_keys = [
                    candidate_key
                    for candidate_key, touched_at in self._usage_touch_cache.items()
                    if touched_at < cutoff
                ]
                for stale_key in stale_keys:
                    self._usage_touch_cache.pop(stale_key, None)
        return False

    def _clear_usage_touch(self, key_id: int) -> None:
        """Clear a key's touch cache entry so the next request can retry a usage write."""
        if self._usage_touch_interval_seconds <= 0:
            return
        with self._usage_touch_cache_guard:
            self._usage_touch_cache.pop(key_id, None)

    @property
    def db_pool(self) -> Optional[DatabasePool]:
        """Current database pool bound to this manager."""
        return getattr(self, "_db_pool", None)

    @db_pool.setter
    def db_pool(self, value: Optional[DatabasePool]) -> None:
        """
        Bind a database pool and reset the cached repository when it changes.

        This keeps the AuthnzApiKeysRepo lifecycle obvious when tests or
        callers swap out the underlying DatabasePool.
        """
        if getattr(self, "_db_pool", None) is not value:
            self._db_pool = value
            if hasattr(self, "_repo"):
                self._repo = None

    def _db_context_hint(self) -> str:
        """
        Return a short, non-sensitive description of the current AuthNZ DB context.

        Used only for error messages to help diagnose misconfigured tests or
        startup issues without logging full connection strings or secrets.
        """
        try:
            auth_mode = getattr(self.settings, "AUTH_MODE", None)
            db_url = getattr(self.settings, "DATABASE_URL", None)
            db_url_set = bool(db_url)
        except (AttributeError, TypeError):
            return "(AuthNZ settings unavailable)"
        return f"(AUTH_MODE={auth_mode}, DATABASE_URL_set={db_url_set})"

    def _get_repo(self) -> "AuthnzApiKeysRepo":
        """
        Lazily construct an AuthnzApiKeysRepo bound to the current db_pool.

        Import is local to avoid circular dependencies between the manager
        and the repository module.

        Raises:
            DatabaseError: If no database pool has been configured.
        """
        if self.db_pool is None:
            raise DatabaseError(
                f"APIKeyManager database pool is not initialized {self._db_context_hint()}"
            )
        from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo

        if self._repo is None or getattr(self._repo, "db_pool", None) is not self.db_pool:
            self._repo = AuthnzApiKeysRepo(self.db_pool)
        return self._repo

    @staticmethod
    def _coerce_json_field(value: Any) -> Optional[Any]:
        """
        Normalize stored JSON/JSONB fields that may be parsed or serialized.

        Preserves previous behavior by raising JSON decode errors to callers.
        """
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str) and value.strip():
            return json.loads(value)
        return None

    def _parse_expires_at(self, expires_at_raw: Any) -> Optional[datetime]:
        """Parse and normalize expires_at to a timezone-aware datetime.

        Returns ``None`` when the value is missing or cannot be parsed.
        """
        if expires_at_raw is None:
            return None

        expires_at: Optional[datetime]
        if isinstance(expires_at_raw, datetime):
            expires_at = expires_at_raw
        elif isinstance(expires_at_raw, str):
            expires_at_str = expires_at_raw.strip()
            if not expires_at_str:
                return None
            if expires_at_str.endswith("Z"):
                expires_at_str = expires_at_str[:-1] + "+00:00"
            try:
                expires_at = datetime.fromisoformat(expires_at_str)
            except ValueError:
                return None
        else:
            return None

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return expires_at

    async def initialize(self) -> None:
        """Initialize database connection and ensure tables exist"""
        if self._initialized:
            return

        # Get database pool
        if self.db_pool is None:
            self.db_pool = await get_db_pool()

        # Create API keys table if it doesn't exist
        await self._create_tables()

        self._initialized = True
        logger.info("APIKeyManager initialized")

    async def _create_tables(self) -> None:
        """Create API keys and related tables if they don't exist"""
        try:
            repo = self._get_repo()
            try:
                await repo.ensure_tables()
            except RuntimeError as exc:
                # SQLite bootstrap: if the AuthNZ migrations have not yet created
                # the api_keys tables, run the centralized migration helper and
                # retry once before surfacing an error. This keeps the behaviour
                # consistent with UsersDB.initialize while preserving explicit
                # failures when migrations remain misconfigured.
                msg = str(exc)
                try:
                    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables  # noqa: WPS433
                except Exception:
                    ensure_authnz_tables = None  # type: ignore[assignment]

                db_path = getattr(self.db_pool, "db_path", None)
                if (
                    ensure_authnz_tables is not None
                    and db_path is not None
                    and "SQLite api_keys tables are missing" in msg
                ):
                    try:
                        from pathlib import Path as _Path  # noqa: WPS433

                        ensure_authnz_tables(_Path(db_path))
                        await repo.ensure_tables()
                    except Exception:
                        # Fall through to the outer handler with the original context
                        raise
                else:
                    raise
            logger.debug("API keys tables and indexes created/verified")
        except Exception as e:
            logger.exception("Failed to create API keys tables")
            raise DatabaseError(
                f"Failed to create API keys tables {self._db_context_hint()}"
            ) from e

    def generate_api_key(self) -> tuple[str, str, str]:
        """
        Generate a new API key

        Returns:
            Tuple of (full_key, key_hash, key_id)
            - full_key: The complete API key to give to the user
            - key_hash: The hash to store in the database
            - key_id: Embedded key identifier for fast lookup
        """
        key_id = generate_api_key_id()
        secret = generate_api_key_secret()
        full_key = format_api_key(key_id, secret)
        key_hash = kdf_hash_api_key(full_key)

        return full_key, key_hash, key_id

    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for storage or comparison.

        New-format keys use a slow PBKDF2-HMAC-SHA256 KDF to protect
        low-entropy secrets. Legacy keys retain HMAC-SHA256 hashing.

        Args:
            api_key: The API key to hash

        Returns:
            Encoded hash string (KDF for new-format keys, HMAC-SHA256 hex for legacy)
        """
        if parse_api_key(api_key):
            return kdf_hash_api_key(api_key)
        candidates = self.hash_candidates(api_key)
        if not candidates:
            raise ValueError("Unable to derive API key hash candidates")
        return candidates[0]

    def hash_candidates(self, api_key: str) -> list[str]:
        """Return ordered HMAC hashes for API keys across active/legacy secrets."""
        hashes: list[str] = []
        try:
            key_candidates = derive_hmac_key_candidates(self.settings)
        except Exception:
            key_candidates = [derive_hmac_key(self.settings)]
        for key in key_candidates:
            digest = hmac.new(key, api_key.encode("utf-8"), hashlib.sha256).hexdigest()
            if digest not in hashes:
                hashes.append(digest)
        return hashes

    async def _verify_new_format_key(
        self,
        api_key: str,
        key_identifier: str,
        repo: "AuthnzApiKeysRepo",
    ) -> Optional[tuple[dict[str, Any], Optional[str]]]:
        """Verify new-format key with embedded key_id."""
        result = await repo.fetch_active_by_key_id(key_identifier)
        if not result:
            return None

        key_info = dict(result)
        stored_hash = key_info.get("key_hash")
        if not stored_hash:
            return None

        if is_kdf_hash(stored_hash):
            if not verify_kdf_hash(api_key, stored_hash):
                return None
            return key_info, None

        hash_candidates = self.hash_candidates(api_key)
        if not hash_candidates:
            return None
        if not any(hmac.compare_digest(stored_hash, cand) for cand in hash_candidates):
            return None
        return key_info, hash_candidates[0]

    async def _verify_legacy_key(
        self,
        api_key: str,
        repo: "AuthnzApiKeysRepo",
    ) -> Optional[tuple[dict[str, Any], Optional[str]]]:
        """Verify legacy key without embedded key_id."""
        hash_candidates = self.hash_candidates(api_key)
        if not hash_candidates:
            return None

        result = await repo.fetch_active_by_hash_candidates(hash_candidates)
        if not result:
            return None

        return dict(result), hash_candidates[0]

    async def create_api_key(
        self,
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Union[str, list[str]] = "read",
        expires_in_days: Optional[int] = 90,
        rate_limit: Optional[int] = None,
        allowed_ips: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        actor_user_id: Optional[int] = None,
        actor_subject: Optional[str] = None,
        actor_kind: Optional[str] = None,
        actor_roles: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Create a new API key for a user

        Args:
            user_id: User ID who owns the key
            name: Optional name for the key
            description: Optional description
            scope: Permission scope string or list of scopes (read, write, admin, service)
            expires_in_days: Days until expiration (None = no expiration)
            rate_limit: Custom rate limit for this key
            allowed_ips: List of allowed IP addresses
            metadata: Additional metadata

        Returns:
            Dictionary with key information including the actual key (only shown once)
        """
        if not self._initialized:
            await self.initialize()

        # Normalize scope for storage (list scopes stored as JSON)
        if scope is None:
            scope = "read"
        stored_scope = json.dumps(scope) if isinstance(scope, (list, tuple)) else scope

        # Generate the key
        full_key, key_hash, key_identifier = self.generate_api_key()
        key_prefix = full_key[:10] + "..."  # Store prefix for identification

        # Calculate expiration
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        repo = self._get_repo()
        try:
            async with self.db_pool.transaction() as conn:
                key_id = await repo.create_api_key_row(
                    user_id=user_id,
                    key_hash=key_hash,
                    key_identifier=key_identifier,
                    key_prefix=key_prefix,
                    name=name,
                    description=description,
                    scope=stored_scope,
                    expires_at=expires_at,
                    rate_limit=rate_limit,
                    allowed_ips=allowed_ips,
                    metadata=metadata,
                    conn=conn,
                )

                await emit_mandatory_api_key_management_audit(
                    user_id=user_id,
                    event_type=AuditEventType.DATA_WRITE,
                    category=AuditEventCategory.DATA_MODIFICATION,
                    action="api_key.create",
                    resource_id=str(key_id),
                    metadata={
                        "scope": scope,
                        "name": name,
                        "expires_in_days": expires_in_days,
                    },
                    actor_user_id=actor_user_id,
                    actor_subject=actor_subject,
                    actor_kind=actor_kind,
                    actor_roles=actor_roles,
                )

            # Keep legacy API-key audit mirror as best-effort compatibility.
            await self._log_action(key_id, "created", user_id)

            if self.settings.PII_REDACT_LOGS:
                logger.info("Created API key for authenticated user (details redacted)")
            else:
                logger.info(f"Created API key {key_id} for user {user_id}")

            return {
                "id": key_id,
                "key": full_key,  # Only returned on creation!
                "key_prefix": key_prefix,
                "name": name,
                "scope": scope,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": "Store this key securely - it will not be shown again"
            }
        except MandatoryAuditWriteError:
            raise
        except Exception as e:
            logger.exception("Failed to create API key")
            raise DatabaseError(
                f"Failed to create API key {self._db_context_hint()}"
            ) from e

    async def create_virtual_key(
        self,
        *,
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        expires_in_days: Optional[int] = 30,
        org_id: Optional[int] = None,
        team_id: Optional[int] = None,
        scope: Optional[Union[str, list[str]]] = None,
        allowed_endpoints: Optional[list[str]] = None,
        allowed_providers: Optional[list[str]] = None,
        allowed_models: Optional[list[str]] = None,
        budget_day_tokens: Optional[int] = None,
        budget_month_tokens: Optional[int] = None,
        budget_day_usd: Optional[float] = None,
        budget_month_usd: Optional[float] = None,
        parent_key_id: Optional[int] = None,
        # Extra generic constraints (stored in metadata)
        allowed_methods: Optional[list[str]] = None,
        allowed_paths: Optional[list[str]] = None,
        max_calls: Optional[int] = None,
        max_runs: Optional[int] = None,
        actor_user_id: Optional[int] = None,
        actor_subject: Optional[str] = None,
        actor_kind: Optional[str] = None,
        actor_roles: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Create a Virtual API Key with LLM endpoint scope and budgets."""
        if not self._initialized:
            await self.initialize()

        full_key, key_hash, key_identifier = self.generate_api_key()
        key_prefix = full_key[:10] + "..."
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        effective_scope = _infer_virtual_key_scope(
            scope=scope,
            allowed_methods=allowed_methods,
            allowed_endpoints=allowed_endpoints,
        )

        repo = self._get_repo()
        try:
            async with self.db_pool.transaction() as conn:
                key_id = await repo.create_virtual_key_row(
                    user_id=user_id,
                    key_hash=key_hash,
                    key_identifier=key_identifier,
                    key_prefix=key_prefix,
                    name=name,
                    description=description,
                    expires_at=expires_at,
                    org_id=org_id,
                    team_id=team_id,
                    scope=effective_scope,
                    allowed_endpoints=allowed_endpoints,
                    allowed_providers=allowed_providers,
                    allowed_models=allowed_models,
                    budget_day_tokens=budget_day_tokens,
                    budget_month_tokens=budget_month_tokens,
                    budget_day_usd=budget_day_usd,
                    budget_month_usd=budget_month_usd,
                    parent_key_id=parent_key_id,
                    allowed_methods=allowed_methods,
                    allowed_paths=allowed_paths,
                    max_calls=max_calls,
                    max_runs=max_runs,
                    conn=conn,
                )

                await emit_mandatory_api_key_management_audit(
                    user_id=user_id,
                    event_type=AuditEventType.DATA_WRITE,
                    category=AuditEventCategory.DATA_MODIFICATION,
                    action="api_key.create_virtual",
                    resource_id=str(key_id),
                    metadata={
                        "scope": effective_scope,
                        "org_id": org_id,
                        "team_id": team_id,
                        "allowed_endpoints": allowed_endpoints or [],
                    },
                    actor_user_id=actor_user_id,
                    actor_subject=actor_subject,
                    actor_kind=actor_kind,
                    actor_roles=actor_roles,
                )

            # Keep legacy API-key audit mirror as best-effort compatibility.
            await self._log_action(key_id, "created_virtual", user_id, {
                "org_id": org_id, "team_id": team_id, "budgets": {
                    "day_tokens": budget_day_tokens,
                    "month_tokens": budget_month_tokens,
                    "day_usd": budget_day_usd,
                    "month_usd": budget_month_usd,
                },
                "allowed_endpoints": allowed_endpoints or [],
                "scope": effective_scope,
            })

            return {
                "id": key_id,
                "key": full_key,
                "key_prefix": key_prefix,
                "name": name,
                "scope": effective_scope,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": "Store this key securely - it will not be shown again"
            }

        except MandatoryAuditWriteError:
            raise
        except Exception as e:
            logger.exception("Failed to create virtual API key")
            raise DatabaseError(
                f"Failed to create virtual API key {self._db_context_hint()}"
            ) from e

    async def validate_api_key(
        self,
        api_key: str,
        required_scope: Optional[str] = None,
        ip_address: Optional[str] = None,
        record_usage: bool = True,
        usage_details: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Validate an API key and return its information

        Args:
            api_key: The API key to validate
            required_scope: Required permission scope
            ip_address: Client IP address for validation and logging
            record_usage: Whether to record usage/audit side effects
            usage_details: Optional structured context persisted in API key usage audit rows

        Returns:
            Key information if valid, None if invalid
        """
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()
            key_id_info = parse_api_key(api_key)
            primary_hash: Optional[str] = None

            if key_id_info:
                key_identifier, _secret = key_id_info
                verification = await self._verify_new_format_key(api_key, key_identifier, repo)
            else:
                verification = await self._verify_legacy_key(api_key, repo)

            if not verification:
                return None

            key_info, primary_hash = verification
            stored_hash = key_info.get("key_hash")

            # Check expiration
            expires_at_raw = key_info.get("expires_at")
            if expires_at_raw:
                expires_at = self._parse_expires_at(expires_at_raw)
                if expires_at is None:
                    if self.settings.PII_REDACT_LOGS:
                        logger.error("API key expires_at could not be parsed; denying access (details redacted)")
                    else:
                        logger.error(
                            f"API key {key_info.get('id')} expires_at could not be parsed; denying access"
                        )
                    return None
                now_utc = datetime.now(timezone.utc)
                if expires_at < now_utc:
                    await self._mark_expired(key_info["id"])
                    return None

            # Check IP restrictions
            if key_info.get("allowed_ips"):
                try:
                    allowed_ips_raw = self._coerce_json_field(key_info.get("allowed_ips"))
                    if allowed_ips_raw is None:
                        # Fail closed when an allowlist value is present but
                        # does not decode into a concrete JSON array.
                        raise TypeError("API key allowlist must be stored as JSON array")
                    if not isinstance(allowed_ips_raw, list):
                        raise TypeError("API key allowlist must be stored as JSON array")
                    allowed_ips = [str(ip).strip() for ip in allowed_ips_raw if str(ip).strip()]
                except (TypeError, ValueError, json.JSONDecodeError) as decode_error:
                    if self.settings.PII_REDACT_LOGS:
                        logger.error("API key allowlist could not be decoded; denying access (details redacted)")
                    else:
                        logger.error(
                            f"API key {key_info.get('id')} allowlist could not be decoded; denying access: {decode_error}"
                        )
                    return None
                if allowed_ips:
                    normalized_ip = (ip_address or "").strip()
                    if not normalized_ip:
                        if self.settings.PII_REDACT_LOGS:
                            logger.warning("API key requires client IP but none was supplied; denying access (details redacted)")
                        else:
                            logger.warning(
                                f"API key {key_info.get('id')} requires client IP but none was supplied; denying access"
                            )
                        return None
                    try:
                        ip_obj = ipaddress.ip_address(normalized_ip)
                    except ValueError:
                        if self.settings.PII_REDACT_LOGS:
                            logger.warning("API key client IP is invalid; denying access (details redacted)")
                        else:
                            logger.warning(
                                f"API key {key_info.get('id')} client IP is invalid: {normalized_ip}"
                            )
                        return None
                    matched = False
                    for entry in allowed_ips:
                        token = entry.strip()
                        if not token:
                            continue
                        try:
                            if "/" in token:
                                if ip_obj in ipaddress.ip_network(token, strict=False):
                                    matched = True
                                    break
                            else:
                                if ip_obj == ipaddress.ip_address(token):
                                    matched = True
                                    break
                        except ValueError as parse_exc:
                            logger.debug(
                                "API key allowlist entry invalid; ignoring (entry={}, error={})",
                                token,
                                parse_exc,
                            )
                            continue
                    if not matched:
                        if self.settings.PII_REDACT_LOGS:
                            logger.warning("API key used from unauthorized IP; denying access (details redacted)")
                        else:
                            logger.warning(
                                f"API key {key_info.get('id')} used from unauthorized IP: {normalized_ip}"
                            )
                        return None

            # Use timing-safe comparison to prevent timing attacks (legacy HMAC path only)
            if primary_hash and stored_hash and not is_kdf_hash(stored_hash):
                if not hmac.compare_digest(stored_hash, primary_hash):
                    try:
                        await repo.update_key_hash(key_info["id"], primary_hash)
                        key_info["key_hash"] = primary_hash
                    except Exception as normalize_exc:
                        if self.settings.PII_REDACT_LOGS:
                            logger.warning("Failed to normalize API key hash (details redacted)")
                        else:
                            logger.warning(
                                f"Failed to normalize API key hash for key {key_info.get('id')}: {normalize_exc}"
                            )
            key_info.pop("key_hash", None)

            # Check scope
            if required_scope:
                key_scope = key_info.get("scope")
                if not self._has_scope(key_scope, required_scope):
                    return None

            if record_usage:
                # Update usage statistics
                await self._update_usage(key_info['id'], ip_address)

                # Optional lightweight audit of usage
                try:
                    if self.settings.API_KEY_AUDIT_LOG_USAGE:
                        await self._log_action(
                            key_info['id'],
                            "used",
                            key_info.get('user_id'),
                            details=usage_details,
                        )
                except Exception as _e:
                    # Do not fail request on audit write
                    logger.debug(f"API key usage audit skipped/failed: {_e}")

            return key_info

        except DatabaseError:
            # Surface explicit database failures so callers can respond with
            # a clear server-side error instead of silently denying access.
            raise
        except Exception:  # noqa: BLE001 - validation failures degrade to 'no key'
            logger.opt(exception=True).error(
                "Failed to validate API key (ip={}, scope={})",
                ip_address,
                required_scope,
            )
            return None

    async def rotate_api_key(
        self,
        key_id: int,
        user_id: int,
        expires_in_days: Optional[int] = 90,
        actor_user_id: Optional[int] = None,
        actor_subject: Optional[str] = None,
        actor_kind: Optional[str] = None,
        actor_roles: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Rotate an API key - atomically create new one and revoke old one.

        This operation is atomic: either both the new key creation and old key
        revocation succeed, or neither happens. This prevents the security issue
        where a new key is created but the old key remains active.

        Args:
            key_id: ID of the key to rotate
            user_id: User requesting rotation (for authorization)
            expires_in_days: Expiration for new key

        Returns:
            New key information
        """
        if not self._initialized:
            await self.initialize()

        repo = self._get_repo()
        try:
            async with self.db_pool.transaction() as conn:
                old_key = await repo.fetch_key_for_user(key_id=key_id, user_id=user_id, conn=conn)

                if not old_key:
                    raise ValueError("API key not found or unauthorized")
                if str(old_key.get("status") or "").lower() != APIKeyStatus.ACTIVE.value:
                    raise ValueError("API key not found or unauthorized")

                raw_allowed_ips = old_key.get("allowed_ips")
                allowed_ips: Optional[list[str]] = None
                if raw_allowed_ips:
                    decoded_allowed_ips = self._coerce_json_field(raw_allowed_ips)
                    if decoded_allowed_ips is None or not isinstance(decoded_allowed_ips, list):
                        raise TypeError("API key allowlist must be stored as JSON array")
                    allowed_ips = decoded_allowed_ips  # type: ignore[assignment]

                full_key, key_hash, key_identifier = self.generate_api_key()
                key_prefix = full_key[:10] + "..."

                expires_at = None
                if expires_in_days is not None:
                    expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

                new_name = f"{old_key['name']} (rotated)" if old_key['name'] else "Rotated key"
                new_metadata = self._coerce_json_field(old_key.get('metadata'))

                new_key_id = await repo.rotate_key_atomic(
                    user_id=user_id,
                    old_key_id=key_id,
                    new_key_hash=key_hash,
                    new_key_identifier=key_identifier,
                    new_key_prefix=key_prefix,
                    new_name=new_name,
                    new_description=old_key['description'],
                    new_scope=old_key['scope'],
                    new_expires_at=expires_at,
                    new_rate_limit=old_key['rate_limit'],
                    new_allowed_ips=allowed_ips,
                    new_metadata=new_metadata,
                    active_status=APIKeyStatus.ACTIVE.value,
                    rotated_status=APIKeyStatus.ROTATED.value,
                    reason="Key rotation",
                    revoked_at=datetime.now(timezone.utc),
                    conn=conn,
                )

                await emit_mandatory_api_key_management_audit(
                    user_id=user_id,
                    event_type=AuditEventType.DATA_UPDATE,
                    category=AuditEventCategory.DATA_MODIFICATION,
                    action="api_key.rotate",
                    resource_id=str(new_key_id),
                    metadata={
                        "old_key_id": key_id,
                        "new_key_id": new_key_id,
                        "expires_in_days": expires_in_days,
                    },
                    actor_user_id=actor_user_id,
                    actor_subject=actor_subject,
                    actor_kind=actor_kind,
                    actor_roles=actor_roles,
                )

            # Keep legacy API-key audit mirror as best-effort compatibility.
            await self._log_action(key_id, "rotated", user_id)
            await self._log_action(new_key_id, "created_from_rotation", user_id)

            logger.info(f"Rotated API key {key_id} to {new_key_id}")

            # Return the new key information
            return {
                "id": new_key_id,
                "key": full_key,  # Only shown once
                "key_prefix": key_prefix,
                "name": new_name,
                "description": old_key['description'],
                "scope": old_key['scope'],
                "expires_at": expires_at.isoformat() if expires_at else None,
                "rate_limit": old_key['rate_limit'],
                "allowed_ips": allowed_ips,
                "metadata": new_metadata,
                "rotated_from": key_id,
            }

        except (ValueError, InvalidTokenError):
            # Preserve auth/not-found semantics; callers map to appropriate 4xx.
            raise
        except MandatoryAuditWriteError:
            raise
        except TransactionError as exc:
            message = str(exc)
            if (
                "API key not found or inactive" in message
                or "API key not found or unauthorized" in message
            ):
                raise ValueError("API key not found or unauthorized") from exc
            logger.exception("Failed to rotate API key")
            raise DatabaseError(
                f"Failed to rotate API key {self._db_context_hint()}"
            ) from exc
        except Exception as e:
            logger.exception("Failed to rotate API key")
            raise DatabaseError(
                f"Failed to rotate API key {self._db_context_hint()}"
            ) from e

    async def revoke_api_key(
        self,
        key_id: int,
        user_id: int,
        reason: Optional[str] = None,
        actor_user_id: Optional[int] = None,
        actor_subject: Optional[str] = None,
        actor_kind: Optional[str] = None,
        actor_roles: Optional[list[str]] = None,
    ) -> bool:
        """
        Revoke an API key

        Args:
            key_id: ID of the key to revoke
            user_id: User requesting revocation
            reason: Reason for revocation

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        repo = self._get_repo()
        reason_text = reason or "Manual revocation"
        try:
            async with self.db_pool.transaction() as conn:
                success = await repo.revoke_api_key_for_user(
                    key_id=key_id,
                    user_id=user_id,
                    revoked_status=APIKeyStatus.REVOKED.value,
                    active_status=APIKeyStatus.ACTIVE.value,
                    reason=reason_text,
                    revoked_at=datetime.now(timezone.utc),
                    conn=conn,
                )

                if success:
                    await emit_mandatory_api_key_management_audit(
                        user_id=user_id,
                        event_type=AuditEventType.DATA_UPDATE,
                        category=AuditEventCategory.DATA_MODIFICATION,
                        action="api_key.revoke",
                        resource_id=str(key_id),
                        metadata={"reason": reason_text},
                        actor_user_id=actor_user_id,
                        actor_subject=actor_subject,
                        actor_kind=actor_kind,
                        actor_roles=actor_roles,
                    )

            if success:
                # Keep legacy API-key audit mirror as best-effort compatibility.
                await self._log_action(key_id, "revoked", user_id, {"reason": reason_text})
                logger.info(f"Revoked API key {key_id}")

            return success

        except MandatoryAuditWriteError:
            raise
        except Exception as e:
            logger.exception("Failed to revoke API key")
            raise DatabaseError(
                f"Failed to revoke API key {self._db_context_hint()}"
            ) from e

    async def list_user_keys(
        self,
        user_id: int,
        include_revoked: bool = False
    ) -> list[dict[str, Any]]:
        """
        List all API keys for a user

        Args:
            user_id: User ID
            include_revoked: Include revoked/expired keys

        Returns:
            List of key information (without actual keys)
        """
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()
            results = await repo.list_user_keys(user_id=user_id, include_revoked=include_revoked)

            keys = []
            for row in results:
                key_dict = dict(row)
                # Never return the actual hash
                key_dict.pop('key_hash', None)
                keys.append(key_dict)

            return keys

        except Exception as e:
            logger.exception("Failed to list user keys")
            raise DatabaseError(
                f"Failed to list user keys {self._db_context_hint()}"
            ) from e

    async def cleanup_expired_keys(self) -> None:
        """Mark expired keys as expired"""
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()
            updated = await repo.expire_keys_before(
                now=datetime.now(timezone.utc),
                expired_status=APIKeyStatus.EXPIRED.value,
                active_status=APIKeyStatus.ACTIVE.value,
            )
            logger.debug(f"Cleaned up expired API keys (updated={updated})")
        except Exception:  # noqa: BLE001 - best-effort cleanup must not break requests
            logger.opt(exception=True).error("Failed to cleanup expired keys")

    def _has_scope(self, key_scope: Optional[Union[str, list[str]]], required_scope: str) -> bool:
        """
        Check if key scope satisfies required scope using explicit matching.

        Uses the module-level has_scope() function with normalize_scope() for
        backward compatibility with both string and list scope formats.

        Args:
            key_scope: The API key's scope (string, list, or None)
            required_scope: The scope required for the operation

        Returns:
            True if the key has the required scope or admin/service bypass
        """
        key_scopes = normalize_scope(key_scope)
        return has_scope(key_scopes, required_scope)

    async def _update_usage(self, key_id: int, ip_address: Optional[str] = None) -> None:
        """Update usage statistics for a key"""
        if self._should_skip_usage_touch(key_id):
            return
        try:
            repo = self._get_repo()
            await repo.increment_usage(key_id=key_id, ip_address=ip_address)
        except Exception:  # noqa: BLE001 - usage updates must not break requests
            self._clear_usage_touch(key_id)
            logger.opt(exception=True).warning(
                "Failed to update API key usage (key_id={})",
                key_id,
            )

    async def _mark_expired(self, key_id: int) -> None:
        """Mark a key as expired"""
        try:
            repo = self._get_repo()
            await repo.mark_key_expired(key_id=key_id, expired_status=APIKeyStatus.EXPIRED.value)
        except Exception:  # noqa: BLE001 - expiration updates must not break requests
            logger.opt(exception=True).warning(
                "Failed to mark API key as expired (key_id={})",
                key_id,
            )

    async def _log_action(
        self,
        key_id: int,
        action: str,
        user_id: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        """Log an action in the audit log"""
        try:
            repo = self._get_repo()
            await repo.insert_audit_log(
                key_id=key_id,
                action=action,
                user_id=user_id,
                details=details,
            )
        except Exception:  # noqa: BLE001 - audit logging should not block requests
            logger.opt(exception=True).warning(
                "Failed to write API key audit log (key_id={}, action={})",
                key_id,
                action,
            )


#######################################################################################################################
#
# Module Functions
#

# Global instance
_api_key_manager: Optional[APIKeyManager] = None
_api_key_manager_lock_guard = threading.Lock()
_api_key_manager_locks: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = WeakKeyDictionary()


def _get_api_key_manager_lock() -> asyncio.Lock:
    """Return a per-event-loop asyncio.Lock to avoid cross-loop binding issues."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
    if loop is None:
        return asyncio.Lock()
    with _api_key_manager_lock_guard:
        lock = _api_key_manager_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _api_key_manager_locks[loop] = lock
        return lock

async def get_api_key_manager() -> APIKeyManager:
    """Get APIKeyManager singleton instance"""
    global _api_key_manager
    # If an instance exists but the HMAC key material has changed (env/settings), recreate it
    try:
        current_settings = get_settings()
        current_fp = _compute_hmac_fingerprint(current_settings)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to compute HMAC fingerprint; will recreate manager: {}", exc)
        current_fp = ""

    async with _get_api_key_manager_lock():
        if _api_key_manager is not None:
            try:
                if getattr(_api_key_manager, "_hmac_key_fingerprint", None) != current_fp:
                    _api_key_manager = None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to check manager fingerprint; will recreate manager: {}", exc)
                _api_key_manager = None

        if not _api_key_manager:
            manager = APIKeyManager()
            await manager.initialize()
            _api_key_manager = manager
        return _api_key_manager


async def reset_api_key_manager() -> None:
    """Reset the APIKeyManager singleton (mainly for testing)."""
    global _api_key_manager
    async with _get_api_key_manager_lock():
        _api_key_manager = None

#
# End of api_key_manager.py
#######################################################################################################################
