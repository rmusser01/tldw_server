# token_blacklist.py
# Description: Token blacklist service for JWT revocation and invalidation
#
# Imports
import asyncio
import contextlib
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger

#
# 3rd-party imports
from redis import asyncio as redis_async
from redis.exceptions import RedisError

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.repos.token_blacklist_repo import (
    AuthnzTokenBlacklistRepo,
)

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings

#######################################################################################################################
#
# Token Blacklist Service
#

class TokenBlacklist:
    """
    Service for managing revoked/blacklisted JWT tokens.

    Supports both Redis (for performance) and database (for persistence) storage.
    Automatically cleans up expired tokens to prevent unbounded growth.
    """

    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize token blacklist service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self._external_db_pool = db_pool is not None
        self.redis_client: Optional[redis_async.Redis] = None
        self._initialized = False

        # In-memory LRU cache of recently seen blacklisted JTIs mapped to expiry
        self._local_cache: dict[str, datetime] = {}
        self._local_order: deque[str] = deque()
        self._cache_size_limit = 1000
        self._ensured_session_columns = False

    def _cache_remove(self, jti: str) -> None:
        """Remove a JTI from the local cache if present."""
        if jti in self._local_cache:
            self._local_cache.pop(jti, None)
            with contextlib.suppress(ValueError):
                self._local_order.remove(jti)

    @staticmethod
    def _normalize_expiry(expires_at: Optional[Any]) -> Optional[datetime]:
        """Normalize stored expiry into a naive UTC datetime when possible."""
        if expires_at is None:
            return None
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is not None:
                return expires_at.astimezone(timezone.utc).replace(tzinfo=None)
            return expires_at
        if isinstance(expires_at, str):
            try:
                parsed = datetime.fromisoformat(expires_at)
            except ValueError:
                return None
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        return None

    def _normalize_expiry_for_storage(self, expires_at: Any) -> datetime:
        """Prepare expiry timestamp for persistence (UTC naive)."""
        normalized = self._normalize_expiry(expires_at)
        if normalized is None:
            raise ValueError("expires_at must be a valid datetime or ISO formatted string")
        return normalized

    def _cache_add(self, jti: str, expires_at: Optional[Any]) -> None:
        """Add a JTI to local LRU cache, respecting expiry."""
        if not jti:
            return
        expiry = self._normalize_expiry(expires_at)
        if expiry is None:
            # Unknown expiry, avoid caching indefinitely
            self._cache_remove(jti)
            return
        if expiry <= datetime.now(timezone.utc).replace(tzinfo=None):
            self._cache_remove(jti)
            return
        # Refresh ordering
        if jti in self._local_cache:
            self._local_cache[jti] = expiry
            with contextlib.suppress(ValueError):
                self._local_order.remove(jti)
            self._local_order.append(jti)
        else:
            self._local_cache[jti] = expiry
            self._local_order.append(jti)
        # Evict expired entries and enforce size limit
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        while self._local_order:
            oldest = self._local_order[0]
            cached_expiry = self._local_cache.get(oldest)
            if cached_expiry and cached_expiry <= now:
                self._local_order.popleft()
                self._local_cache.pop(oldest, None)
                continue
            if len(self._local_cache) > self._cache_size_limit:
                self._local_order.popleft()
                self._local_cache.pop(oldest, None)
                continue
            break

    def hint_blacklisted(self, jti: str, expires_at: Optional[datetime]) -> None:
        """
        Optimistically mark a token as blacklisted in the local cache.

        This gives synchronous helpers a fast fail path while asynchronous
        persistence (database, Redis) catches up.
        """
        normalized_expiry = self._normalize_expiry_for_storage(expires_at)
        self._cache_add(jti, normalized_expiry)

    async def initialize(self):
        """Initialize blacklist service and create tables if needed"""
        if self._initialized:
            return

        # Get database pool
        self.db_pool = await self._ensure_db_pool()

        # Create blacklist table if it doesn't exist
        await self._create_tables()

        # Initialize Redis if configured
        if self.settings.REDIS_URL:
            try:
                self.redis_client = redis_async.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1
                )
                await self.redis_client.ping()
                logger.debug("Redis connected for token blacklist")
            except (OSError, RedisError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Redis unavailable for token blacklist: {e}")
                self.redis_client = None

        self._initialized = True
        logger.info("TokenBlacklist service initialized")

    async def _ensure_db_pool(self) -> DatabasePool:
        """Ensure the blacklist has a usable database pool for the active loop."""
        current_settings = get_settings()

        if not self._external_db_pool:
            global_pool = await get_db_pool()
            if self.db_pool is not global_pool:
                logger.debug("TokenBlacklist adopting refreshed AuthNZ DatabasePool instance")
                self.db_pool = global_pool
            # Adopt global settings when we manage the pool ourselves
            self.settings = current_settings
        else:
            # External pools rely on caller to manage lifecycle; preserve provided settings
            # to ensure consistent behavior within the caller's configured mode.
            if self.settings is None:
                self.settings = current_settings

        if not self.db_pool:
            self.db_pool = await get_db_pool()
            return self.db_pool

        pool_ref = getattr(self.db_pool, "pool", None)
        pool_closed = bool(pool_ref) and getattr(pool_ref, "closed", False)

        loop_changed = False
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        stored_loop = getattr(self.db_pool, "_loop", None)
        if pool_ref and stored_loop and current_loop and stored_loop is not current_loop:
            loop_changed = True

        if pool_closed or loop_changed:
            logger.debug(
                "TokenBlacklist refreshing database pool "
                f"(pool_closed={pool_closed}, loop_changed={loop_changed})"
            )
            if not self._external_db_pool:
                await reset_db_pool()
                self.db_pool = await get_db_pool()
                return self.db_pool
            await self.db_pool.close()
            await self.db_pool.initialize()
            return self.db_pool

        if not getattr(self.db_pool, "_initialized", False):
            await self.db_pool.initialize()

        return self.db_pool

    async def _create_tables(self):
        """Create token blacklist table if it doesn't exist"""
        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzTokenBlacklistRepo(db_pool)
            await repo.ensure_schema()

        except Exception as e:
            logger.error(f"Failed to create token blacklist table: {e}")
            raise DatabaseError(f"Failed to create blacklist table: {e}") from e

    async def _ensure_session_revocation_columns(self, conn) -> None:
        """Ensure legacy SQLite session tables include revocation columns."""
        if self._ensured_session_columns:
            return
        try:
            cursor = await conn.execute("PRAGMA table_info(sessions)")
            rows = await cursor.fetchall()

            # If the sessions table does not exist yet, defer harmonization so a later
            # call (after the table is created) can retry instead of caching failure.
            if not rows:
                logger.debug("TokenBlacklist: sessions table missing; deferring revocation column harmonization")
                return

            columns = {row[1] for row in rows}
            alterations = [
                ("is_active", "is_active INTEGER DEFAULT 1"),
                ("is_revoked", "is_revoked INTEGER DEFAULT 0"),
                ("revoked_at", "revoked_at TIMESTAMP"),
                ("revoked_by", "revoked_by INTEGER"),
                ("revoke_reason", "revoke_reason TEXT"),
            ]
            altered = False
            for name, decl in alterations:
                if name not in columns:
                    await conn.execute(f"ALTER TABLE sessions ADD COLUMN {decl}")
                    altered = True
            if altered:
                await conn.commit()
            self._ensured_session_columns = True
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(f"TokenBlacklist: unable to harmonize session columns: {exc}")

    async def revoke_token(
        self,
        jti: str,
        expires_at: datetime,
        user_id: Optional[int] = None,
        token_type: str = "access",
        reason: Optional[str] = None,
        revoked_by: Optional[int] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Add a token to the blacklist

        Args:
            jti: JWT ID (unique identifier for the token)
            expires_at: Token expiration time
            user_id: User who owns the token
            token_type: Type of token (access, refresh, etc.)
            reason: Reason for revocation
            revoked_by: User who revoked the token (for admin actions)
            ip_address: IP address of revocation request

        Returns:
            True if successfully blacklisted
        """
        if not self._initialized:
            await self.initialize()

        normalized_expiry = self._normalize_expiry_for_storage(expires_at)

        # Add to local cache (LRU) with a small grace buffer to account for
        # initialization/IO latency so that immediate post-revocation checks
        # reliably observe the blacklisted state even for very short expiries.
        # This is conservative (tokens remain blacklisted slightly longer).
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        min_grace = timedelta(seconds=1)
        effective_cache_expiry = (
            now_utc + min_grace if normalized_expiry <= now_utc + min_grace else normalized_expiry
        )
        self._cache_add(jti, effective_cache_expiry)

        # Add to Redis if available
        if self.redis_client:
            try:
                key = f"blacklist:{jti}"
                ttl = int((normalized_expiry - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds())

                if ttl > 0:
                    await self.redis_client.setex(
                        key,
                        ttl,
                        json.dumps({
                            "user_id": user_id,
                            "token_type": token_type,
                            "reason": reason,
                            "revoked_at": datetime.now(timezone.utc).isoformat()
                        })
                    )
                    if self.settings.PII_REDACT_LOGS:
                        logger.debug("Token added to Redis blacklist (details redacted)")
                    else:
                        logger.debug(f"Token {jti} added to Redis blacklist")

            except (OSError, RedisError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to add token to Redis blacklist: {e}")

        # Add to database for persistence
        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzTokenBlacklistRepo(db_pool)
            await repo.insert_blacklisted_token(
                jti=jti,
                user_id=user_id,
                token_type=token_type,
                expires_at=normalized_expiry,
                reason=reason,
                revoked_by=revoked_by,
                ip_address=ip_address,
            )

            if self.settings.PII_REDACT_LOGS:
                logger.info(
                    "Token blacklisted for authenticated user (details redacted) - Reason: {}",
                    reason,
                )
            else:
                logger.info(
                    "Token {} blacklisted for user {} - Reason: {}",
                    jti,
                    user_id,
                    reason,
                )
            return True

        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False

    async def is_blacklisted(self, jti: str) -> bool:
        """
        Check if a token is blacklisted

        Args:
            jti: JWT ID to check

        Returns:
            True if token is blacklisted
        """
        if not jti:
            return False

        # Check local cache first (fastest)
        cached_expiry = self._local_cache.get(jti)
        if cached_expiry:
            if cached_expiry > datetime.now(timezone.utc).replace(tzinfo=None):
                return True
            self._cache_remove(jti)

        if not self._initialized:
            await self.initialize()

        # Check Redis if available
        if self.redis_client:
            try:
                key = f"blacklist:{jti}"
                exists = await self.redis_client.exists(key)
                if exists:
                    ttl = await self.redis_client.ttl(key)
                    expiry = None
                    if isinstance(ttl, (int, float)) and ttl > 0:
                        expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=int(ttl))
                    # Add to local cache for next time if expiry known
                    self._cache_add(jti, expiry)
                    return True
            except (OSError, RedisError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Redis error checking blacklist: {e}")

        # Check database
        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzTokenBlacklistRepo(db_pool)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            expires_at = await repo.get_active_expiry_for_jti(jti=jti, now=now)

            if expires_at:
                self._cache_add(jti, expires_at)
                return True

            self._cache_remove(jti)

        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Database error checking blacklist: {e}")
            # Fail closed - treat as blacklisted on error
            return True

        return False

    async def revoke_all_user_tokens(
        self,
        user_id: int,
        reason: str = "User requested logout from all devices",
        revoked_by: Optional[int] = None,
        ip_address: Optional[str] = None,
        except_session_id: Optional[int] = None,
    ) -> int:
        """
        Revoke all tokens for a specific user

        Args:
            user_id: User whose tokens to revoke
            reason: Reason for revocation
            revoked_by: User who initiated revocation
            ip_address: IP address of request
            except_session_id: Session that must remain active

        Returns:
            Number of tokens revoked
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzSessionsRepo(db_pool)

            # For SQLite-based pools, ensure legacy session tables have the
            # revocation columns before we attempt to update them.
            if getattr(db_pool, "pool", None) is None:
                async with db_pool.acquire() as conn:
                    await self._ensure_session_revocation_columns(conn)

            # Snapshot the sessions' token metadata for blacklist use
            sessions = await repo.fetch_session_token_metadata_for_user(
                user_id,
                except_session_id=except_session_id,
            )
            # Mark sessions as revoked with audit metadata
            await repo.mark_sessions_revoked_for_user_with_audit(
                user_id=user_id,
                revoked_by=revoked_by,
                reason=reason,
                except_session_id=except_session_id,
            )

            def _to_datetime(value: Optional[Any]) -> Optional[datetime]:
                if value is None:
                    return None
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value)
                    except ValueError:
                        return None
                return None

            # Blacklist stored JTIs without needing token decryption
            tokens_revoked = 0
            sessions_count = len(sessions)
            for session in sessions:
                access_jti = session.get("access_jti")
                refresh_jti = session.get("refresh_jti")
                access_exp = _to_datetime(session.get("expires_at"))
                refresh_exp = _to_datetime(session.get("refresh_expires_at"))

                if access_jti and access_exp and await self.revoke_token(
                    jti=access_jti,
                    expires_at=access_exp,
                    user_id=user_id,
                    token_type="access",
                    reason=reason,
                    revoked_by=revoked_by,
                    ip_address=ip_address,
                ):
                    tokens_revoked += 1
                if refresh_jti and refresh_exp and await self.revoke_token(
                    jti=refresh_jti,
                    expires_at=refresh_exp,
                    user_id=user_id,
                    token_type="refresh",
                    reason=reason,
                    revoked_by=revoked_by,
                    ip_address=ip_address,
                ):
                    tokens_revoked += 1

            if self.settings.PII_REDACT_LOGS:
                logger.info(
                    f"Revoked {tokens_revoked} token(s) across {len(sessions)} session(s) for authenticated user (details redacted)"
                )
            else:
                logger.info(
                    f"Revoked {tokens_revoked} token(s) across {len(sessions)} session(s) for user {user_id}"
                )
            # Return semantics:
            # - single_user: number of sessions affected (devices) for clearer UX
            # - multi_user: number of tokens revoked to preserve existing unit-test expectations
            try:
                mode = getattr(self.settings, "AUTH_MODE", "single_user")
            except (AttributeError, TypeError):
                mode = "single_user"
            if mode == "single_user":
                return sessions_count
            return tokens_revoked

        except (DatabaseError, OSError, RedisError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to revoke user tokens: {e}")
            return 0

    async def cleanup_expired(self) -> int:
        """
        Remove expired tokens from the blacklist

        Returns:
            Number of tokens removed
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzTokenBlacklistRepo(db_pool)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            count = await repo.cleanup_expired(now=now)

            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens from blacklist")

            # Clear local cache periodically if it grew too large (soft reset)
            if len(self._local_cache) > self._cache_size_limit * 2:
                self._local_cache.clear()
                self._local_order.clear()

            return count

        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0

    async def get_blacklist_stats(self, user_id: Optional[int] = None) -> dict[str, Any]:
        """
        Get statistics about blacklisted tokens

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with blacklist statistics
        """
        if not self._initialized:
            await self.initialize()

        try:
            db_pool = await self._ensure_db_pool()
            repo = AuthnzTokenBlacklistRepo(db_pool)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            return await repo.get_blacklist_stats(now=now, user_id=user_id)
        except (DatabaseError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to get blacklist stats: {e}")

        return {
            "total": 0,
            "unique_users": 0,
            "access_tokens": 0,
            "refresh_tokens": 0,
        }


#######################################################################################################################
#
# Module Functions for convenience
#

# Global instance
_token_blacklist: Optional[TokenBlacklist] = None


def get_token_blacklist() -> TokenBlacklist:
    """Get token blacklist singleton instance"""
    global _token_blacklist
    if not _token_blacklist:
        _token_blacklist = TokenBlacklist()
    return _token_blacklist


async def revoke_token(
    jti: str,
    expires_at: datetime,
    user_id: Optional[int] = None,
    reason: Optional[str] = None
) -> bool:
    """Convenience function to revoke a token"""
    blacklist = get_token_blacklist()
    return await blacklist.revoke_token(jti, expires_at, user_id, reason=reason)


async def is_token_blacklisted(jti: str) -> bool:
    """Convenience function to check if token is blacklisted"""
    blacklist = get_token_blacklist()
    return await blacklist.is_blacklisted(jti)


async def revoke_all_user_tokens(
    user_id: int,
    reason: str = "User logout",
    except_session_id: Optional[int] = None,
) -> int:
    """Convenience function to revoke all user tokens"""
    blacklist = get_token_blacklist()
    return await blacklist.revoke_all_user_tokens(
        user_id,
        reason,
        except_session_id=except_session_id,
    )


async def reset_token_blacklist():
    """Reset token blacklist singleton (primarily for testing)."""
    global _token_blacklist
    if _token_blacklist:
        try:
            if _token_blacklist.redis_client:
                await _token_blacklist.redis_client.close()
        except (OSError, RedisError, RuntimeError, TypeError, ValueError) as e:
            logger.debug(f"TokenBlacklist reset ignored Redis shutdown error: {e}")
        finally:
            _token_blacklist = None


#
# End of token_blacklist.py
#######################################################################################################################
