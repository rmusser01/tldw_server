# storage_quota_service.py
# Description: User storage quota management service with async operations
#
# Imports
import asyncio
import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

#
# 3rd-party imports
from cachetools import TTLCache
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError, StorageError, UserNotFoundError
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    FILE_CATEGORY_VOICE_CLONE,
    AuthnzGeneratedFilesRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import (
    AuthnzStorageQuotasRepo,
)

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Metrics import get_metrics_registry

#######################################################################################################################
#
# Storage Quota Service Class

class StorageQuotaService:
    """Service for managing user storage quotas and tracking usage"""

    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize storage quota service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool

        # TTL cache for quota checks (5 minutes)
        self.quota_cache = TTLCache(maxsize=1000, ttl=300)

        # TTL cache for storage calculations (10 minutes)
        self.storage_cache = TTLCache(maxsize=1000, ttl=600)

        self._initialized = False

    async def initialize(self):
        """Initialize the storage quota service"""
        if self._initialized:
            return

        if not self.db_pool:
            self.db_pool = await get_db_pool()

        self._initialized = True
        logger.info("StorageQuotaService initialized")

    def invalidate_user_cache(self, user_id: int) -> None:
        """Invalidate in-memory quota/storage cache entries for a user."""
        self.quota_cache.pop(f"quota:{user_id}", None)
        self.storage_cache.pop(f"storage_calc:{user_id}", None)

    def _is_postgres_backend(self) -> bool:
        """
        Return True when the underlying DatabasePool is using PostgreSQL.

        Backend detection should key off DatabasePool state rather than probing
        connection method presence, which can be misleading with shimmed
        connections.
        """
        if not self.db_pool:
            return False
        if getattr(self.db_pool, "pool", None):
            return True
        backend = getattr(self.db_pool, "backend", None)
        if isinstance(backend, str):
            return backend.strip().lower() in {"postgres", "postgresql", "pg"}
        return False

    async def calculate_user_storage(
        self,
        user_id: int,
        update_database: bool = True
    ) -> dict[str, Any]:
        """
        Calculate actual storage usage for a user and optionally persist it.
        """
        if not self._initialized:
            await self.initialize()

        # Check cache first
        cache_key = f"storage_calc:{user_id}"
        if cache_key in self.storage_cache and not update_database:
            return self.storage_cache[cache_key]

        # Calculate storage in thread pool
        user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
        size_bytes = await asyncio.to_thread(
            self._calculate_directory_size,
            str(user_dir)
        )

        # Also calculate ChromaDB if configured
        chroma_bytes = 0
        if self.settings.CHROMADB_BASE_PATH:
            chroma_dir = Path(self.settings.CHROMADB_BASE_PATH) / str(user_id)
            chroma_bytes = await asyncio.to_thread(
                self._calculate_directory_size,
                str(chroma_dir)
            )

        total_bytes = size_bytes + chroma_bytes
        total_mb = total_bytes / (1024 * 1024)

        # Get quota from database
        user_info = await self._get_user_storage_info(user_id)
        if not user_info:
            raise UserNotFoundError(f"User {user_id}")
        quota_mb = user_info['storage_quota_mb']

        # Update database if requested
        if update_database:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    # PostgreSQL
                    await conn.execute(
                        "UPDATE users SET storage_used_mb = $1 WHERE id = $2",
                        total_mb, user_id
                    )
                else:
                    # SQLite
                    await conn.execute(
                        "UPDATE users SET storage_used_mb = ? WHERE id = ?",
                        (total_mb, user_id)
                    )
                    await conn.commit()

            # Invalidate quota cache
            self.quota_cache.pop(f"quota:{user_id}", None)
            logger.info(
                f"Recalculated storage for user {user_id}: {total_mb:.2f}MB / {quota_mb}MB"
            )

        result = {
            "user_id": user_id,
            "user_data_bytes": size_bytes,
            "user_data_mb": round(size_bytes / (1024 * 1024), 2),
            "chromadb_bytes": chroma_bytes,
            "chromadb_mb": round(chroma_bytes / (1024 * 1024), 2),
            "total_bytes": total_bytes,
            "total_mb": round(total_mb, 2),
            "quota_mb": quota_mb,
            "available_mb": round(max(0, quota_mb - total_mb), 2),
            "usage_percentage": round((total_mb / quota_mb * 100) if quota_mb > 0 else 0, 1),
            "calculated_at": datetime.utcnow().isoformat()
        }

        # Update cache
        self.storage_cache[cache_key] = result
        return result

    async def check_quota(
        self,
        user_id: int,
        new_bytes: int,
        raise_on_exceed: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if user has quota for new content

        Args:
            user_id: User's database ID
            new_bytes: Size of new content in bytes
            raise_on_exceed: Raise exception if quota exceeded

        Returns:
            Tuple of (has_quota, quota_info)

        Raises:
            QuotaExceededError: If quota exceeded and raise_on_exceed is True
        """
        if not self._initialized:
            await self.initialize()

        # Check cache first
        cache_key = f"quota:{user_id}"
        if cache_key in self.quota_cache:
            current_mb, quota_mb = self.quota_cache[cache_key]
        else:
            # Fetch from database
            user_info = await self._get_user_storage_info(user_id)
            if not user_info:
                raise UserNotFoundError(f"User {user_id}")

            current_mb = float(user_info['storage_used_mb'])
            quota_mb = user_info['storage_quota_mb']

            # Update cache
            self.quota_cache[cache_key] = (current_mb, quota_mb)

        # Emit gauges for current values
        try:
            reg = get_metrics_registry()
            reg.set_gauge("user_storage_used_mb", float(current_mb), labels={"user_id": str(user_id)})
            reg.set_gauge("user_storage_quota_mb", float(quota_mb), labels={"user_id": str(user_id)})
        except Exception as e:
            logger.debug(f"storage_quota: failed to record gauges for user {user_id}: {e}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "storage_quota", "event": "metrics_record_failed"},
                )
            except Exception:
                logger.debug("metrics increment failed for storage_quota metrics_record_failed")

        # Calculate new usage
        new_mb = new_bytes / (1024 * 1024)
        projected_mb = current_mb + new_mb
        has_quota = projected_mb <= quota_mb

        quota_info = {
            "user_id": user_id,
            "current_usage_mb": round(current_mb, 2),
            "quota_mb": quota_mb,
            "new_size_mb": round(new_mb, 2),
            "projected_usage_mb": round(projected_mb, 2),
            "available_mb": round(max(0, quota_mb - current_mb), 2),
            "usage_percentage": round((current_mb / quota_mb * 100) if quota_mb > 0 else 0, 1),
            "has_quota": has_quota
        }

        if not has_quota and raise_on_exceed:
            raise QuotaExceededError(projected_mb, quota_mb)

        return has_quota, quota_info

    async def update_usage(
        self,
        user_id: int,
        bytes_delta: int,
        operation: str = "add"
    ) -> dict[str, Any]:
        """
        Update storage usage for a user

        Args:
            user_id: User's database ID
            bytes_delta: Bytes to add or remove
            operation: 'add' or 'remove'

        Returns:
            Updated storage information
        """
        if not self._initialized:
            await self.initialize()

        mb_delta = bytes_delta / (1024 * 1024)
        if operation == "remove":
            mb_delta = -mb_delta

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    # PostgreSQL
                    result = await conn.fetchrow(
                        """
                        UPDATE users
                        SET storage_used_mb = GREATEST(0, storage_used_mb + $1)
                        WHERE id = $2
                        RETURNING storage_used_mb, storage_quota_mb
                        """,
                        mb_delta, user_id
                    )

                    if not result:
                        raise UserNotFoundError(f"User {user_id}")

                    new_usage = float(result['storage_used_mb'])
                    quota = result['storage_quota_mb']

                else:
                    # SQLite
                    await conn.execute(
                        """
                        UPDATE users
                        SET storage_used_mb = MAX(0, storage_used_mb + ?)
                        WHERE id = ?
                        """,
                        (mb_delta, user_id)
                    )

                    cursor = await conn.execute(
                        "SELECT storage_used_mb, storage_quota_mb FROM users WHERE id = ?",
                        (user_id,)
                    )
                    result = await cursor.fetchone()

                    if not result:
                        raise UserNotFoundError(f"User {user_id}")

                    new_usage = float(result[0])
                    quota = result[1]

                    await conn.commit()

                # Check if over quota
                if new_usage > quota:
                    logger.warning(
                        f"User {user_id} exceeded quota: {new_usage:.2f}MB / {quota}MB"
                    )

                # Invalidate cache
                cache_key = f"quota:{user_id}"
                self.quota_cache.pop(cache_key, None)

                # Log significant changes
                if abs(mb_delta) > 10:  # Log changes over 10MB
                    logger.info(
                        f"Updated storage for user {user_id}: "
                        f"{operation} {abs(mb_delta):.2f}MB "
                        f"(new total: {new_usage:.2f}MB / {quota}MB)"
                    )

                # Update gauges
                try:
                    reg = get_metrics_registry()
                    reg.set_gauge("user_storage_used_mb", float(new_usage), labels={"user_id": str(user_id)})
                    reg.set_gauge("user_storage_quota_mb", float(quota), labels={"user_id": str(user_id)})
                except Exception as e:
                    logger.debug(f"storage_quota: failed to record updated gauges for user {user_id}: {e}")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "storage_quota", "event": "metrics_record_failed"},
                        )
                    except Exception:
                        logger.debug("metrics increment failed for storage_quota metrics_record_failed")

                return {
                    "user_id": user_id,
                    "storage_used_mb": round(new_usage, 2),
                    "storage_quota_mb": quota,
                    "available_mb": round(max(0, quota - new_usage), 2),
                    "usage_percentage": round((new_usage / quota * 100) if quota > 0 else 0, 1)
                }

        except Exception as e:
            logger.error(f"Failed to update storage usage: {e}")
            raise StorageError(f"Failed to update storage usage: {e}") from e


    # ---- Filesystem and reporting helpers ----
    def _calculate_directory_size(self, path: str) -> int:
        """Calculate directory size (runs in thread pool)."""
        total = 0
        path_obj = Path(path)
        if not path_obj.exists():
            return 0
        try:
            for entry in path_obj.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as e:
            logger.error(f"Error calculating size for {path}: {e}")
        return total

    async def get_storage_breakdown(self, user_id: int) -> dict[str, Any]:
        """Get detailed storage breakdown by file type for a user."""
        if not self._initialized:
            await self.initialize()
        user_dir = Path(self.settings.USER_DATA_BASE_PATH) / str(user_id)
        breakdown = await asyncio.to_thread(
            self._calculate_storage_breakdown,
            str(user_dir)
        )
        user_info = await self._get_user_storage_info(user_id)
        if not user_info:
            raise UserNotFoundError(f"User {user_id}")
        breakdown.update({
            "user_id": user_id,
            "quota_mb": user_info['storage_quota_mb'],
            "current_usage_mb": float(user_info['storage_used_mb'])
        })
        return breakdown

    def _calculate_storage_breakdown(self, path: str) -> dict[str, Any]:
        """Calculate storage breakdown by top-level category under the user dir."""
        breakdown = {
            "media": {"count": 0, "bytes": 0},
            "notes": {"count": 0, "bytes": 0},
            "embeddings": {"count": 0, "bytes": 0},
            "exports": {"count": 0, "bytes": 0},
            "temp": {"count": 0, "bytes": 0},
            "other": {"count": 0, "bytes": 0}
        }
        path_obj = Path(path)
        if not path_obj.exists():
            return breakdown
        try:
            for entry in path_obj.rglob('*'):
                if entry.is_file():
                    try:
                        size = entry.stat().st_size
                        relative_path = entry.relative_to(path_obj)
                        parts = relative_path.parts
                        category = parts[0] if parts and parts[0] in breakdown else "other"
                        breakdown[category]["count"] += 1
                        breakdown[category]["bytes"] += size
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as e:
            logger.error(f"Error calculating breakdown for {path}: {e}")
        for category in breakdown:
            breakdown[category]["mb"] = round(breakdown[category]["bytes"] / (1024 * 1024), 2)
        return breakdown

    async def set_user_quota(self, user_id: int, quota_mb: int) -> dict[str, Any]:
        """Set storage quota for a user (min 100MB)."""
        if not self._initialized:
            await self.initialize()
        if quota_mb < 100:
            quota_mb = 100
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres_backend():
                    result = await conn.fetchrow(
                        """
                        UPDATE users
                        SET storage_quota_mb = $1
                        WHERE id = $2
                        RETURNING storage_used_mb, storage_quota_mb
                        """,
                        quota_mb, user_id
                    )
                    if not result:
                        raise UserNotFoundError(f"User {user_id}")
                    current_usage = float(result['storage_used_mb'])
                else:
                    await conn.execute(
                        "UPDATE users SET storage_quota_mb = ? WHERE id = ?",
                        (quota_mb, user_id)
                    )
                    cursor = await conn.execute(
                        "SELECT storage_used_mb FROM users WHERE id = ?",
                        (user_id,)
                    )
                    row = await cursor.fetchone()
                    if not row:
                        raise UserNotFoundError(f"User {user_id}")
                    current_usage = float(row[0])
                    await conn.commit()
            self.quota_cache.pop(f"quota:{user_id}", None)
            logger.info(f"Set quota for user {user_id}: {quota_mb}MB")
            return {
                "user_id": user_id,
                "storage_quota_mb": quota_mb,
                "storage_used_mb": round(current_usage, 2),
                "available_mb": round(max(0, quota_mb - current_usage), 2),
                "usage_percentage": round((current_usage / quota_mb * 100) if quota_mb > 0 else 0, 1)
            }
        except Exception as e:
            logger.error(f"Failed to set user quota: {e}")
            raise StorageError(f"Failed to set quota: {e}") from e

    async def get_all_users_storage(self) -> list[dict[str, Any]]:
        """List storage usage for all active users, sorted by usage desc."""
        if not self._initialized:
            await self.initialize()

        async with self.db_pool.acquire() as conn:
            if self._is_postgres_backend():
                # PostgreSQL
                users = await conn.fetch(
                    """
                    SELECT id, username, storage_used_mb, storage_quota_mb
                    FROM users
                    WHERE is_active = TRUE
                    ORDER BY storage_used_mb DESC
                    """
                )
            else:
                # SQLite
                cursor = await conn.execute(
                    """
                    SELECT id, username, storage_used_mb, storage_quota_mb
                    FROM users
                    WHERE is_active = 1
                    ORDER BY storage_used_mb DESC
                    """
                )
                rows = await cursor.fetchall()
                cols = [desc[0] for desc in cursor.description] if cursor.description else []
                users = [dict(zip(cols, row)) for row in rows]

        result = []
        for user in users:
            used = float(user.get('storage_used_mb', 0) or 0)
            quota = user.get('storage_quota_mb', 0) or 0
            result.append({
                "user_id": user.get('id'),
                "username": user.get('username'),
                "storage_used_mb": round(used, 2),
                "storage_quota_mb": quota,
                "available_mb": round(max(0, quota - used), 2),
                "usage_percentage": round((used / quota * 100) if quota > 0 else 0, 1)
            })
        return result

    async def cleanup_temp_files(self, user_id: Optional[int] = None, older_than_hours: int = 24) -> dict[str, Any]:
        """Delete temp files older than the threshold for one or all users."""
        if not self._initialized:
            await self.initialize()
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        if user_id:
            temp_dirs = [Path(self.settings.USER_DATA_BASE_PATH) / str(user_id) / "temp"]
        else:
            base_path = Path(self.settings.USER_DATA_BASE_PATH)
            temp_dirs = list(base_path.glob("*/temp"))
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,  # Use default executor (self.executor was undefined)
            self._cleanup_temp_directories,
            temp_dirs,
            cutoff_time
        )
        if stats['files_deleted'] > 0:
            logger.info(
                f"Cleaned up {stats['files_deleted']} temp files ({stats['bytes_freed'] / (1024*1024):.2f}MB)"
            )
        return stats

    def _cleanup_temp_directories(self, temp_dirs: list[Path], cutoff_time: datetime) -> dict[str, Any]:
        """Filesystem worker to cleanup temp directories."""
        stats = {"files_deleted": 0, "bytes_freed": 0, "errors": 0}
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            try:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if mtime < cutoff_time:
                                size = file_path.stat().st_size
                                file_path.unlink()
                                stats['files_deleted'] += 1
                                stats['bytes_freed'] += size
                        except (OSError, PermissionError):
                            stats['errors'] += 1
                            continue
            except (OSError, PermissionError):
                stats['errors'] += 1
                continue
        return stats

    async def _get_user_storage_info(self, user_id: int) -> Optional[dict[str, Any]]:
        """Get user storage info from the database."""
        return await self.db_pool.fetchone(
            "SELECT storage_used_mb, storage_quota_mb FROM users WHERE id = ?",
            user_id
        )

    async def shutdown(self):
        """Shutdown hook retained for compatibility."""
        self._initialized = False
        self.quota_cache.clear()
        self.storage_cache.clear()
        logger.info("StorageQuotaService shutdown complete (no dedicated executor)")

    # =========================================================================
    # Team/Org Quota Management (using storage_quotas table)
    # =========================================================================

    async def get_storage_quotas_repo(self) -> AuthnzStorageQuotasRepo:
        """Get the storage quotas repository."""
        if not self._initialized:
            await self.initialize()
        return AuthnzStorageQuotasRepo(db_pool=self.db_pool)

    async def get_generated_files_repo(self) -> AuthnzGeneratedFilesRepo:
        """Get the generated files repository."""
        if not self._initialized:
            await self.initialize()
        return AuthnzGeneratedFilesRepo(db_pool=self.db_pool)

    async def get_org_quota(self, org_id: int) -> dict[str, Any]:
        """Get storage quota status for an organization."""
        repo = await self.get_storage_quotas_repo()
        status = await repo.check_quota_status(org_id=org_id)
        status["org_id"] = org_id
        return status

    async def get_team_quota(self, team_id: int) -> dict[str, Any]:
        """Get storage quota status for a team."""
        repo = await self.get_storage_quotas_repo()
        status = await repo.check_quota_status(team_id=team_id)
        status["team_id"] = team_id
        return status

    async def set_org_quota(
        self,
        org_id: int,
        quota_mb: int,
        soft_limit_pct: int = 80,
        hard_limit_pct: int = 100,
    ) -> dict[str, Any]:
        """Set storage quota for an organization."""
        if not self._initialized:
            await self.initialize()

        if quota_mb < 100:
            quota_mb = 100

        repo = await self.get_storage_quotas_repo()
        result = await repo.upsert_org_quota(
            org_id,
            quota_mb=quota_mb,
            soft_limit_pct=soft_limit_pct,
            hard_limit_pct=hard_limit_pct,
        )
        logger.info(f"Set org {org_id} quota: {quota_mb}MB")
        return result

    async def set_team_quota(
        self,
        team_id: int,
        quota_mb: int,
        soft_limit_pct: int = 80,
        hard_limit_pct: int = 100,
    ) -> dict[str, Any]:
        """Set storage quota for a team."""
        if not self._initialized:
            await self.initialize()

        if quota_mb < 100:
            quota_mb = 100

        repo = await self.get_storage_quotas_repo()
        result = await repo.upsert_team_quota(
            team_id,
            quota_mb=quota_mb,
            soft_limit_pct=soft_limit_pct,
            hard_limit_pct=hard_limit_pct,
        )
        logger.info(f"Set team {team_id} quota: {quota_mb}MB")
        return result

    async def check_combined_quota(
        self,
        user_id: int,
        new_bytes: int,
        org_id: Optional[int] = None,
        team_id: Optional[int] = None,
        raise_on_exceed: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check combined quota across user, team, and org levels.

        The quota model is a shared pool:
        - User has their own quota (from users table)
        - Team/org has a shared pool quota (from storage_quotas table)
        - All levels must have sufficient space

        Returns:
            Tuple of (has_quota, quota_info)
        """
        if not self._initialized:
            await self.initialize()

        # Check user quota
        has_user_quota, user_info = await self.check_quota(
            user_id, new_bytes, raise_on_exceed=False
        )

        # Check team quota if applicable
        team_info = None
        has_team_quota = True
        if team_id:
            repo = await self.get_storage_quotas_repo()
            can_allocate, reason = await repo.can_allocate(
                new_bytes, team_id=team_id
            )
            has_team_quota = can_allocate
            team_info = await repo.check_quota_status(team_id=team_id)
            team_info["reason"] = reason

        # Check org quota if applicable
        org_info = None
        has_org_quota = True
        if org_id:
            repo = await self.get_storage_quotas_repo()
            can_allocate, reason = await repo.can_allocate(
                new_bytes, org_id=org_id
            )
            has_org_quota = can_allocate
            org_info = await repo.check_quota_status(org_id=org_id)
            org_info["reason"] = reason

        # Combined check
        has_quota = has_user_quota and has_team_quota and has_org_quota

        combined_info = {
            "user_id": user_id,
            "has_quota": has_quota,
            "new_size_bytes": new_bytes,
            "new_size_mb": round(new_bytes / (1024 * 1024), 2),
            "user": user_info,
            "team": team_info,
            "org": org_info,
            "blocking_level": None,
        }

        if not has_quota:
            if not has_user_quota:
                combined_info["blocking_level"] = "user"
            elif not has_team_quota:
                combined_info["blocking_level"] = "team"
            elif not has_org_quota:
                combined_info["blocking_level"] = "org"

            if raise_on_exceed:
                blocking = combined_info["blocking_level"]
                raise QuotaExceededError(
                    new_bytes / (1024 * 1024),
                    user_info.get("quota_mb", 0),
                    f"Quota exceeded at {blocking} level"
                )

        return has_quota, combined_info

    async def update_org_usage(self, org_id: int, bytes_delta: int) -> dict[str, Any]:
        """Update storage usage for an organization."""
        if not self._initialized:
            await self.initialize()

        mb_delta = bytes_delta / (1024 * 1024)
        repo = await self.get_storage_quotas_repo()
        new_used = await repo.increment_org_used_mb(org_id, mb_delta)
        status = await repo.check_quota_status(org_id=org_id)

        logger.debug(f"Updated org {org_id} usage: +{mb_delta:.2f}MB (total: {new_used:.2f}MB)")
        return status

    async def update_team_usage(self, team_id: int, bytes_delta: int) -> dict[str, Any]:
        """Update storage usage for a team."""
        if not self._initialized:
            await self.initialize()

        mb_delta = bytes_delta / (1024 * 1024)
        repo = await self.get_storage_quotas_repo()
        new_used = await repo.increment_team_used_mb(team_id, mb_delta)
        status = await repo.check_quota_status(team_id=team_id)

        logger.debug(f"Updated team {team_id} usage: +{mb_delta:.2f}MB (total: {new_used:.2f}MB)")
        return status

    # =========================================================================
    # Generated Files Integration
    # =========================================================================

    async def register_generated_file(
        self,
        *,
        user_id: int,
        filename: str,
        storage_path: str,
        file_category: str,
        source_feature: str,
        file_size_bytes: int,
        org_id: Optional[int] = None,
        team_id: Optional[int] = None,
        original_filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        checksum: Optional[str] = None,
        source_ref: Optional[str] = None,
        folder_tag: Optional[str] = None,
        tags: Optional[list[str]] = None,
        is_transient: bool = False,
        expires_at: Optional[datetime] = None,
        retention_policy: str = "user_default",
        check_quota: bool = True,
    ) -> dict[str, Any]:
        """
        Register a generated file and update storage usage.

        This is the main entry point for tracking generated files.
        It checks quota, creates the file record, and updates usage counters.

        Args:
            user_id: Owner user ID
            filename: Stored filename
            storage_path: Relative path to file
            file_category: Category (tts_audio, image, voice_clone, etc.)
            source_feature: Feature that generated the file
            file_size_bytes: Size in bytes
            org_id: Optional organization ID
            team_id: Optional team ID
            original_filename: Original filename
            mime_type: MIME type
            checksum: SHA-256 checksum
            source_ref: Reference to source entity
            folder_tag: Virtual folder tag
            tags: Additional tags
            is_transient: Whether file is temporary
            expires_at: Expiration timestamp
            retention_policy: Retention policy
            check_quota: Whether to check quota before registering

        Returns:
            Created file record with quota info
        """
        if not self._initialized:
            await self.initialize()

        # Validate file size upper limit (10 GB max per file)
        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            raise StorageError(
                f"File size {file_size_bytes / (1024*1024*1024):.2f} GB exceeds "
                f"maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024*1024*1024):.0f} GB"
            )

        # Check quota if requested
        if check_quota:
            has_quota, quota_info = await self.check_combined_quota(
                user_id, file_size_bytes,
                org_id=org_id, team_id=team_id,
                raise_on_exceed=True
            )

        # Create file record
        files_repo = await self.get_generated_files_repo()
        file_record = await files_repo.create_file(
            user_id=user_id,
            filename=filename,
            storage_path=storage_path,
            file_category=file_category,
            source_feature=source_feature,
            file_size_bytes=file_size_bytes,
            org_id=org_id,
            team_id=team_id,
            original_filename=original_filename,
            mime_type=mime_type,
            checksum=checksum,
            source_ref=source_ref,
            folder_tag=folder_tag,
            tags=tags,
            is_transient=is_transient,
            expires_at=expires_at,
            retention_policy=retention_policy,
        )

        # Update usage counters
        await self.update_usage(user_id, file_size_bytes, operation="add")

        if org_id:
            await self.update_org_usage(org_id, file_size_bytes)

        if team_id:
            await self.update_team_usage(team_id, file_size_bytes)

        logger.info(
            f"Registered generated file: {file_category}/{filename} "
            f"({file_size_bytes} bytes) for user {user_id}"
        )

        return file_record

    def _unlink_voice_clone_file(self, file_record: dict[str, Any]) -> None:
        """Best-effort cleanup for generated voice clone files removed from the DB."""
        user_id = file_record.get("user_id")
        file_category = file_record.get("file_category")
        storage_path = str(file_record.get("storage_path") or "")
        if file_category != FILE_CATEGORY_VOICE_CLONE or not user_id or not storage_path:
            return

        try:
            base_dir = DatabasePaths.get_user_voices_dir(user_id)
            resolved_path = (base_dir / storage_path).resolve()
            if resolved_path.is_relative_to(base_dir.resolve()) and resolved_path.exists():
                resolved_path.unlink()
        except Exception as exc:
            logger.debug(f"Failed to remove voice clone file {file_record.get('id')} from disk: {exc}")

    async def _apply_generated_file_usage_delta(
        self,
        file_records: list[dict[str, Any]],
        *,
        operation: str,
    ) -> None:
        """Apply aggregated user/org/team usage changes for generated files."""
        if not file_records:
            return

        user_deltas: dict[int, int] = {}
        org_deltas: dict[int, int] = {}
        team_deltas: dict[int, int] = {}
        for file_record in file_records:
            size = int(file_record.get("file_size_bytes", 0) or 0)
            if size <= 0:
                continue
            user_id = file_record.get("user_id")
            org_id = file_record.get("org_id")
            team_id = file_record.get("team_id")
            if user_id:
                user_deltas[int(user_id)] = user_deltas.get(int(user_id), 0) + size
            if org_id:
                org_deltas[int(org_id)] = org_deltas.get(int(org_id), 0) + size
            if team_id:
                team_deltas[int(team_id)] = team_deltas.get(int(team_id), 0) + size

        org_multiplier = -1 if operation == "remove" else 1
        for user_id, bytes_delta in user_deltas.items():
            await self.update_usage(user_id, bytes_delta, operation=operation)
        for org_id, bytes_delta in org_deltas.items():
            await self.update_org_usage(org_id, org_multiplier * bytes_delta)
        for team_id, bytes_delta in team_deltas.items():
            await self.update_team_usage(team_id, org_multiplier * bytes_delta)

    async def unregister_generated_file(
        self,
        file_id: int,
        hard_delete: bool = False,
    ) -> bool:
        """
        Unregister a generated file and update storage usage.

        Args:
            file_id: File ID to unregister
            hard_delete: If True, permanently delete; otherwise soft delete

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        files_repo = await self.get_generated_files_repo()

        # Get file info first
        file_record = await files_repo.get_file_by_id(file_id)
        if not file_record:
            return False

        was_deleted = bool(file_record.get("is_deleted"))
        user_id = file_record.get("user_id")
        org_id = file_record.get("org_id")
        team_id = file_record.get("team_id")
        file_size_bytes = file_record.get("file_size_bytes", 0)

        if hard_delete:
            success = await files_repo.hard_delete_file(file_id)
        else:
            success = await files_repo.soft_delete_file(file_id)

        if success and file_size_bytes > 0 and not was_deleted:
            # Update usage counters (subtract)
            await self.update_usage(user_id, file_size_bytes, operation="remove")

            if org_id:
                await self.update_org_usage(org_id, -file_size_bytes)

            if team_id:
                await self.update_team_usage(team_id, -file_size_bytes)

        if success and hard_delete:
            self._unlink_voice_clone_file(file_record)

        return success

    async def unregister_generated_files(
        self,
        file_records: list[dict[str, Any]],
        hard_delete: bool = False,
    ) -> int:
        """Unregister generated files from already-loaded records and update usage in aggregate."""
        if not self._initialized:
            await self.initialize()

        if not file_records:
            return 0

        files_repo = await self.get_generated_files_repo()

        if hard_delete:
            deleted_records: list[dict[str, Any]] = []
            for file_record in file_records:
                file_id = file_record.get("id")
                if file_id and await files_repo.hard_delete_file(file_id):
                    deleted_records.append(file_record)
                    self._unlink_voice_clone_file(file_record)
        else:
            active_records = [record for record in file_records if not bool(record.get("is_deleted"))]
            deleted_count = await files_repo.bulk_soft_delete(
                [int(record["id"]) for record in active_records if record.get("id")]
            )
            deleted_records = active_records[:deleted_count]

        usage_records = [record for record in deleted_records if not bool(record.get("is_deleted"))]
        await self._apply_generated_file_usage_delta(usage_records, operation="remove")
        return len(deleted_records)

    async def restore_generated_file(
        self,
        file_id: int,
        *,
        file_record: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Restore a generated file and update usage counters in the service layer."""
        if not self._initialized:
            await self.initialize()

        files_repo = await self.get_generated_files_repo()
        file_record = file_record or await files_repo.get_file_by_id(file_id)
        if not file_record or not file_record.get("is_deleted"):
            return None

        success = await files_repo.restore_file(file_id)
        if not success:
            return None

        await self._apply_generated_file_usage_delta([file_record], operation="add")
        updated = await files_repo.get_file_by_id(file_id)
        return updated or file_record

    async def get_user_generated_files_usage(self, user_id: int) -> dict[str, Any]:
        """Get detailed usage breakdown for user's generated files."""
        if not self._initialized:
            await self.initialize()

        files_repo = await self.get_generated_files_repo()
        usage = await files_repo.get_user_storage_usage(user_id)

        # Get user quota info
        user_info = await self._get_user_storage_info(user_id)
        if user_info:
            usage["quota_mb"] = user_info.get("storage_quota_mb", 0)
            usage["quota_used_mb"] = user_info.get("storage_used_mb", 0)

        return usage

    async def list_user_generated_files(
        self,
        user_id: int,
        *,
        offset: int = 0,
        limit: int = 50,
        file_category: Optional[str] = None,
        folder_tag: Optional[str] = None,
        search: Optional[str] = None,
        include_deleted: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """List generated files for a user."""
        if not self._initialized:
            await self.initialize()

        files_repo = await self.get_generated_files_repo()
        return await files_repo.list_files(
            user_id=user_id,
            offset=offset,
            limit=limit,
            file_category=file_category,
            folder_tag=folder_tag,
            search=search,
            include_deleted=include_deleted,
        )

    async def get_user_folders(self, user_id: int) -> list[dict[str, Any]]:
        """List virtual folders for a user."""
        if not self._initialized:
            await self.initialize()

        files_repo = await self.get_generated_files_repo()
        return await files_repo.list_folders(user_id)

    async def recalculate_org_usage(self, org_id: int) -> dict[str, Any]:
        """Recalculate and update org storage usage from generated files."""
        if not self._initialized:
            await self.initialize()

        # Sum all non-deleted files for users in the org
        async with self.db_pool.acquire() as conn:
            if self._is_postgres_backend():
                total_bytes = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(file_size_bytes), 0)
                    FROM generated_files
                    WHERE org_id = $1 AND is_deleted = FALSE
                    """,
                    org_id,
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT COALESCE(SUM(file_size_bytes), 0)
                    FROM generated_files
                    WHERE org_id = ? AND is_deleted = 0
                    """,
                    (org_id,),
                )
                row = await cursor.fetchone()
                total_bytes = row[0] if row else 0

        total_mb = total_bytes / (1024 * 1024)
        repo = await self.get_storage_quotas_repo()
        await repo.update_org_used_mb(org_id, total_mb)

        status = await repo.check_quota_status(org_id=org_id)
        logger.info(f"Recalculated org {org_id} usage: {total_mb:.2f}MB")
        return status

    async def recalculate_team_usage(self, team_id: int) -> dict[str, Any]:
        """Recalculate and update team storage usage from generated files."""
        if not self._initialized:
            await self.initialize()

        # Sum all non-deleted files for users in the team
        async with self.db_pool.acquire() as conn:
            if self._is_postgres_backend():
                total_bytes = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(file_size_bytes), 0)
                    FROM generated_files
                    WHERE team_id = $1 AND is_deleted = FALSE
                    """,
                    team_id,
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT COALESCE(SUM(file_size_bytes), 0)
                    FROM generated_files
                    WHERE team_id = ? AND is_deleted = 0
                    """,
                    (team_id,),
                )
                row = await cursor.fetchone()
                total_bytes = row[0] if row else 0

        total_mb = total_bytes / (1024 * 1024)
        repo = await self.get_storage_quotas_repo()
        await repo.update_team_used_mb(team_id, total_mb)

        status = await repo.check_quota_status(team_id=team_id)
        logger.info(f"Recalculated team {team_id} usage: {total_mb:.2f}MB")
        return status


# Singleton accessor
_quota_service: Optional[StorageQuotaService] = None


def get_storage_quota_service() -> StorageQuotaService:
    global _quota_service
    if _quota_service is None:
        _quota_service = StorageQuotaService()
    return _quota_service


#######################################################################################################################
#
# Module Functions

# Global instance
_storage_service: Optional[StorageQuotaService] = None


async def get_storage_service() -> StorageQuotaService:
    """Get storage service singleton"""
    global _storage_service
    if not _storage_service:
        _storage_service = StorageQuotaService()
        await _storage_service.initialize()
    return _storage_service


def invalidate_storage_cache_for_user(user_id: int) -> None:
    """Best-effort invalidation for module-level storage service singletons."""
    candidates: list[StorageQuotaService] = []
    if _storage_service is not None:
        candidates.append(_storage_service)
    if _quota_service is not None and _quota_service is not _storage_service:
        candidates.append(_quota_service)
    for service in candidates:
        service.invalidate_user_cache(int(user_id))


async def reset_storage_service() -> None:
    """Reset module-level storage service singletons and release cached state."""
    global _storage_service, _quota_service
    candidates: list[StorageQuotaService] = []
    if _storage_service is not None:
        candidates.append(_storage_service)
    if _quota_service is not None and _quota_service is not _storage_service:
        candidates.append(_quota_service)
    _storage_service = None
    _quota_service = None
    for service in candidates:
        with contextlib.suppress(Exception):
            await service.shutdown()


#
# End of storage_quota_service.py
#######################################################################################################################
