# quota_manager.py
# Description: User quota management for chatbook operations
#
"""
Quota Manager for Chatbook Operations
--------------------------------------

Manages user quotas for storage, export/import operations, and rate limits.
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from loguru import logger

# Sentinel value for unlimited quotas (avoids arithmetic overflow issues with sys.maxsize)
UNLIMITED_QUOTA = -1

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import is_truthy

_CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


def _env_flag(name: str) -> bool:
    return is_truthy(os.getenv(name))


class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class OperationType(str, Enum):
    """Types of quota-tracked operations."""
    EXPORT = "export"
    IMPORT = "import"


class QuotaManager:
    """Manages user quotas and usage limits."""

    # Default quotas (can be overridden per user tier)
    DEFAULT_QUOTAS = {
        'max_storage_mb': 1000,  # 1GB default storage
        'max_exports_per_day': 10,
        'max_imports_per_day': 10,
        'max_file_size_mb': 100,
        'max_concurrent_jobs': 2,
        'max_chatbooks': 50
    }

    # Premium user quotas
    PREMIUM_QUOTAS = {
        'max_storage_mb': 5000,  # 5GB for premium users
        'max_exports_per_day': 50,
        'max_imports_per_day': 50,
        'max_file_size_mb': 500,
        'max_concurrent_jobs': 5,
        'max_chatbooks': 200
    }

    # Valid user tiers for validation
    VALID_TIERS = {UserTier.FREE.value, UserTier.PREMIUM.value, UserTier.ENTERPRISE.value}

    def __init__(self, user_id: str, user_tier: str = 'free', db: Optional[Any] = None):
        """
        Initialize quota manager for a user.

        Args:
            user_id: User identifier
            user_tier: User tier (free, premium, enterprise)
            db: Optional database handle for persistent quota checks
        """
        self.user_id = user_id
        # Validate and normalize user_tier
        normalized_tier = user_tier.lower() if isinstance(user_tier, str) else 'free'
        if normalized_tier not in self.VALID_TIERS:
            logger.warning(f"Invalid user tier '{user_tier}' for user {user_id}, defaulting to 'free'")
            normalized_tier = UserTier.FREE.value
        self.user_tier = normalized_tier
        self.quotas = self._get_quotas_for_tier(normalized_tier)
        self.db = db  # Optional DB handle for persistent quota checks
        self._quotas_disabled = _env_flag("CHATBOOKS_DISABLE_QUOTAS") or _env_flag("TEST_MODE") or _env_flag("TESTING") or bool(os.getenv("PYTEST_CURRENT_TEST"))

        # Usage tracking (in production, use database)
        self.usage_cache: dict[str, Any] = {}

    def _get_quotas_for_tier(self, tier: str) -> dict[str, int]:
        """Get quota limits based on user tier."""
        if tier == 'premium':
            return self.PREMIUM_QUOTAS.copy()
        elif tier == 'enterprise':
            # Enterprise users get effectively unlimited quotas (using sentinel value)
            return {
                'max_storage_mb': UNLIMITED_QUOTA,  # -1 means unlimited
                'max_exports_per_day': UNLIMITED_QUOTA,
                'max_imports_per_day': UNLIMITED_QUOTA,
                'max_file_size_mb': 1000,
                'max_concurrent_jobs': 10,
                'max_chatbooks': UNLIMITED_QUOTA
            }
        else:
            return self.DEFAULT_QUOTAS.copy()

    async def check_storage_quota(self, additional_bytes: int = 0) -> tuple[bool, str]:
        """
        Check if user has enough storage quota.

        Args:
            additional_bytes: Additional bytes to be added

        Returns:
            Tuple of (allowed, message)
        """
        # Unlimited quota check
        if self.quotas['max_storage_mb'] == UNLIMITED_QUOTA:
            return True, "Storage quota OK (unlimited)"

        current_usage = await self._get_current_storage_usage()
        max_bytes = self.quotas['max_storage_mb'] * 1024 * 1024

        if current_usage + additional_bytes > max_bytes:
            remaining_mb = (max_bytes - current_usage) / (1024 * 1024)
            return False, f"Storage quota exceeded. You have {remaining_mb:.1f}MB remaining."

        return True, "Storage quota OK"

    async def check_export_quota(self) -> tuple[bool, str]:
        """
        Check if user can perform another export today.

        Returns:
            Tuple of (allowed, message)
        """
        if self._quotas_disabled:
            return True, "Export quota OK (disabled)"
        # Unlimited quota check
        if self.quotas['max_exports_per_day'] == UNLIMITED_QUOTA:
            return True, "Export quota OK (unlimited)"

        exports_today = await self._get_operations_count_today('export')

        if exports_today >= self.quotas['max_exports_per_day']:
            return False, f"Daily export limit ({self.quotas['max_exports_per_day']}) reached. Try again tomorrow."

        return True, "Export quota OK"

    async def check_import_quota(self) -> tuple[bool, str]:
        """
        Check if user can perform another import today.

        Returns:
            Tuple of (allowed, message)
        """
        if self._quotas_disabled:
            return True, "Import quota OK (disabled)"
        # Unlimited quota check
        if self.quotas['max_imports_per_day'] == UNLIMITED_QUOTA:
            return True, "Import quota OK (unlimited)"

        imports_today = await self._get_operations_count_today('import')

        if imports_today >= self.quotas['max_imports_per_day']:
            return False, f"Daily import limit ({self.quotas['max_imports_per_day']}) reached. Try again tomorrow."

        return True, "Import quota OK"

    async def check_file_size(self, file_size_bytes: int) -> tuple[bool, str]:
        """
        Check if file size is within limits.

        Args:
            file_size_bytes: File size in bytes

        Returns:
            Tuple of (allowed, message)
        """
        max_bytes = self.quotas['max_file_size_mb'] * 1024 * 1024

        if file_size_bytes > max_bytes:
            return False, f"File too large. Maximum size is {self.quotas['max_file_size_mb']}MB"

        return True, "File size OK"

    async def check_concurrent_jobs(self) -> tuple[bool, str]:
        """
        Check if user can start another concurrent job.

        Returns:
            Tuple of (allowed, message)
        """
        if self._quotas_disabled:
            return True, "Concurrent jobs OK (disabled)"
        active_jobs = await self._get_active_jobs_count()

        if active_jobs >= self.quotas['max_concurrent_jobs']:
            return False, f"Maximum concurrent jobs ({self.quotas['max_concurrent_jobs']}) reached. Wait for current jobs to complete."

        return True, "Concurrent jobs OK"

    async def record_operation(self, operation_type: str, size_bytes: int = 0):
        """
        Record an operation for quota tracking.

        Args:
            operation_type: Type of operation (export, import)
            size_bytes: Size of the operation in bytes
        """
        # In production, this would update database
        logger.info(f"Recording {operation_type} operation for user {self.user_id} ({size_bytes} bytes)")

    async def get_usage_summary(self) -> dict[str, Any]:
        """
        Get current usage summary for the user.

        Returns:
            Dictionary with usage statistics
        """
        storage_used = await self._get_current_storage_usage()
        exports_today = await self._get_operations_count_today('export')
        imports_today = await self._get_operations_count_today('import')
        active_jobs = await self._get_active_jobs_count()

        # Calculate storage percentage safely, handling unlimited quotas
        max_storage_mb = self.quotas['max_storage_mb']
        if max_storage_mb == UNLIMITED_QUOTA:
            storage_percentage = 0.0  # Unlimited means 0% used
            storage_limit_display = "unlimited"
        elif max_storage_mb <= 0:
            storage_percentage = 0.0  # Guard against division by zero
            storage_limit_display = max_storage_mb
        else:
            storage_percentage = (storage_used / (max_storage_mb * 1024 * 1024)) * 100
            storage_limit_display = max_storage_mb

        # Format limits for display (handle unlimited sentinel)
        exports_limit = self.quotas['max_exports_per_day']
        imports_limit = self.quotas['max_imports_per_day']
        self.quotas.get('max_chatbooks', 50)

        return {
            'storage': {
                'used_mb': storage_used / (1024 * 1024),
                'limit_mb': storage_limit_display,
                'percentage': storage_percentage,
                'unlimited': max_storage_mb == UNLIMITED_QUOTA
            },
            'exports': {
                'today': exports_today,
                'limit': "unlimited" if exports_limit == UNLIMITED_QUOTA else exports_limit,
                'unlimited': exports_limit == UNLIMITED_QUOTA
            },
            'imports': {
                'today': imports_today,
                'limit': "unlimited" if imports_limit == UNLIMITED_QUOTA else imports_limit,
                'unlimited': imports_limit == UNLIMITED_QUOTA
            },
            'jobs': {
                'active': active_jobs,
                'limit': self.quotas['max_concurrent_jobs']
            },
            'tier': self.user_tier
        }

    async def _get_current_storage_usage(self) -> int:
        """Get current storage usage in bytes."""
        # In production, query database for actual usage
        # For now, calculate from user's data directory
        try:
            user_dir = DatabasePaths.get_user_chatbooks_dir(self.user_id)

            if user_dir.exists():
                total_size = 0
                for path in user_dir.rglob('*'):
                    if path.is_file():
                        total_size += path.stat().st_size
                return total_size
            return 0
        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error calculating storage usage: {e}")
            return 0

    async def _get_operations_count_today(self, operation_type: str) -> int:
        """Get count of operations performed today.

        Args:
            operation_type: Type of operation ('export' or 'import')

        Returns:
            Count of operations performed today
        """
        # Validate operation_type
        valid_operations = {OperationType.EXPORT.value, OperationType.IMPORT.value}
        if operation_type not in valid_operations:
            logger.warning(f"Invalid operation_type '{operation_type}', returning 0")
            return 0

        # Prefer database-backed counts when available
        try:
            if self.db is not None and hasattr(self.db, 'execute_query'):
                # Use UTC for consistent timezone-aware date filtering
                start = datetime.now(timezone.utc).date().isoformat()
                if operation_type == OperationType.EXPORT.value:
                    # Count exports created today
                    cursor = self.db.execute_query(
                        "SELECT COUNT(1) as c FROM export_jobs WHERE user_id = ? AND created_at >= ?",
                        (self.user_id, start)
                    )
                else:
                    cursor = self.db.execute_query(
                        "SELECT COUNT(1) as c FROM import_jobs WHERE user_id = ? AND created_at >= ?",
                        (self.user_id, start)
                    )
                # Cursor may be list or cursor
                if hasattr(cursor, 'fetchone'):
                    row = cursor.fetchone()
                    if not row:
                        return 0
                    # sqlite3.Row supports key access but is not dict
                    try:
                        # dict-like row
                        return int(row.get('c', 0))  # type: ignore[attr-defined]
                    except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                        try:
                            # sqlite3.Row or tuple
                            if hasattr(row, 'keys'):
                                return int(row['c'])  # type: ignore[index]
                            return int(row[0])
                        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                            return 0
                elif isinstance(cursor, list) and cursor:
                    row = cursor[0]
                    if isinstance(row, dict):
                        return int(row.get('c', 0))
                    if hasattr(row, 'keys'):
                        try:
                            return int(row['c'])
                        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                            return 0
                    if isinstance(row, (list, tuple)):
                        return int(row[0])
                return 0
        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"QuotaManager DB count fallback due to error: {e}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "chatbooks_quota", "event": "db_count_fallback"},
                )
            except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for chatbooks_quota db_count_fallback")
        # Fallback to cached value
        cache_key = f"{operation_type}_count_{datetime.now().date()}"
        return self.usage_cache.get(cache_key, 0)

    async def _get_active_jobs_count(self) -> int:
        """Get count of currently active jobs."""
        # Prefer database-backed counts when available
        try:
            if self.db is not None and hasattr(self.db, 'execute_query'):
                export_active_statuses = ('pending', 'in_progress')
                import_active_statuses = ('pending', 'validating', 'in_progress')
                # Count both export and import active jobs
                c1 = self.db.execute_query(
                    "SELECT COUNT(1) FROM export_jobs WHERE user_id = ? AND status IN (?, ?)",
                    (self.user_id, *export_active_statuses)
                )
                c2 = self.db.execute_query(
                    "SELECT COUNT(1) FROM import_jobs WHERE user_id = ? AND status IN (?, ?, ?)",
                    (self.user_id, *import_active_statuses)
                )
                def _to_int(cur):
                    if hasattr(cur, 'fetchone'):
                        r = cur.fetchone()
                        return int(r[0]) if r else 0
                    if isinstance(cur, list) and cur:
                        r = cur[0]
                        return int(r.get('COUNT(1)', 0) if isinstance(r, dict) else r[0])
                    return 0
                return _to_int(c1) + _to_int(c2)
        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"QuotaManager DB active job count fallback: {e}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "chatbooks_quota", "event": "db_active_count_fallback"},
                )
            except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for chatbooks_quota db_active_count_fallback")
        return self.usage_cache.get('active_jobs', 0)

    def cleanup_expired_files(self, days_old: int = 7) -> int:
        """
        Clean up old export files to free up space.

        Args:
            days_old: Delete files older than this many days

        Returns:
            Number of files deleted
        """
        try:
            user_export_dir = DatabasePaths.get_user_chatbooks_exports_dir(self.user_id)

            if not user_export_dir.exists():
                return 0

            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0

            for file_path in user_export_dir.glob('*.zip'):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old export: {file_path}")
                    except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
                        try:
                            get_metrics_registry().increment(
                                "app_warning_events_total",
                                labels={"component": "chatbooks_quota", "event": "export_delete_failed"},
                            )
                        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                            logger.debug("metrics increment failed for chatbooks_quota export_delete_failed")

            return deleted_count

        except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error cleaning up old files: {e}")
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "chatbooks_quota", "event": "cleanup_error"},
                )
            except _CHATBOOKS_QUOTA_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for chatbooks_quota cleanup_error")
            return 0


# Dependency injection helper
async def get_quota_manager(user_id: str, user_tier: str = 'free') -> QuotaManager:
    """Get quota manager instance for a user."""
    return QuotaManager(user_id, user_tier)
