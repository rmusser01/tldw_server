# scheduler.py
# Description: Scheduled jobs for AuthNZ maintenance tasks
#
# Imports
import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

#
# 3rd-party imports
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.alerting import get_security_alert_dispatcher
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.monitoring_repo import AuthnzMonitoringRepo
from tldw_Server_API.app.core.AuthNZ.repos.byok_oauth_state_repo import AuthnzByokOAuthStateRepo
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Metrics import set_gauge

_AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS = (
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
    ValueError,
)

#######################################################################################################################
#
# Scheduled Jobs
#

class AuthNZScheduler:
    """Manages scheduled maintenance tasks for the AuthNZ module"""

    def __init__(self):
        """Initialize the scheduler"""
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.settings = get_settings()
        self._started = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self):
        """Start the scheduler and register all jobs"""
        loop = asyncio.get_running_loop()

        if self._started and self._loop is loop and self.scheduler and self.scheduler.running:
            logger.warning("AuthNZ scheduler already started on current event loop")
            return

        # If we were previously started on a different loop or have a stale scheduler, tear it down
        if self.scheduler and self._loop is not loop:
            logger.info("Restarting AuthNZ scheduler on new event loop")
            try:
                self.scheduler.shutdown(wait=True)
            except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Ignoring scheduler shutdown error during restart: {e}")
            finally:
                self.scheduler = None
                self._started = False
                self._loop = None

        # Always create a fresh scheduler when starting to avoid stale loop bindings
        if not self.scheduler:
            self.scheduler = AsyncIOScheduler(event_loop=loop)
            self._loop = loop

        try:
            try:
                from tldw_Server_API.app.core.AuthNZ.retention_policies import apply_retention_overrides

                await apply_retention_overrides(self.settings)
            except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"AuthNZ scheduler: failed to apply retention overrides: {exc}")

            # Register cleanup jobs
            self._register_session_cleanup()
            self._register_api_key_cleanup()
            self._register_audit_log_cleanup()
            self._register_expired_registration_cleanup()
            self._register_byok_oauth_state_cleanup()

            # Register monitoring jobs
            self._register_auth_failure_monitor()
            self._register_api_usage_monitor()
            # Evaluations: idempotency keys cleanup
            self._register_evaluations_idempotency_cleanup()

            # Usage log pruning jobs
            self._register_usage_log_cleanup()
            self._register_llm_usage_log_cleanup()
            # Daily aggregates pruning jobs
            self._register_usage_daily_cleanup()
            self._register_llm_usage_daily_cleanup()
            # Privilege snapshot retention housekeeping
            self._register_privilege_snapshot_retention()

            # Start the scheduler
            self.scheduler.start()
            self._started = True
            self._loop = loop
            logger.info("AuthNZ scheduler started with all jobs registered")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            # Cleanup scheduler on initialization failure to prevent resource leak
            logger.error(f"Failed to initialize scheduler jobs: {e}")
            if self.scheduler:
                with contextlib.suppress(_AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS):
                    self.scheduler.shutdown(wait=False)
            self.scheduler = None
            self._started = False
            self._loop = None
            raise

    async def stop(self):
        """Stop the scheduler"""
        if not self.scheduler:
            self._started = False
            self._loop = None
            return

        if self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=True)
            except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Ignoring scheduler shutdown error: {e}")

        self._started = False
        self._loop = None
        self.scheduler = None
        logger.info("AuthNZ scheduler stopped")

    def _register_session_cleanup(self):
        """Register job to clean up expired sessions"""
        self.scheduler.add_job(
            self._cleanup_expired_sessions,
            trigger=IntervalTrigger(
                hours=self.settings.SESSION_CLEANUP_INTERVAL_HOURS
            ),
            id='session_cleanup',
            name='Clean up expired sessions',
            replace_existing=True,
            max_instances=1
        )
        logger.debug(f"Registered session cleanup job (every {self.settings.SESSION_CLEANUP_INTERVAL_HOURS} hours)")

    def _register_api_key_cleanup(self):
        """Register job to clean up expired API keys"""
        self.scheduler.add_job(
            self._cleanup_expired_api_keys,
            trigger=CronTrigger(
                hour=2,  # Run at 2 AM daily
                minute=0
            ),
            id='api_key_cleanup',
            name='Clean up expired API keys',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered API key cleanup job (daily at 2 AM)")

    def _register_audit_log_cleanup(self):
        """Register job to prune old audit logs"""
        self.scheduler.add_job(
            self._prune_audit_logs,
            trigger=CronTrigger(
                day=1,  # First day of month
                hour=3,
                minute=0
            ),
            id='audit_log_cleanup',
            name='Prune old audit logs',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered audit log cleanup job (monthly)")

    def _register_usage_log_cleanup(self):
        """Register job to prune old usage_log rows"""
        self.scheduler.add_job(
            self._prune_usage_logs,
            trigger=CronTrigger(hour=3, minute=15),  # Daily at 03:15
            id='usage_log_cleanup',
            name='Prune old usage logs',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered usage log cleanup job (daily at 03:15)")

    def _register_llm_usage_log_cleanup(self):
        """Register job to prune old llm_usage_log rows"""
        self.scheduler.add_job(
            self._prune_llm_usage_logs,
            trigger=CronTrigger(hour=3, minute=30),  # Daily at 03:30
            id='llm_usage_log_cleanup',
            name='Prune old LLM usage logs',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered LLM usage log cleanup job (daily at 03:30)")

    def _register_usage_daily_cleanup(self):
        """Register job to prune old usage_daily rows"""
        self.scheduler.add_job(
            self._prune_usage_daily,
            trigger=CronTrigger(hour=3, minute=40),  # Daily at 03:40
            id='usage_daily_cleanup',
            name='Prune old usage_daily rows',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered usage_daily cleanup job (daily at 03:40)")

    def _register_llm_usage_daily_cleanup(self):
        """Register job to prune old llm_usage_daily rows"""
        self.scheduler.add_job(
            self._prune_llm_usage_daily,
            trigger=CronTrigger(hour=3, minute=45),  # Daily at 03:45
            id='llm_usage_daily_cleanup',
            name='Prune old llm_usage_daily rows',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered llm_usage_daily cleanup job (daily at 03:45)")

    def _register_privilege_snapshot_retention(self):
        """Register job to enforce privilege snapshot retention policy."""
        self.scheduler.add_job(
            self._prune_privilege_snapshots,
            trigger=CronTrigger(hour=2, minute=20),  # Daily at 02:20
            id='privilege_snapshot_retention',
            name='Prune privilege snapshots per retention policy',
            replace_existing=True,
            max_instances=1,
        )
        logger.debug("Registered privilege snapshot retention job (daily at 02:20)")

    def _register_expired_registration_cleanup(self):
        """Register job to clean up expired registration codes"""
        self.scheduler.add_job(
            self._cleanup_expired_registration_codes,
            trigger=CronTrigger(
                hour=1,  # Run at 1 AM daily
                minute=30
            ),
            id='registration_cleanup',
            name='Clean up expired registration codes',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered registration code cleanup job (daily at 1:30 AM)")

    def _register_byok_oauth_state_cleanup(self):
        """Register job to purge consumed/expired BYOK OAuth state records."""
        self.scheduler.add_job(
            self._cleanup_byok_oauth_state,
            trigger=CronTrigger(hour=1, minute=45),
            id='byok_oauth_state_cleanup',
            name='Clean up BYOK OAuth state records',
            replace_existing=True,
            max_instances=1,
        )
        logger.debug("Registered BYOK OAuth state cleanup job (daily at 1:45 AM)")

    def _register_auth_failure_monitor(self):
        """Register job to monitor authentication failures"""
        self.scheduler.add_job(
            self._monitor_auth_failures,
            trigger=IntervalTrigger(minutes=5),  # Every 5 minutes
            id='auth_failure_monitor',
            name='Monitor authentication failures',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered auth failure monitor (every 5 minutes)")

    def _register_api_usage_monitor(self):
        """Register job to monitor API key usage patterns"""
        self.scheduler.add_job(
            self._monitor_api_usage,
            trigger=IntervalTrigger(hours=1),  # Every hour
            id='api_usage_monitor',
            name='Monitor API key usage',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered API usage monitor (hourly)")

    def _register_evaluations_idempotency_cleanup(self):
        """Register job to cleanup stale idempotency keys in Evaluations DBs."""
        # Daily at 4:00 AM
        self.scheduler.add_job(
            self._cleanup_evaluations_idempotency,
            trigger=CronTrigger(hour=4, minute=0),
            id='evaluations_idempotency_cleanup',
            name='Cleanup Evaluations idempotency keys',
            replace_existing=True,
            max_instances=1,
        )
        logger.debug("Registered evaluations idempotency cleanup (daily at 04:00)")

    async def _cleanup_evaluations_idempotency(self):
        """Iterate user evaluation DBs and purge old idempotency keys."""
        try:
            from pathlib import Path

            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
            from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase as _EDB
            # Discover user database base dir (reuse DatabasePaths fallback by building a known path)
            base = Path(_DP.get_user_base_directory(_DP.get_single_user_id())).parent
            deleted_total = 0
            # Include single-user fixed id explicitly
            candidate_ids = set()
            with contextlib.suppress(_AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS):
                candidate_ids.add(int(_DP.get_single_user_id()))
            try:
                if base.exists():
                    for entry in base.iterdir():
                        if entry.is_dir():
                            try:
                                candidate_ids.add(int(entry.name))
                            except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS:
                                continue
            except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS:
                pass
            for uid in sorted(candidate_ids):
                try:
                    db_path = _DP.get_evaluations_db_path(uid)
                    if not db_path.exists():
                        continue
                    db = _EDB(str(db_path))
                    deleted = db.cleanup_idempotency_keys(ttl_hours=72)
                    deleted_total += int(deleted)
                except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS:
                    continue
            if deleted_total:
                logger.info(f"Evaluations idempotency cleanup removed {deleted_total} rows across user DBs")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed evaluations idempotency cleanup: {e}")

    # Cleanup Jobs

    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions from the database"""
        try:
            session_manager = await get_session_manager()
            count = await session_manager.cleanup_expired_sessions()
            if count and count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")

    async def _cleanup_expired_api_keys(self):
        """Mark expired API keys as expired"""
        try:
            api_key_manager = await get_api_key_manager()
            await api_key_manager.cleanup_expired_keys()
            logger.info("Completed API key expiration check")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to cleanup expired API keys: {e}")

    async def _prune_audit_logs(self):
        """Prune audit logs older than retention period"""
        try:
            db_pool = await get_db_pool()
            retention_days = self.settings.AUDIT_LOG_RETENTION_DAYS
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            repo = AuthnzMonitoringRepo(db_pool)
            count = await repo.delete_audit_logs_before(cutoff_date)

            if count > 0:
                logger.info(f"Pruned {count} audit log entries older than {retention_days} days")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to prune audit logs: {e}")

    async def _prune_usage_logs(self):
        """Prune usage_log rows older than retention period."""
        try:
            db_pool = await get_db_pool()
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.USAGE_LOG_RETENTION_DAYS)

            repo = AuthnzUsageRepo(db_pool)
            count = await repo.prune_usage_log_before(cutoff)
            if count:
                logger.info(f"Pruned {count} usage_log rows older than {self.settings.USAGE_LOG_RETENTION_DAYS} days")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to prune usage_log: {e}")

    async def _prune_llm_usage_logs(self):
        """Prune llm_usage_log rows older than retention period."""
        try:
            db_pool = await get_db_pool()
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.LLM_USAGE_LOG_RETENTION_DAYS)

            repo = AuthnzUsageRepo(db_pool)
            count = await repo.prune_llm_usage_log_before(cutoff)
            if count:
                logger.info(f"Pruned {count} llm_usage_log rows older than {self.settings.LLM_USAGE_LOG_RETENTION_DAYS} days")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to prune llm_usage_log: {e}")

    async def _prune_usage_daily(self):
        """Prune usage_daily rows older than retention period"""
        try:
            db_pool = await get_db_pool()
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _gs
            retention_days = _gs().USAGE_DAILY_RETENTION_DAYS
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            repo = AuthnzUsageRepo(db_pool)
            count = await repo.prune_usage_daily_before(cutoff_date.date())
            if count:
                logger.info(f"Pruned {count} usage_daily rows older than {retention_days} days")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to prune usage_daily: {e}")

    async def _prune_llm_usage_daily(self):
        """Prune llm_usage_daily rows older than retention period"""
        try:
            db_pool = await get_db_pool()
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _gs
            retention_days = _gs().LLM_USAGE_DAILY_RETENTION_DAYS
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            repo = AuthnzUsageRepo(db_pool)
            count = await repo.prune_llm_usage_daily_before(cutoff_date.date())
            if count:
                logger.info(f"Pruned {count} llm_usage_daily rows older than {retention_days} days")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to prune llm_usage_daily: {e}")

    async def _prune_privilege_snapshots(self):
        """Enforce privilege snapshot retention (daily + weekly) and emit metrics."""
        try:
            db_pool = await get_db_pool()
            retention_days = max(int(getattr(self.settings, "PRIVILEGE_SNAPSHOT_RETENTION_DAYS", 90)), 0)
            weekly_retention_days = max(
                int(getattr(self.settings, "PRIVILEGE_SNAPSHOT_WEEKLY_RETENTION_DAYS", 365)),
                retention_days,
            )
            now = datetime.now(timezone.utc)
            weekly_cutoff = now - timedelta(days=weekly_retention_days) if weekly_retention_days > 0 else None
            primary_cutoff = now - timedelta(days=retention_days) if retention_days > 0 else None

            def _normalize_rowcount(value: Optional[int]) -> int:
                if value is None:
                    return 0
                try:
                    count = int(value)
                except (TypeError, ValueError):
                    return 0
                return count if count > 0 else 0

            purged_legacy = 0
            purged_duplicates = 0

            async with db_pool.transaction() as conn:
                is_postgres = bool(getattr(db_pool, "pool", None))

                # Purge anything older than the weekly retention window
                if weekly_cutoff is not None:
                    if is_postgres:
                        result = await conn.execute(
                            "DELETE FROM privilege_snapshots WHERE generated_at::timestamptz < $1",
                            weekly_cutoff,
                        )
                        if isinstance(result, str):
                            try:
                                purged_legacy = int(result.split()[-1])
                            except (ValueError, IndexError):
                                purged_legacy = 0
                    else:
                        # SQLite's datetime() doesn't reliably parse ISO8601 with timezone offsets.
                        # Compare ISO strings directly (stored as ISO8601) for robust behavior.
                        cursor = await conn.execute(
                            "DELETE FROM privilege_snapshots WHERE generated_at < ?",
                            (weekly_cutoff.isoformat(),),
                        )
                        purged_legacy = _normalize_rowcount(getattr(cursor, "rowcount", None))

                # Downsample older snapshots (retain first per ISO week per org/team)
                if (
                    primary_cutoff is not None
                    and weekly_cutoff is not None
                    and weekly_retention_days > retention_days
                ):
                    if is_postgres:
                        dedupe_sql = """
                        WITH ranked AS (
                            SELECT
                                snapshot_id,
                                COALESCE(org_id, '__global__') AS org_bucket,
                                COALESCE(team_id, '__none__') AS team_bucket,
                                to_char(generated_at::timestamptz, 'IYYY-IW') AS iso_week,
                                ROW_NUMBER() OVER (
                                    PARTITION BY
                                        COALESCE(org_id, '__global__'),
                                        COALESCE(team_id, '__none__'),
                                        to_char(generated_at::timestamptz, 'IYYY-IW')
                                    ORDER BY generated_at::timestamptz ASC
                                ) AS rn
                            FROM privilege_snapshots
                            WHERE generated_at::timestamptz < $1
                              AND generated_at::timestamptz >= $2
                        )
                        DELETE FROM privilege_snapshots
                        WHERE snapshot_id IN (
                            SELECT snapshot_id FROM ranked WHERE rn > 1
                        )
                        """
                        result = await conn.execute(dedupe_sql, primary_cutoff, weekly_cutoff)
                        if isinstance(result, str):
                            try:
                                purged_duplicates = int(result.split()[-1])
                            except (ValueError, IndexError):
                                purged_duplicates = 0
                    else:
                        # Use string-based comparisons and ISO week bucketing compatible with ISO8601 strings.
                        # SQLite lacks native ISO week formatting, so anchor each date to its ISO-week Thursday.
                        dedupe_sql = """
                        WITH normalized AS (
                            SELECT
                                snapshot_id,
                                COALESCE(org_id, '__global__') AS org_bucket,
                                COALESCE(team_id, '__none__') AS team_bucket,
                                date(
                                    replace(replace(generated_at, 'Z',''), '+00:00',''),
                                    '-3 days',
                                    'weekday 4'
                                ) AS iso_week_anchor,
                                generated_at
                            FROM privilege_snapshots
                            WHERE generated_at < ?
                              AND generated_at >= ?
                        ),
                        ranked AS (
                            SELECT
                                snapshot_id,
                                ROW_NUMBER() OVER (
                                    PARTITION BY
                                        org_bucket,
                                        team_bucket,
                                        strftime('%Y', iso_week_anchor) || '-' || printf('%02d', cast((strftime('%j', iso_week_anchor) - 1) / 7 + 1 as integer))
                                    ORDER BY generated_at ASC
                                ) AS rn
                            FROM normalized
                        )
                        DELETE FROM privilege_snapshots
                        WHERE snapshot_id IN (
                            SELECT snapshot_id FROM ranked WHERE rn > 1
                        )
                        """
                        cursor = await conn.execute(
                            dedupe_sql,
                            (primary_cutoff.isoformat(), weekly_cutoff.isoformat()),
                        )
                        purged_duplicates = _normalize_rowcount(getattr(cursor, "rowcount", None))

            row_count = await db_pool.fetchval("SELECT COUNT(*) FROM privilege_snapshots") or 0
            size_bytes = None
            if is_postgres:
                try:
                    size_bytes = await db_pool.fetchval(
                        "SELECT pg_total_relation_size('privilege_snapshots')"
                    )
                except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS:
                    size_bytes = None

            if size_bytes is not None:
                set_gauge("privilege_snapshots_table_bytes", float(size_bytes))
            set_gauge("privilege_snapshots_table_rows", float(row_count))

            logger.info(
                'Privilege snapshot retention pruned {} legacy rows (> {} days) and {} weekly duplicates (> {} days); remaining={} rows',
                purged_legacy,
                weekly_retention_days,
                purged_duplicates,
                retention_days,
                row_count,
            )
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to prune privilege snapshots: {e}")

    async def _cleanup_expired_registration_codes(self):
        """Clean up expired registration codes"""
        try:
            db_pool = await get_db_pool()

            from tldw_Server_API.app.core.AuthNZ.repos.registration_codes_repo import (
                AuthnzRegistrationCodesRepo,
            )

            repo = AuthnzRegistrationCodesRepo(db_pool)
            count = await repo.deactivate_expired_codes(datetime.now(timezone.utc))

            if count > 0:
                logger.info(f"Deactivated {count} expired registration codes")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to cleanup registration codes: {e}")

    async def _cleanup_byok_oauth_state(self):
        """Purge consumed and expired BYOK OAuth state records."""
        try:
            db_pool = await get_db_pool()
            repo = AuthnzByokOAuthStateRepo(db_pool)
            await repo.ensure_tables()
            purged = await repo.purge_expired()
            if purged:
                logger.info(f"Purged {purged} BYOK OAuth state rows")
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to cleanup BYOK OAuth state: {e}")

    # Monitoring Jobs

    async def _monitor_auth_failures(self):
        """Monitor and alert on authentication failures"""
        try:
            db_pool = await get_db_pool()
            threshold = 10  # Alert if more than 10 failures in 5 minutes
            time_window = datetime.now(timezone.utc) - timedelta(minutes=5)
            is_postgres = getattr(db_pool, "pool", None) is not None
            cutoff = time_window if is_postgres else time_window.isoformat()

            if is_postgres:
                result = await db_pool.fetchone(
                    """
                    SELECT COUNT(*) as failure_count,
                           COUNT(DISTINCT ip_address) as unique_ips
                    FROM audit_logs
                    WHERE action = ANY($1)
                    AND created_at > $2
                    """,
                    ['login_failed', 'invalid_api_key', 'invalid_token'],
                    cutoff,
                )
            else:
                result = await db_pool.fetchone(
                    """
                    SELECT COUNT(*) as failure_count,
                           COUNT(DISTINCT ip_address) as unique_ips
                    FROM audit_logs
                    WHERE action IN (?, ?, ?)
                    AND created_at > ?
                    """,
                    ('login_failed', 'invalid_api_key', 'invalid_token', cutoff),
                )

            if result:
                failure_count = result['failure_count'] or 0
                unique_ips = result['unique_ips'] or 0

                if failure_count > threshold:
                    if self.settings.PII_REDACT_LOGS:
                        logger.warning("⚠️ High authentication failure rate detected (details redacted)")
                    else:
                        logger.warning(
                            f"⚠️ High authentication failure rate detected: "
                            f"{failure_count} failures from {unique_ips} unique IPs in last 5 minutes"
                        )

                    # Here you would trigger actual alerts (email, Slack, etc.)
                    await self._send_security_alert(
                        "High Authentication Failure Rate",
                        f"{failure_count} failures from {unique_ips} IPs" if not self.settings.PII_REDACT_LOGS else "Details redacted",
                        severity="high",
                        metadata={
                            "failure_count": failure_count,
                            "unique_ips": unique_ips,
                            "window_minutes": 5,
                        },
                    )
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to monitor auth failures: {e}")

    async def _monitor_api_usage(self):
        """Monitor API key usage patterns"""
        try:
            db_pool = await get_db_pool()

            # Get API usage statistics for the last hour
            time_window = datetime.now(timezone.utc) - timedelta(hours=1)
            is_postgres = getattr(db_pool, "pool", None) is not None
            cutoff = time_window if is_postgres else time_window.isoformat()

            results = await db_pool.fetchall(
                """
                SELECT
                    k.id,
                    k.name,
                    k.user_id,
                    COUNT(l.id) as usage_count,
                    k.rate_limit
                FROM api_keys k
                LEFT JOIN api_key_audit_log l ON k.id = l.api_key_id
                WHERE k.status = ?
                AND l.created_at > ?
                GROUP BY k.id, k.name, k.user_id, k.rate_limit
                HAVING COUNT(l.id) > 0
                ORDER BY usage_count DESC
                LIMIT 10
                """,
                ('active', cutoff),
            )

            for row in results:
                usage = row['usage_count']
                rate_limit = row['rate_limit'] or 60  # Default rate limit

                # Alert if usage is approaching rate limit
                if usage > rate_limit * 0.8:  # 80% of rate limit
                    logger.warning(
                        f"API key '{row['name']}' (ID: {row['id']}) "
                        f"approaching rate limit: {usage}/{rate_limit} requests/hour"
                    )

            # Log summary
            if results:
                total_usage = sum(r['usage_count'] for r in results)
                logger.info(f"API usage monitoring: {total_usage} total requests in last hour")

        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to monitor API usage: {e}")

    async def _send_security_alert(
        self,
        subject: str,
        message: str,
        *,
        severity: str = "high",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Dispatch a security alert using the configured dispatcher.

        Returns:
            True if the dispatcher attempted to send the alert, False otherwise.
        """
        dispatcher = get_security_alert_dispatcher()
        payload_metadata: dict[str, Any] = {"source": "authnz_scheduler"}
        if metadata:
            payload_metadata.update(metadata)

        try:
            return await dispatcher.dispatch(
                subject=subject,
                message=message,
                severity=severity,
                metadata=payload_metadata,
            )
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"Security alert dispatch failed: {exc}")
            logger.critical(f"🚨 SECURITY ALERT [{severity.upper()}]: {subject} - {message}")
            return False


#######################################################################################################################
#
# Module Functions
#

# Global scheduler instance
_scheduler: Optional[AuthNZScheduler] = None

async def get_authnz_scheduler() -> AuthNZScheduler:
    """Get the AuthNZ scheduler singleton"""
    global _scheduler
    if not _scheduler:
        _scheduler = AuthNZScheduler()
    return _scheduler

async def start_authnz_scheduler():
    """Start the AuthNZ scheduler"""
    scheduler = await get_authnz_scheduler()
    await scheduler.start()
    logger.info("AuthNZ scheduled jobs started")

async def stop_authnz_scheduler():
    """Stop the AuthNZ scheduler"""
    global _scheduler
    if not _scheduler:
        return
    await _scheduler.stop()
    logger.info("AuthNZ scheduled jobs stopped")

async def reset_authnz_scheduler():
    """Reset scheduler singleton (primarily for tests)."""
    global _scheduler
    if _scheduler:
        try:
            await _scheduler.stop()
        except _AUTHNZ_SCHEDULER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Ignoring scheduler stop error during reset: {e}")
        finally:
            _scheduler = None

#
# End of scheduler.py
#######################################################################################################################
