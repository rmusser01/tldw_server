"""
Stripe Usage Metering Reconciliation Service.

Syncs aggregated usage from usage_daily table to Stripe's metering API.
Runs as a periodic background task to keep billing in sync.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from tldw_Server_API.app.core.DB_Management.AuthNZ_Metering_Repository import (
    AuthNZBillingSubscriptionRepository,
    AuthNZMeteringSyncLogRepository,
    AuthNZUsageDailyRepository,
)

# Stripe import is optional - mirrors stripe_client.py pattern
try:
    import stripe
    import stripe.error

    STRIPE_AVAILABLE = True
except ImportError:
    stripe = None  # type: ignore[assignment]
    STRIPE_AVAILABLE = False


class StripeMeteringService:
    """Reconciles local usage tracking with Stripe metering."""

    def __init__(
        self,
        *,
        db_pool: Any | None = None,
        usage_repo: AuthNZUsageDailyRepository | None = None,
        subscription_repo: AuthNZBillingSubscriptionRepository | None = None,
        sync_log_repo: AuthNZMeteringSyncLogRepository | None = None,
    ) -> None:
        self._enabled = os.getenv("BILLING_ENABLED", "false").lower() in ("1", "true")
        self._stripe_key = os.getenv("STRIPE_API_KEY", "")
        self._meter_event_name = os.getenv("STRIPE_METER_EVENT_NAME", "api_requests")
        self._db_pool = db_pool
        injected_repo_count = sum(
            repo is not None for repo in (usage_repo, subscription_repo, sync_log_repo)
        )
        if db_pool is None and 0 < injected_repo_count < 3:
            raise ValueError(  # noqa: TRY003
                "Inject all metering repositories together or provide db_pool"
            )
        self._usage_repo = usage_repo or AuthNZUsageDailyRepository(db_pool=db_pool)
        self._subscription_repo = subscription_repo or AuthNZBillingSubscriptionRepository(
            db_pool=db_pool
        )
        self._sync_log_repo = sync_log_repo or AuthNZMeteringSyncLogRepository(
            db_pool=db_pool
        )
        self._use_repository_owned_pool = db_pool is not None or injected_repo_count == 3

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_db_pool(self) -> Any:
        """Lazily acquire the AuthNZ database pool."""
        if self._db_pool is not None:
            return self._db_pool

        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        self._db_pool = await get_db_pool()
        return self._db_pool

    @staticmethod
    def _is_postgres(conn: Any) -> bool:
        """Detect whether *conn* is a PostgreSQL connection (asyncpg)."""
        return hasattr(conn, "fetchrow")

    @staticmethod
    def _is_missing_usage_column_error(exc: Exception) -> bool:
        """Return True when *exc* indicates legacy usage_daily schema."""
        message = str(exc).lower()
        return "bytes_in_total" in message and (
            "no such column" in message
            or "does not exist" in message
            or "undefined column" in message
        )

    @staticmethod
    def _sqlite_rows_to_dicts(
        raw_rows: list[tuple[Any, ...]],
        description: list[tuple[Any, ...]] | None,
        *,
        include_bytes_in_total: bool,
    ) -> list[dict[str, Any]]:
        if not raw_rows:
            return []
        columns = [col[0] for col in (description or [])]
        rows = [dict(zip(columns, row)) for row in raw_rows]
        if not include_bytes_in_total:
            for row in rows:
                row["bytes_in_total"] = 0
        return rows

    @staticmethod
    def _pool_bound_usage_repo(pool: Any | None) -> AuthNZUsageDailyRepository | None:
        if pool is None:
            return None
        return AuthNZUsageDailyRepository(db_pool=pool)

    @staticmethod
    def _pool_bound_subscription_repo(
        pool: Any | None,
    ) -> AuthNZBillingSubscriptionRepository | None:
        if pool is None:
            return None
        return AuthNZBillingSubscriptionRepository(db_pool=pool)

    @staticmethod
    def _pool_bound_sync_log_repo(
        pool: Any | None,
    ) -> AuthNZMeteringSyncLogRepository | None:
        if pool is None:
            return None
        return AuthNZMeteringSyncLogRepository(db_pool=pool)

    @staticmethod
    def _is_missing_stripe_resource_error(exc: Exception) -> bool:
        """Return True when Stripe reports a missing subscription/resource."""
        code = str(getattr(exc, "code", "") or "").lower()
        if code == "resource_missing":
            return True
        message = str(exc).lower()
        return "no such subscription" in message or "resource missing" in message

    async def _query_usage_for_date(
        self,
        pool: Any | None,
        target_date: str,
    ) -> list[dict[str, Any]]:
        """Fetch usage_daily rows for *target_date*.

        Returns a list of dicts with keys: user_id, requests, errors,
        bytes_total, bytes_in_total, latency_avg_ms.
        """
        repo = self._pool_bound_usage_repo(pool) or self._usage_repo
        return await repo.fetch_usage_for_date(target_date)

    async def _query_user_subscription(
        self,
        pool: Any | None,
        user_id: int,
    ) -> dict[str, Any] | None:
        """Look up the active Stripe subscription for a user.

        Joins through org_members -> org_subscriptions to find the user's
        organisation subscription that has a Stripe subscription ID.
        Falls back to checking the ``organizations.owner_user_id`` path.
        """
        repo = self._pool_bound_subscription_repo(pool) or self._subscription_repo
        return await repo.get_active_subscription_for_user(user_id)

    async def _get_subscription_metered_item(
        self, subscription_id: str
    ) -> str | None:
        """Return the first metered subscription-item ID on *subscription_id*.

        Stripe usage records are attached to a *subscription item*, not the
        subscription itself.  This helper retrieves the subscription and picks
        the first item whose price has ``usage_type == 'metered'``.
        """
        if not STRIPE_AVAILABLE:
            return None
        try:
            sub = await asyncio.to_thread(
                stripe.Subscription.retrieve,
                subscription_id,
                expand=["items.data.price"],
            )
            for item in sub.get("items", {}).get("data", []):
                price = item.get("price", {})
                if price.get("recurring", {}).get("usage_type") == "metered":
                    return item["id"]
            # No metered item found — return None so caller can skip
            return None
        except Exception as exc:
            if self._is_missing_stripe_resource_error(exc):
                logger.warning(
                    "Stripe subscription {} is missing; skipping metering sync",
                    subscription_id,
                )
                return None
            logger.warning(
                "Failed to retrieve subscription items for {}: {}",
                subscription_id,
                exc,
            )
            raise

    async def _ensure_metering_sync_table(self, pool: Any | None) -> None:
        """Create the ``metering_sync_log`` tracking table if it does not exist."""
        repo = self._pool_bound_sync_log_repo(pool) or self._sync_log_repo
        await repo.ensure_schema()

    async def _already_synced(
        self,
        pool: Any | None,
        user_id: int,
        day: str,
        subscription_id: str,
    ) -> bool:
        """Return True if usage for this user/day/subscription was already synced."""
        repo = self._pool_bound_sync_log_repo(pool) or self._sync_log_repo
        return await repo.already_synced(
            user_id=user_id,
            day=day,
            subscription_id=subscription_id,
        )

    async def _record_sync(
        self,
        pool: Any | None,
        user_id: int,
        day: str,
        subscription_id: str,
        requests: int,
        bytes_total: int,
    ) -> None:
        """Record a successful sync in metering_sync_log."""
        repo = self._pool_bound_sync_log_repo(pool) or self._sync_log_repo
        await repo.record_sync(
            user_id=user_id,
            day=day,
            subscription_id=subscription_id,
            requests=requests,
            bytes_total=bytes_total,
        )

    async def _report_usage_to_stripe(
        self, subscription_item_id: str, quantity: int, timestamp: int
    ) -> None:
        """Create a Stripe usage record on the given subscription item.

        Uses the legacy ``SubscriptionItem.create_usage_record`` API which is
        widely supported across stripe-python versions.
        """
        if not STRIPE_AVAILABLE:
            raise RuntimeError("stripe package is not installed")

        await asyncio.to_thread(
            stripe.SubscriptionItem.create_usage_record,
            subscription_item_id,
            quantity=quantity,
            timestamp=timestamp,
            action="set",
        )

    async def _query_sync_totals(
        self,
        pool: Any | None,
        target_date: str,
    ) -> list[dict[str, Any]]:
        """Fetch synced totals from metering_sync_log for *target_date*."""
        repo = self._pool_bound_sync_log_repo(pool) or self._sync_log_repo
        return await repo.fetch_sync_totals(target_date)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def sync_daily_usage(self, date: str | None = None) -> dict[str, Any]:
        """Sync a day's usage to Stripe metering.

        Args:
            date: ISO date string (YYYY-MM-DD). Defaults to yesterday.

        Returns:
            Summary of synced records.
        """
        if not self._enabled or not self._stripe_key:
            return {"status": "skipped", "reason": "billing_not_enabled"}

        if not STRIPE_AVAILABLE:
            return {"status": "skipped", "reason": "stripe_package_not_installed"}

        target_date = date or (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        logger.info("Stripe metering sync for {}: started", target_date)

        # Configure stripe key (stripe is guaranteed non-None since STRIPE_AVAILABLE is True)
        stripe.api_key = self._stripe_key  # type: ignore[union-attr]

        pool = None
        if not self._use_repository_owned_pool:
            try:
                pool = await self._get_db_pool()
            except Exception as exc:
                logger.error("Stripe metering sync: failed to get DB pool: {}", exc)
                return {
                    "status": "error",
                    "date": target_date,
                    "error": f"db_pool_unavailable: {exc}",
                }

        # Ensure tracking table exists
        try:
            await self._ensure_metering_sync_table(pool)
        except Exception as exc:
            logger.warning(
                "Stripe metering sync: could not ensure sync table: {}", exc
            )
            # Non-fatal — table may already exist, or usage_daily may not exist either

        # Query usage for the target date
        try:
            usage_rows = await self._query_usage_for_date(pool, target_date)
        except Exception as exc:
            logger.error(
                "Stripe metering sync for {}: failed to query usage: {}",
                target_date,
                exc,
            )
            return {
                "status": "error",
                "date": target_date,
                "error": f"usage_query_failed: {exc}",
            }

        if not usage_rows:
            logger.info("Stripe metering sync for {}: no usage data", target_date)
            return {
                "status": "completed",
                "date": target_date,
                "synced_users": 0,
                "skipped_users": 0,
                "errors": 0,
                "message": "no_usage_data",
            }

        # Compute the epoch timestamp for end-of-day (23:59:59 UTC)
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            usage_timestamp = int(dt.timestamp())
        except ValueError:
            usage_timestamp = int(datetime.now(timezone.utc).timestamp())

        synced = 0
        skipped = 0
        errors = 0

        for row in usage_rows:
            user_id = row["user_id"]
            requests = row.get("requests", 0) or 0
            bytes_total = row.get("bytes_total", 0) or 0

            if requests == 0:
                skipped += 1
                continue

            try:
                # Look up user's Stripe subscription
                sub_info = await self._query_user_subscription(pool, user_id)
                if not sub_info:
                    skipped += 1
                    continue

                subscription_id = sub_info["stripe_subscription_id"]

                # Check for duplicate sync
                try:
                    if await self._already_synced(
                        pool, user_id, target_date, subscription_id
                    ):
                        logger.debug(
                            "Skipping already-synced user {} for {}",
                            user_id,
                            target_date,
                        )
                        skipped += 1
                        continue
                except Exception as exc:
                    # Table might not exist yet; continue without duplicate-sync protection.
                    logger.debug(
                        "Metering sync precheck unavailable for user {} on {}: {}",
                        user_id,
                        target_date,
                        exc,
                    )

                # Find the metered subscription item
                item_id = await self._get_subscription_metered_item(subscription_id)
                if not item_id:
                    logger.debug(
                        "No metered item on subscription {} for user {}",
                        subscription_id,
                        user_id,
                    )
                    skipped += 1
                    continue

                # Report usage to Stripe
                await self._report_usage_to_stripe(
                    item_id, requests, usage_timestamp
                )

                logger.debug(
                    "Synced usage for user {}: requests={}, bytes={}",
                    user_id,
                    requests,
                    bytes_total,
                )

                # Record the sync to prevent double-counting
                try:
                    await self._record_sync(
                        pool,
                        user_id,
                        target_date,
                        subscription_id,
                        requests,
                        bytes_total,
                    )
                except Exception as rec_exc:
                    logger.warning(
                        "Failed to record sync for user {}: {}",
                        user_id,
                        rec_exc,
                    )

                synced += 1

            except Exception as exc:
                logger.error(
                    "Failed to sync usage for user {}: {}", user_id, exc
                )
                errors += 1

        logger.info(
            "Stripe metering sync for {}: completed (synced={}, skipped={}, errors={})",
            target_date,
            synced,
            skipped,
            errors,
        )

        return {
            "status": "completed",
            "date": target_date,
            "synced_users": synced,
            "skipped_users": skipped,
            "errors": errors,
        }

    async def check_reconciliation(
        self, date: str | None = None
    ) -> dict[str, Any]:
        """Compare local usage totals with synced records for drift detection.

        Args:
            date: ISO date string (YYYY-MM-DD). Defaults to yesterday.

        Returns:
            Reconciliation report with any discrepancies found.
        """
        if not self._enabled:
            return {"status": "skipped", "reason": "billing_not_enabled"}

        target_date = date or (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        pool = None
        if not self._use_repository_owned_pool:
            try:
                pool = await self._get_db_pool()
            except Exception as exc:
                logger.error("Reconciliation check: failed to get DB pool: {}", exc)
                return {
                    "status": "error",
                    "date": target_date,
                    "error": f"db_pool_unavailable: {exc}",
                    "discrepancies": [],
                }

        # Fetch local usage totals
        try:
            usage_rows = await self._query_usage_for_date(pool, target_date)
        except Exception as exc:
            logger.error(
                "Reconciliation for {}: failed to query usage: {}",
                target_date,
                exc,
            )
            return {
                "status": "error",
                "date": target_date,
                "error": f"usage_query_failed: {exc}",
                "discrepancies": [],
            }

        # Fetch synced totals
        try:
            sync_rows = await self._query_sync_totals(pool, target_date)
        except Exception as exc:
            logger.warning(
                "Reconciliation for {}: sync log query failed (table may not exist): {}",
                target_date,
                exc,
            )
            sync_rows = []

        # Build lookup: user_id -> synced requests
        synced_by_user: dict[int, int] = {}
        for sr in sync_rows:
            uid = sr["user_id"]
            synced_by_user[uid] = synced_by_user.get(uid, 0) + sr.get(
                "requests_synced", 0
            )

        # Compare
        discrepancies: list[dict[str, Any]] = []
        total_local_requests = 0
        total_synced_requests = 0

        for row in usage_rows:
            uid = row["user_id"]
            local_requests = row.get("requests", 0) or 0
            synced_requests = synced_by_user.pop(uid, 0)

            total_local_requests += local_requests
            total_synced_requests += synced_requests

            if local_requests != synced_requests:
                discrepancies.append(
                    {
                        "user_id": uid,
                        "local_requests": local_requests,
                        "synced_requests": synced_requests,
                        "drift": local_requests - synced_requests,
                    }
                )

        # Any users in sync log but not in usage_daily (shouldn't happen normally)
        for uid, extra_synced in synced_by_user.items():
            total_synced_requests += extra_synced
            discrepancies.append(
                {
                    "user_id": uid,
                    "local_requests": 0,
                    "synced_requests": extra_synced,
                    "drift": -extra_synced,
                }
            )

        return {
            "status": "completed",
            "date": target_date,
            "total_local_requests": total_local_requests,
            "total_synced_requests": total_synced_requests,
            "discrepancies": discrepancies,
        }

    @property
    def is_enabled(self) -> bool:
        """Whether Stripe metering is enabled."""
        return self._enabled and bool(self._stripe_key) and STRIPE_AVAILABLE
