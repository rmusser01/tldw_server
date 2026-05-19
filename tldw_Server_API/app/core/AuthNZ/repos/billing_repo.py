"""
billing_repo.py

Repository for billing-related database operations.
Handles subscription plans, org subscriptions, payment history, and billing audit logs.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.Billing.plan_limits import get_plan_limits


@dataclass
class AuthnzBillingRepo:
    """Repository for billing operations."""

    db_pool: DatabasePool

    def _is_postgres(self, conn: Any | None = None) -> bool:
        """Detect whether the current AuthNZ backend is PostgreSQL.

        Relies on the DatabasePool backend configuration rather than
        per-connection heuristics.
        """
        return getattr(self.db_pool, "pool", None) is not None

    @staticmethod
    def _row_to_dict(cursor, row: tuple) -> dict[str, Any]:
        """Convert a SQLite row tuple to a dict using cursor.description."""
        if row is None:
            return {}
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    @staticmethod
    def _rows_to_dicts(cursor, rows: list[tuple]) -> list[dict[str, Any]]:
        """Convert multiple SQLite row tuples to dicts using cursor.description."""
        if not rows:
            return []
        columns = [col[0] for col in cursor.description]
        return [{columns[idx]: val for idx, val in enumerate(row)} for row in rows]

    @staticmethod
    def _normalize_storage_limits(limits: dict[str, Any]) -> dict[str, Any]:
        """Normalize storage limit keys to storage_mb."""
        if not limits:
            return limits

        normalized = dict(limits)
        if "storage_mb" in normalized:
            normalized.pop("storage_gb", None)
            return normalized

        if "storage_gb" in normalized:
            try:
                storage_mb = int(float(normalized["storage_gb"]) * 1024)
                normalized["storage_mb"] = storage_mb
            except (TypeError, ValueError):
                pass
            normalized.pop("storage_gb", None)

        return normalized

    # =========================================================================
    # Subscription Plans
    # =========================================================================

    async def list_plans(
        self,
        *,
        active_only: bool = True,
        public_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List subscription plans.

        Args:
            active_only: Only return active plans
            public_only: Only return publicly visible plans

        Returns:
            List of plan dicts with parsed limits_json
        """
        conditions = []
        if active_only:
            conditions.append("is_active = 1" if not self._is_postgres() else "is_active = TRUE")
        if public_only:
            conditions.append("is_public = 1" if not self._is_postgres() else "is_public = TRUE")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    list_plans_sql_template = """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        {where_clause}
                        ORDER BY sort_order ASC, created_at ASC
                        """
                    list_plans_sql = list_plans_sql_template.format_map(locals())  # nosec B608
                    rows = await conn.fetch(
                        list_plans_sql
                    )
                    return [self._plan_row_to_dict(dict(r)) for r in rows]
                else:
                    list_plans_sql_template = """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        {where_clause}
                        ORDER BY sort_order ASC, created_at ASC
                        """
                    list_plans_sql = list_plans_sql_template.format_map(locals())  # nosec B608
                    cur = await conn.execute(
                        list_plans_sql
                    )
                    rows = await cur.fetchall()
                    row_dicts = self._rows_to_dicts(cur, rows)
                    # Convert bool fields for SQLite (stored as 0/1)
                    for rd in row_dicts:
                        rd["is_active"] = bool(rd.get("is_active"))
                        rd["is_public"] = bool(rd.get("is_public"))
                    return [self._plan_row_to_dict(rd) for rd in row_dicts]
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.list_plans failed: {exc}")
            raise

    async def get_plan_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a subscription plan by name."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        WHERE name = $1
                        """,
                        name,
                    )
                    return self._plan_row_to_dict(dict(row)) if row else None
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        WHERE name = ?
                        """,
                        (name,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    row_dict = self._row_to_dict(cur, row)
                    row_dict["is_active"] = bool(row_dict.get("is_active"))
                    row_dict["is_public"] = bool(row_dict.get("is_public"))
                    return self._plan_row_to_dict(row_dict)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_plan_by_name failed: {exc}")
            raise

    async def get_plan_by_stripe_price_id(self, price_id: str) -> dict[str, Any] | None:
        """Get a subscription plan by its Stripe price ID."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        WHERE stripe_price_id = $1 OR stripe_price_id_yearly = $1
                        """,
                        price_id,
                    )
                    return self._plan_row_to_dict(dict(row)) if row else None
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        WHERE stripe_price_id = ? OR stripe_price_id_yearly = ?
                        """,
                        (price_id, price_id),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    row_dict = self._row_to_dict(cur, row)
                    row_dict["is_active"] = bool(row_dict.get("is_active"))
                    row_dict["is_public"] = bool(row_dict.get("is_public"))
                    return self._plan_row_to_dict(row_dict)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_plan_by_stripe_price_id failed: {exc}")
            raise

    async def get_plan_by_stripe_product_id(self, product_id: str) -> dict[str, Any] | None:
        """Get a subscription plan by its Stripe product ID."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        WHERE stripe_product_id = $1
                        """,
                        product_id,
                    )
                    return self._plan_row_to_dict(dict(row)) if row else None
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans
                        WHERE stripe_product_id = ?
                        """,
                        (product_id,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    row_dict = self._row_to_dict(cur, row)
                    row_dict["is_active"] = bool(row_dict.get("is_active"))
                    row_dict["is_public"] = bool(row_dict.get("is_public"))
                    return self._plan_row_to_dict(row_dict)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_plan_by_stripe_product_id failed: {exc}")
            raise

    async def get_plan_by_id(self, plan_id: int) -> dict[str, Any] | None:
        """Get a subscription plan by ID."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans WHERE id = $1
                        """,
                        plan_id,
                    )
                    return self._plan_row_to_dict(dict(row)) if row else None
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, name, display_name, description, stripe_product_id, stripe_price_id,
                               stripe_price_id_yearly, price_usd_monthly, price_usd_yearly, limits_json, is_active,
                               is_public,
                               sort_order, created_at
                        FROM subscription_plans WHERE id = ?
                        """,
                        (plan_id,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    row_dict = self._row_to_dict(cur, row)
                    row_dict["is_active"] = bool(row_dict.get("is_active"))
                    row_dict["is_public"] = bool(row_dict.get("is_public"))
                    return self._plan_row_to_dict(row_dict)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_plan_by_id failed: {exc}")
            raise

    def _plan_row_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert plan row to dict with parsed limits."""
        result = dict(row)
        if result.get("limits_json"):
            try:
                if isinstance(result["limits_json"], str):
                    result["limits"] = json.loads(result["limits_json"])
                else:
                    result["limits"] = result["limits_json"]
            except (json.JSONDecodeError, TypeError):
                result["limits"] = {}
        else:
            result["limits"] = {}
        result["limits"] = self._normalize_storage_limits(result["limits"])
        return result

    # =========================================================================
    # Organization Subscriptions
    # =========================================================================

    async def get_org_subscription(self, org_id: int) -> dict[str, Any] | None:
        """Get the subscription for an organization."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        SELECT os.id, os.org_id, os.plan_id, os.stripe_customer_id, os.stripe_subscription_id,
                               os.stripe_subscription_status, os.billing_cycle, os.current_period_start,
                               os.current_period_end, os.status, os.trial_end, os.cancel_at_period_end,
                               os.custom_limits_json,
                               os.created_at, sp.name as plan_name, sp.display_name as plan_display_name,
                               sp.limits_json as plan_limits_json
                        FROM org_subscriptions os
                        JOIN subscription_plans sp ON os.plan_id = sp.id
                        WHERE os.org_id = $1
                        """,
                        org_id,
                    )
                    return self._subscription_row_to_dict(dict(row)) if row else None
                else:
                    cur = await conn.execute(
                        """
                        SELECT os.id, os.org_id, os.plan_id, os.stripe_customer_id, os.stripe_subscription_id,
                               os.stripe_subscription_status, os.billing_cycle, os.current_period_start,
                               os.current_period_end, os.status, os.trial_end, os.cancel_at_period_end,
                               os.custom_limits_json,
                               os.created_at, sp.name as plan_name, sp.display_name as plan_display_name,
                               sp.limits_json as plan_limits_json
                        FROM org_subscriptions os
                        JOIN subscription_plans sp ON os.plan_id = sp.id
                        WHERE os.org_id = ?
                        """,
                        (org_id,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return self._subscription_row_to_dict(self._row_to_dict(cur, row))
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_org_subscription failed: {exc}")
            raise

    async def create_org_subscription(
        self,
        *,
        org_id: int,
        plan_id: int,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        billing_cycle: str = "monthly",
        status: str = "active",
        trial_days: int | None = None,
    ) -> dict[str, Any]:
        """Create a subscription for an organization."""
        trial_end = None
        if trial_days:
            trial_end = datetime.now(timezone.utc) + timedelta(days=trial_days)

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        INSERT INTO org_subscriptions (org_id, plan_id, stripe_customer_id, stripe_subscription_id,
                                                       billing_cycle, status, trial_end)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (org_id) DO UPDATE SET
                            plan_id = EXCLUDED.plan_id,
                            stripe_customer_id = EXCLUDED.stripe_customer_id,
                            stripe_subscription_id = EXCLUDED.stripe_subscription_id,
                            billing_cycle = EXCLUDED.billing_cycle,
                            status = EXCLUDED.status,
                            trial_end = EXCLUDED.trial_end
                        RETURNING id, org_id, plan_id, stripe_customer_id, stripe_subscription_id,
                                  stripe_subscription_status, billing_cycle, current_period_start,
                                  current_period_end, status, trial_end, cancel_at_period_end,
                                  custom_limits_json, created_at
                        """,
                        org_id, plan_id, stripe_customer_id, stripe_subscription_id,
                        billing_cycle, status, trial_end,
                    )
                    return dict(row)
                else:
                    await conn.execute(
                        """
                        INSERT INTO org_subscriptions (org_id, plan_id, stripe_customer_id, stripe_subscription_id,
                                                       billing_cycle, status, trial_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (org_id) DO UPDATE SET
                            plan_id = excluded.plan_id,
                            stripe_customer_id = excluded.stripe_customer_id,
                            stripe_subscription_id = excluded.stripe_subscription_id,
                            billing_cycle = excluded.billing_cycle,
                            status = excluded.status,
                            trial_end = excluded.trial_end
                        """,
                        (org_id, plan_id, stripe_customer_id, stripe_subscription_id,
                         billing_cycle, status, trial_end.isoformat() if trial_end else None),
                    )
                    cur = await conn.execute(
                        "SELECT id, org_id, plan_id, stripe_customer_id, stripe_subscription_id, "
                        "stripe_subscription_status, billing_cycle, current_period_start, "
                        "current_period_end, status, trial_end, custom_limits_json, created_at "
                        "FROM org_subscriptions WHERE org_id = ?",
                        (org_id,),
                    )
                    row = await cur.fetchone()
                    return self._row_to_dict(cur, row)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.create_org_subscription failed: {exc}")
            raise

    async def update_org_subscription(
        self,
        org_id: int,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Update an organization's subscription."""
        if not updates:
            return await self.get_org_subscription(org_id)

        # Handle special cases
        if "custom_limits" in updates:
            updates["custom_limits_json"] = json.dumps(updates.pop("custom_limits"))

        allowed_fields = {
            "plan_id", "stripe_customer_id", "stripe_subscription_id",
            "stripe_subscription_status", "billing_cycle", "current_period_start",
            "current_period_end", "status", "trial_end", "cancel_at_period_end", "custom_limits_json",
        }
        updates = {k: v for k, v in updates.items() if k in allowed_fields}

        if not updates:
            return await self.get_org_subscription(org_id)

        # SECURITY: Verify column names are in the allowed whitelist before dynamic SQL
        # This assertion should never fail since we filtered above, but provides defense-in-depth
        assert all(k in allowed_fields for k in updates), "Invalid column name in updates"

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
                    params = [org_id] + list(updates.values())
                    await conn.execute(
                        f"UPDATE org_subscriptions SET {set_clause} WHERE org_id = $1",  # nosec B608
                        *params,
                    )
                else:
                    set_clause = ", ".join(f"{k} = ?" for k in updates)
                    params = list(updates.values()) + [org_id]
                    await conn.execute(
                        f"UPDATE org_subscriptions SET {set_clause} WHERE org_id = ?",  # nosec B608
                        tuple(params),
                    )

            return await self.get_org_subscription(org_id)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.update_org_subscription failed: {exc}")
            raise

    async def get_subscription_by_stripe_customer(self, stripe_customer_id: str) -> dict[str, Any] | None:
        """Get subscription by Stripe customer ID."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        "SELECT org_id FROM org_subscriptions WHERE stripe_customer_id = $1",
                        stripe_customer_id,
                    )
                    if row:
                        return await self.get_org_subscription(row["org_id"])
                    return None
                else:
                    cur = await conn.execute(
                        "SELECT org_id FROM org_subscriptions WHERE stripe_customer_id = ?",
                        (stripe_customer_id,),
                    )
                    row = await cur.fetchone()
                    if row:
                        return await self.get_org_subscription(row[0])
                    return None
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_subscription_by_stripe_customer failed: {exc}")
            raise

    def _subscription_row_to_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert subscription row to dict with parsed limits."""
        result = dict(row)
        # Parse custom limits
        if result.get("custom_limits_json"):
            try:
                if isinstance(result["custom_limits_json"], str):
                    result["custom_limits"] = json.loads(result["custom_limits_json"])
                else:
                    result["custom_limits"] = result["custom_limits_json"]
            except (json.JSONDecodeError, TypeError):
                result["custom_limits"] = {}
        else:
            result["custom_limits"] = {}
        result["custom_limits"] = self._normalize_storage_limits(result["custom_limits"])

        # Parse plan limits
        if result.get("plan_limits_json"):
            try:
                if isinstance(result["plan_limits_json"], str):
                    result["plan_limits"] = json.loads(result["plan_limits_json"])
                else:
                    result["plan_limits"] = result["plan_limits_json"]
            except (json.JSONDecodeError, TypeError):
                result["plan_limits"] = {}
        else:
            result["plan_limits"] = {}
        result["plan_limits"] = self._normalize_storage_limits(result["plan_limits"])

        # Merge limits: custom overrides plan
        result["effective_limits"] = {**result["plan_limits"], **result["custom_limits"]}
        return result

    async def list_all_subscriptions(self) -> list[dict[str, Any]]:
        """List all org subscriptions with plan details for analytics."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    rows = await conn.fetch(
                        """
                        SELECT os.id, os.org_id, os.plan_id, os.stripe_customer_id,
                               os.stripe_subscription_id, os.stripe_subscription_status,
                               os.billing_cycle, os.current_period_start,
                               os.current_period_end, os.status, os.trial_end,
                               os.cancel_at_period_end, os.custom_limits_json,
                               os.created_at,
                               sp.name as plan_name, sp.display_name as plan_display_name,
                               sp.price_usd_monthly, sp.price_usd_yearly,
                               sp.limits_json as plan_limits_json
                        FROM org_subscriptions os
                        JOIN subscription_plans sp ON os.plan_id = sp.id
                        ORDER BY os.created_at DESC
                        """
                    )
                    return [self._subscription_row_to_dict(dict(r)) for r in rows]
                else:
                    cur = await conn.execute(
                        """
                        SELECT os.id, os.org_id, os.plan_id, os.stripe_customer_id,
                               os.stripe_subscription_id, os.stripe_subscription_status,
                               os.billing_cycle, os.current_period_start,
                               os.current_period_end, os.status, os.trial_end,
                               os.cancel_at_period_end, os.custom_limits_json,
                               os.created_at,
                               sp.name as plan_name, sp.display_name as plan_display_name,
                               sp.price_usd_monthly, sp.price_usd_yearly,
                               sp.limits_json as plan_limits_json
                        FROM org_subscriptions os
                        JOIN subscription_plans sp ON os.plan_id = sp.id
                        ORDER BY os.created_at DESC
                        """
                    )
                    rows = await cur.fetchall()
                    row_dicts = self._rows_to_dicts(cur, rows)
                    return [self._subscription_row_to_dict(rd) for rd in row_dicts]
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.list_all_subscriptions failed: {exc}")
            raise

    # =========================================================================
    # Payment History
    # =========================================================================

    async def add_payment(
        self,
        *,
        org_id: int,
        stripe_invoice_id: str | None = None,
        amount_cents: int,
        currency: str = "usd",
        status: str = "succeeded",
        description: str | None = None,
        invoice_pdf_url: str | None = None,
    ) -> dict[str, Any]:
        """Record a payment in history."""
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        INSERT INTO payment_history (org_id, stripe_invoice_id, amount_cents, currency,
                                                     status, description, invoice_pdf_url)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id, org_id, stripe_invoice_id, amount_cents, currency, status,
                                  description, invoice_pdf_url, created_at
                        """,
                        org_id, stripe_invoice_id, amount_cents, currency, status,
                        description, invoice_pdf_url,
                    )
                    return dict(row)
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO payment_history (org_id, stripe_invoice_id, amount_cents, currency,
                                                     status, description, invoice_pdf_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (org_id, stripe_invoice_id, amount_cents, currency, status,
                         description, invoice_pdf_url),
                    )
                    payment_id = cur.lastrowid
                    cur2 = await conn.execute(
                        "SELECT id, org_id, stripe_invoice_id, amount_cents, currency, status, "
                        "description, invoice_pdf_url, created_at FROM payment_history WHERE id = ?",
                        (payment_id,),
                    )
                    row = await cur2.fetchone()
                    return self._row_to_dict(cur2, row)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.add_payment failed: {exc}")
            raise

    async def list_payments(
        self,
        org_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List payment history for an organization."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    rows = await conn.fetch(
                        """
                        SELECT id, org_id, stripe_invoice_id, amount_cents, currency, status,
                               description, invoice_pdf_url, created_at
                        FROM payment_history
                        WHERE org_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2 OFFSET $3
                        """,
                        org_id, limit, offset,
                    )
                    total = await conn.fetchval(
                        "SELECT COUNT(*) FROM payment_history WHERE org_id = $1",
                        org_id,
                    )
                    return [dict(r) for r in rows], int(total or 0)
                else:
                    cur = await conn.execute(
                        """
                        SELECT id, org_id, stripe_invoice_id, amount_cents, currency, status,
                               description, invoice_pdf_url, created_at
                        FROM payment_history
                        WHERE org_id = ?
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                        """,
                        (org_id, limit, offset),
                    )
                    rows = await cur.fetchall()
                    cur2 = await conn.execute(
                        "SELECT COUNT(*) FROM payment_history WHERE org_id = ?",
                        (org_id,),
                    )
                    total_row = await cur2.fetchone()
                    payments = self._rows_to_dicts(cur, rows)
                    return payments, int(total_row[0]) if total_row else 0
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.list_payments failed: {exc}")
            raise

    # =========================================================================
    # Billing Audit Log
    # =========================================================================

    async def log_billing_action(
        self,
        *,
        org_id: int,
        action: str,
        user_id: int | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> dict[str, Any]:
        """Log a billing-related action for audit purposes."""
        details_json = json.dumps(details) if details else None

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        """
                        INSERT INTO billing_audit_log (org_id, user_id, action, details, ip_address)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id, org_id, user_id, action, details, ip_address, created_at
                        """,
                        org_id, user_id, action, details_json, ip_address,
                    )
                    return dict(row)
                else:
                    cur = await conn.execute(
                        """
                        INSERT INTO billing_audit_log (org_id, user_id, action, details, ip_address)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (org_id, user_id, action, details_json, ip_address),
                    )
                    log_id = cur.lastrowid
                    cur2 = await conn.execute(
                        "SELECT id, org_id, user_id, action, details, ip_address, created_at "
                        "FROM billing_audit_log WHERE id = ?",
                        (log_id,),
                    )
                    row = await cur2.fetchone()
                    return self._row_to_dict(cur2, row)
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.log_billing_action failed: {exc}")
            raise

    # =========================================================================
    # Stripe Webhook Events
    # =========================================================================

    async def record_webhook_event(
        self,
        stripe_event_id: str,
        event_type: str,
        event_data: dict[str, Any],
    ) -> bool:
        """
        Record a Stripe webhook event for idempotency.

        Returns True if this is a new event, False if already processed.
        """
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    # Try to insert, ignore conflict (idempotency)
                    row = await conn.fetchrow(
                        """
                        INSERT INTO stripe_webhook_events (stripe_event_id, event_type, event_data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (stripe_event_id) DO NOTHING
                        RETURNING id
                        """,
                        stripe_event_id, event_type, json.dumps(event_data),
                    )
                    return row is not None
                else:
                    try:
                        await conn.execute(
                            """
                            INSERT INTO stripe_webhook_events (stripe_event_id, event_type, event_data)
                            VALUES (?, ?, ?)
                            """,
                            (stripe_event_id, event_type, json.dumps(event_data)),
                        )
                        return True
                    except sqlite3.IntegrityError:
                        # Likely unique constraint violation
                        return False
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.record_webhook_event failed: {exc}")
            raise

    async def try_claim_webhook_event(
        self,
        stripe_event_id: str,
        *,
        processing_timeout_seconds: int | None = None,
    ) -> bool:
        """
        Atomically try to claim a webhook event for processing.

        Uses UPDATE ... WHERE status IN ('pending', 'failed') to ensure only one
        processor can claim the event, preventing race conditions and allowing
        manual retries of failed events. Optionally allows reclaiming stale
        events stuck in 'processing' past a timeout.

        Returns True if successfully claimed, False if already claimed/processed.
        """
        timeout_seconds: int | None = None
        if processing_timeout_seconds is not None:
            try:
                timeout_seconds = max(1, int(processing_timeout_seconds))
            except (TypeError, ValueError):
                timeout_seconds = None

        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    if timeout_seconds is None:
                        # Atomic claim: only succeeds if status is still pending/failed.
                        result = await conn.execute(
                            """
                            UPDATE stripe_webhook_events
                            SET status = 'processing',
                                processed_at = CURRENT_TIMESTAMP,
                                error_message = NULL
                            WHERE stripe_event_id = $1 AND status IN ('pending', 'failed')
                            """,
                            stripe_event_id,
                        )
                    else:
                        # Allow stale processing claims to be reclaimed.
                        result = await conn.execute(
                            """
                            UPDATE stripe_webhook_events
                            SET status = 'processing',
                                processed_at = CURRENT_TIMESTAMP,
                                error_message = NULL
                            WHERE stripe_event_id = $1
                              AND (
                                    status IN ('pending', 'failed')
                                    OR (
                                        status = 'processing'
                                        AND COALESCE(processed_at, TIMESTAMP 'epoch')
                                            <= CURRENT_TIMESTAMP - ($2::integer * INTERVAL '1 second')
                                    )
                                  )
                            """,
                            stripe_event_id,
                            timeout_seconds,
                        )
                    # PostgreSQL returns "UPDATE N" - extract row count
                    return result and "UPDATE 1" in result
                else:
                    if timeout_seconds is None:
                        cur = await conn.execute(
                            """
                            UPDATE stripe_webhook_events
                            SET status = 'processing',
                                processed_at = CURRENT_TIMESTAMP,
                                error_message = NULL
                            WHERE stripe_event_id = ? AND status IN ('pending', 'failed')
                            """,
                            (stripe_event_id,),
                        )
                    else:
                        timeout_modifier = f"-{timeout_seconds} seconds"
                        cur = await conn.execute(
                            """
                            UPDATE stripe_webhook_events
                            SET status = 'processing',
                                processed_at = CURRENT_TIMESTAMP,
                                error_message = NULL
                            WHERE stripe_event_id = ?
                              AND (
                                    status IN ('pending', 'failed')
                                    OR (
                                        status = 'processing'
                                        AND (
                                            processed_at IS NULL
                                            OR datetime(processed_at) <= datetime('now', ?)
                                        )
                                    )
                                  )
                            """,
                            (stripe_event_id, timeout_modifier),
                        )
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.try_claim_webhook_event failed: {exc}")
            raise

    async def get_webhook_event_status(
        self,
        stripe_event_id: str,
    ) -> str | None:
        """Get the current status for a webhook event."""
        try:
            async with self.db_pool.acquire() as conn:
                if self._is_postgres(conn):
                    row = await conn.fetchrow(
                        "SELECT status FROM stripe_webhook_events WHERE stripe_event_id = $1",
                        stripe_event_id,
                    )
                    return row["status"] if row else None
                else:
                    cur = await conn.execute(
                        "SELECT status FROM stripe_webhook_events WHERE stripe_event_id = ?",
                        (stripe_event_id,),
                    )
                    row = await cur.fetchone()
                    return row[0] if row else None
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.get_webhook_event_status failed: {exc}")
            raise

    async def mark_webhook_processed(
        self,
        stripe_event_id: str,
        *,
        error_message: str | None = None,
    ) -> None:
        """Mark a webhook event as processed."""
        status = "failed" if error_message else "processed"
        try:
            async with self.db_pool.transaction() as conn:
                if self._is_postgres(conn):
                    await conn.execute(
                        """
                        UPDATE stripe_webhook_events
                        SET status = $2,
                            processed_at = CURRENT_TIMESTAMP,
                            error_message = $3::text,
                            retry_count = CASE
                                WHEN $3::text IS NULL THEN retry_count
                                ELSE COALESCE(retry_count, 0) + 1
                            END
                        WHERE stripe_event_id = $1
                        """,
                        stripe_event_id, status, error_message,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE stripe_webhook_events
                        SET status = ?,
                            processed_at = CURRENT_TIMESTAMP,
                            error_message = ?,
                            retry_count = CASE
                                WHEN ? IS NULL THEN retry_count
                                ELSE COALESCE(retry_count, 0) + 1
                            END
                        WHERE stripe_event_id = ?
                        """,
                        (status, error_message, error_message, stripe_event_id),
                    )
        except Exception as exc:
            logger.error(f"AuthnzBillingRepo.mark_webhook_processed failed: {exc}")
            raise

    # =========================================================================
    # Effective Limits Helper
    # =========================================================================

    async def get_org_limits(self, org_id: int) -> dict[str, Any]:
        """
        Get the effective limits for an organization.

        Returns merged limits from plan + custom overrides.
        Falls back to free tier if no subscription exists.
        """
        subscription = await self.get_org_subscription(org_id)

        if not subscription:
            base_limits = get_plan_limits("free")
            # Fall back to free plan from DB if present, merged over canonical defaults
            # so newly introduced categories are not silently treated as unlimited.
            free_plan = await self.get_plan_by_name("free")
            if free_plan:
                free_plan_limits = self._normalize_storage_limits(free_plan.get("limits", {}) or {})
                return {**base_limits, **free_plan_limits}
            # Ultimate fallback to canonical defaults.
            return base_limits

        # Only billable/active subscriptions should retain paid limits.
        # Other statuses (e.g. pending, past_due, canceled) fall back to free-tier
        # limits to avoid stale paid access after billing failures.
        status = str(subscription.get("status", "active")).strip().lower()
        paid_statuses = {"active", "trialing", "canceling"}
        if status not in paid_statuses:
            base_limits = get_plan_limits("free")
            free_plan = await self.get_plan_by_name("free")
            if free_plan:
                free_plan_limits = self._normalize_storage_limits(free_plan.get("limits", {}) or {})
                return {**base_limits, **free_plan_limits}
            return base_limits

        plan_name = subscription.get("plan_name", "free")
        base_limits = get_plan_limits(plan_name)
        effective_limits = subscription.get("effective_limits", {})
        # Merge defaults so newly added categories are not silently unlimited.
        return {**base_limits, **effective_limits}
