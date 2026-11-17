"""Commit guard for caller-owned PostgreSQL Jobs transactions."""

from __future__ import annotations

from typing import Any


def _commit_postgres_transaction(conn: Any, *, operation: str) -> None:
    """Commit a PostgreSQL mutation, rejecting already-aborted transactions."""

    from psycopg.pq import TransactionStatus

    if conn.info.transaction_status == TransactionStatus.INERROR:
        conn.rollback()
        raise RuntimeError(f"Cannot commit PostgreSQL {operation}: transaction is aborted")
    conn.commit()
