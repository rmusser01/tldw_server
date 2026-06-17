from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.scheduler import AuthNZScheduler
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


def _same_iso_week_across_rough_bucket_boundary(
    now: datetime,
) -> tuple[datetime, datetime]:
    """Return old timestamps that share an ISO week but cross the former rough bucket."""
    primary_cutoff = now - timedelta(days=90)
    for days_back in range(91, 360):
        first = (now - timedelta(days=days_back)).replace(
            hour=22,
            minute=0,
            second=0,
            microsecond=0,
        )
        second = first + timedelta(hours=4)
        same_iso_week = first.isocalendar()[:2] == second.isocalendar()[:2]
        rough_first = (first.year, (first.timetuple().tm_yday - 1) // 7 + 1)
        rough_second = (second.year, (second.timetuple().tm_yday - 1) // 7 + 1)
        both_weekly_retained = second < primary_cutoff
        if same_iso_week and rough_first != rough_second and both_weekly_retained:
            return first, second
    raise AssertionError("Could not find a stable same-ISO-week boundary pair")


@pytest.mark.asyncio
async def test_privilege_snapshot_retention_job(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "unit-test-secret-key-priv-snapshots-1234567890")
    reset_settings()
    await reset_db_pool()
    ensure_authnz_tables(Path(db_path))

    pool = await get_db_pool()
    now = datetime.now(timezone.utc)

    async with pool.transaction() as conn:
        base_recent = (now - timedelta(days=10)).isoformat()
        older_first_dt, older_same_week_dt = _same_iso_week_across_rough_bucket_boundary(now)
        older_first = older_first_dt.isoformat()
        older_same_week = older_same_week_dt.isoformat()
        team_base = (now - timedelta(days=150)).isoformat()
        ancient = (now - timedelta(days=410)).isoformat()

        payloads = [
            ("snap-recent", base_recent, "seed", "org-alpha", None),
            ("snap-week-keep", older_first, "seed", "org-alpha", None),
            ("snap-week-drop", older_same_week, "seed", "org-alpha", None),
            ("snap-team-keep", team_base, "seed", "org-alpha", "team-1"),
            ("snap-ancient-drop", ancient, "seed", "org-beta", None),
        ]

        for snapshot_id, generated_at, generated_by, org_id, team_id in payloads:
            await conn.execute(
                """
                INSERT INTO privilege_snapshots (
                    snapshot_id,
                    generated_at,
                    generated_by,
                    org_id,
                    team_id,
                    catalog_version,
                    summary_json,
                    scope_index
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    generated_at,
                    generated_by,
                    org_id,
                    team_id,
                    "1.0.0",
                    '{"users": 1, "scope_ids": ["media.ingest"]}',
                    "|media.ingest|",
                ),
            )

    scheduler = AuthNZScheduler()
    await scheduler._prune_privilege_snapshots()

    rows = await pool.fetchall(
        "SELECT snapshot_id, generated_at FROM privilege_snapshots ORDER BY snapshot_id"
    )
    remaining_ids = {row["snapshot_id"] for row in rows}

    assert "snap-recent" in remaining_ids
    assert "snap-week-keep" in remaining_ids
    assert "snap-week-drop" not in remaining_ids, "Only the earliest weekly snapshot should persist"
    assert "snap-ancient-drop" not in remaining_ids

    registry = get_metrics_registry()
    gauge_values = registry.values.get("privilege_snapshots_table_rows")
    assert gauge_values, "Snapshot row gauge should record the latest count"
    assert gauge_values[-1].value == len(remaining_ids)

    await reset_db_pool()
    reset_settings()
