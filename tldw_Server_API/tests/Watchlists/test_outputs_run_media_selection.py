"""Output rendering "by run" resolves media through the Watchlists owner (TASK-13317).

It used to probe the Media database for the run-to-media mapping, which lives in the
Watchlists database, so it always found nothing.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import outputs_templates
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

pytestmark = pytest.mark.unit


def test_run_media_ids_come_from_the_callers_own_run(tmp_path, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "watchlists.db"))
    )
    dbs = {uid: WatchlistsDatabase(user_id=uid, backend=backend) for uid in (1, 2)}
    monkeypatch.setattr(WatchlistsDatabase, "for_user", classmethod(lambda _cls, uid: dbs[int(uid)]))

    owner = dbs[1]
    job = owner.create_job(
        name="monitor",
        description=None,
        scope_json=None,
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )
    run = owner.create_run(int(job.id))
    for media_id in (30, 10, 20):
        owner.append_run_item(int(run.id), media_id)

    assert outputs_templates._select_media_ids_for_run(1, int(run.id), 2) == [10, 20]
    assert outputs_templates._select_media_ids_for_run(2, int(run.id), 10) == []
    assert outputs_templates._select_media_ids_for_run(1, 99999, 10) == []
