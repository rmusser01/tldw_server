from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Slides import standalone_html_registry as registry_module

UTC = timezone.utc
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def _insert_archive(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    job_uuid: str | None,
    owner: str | None = "owner-1",
    idempotency_key: str | None = "idem-1",
    payload: str = "{}",
) -> None:
    conn.execute(
        """
        INSERT INTO jobs_archive (
            id, uuid, domain, queue, job_type, owner_user_id,
            idempotency_key, payload, status, archived_at
        ) VALUES (?, ?, 'slides', 'default', 'presentation.generate', ?, ?, ?, 'completed', ?)
        """,
        (job_id, job_uuid, owner, idempotency_key, payload, NOW.isoformat()),
    )


def test_sqlite_migration_adds_exact_archive_indexes_shared_tables_and_narrow_uuid_trigger(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-schema.db")
    with sqlite3.connect(db_path) as conn:
        objects = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')")
        }
        assert "slides_standalone_key_registry" in objects
        assert "slides_standalone_reconciliation" in objects
        assert "idx_jobs_archive_slides_scope" in objects
        assert "idx_jobs_archive_uuid_unique" in objects
        assert "trg_jobs_slides_generation_uuid_required" in objects

        scope_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_jobs_archive_slides_scope'"
        ).fetchone()[0]
        normalized_scope = " ".join(scope_sql.split()).lower().replace("( ", "(").replace(" )", ")")
        assert "(domain, queue, job_type, idempotency_key, owner_user_id, archived_at desc)" in normalized_scope
        assert "where idempotency_key is not null" in normalized_scope

        uuid_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_jobs_archive_uuid_unique'"
        ).fetchone()[0]
        normalized_uuid = " ".join(uuid_sql.split()).lower()
        assert "unique index" in normalized_uuid
        assert "where uuid is not null" in normalized_uuid

        registry_columns = {row[1] for row in conn.execute("PRAGMA table_info(slides_standalone_key_registry)")}
        assert registry_columns == {
            "key_id",
            "state",
            "activated_at",
            "retired_at",
            "config_revision",
        }
        assert not any("secret" in column or "digest" in column for column in registry_columns)
        coordination_columns = {row[1] for row in conn.execute("PRAGMA table_info(slides_standalone_reconciliation)")}
        for expected in (
            "holder_uuid",
            "lease_expires_at",
            "fencing_token",
            "cursor",
            "config_revision",
            "startup_complete_epoch",
            "last_complete_epoch",
            "lag",
            "diagnostic_code",
            "diagnostic_count",
            "diagnostic_at",
        ):
            assert expected in coordination_columns

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO jobs (uuid, domain, queue, job_type, status)
                VALUES (NULL, 'slides', 'default', 'presentation.generate', 'queued')
                """
            )
        conn.execute(
            """
            INSERT INTO jobs (uuid, domain, queue, job_type, status)
            VALUES (NULL, 'unrelated', 'default', 'legacy', 'queued')
            """
        )


def test_sqlite_duplicate_archive_uuid_is_diagnosed_without_deduping_or_breaking_jobs(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-duplicate.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_jobs_archive_uuid_unique")
        _insert_archive(conn, job_id=1, job_uuid="duplicate-uuid", idempotency_key="idem-a")
        _insert_archive(conn, job_id=2, job_uuid="duplicate-uuid", idempotency_key="idem-b")
        conn.commit()

    ensure_jobs_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs_archive WHERE uuid='duplicate-uuid'").fetchone()[0] == 2
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_jobs_archive_uuid_unique'"
            ).fetchone()
            is None
        )
        diagnostic = conn.execute(
            """
            SELECT diagnostic_code, diagnostic_count, diagnostic_at
            FROM slides_standalone_reconciliation WHERE singleton_id=1
            """
        ).fetchone()
        assert diagnostic[0] == "duplicate_archive_uuid"
        assert diagnostic[1] >= 2
        assert diagnostic[2]

    jm = JobManager(db_path)
    unrelated = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )
    assert unrelated["uuid"]
    readiness = jm.get_slides_generation_readiness()
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "duplicate_archive_uuid"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              uuid, domain, queue, job_type, owner_user_id, idempotency_key,
              payload, status, completed_at
            ) VALUES (
              'duplicate-uuid', 'slides', 'default', 'presentation.generate',
              'owner-3', 'idem-c', '{"new":true}', 'completed', '2000-01-01 00:00:00'
            )
            """
        )
        conn.commit()
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
    assert jm.prune_jobs(older_than_days=1, domain="slides") == 1
    with sqlite3.connect(db_path) as conn:
        archived_payloads = conn.execute(
            """
            SELECT payload, payload_compressed FROM jobs_archive
            WHERE uuid='duplicate-uuid' ORDER BY id
            """
        ).fetchall()
    assert archived_payloads == [
        ("{}", None),
        ("{}", None),
        ('{"new":true}', None),
    ]


@pytest.mark.parametrize(
    ("job_uuid", "owner", "idempotency_key"),
    [
        (None, "owner-1", "idem-1"),
        ("", "owner-1", "idem-1"),
        ("uuid-1", None, "idem-1"),
        ("uuid-1", "owner-1", None),
    ],
)
def test_sqlite_ambiguous_legacy_generation_rows_fail_only_standalone_readiness(
    tmp_path,
    job_uuid,
    owner,
    idempotency_key,
):
    db_path = ensure_jobs_tables(tmp_path / f"slides-ambiguous-{job_uuid}-{owner}-{idempotency_key}.db")
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=1,
            job_uuid=job_uuid,
            owner=owner,
            idempotency_key=idempotency_key,
        )
        conn.commit()

    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    readiness = jm.get_slides_generation_readiness()
    assert readiness["ready"] is False
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"
    assert readiness["diagnostic_count"] >= 1
    assert jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )["uuid"]


def test_sqlite_multiple_uuid_candidates_for_one_full_scope_are_diagnosed(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-multiple-candidates.db")
    jm = JobManager(db_path)
    active = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="same-correlation",
    )
    with sqlite3.connect(db_path) as conn:
        _insert_archive(
            conn,
            job_id=int(active["id"]) + 100,
            job_uuid="different-uuid",
            owner="owner-1",
            idempotency_key="same-correlation",
        )
        conn.commit()

    ensure_jobs_tables(db_path)

    readiness = JobManager(db_path).get_slides_generation_readiness()
    assert readiness["diagnostic_code"] == "ambiguous_generation_legacy_row"


def test_sqlite_generation_lookup_requires_uuid_owner_scope_key_and_rejects_numeric_reuse(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-lookup.db")
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"digest_key_id": "key-1"},
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
    )
    assert job["uuid"]

    found = jm.resolve_slides_generation_job(
        job_uuid=str(job["uuid"]),
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
        job_id=int(job["id"]),
    )
    assert found is not None
    assert found["uuid"] == job["uuid"]
    assert found["archived"] is False
    assert (
        jm.resolve_slides_generation_job(
            job_uuid=str(job["uuid"]),
            owner_user_id="owner-2",
            idempotency_key="idem-lookup",
            job_id=int(job["id"]),
        )
        is None
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (int(job["id"]),)).fetchone()
        columns = [description[0] for description in conn.execute("SELECT * FROM jobs LIMIT 0").description]
        values = dict(zip(columns, row))
        archive_columns = {item[1] for item in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()}
        insert_columns = [column for column in columns if column in archive_columns]
        conn.execute(
            f"INSERT INTO jobs_archive ({','.join(insert_columns)}) VALUES ({','.join('?' for _ in insert_columns)})",
            tuple(values[column] for column in insert_columns),
        )
        conn.execute("DELETE FROM jobs WHERE id=?", (int(job["id"]),))
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid="reused-numeric-id-uuid",
            owner="owner-2",
            idempotency_key="other-key",
        )
        conn.commit()

    archived = jm.resolve_slides_generation_job(
        job_uuid=str(job["uuid"]),
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
        job_id=int(job["id"]),
    )
    assert archived is not None
    assert archived["archived"] is True
    assert (
        jm.resolve_slides_generation_job(
            job_uuid="reused-numeric-id-uuid",
            owner_user_id="owner-1",
            idempotency_key="idem-lookup",
            job_id=int(job["id"]),
        )
        is None
    )
    replayed = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"digest_key_id": "key-1"},
        owner_user_id="owner-1",
        idempotency_key="idem-lookup",
    )
    assert replayed["uuid"] == job["uuid"]
    assert replayed["archived"] is True


def test_sqlite_generation_idempotency_never_returns_wrong_owner_or_legacy_null_uuid(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-create-authority.db")
    jm = JobManager(db_path)
    first = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="idem-owner",
    )
    assert first["uuid"]
    with pytest.raises(ValueError):
        jm.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={},
            owner_user_id="owner-2",
            idempotency_key="idem-owner",
        )

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TRIGGER trg_jobs_slides_generation_uuid_required")
        conn.execute(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id,
                idempotency_key, payload, status
            ) VALUES (NULL, 'slides', 'default', 'presentation.generate', 'owner-legacy',
                      'legacy-null-winner', '{}', 'queued')
            """
        )
        conn.commit()

    legacy_manager = JobManager(db_path)
    with pytest.raises(ValueError):
        legacy_manager.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={},
            owner_user_id="owner-legacy",
            idempotency_key="legacy-null-winner",
        )


def test_sqlite_archive_compression_targets_uuid_and_skips_legacy_null_uuid(
    tmp_path,
    monkeypatch,
):
    db_path = ensure_jobs_tables(tmp_path / "slides-archive-compress.db")
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"digest_key_id": "key-1"},
        owner_user_id="owner-1",
        idempotency_key="archive-compress",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs SET status='completed', completed_at='2000-01-01 00:00:00'
            WHERE id=?
            """,
            (int(job["id"]),),
        )
        _insert_archive(
            conn,
            job_id=int(job["id"]),
            job_uuid="reused-before-prune",
            owner="owner-2",
            idempotency_key="other",
            payload='{"sentinel":true}',
        )
        conn.execute("UPDATE jobs_archive SET payload_compressed='sentinel' WHERE uuid='reused-before-prune'")
        conn.execute("DROP TRIGGER trg_jobs_slides_generation_uuid_required")
        conn.execute(
            """
            INSERT INTO jobs (
                uuid, domain, queue, job_type, owner_user_id, idempotency_key,
                payload, status, completed_at
            ) VALUES (NULL, 'slides', 'default', 'presentation.generate', 'legacy-owner',
                      'legacy-null-archive', '{"legacy":true}', 'completed', '2000-01-01 00:00:00')
            """
        )
        conn.commit()

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
    assert (
        jm.prune_jobs(
            older_than_days=1,
            domain="slides",
            queue="default",
            job_type="presentation.generate",
        )
        == 2
    )

    with sqlite3.connect(db_path) as conn:
        reused = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive WHERE uuid='reused-before-prune'"
        ).fetchone()
        archived = conn.execute(
            "SELECT payload, payload_compressed FROM jobs_archive WHERE uuid=?",
            (job["uuid"],),
        ).fetchone()
        legacy = conn.execute(
            """
            SELECT payload, payload_compressed FROM jobs_archive
            WHERE uuid IS NULL AND idempotency_key='legacy-null-archive'
            """
        ).fetchone()
        assert reused == ('{"sentinel":true}', "sentinel")
        assert archived[0] is None
        assert archived[1].startswith("gzip64:")
        assert legacy == ('{"legacy":true}', None)
        diagnostic = conn.execute(
            "SELECT diagnostic_code FROM slides_standalone_reconciliation WHERE singleton_id=1"
        ).fetchone()[0]
        assert diagnostic == "ambiguous_generation_legacy_row"


def test_sqlite_reconciliation_lease_fences_takeover_and_preserves_progress(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-fencing.db")
    first_manager = JobManager(db_path)
    second_manager = JobManager(db_path)
    first = first_manager.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW,
    )
    assert first is not None
    assert first["fencing_token"] == 1
    assert (
        first_manager.acquire_slides_reconciliation_lease(
            holder_uuid="holder-a",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=1),
        )
        is None
    )
    assert (
        second_manager.acquire_slides_reconciliation_lease(
            holder_uuid="holder-b",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=1),
        )
        is None
    )
    assert first_manager.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        cursor="cursor-10",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=4,
        now=NOW + timedelta(seconds=2),
    )
    assert first_manager.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=3),
    )

    takeover = second_manager.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=40),
    )
    assert takeover is not None
    assert takeover["fencing_token"] == 2
    assert takeover["cursor"] == "cursor-10"
    assert not first_manager.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=41),
    )
    assert not first_manager.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        cursor="stale",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(seconds=41),
    )
    assert not first_manager.release_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=1,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=41),
    )
    assert second_manager.release_slides_reconciliation_lease(
        holder_uuid="holder-b",
        fencing_token=2,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=100),
    )
    released = second_manager.get_slides_reconciliation_state()
    assert released["holder_uuid"] is None
    assert released["fencing_token"] == 2
    assert released["cursor"] == "cursor-10"


def test_sqlite_revision_change_atomically_invalidates_readiness_cursor_and_lag(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-revision.db")
    jm = JobManager(db_path)
    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW,
    )
    assert lease is not None
    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        cursor="cursor-a",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=9,
        now=NOW + timedelta(seconds=1),
    )
    assert jm.release_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        now=NOW + timedelta(seconds=2),
    )

    changed = jm.acquire_slides_reconciliation_lease(
        holder_uuid="holder-b",
        lease_seconds=30,
        config_revision="revision-b",
        now=NOW + timedelta(seconds=3),
    )
    assert changed is not None
    assert changed["fencing_token"] == lease["fencing_token"] + 1
    assert changed["config_revision"] == "revision-b"
    assert changed["cursor"] is None
    assert changed["startup_complete_epoch"] is None
    assert changed["last_complete_epoch"] is None
    assert changed["lag"] == 0


def test_sqlite_key_rotation_fences_reconciliation_and_stale_holder(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-key-revision-fence.db")
    jm = JobManager(db_path)
    initial = jm.compare_and_swap_slides_current_key(
        expected_current_key_id=None,
        expected_config_revision=None,
        new_current_key_id="key-a",
        new_config_revision="revision-a",
        changed_at=NOW,
    )
    assert initial is not None
    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="holder-a",
        lease_seconds=30,
        config_revision="revision-a",
        now=NOW + timedelta(seconds=1),
    )
    assert lease is not None
    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        cursor="cursor-a",
        startup_complete_epoch="revision-a",
        last_complete_epoch=NOW.timestamp(),
        lag=7,
        now=NOW + timedelta(seconds=2),
    )

    rotated = jm.compare_and_swap_slides_current_key(
        expected_current_key_id="key-a",
        expected_config_revision="revision-a",
        new_current_key_id="key-b",
        new_config_revision="revision-b",
        changed_at=NOW + timedelta(seconds=3),
    )
    assert rotated is not None and rotated["applied_here"]
    state = jm.get_slides_reconciliation_state()
    assert state["config_revision"] == "revision-b"
    assert state["fencing_token"] == lease["fencing_token"] + 1
    assert state["holder_uuid"] is None
    assert state["cursor"] is None
    assert state["startup_complete_epoch"] is None
    assert state["last_complete_epoch"] is None
    assert state["lag"] == 0
    assert not jm.renew_slides_reconciliation_lease(
        holder_uuid="holder-a",
        fencing_token=lease["fencing_token"],
        config_revision="revision-a",
        lease_seconds=30,
        now=NOW + timedelta(seconds=4),
    )
    assert (
        jm.acquire_slides_reconciliation_lease(
            holder_uuid="stale-revision-holder",
            lease_seconds=30,
            config_revision="revision-a",
            now=NOW + timedelta(seconds=4),
        )
        is None
    )


@pytest.mark.asyncio
async def test_job_manager_registry_adapter_uses_only_source_free_shared_state(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "slides-registry-adapter.db")
    jm = JobManager(db_path)
    store = registry_module.JobManagerDigestKeyRegistryStore(jm)

    empty = await store.load_digest_key_registry()
    assert empty.records == ()
    activated_at = NOW - timedelta(days=90)
    initial = await store.compare_and_swap_current_key(
        expected_current_key_id=None,
        expected_config_epoch=None,
        new_current_key_id="old-key",
        new_config_epoch="revision-a",
        changed_at=activated_at,
    )
    assert initial is not None and initial.applied_here
    rotated_at = NOW - timedelta(days=33)
    rotated = await store.compare_and_swap_current_key(
        expected_current_key_id="old-key",
        expected_config_epoch="revision-a",
        new_current_key_id="new-key",
        new_config_epoch="revision-b",
        changed_at=rotated_at,
    )
    assert rotated is not None and rotated.applied_here
    assert all(record.activated_at.tzinfo == UTC for record in rotated.state.records)

    lease = jm.acquire_slides_reconciliation_lease(
        holder_uuid="sweep-holder",
        lease_seconds=30,
        config_revision="revision-b",
        now=NOW,
    )
    assert lease is not None
    assert jm.checkpoint_slides_reconciliation(
        holder_uuid="sweep-holder",
        fencing_token=lease["fencing_token"],
        config_revision="revision-b",
        cursor=None,
        startup_complete_epoch="revision-b",
        last_complete_epoch=NOW.timestamp(),
        lag=0,
        now=NOW + timedelta(seconds=1),
        completed=True,
        sweep_key_id="old-key",
        sweep_started_at=NOW,
    )
    proof = await store.load_dormant_sweep_proof(key_id="old-key")
    assert proof is not None
    assert proof.complete
    assert proof.unexpired_reference_count == 0
    assert proof.sweep_started_at.tzinfo == UTC

    removed = await store.compare_and_swap_remove_key(
        key_id="old-key",
        expected_retired_at=rotated_at,
        expected_config_epoch="revision-b",
    )
    assert removed is not None
    assert [record.key_id for record in removed.records] == ["new-key"]

    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            "SELECT key_id, state, activated_at, retired_at, config_revision FROM slides_standalone_key_registry"
        ).fetchall()
        rendered = json.dumps(raw)
        assert "secret" not in rendered.lower()
        assert "hmac" not in rendered.lower()
