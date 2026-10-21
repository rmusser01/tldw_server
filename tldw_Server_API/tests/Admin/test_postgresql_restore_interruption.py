"""Real tool interruption must roll back the target before a legitimate retry."""

import re
import shutil
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.DB_Backups import (
    PostgresToolTimeout,
    assert_postgres_database_offline,
    create_postgres_backup,
    restore_postgres_backup,
)
from tldw_Server_API.tests._plugins.postgres import pg_restricted_backend as official_restricted

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def separate_interruption_target(pg_secondary_database_config):
    yield from official_restricted.__wrapped__(pg_secondary_database_config)


def test_real_restore_timeout_rolls_back_then_accepts_legitimate_retry(
    tmp_path, monkeypatch, pg_restricted_backend, separate_interruption_target
):
    source, target = pg_restricted_backend, separate_interruption_target
    target_name = target.config.pg_database
    if not re.fullmatch(r"[a-zA-Z0-9_]+", target_name) or source.config.pg_database == target_name:
        pytest.fail("Distinct official database identifiers are required")
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    for key in ("PGHOSTADDR", "PGSERVICE", "PGOPTIONS", "PGPORT"):
        monkeypatch.delenv(key, raising=False)
    from tldw_Server_API.app.core.DB_Management import DB_Backups

    real_tool = DB_Backups._run_postgres_tool

    def with_progress(command, **kwargs):
        # pg_restore emits server notices at INFO level, enabled only by --verbose.
        if Path(command[0]).name == "pg_restore":
            command = [command[0], "--verbose", *command[1:]]
        return real_tool(command, **kwargs)

    monkeypatch.setattr(DB_Backups, "_run_postgres_tool", with_progress)
    directory = tmp_path / "interruption-backup"
    completed = False
    try:
        # The source is fast; only COPY into this exact disposable target pauses.
        source.execute(
            "CREATE FUNCTION c09_restore_pause() RETURNS boolean LANGUAGE plpgsql AS $$ "
            "BEGIN IF current_database() = '"
            + target_name
            + "' THEN RAISE WARNING 'c09-copy-entered'; PERFORM pg_sleep(3); END IF; "
            "RETURN TRUE; END $$"
        )  # nosec B608 -- target_name is an official fixture identifier validated above.
        source.execute("CREATE TABLE c09_incoming (value text NOT NULL CHECK (c09_restore_pause()))")
        source.execute("INSERT INTO c09_incoming VALUES (?)", ("complete source row",))
        target.execute("CREATE TABLE c09_retained (value text NOT NULL)")
        target.execute("INSERT INTO c09_retained VALUES (?)", ("untouched target sentinel",))
        source.get_pool().close_all()
        target.get_pool().close_all()
        dump = create_postgres_backup(source, backup_dir=str(directory), label="postgresql", timeout_seconds=15)
        assert_postgres_database_offline(target.config, timeout_seconds=15)
        with pytest.raises(PostgresToolTimeout) as interrupted:
            restore_postgres_backup(target, dump_file=dump, timeout_seconds=1)
        error = interrupted.value
        assert not error.process_running and not error.cleanup_exceeded and error.process.poll() is not None  # nosec B101 -- a timeout cannot release an unreaped child.
        assert "c09-copy-entered" in error.process.communicate(timeout=1)[1]  # nosec B101 -- actual COPY reached the target-only delay before timeout.
        target = DatabaseBackendFactory.create_backend(target.config)
        assert (
            not target.table_exists("c09_incoming")
            and target.execute("SELECT to_regprocedure('public.c09_restore_pause()') IS NULL AS absent").rows[0][
                "absent"
            ]
        )  # nosec B101 -- transaction-created objects must not survive interruption.
        assert target.execute("SELECT value FROM c09_retained").rows == [{"value": "untouched target sentinel"}]  # nosec B101 -- failed restore cannot damage prior target data.
        target.get_pool().close_all()
        assert_postgres_database_offline(target.config, timeout_seconds=15)
        assert restore_postgres_backup(target, dump_file=dump, timeout_seconds=15) == "ok"  # nosec B101 -- one legitimate normal-budget retry must complete.
        target = DatabaseBackendFactory.create_backend(target.config)
        source = DatabaseBackendFactory.create_backend(source.config)
        assert target.execute("SELECT value FROM c09_incoming").rows == [{"value": "complete source row"}]  # nosec B101 -- retry restores complete incoming data.
        assert target.execute("SELECT value FROM c09_retained").rows == [{"value": "untouched target sentinel"}]  # nosec B101 -- unrelated target sentinel remains intact.
        assert source.execute("SELECT value FROM c09_incoming").rows == [{"value": "complete source row"}]  # nosec B101 -- original source stays intact.
        completed = True
    finally:
        source.get_pool().close_all()
        target.get_pool().close_all()
        if completed and directory.exists():
            shutil.rmtree(directory)
