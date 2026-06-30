from __future__ import annotations

import sqlite3


def test_sqlite_store_migrations_add_new_columns(tmp_path) -> None:


    db_path = tmp_path / "sandbox_store.db"
    # Create old schema missing runtime_version and resource_usage
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE sandbox_runs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            spec_version TEXT,
            runtime TEXT,
            base_image TEXT,
            phase TEXT,
            exit_code INTEGER,
            started_at TEXT,
            finished_at TEXT,
            message TEXT,
            image_digest TEXT,
            policy_hash TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE sandbox_vz_sessions (
            id TEXT PRIMARY KEY,
            runtime TEXT,
            vm_id TEXT,
            template_id TEXT,
            workspace_mount TEXT,
            agent_ready INTEGER,
            created_at REAL,
            updated_at REAL
        );
        """
    )
    con.commit()
    con.close()

    # Instantiate store; should run ALTER TABLE migrations
    from tldw_Server_API.app.core.Sandbox.store import SQLiteStore

    SQLiteStore(db_path=str(db_path))

    # Verify columns exist
    con2 = sqlite3.connect(str(db_path))
    cur = con2.execute("PRAGMA table_info(sandbox_runs)")
    cols = {row[1] for row in cur.fetchall()}
    assert "runtime_version" in cols
    assert "resource_usage" in cols
    assert "session_id" in cols
    assert "persona_id" in cols
    assert "workspace_id" in cols
    assert "workspace_group_id" in cols
    assert "scope_snapshot_id" in cols
    assert "claim_owner" in cols
    assert "claim_expires_at" in cols

    cur_acp = con2.execute("PRAGMA table_info(sandbox_acp_sessions)")
    acp_cols = {row[1] for row in cur_acp.fetchall()}
    assert "id" in acp_cols
    assert "user_id" in acp_cols
    assert "sandbox_session_id" in acp_cols
    assert "run_id" in acp_cols
    assert "ssh_private_key" in acp_cols
    assert "persona_id" in acp_cols
    assert "workspace_id" in acp_cols
    assert "workspace_group_id" in acp_cols
    assert "scope_snapshot_id" in acp_cols

    cur_vz = con2.execute("PRAGMA table_info(sandbox_vz_sessions)")
    vz_cols = {row[1] for row in cur_vz.fetchall()}
    assert "helper_instance_id" in vz_cols
    assert "helper_started_at" in vz_cols
    con2.close()
