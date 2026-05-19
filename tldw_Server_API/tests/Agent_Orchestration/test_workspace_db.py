"""Tests for ACPWorkspace DB CRUD and schema migration."""
import tempfile

import pytest

from tldw_Server_API.app.core.DB_Management.Orchestration_DB import (
    CanonicalWorkspaceBridgeConflictError,
    OrchestrationDB,
    OrchestrationNotFoundError,
)


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        instance = OrchestrationDB(user_id=1, db_dir=tmp)
        yield instance
        instance.close()


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_fresh_db_creates_v2_schema(self, db):
        """A fresh DB should have all current workspace tables and columns."""
        db._ensure_schema()
        conn = db._get_conn()
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "acp_workspaces" in tables
        assert "acp_workspace_mcp_servers" in tables
        assert "projects" in tables
        workspace_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(acp_workspaces)").fetchall()
        }
        assert "canonical_workspace_id" in workspace_cols
        # Check user_version
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3

    def test_projects_has_workspace_id_column(self, db):
        """The projects table should have a workspace_id column after migration."""
        db._ensure_schema()
        conn = db._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(projects)").fetchall()}
        assert "workspace_id" in cols

    def test_v1_to_current_migration(self):
        """Simulate a v1 DB and verify migration to the current schema works."""
        import os
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "orchestration.db")
            # Create a v1 DB manually
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    user_id INTEGER NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );
                PRAGMA user_version=1;
            """)
            conn.execute(
                "INSERT INTO projects (name, description, user_id, metadata, created_at) "
                "VALUES ('test', '', 1, '{}', '2025-01-01')"
            )
            conn.commit()
            conn.close()

            # Open with OrchestrationDB — should migrate
            db = OrchestrationDB(user_id=1, db_dir=tmp)
            db._ensure_schema()
            c = db._get_conn()

            # Check v2 tables exist
            tables = {
                r[0]
                for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "acp_workspaces" in tables
            assert "acp_workspace_mcp_servers" in tables

            # Check workspace_id column added
            cols = {r[1] for r in c.execute("PRAGMA table_info(projects)").fetchall()}
            assert "workspace_id" in cols

            # Existing project data preserved
            row = c.execute("SELECT * FROM projects WHERE id = 1").fetchone()
            assert row is not None
            assert row[1] == "test"  # name

            workspace_cols = {
                r[1] for r in c.execute("PRAGMA table_info(acp_workspaces)").fetchall()
            }
            assert "canonical_workspace_id" in workspace_cols

            version = c.execute("PRAGMA user_version").fetchone()[0]
            assert version == 3
            db.close()

    def test_v2_to_v3_migration_backfills_canonical_workspace_id(self):
        """Existing metadata links should backfill the dedicated canonical column."""
        import os
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "orchestration.db")
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE acp_workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    workspace_type TEXT NOT NULL DEFAULT 'manual',
                    parent_workspace_id INTEGER,
                    env_vars TEXT NOT NULL DEFAULT '{}',
                    git_remote_url TEXT,
                    git_default_branch TEXT,
                    git_current_branch TEXT,
                    git_is_dirty INTEGER,
                    last_health_check TEXT,
                    health_status TEXT NOT NULL DEFAULT 'unknown',
                    user_id INTEGER NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );
                CREATE UNIQUE INDEX idx_acp_workspaces_name ON acp_workspaces(user_id, name);
                CREATE UNIQUE INDEX idx_acp_workspaces_root ON acp_workspaces(user_id, root_path);
                PRAGMA user_version=2;
            """)
            conn.execute(
                "INSERT INTO acp_workspaces "
                "(name, root_path, user_id, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "Linked",
                    "/tmp/linked",
                    1,
                    '{"canonical_workspace_id":"workspace-alpha"}',
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            conn.commit()
            conn.close()

            db = OrchestrationDB(user_id=1, db_dir=tmp)
            db._ensure_schema()

            found = db.get_workspace_by_canonical_workspace_id("workspace-alpha")
            assert found is not None
            assert found.name == "Linked"
            db.close()

    def test_v2_to_v3_migration_rejects_duplicate_canonical_links(self):
        """Duplicate metadata links should fail migration with remediation context."""
        import os
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "orchestration.db")
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE acp_workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    workspace_type TEXT NOT NULL DEFAULT 'manual',
                    parent_workspace_id INTEGER,
                    env_vars TEXT NOT NULL DEFAULT '{}',
                    git_remote_url TEXT,
                    git_default_branch TEXT,
                    git_current_branch TEXT,
                    git_is_dirty INTEGER,
                    last_health_check TEXT,
                    health_status TEXT NOT NULL DEFAULT 'unknown',
                    user_id INTEGER NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                );
                PRAGMA user_version=2;
            """)
            for name, root_path in (("Linked A", "/tmp/a"), ("Linked B", "/tmp/b")):
                conn.execute(
                    "INSERT INTO acp_workspaces "
                    "(name, root_path, user_id, metadata, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        name,
                        root_path,
                        1,
                        '{"canonical_workspace_id":"workspace-alpha"}',
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
            conn.commit()
            conn.close()

            db = OrchestrationDB(user_id=1, db_dir=tmp)
            with pytest.raises(ValueError, match="Duplicate canonical workspace links"):
                db._ensure_schema()
            db.close()


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------


class TestWorkspaceCRUD:
    def test_create_workspace(self, db):
        ws = db.create_workspace(name="My Project", root_path="/home/user/project")
        assert ws.id > 0
        assert ws.name == "My Project"
        assert ws.root_path == "/home/user/project"
        assert ws.workspace_type == "manual"
        assert ws.health_status == "unknown"
        assert ws.user_id == 1

    def test_get_workspace(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        fetched = db.get_workspace(ws.id)
        assert fetched is not None
        assert fetched.name == "WS1"
        assert fetched.root_path == "/tmp/ws1"

    def test_get_workspace_not_found(self, db):
        assert db.get_workspace(999) is None

    def test_get_workspace_wrong_user(self):
        """Workspace created by user 1 should not be visible to user 2."""
        with tempfile.TemporaryDirectory() as tmp:
            db1 = OrchestrationDB(user_id=1, db_dir=tmp)
            db2 = OrchestrationDB(user_id=2, db_dir=tmp)
            ws = db1.create_workspace(name="WS1", root_path="/tmp/ws1")
            # User 2 can't see user 1's workspace
            assert db2.get_workspace(ws.id) is None
            db1.close()
            db2.close()

    def test_list_workspaces(self, db):
        db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_workspace(name="WS2", root_path="/tmp/ws2")
        workspaces = db.list_workspaces()
        assert len(workspaces) == 2

    def test_list_workspaces_filter_type(self, db):
        db.create_workspace(name="WS1", root_path="/tmp/ws1", workspace_type="manual")
        db.create_workspace(name="WS2", root_path="/tmp/ws2", workspace_type="discovered")
        manual = db.list_workspaces(workspace_type="manual")
        assert len(manual) == 1
        assert manual[0].name == "WS1"

    def test_list_workspaces_filter_health(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.update_workspace_health(ws.id, health_status="healthy")
        db.create_workspace(name="WS2", root_path="/tmp/ws2")
        healthy = db.list_workspaces(health_status="healthy")
        assert len(healthy) == 1
        assert healthy[0].name == "WS1"

    def test_update_workspace(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        updated = db.update_workspace(ws.id, name="WS1-Renamed", description="Updated")
        assert updated.name == "WS1-Renamed"
        assert updated.description == "Updated"
        assert updated.updated_at is not None

    def test_update_workspace_not_found(self, db):
        with pytest.raises(OrchestrationNotFoundError):
            db.update_workspace(999, name="New")

    def test_update_workspace_env_vars(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        updated = db.update_workspace(ws.id, env_vars={"KEY": "value"})
        assert updated.env_vars == {"KEY": "value"}
        # Verify persisted
        fetched = db.get_workspace(ws.id)
        assert fetched.env_vars == {"KEY": "value"}

    def test_delete_workspace(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        assert db.delete_workspace(ws.id) is True
        assert db.get_workspace(ws.id) is None

    def test_delete_workspace_not_found(self, db):
        assert db.delete_workspace(999) is False

    def test_unique_name_constraint(self, db):
        db.create_workspace(name="WS1", root_path="/tmp/ws1")
        with pytest.raises(ValueError, match="already exists"):
            db.create_workspace(name="WS1", root_path="/tmp/ws_different")

    def test_unique_root_path_constraint(self, db):
        db.create_workspace(name="WS1", root_path="/tmp/ws1")
        with pytest.raises(ValueError, match="already exists"):
            db.create_workspace(name="WS2", root_path="/tmp/ws1")

    def test_update_workspace_unique_constraint(self, db):
        db.create_workspace(name="WS1", root_path="/tmp/ws1")
        ws2 = db.create_workspace(name="WS2", root_path="/tmp/ws2")
        with pytest.raises(ValueError, match="conflicts"):
            db.update_workspace(ws2.id, name="WS1")

    def test_get_workspace_by_root_path(self, db):
        db.create_workspace(name="WS1", root_path="/tmp/ws1")
        found = db.get_workspace_by_root_path("/tmp/ws1")
        assert found is not None
        assert found.name == "WS1"
        assert db.get_workspace_by_root_path("/tmp/nonexistent") is None

    def test_get_workspaces_by_ids_returns_current_user_workspaces(self, db):
        ws1 = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        ws2 = db.create_workspace(name="WS2", root_path="/tmp/ws2")

        found = db.get_workspaces_by_ids([ws1.id, ws2.id, ws1.id, 999])

        assert set(found) == {ws1.id, ws2.id}
        assert found[ws1.id].name == "WS1"
        assert found[ws2.id].name == "WS2"

    def test_workspace_with_env_vars(self, db):
        ws = db.create_workspace(
            name="WS1",
            root_path="/tmp/ws1",
            env_vars={"API_KEY": "secret", "DEBUG": "1"},
        )
        assert ws.env_vars == {"API_KEY": "secret", "DEBUG": "1"}
        fetched = db.get_workspace(ws.id)
        assert fetched.env_vars == {"API_KEY": "secret", "DEBUG": "1"}

    def test_workspace_with_metadata(self, db):
        ws = db.create_workspace(
            name="WS1",
            root_path="/tmp/ws1",
            metadata={"framework": "fastapi", "version": "0.100"},
        )
        assert ws.metadata["framework"] == "fastapi"

    def test_get_workspace_by_canonical_workspace_id(self, db):
        db.create_workspace(name="Other", root_path="/tmp/other")
        linked = db.create_workspace(
            name="Linked",
            root_path="/tmp/linked",
            metadata={
                "canonical_workspace_id": "workspace-alpha",
                "canonical_workspace_source": "workspace_playground",
                "link_status": "linked",
            },
        )

        found = db.get_workspace_by_canonical_workspace_id("workspace-alpha")

        assert found is not None
        assert found.id == linked.id
        assert db.get_workspace_by_canonical_workspace_id("workspace-missing") is None

    def test_duplicate_canonical_workspace_id_raises(self, db):
        db.create_workspace(
            name="Linked A",
            root_path="/tmp/linked-a",
            metadata={"canonical_workspace_id": "workspace-alpha"},
        )

        with pytest.raises(ValueError, match="canonical_workspace_id"):
            db.create_workspace(
                name="Linked B",
                root_path="/tmp/linked-b",
                metadata={"canonical_workspace_id": "workspace-alpha"},
            )

    def test_link_workspace_to_canonical_merges_metadata_without_duplication(self, db):
        ws = db.create_workspace(
            name="Existing Root",
            root_path="/tmp/existing-root",
            metadata={"existing": "kept"},
        )

        linked = db.link_workspace_to_canonical(
            ws.id,
            canonical_workspace_id="workspace-alpha",
            canonical_workspace_source="workspace_playground",
        )

        assert linked.id == ws.id
        assert linked.metadata == {
            "existing": "kept",
            "canonical_workspace_id": "workspace-alpha",
            "canonical_workspace_source": "workspace_playground",
            "link_status": "linked",
        }
        assert len(db.list_workspaces()) == 1
        assert db.get_workspace_by_canonical_workspace_id("workspace-alpha").id == ws.id

    def test_link_workspace_to_canonical_rejects_duplicate_canonical_id(self, db):
        first = db.create_workspace(name="First", root_path="/tmp/first")
        second = db.create_workspace(name="Second", root_path="/tmp/second")
        db.link_workspace_to_canonical(
            first.id,
            canonical_workspace_id="workspace-alpha",
            canonical_workspace_source="workspace_playground",
        )

        with pytest.raises(ValueError, match="already linked"):
            db.link_workspace_to_canonical(
                second.id,
                canonical_workspace_id="workspace-alpha",
                canonical_workspace_source="workspace_playground",
            )

    def test_link_workspace_to_canonical_rejects_overwriting_existing_link(self, db):
        ws = db.create_workspace(name="Existing Root", root_path="/tmp/existing-root")
        db.link_workspace_to_canonical(
            ws.id,
            canonical_workspace_id="workspace-alpha",
            canonical_workspace_source="workspace_playground",
        )

        with pytest.raises(
            CanonicalWorkspaceBridgeConflictError,
            match="different canonical workspace",
        ):
            db.link_workspace_to_canonical(
                ws.id,
                canonical_workspace_id="workspace-beta",
                canonical_workspace_source="workspace_playground",
            )

        assert (
            db.get_workspace_by_canonical_workspace_id("workspace-alpha").id == ws.id
        )
        assert db.get_workspace_by_canonical_workspace_id("workspace-beta") is None

    def test_workspace_with_git_info(self, db):
        ws = db.create_workspace(
            name="WS1",
            root_path="/tmp/ws1",
            git_remote_url="https://github.com/user/repo.git",
            git_default_branch="main",
            git_current_branch="feature/x",
            git_is_dirty=True,
        )
        assert ws.git_remote_url == "https://github.com/user/repo.git"
        assert ws.git_is_dirty is True
        fetched = db.get_workspace(ws.id)
        assert fetched.git_is_dirty is True


# ---------------------------------------------------------------------------
# Workspace health
# ---------------------------------------------------------------------------


class TestWorkspaceHealth:
    def test_update_workspace_health(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        updated = db.update_workspace_health(
            ws.id,
            health_status="healthy",
            git_remote_url="https://github.com/user/repo.git",
            git_current_branch="main",
            git_is_dirty=False,
        )
        assert updated.health_status == "healthy"
        assert updated.git_remote_url == "https://github.com/user/repo.git"
        assert updated.git_is_dirty is False
        assert updated.last_health_check is not None

    def test_update_workspace_health_not_found(self, db):
        with pytest.raises(OrchestrationNotFoundError):
            db.update_workspace_health(999, health_status="missing")


# ---------------------------------------------------------------------------
# Workspace parent/children
# ---------------------------------------------------------------------------


class TestWorkspaceHierarchy:
    def test_parent_child_relationship(self, db):
        parent = db.create_workspace(name="Monorepo", root_path="/tmp/mono")
        child = db.create_workspace(
            name="Frontend",
            root_path="/tmp/mono/frontend",
            workspace_type="monorepo_child",
            parent_workspace_id=parent.id,
        )
        assert child.parent_workspace_id == parent.id
        children = db.list_workspace_children(parent.id)
        assert len(children) == 1
        assert children[0].name == "Frontend"

    def test_parent_not_found(self, db):
        with pytest.raises(OrchestrationNotFoundError, match="Parent workspace"):
            db.create_workspace(
                name="Orphan",
                root_path="/tmp/orphan",
                parent_workspace_id=999,
            )

    def test_delete_parent_sets_null(self, db):
        parent = db.create_workspace(name="Parent", root_path="/tmp/parent")
        child = db.create_workspace(
            name="Child",
            root_path="/tmp/child",
            parent_workspace_id=parent.id,
        )
        db.delete_workspace(parent.id)
        updated_child = db.get_workspace(child.id)
        assert updated_child is not None
        assert updated_child.parent_workspace_id is None


# ---------------------------------------------------------------------------
# Workspace MCP servers
# ---------------------------------------------------------------------------


class TestWorkspaceMCPServers:
    def test_create_mcp_server(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        server = db.create_workspace_mcp_server(
            workspace_id=ws.id,
            server_name="my-mcp",
            server_type="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={"HOME": "/tmp"},
        )
        assert server["server_name"] == "my-mcp"
        assert server["command"] == "npx"
        assert server["args"] == ["-y", "@modelcontextprotocol/server-filesystem"]

    def test_list_mcp_servers(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_workspace_mcp_server(ws.id, "server-a", command="cmd-a")
        db.create_workspace_mcp_server(ws.id, "server-b", command="cmd-b")
        servers = db.list_workspace_mcp_servers(ws.id)
        assert len(servers) == 2

    def test_delete_mcp_server(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        server = db.create_workspace_mcp_server(ws.id, "server-a", command="cmd")
        assert db.delete_workspace_mcp_server(ws.id, server["id"]) is True
        assert len(db.list_workspace_mcp_servers(ws.id)) == 0

    def test_delete_mcp_server_not_found(self, db):
        assert db.delete_workspace_mcp_server(999, 999) is False

    def test_delete_mcp_server_wrong_workspace(self, db):
        ws1 = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        ws2 = db.create_workspace(name="WS2", root_path="/tmp/ws2")
        server = db.create_workspace_mcp_server(ws1.id, "server-a", command="cmd")
        # Should not delete a server from wrong workspace
        assert db.delete_workspace_mcp_server(ws2.id, server["id"]) is False
        # Server still exists
        assert len(db.list_workspace_mcp_servers(ws1.id)) == 1

    def test_duplicate_server_name_raises(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_workspace_mcp_server(ws.id, "server-a", command="cmd")
        with pytest.raises(ValueError, match="already exists"):
            db.create_workspace_mcp_server(ws.id, "server-a", command="cmd2")

    def test_workspace_not_found_for_mcp_server(self, db):
        with pytest.raises(OrchestrationNotFoundError):
            db.create_workspace_mcp_server(999, "server-a", command="cmd")

    def test_cascade_delete_mcp_servers(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_workspace_mcp_server(ws.id, "server-a", command="cmd")
        db.delete_workspace(ws.id)
        # MCP servers should be gone (CASCADE)
        assert len(db.list_workspace_mcp_servers(ws.id)) == 0


# ---------------------------------------------------------------------------
# Project-Workspace FK binding
# ---------------------------------------------------------------------------


class TestProjectWorkspaceBinding:
    def test_create_project_with_workspace(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        project = db.create_project(name="P1", workspace_id=ws.id)
        assert project.workspace_id == ws.id

    def test_create_project_invalid_workspace(self, db):
        with pytest.raises(OrchestrationNotFoundError, match="Workspace"):
            db.create_project(name="P1", workspace_id=999)

    def test_create_project_without_workspace(self, db):
        project = db.create_project(name="P1")
        assert project.workspace_id is None

    def test_delete_workspace_sets_project_null(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        project = db.create_project(name="P1", workspace_id=ws.id)
        db.delete_workspace(ws.id)
        updated = db.get_project(project.id)
        assert updated is not None
        assert updated.workspace_id is None

    def test_list_projects_no_filter(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_project(name="P1", workspace_id=ws.id)
        db.create_project(name="P2")
        # Default (no filter) returns all
        projects = db.list_projects()
        assert len(projects) == 2

    def test_list_projects_filter_by_workspace(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_project(name="P1", workspace_id=ws.id)
        db.create_project(name="P2")
        # Filter by workspace_id returns only bound projects
        bound = db.list_projects(workspace_id=ws.id)
        assert len(bound) == 1
        assert bound[0].name == "P1"

    def test_list_projects_filter_unbound(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        db.create_project(name="P1", workspace_id=ws.id)
        db.create_project(name="P2")
        # Filter by workspace_id=None returns only unbound projects
        unbound = db.list_projects(workspace_id=None)
        assert len(unbound) == 1
        assert unbound[0].name == "P2"

    def test_project_to_dict_includes_workspace_id(self, db):
        ws = db.create_workspace(name="WS1", root_path="/tmp/ws1")
        project = db.create_project(name="P1", workspace_id=ws.id)
        d = project.to_dict()
        assert d["workspace_id"] == ws.id
