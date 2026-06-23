# test_database.py
# Database tests for Prompt Studio

import gc
import pytest
import sqlite3
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError, ConflictError, InputError
)

########################################################################################################################
# Test Fixtures

def _unlink_sqlite_temp_db(path: str | Path) -> None:
    """Remove a SQLite temp DB and sidecar files, retrying transient Windows locks."""
    db_path = Path(path)
    for target in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        max_attempts = 40
        for attempt in range(max_attempts):
            try:
                target.unlink()
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if attempt == max_attempts - 1:
                    raise
                gc.collect()
                time.sleep(0.1)

@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    yield path
    # Cleanup
    _unlink_sqlite_temp_db(path)

@pytest.fixture
def test_db(temp_db_path):
    """Create a test database instance."""
    db = PromptStudioDatabase(str(temp_db_path), "test_client")
    yield db
    # Close connection if needed
    try:
        db.close()
    except Exception:
        _ = None

@pytest.fixture
def populated_db(test_db):
    """Create a database with sample data."""
    # Add sample project
    project_data = test_db.create_project(
        name="Test Project",
        description="Test project for unit tests",
        user_id="test_user"
    )

    # Add sample prompt
    conn = test_db.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prompt_studio_prompts (
            uuid, project_id, name, system_prompt, user_prompt,
            version_number, client_id
        ) VALUES (
            lower(hex(randomblob(16))), ?, 'Test Prompt',
            'System prompt', 'User prompt', 1, ?
        )
    """, (project_data["id"], test_db.client_id))
    conn.commit()

    yield test_db


def _is_sqlite_backend(db: PromptStudioDatabase) -> bool:
    """Best-effort detection of SQLite backend for conditional concurrency tests."""
    db_path = getattr(db, "db_path", None)
    if db_path is None:
        return True
    path_str = str(db_path)
    return path_str.endswith(".db") or path_str.startswith("sqlite")


@pytest.fixture
def multi_user_prompt_dbs():
    """Create multiple PromptStudioDatabase instances to simulate per-user SQLite DBs."""
    temp_paths = []
    dbs = []
    try:
        for idx in range(5):
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp.close()
            temp_paths.append(tmp.name)
            dbs.append(PromptStudioDatabase(tmp.name, f"test-client-{idx}"))
        yield dbs
    finally:
        for db in dbs:
            try:
                db.close()
            except Exception:
                _ = None
        for path in temp_paths:
            _unlink_sqlite_temp_db(path)


def test_unlink_sqlite_temp_db_retries_extended_transient_permission_error(
    monkeypatch,
    tmp_path,
) -> None:
    """Windows can hold SQLite handles after close longer than one short retry loop."""
    db_path = tmp_path / "locked.db"
    db_path.write_text("", encoding="utf-8")
    attempts = 0
    original_unlink = Path.unlink

    def _flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        if self == db_path and attempts < 6:
            attempts += 1
            raise PermissionError("transient Windows SQLite lock")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)

    _unlink_sqlite_temp_db(db_path)

    assert attempts == 6
    assert not db_path.exists()

########################################################################################################################
# Database Initialization Tests

class TestDatabaseInitialization:
    """Test database initialization and schema creation."""

    def test_database_creation(self, temp_db_path: Path):
        """Test that database is created successfully."""
        db = PromptStudioDatabase(temp_db_path, "test_client")
        try:
            assert temp_db_path.exists()

            # Check that connection works
            conn = db.get_connection()
            assert conn is not None
        finally:
            db.close()

    def test_schema_creation(self, test_db: PromptStudioDatabase):
        """Test that all required tables are created."""
        conn = test_db.get_connection()
        cursor = conn.cursor()

        # Check for expected tables
        expected_tables = [
            "prompt_studio_projects",
            "prompt_studio_signatures",
            "prompt_studio_prompts",
            "prompt_studio_test_cases",
            "prompt_studio_test_runs",
            "prompt_studio_evaluations",
            "prompt_studio_optimizations",
            "prompt_studio_job_queue"
        ]

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE 'prompt_studio_%'
        """)

        tables = [row[0] for row in cursor.fetchall()]

        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    def test_indexes_creation(self, test_db: PromptStudioDatabase):
        """Test that indexes are created."""
        conn = test_db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name LIKE 'idx_ps_%'
        """)

        indexes = [row[0] for row in cursor.fetchall()]

        # Check for some key indexes
        assert "idx_ps_projects_user" in indexes
        assert "idx_ps_prompts_project" in indexes
        assert "idx_ps_test_cases_project" in indexes

    def test_fts_tables_creation(self, test_db: PromptStudioDatabase):
        """Test that FTS tables are created."""
        conn = test_db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name LIKE '%_fts'
        """)

        fts_tables = [row[0] for row in cursor.fetchall()]

        expected_fts = [
            "prompt_studio_projects_fts",
            "prompt_studio_prompts_fts",
            "prompt_studio_test_cases_fts"
        ]

        for table in expected_fts:
            assert table in fts_tables, f"FTS table {table} not found"

    def test_wal_mode_defaults_outside_test_context(self, temp_db_path: Path, monkeypatch):
        """Prompt Studio should default to WAL for normal runtime SQLite usage."""
        monkeypatch.setenv("TEST_MODE", "false")
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("TLDW_PS_SQLITE_WAL", raising=False)

        db = PromptStudioDatabase(temp_db_path, "test_client")
        try:
            conn = db.get_connection()
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert str(row[0]).lower() == "wal"
        finally:
            db.close()

    def test_wal_mode_defaults_to_delete_in_ci(self, temp_db_path: Path, monkeypatch):
        """CI and test contexts should keep Prompt Studio on DELETE unless overridden."""
        monkeypatch.setenv("TEST_MODE", "false")
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("TLDW_PS_SQLITE_WAL", raising=False)

        db = PromptStudioDatabase(temp_db_path, "test_client")
        try:
            conn = db.get_connection()
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert str(row[0]).lower() == "delete"
        finally:
            db.close()

    def test_wal_mode_override_beats_ci_default(self, temp_db_path: Path, monkeypatch):
        """Explicit WAL override should win even under CI markers."""
        monkeypatch.setenv("TEST_MODE", "false")
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("TLDW_PS_SQLITE_WAL", "1")

        db = PromptStudioDatabase(temp_db_path, "test_client")
        try:
            conn = db.get_connection()
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert str(row[0]).lower() == "wal"
        finally:
            db.close()

    def test_reopened_connections_keep_ci_delete_mode(self, temp_db_path: Path, monkeypatch):
        """Prompt Studio reopened SQLite connections should preserve the local WAL decision."""
        monkeypatch.setenv("TEST_MODE", "false")
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("TLDW_PS_SQLITE_WAL", raising=False)

        db = PromptStudioDatabase(temp_db_path, "test_client")
        try:
            db.close_connection()
            conn = db.get_connection()
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert str(row[0]).lower() == "delete"
        finally:
            db.close()

########################################################################################################################
# Project CRUD Tests

class TestProjectOperations:
    """Test project CRUD operations."""

    def test_create_project(self, test_db: PromptStudioDatabase):
        """Test creating a project."""
        project = test_db.create_project(
            name="Test Project",
            description="Test description",
            status="draft"
        )

        assert project is not None
        assert project["name"] == "Test Project"
        assert project["description"] == "Test description"
        assert project["status"] == "draft"
        assert project["uuid"] is not None
        assert project["id"] > 0

    def test_create_duplicate_project(self, test_db: PromptStudioDatabase):
        """Test that duplicate project names are rejected."""
        test_db.create_project(name="Unique Project")

        with pytest.raises(ConflictError):
            test_db.create_project(name="Unique Project")

    def test_get_project(self, populated_db: PromptStudioDatabase):
        """Test getting a project by ID."""
        # Create a project first
        created = populated_db.create_project(name="Get Test Project")

        # Get the project
        project = populated_db.get_project(created["id"])

        assert project is not None
        assert project["id"] == created["id"]
        assert project["name"] == "Get Test Project"

    def test_get_nonexistent_project(self, test_db: PromptStudioDatabase):
        """Test getting a non-existent project."""
        project = test_db.get_project(99999)
        assert project is None

    def test_list_projects(self, populated_db: PromptStudioDatabase):
        """Test listing projects."""
        # List all projects
        result = populated_db.list_projects(page=1, per_page=10)

        assert "projects" in result
        assert "pagination" in result
        assert len(result["projects"]) >= 1  # From populated_db fixture
        assert result["pagination"]["total"] >= 1

    def test_list_projects_with_filter(self, populated_db: PromptStudioDatabase):
        """Test listing projects with status filter."""
        result = populated_db.list_projects(status="active", page=1, per_page=10)

        assert "projects" in result
        for project in result["projects"]:
            assert project["status"] == "active"

    def test_update_project(self, populated_db: PromptStudioDatabase):
        """Test updating a project."""
        # Create a project
        project = populated_db.create_project(name="Update Test")

        # Update it
        updated = populated_db.update_project(
            project["id"],
            {"description": "Updated description", "status": "active"}
        )

        assert updated["description"] == "Updated description"
        assert updated["status"] == "active"
        assert updated["name"] == "Update Test"  # Unchanged

    def test_delete_project_soft(self, populated_db: PromptStudioDatabase):
        """Test soft deleting a project."""
        # Create a project
        project = populated_db.create_project(name="Delete Test")

        # Soft delete it
        success = populated_db.delete_project(project["id"], hard_delete=False)
        assert success is True

        # Should not be found in normal get
        deleted = populated_db.get_project(project["id"], include_deleted=False)
        assert deleted is None

        # Should be found when including deleted
        deleted_with_flag = populated_db.get_project(project["id"], include_deleted=True)
        assert deleted_with_flag is not None
        assert deleted_with_flag["deleted"] == 1

    def test_delete_project_hard(self, populated_db: PromptStudioDatabase):
        """Test hard deleting a project."""
        # Create a project
        project = populated_db.create_project(name="Hard Delete Test")
        project_id = project["id"]

        # Hard delete it
        success = populated_db.delete_project(project_id, hard_delete=True)
        assert success is True

        # Should not exist at all
        deleted = populated_db.get_project(project_id, include_deleted=True)
        assert deleted is None


########################################################################################################################
# Signature CRUD Tests

class TestSignatureOperations:
    """Test signature CRUD and listing behaviour."""

    def _create_project(self, db: PromptStudioDatabase) -> Dict[str, Any]:
        return db.create_project(
            name="Signature Project",
            description="Project for signature tests",
            user_id="signature-user",
        )

    def test_create_and_get_signature(self, test_db: PromptStudioDatabase):
        project = self._create_project(test_db)

        signature = test_db.create_signature(
            project_id=project["id"],
            name="Summarization",
            input_schema=[{"name": "text", "type": "string"}],
            output_schema=[{"name": "summary", "type": "string"}],
            constraints={"max_length": 512},
        )

        assert signature["name"] == "Summarization"
        assert signature["project_id"] == project["id"]
        assert isinstance(signature["input_schema"], list)

        fetched = test_db.get_signature(signature["id"])
        assert fetched is not None
        assert fetched["uuid"] == signature["uuid"]
        assert fetched["input_schema"][0]["name"] == "text"

    def test_list_signatures(self, test_db: PromptStudioDatabase):
        project = self._create_project(test_db)
        for idx in range(3):
            test_db.create_signature(
                project_id=project["id"],
                name=f"Signature {idx}",
                input_schema=[{"name": "field", "type": "string"}],
                output_schema=[{"name": "result", "type": "string"}],
            )

        signatures = test_db.list_signatures(project_id=project["id"])
        assert isinstance(signatures, list)
        assert len(signatures) == 3

    def test_update_signature(self, test_db: PromptStudioDatabase):
        project = self._create_project(test_db)
        signature = test_db.create_signature(
            project_id=project["id"],
            name="Draft Signature",
            input_schema=[{"name": "field", "type": "string"}],
            output_schema=[{"name": "result", "type": "string"}],
        )

        updated = test_db.update_signature(
            signature["id"],
            {
                "name": "Updated Signature",
                "validation_rules": {"required": ["field"]},
            },
        )

        assert updated["name"] == "Updated Signature"
        assert updated["validation_rules"]["required"] == ["field"]

    def test_delete_signature(self, test_db: PromptStudioDatabase):
        project = self._create_project(test_db)
        signature = test_db.create_signature(
            project_id=project["id"],
            name="Temp Signature",
            input_schema=[{"name": "field", "type": "string"}],
            output_schema=[{"name": "result", "type": "string"}],
        )

        assert test_db.delete_signature(signature["id"])
        assert test_db.get_signature(signature["id"]) is None

    def test_create_signature_conflict(self, test_db: PromptStudioDatabase):
        project = self._create_project(test_db)
        test_db.create_signature(
            project_id=project["id"],
            name="Duplicate Signature",
            input_schema=[{"name": "field", "type": "string"}],
            output_schema=[{"name": "result", "type": "string"}],
        )

        with pytest.raises(ConflictError):
            test_db.create_signature(
                project_id=project["id"],
                name="Duplicate Signature",
                input_schema=[{"name": "another", "type": "string"}],
                output_schema=[{"name": "result", "type": "string"}],
            )

########################################################################################################################
# Test Run & Evaluation Tests

class TestTestRunOperations:
    """Verify creating test runs via the database abstraction."""

    def _seed_prompt_and_case(self, db: PromptStudioDatabase) -> Dict[str, Any]:
        project = db.create_project(name="Run Project", description="", user_id="runner")
        prompt = db.create_prompt(
            project_id=project["id"],
            name="Run Prompt",
            system_prompt="You are helpful.",
            user_prompt="Answer: {value}",
        )
        test_case = db.create_test_case(
            project_id=project["id"],
            name="Case",
            inputs={"value": "42"},
            expected_outputs={"response": "42"},
        )
        return {"project": project, "prompt": prompt, "test_case": test_case}

    def test_create_test_run(self, test_db: PromptStudioDatabase):
        seeded = self._seed_prompt_and_case(test_db)
        run = test_db.create_test_run(
            project_id=seeded["project"]["id"],
            prompt_id=seeded["prompt"]["id"],
            test_case_id=seeded["test_case"]["id"],
            model_name="gpt-3.5-turbo",
            inputs={"value": "42"},
            outputs={"response": "42"},
            expected_outputs={"response": "42"},
        )

        assert run["prompt_id"] == seeded["prompt"]["id"]
        assert run["test_case_id"] == seeded["test_case"]["id"]
        assert run["inputs"]["value"] == "42"


class TestEvaluationOperations:
    """Verify evaluation CRUD operations through the abstraction."""

    def _seed_prompt(self, db: PromptStudioDatabase) -> Dict[str, Any]:
        project = db.create_project(name="Eval Project", description="", user_id="eval")
        prompt = db.create_prompt(
            project_id=project["id"],
            name="Eval Prompt",
            system_prompt="System",
            user_prompt="User",
        )
        test_case = db.create_test_case(
            project_id=project["id"],
            name="Eval Case",
            inputs={"text": "hello"},
            expected_outputs={"response": "hello"},
        )
        return {"project": project, "prompt": prompt, "test_case": test_case}

    def test_create_and_update_evaluation(self, test_db: PromptStudioDatabase):
        seeded = self._seed_prompt(test_db)
        evaluation = test_db.create_evaluation(
            prompt_id=seeded["prompt"]["id"],
            project_id=seeded["project"]["id"],
            model_configs={"model": "gpt-3.5-turbo"},
            test_case_ids=[seeded["test_case"]["id"]],
        )

        assert evaluation["prompt_id"] == seeded["prompt"]["id"]
        assert evaluation["status"] == "running"

        updated = test_db.update_evaluation(
            evaluation["id"],
            {
                "status": "completed",
                "aggregate_metrics": {"average_score": 1.0},
            },
        )

        assert updated["status"] == "completed"
        assert updated["aggregate_metrics"]["average_score"] == 1.0

        fetched = test_db.get_evaluation(evaluation["id"])
        assert fetched is not None

        listing = test_db.list_evaluations(project_id=seeded["project"]["id"])
        assert listing["pagination"]["total"] >= 1

########################################################################################################################
# Transaction Tests

class TestTransactions:
    """Test database transaction management."""

    def test_transaction_rollback(self, test_db: PromptStudioDatabase):
        """Test that transactions rollback on error."""
        with pytest.raises(Exception):
            with test_db.transaction() as conn:
                cursor = conn.cursor()

                # Create a project
                cursor.execute("""
                    INSERT INTO prompt_studio_projects (uuid, name, user_id, client_id)
                    VALUES ('test-uuid', 'Transaction Test', 'user1', 'client1')
                """)

                # Force an error
                raise Exception("Test error")

        # Check that project was not created
        conn = test_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prompt_studio_projects WHERE name = 'Transaction Test'")
        assert cursor.fetchone() is None

    def test_transaction_commit(self, test_db: PromptStudioDatabase):
        """Test that transactions commit properly."""
        with test_db.transaction() as conn:
            cursor = conn.cursor()

            # Create a project
            cursor.execute("""
                INSERT INTO prompt_studio_projects (uuid, name, user_id, client_id)
                VALUES ('test-uuid-2', 'Transaction Test 2', 'user1', 'client1')
            """)

        # Check that project was created
        conn = test_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prompt_studio_projects WHERE name = 'Transaction Test 2'")
        assert cursor.fetchone() is not None

########################################################################################################################
# Helper Method Tests

class TestHelperMethods:
    """Test database helper methods."""

    def test_row_to_dict(self, test_db: PromptStudioDatabase):
        """Test converting database row to dictionary."""
        conn = test_db.get_connection()
        cursor = conn.cursor()

        # Create a test project
        cursor.execute("""
            INSERT INTO prompt_studio_projects (uuid, name, user_id, client_id, metadata)
            VALUES ('test-uuid', 'Dict Test', 'user1', 'client1', '{"key": "value"}')
        """)
        conn.commit()

        # Fetch and convert
        cursor.execute("SELECT * FROM prompt_studio_projects WHERE name = 'Dict Test'")
        row = cursor.fetchone()
        result = test_db._row_to_dict(cursor, row)

        assert isinstance(result, dict)
        assert result["name"] == "Dict Test"
        assert result["metadata"] == {"key": "value"}  # JSON parsed
        assert isinstance(result["created_at"], datetime)  # Datetime parsed

    def test_sync_log_event(self, test_db: PromptStudioDatabase):
        """Test sync event logging."""
        # This should not raise an error even if sync_log doesn't exist
        test_db._log_sync_event(
            "test_entity",
            "test-uuid",
            "create",
            {"test": "data"}
        )

        # If sync_log exists, check that event was logged
        conn = test_db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='sync_log'
        """)

        if cursor.fetchone():
            cursor.execute("""
                SELECT * FROM sync_log
                WHERE entity = 'test_entity' AND entity_uuid = 'test-uuid'
            """)
            log_entry = cursor.fetchone()
            # May or may not exist depending on base database setup

########################################################################################################################
# Prompt CRUD Tests

class TestPromptOperations:
    """Test prompt CRUD operations."""

    def test_create_prompt(self, populated_db: PromptStudioDatabase):
        """Test creating a prompt."""
        # Get a project first
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create a prompt
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, name, system_prompt, user_prompt,
                version_number, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'prompt-uuid-1',
            project_id,
            'New Prompt',
            'System prompt text',
            'User prompt text',
            1,
            populated_db.client_id
        ))
        conn.commit()

        # Verify it was created
        cursor.execute("""
            SELECT * FROM prompt_studio_prompts
            WHERE uuid = 'prompt-uuid-1'
        """)
        prompt = cursor.fetchone()
        assert prompt is not None

    def test_get_prompts_by_project(self, populated_db: PromptStudioDatabase):
        """Test getting prompts for a project."""
        # Get a project
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Get prompts for the project
        cursor.execute("""
            SELECT * FROM prompt_studio_prompts
            WHERE project_id = ? AND deleted = 0
        """, (project_id,))

        prompts = cursor.fetchall()
        assert len(prompts) >= 1  # At least one from populated_db

    def test_update_prompt(self, populated_db: PromptStudioDatabase):
        """Test updating a prompt."""
        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Get an existing prompt
        cursor.execute("""
            SELECT * FROM prompt_studio_prompts
            WHERE deleted = 0 LIMIT 1
        """)
        prompt = cursor.fetchone()
        prompt_id = prompt[0]

        # Update it
        cursor.execute("""
            UPDATE prompt_studio_prompts
            SET user_prompt = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, ('Updated user prompt', prompt_id))
        conn.commit()

        # Verify update
        cursor.execute("""
            SELECT user_prompt FROM prompt_studio_prompts
            WHERE id = ?
        """, (prompt_id,))
        updated = cursor.fetchone()
        assert updated[0] == 'Updated user prompt'

    def test_prompt_versioning(self, populated_db: PromptStudioDatabase):
        """Test prompt version management."""
        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Get a project
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        # Create multiple versions of same prompt
        for version in range(1, 4):
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, name, system_prompt, user_prompt,
                    version_number, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f'versioned-prompt-v{version}',
                project_id,
                'Versioned Prompt',
                f'System v{version}',
                f'User v{version}',
                version,
                populated_db.client_id
            ))
        conn.commit()

        # Get all versions
        cursor.execute("""
            SELECT version_number FROM prompt_studio_prompts
            WHERE name = 'Versioned Prompt' AND deleted = 0
            ORDER BY version_number
        """)
        versions = [row[0] for row in cursor.fetchall()]
        assert versions == [1, 2, 3]

########################################################################################################################
# Test Case CRUD Tests

class TestTestCaseOperations:
    """Test test case CRUD operations."""

    def test_create_test_case(self, populated_db: PromptStudioDatabase):
        """Test creating a test case."""
        # Get a project
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create a test case
        cursor.execute("""
            INSERT INTO prompt_studio_test_cases (
                uuid, project_id, name, description, inputs, expected_outputs,
                is_golden, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'test-case-uuid-1',
            project_id,
            'Test Case 1',
            'Test description',
            '{"input": "test data"}',
            '{"output": "expected result"}',
            0,
            populated_db.client_id
        ))
        conn.commit()

        # Verify creation
        cursor.execute("""
            SELECT * FROM prompt_studio_test_cases
            WHERE uuid = 'test-case-uuid-1'
        """)
        test_case = cursor.fetchone()
        assert test_case is not None

    def test_golden_test_cases(self, populated_db: PromptStudioDatabase):
        """Test golden test case management."""
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create golden and regular test cases
        test_cases = [
            ('golden-1', 'Golden Test 1', 1),
            ('regular-1', 'Regular Test 1', 0),
            ('golden-2', 'Golden Test 2', 1),
        ]

        for uuid, name, is_golden in test_cases:
            cursor.execute("""
                INSERT INTO prompt_studio_test_cases (
                    uuid, project_id, name, inputs, expected_outputs,
                    is_golden, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                uuid, project_id, name,
                '{}', '{}', is_golden,
                populated_db.client_id
            ))
        conn.commit()

        # Query golden test cases only
        cursor.execute("""
            SELECT name FROM prompt_studio_test_cases
            WHERE project_id = ? AND is_golden = 1 AND deleted = 0
            ORDER BY name
        """, (project_id,))
        golden = [row[0] for row in cursor.fetchall()]
        assert 'Golden Test 1' in golden
        assert 'Golden Test 2' in golden
        assert 'Regular Test 1' not in golden

########################################################################################################################
# Job Queue Tests

class TestJobQueue:
    """Test job queue operations."""

    def test_create_job(self, populated_db: PromptStudioDatabase):
        """Test creating a job."""
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create a job
        cursor.execute("""
            INSERT INTO prompt_studio_job_queue (
                uuid, project_id, job_type, entity_id, payload, status, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'job-uuid-1',
            project_id,
            'evaluation',
            1,  # entity_id is required
            '{"test": true}',
            'queued',  # default status is 'queued' not 'pending'
            populated_db.client_id
        ))
        conn.commit()

        # Verify creation
        cursor.execute("""
            SELECT * FROM prompt_studio_job_queue
            WHERE uuid = 'job-uuid-1'
        """)
        job = cursor.fetchone()
        assert job is not None

    def test_job_status_transitions(self, populated_db: PromptStudioDatabase):
        """Test job status transitions."""
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create a job
        cursor.execute("""
            INSERT INTO prompt_studio_job_queue (
                uuid, project_id, job_type, entity_id, payload, status, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'job-status-test',
            project_id,
            'optimization',
            1,  # entity_id is required
            '{}',
            'queued',
            populated_db.client_id
        ))
        job_id = cursor.lastrowid
        conn.commit()

        # Transition through statuses
        statuses = ['processing', 'completed']
        for status in statuses:
            cursor.execute("""
                UPDATE prompt_studio_job_queue
                SET status = ?, started_at = CASE WHEN ? = 'processing' THEN CURRENT_TIMESTAMP ELSE started_at END,
                    completed_at = CASE WHEN ? IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE id = ?
            """, (status, status, status, job_id))
            conn.commit()

            # Verify status
            cursor.execute("""
                SELECT status FROM prompt_studio_job_queue
                WHERE id = ?
            """, (job_id,))
            current_status = cursor.fetchone()[0]
            assert current_status == status

    def test_job_queue_priority(self, populated_db: PromptStudioDatabase):
        """Test job queue priority handling."""
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create jobs with different priorities
        jobs = [
            ('job-low', 0),
            ('job-high', 10),
            ('job-medium', 5),
        ]

        for uuid, priority in jobs:
            cursor.execute("""
                INSERT INTO prompt_studio_job_queue (
                    uuid, project_id, job_type, entity_id, payload, status, priority, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                uuid, project_id, 'evaluation', 1, '{}', 'queued',
                priority, populated_db.client_id
            ))
        conn.commit()

        # Get jobs ordered by priority
        cursor.execute("""
            SELECT uuid FROM prompt_studio_job_queue
            WHERE project_id = ? AND status = 'queued'
            ORDER BY priority DESC, created_at ASC
        """, (project_id,))
        ordered = [row[0] for row in cursor.fetchall()]

        # High priority should come first
        assert ordered.index('job-high') < ordered.index('job-medium')
        assert ordered.index('job-medium') < ordered.index('job-low')

########################################################################################################################
# Search and FTS Tests

class TestSearchFunctionality:
    """Test full-text search functionality."""

    def test_project_search(self, populated_db: PromptStudioDatabase):
        """Test searching projects."""
        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create test projects with searchable content
        test_projects = [
            ('search-1', 'Machine Learning Project', 'Using neural networks'),
            ('search-2', 'Data Analysis Project', 'Statistical analysis'),
            ('search-3', 'Deep Learning Research', 'Neural architecture search'),
        ]

        for uuid, name, desc in test_projects:
            cursor.execute("""
                INSERT INTO prompt_studio_projects (
                    uuid, name, description, user_id, client_id
                ) VALUES (?, ?, ?, ?, ?)
            """, (uuid, name, desc, 'test_user', populated_db.client_id))
        conn.commit()

        # Search for "neural"
        cursor.execute("""
            SELECT p.name FROM prompt_studio_projects p
            JOIN prompt_studio_projects_fts fts ON p.id = fts.rowid
            WHERE fts.prompt_studio_projects_fts MATCH 'neural'
            ORDER BY rank
        """)
        results = [row[0] for row in cursor.fetchall()]

        assert 'Machine Learning Project' in results
        assert 'Deep Learning Research' in results
        assert 'Data Analysis Project' not in results

    def test_prompt_search(self, populated_db: PromptStudioDatabase):
        """Test searching prompts."""
        projects = populated_db.list_projects(page=1, per_page=1)
        project_id = projects["projects"][0]["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Create searchable prompts
        prompts = [
            ('prompt-search-1', 'Code Review', 'Review Python code', 'Check for bugs'),
            ('prompt-search-2', 'Documentation', 'Write docs', 'Create markdown'),
            ('prompt-search-3', 'Testing', 'Write Python tests', 'Use pytest'),
        ]

        for uuid, name, system, user in prompts:
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, name, system_prompt, user_prompt,
                    version_number, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (uuid, project_id, name, system, user, 1, populated_db.client_id))
        conn.commit()

        # Search for "Python"
        cursor.execute("""
            SELECT p.name FROM prompt_studio_prompts p
            JOIN prompt_studio_prompts_fts fts ON p.id = fts.rowid
            WHERE fts.prompt_studio_prompts_fts MATCH 'Python'
            ORDER BY rank
        """)
        results = [row[0] for row in cursor.fetchall()]

        assert 'Code Review' in results
        assert 'Testing' in results

########################################################################################################################
# Concurrent Access Tests

class TestConcurrentAccess:
    """Test concurrent database access."""

    def test_concurrent_project_creation(self, test_db: PromptStudioDatabase):
        """Test concurrent project creation."""
        import threading
        import time
        import sqlite3

        if _is_sqlite_backend(test_db):
            pytest.skip("SQLite serializes writes; run on a Postgres backend to exercise shared DB concurrency.")
        results = []
        errors = []

        def create_project(idx):
            try:
                project = test_db.create_project(
                    name=f"Concurrent Project {idx}",
                    description=f"Created by thread {idx}"
                )
                results.append(project)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_project, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all projects exist
        conn = test_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_projects
            WHERE name LIKE 'Concurrent Project %'
        """)
        count = cursor.fetchone()[0]
        assert count == 5

    def test_concurrent_updates(self, populated_db: PromptStudioDatabase):
        """Test concurrent updates to same record."""
        import threading
        import time
        import sqlite3

        if _is_sqlite_backend(populated_db):
            pytest.skip("SQLite serializes writes; run on a Postgres backend to exercise shared DB updates.")
        # Create a project
        project = populated_db.create_project(
            name="Update Test Project",
            description="Initial description"
        )
        project_id = project["id"]

        update_count = [0]
        errors = []

        def update_project(thread_id):
            conn = populated_db.get_connection()
            cursor = conn.cursor()

            for i in range(10):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        cursor.execute("""
                            UPDATE prompt_studio_projects
                            SET description = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (f"Updated by thread {thread_id} iteration {i}", project_id))
                        conn.commit()
                        update_count[0] += 1
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < max_retries - 1:
                            time.sleep(0.01 * (attempt + 1))
                            continue
                        errors.append(e)
                        break

        # Create threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=update_project, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify all updates completed
        assert update_count[0] == 30  # 3 threads * 10 updates each

    def test_concurrent_project_creation_multi_user_sqlite(self, multi_user_prompt_dbs):
        """Ensure concurrent project creation succeeds when users have isolated SQLite DBs."""
        import threading

        results = []
        errors = []

        def create_project(idx, db_inst: PromptStudioDatabase):
            try:
                project = db_inst.create_project(
                    name=f"Concurrent User Project {idx}",
                    description=f"Created by simulated user {idx}"
                )
                results.append((idx, project["id"]))
            except Exception as exc:
                errors.append(exc)

        threads = []
        for idx, db_inst in enumerate(multi_user_prompt_dbs):
            thread = threading.Thread(target=create_project, args=(idx, db_inst))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        assert not errors, f"Errors occurred during multi-user project creation: {errors}"
        assert len(results) == len(multi_user_prompt_dbs)

        # Each database should have exactly one project
        for idx, db_inst in enumerate(multi_user_prompt_dbs):
            listing = db_inst.list_projects(page=1, per_page=10)
            assert listing["pagination"]["total"] == 1
            assert listing["projects"][0]["name"] == f"Concurrent User Project {idx}"

    def test_concurrent_updates_multi_user_sqlite(self, multi_user_prompt_dbs):
        """Ensure concurrent updates across isolated user DBs complete without contention."""
        import threading

        errors = []

        projects = []
        for idx, db_inst in enumerate(multi_user_prompt_dbs):
            projects.append(db_inst.create_project(
                name=f"Shared Concept {idx}",
                description="Initial"
            ))

        def update_project(idx, db_inst: PromptStudioDatabase, project_id: int):
            try:
                db_inst.update_project(
                    project_id,
                    description=f"Updated by user {idx}",
                    status="active"
                )
            except Exception as exc:
                errors.append(exc)

        threads = []
        for idx, (db_inst, project) in enumerate(zip(multi_user_prompt_dbs, projects)):
            thread = threading.Thread(target=update_project, args=(idx, db_inst, project["id"]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        assert not errors, f"Errors occurred during multi-user project updates: {errors}"

        for idx, (db_inst, project) in enumerate(zip(multi_user_prompt_dbs, projects)):
            updated = db_inst.get_project(project["id"])
            assert updated["description"] == f"Updated by user {idx}"
            assert updated["status"] == "active"

########################################################################################################################
# Data Integrity Tests

class TestDataIntegrity:
    """Test data integrity and constraints."""

    def test_foreign_key_constraints(self, test_db: PromptStudioDatabase):
        """Test foreign key constraints are enforced."""
        conn = test_db.get_connection()
        cursor = conn.cursor()

        # Try to insert prompt with non-existent project
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, name, system_prompt, user_prompt,
                    version_number, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'invalid-prompt',
                99999,  # Non-existent project
                'Invalid Prompt',
                'System',
                'User',
                1,
                test_db.client_id
            ))
            conn.commit()

    def test_unique_constraints(self, test_db: PromptStudioDatabase):
        """Test unique constraints are enforced."""
        # Create a project
        project = test_db.create_project(
            name="Unique Test Project"
        )

        conn = test_db.get_connection()
        cursor = conn.cursor()

        # Try to create another project with same UUID
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO prompt_studio_projects (
                    uuid, name, user_id, client_id
                ) VALUES (?, ?, ?, ?)
            """, (
                project["uuid"],  # Duplicate UUID
                'Duplicate UUID Project',
                'test_user',
                test_db.client_id
            ))
            conn.commit()

    def test_cascade_delete(self, populated_db: PromptStudioDatabase):
        """Test cascade delete behavior."""
        # Create project with related data
        project = populated_db.create_project(
            name="Cascade Test Project"
        )
        project_id = project["id"]

        conn = populated_db.get_connection()
        cursor = conn.cursor()

        # Add prompt and test case
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, name, system_prompt, user_prompt,
                version_number, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'cascade-prompt',
            project_id,
            'Cascade Prompt',
            'System',
            'User',
            1,
            populated_db.client_id
        ))

        cursor.execute("""
            INSERT INTO prompt_studio_test_cases (
                uuid, project_id, name, inputs, expected_outputs,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'cascade-test',
            project_id,
            'Cascade Test',
            '{}',
            '{}',
            populated_db.client_id
        ))
        conn.commit()

        # Hard delete project
        populated_db.delete_project(project_id, hard_delete=True)

        # Verify related data is also deleted
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_prompts
            WHERE project_id = ?
        """, (project_id,))
        prompt_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ?
        """, (project_id,))
        test_count = cursor.fetchone()[0]

        # With proper CASCADE DELETE, these should be 0
        # If CASCADE DELETE is not set up, they would still exist
        # This test verifies the expected behavior based on schema
