# tests/conftest.py
from contextlib import contextmanager
import pytest
import json
import tempfile
import sys
import os
from pathlib import Path
import sqlite3 # Keep for potential type hints if needed

from tldw_Server_API.app.core.DB_Management.Media_DB import (
    Database,
    )
# Add the project root directory to sys.path
# (Assuming your tests folder is directly inside the project root)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
print(f"Project root added to sys.path: {project_root}")

# Dynamically find and import the Database class
# Adjust the path based on your actual project structure
try:
    # Example: Assuming Database is in project_root/app/db/database_setup.py
    from app.db.database_setup import Database, DatabaseError, InputError # Adjust import path as needed
    print("Successfully imported Database class.")
except ImportError as e:
    print(f"Error importing Database class: {e}")
    print("Please ensure the path to your Database class is correct relative to the project root.")
    # Optionally raise the error to halt tests if the class is essential
    # raise

# (Keep pytest_configure as is, it seems okay for path setup)
def pytest_configure(config):
    # Get the directory of the current file (conftest.py)
    current_dir = Path(__file__).resolve().parent
    # Navigate to the root directory of your project (assuming tests/ is in root)
    project_root = current_dir.parent
    # Add the project root to sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Set the working directory to the project root (optional but can help with relative paths)
    # os.chdir(project_root)
    print(f"Project root for tests: {project_root}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    # (Keep workflow check if needed)


@pytest.fixture(scope="function")
def db_instance() -> Database:
    """
    Provides a clean, temporary SQLite database instance for each test function.
    The schema is created by the Database class constructor.
    """
    # Create a temporary file path for the database
    _, db_path = tempfile.mkstemp(suffix='.db', prefix='test_db_')
    print(f"\nCreating test database: {db_path}")
    # Instantiate the Database class - this should create the schema
    db = Database(db_path)

    # Yield the database instance for the test
    yield db

    # Clean up: close connection and remove the temporary file
    print(f"Cleaning up test database: {db_path}")
    try:
        db.close_connection() # Ensure connection is closed via the Database class method
    except Exception as e:
        print(f"Warning: Error closing connection for {db_path}: {e}")

    # Attempt to delete the file
    try:
        os.unlink(db_path)
    except PermissionError:
        print(f"Warning: PermissionError deleting temporary database file: {db_path}. It might still be locked.")
    except FileNotFoundError:
         print(f"Warning: FileNotFoundError for {db_path}. Already deleted?")
    except Exception as e:
        print(f"Warning: Error deleting temporary database file {db_path}: {e}")


@pytest.fixture(scope="function")
def memory_db_instance() -> Database:
    """ Provides an in-memory SQLite database instance using the Database class. """
    print("\nCreating in-memory test database.")
    # Use ":memory:" for the path
    db = Database(":memory:")
    # Schema is created by __init__
    yield db
    # Cleanup for :memory: db is handled when the connection closes
    print("Cleaning up in-memory test database.")
    db.close_connection()


# (Keep mock_workflows_json if used elsewhere)
@pytest.fixture
def mock_workflows_json(tmp_path):
    workflows_data = {
        "example_workflow": {
            "steps": ["step1", "step2"]
        }
    }
    # Ensure parent directories exist if needed by your setup
    workflows_dir = tmp_path / 'Helper_Scripts' / 'Workflows'
    workflows_dir.mkdir(parents=True, exist_ok=True)
    mock_file = workflows_dir / "Workflows.json"
    mock_file.write_text(json.dumps(workflows_data))
    return mock_file