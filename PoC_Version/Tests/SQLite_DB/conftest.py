# tests/conftest.py
from contextlib import contextmanager

import pytest
import json
import tempfile
import sys
import os


# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tldw')))
print("Current sys.path:", sys.path)

from App_Function_Libraries.DB.SQLite_DB import Database

from pathlib import Path


def pytest_configure(config):
    # Get the directory of the current file (conftest.py)
    current_dir = Path(__file__).resolve().parent

    # Navigate to the root directory of your project
    project_root = current_dir.parent

    # Add the project root to sys.path
    sys.path.insert(0, str(project_root))

    # Set the working directory to the project root
    os.chdir(project_root)

    # Print debug information
    print(f"Project root set to: {project_root}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")

    # Check if the Workflows.json file exists
    workflows_path = project_root / 'Helper_Scripts' / 'Workflows' / 'Workflows.json'
    print(f"Checking for Workflows.json at: {workflows_path}")
    print(f"File exists: {workflows_path.exists()}")

    # If the file doesn't exist, list the contents of the directories
    if not workflows_path.exists():
        print("Contents of project root:")
        for item in project_root.iterdir():
            print(f"  {item}")

        helper_scripts_path = project_root / 'Helper_Scripts'
        if helper_scripts_path.exists():
            print("Contents of Helper_Scripts:")
            for item in helper_scripts_path.iterdir():
                print(f"  {item}")

def pytest_ignore_collect(path, config):
    return os.path.isdir(str(path)) and os.path.basename(str(path)) != "."

@contextmanager
def temp_db():
    _, db_path = tempfile.mkstemp(suffix='.db')
    db = Database(db_path)
    try:
        yield db
    finally:
        db.close()
        try:
            os.unlink(db_path)
        except PermissionError:
            print(f"Warning: Unable to delete temporary database file: {db_path}")

@pytest.fixture(scope="function")
def empty_db():
    # Create a temporary file to serve as the SQLite database
    _, db_path = tempfile.mkstemp(suffix='.db')
    db = Database(db_path)

    # Initialize the schema (create tables like Media)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # Example schema for the Media table
        cursor.execute('''
            CREATE TABLE Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                ingestion_date TEXT,
                transcription_model TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE DocumentVersions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                version_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(media_id) REFERENCES Media(id)
            )
        ''')
        conn.commit()

    # Yield the database for the tests
    yield db

    # Clean up and remove the temporary database file after the test
    try:
        os.unlink(db_path)
    except PermissionError:
        print(f"Warning: Unable to delete temporary database file: {db_path}")

@pytest.fixture
def mock_workflows_json(tmp_path):
    workflows_data = {
        "example_workflow": {
            "steps": ["step1", "step2"]
        }
    }
    mock_file = tmp_path / "Workflows.json"
    mock_file.write_text(json.dumps(workflows_data))
    return mock_file