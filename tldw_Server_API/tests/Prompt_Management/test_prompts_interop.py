# tests/unit/core/Prompts_Management/test_prompts_interop.py
# Description:
#
# Imports
import pytest
from pathlib import Path
import os
#
# Local Imports
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    initialize_interop,
    shutdown_interop,
    get_db_instance,
    is_initialized,
    add_keyword as interop_add_keyword,
    add_prompt as interop_add_prompt,
    fetch_prompt_details as interop_fetch_prompt_details,
    # Import standalone wrappers
    add_or_update_prompt_interop,
    export_prompts_formatted_interop,
    DatabaseError,
    InputError
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
#
#######################################################################################################################
#
# Functions:
TEST_INTEROP_CLIENT_ID = "test_interop_client"

@pytest.fixture(scope="function") # function scope to ensure clean init/shutdown for each test
def interop_manager(tmp_path):
    """Manages initialization and shutdown of the interop layer for tests."""
    db_file = tmp_path / "interop_test_prompts.db"
    try:
        initialize_interop(db_path=db_file, client_id=TEST_INTEROP_CLIENT_ID)
        yield db_file # provide the path if needed by tests
    finally:
        shutdown_interop()
        if os.path.exists(db_file):
            os.remove(db_file)

def test_interop_initialization_and_shutdown(interop_manager):
    assert is_initialized() is True
    db_instance = get_db_instance()
    assert db_instance is not None
    assert isinstance(db_instance, PromptsDatabase)
    assert db_instance.client_id == TEST_INTEROP_CLIENT_ID

    shutdown_interop() # Explicitly call shutdown within test for this case
    assert is_initialized() is False
    with pytest.raises(RuntimeError, match="Prompts Interop Library not initialized"):
        get_db_instance()
    # Re-initialize for the fixture's finally block to work correctly
    initialize_interop(db_path=interop_manager, client_id=TEST_INTEROP_CLIENT_ID)


def test_interop_add_prompt_via_global_instance(interop_manager):
    assert is_initialized() is True # Ensure interop_manager fixture worked
    p_id, p_uuid, msg = interop_add_prompt(
        name="Interop Prompt", author="Interop", details="Via global instance"
    )
    assert p_id is not None
    assert "added" in msg

    details = interop_fetch_prompt_details(p_uuid)
    assert details is not None
    assert details['name'] == "Interop Prompt"

# --- Testing standalone wrapper functions from interop ---
# These take a db_instance, so we need to provide one.
# For these, the interop's global instance isn't directly used by the function itself,
# but the test setup might rely on `initialize_interop` if the function internally calls `get_db_instance`.
# The functions like `add_or_update_prompt_interop` DO use `get_db_instance()`.

def test_interop_standalone_add_or_update_prompt(interop_manager):
    p_id, _, msg = add_or_update_prompt_interop(
        name="Interop SU Prompt", author="SU", details="Details"
    )
    assert p_id is not None
    assert "added" in msg or "updated" in msg

    db_instance = get_db_instance() # Get the globally managed instance
    fetched = db_instance.get_prompt_by_name("Interop SU Prompt")
    assert fetched is not None
    assert fetched['author'] == "SU"

def test_interop_standalone_export_formatted(interop_manager):
    add_or_update_prompt_interop(name="Export Me Interop", author="Exporter", details="...")
    status_msg, file_path_str = export_prompts_formatted_interop(export_format='csv')

    assert "Successfully exported" in status_msg
    assert file_path_str != "None"
    if os.path.exists(file_path_str): # file_path_str is temp file path
        with open(file_path_str, 'r') as f:
            assert "Export Me Interop" in f.read()
        os.remove(file_path_str)
    else:
        # This might happen if the DB was in-memory and export didn't write to disk for some reason
        # or if the test setup of interop_manager used :memory: (it uses tmp_path now)
        pytest.fail(f"Exported file {file_path_str} not found.")

def test_interop_error_propagation(interop_manager):
    # Try to add a prompt with an empty name, should raise InputError from DB layer
    with pytest.raises(InputError):
        interop_add_prompt(name="", author="Test", details="...")

# Test calling get_db_instance when not initialized (outside fixture)
def test_get_db_instance_not_initialized():
    # Ensure it's shutdown if a previous test didn't clean up fully in some error case
    if is_initialized():
        shutdown_interop()
    with pytest.raises(RuntimeError, match="Prompts Interop Library not initialized"):
        get_db_instance()

#
# End of test_prompts_interop.py
#######################################################################################################################
