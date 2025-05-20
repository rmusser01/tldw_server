# tests/unit/core/Prompts_Management/test_prompts_db_v2.py
# Description:
#
# Imports
import pytest
import sqlite3
import uuid
from pathlib import Path
import os
import time # For testing last_modified updates
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    SchemaError,
    InputError,
    ConflictError,
    # Standalone functions
    add_or_update_prompt as standalone_add_or_update_prompt,
    load_prompt_details_for_ui as standalone_load_prompt_details_for_ui,
    export_prompt_keywords_to_csv as standalone_export_prompt_keywords_to_csv,
    view_prompt_keywords_markdown as standalone_view_prompt_keywords_markdown,
    export_prompts_formatted as standalone_export_prompts_formatted
)
#
########################################################################################################################
#
# Functions:

TEST_CLIENT_ID = "test_db_client"

@pytest.fixture
def memory_db():
    """Provides an in-memory PromptsDatabase instance for testing."""
    db = PromptsDatabase(db_path=":memory:", client_id=TEST_CLIENT_ID)
    yield db
    db.close_connection()

@pytest.fixture
def file_db(tmp_path):
    """Provides a file-based PromptsDatabase instance for testing."""
    db_file = tmp_path / "test_prompts.db"
    db = PromptsDatabase(db_path=db_file, client_id=TEST_CLIENT_ID)
    yield db
    db.close_connection()
    if os.path.exists(db_file):
        os.remove(db_file)

# --- Test PromptsDatabase Class ---

def test_database_initialization_memory(memory_db):
    assert memory_db is not None
    assert memory_db.client_id == TEST_CLIENT_ID
    assert memory_db.is_memory_db is True
    # Check if schema version table exists and has version 1
    conn = memory_db.get_connection()
    cursor = conn.execute("SELECT version FROM schema_version")
    assert cursor.fetchone()['version'] == PromptsDatabase._CURRENT_SCHEMA_VERSION

def test_database_initialization_file(file_db):
    assert file_db is not None
    assert file_db.client_id == TEST_CLIENT_ID
    assert file_db.is_memory_db is False
    assert os.path.exists(file_db.db_path)
    conn = file_db.get_connection()
    cursor = conn.execute("SELECT version FROM schema_version")
    assert cursor.fetchone()['version'] == PromptsDatabase._CURRENT_SCHEMA_VERSION

def test_initialization_empty_client_id():
    with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
        PromptsDatabase(db_path=":memory:", client_id="")

def test_add_keyword(memory_db: PromptsDatabase):
    kw_id, kw_uuid = memory_db.add_keyword("test_keyword")
    assert kw_id is not None
    assert isinstance(kw_id, int)
    assert kw_uuid is not None
    assert isinstance(uuid.UUID(kw_uuid, version=4), uuid.UUID)

    # Check if keyword exists
    res = memory_db.execute_query("SELECT * FROM PromptKeywordsTable WHERE id = ?", (kw_id,)).fetchone()
    assert res is not None
    assert res['keyword'] == "test_keyword" # Normalized
    assert res['deleted'] == 0

    # Add same keyword again (should return existing)
    kw_id_2, kw_uuid_2 = memory_db.add_keyword(" TeSt_KeYwOrD ")
    assert kw_id_2 == kw_id
    assert kw_uuid_2 == kw_uuid

    # Add empty keyword
    with pytest.raises(InputError):
        memory_db.add_keyword("  ")

def test_add_prompt(memory_db: PromptsDatabase):
    p_id, p_uuid, msg = memory_db.add_prompt(
        name="My Test Prompt",
        author="Tester",
        details="A prompt for testing.",
        system_prompt="System instructions.",
        user_prompt="User query.",
        keywords=["test", "example"]
    )
    assert p_id is not None
    assert isinstance(p_id, int)
    assert p_uuid is not None
    assert "added" in msg.lower()

    prompt_data = memory_db.fetch_prompt_details(p_id)
    assert prompt_data is not None
    assert prompt_data['name'] == "My Test Prompt"
    assert prompt_data['author'] == "Tester"
    assert "test" in prompt_data['keywords']
    assert "example" in prompt_data['keywords']

    # Try adding same prompt name without overwrite
    p_id2, p_uuid2, msg2 = memory_db.add_prompt(name="My Test Prompt", author="New Author", details=None)
    assert p_id2 == p_id
    assert "skipped" in msg2.lower() # or "already exists"

    # Add with overwrite
    p_id3, p_uuid3, msg3 = memory_db.add_prompt(
        name="My Test Prompt",
        author="Updated Author",
        details="Updated details.",
        overwrite=True
    )
    assert p_id3 == p_id
    assert "updated" in msg3.lower()
    updated_prompt = memory_db.fetch_prompt_details(p_id)
    assert updated_prompt['author'] == "Updated Author"

def test_soft_delete_and_undelete_prompt(memory_db: PromptsDatabase):
    p_id, _, _ = memory_db.add_prompt(name="Deletable Prompt", author="Test", details="Details")
    assert p_id is not None

    # Soft delete
    deleted = memory_db.soft_delete_prompt(p_id)
    assert deleted is True
    assert memory_db.get_prompt_by_id(p_id) is None
    assert memory_db.get_prompt_by_id(p_id, include_deleted=True) is not None

    # Try deleting again (should return False)
    deleted_again = memory_db.soft_delete_prompt(p_id)
    assert deleted_again is False

    # Undelete (by adding with overwrite=True)
    memory_db.add_prompt(name="Deletable Prompt", author="Test", details="Restored", overwrite=True)
    restored_prompt = memory_db.get_prompt_by_id(p_id)
    assert restored_prompt is not None
    assert restored_prompt['deleted'] == 0
    assert restored_prompt['details'] == "Restored"

def test_soft_delete_keyword_and_links(memory_db: PromptsDatabase):
    memory_db.add_prompt(name="Prompt With Keyword", author="Test", details="...", keywords=["deletable_kw"])
    kw_info = memory_db.execute_query("SELECT id FROM PromptKeywordsTable WHERE keyword='deletable_kw'").fetchone()
    assert kw_info is not None
    kw_id = kw_info['id']

    # Check link exists
    link = memory_db.execute_query("SELECT * FROM PromptKeywordLinks WHERE keyword_id=?", (kw_id,)).fetchone()
    assert link is not None

    # Soft delete keyword
    deleted = memory_db.soft_delete_keyword("deletable_kw")
    assert deleted is True

    # Verify keyword is deleted
    assert memory_db.execute_query("SELECT id FROM PromptKeywordsTable WHERE keyword='deletable_kw' AND deleted=0").fetchone() is None
    # Verify link is gone (due to cascade or explicit delete in soft_delete_keyword)
    assert memory_db.execute_query("SELECT * FROM PromptKeywordLinks WHERE keyword_id=?", (kw_id,)).fetchone() is None

def test_update_keywords_for_prompt(memory_db: PromptsDatabase):
    p_id, _, _ = memory_db.add_prompt(name="Keyword Update Prompt", author="Test", details="...", keywords=["initial1", "initial2"])
    assert p_id is not None

    memory_db.update_keywords_for_prompt(p_id, ["initial2", "new1", "new2"])
    updated_keywords = memory_db.fetch_keywords_for_prompt(p_id)
    assert sorted(updated_keywords) == sorted(["initial2", "new1", "new2"])

    memory_db.update_keywords_for_prompt(p_id, []) # Remove all
    assert memory_db.fetch_keywords_for_prompt(p_id) == []

def test_search_prompts_fts(memory_db: PromptsDatabase):
    memory_db.add_prompt(name="Alpha Search", author="AuthorA", details="Unique detail alpha", keywords=["common", "alpha_k"])
    memory_db.add_prompt(name="Beta Search", author="AuthorB", details="Common detail beta", keywords=["common", "beta_k"])
    memory_db.add_prompt(name="Gamma NonMatch", author="AuthorC", details="Different info", keywords=["other"])
    time.sleep(0.1) # Allow FTS to potentially catch up if there were async aspects (though SQLite FTS is sync)

    results, total = memory_db.search_prompts(search_query="alpha")
    assert total == 1
    assert len(results) == 1
    assert results[0]['name'] == "Alpha Search"

    results_k, total_k = memory_db.search_prompts(search_query="common", search_fields=["keywords"])
    assert total_k == 2
    assert len(results_k) == 2

    results_detail, total_detail = memory_db.search_prompts(search_query="detail", search_fields=["details"])
    assert total_detail == 2 # "Unique detail alpha", "Common detail beta"

    # Test FTS on system/user prompts
    memory_db.add_prompt(name="SysUserPrompt", system_prompt="System test phrase", user_prompt="User specific content")
    results_sys, _ = memory_db.search_prompts(search_query="phrase", search_fields=["system_prompt"])
    assert len(results_sys) == 1
    assert results_sys[0]['name'] == "SysUserPrompt"


def test_sync_log(memory_db: PromptsDatabase):
    p_id, p_uuid, _ = memory_db.add_prompt(name="Sync Log Test Prompt", author="Sync", details="...")
    kw_id, kw_uuid = memory_db.add_keyword("sync_keyword")
    memory_db.update_keywords_for_prompt(p_id, ["sync_keyword"]) # This will log a link

    log_entries = memory_db.get_sync_log_entries()
    assert len(log_entries) >= 3 # create prompt, create keyword, link them

    create_prompt_entry = next(e for e in log_entries if e['entity_uuid'] == p_uuid and e['operation'] == 'create')
    assert create_prompt_entry is not None
    assert create_prompt_entry['payload']['name'] == "Sync Log Test Prompt"

    create_kw_entry = next(e for e in log_entries if e['entity_uuid'] == kw_uuid and e['operation'] == 'create')
    assert create_kw_entry is not None

    link_entry = next(e for e in log_entries if e['entity'] == 'PromptKeywordLinks' and e['operation'] == 'link')
    assert link_entry is not None
    assert link_entry['payload']['prompt_uuid'] == p_uuid
    assert link_entry['payload']['keyword_uuid'] == kw_uuid

    # Test deleting sync log entries
    change_ids_to_delete = [e['change_id'] for e in log_entries[:2]]
    deleted_count = memory_db.delete_sync_log_entries(change_ids_to_delete)
    assert deleted_count == len(change_ids_to_delete)
    remaining_entries = memory_db.get_sync_log_entries()
    assert len(remaining_entries) == len(log_entries) - deleted_count


def test_versioning_and_conflict(memory_db: PromptsDatabase):
    p_id, p_uuid, _ = memory_db.add_prompt(name="Version Test", author="V1", details="Initial")
    prompt_v1 = memory_db.get_prompt_by_id(p_id)
    assert prompt_v1['version'] == 1

    # Simulate an update with correct version increment (via add_prompt with overwrite)
    memory_db.add_prompt(name="Version Test", author="V2", details="Updated", overwrite=True)
    prompt_v2 = memory_db.get_prompt_by_id(p_id)
    assert prompt_v2['version'] == 2

    # Simulate a direct DB update with incorrect version (should be blocked by trigger)
    conn = memory_db.get_connection()
    with pytest.raises(sqlite3.IntegrityError, match="Sync Error (Prompts): Version must increment by exactly 1."):
        with memory_db.transaction(): # Use transaction context
            conn.execute(
                "UPDATE Prompts SET details = ?, version = ?, client_id = ?, last_modified = ? WHERE id = ?",
                ("Conflict attempt", prompt_v2['version'] + 2, TEST_CLIENT_ID, memory_db._get_current_utc_timestamp_str(), p_id)
            )
            # The transaction context will attempt to commit, which is when the trigger's RAISE(ABORT) takes effect.

    # Test ConflictError on soft_delete_prompt if version mismatch (harder to simulate without direct version manipulation)
    # PromptsDatabase.soft_delete_prompt internally fetches current version, so direct conflict is less likely unless concurrent access
    # To test ConflictError from soft_delete_prompt, one would need to:
    # 1. Fetch prompt (gets current_version_A)
    # 2. Concurrently, another process updates the prompt (version becomes current_version_A + 1)
    # 3. The first process tries to soft_delete using current_version_A, which now mismatches.
    # This is hard to test in a single-threaded unit test without complex mocking.
    # The trigger test above covers the core version integrity.

# --- Test Standalone Functions ---

def test_standalone_add_or_update_prompt(memory_db: PromptsDatabase):
    p_id, p_uuid, msg = standalone_add_or_update_prompt(
        memory_db, name="Standalone Prompt", author="Standalone", details="Details"
    )
    assert p_id is not None
    assert "added" in msg or "updated" in msg # First time it's added

    p_id2, _, msg2 = standalone_add_or_update_prompt(
        memory_db, name="Standalone Prompt", author="Standalone Updated", details="New Details"
    )
    assert p_id2 == p_id
    assert "updated" in msg2
    updated_prompt = memory_db.get_prompt_by_name("Standalone Prompt")
    assert updated_prompt['author'] == "Standalone Updated"


def test_standalone_load_prompt_details_for_ui(memory_db: PromptsDatabase):
    standalone_add_or_update_prompt(
        memory_db, name="UI Prompt", author="UI Author", details="UI Details",
        system_prompt="Sys UI", user_prompt="User UI", keywords=["ui_kw1", "ui_kw2"]
    )
    name, author, details, system, user, kws_str = standalone_load_prompt_details_for_ui(memory_db, "UI Prompt")
    assert name == "UI Prompt"
    assert author == "UI Author"
    assert details == "UI Details"
    assert system == "Sys UI"
    assert user == "User UI"
    assert "ui_kw1" in kws_str and "ui_kw2" in kws_str

    # Test non-existent prompt
    name_nf, _, _, _, _, _ = standalone_load_prompt_details_for_ui(memory_db, "NonExistentPrompt")
    assert name_nf == ""

def test_standalone_export_functions(memory_db: PromptsDatabase, tmp_path: Path):
    memory_db.add_prompt("Export Prompt 1", "Export Author", "Details1", keywords=["export_kw", "common_kw"])
    memory_db.add_prompt("Export Prompt 2", "Export Author", "Details2", keywords=["another_kw", "common_kw"])

    # Test export_prompts_formatted (CSV)
    status_csv, path_csv_str = standalone_export_prompts_formatted(memory_db, export_format='csv')
    path_csv = Path(path_csv_str)
    assert "Successfully exported" in status_csv
    assert path_csv.exists()
    assert path_csv.suffix == ".csv"
    with open(path_csv, 'r') as f:
        content = f.read()
        assert "Export Prompt 1" in content
        assert "common_kw" in content # Assuming keywords are exported
    os.remove(path_csv)

    # Test export_prompts_formatted (Markdown)
    status_md, path_md_zip_str = standalone_export_prompts_formatted(memory_db, export_format='markdown')
    path_md_zip = Path(path_md_zip_str)
    assert "Successfully exported" in status_md
    assert path_md_zip.exists()
    assert path_md_zip.suffix == ".zip" # It creates a zip of markdown files
    # Further inspection of zip contents could be done here.
    os.remove(path_md_zip)

    # Test export_prompt_keywords_to_csv
    status_kw_csv, path_kw_csv_str = standalone_export_prompt_keywords_to_csv(memory_db)
    path_kw_csv = Path(path_kw_csv_str)
    assert "Successfully exported" in status_kw_csv
    assert path_kw_csv.exists()
    assert path_kw_csv.suffix == ".csv"
    with open(path_kw_csv, 'r') as f:
        content = f.read()
        assert "export_kw" in content
        assert "common_kw" in content
    os.remove(path_kw_csv)

    # Test view_prompt_keywords_markdown
    md_output = standalone_view_prompt_keywords_markdown(memory_db)
    assert "Current Active Prompt Keywords" in md_output
    assert "export_kw" in md_output
    assert "common_kw (2 active prompts)" in md_output # Check count

def test_get_next_version_logic(memory_db: PromptsDatabase):
    # This is an internal helper, but its logic is critical
    p_id, _, _ = memory_db.add_prompt(name="Version Helper Test", author="Test", details="...")
    prompt_data = memory_db.get_prompt_by_id(p_id)
    assert prompt_data['version'] == 1 # Initial version

    conn = memory_db.get_connection()
    version_info = memory_db._get_next_version(conn, "Prompts", "id", p_id)
    assert version_info is not None
    current_v, next_v = version_info
    assert current_v == 1
    assert next_v == 2

    # Simulate an update
    memory_db.add_prompt(name="Version Helper Test", author="Test", details="Updated", overwrite=True)
    version_info_after_update = memory_db._get_next_version(conn, "Prompts", "id", p_id)
    assert version_info_after_update is not None
    current_v_up, next_v_up = version_info_after_update
    assert current_v_up == 2
    assert next_v_up == 3

    # Test for non-existent record
    assert memory_db._get_next_version(conn, "Prompts", "id", 99999) is None