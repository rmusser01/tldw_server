# DB_Manager.py
# Description: This file contains the DatabaseManager class, which is responsible for managing the database connection, i.e. either SQLite or Elasticsearch.
#
# Imports
import configparser
import os
from typing import Tuple, List, Union, Dict
#
# 3rd-Party Libraries
#from elasticsearch import Elasticsearch
#
# Import your existing SQLite functions
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import DatabaseError
from PoC_Version.App_Function_Libraries.DB.Prompts_DB import list_prompts as sqlite_list_prompts, \
    fetch_prompt_details as sqlite_fetch_prompt_details, add_prompt as sqlite_add_prompt, \
    search_prompts as sqlite_search_prompts, add_or_update_prompt as sqlite_add_or_update_prompt, \
    load_prompt_details as sqlite_load_prompt_details, insert_prompt_to_db as sqlite_insert_prompt_to_db, \
    delete_prompt as sqlite_delete_prompt
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import (
    update_media_content as sqlite_update_media_content,
    search_and_display as sqlite_search_and_display,
    keywords_browser_interface as sqlite_keywords_browser_interface,
    add_keyword as sqlite_add_keyword,
    delete_keyword as sqlite_delete_keyword,
    export_keywords_to_csv as sqlite_export_keywords_to_csv,
    ingest_article_to_db as sqlite_ingest_article_to_db,
    add_media_to_database as sqlite_add_media_to_database,
    import_obsidian_note_to_db as sqlite_import_obsidian_note_to_db,
    view_database as sqlite_view_database,
    get_transcripts as sqlite_get_transcripts,
    get_trashed_items as sqlite_get_trashed_items,
    user_delete_item as sqlite_user_delete_item,
    empty_trash as sqlite_empty_trash,
    create_automated_backup as sqlite_create_automated_backup,
    search_and_display_items as sqlite_search_and_display_items,
    add_media_with_keywords as sqlite_add_media_with_keywords,
    check_media_and_whisper_model as sqlite_check_media_and_whisper_model, \
    create_document_version as sqlite_create_document_version,
    get_document_version as sqlite_get_document_version, search_media_db as sqlite_search_media_db, add_media_chunk as sqlite_add_media_chunk,
    sqlite_update_fts_for_media, get_unprocessed_media as sqlite_get_unprocessed_media, fetch_item_details as sqlite_fetch_item_details, \
    search_media_database as sqlite_search_media_database, mark_as_trash as sqlite_mark_as_trash, \
    get_media_transcripts as sqlite_get_media_transcripts, get_specific_transcript as sqlite_get_specific_transcript, \
    get_media_summaries as sqlite_get_media_summaries, get_specific_summary as sqlite_get_specific_summary, \
    get_media_prompts as sqlite_get_media_prompts, get_specific_prompt as sqlite_get_specific_prompt, \
    delete_specific_transcript as sqlite_delete_specific_transcript,
    delete_specific_summary as sqlite_delete_specific_summary, \
    delete_specific_prompt as sqlite_delete_specific_prompt,
    fetch_keywords_for_media as sqlite_fetch_keywords_for_media, \
    update_keywords_for_media as sqlite_update_keywords_for_media, check_media_exists as sqlite_check_media_exists, \
    get_media_content as sqlite_get_media_content, get_paginated_files as sqlite_get_paginated_files, \
    get_media_title as sqlite_get_media_title, get_all_content_from_database as sqlite_get_all_content_from_database, \
    get_next_media_id as sqlite_get_next_media_id, batch_insert_chunks as sqlite_batch_insert_chunks, Database, \
    save_workflow_chat_to_db as sqlite_save_workflow_chat_to_db, get_workflow_chat as sqlite_get_workflow_chat, \
    update_media_content_with_version as sqlite_update_media_content_with_version, \
    check_existing_media as sqlite_check_existing_media, get_all_document_versions as sqlite_get_all_document_versions, \
    fetch_paginated_data as sqlite_fetch_paginated_data, get_latest_transcription as sqlite_get_latest_transcription, \
    mark_media_as_processed as sqlite_mark_media_as_processed,
)
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import start_new_conversation as sqlite_start_new_conversation, \
    save_message as sqlite_save_message, load_chat_history as sqlite_load_chat_history, \
    get_all_conversations as sqlite_get_all_conversations, get_notes_by_keywords as sqlite_get_notes_by_keywords, \
    get_note_by_id as sqlite_get_note_by_id, update_note as sqlite_update_note, save_notes as sqlite_save_notes, \
    clear_keywords_from_note as sqlite_clear_keywords_from_note, add_keywords_to_note as sqlite_add_keywords_to_note, \
    add_keywords_to_conversation as sqlite_add_keywords_to_conversation, \
    get_keywords_for_note as sqlite_get_keywords_for_note, delete_note as sqlite_delete_note, \
    search_conversations_by_keywords as sqlite_search_conversations_by_keywords, \
    delete_conversation as sqlite_delete_conversation, get_conversation_title as sqlite_get_conversation_title, \
    update_conversation_title as sqlite_update_conversation_title, \
    fetch_all_conversations as sqlite_fetch_all_conversations, fetch_all_notes as sqlite_fetch_all_notes, \
    fetch_conversations_by_ids as sqlite_fetch_conversations_by_ids, fetch_notes_by_ids as sqlite_fetch_notes_by_ids, \
    delete_messages_in_conversation as sqlite_delete_messages_in_conversation, \
    get_conversation_text as sqlite_get_conversation_text, search_notes_titles as sqlite_search_notes_titles
from PoC_Version.App_Function_Libraries.DB.Character_Chat_DB import (
    add_character_card as sqlite_add_character_card, get_character_cards as sqlite_get_character_cards, \
    get_character_card_by_id as sqlite_get_character_card_by_id, update_character_card as sqlite_update_character_card, \
    delete_character_card as sqlite_delete_character_card, add_character_chat as sqlite_add_character_chat, \
    get_character_chats as sqlite_get_character_chats, get_character_chat_by_id as sqlite_get_character_chat_by_id, \
    update_character_chat as sqlite_update_character_chat, delete_character_chat as sqlite_delete_character_chat
)
#
# Local Imports
from PoC_Version.App_Function_Libraries.Utils.Utils import load_comprehensive_config, get_database_path, get_project_relative_path, \
    logger, logging

#
# End of imports
############################################################################################################

############################################################################################################
#
# Database Config loading
config_path = get_project_relative_path('Config_Files/config.txt')
config = configparser.ConfigParser()
config.read(config_path)

db_path: str = config.get('Database', 'sqlite_path', fallback='./Databases/media_summary.db')
backup_path: str = config.get('Database', 'backup_path', fallback='database_backups')
backup_dir: Union[str, bytes] = os.environ.get('DB_BACKUP_DIR', backup_path)

def get_db_config():
    try:
        config = load_comprehensive_config()

        if 'Database' not in config:
            print("Warning: 'Database' section not found in config. Using default values.")
            return default_db_config()

        return {
            'type': config.get('Database', 'type', fallback='sqlite'),
            'sqlite_path': config.get('Database', 'sqlite_path', fallback='Databases/media_summary.db'),
            'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
            'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200)
        }
    except FileNotFoundError:
        print("Warning: Config file not found. Using default database configuration.")
        return default_db_config()
    except Exception as e:
        print(f"Error reading config: {str(e)}. Using default database configuration.")
        return default_db_config()

def default_db_config():
    return {
        'type': 'sqlite',
        'sqlite_path': get_database_path('media_summary.db'),
        'elasticsearch_host': 'localhost',
        'elasticsearch_port': 9200
    }

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

db_config = get_db_config()
db_type = db_config['type']

if db_type == 'sqlite':
    db = Database(os.path.basename(db_config['sqlite_path']))
elif db_type == 'elasticsearch':
    raise NotImplementedError("Elasticsearch support not yet implemented")
else:
    raise ValueError(f"Unsupported database type: {db_type}")

print(f"Database path: {db.db_path}")

def get_db_config():
    try:
        config = load_comprehensive_config()

        if 'Database' not in config:
            print("Warning: 'Database' section not found in config. Using default values.")
            return default_db_config()

        return {
            'type': config.get('Database', 'type', fallback='sqlite'),
            'sqlite_path': config.get('Database', 'sqlite_path', fallback='Databases/media_summary.db'),
            'elasticsearch_host': config.get('Database', 'elasticsearch_host', fallback='localhost'),
            'elasticsearch_port': config.getint('Database', 'elasticsearch_port', fallback=9200)
        }
    except FileNotFoundError:
        print("Warning: Config file not found. Using default database configuration.")
        return default_db_config()
    except Exception as e:
        print(f"Error reading config: {str(e)}. Using default database configuration.")
        return default_db_config()


def default_db_config():
    """Return the default database configuration with project-relative paths."""
    return {
        'type': 'sqlite',
        'sqlite_path': get_database_path('media_summary.db'),
        'elasticsearch_host': 'localhost',
        'elasticsearch_port': 9200
    }


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Use the config to set up the database
db_config = get_db_config()
db_type = db_config['type']

if db_type == 'sqlite':
    db = Database(os.path.basename(db_config['sqlite_path']))
elif db_type == 'elasticsearch':
    # Implement Elasticsearch setup here if needed
    raise NotImplementedError("Elasticsearch support not yet implemented")
else:
    raise ValueError(f"Unsupported database type: {db_type}")

# Print database path for debugging
print(f"Database path: {db.db_path}")

# Sanity Check for SQLite DB
# FIXME - Remove this after testing / Writing Unit tests
# try:
#     db.execute_query("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)")
#     logger.info("Successfully created test table")
# except DatabaseError as e:
#     logger.error(f"Failed to create test table: {e}")

#
# End of Database Config loading
############################################################################################################
#
# DB Search functions

def search_media_db(search_query: str, search_fields: List[str], keywords: str, page: int = 1, results_per_page: int = 10):
    if db_type == 'sqlite':
        return sqlite_search_media_db(search_query, search_fields, keywords, page, results_per_page)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of search_db not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def view_database(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_view_database(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_and_display_items(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_and_display_items(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_all_content_from_database():
    if db_type == 'sqlite':
        return sqlite_get_all_content_from_database()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_and_display(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_and_display(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def check_media_exists(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_media_exists(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_paginated_files(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_paginated_files(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_media_title(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_media_title(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_next_media_id():
    if db_type == 'sqlite':
        return sqlite_get_next_media_id()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of DB-Searching functions
############################################################################################################


############################################################################################################
#
# Transcript-related Functions

def get_transcripts(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_transcripts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of Transcript-related Functions
############################################################################################################


############################################################################################################
#
# DB-Ingestion functions

def add_media_to_database(*args, **kwargs):
    if db_type == 'sqlite':
        result = sqlite_add_media_to_database(*args, **kwargs)

        # Extract content
        segments = kwargs.get('segments') if 'segments' in kwargs else args[2] if len(args) > 2 else None
        if segments is None:
            raise ValueError("Segments not provided in arguments")

        if isinstance(segments, list):
            content = ' '.join([segment.get('Text', '') for segment in segments if 'Text' in segment])
        elif isinstance(segments, dict):
            content = segments.get('text', '') or segments.get('content', '')
        else:
            content = str(segments)

        # Extract media_id from the result
        # Assuming the result is in the format "Media 'Title' added/updated successfully with ID: {media_id}"
        import re
        match = re.search(r"with ID: (\d+)", result)
        if match:
            media_id = int(match.group(1))

            # Create initial document version
            sqlite_create_document_version(media_id, content)

        return result
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_to_database not yet implemented")

def check_existing_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_existing_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of check_existing_media not yet implemented")

def update_media_content_with_version(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_media_content_with_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_media_content not yet implemented")

def import_obsidian_note_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_import_obsidian_note_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")


def update_media_content(*args, **kwargs):
    if db_type == 'sqlite':
        result = sqlite_update_media_content(*args, **kwargs)

        # Extract media_id and content
        selected_item = args[0]
        item_mapping = args[1]
        content_input = args[2]

        if selected_item and item_mapping and selected_item in item_mapping:
            media_id = item_mapping[selected_item]

            # Create new document version
            sqlite_create_document_version(media_id, content_input)

        return result
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_media_content not yet implemented")


def add_media_with_keywords(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_media_with_keywords(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def check_media_and_whisper_model(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_media_and_whisper_model(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of check_media_and_whisper_model not yet implemented")

def ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date, custom_prompt):
    if db_type == 'sqlite':
        return sqlite_ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date, custom_prompt)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of ingest_article_to_db not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def add_media_chunk(*args, **kwargs):
    if db_type == 'sqlite':
        sqlite_add_media_chunk(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def batch_insert_chunks(*args, **kwargs):
    if db_type == 'sqlite':
        sqlite_batch_insert_chunks(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def update_fts_for_media(media_id: int):
    if db_type == 'sqlite':
        sqlite_update_fts_for_media(db, media_id)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_unprocessed_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_unprocessed_media(db)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_unprocessed_media not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def mark_media_as_processed(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_mark_media_as_processed(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of mark_media_as_processed not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


#
# End of DB-Ingestion functions
############################################################################################################


############################################################################################################
#
# Prompt-related functions #FIXME rename /resort

def list_prompts(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_list_prompts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_prompts(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_prompts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def fetch_prompt_details(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_prompt_details(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def add_prompt(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")


def add_or_update_prompt(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_or_update_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def load_prompt_details(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_load_prompt_details(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def insert_prompt_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_insert_prompt_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_prompt(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_media_database(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_media_database(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of search_media_database not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def mark_as_trash(media_id: int) -> None:
    if db_type == 'sqlite':
        return sqlite_mark_as_trash(media_id)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of mark_as_trash not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_latest_transcription(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_latest_transcription(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_latest_transcription not yet implemented")

def fetch_paginated_data(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_paginated_data(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of fetch_paginated_data not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

        
def get_media_content(media_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_get_media_content(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_content not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_transcripts(media_id: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_transcripts(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_transcripts not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_transcript(transcript_id: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_transcript(transcript_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_transcript not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_summaries(media_id: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_summaries(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_summaries not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_summary(summary_id: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_summary(summary_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_summary not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def fetch_item_details_single(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_item_details(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of fetch_item_details not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_all_document_versions(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_all_document_versions(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_all_document_versions not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
#
#
############################################################################################################
#
# Prompt Functions:

def get_media_prompts(media_id: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_prompts(media_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_prompts not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_prompt(prompt_id: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_prompt(prompt_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_prompt not yet implemented")
    else:
        return {'error': f"Unsupported database type: {db_type}"}

def delete_specific_transcript(transcript_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_delete_specific_transcript(transcript_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_transcript not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def delete_specific_summary(summary_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_delete_specific_summary(summary_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_summary not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def delete_specific_prompt(prompt_id: int) -> str:
    if db_type == 'sqlite':
        return sqlite_delete_specific_prompt(prompt_id)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_prompt not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


#
# End of Prompt-related functions
############################################################################################################

############################################################################################################
#
# Keywords-related Functions

def keywords_browser_interface(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_keywords_browser_interface()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def add_keyword(*args, **kwargs):
    if db_type == 'sqlite':
        with db.get_connection() as conn:
            cursor = conn.cursor()
        return sqlite_add_keyword(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_keyword(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_keyword(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def export_keywords_to_csv(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_export_keywords_to_csv()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def update_keywords_for_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def fetch_keywords_for_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of Keywords-related Functions
############################################################################################################

############################################################################################################
#
# Chat-related Functions

def search_notes_titles(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_notes_titles(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def save_message(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_save_message(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def load_chat_history(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_load_chat_history(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def start_new_conversation(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_start_new_conversation(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_all_conversations(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_all_conversations(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_notes_by_keywords(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_notes_by_keywords(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_note_by_id(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_note_by_id(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def add_keywords_to_conversation(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_keywords_to_conversation(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_keywords_for_note(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_keywords_for_note(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_note(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_note(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def search_conversations_by_keywords(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_search_conversations_by_keywords(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_conversation(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_conversation(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def get_conversation_title(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_conversation_title(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def update_conversation_title(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_conversation_title(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def fetch_all_conversations(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_all_conversations()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def fetch_all_notes(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_all_notes()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def delete_messages_in_conversation(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_messages_in_conversation(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of delete_messages_in_conversation not yet implemented")

def get_conversation_text(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_conversation_text(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_conversation_text not yet implemented")

#
# End of Chat-related Functions
############################################################################################################


############################################################################################################
#
# Character Chat-related Functions

def add_character_card(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_character_card(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_character_card not yet implemented")

def get_character_cards():
    if db_type == 'sqlite':
        return sqlite_get_character_cards()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_character_cards not yet implemented")

def get_character_card_by_id(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_character_card_by_id(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_character_card_by_id not yet implemented")

def update_character_card(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_character_card(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_character_card not yet implemented")

def delete_character_card(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_character_card(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of delete_character_card not yet implemented")

def add_character_chat(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_character_chat(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_character_chat not yet implemented")

def get_character_chats(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_character_chats(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_character_chats not yet implemented")

def get_character_chat_by_id(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_character_chat_by_id(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_character_chat_by_id not yet implemented")

def update_character_chat(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_character_chat(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_character_chat not yet implemented")

def delete_character_chat(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_delete_character_chat(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of delete_character_chat not yet implemented")

def update_note(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_update_note(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_note not yet implemented")

def save_notes(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_save_notes(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of save_notes not yet implemented")

def clear_keywords(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_clear_keywords_from_note(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of clear_keywords not yet implemented")

def clear_keywords_from_note(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_clear_keywords_from_note(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of clear_keywords_from_note not yet implemented")

def add_keywords_to_note(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_add_keywords_to_note(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_keywords_to_note not yet implemented")

def fetch_conversations_by_ids(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_conversations_by_ids(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of fetch_conversations_by_ids not yet implemented")

def fetch_notes_by_ids(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_notes_by_ids(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of fetch_notes_by_ids not yet implemented")

#
# End of Character Chat-related Functions
############################################################################################################


############################################################################################################
#
# Trash-related Functions

def get_trashed_items(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_trashed_items()
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def user_delete_item(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_user_delete_item(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

def empty_trash(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_empty_trash(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")


def fetch_item_details(media_id: int) -> Tuple[str, str, str]:
    """
    Fetch the details of a media item including content, prompt, and summary.

    Args:
        media_id (int): The ID of the media item.

    Returns:
        Tuple[str, str, str]: A tuple containing (content, prompt, summary).
        If an error occurs, it returns empty strings for each field.
    """
    if db_type == 'sqlite':
        return sqlite_fetch_item_details(media_id)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version when available
        raise NotImplementedError("Elasticsearch version of fetch_item_details not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of Trash-related Functions
############################################################################################################


############################################################################################################
#
# DB-Backup Functions

def create_automated_backup(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_create_automated_backup(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")

#
# End of DB-Backup Functions
############################################################################################################


############################################################################################################
#
# Document Versioning Functions

def create_document_version(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_create_document_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of create_document_version not yet implemented")

def get_document_version(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_document_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_document_version not yet implemented")

#
# End of Document Versioning Functions
############################################################################################################


############################################################################################################
#
# Workflow Functions

def get_workflow_chat(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_workflow_chat(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of get_workflow_chat not yet implemented")


def save_workflow_chat_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        # FIXME
        return sqlite_save_workflow_chat_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of save_workflow_chat_to_db not yet implemented")

#
# End of Workflow Functions
############################################################################################################

# Dead code FIXME
# def close_connection():
#     if db_type == 'sqlite':
#         db.get_connection().close()

#
# End of file
############################################################################################################
