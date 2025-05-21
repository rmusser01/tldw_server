# DB_Manager.py
# Description: This file contains the DatabaseManager class, which is responsible for managing the database connection, i.e. either SQLite or Elasticsearch.
#
# Imports
import configparser
import os
from typing import List, Tuple, Union, Dict
#
# 3rd-Party Libraries
#from elasticsearch import Elasticsearch
#
# Local Imports
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.Utils.Utils import get_database_path, get_project_relative_path
#from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    #list_prompts as sqlite_list_prompts,
    #fetch_prompt_details as sqlite_fetch_prompt_details,
    #add_prompt as sqlite_add_prompt,
    #search_prompts as sqlite_search_prompts,
    #add_or_update_prompt as sqlite_add_or_update_prompt,
    #load_prompt_details as sqlite_load_prompt_details,
    # insert_prompt_to_db as sqlite_insert_prompt_to_db,
    #delete_prompt as sqlite_delete_prompt
#)
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import (
    MediaDatabase,
    import_obsidian_note_to_db as sqlite_import_obsidian_note_to_db,
    empty_trash as sqlite_empty_trash,
    create_automated_backup as sqlite_create_automated_backup,
    check_media_and_whisper_model as sqlite_check_media_and_whisper_model, \
    get_document_version as sqlite_get_document_version,
    get_media_transcripts as sqlite_get_media_transcripts,
    get_specific_transcript as sqlite_get_specific_transcript, \
    get_specific_analysis as sqlite_get_specific_summary, \
    get_media_prompts as sqlite_get_media_prompts,
    get_specific_prompt as sqlite_get_specific_prompt, \
    fetch_keywords_for_media as sqlite_fetch_keywords_for_media, \
    check_media_exists as sqlite_check_media_exists, \
    get_all_content_from_database as sqlite_get_all_content_from_database, \
    get_latest_transcription as sqlite_get_latest_transcription, \
    mark_media_as_processed as sqlite_mark_media_as_processed,
    ingest_article_to_db_new as sqlite_ingest_article_to_db, \
    get_unprocessed_media as sqlite_get_unprocessed_media,\
    )
#from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import
#
# End of imports
############################################################################################################

############################################################################################################
#
# Database Config loading
single_user_config_path = get_project_relative_path('Config_Files/config.txt')
single_user_config = configparser.ConfigParser()
single_user_config.read(single_user_config_path)

single_user_db_path: str = single_user_config.get('Database', 'sqlite_path', fallback='./Databases/server_media_summary.db')
single_user_backup_path: str = single_user_config.get('Database', 'backup_path', fallback='database_backups')
single_user_backup_dir: Union[str, bytes] = os.environ.get('DB_BACKUP_DIR', single_user_backup_path)


def get_db_config():
    try:
        config = load_comprehensive_config()

        if 'Database' not in config:
            print("Warning: 'Database' section not found in config. Using default values.")
            return default_db_config()

        return {
            'type': config.get('Database', 'type', fallback='sqlite'),
            'sqlite_path': config.get('Database', 'sqlite_path', fallback='Databases/server_media_summary.db'),
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
        'sqlite_path': get_database_path('server_media_summary.db'),
        'elasticsearch_host': 'localhost',
        'elasticsearch_port': 9200
    }

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

BIGSEARCH = single_user_config.getboolean('Database', 'bigsearch', fallback=False)
if not BIGSEARCH == True:
    db_type = 'sqlite'
elif BIGSEARCH == True and single_user_config.get('Database', 'type') == 'elasticsearch':
    db_type = 'elasticsearch'
elif BIGSEARCH == True and single_user_config.get('Database', 'type') == 'postgres':
    db_type = 'postgres'


#
# End of Database Config loading
############################################################################################################
#
# DB Search functions

def get_all_content_from_database(*args, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.get_all_content_from_database(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def check_media_exists(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_media_exists(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def get_full_media_details2(*args, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.get_full_media_details(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def get_paginated_files(*args, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.get_paginated_files(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

#
# End of DB-Searching functions
############################################################################################################


############################################################################################################
#
# DB-Ingestion functions

def import_obsidian_note_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_import_obsidian_note_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
    return None


def add_media_with_keywords(*args, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.add_media_with_keywords(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    return None


def check_media_and_whisper_model(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_check_media_and_whisper_model(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of check_media_and_whisper_model not yet implemented")
    return None


def ingest_article_to_db(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_ingest_article_to_db(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of ingest_article_to_db not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def add_media_chunk(*args, **kwargs):
    if db_type == 'sqlite':
        MediaDatabase.add_media_chunk(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def batch_insert_chunks(*args, **kwargs):
    if db_type == 'sqlite':
        MediaDatabase.batch_insert_chunks(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_unprocessed_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_get_unprocessed_media(*args, **kwargs)
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


def update_keywords_for_media(*args, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.update_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of update_keywords_for_media not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of update_keywords_for_media not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def rollback_to_version(*arg, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.rollback_to_version(*arg, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of rollback_to_version not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of rollback_to_version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def delete_document_version(*args, **kwargs):
    if db_type == 'sqlite':
        return MediaDatabase.soft_delete_document_version(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of delete_document_version not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of delete_document_version not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

#
# End of DB-Ingestion functions
############################################################################################################


############################################################################################################
#
# Prompt-related functions #FIXME rename /resort

# def list_prompts(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_list_prompts(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def search_prompts(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_search_prompts(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def fetch_prompt_details(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_prompt_details(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def add_prompt(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_prompt(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
#
# def add_or_update_prompt(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_or_update_prompt(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#     return None
#
#
# def load_prompt_details(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_load_prompt_details(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#     return None
#
#
# def insert_prompt_to_db(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_insert_prompt_to_db(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#     return None
#
#
# def delete_prompt(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_prompt(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def mark_as_trash(*args, **kwargs: int) -> bool:
    if db_type == 'sqlite':
        return MediaDatabase.mark_as_trash(*args, **kwargs)
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
        return MediaDatabase.fetch_paginated_data(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of fetch_paginated_data not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_media_transcripts(*args, **kwargs: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_transcripts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_transcripts not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_transcript(*args, **kwargs: int) -> Dict:
    if db_type == 'sqlite':
        return sqlite_get_specific_transcript(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_transcript not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_all_document_versions(db_instance: MediaDatabase, media_id: int, **kwargs):
    """
    Wrapper to get all document versions for a given media_id from a Database instance.
    """
    # db_type check might be relevant if you support multiple DB backends via DB_Manager
    # For now, assume db_instance is always a Media_DB_v2.Database instance.
    if isinstance(db_instance, MediaDatabase):
        # Call the INSTANCE method, passing only the relevant kwargs
        # The instance method itself is get_all_document_versions(self, media_id, include_content=True, include_deleted=False, limit=None, offset=None)
        # So we need to ensure only those valid arguments are passed from kwargs.

        # Extract known arguments for the instance method
        limit = kwargs.get('limit')
        offset = kwargs.get('offset')
        include_content = kwargs.get('include_content', True)  # Default if not in test call
        include_deleted = kwargs.get('include_deleted', False)  # Default if not in test call

        return db_instance.get_all_document_versions(
            media_id=media_id,
            include_content=include_content,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset
        )

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

def get_media_prompts(*args, **kwargs: int) -> List[Dict]:
    if db_type == 'sqlite':
        return sqlite_get_media_prompts(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_media_prompts not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_specific_prompt(*args, **kwargs: int) -> Dict:
    if db_type == 'sqlite':
        return get_specific_prompt(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of get_specific_prompt not yet implemented")
    else:
        return {'error': f"Unsupported database type: {db_type}"}

def delete_specific_summary(*args, **kwargs: int) -> str:
    if db_type == 'sqlite':
        return delete_specific_summary(*args, **kwargs)
    elif db_type == 'elasticsearch':
        raise NotImplementedError("Elasticsearch version of delete_specific_summary not yet implemented")
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def delete_specific_prompt(*args, **kwargs: int) -> str:
    if db_type == 'sqlite':
        return delete_specific_prompt(*args, **kwargs)
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
        return keywords_browser_interface(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def add_keyword(*args, **kwargs):
    if db_type == 'sqlite':
        return add_keyword(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def delete_keyword(*args, **kwargs):
    if db_type == 'sqlite':
        return delete_keyword(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def export_keywords_to_csv(*args, **kwargs):
    if db_type == 'sqlite':
        return export_keywords_to_csv(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def fetch_keywords_for_media(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_fetch_keywords_for_media(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

#
# End of Keywords-related Functions
############################################################################################################

############################################################################################################
#
# Chat-related Functions
# FIXME
# def search_notes_titles(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_search_notes_titles(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def save_message(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_save_message(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def load_chat_history(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_load_chat_history(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def start_new_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_start_new_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_all_conversations(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_all_conversations(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_notes_by_keywords(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_notes_by_keywords(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_note_by_id(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_note_by_id(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def add_keywords_to_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_keywords_to_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_keywords_for_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_keywords_for_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def delete_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def search_conversations_by_keywords(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_search_conversations_by_keywords(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def delete_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def get_conversation_title(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_conversation_title(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def update_conversation_title(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_conversation_title(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def fetch_all_conversations(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_all_conversations()
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def fetch_all_notes(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_all_notes()
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
#     elif db_type == 'postgres':
#         # Implement Postgres version
#         raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")
#
# def delete_messages_in_conversation(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_messages_in_conversation(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of delete_messages_in_conversation not yet implemented")
#
# def get_conversation_text(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_conversation_text(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_conversation_text not yet implemented")

#
# End of Chat-related Functions
############################################################################################################


############################################################################################################
#
# Character Chat-related Functions
# FIXME
# def add_character_card(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_character_card(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_character_card not yet implemented")
#
# def get_character_cards():
#     if db_type == 'sqlite':
#         return sqlite_get_character_cards()
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_cards not yet implemented")
#
# def get_character_card_by_id(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_character_card_by_id(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_card_by_id not yet implemented")
#
# def update_character_card(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_character_card(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of update_character_card not yet implemented")
#
# def delete_character_card(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_character_card(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of delete_character_card not yet implemented")
#
# def add_character_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_character_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_character_chat not yet implemented")
#
# def get_character_chats(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_character_chats(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_chats not yet implemented")
#
# def get_character_chat_by_id(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_character_chat_by_id(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_character_chat_by_id not yet implemented")
#
# def update_character_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_character_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of update_character_chat not yet implemented")
#
# def delete_character_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_delete_character_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of delete_character_chat not yet implemented")
#
# def update_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_update_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of update_note not yet implemented")
#
# def save_notes(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_save_notes(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of save_notes not yet implemented")
#
# def clear_keywords(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_clear_keywords_from_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of clear_keywords not yet implemented")
#
# def clear_keywords_from_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_clear_keywords_from_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of clear_keywords_from_note not yet implemented")
#
# def add_keywords_to_note(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_add_keywords_to_note(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of add_keywords_to_note not yet implemented")
#
# def fetch_conversations_by_ids(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_conversations_by_ids(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of fetch_conversations_by_ids not yet implemented")
#
# def fetch_notes_by_ids(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_fetch_notes_by_ids(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of fetch_notes_by_ids not yet implemented")

#
# End of Character Chat-related Functions
############################################################################################################


############################################################################################################
#
# Trash-related Functions

def get_trashed_items(*args, **kwargs):
    if db_type == 'sqlite':
        return get_trashed_items(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def user_delete_item(*args, **kwargs):
    if db_type == 'sqlite':
        return user_delete_item(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

def empty_trash(*args, **kwargs):
    if db_type == 'sqlite':
        return sqlite_empty_trash(*args, **kwargs)
    elif db_type == 'elasticsearch':
        # Implement Elasticsearch version
        raise NotImplementedError("Elasticsearch version of add_media_with_keywords not yet implemented")
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")


def fetch_item_details(*args, **kwargs) -> Tuple[str, str, str]:
    """
    Fetch the details of a media item including content, prompt, and summary.

    Args:
        media_id (int): The ID of the media item.

    Returns:
        Tuple[str, str, str]: A tuple containing (content, prompt, summary).
        If an error occurs, it returns empty strings for each field.
    """
    if db_type == 'sqlite':
        return fetch_item_details(*args, **kwargs)
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
    elif db_type == 'postgres':
        # Implement Postgres version
        raise NotImplementedError("Postgres version of add_media_with_keywords not yet implemented")

#
# End of DB-Backup Functions
############################################################################################################


############################################################################################################
#
# Document Versioning Functions

def create_document_version(*args, **kwargs):
    if db_type == 'sqlite':
        return create_document_version(*args, **kwargs)
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
#
# def get_workflow_chat(*args, **kwargs):
#     if db_type == 'sqlite':
#         return sqlite_get_workflow_chat(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of get_workflow_chat not yet implemented")
#
#
# def save_workflow_chat_to_db(*args, **kwargs):
#     if db_type == 'sqlite':
#         # FIXME
#         return sqlite_save_workflow_chat_to_db(*args, **kwargs)
#     elif db_type == 'elasticsearch':
#         # Implement Elasticsearch version
#         raise NotImplementedError("Elasticsearch version of save_workflow_chat_to_db not yet implemented")
#
# #
# End of Workflow Functions
############################################################################################################

# Dead code FIXME
# def close_connection():
#     if db_type == 'sqlite':
#         db.get_connection().close()

#
# End of file
############################################################################################################
class DatabaseError:
    pass