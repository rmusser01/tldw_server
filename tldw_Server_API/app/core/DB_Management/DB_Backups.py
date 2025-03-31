# Backup_Manager.py
#
# Imports:
import os
import shutil
import sqlite3
from datetime import datetime
#
# Local Imports:
from PoC_Version.App_Function_Libraries.DB.Character_Chat_DB import chat_DB_PATH
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import get_rag_qa_db_path
from PoC_Version.App_Function_Libraries.Utils import get_project_relative_path, logging


#
# End of Imports
#######################################################################################################################
#
# Functions:

def init_backup_directory(backup_base_dir: str, db_name: str) -> str:
    """Initialize backup directory for a specific database."""
    backup_dir = os.path.join(backup_base_dir, db_name)
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


def create_backup(db_path: str, backup_dir: str, db_name: str) -> str:
    """Create a full backup of the database."""
    try:
        db_path = os.path.abspath(db_path)
        backup_dir = os.path.abspath(backup_dir)

        logging.info(f"Creating backup:")
        logging.info(f"  DB Path: {db_path}")
        logging.info(f"  Backup Dir: {backup_dir}")
        logging.info(f"  DB Name: {db_name}")

        # Create subdirectory based on db_name
        specific_backup_dir = os.path.join(backup_dir, db_name)
        os.makedirs(specific_backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(specific_backup_dir, f"{db_name}_backup_{timestamp}.db")
        logging.info(f"  Full backup path: {backup_file}")

        # Create a backup using SQLite's backup API
        with sqlite3.connect(db_path) as source, \
                sqlite3.connect(backup_file) as target:
            source.backup(target)

        logging.info(f"Backup created successfully: {backup_file}")
        return f"Backup created: {backup_file}"
    except Exception as e:
        error_msg = f"Failed to create backup: {str(e)}"
        logging.error(error_msg)
        return error_msg


def create_incremental_backup(db_path: str, backup_dir: str, db_name: str) -> str:
    """Create an incremental backup using VACUUM INTO."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir,
                                   f"{db_name}_incremental_{timestamp}.sqlib")

        with sqlite3.connect(db_path) as conn:
            conn.execute(f"VACUUM INTO '{backup_file}'")

        logging.info(f"Incremental backup created: {backup_file}")
        return f"Incremental backup created: {backup_file}"
    except Exception as e:
        error_msg = f"Failed to create incremental backup: {str(e)}"
        logging.error(error_msg)
        return error_msg


def list_backups(backup_dir: str) -> str:
    """List all available backups."""
    try:
        backups = [f for f in os.listdir(backup_dir)
                   if f.endswith(('.db', '.sqlib'))]
        backups.sort(reverse=True)  # Most recent first
        return "\n".join(backups) if backups else "No backups found"
    except Exception as e:
        error_msg = f"Failed to list backups: {str(e)}"
        logging.error(error_msg)
        return error_msg


def restore_single_db_backup(db_path: str, backup_dir: str, db_name: str, backup_name: str) -> str:
    """Restore database from a backup file."""
    try:
        logging.info(f"Restoring backup: {backup_name}")
        backup_path = os.path.join(backup_dir, backup_name)
        if not os.path.exists(backup_path):
            logging.error(f"Backup file not found: {backup_name}")
            return f"Backup file not found: {backup_name}"

        # Create a timestamp for the current db
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_backup = os.path.join(backup_dir,
                                      f"{db_name}_pre_restore_{timestamp}.db")

        # Backup current database before restore
        logging.info(f"Creating backup of current database: {current_backup}")
        shutil.copy2(db_path, current_backup)

        # Restore the backup
        logging.info(f"Restoring database from {backup_name}")
        shutil.copy2(backup_path, db_path)

        logging.info(f"Database restored from {backup_name}")
        return f"Database restored from {backup_name}"
    except Exception as e:
        error_msg = f"Failed to restore backup: {str(e)}"
        logging.error(error_msg)
        return error_msg


def setup_backup_config():
    """Setup configuration for database backups."""
    backup_base_dir = get_project_relative_path('tldw_DB_Backups')
    logging.info(f"Base backup directory: {os.path.abspath(backup_base_dir)}")

    # RAG Chat DB configuration
    rag_db_path = get_rag_qa_db_path()
    rag_backup_dir = os.path.join(backup_base_dir, 'rag_chat')
    os.makedirs(rag_backup_dir, exist_ok=True)
    logging.info(f"RAG backup directory: {os.path.abspath(rag_backup_dir)}")

    rag_db_config = {
        'db_path': rag_db_path,
        'backup_dir': rag_backup_dir,  # Make sure we use the full path
        'db_name': 'rag_qa'
    }

    # Character Chat DB configuration
    char_backup_dir = os.path.join(backup_base_dir, 'character_chat')
    os.makedirs(char_backup_dir, exist_ok=True)
    logging.info(f"Character backup directory: {os.path.abspath(char_backup_dir)}")

    char_db_config = {
        'db_path': chat_DB_PATH,
        'backup_dir': char_backup_dir,  # Make sure we use the full path
        'db_name': 'chatDB'
    }

    # Media DB configuration (based on your logs)
    media_backup_dir = os.path.join(backup_base_dir, 'media')
    os.makedirs(media_backup_dir, exist_ok=True)
    logging.info(f"Media backup directory: {os.path.abspath(media_backup_dir)}")

    media_db_config = {
        'db_path': os.path.join(os.path.dirname(chat_DB_PATH), 'server_media_summary.db'),
        'backup_dir': media_backup_dir,
        'db_name': 'media'
    }

    return rag_db_config, char_db_config, media_db_config

