# Backup_Functionality.py
# Functionality for managing database backups
#
# Imports:
import os
import shutil
import gradio as gr
from typing import Dict, List
#
# Local Imports:
from PoC_Version.App_Function_Libraries.DB.DB_Manager import create_automated_backup
from PoC_Version.App_Function_Libraries.DB.DB_Backups import create_backup, create_incremental_backup, restore_single_db_backup


#
# End of Imports
#######################################################################################################################
#
# Functions:

def get_db_specific_backups(backup_dir: str, db_name: str) -> List[str]:
    """Get list of backups specific to a database."""
    all_backups = [f for f in os.listdir(backup_dir) if f.endswith(('.db', '.sqlib'))]
    db_specific_backups = [
        backup for backup in all_backups
        if backup.startswith(f"{db_name}_")
    ]
    return sorted(db_specific_backups, reverse=True)  # Most recent first

def create_backup_tab(db_path: str, backup_dir: str, db_name: str):
    """Create the backup creation tab for a database."""
    gr.Markdown("## Create Database Backup")
    gr.Markdown(f"This will create a backup in the directory: `{backup_dir}`")
    with gr.Row():
        with gr.Column():
            #automated_backup_btn = gr.Button("Create Simple Backup")
            full_backup_btn = gr.Button("Create Full Backup")
            incr_backup_btn = gr.Button("Create Incremental Backup")
        with gr.Column():
            backup_output = gr.Textbox(label="Result")

    def create_db_backup():
        backup_file = create_automated_backup(db_path, backup_dir)
        return f"Backup created: {backup_file}"

    # automated_backup_btn.click(
    #     fn=create_db_backup,
    #     inputs=[],
    #     outputs=[backup_output]
    # )
    full_backup_btn.click(
        fn=lambda: create_backup(db_path, backup_dir, db_name),
        inputs=[],
        outputs=[backup_output]
    )
    incr_backup_btn.click(
        fn=lambda: create_incremental_backup(db_path, backup_dir, db_name),
        inputs=[],
        outputs=[backup_output]
    )

def create_view_backups_tab(backup_dir: str, db_name: str):
    """Create the backup viewing tab for a database."""
    gr.Markdown("## Available Backups")
    with gr.Row():
        with gr.Column():
            view_btn = gr.Button("Refresh Backup List")
        with gr.Column():
            backup_list = gr.Textbox(label="Available Backups")

    def list_db_backups():
        """List backups specific to this database."""
        backups = get_db_specific_backups(backup_dir, db_name)
        return "\n".join(backups) if backups else f"No backups found for {db_name} database"

    view_btn.click(
        fn=list_db_backups,
        inputs=[],
        outputs=[backup_list]
    )

def validate_backup_name(backup_name: str, db_name: str) -> bool:
    """Validate that the backup name matches the database being restored."""
    # Check if backup name starts with the database name prefix and has valid extension
    valid_prefixes = [
        f"{db_name}_backup_",    # Full backup prefix
        f"{db_name}_incremental_"  # Incremental backup prefix
    ]
    has_valid_prefix = any(backup_name.startswith(prefix) for prefix in valid_prefixes)
    has_valid_extension = backup_name.endswith(('.db', '.sqlib'))
    return has_valid_prefix and has_valid_extension

def create_restore_backup_tab(db_path: str, backup_dir: str, db_name: str):
    """Create the backup restoration tab for a database."""
    gr.Markdown("## Restore Database")
    gr.Markdown("⚠️ **Warning**: Restoring a backup will overwrite the current database.")
    with gr.Row():
        with gr.Column():
            backup_input = gr.Textbox(label="Backup Filename")
            restore_btn = gr.Button("Restore", variant="primary")
        with gr.Column():
            restore_output = gr.Textbox(label="Result")

    def secure_restore(backup_name: str) -> str:
        """Restore backup with validation checks."""
        if not backup_name:
            return "Please enter a backup filename"

        # Validate backup name format
        if not validate_backup_name(backup_name, db_name):
            return f"Invalid backup file. Please select a backup file that starts with '{db_name}_backup_' or '{db_name}_incremental_'"

        # Check if backup exists
        backup_path = os.path.join(backup_dir, backup_name)
        if not os.path.exists(backup_path):
            return f"Backup file not found: {backup_name}"

        # Proceed with restore
        return restore_single_db_backup(db_path, backup_dir, db_name, backup_name)

    restore_btn.click(
        fn=secure_restore,
        inputs=[backup_input],
        outputs=[restore_output]
    )

def create_media_db_tabs(db_config: Dict[str, str]):
    """Create all tabs for the Media database."""
    create_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='media'
    )
    create_view_backups_tab(
        backup_dir=db_config['backup_dir'],
        db_name='media'
    )
    create_restore_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='media'
    )

def create_rag_chat_tabs(db_config: Dict[str, str]):
    """Create all tabs for the RAG Chat database."""
    create_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='rag_qa'  # Updated to match DB_Manager.py
    )
    create_view_backups_tab(
        backup_dir=db_config['backup_dir'],
        db_name='rag_qa'  # Updated to match DB_Manager.py
    )
    create_restore_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='rag_qa'  # Updated to match DB_Manager.py
    )

def create_character_chat_tabs(db_config: Dict[str, str]):
    """Create all tabs for the Character Chat database."""
    create_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='chatDB'  # Updated to match DB_Manager.py
    )
    create_view_backups_tab(
        backup_dir=db_config['backup_dir'],
        db_name='chatDB'  # Updated to match DB_Manager.py
    )
    create_restore_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='chatDB'
    )

def create_database_management_interface(
        media_db_config: Dict[str, str],
        rag_db_config: Dict[str, str],
        char_db_config: Dict[str, str]
):
    """Create the main database management interface with tabs for each database."""
    with gr.TabItem("Media Database", id="media_db_group", visible=True):
        create_media_db_tabs(media_db_config)

    with gr.TabItem("RAG Chat Database", id="rag_chat_group", visible=True):
        create_rag_chat_tabs(rag_db_config)

    with gr.TabItem("Character Chat Database", id="character_chat_group", visible=True):
        create_character_chat_tabs(char_db_config)

#
# End of Functions
#######################################################################################################################
