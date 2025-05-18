# Backup_Functionality.py
# Functionality for exporting items as markdown files
#
# Imports:
import os
import shutil
import gradio as gr
#
# Local Imports:
from App_Function_Libraries.DB.DB_Manager import create_automated_backup, db_path, backup_dir
#
# End of Imports
#######################################################################################################################
#
# Functions:

def create_db_backup():
    backup_file = create_automated_backup(db_path, backup_dir)
    return f"Backup created: {backup_file}"


def list_backups():
    backups = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
    return "\n".join(backups)


def restore_backup(backup_name: str) -> str:
    backup_path_location: str = os.path.join(str(backup_dir), backup_name)
    if os.path.exists(backup_path_location):
        shutil.copy2(str(backup_path_location), str(db_path))
        return f"Database restored from {backup_name}"
    else:
        return "Backup file not found"


def create_backup_tab():
    with gr.Tab("Create Backup", visible=True):
        gr.Markdown("# Create a backup of the database")
        gr.Markdown("This will create a backup of the database in the backup directory(the default backup directory is `/tldw_DB_Backups/')")
        with gr.Row():
            with gr.Column():
                create_button = gr.Button("Create Backup")
                create_output = gr.Textbox(label="Result")
            with gr.Column():
                create_button.click(create_db_backup, inputs=[], outputs=create_output)


def create_restore_backup_tab():
    with gr.TabItem("Restore Backup", visible=True):
        gr.Markdown("# Restore a backup of the database")
        with gr.Column():
            backup_input = gr.Textbox(label="Backup Filename")
            restore_button = gr.Button("Restore")
        with gr.Column():
            restore_output = gr.Textbox(label="Result")
            restore_button.click(restore_backup, inputs=[backup_input], outputs=restore_output)

#
# End of Functions
#######################################################################################################################
