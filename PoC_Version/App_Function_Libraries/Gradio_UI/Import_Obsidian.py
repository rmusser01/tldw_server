# Import_Obsidian.py
# Functionality to import Obsidian Vault content into the DB
#
# Imports
import os
import re
import shutil
import tempfile
import traceback
import zipfile
from time import sleep
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import import_obsidian_note_to_db
from App_Function_Libraries.Utils.Utils import logger
#
###############################################################
#
# Obsidian Vault Import Functionality

def process_obsidian_zip(zip_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            imported_files, total_files, errors = import_obsidian_vault(temp_dir)

            return imported_files, total_files, errors
        except zipfile.BadZipFile:
            error_msg = "The uploaded file is not a valid zip file."
            logger.error(error_msg)
            return 0, 0, [error_msg]
        except Exception as e:
            error_msg = f"Error processing zip file: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return 0, 0, [error_msg]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# FIXME - File path issue
def scan_obsidian_vault(vault_path):
    markdown_files = []
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files


def parse_obsidian_note(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    frontmatter = {}
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        import yaml
        frontmatter = yaml.safe_load(frontmatter_text)
        content = content[frontmatter_match.end():]

    tags = re.findall(r'#(\w+)', content)
    links = re.findall(r'\[\[(.*?)\]\]', content)

    return {
        'title': os.path.basename(file_path).replace('.md', ''),
        'content': content,
        'frontmatter': frontmatter,
        'tags': tags,
        'links': links,
        'file_path': file_path  # Add this line
    }


def import_vault(vault_path, vault_zip):
    if vault_zip:
        imported, total, errors = process_obsidian_zip(vault_zip.name)
    elif vault_path:
        imported, total, errors = import_obsidian_vault(vault_path)
    else:
        return "Please provide either a local vault path or upload a zip file."

    status = f"Imported {imported} out of {total} files.\n"
    if errors:
        status += f"Encountered {len(errors)} errors:\n" + "\n".join(errors)
    return status


def create_import_obsidian_vault_tab():
    with gr.TabItem("Import Obsidian Vault", visible=True):
        gr.Markdown("## Import Obsidian Vault")
        with gr.Row():
            with gr.Column():
                vault_path_input = gr.Textbox(label="Obsidian Vault Path (Local)")
                vault_zip_input = gr.File(label="Upload Obsidian Vault (Zip)")
            with gr.Column():
                import_vault_button = gr.Button("Import Obsidian Vault")
                import_status = gr.Textbox(label="Import Status", interactive=False)

    import_vault_button.click(
        fn=import_vault,
        inputs=[vault_path_input, vault_zip_input],
        outputs=[import_status],
    )


def import_obsidian_vault(vault_path, progress=gr.Progress()):
    try:
        markdown_files = scan_obsidian_vault(vault_path)
        total_files = len(markdown_files)
        imported_files = 0
        errors = []

        for i, file_path in enumerate(markdown_files):
            try:
                note_data = parse_obsidian_note(file_path)
                success, error_msg = import_obsidian_note_to_db(note_data)
                if success:
                    imported_files += 1
                else:
                    errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

            progress((i + 1) / total_files, f"Imported {imported_files} of {total_files} files")
            sleep(0.1)  # Small delay to prevent UI freezing

        return imported_files, total_files, errors
    except Exception as e:
        error_msg = f"Error scanning vault: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return 0, 0, [error_msg]

#
# End of Obsidian Vault Import Functionality
#######################################################################################################################
