#!/usr/bin/env python3

import logging
from pathlib import Path
import sys
from typing import Optional  # Added for type hinting

# --- Attempt to import questionary ---
try:
    import questionary
except ImportError:
    print("Error: The 'questionary' library is not installed. Please install it by running:")
    print("pip install questionary")
    sys.exit(1)

# --- Configuration for Prompts_DB_v2 library location ---
try:
    from Prompts_DB_v2 import PromptsDatabase, DatabaseError, InputError, ConflictError
except ImportError as e:
    script_dir = Path(__file__).resolve().parent
    print(f"Error: Could not import 'PromptsDatabase' from 'Prompts_DB_v2.py'.\n"
          f"Details: {e}\n"
          f"Please ensure 'Prompts_DB_v2.py' is in the same directory as this script ('{script_dir}') "
          f"or in a directory included in your PYTHONPATH.\n"
          f"Current sys.path: {sys.path}")
    sys.exit(1)

# Configure basic logging - level will be set based on user choice later
# Set a base level; we'll adjust the script's specific logger.
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)  # Get a logger for this script


def read_file_content(file_path: Path) -> Optional[str]:
    """
    Reads content from a text file if it exists.
    Returns None if the file doesn't exist or an error occurs during reading.
    """
    if file_path.is_file():
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None
    return None


def import_prompts_from_source(db_path: str, client_id: str, source_folder: Path,
                               default_keyword: str, default_author: Optional[str],
                               overwrite_existing: bool, use_readme_for_details: bool):
    """
    Imports prompts from the source_folder into the database.
    (This function remains largely the same as in the CLI version)
    """
    try:
        db = PromptsDatabase(db_path=db_path, client_id=client_id)
        logger.info(f"Successfully connected to database: {db_path}")
    except (DatabaseError, ValueError) as e:
        logger.error(f"Failed to initialize database at {db_path}: {e}")
        return

    if not source_folder.is_dir():
        logger.error(f"Source folder '{source_folder}' does not exist or is not a directory.")
        return

    imported_count = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0

    logger.info(f"Scanning source folder: {source_folder}")

    for item_path in source_folder.iterdir():
        if item_path.is_dir():
            prompt_name = item_path.name
            logger.debug(f"Processing potential prompt folder: {prompt_name}")

            system_prompt_file = item_path / "system.md"
            user_prompt_file = item_path / "user.md"
            readme_file = item_path / "README.md"

            system_prompt_content = read_file_content(system_prompt_file)
            user_prompt_content = read_file_content(user_prompt_file)

            details_content = None
            if use_readme_for_details:
                details_content = read_file_content(readme_file)

            if system_prompt_content is None and user_prompt_content is None:
                logger.warning(f"Skipping '{prompt_name}': No 'system.md' or 'user.md' found or readable.")
                skipped_count += 1
                continue

            keywords_to_add = []
            if default_keyword and default_keyword.strip():
                keywords_to_add.append(default_keyword.strip())

            try:
                prompt_id, prompt_uuid, message = db.add_prompt(
                    name=prompt_name,
                    author=default_author,
                    details=details_content,
                    system_prompt=system_prompt_content,
                    user_prompt=user_prompt_content,
                    keywords=keywords_to_add,
                    overwrite=overwrite_existing
                )

                if prompt_id is not None:
                    log_message = f"Prompt '{prompt_name}': {message} (ID: {prompt_id}, UUID: {prompt_uuid})"
                    if "added" in message.lower():
                        logger.info(log_message)
                        imported_count += 1
                    elif "updated" in message.lower():
                        logger.info(log_message)
                        updated_count += 1
                    elif "skipped" in message.lower() or "exists" in message.lower():
                        logger.info(log_message)
                        skipped_count += 1
                    else:
                        logger.info(log_message)
                else:
                    logger.warning(
                        f"Prompt '{prompt_name}': {message} (No ID returned, likely skipped or pre-existing without overwrite)")
                    if "skipped" in message.lower() or "exists" in message.lower():
                        skipped_count += 1
                    else:
                        error_count += 1

            except ConflictError as e:
                logger.info(
                    f"Skipped prompt '{prompt_name}' due to conflict (already exists and overwrite is false): {e}")
                skipped_count += 1
            except (InputError, DatabaseError) as e:
                logger.error(f"Failed to process prompt '{prompt_name}': {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing prompt '{prompt_name}': {e}",
                             exc_info=True)
                error_count += 1
        else:
            logger.debug(f"Skipping non-directory item in source folder: {item_path.name}")

    questionary.print("--------------------------------------------------", style="bold yellow")
    questionary.print("Import process finished.", style="bold green")
    questionary.print(f"  Prompts newly added: {imported_count}", style="fg:ansigreen")
    questionary.print(f"  Prompts updated:     {updated_count}", style="fg:ansicyan")
    questionary.print(f"  Prompts skipped:     {skipped_count}", style="fg:ansiyellow")
    questionary.print(f"  Errors encountered:  {error_count}", style="fg:ansired" if error_count > 0 else "")
    questionary.print("--------------------------------------------------", style="bold yellow")


def run_interactive_importer():
    """
    Runs the interactive prompt importer using questionary.
    """
    questionary.print("Welcome to the Interactive Prompt Importer!", style="bold italic fg:ansiblue")
    questionary.print("This script will guide you through importing prompts into your database.\n"
                      "Each direct subdirectory in the source folder you specify will be treated as a single prompt. "
                      "The subdirectory's name becomes the prompt's name. Inside each prompt subdirectory, "
                      "the script looks for 'system.md', 'user.md', and optionally 'README.md'.\n")

    questions = [
        {
            "type": "path",
            "name": "source_folder",
            "message": "Enter the path to the folder containing your prompt subdirectories (e.g., a cloned GitHub repo):",
            "only_directories": True,
            "validate": lambda p: Path(p).exists() or "Path does not exist. Please enter a valid directory path.",
        },
        {
            "type": "path",  # Using path for potential autocompletion and better UX
            "name": "db_path",
            "message": "Enter the path for your SQLite database file (e.g., ./my_prompts.db):",
            "validate": lambda p: Path(p).name.strip() != "" or "Database path cannot be empty.",
            "default": "./prompts.db"
        },
        {
            "type": "text",
            "name": "client_id",
            "message": "Enter a client ID for this import session (used for DB logging):",
            "default": "interactive_importer_session",
            "validate": lambda text: text.strip() != "" or "Client ID cannot be empty."
        },
        {
            "type": "text",
            "name": "keyword",
            "message": "Enter a keyword to associate with ALL imported prompts (e.g., 'from_github_collection'):",
            "validate": lambda text: text.strip() != "" or "Keyword cannot be empty."
        },
        {
            "type": "text",
            "name": "author",
            "message": "Enter the default author for these prompts (press Enter to use default):",
            "default": "Interactive Importer"
        },
        {
            "type": "confirm",
            "name": "overwrite",
            "message": "Overwrite prompts if they already exist in the database?",
            "default": False  # Safer default
        },
        {
            "type": "confirm",
            "name": "readme_as_details",
            "message": "Use content from README.md files (if found in prompt folders) as prompt 'details'?",
            "default": True
        },
        {
            "type": "confirm",
            "name": "verbose_logging",
            "message": "Enable detailed logging (shows more progress information during import)?",
            "default": True
        }
    ]

    try:
        answers = questionary.prompt(questions)
    except KeyboardInterrupt:
        questionary.print("\nImport process cancelled by user.", style="bold red")
        sys.exit(0)

    if not answers:  # User pressed Ctrl+C or Esc at the first question
        questionary.print("Import process cancelled.", style="bold red")
        return

    # Set logging level based on user choice
    if answers.get("verbose_logging", False):  # Default to False if key somehow missing
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # Resolve the source folder path to an absolute path
    source_path = Path(answers["source_folder"]).resolve()
    db_path = Path(answers["db_path"]).resolve()  # Resolve db_path as well for consistency

    import_prompts_from_source(
        db_path=str(db_path),  # PromptsDatabase expects string path
        client_id=answers["client_id"],
        source_folder=source_path,
        default_keyword=answers["keyword"],
        default_author=answers["author"] if answers["author"].strip() else None,
        overwrite_existing=answers["overwrite"],
        use_readme_for_details=answers["readme_as_details"]
    )


def main():
    run_interactive_importer()


if __name__ == "__main__":
    main()