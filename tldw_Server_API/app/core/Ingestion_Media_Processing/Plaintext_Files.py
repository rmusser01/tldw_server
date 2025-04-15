# Plaintext_Files.py
# Description: This file contains functions for reading and writing plaintext files.
#
# Import necessary libraries
import json
import os
import tempfile
import zipfile
#
# External Imports
from docx2txt import docx2txt
from pypandoc import convert_file

from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_to_database
#
# Local Imports
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import perform_summarization
from tldw_Server_API.app.core.Utils.Utils import logging
#
#######################################################################################################################
#
# Function Definitions

def import_data(file, title, author, keywords, custom_prompt, summary, auto_summarize, api_name, api_key, system_prompt):
    logging.debug(f"Starting import_data with file: {file} / Title: {title} / Author: {author} / Keywords: {keywords}")
    if file is None:
        return "No file uploaded. Please upload a file."

    # We'll define this here so we can clean it up at the very end
    temp_file_path = None

    try:
        logging.debug(f"File object type: {type(file)}")
        logging.debug(f"File object attributes: {dir(file)}")

        if hasattr(file, 'name'):
            file_name = file.name
        else:
            file_name = 'unknown_file'

        # Create a temporary file for reading the content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            # Keep track of the full path so we can remove it later
            temp_file_path = temp_file.name

            if isinstance(file, str):
                # 'file' is actually a string of content
                temp_file.write(file)
            elif hasattr(file, 'read'):
                # 'file' is a file-like object
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                temp_file.write(content)
            else:
                # If neither a string nor file-like, force string conversion
                temp_file.write(str(file))

            temp_file.flush()  # Make sure data is written
            temp_file.seek(0)
            file_content = temp_file.read()

        logging.debug(f"File name: {file_name}")
        logging.debug(f"File content (first 100 chars): {file_content[:100]}")

        # Build info_dict for DB storage
        info_dict = {
            'title': title or 'Untitled',
            'uploader': author or 'Unknown',
        }

        # Prepare segments (right now just one segment for everything)
        # If you intend to chunk, you can do that here:
        segments = [{'Text': file_content}]

        # Process keywords into a list
        if keywords:
            keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        else:
            keyword_list = []

        # If auto-summarize is enabled and we have an API, do summarization
        if auto_summarize and api_name and api_key:
            # FIXME - Make sure custom_prompt is system prompt
            summary = perform_summarization(api_name, file_content, custom_prompt, api_key, False, None, system_prompt)
        # If there's no user-provided summary, and we haven't auto-summarized:
        elif not summary:
            summary = "No analysis provided"

        # --- ALWAYS add to database after we've finalized `summary` ---
        result = add_media_to_database(
            url=file_name,             # or any unique identifier
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=system_prompt + "\n\nCustom Prompt:\n\n" + custom_prompt,
            whisper_model="Imported",  # indicates it was an imported file
            media_type="document",
            overwrite=False
        )

        return f"File '{file_name}' import attempt complete. Database result: {result}"

    except Exception as e:
        logging.exception(f"Error importing file: {str(e)}")
        return f"Error importing file: {str(e)}"

    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


###############################################################
#
# Plaintext/Markdown/RTF/Docx Import Functionality

def preview_import_handler(
        files,
        author,
        keywords,
        system_prompt,
        user_prompt,
        auto_summarize,
        api_name,
        api_key
):
    """
    Step 1: Read/convert files (or ZIP of multiple text files) + optionally auto-summarize,
    but DO NOT store in the DB.

    Returns:
      - A user-facing status string
      - A JSON string (preview_data_json) containing a list of file results:
          [
            {
               "filename": "...",
               "title": "...",
               "content": "...",
               "summary": "...",
               "author": "...",
               "keywords": [...],
               "system_prompt": "...",
               "user_prompt": "...",
               ...
            },
            ...
          ]
    """
    if not files:
        return "No files uploaded.", None

    results_for_ui = []
    preview_list = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file_obj in files:
            filename = os.path.basename(file_obj.name)
            try:
                # Make a temp copy
                temp_path = os.path.join(temp_dir, filename)
                with open(temp_path, 'wb') as f:
                    f.write(open(file_obj.name, 'rb').read())

                # If the file is a ZIP, extract and preview each valid item
                if temp_path.lower().endswith('.zip'):
                    with tempfile.TemporaryDirectory() as zip_temp_dir:
                        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                            zip_ref.extractall(zip_temp_dir)

                        for root, _, extracted_files in os.walk(zip_temp_dir):
                            for extracted_filename in extracted_files:
                                if extracted_filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                                    extracted_path = os.path.join(root, extracted_filename)
                                    file_info = _preview_single_file(
                                        extracted_path,
                                        author,
                                        keywords,
                                        system_prompt,
                                        user_prompt,
                                        auto_summarize,
                                        api_name,
                                        api_key
                                    )
                                    preview_list.append(file_info)
                                    results_for_ui.append(f"📄 [ZIP] {extracted_filename} => Success")

                    results_for_ui.append(f"📦 {filename} => Extracted successfully.")

                else:
                    # Single file scenario
                    file_info = _preview_single_file(
                        temp_path,
                        author,
                        keywords,
                        system_prompt,
                        user_prompt,
                        auto_summarize,
                        api_name,
                        api_key
                    )
                    preview_list.append(file_info)
                    results_for_ui.append(f"📄 {filename} => Success")

            except Exception as e:
                logging.exception(f"❌ Error with file: {filename}")
                results_for_ui.append(f"❌ {filename} => {str(e)}")

    # Convert list of file info dicts to JSON so we can store in gr.State or similar
    preview_data_json = json.dumps(preview_list, ensure_ascii=True)
    status_message = "\n".join(results_for_ui)

    return status_message, preview_data_json


def _preview_single_file(
        file_path,
        author,
        keywords,
        system_prompt,
        user_prompt,
        auto_summarize,
        api_name,
        api_key
):
    """
    Internal helper to read/convert a single file into plain text,
    optionally auto-summarize, and return a dictionary describing the
    would-be DB record (but does not ingest).
    """
    log_counter("file_preview_attempt", labels={"file_path": file_path})

    # Derive a filename-based title
    filename = os.path.basename(file_path)
    title = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1].lower()

    # 1) Read/convert content
    try:
        if extension == '.rtf':
            with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_md:
                convert_file(file_path, 'md', outputfile=temp_md.name)
                file_path_md = temp_md.name
            with open(file_path_md, 'r', encoding='utf-8') as f:
                content = f.read()
        elif extension == '.docx':
            content = docx2txt.process(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except Exception as e:
        logging.error(f"Error reading file content: {str(e)}")
        return {
            "filename": filename,
            "title": title,
            "content": f"Error reading file: {e}",
            "summary": None,
            "author": author,
            "keywords": keywords.split(",") if keywords else [],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "auto_summarize": auto_summarize,
            "api_name": api_name,
            "api_key": api_key,
        }

    # 2) Optionally auto-summarize
    summary = None
    if auto_summarize and api_name and api_key:
        combined_prompt = (system_prompt or "") + "\n\n" + (user_prompt or "")
        summary = perform_summarization(
            api_name=api_name,
            input_data=content,
            custom_prompt=combined_prompt,
            api_key=api_key,
            recursive_summarization=False,
        )

    if not summary:
        summary = "No summary provided"

    # 3) Return the file info dict (not ingested yet)
    return {
        "filename": filename,
        "title": title,
        "content": content,
        "summary": summary,
        "author": author,
        "keywords": keywords.split(",") if keywords else [],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "auto_summarize": auto_summarize,
        "api_name": api_name,
        "api_key": api_key
    }


def final_ingest_handler(preview_data_json, updated_metadata_json):
    """
    Step 2: Actually ingest data into the database using add_media_to_database.

    - preview_data_json: The JSON output from preview_import_handler()
    - updated_metadata_json: (Optional) JSON from the user specifying
        overrides for each file.

    Returns a status string (success/fail for each file).
    """
    if not preview_data_json:
        return "No preview data found. Please run the preview step first."

    try:
        preview_list = json.loads(preview_data_json)
    except Exception as e:
        logging.exception("Error loading preview data.")
        return f"Error parsing preview data: {e}"

    # Parse user-supplied overrides (if any)
    if updated_metadata_json:
        try:
            overrides_dict = json.loads(updated_metadata_json)
        except Exception as e:
            logging.exception("Error loading user metadata overrides.")
            overrides_dict = {}
    else:
        overrides_dict = {}

    results = []
    for file_info in preview_list:
        fname = file_info["filename"]
        # Attempt to match user overrides by filename (the base name without extension, or the full fname).
        # Typically the user might key by "Doc1" vs "Doc1.txt". Decide how you want to match.
        # Here we assume the user’s JSON keys match exactly the 'filename' in file_info.
        this_file_overrides = overrides_dict.get(fname, {})

        # Combine final metadata
        final_author = this_file_overrides.get("author", file_info["author"])
        final_keywords = this_file_overrides.get("keywords", file_info["keywords"])
        final_title = this_file_overrides.get("title", file_info["title"])
        final_summary = this_file_overrides.get("summary", file_info["summary"])

        text_content = file_info["content"]  # The converted text

        # Construct combined prompts if needed or just store them
        combined_prompt = (file_info["system_prompt"] or "") + "\n\n" + (file_info["user_prompt"] or "")

        # Now do the actual DB ingestion
        try:
            db_result = add_media_to_database(
                url=fname,  # or some unique identifier
                info_dict={
                    "title": final_title,
                    "uploader": final_author,
                },
                segments=[{"Text": text_content}],
                summary=final_summary,
                keywords=final_keywords,
                custom_prompt_input=combined_prompt,
                whisper_model="Imported",
                media_type="document",
                overwrite=False
            )
            results.append(f"✅ {fname} => {db_result}")
        except Exception as e:
            logging.exception(f"Error ingesting file {fname}")
            results.append(f"❌ {fname} => {str(e)}")

    # Return an overall string
    return "\n".join(results)

#
# End of Plaintext/Markdown/RTF/Docx Import Functionality
###############################################################



###############################################################
#
# End of Plaintext/Markdown/RTF/Docx Import Functionality

def import_plain_text_file(file_path, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    """Import a single plain text file."""
    try:
        log_counter("file_processing_attempt", labels={"file_path": file_path})

        # Extract title from filename
        title = os.path.splitext(os.path.basename(file_path))[0]

        # Determine the file type and convert if necessary
        file_extension = os.path.splitext(file_path)[1].lower()

        # Get the content based on file type
        try:
            if file_extension == '.rtf':
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    convert_file(file_path, 'md', outputfile=temp_file.name)
                    file_path = temp_file.name
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                log_counter("rtf_conversion_success", labels={"file_path": file_path})
            elif file_extension == '.docx':
                content = docx2txt.process(file_path)
                log_counter("docx_conversion_success", labels={"file_path": file_path})
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
        except Exception as e:
            logging.error(f"Error reading file content: {str(e)}")
            return f"Error reading file content: {str(e)}"

        # Import the content
        result = import_data(
            content,  # Pass the content directly
            title,
            author,
            keywords,
            user_prompt,  # This is the custom_prompt parameter
            None,        # No summary - let auto_summarize handle it
            auto_summarize,
            api_name,
            api_key,
            system_prompt
        )

        log_counter("file_processing_success", labels={"file_path": file_path})
        return result

    except Exception as e:
        logging.exception(f"Error processing file {file_path}")
        log_counter("file_processing_error", labels={"file_path": file_path, "error": str(e)})
        return f"Error processing file {os.path.basename(file_path)}: {str(e)}"


def process_plain_text_zip_file(zip_file, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    """Process multiple text files from a zip archive."""
    results = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for filename in os.listdir(temp_dir):
                if filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                    file_path = os.path.join(temp_dir, filename)
                    result = import_plain_text_file(
                        file_path=file_path,
                        author=author,
                        keywords=keywords,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        auto_summarize=auto_summarize,
                        api_name=api_name,
                        api_key=api_key
                    )
                    results.append(f"📄 {filename}: {result}")

        return "\n\n".join(results)
    except Exception as e:
        logging.exception(f"Error processing zip file: {str(e)}")
        return f"Error processing zip file: {str(e)}"


def import_file_handler(files, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    """
    Handle file imports with proper temp file handling.
    This function is wired to the Gradio import button; it must return a single string
    if we only have one output in the UI.
    """
    try:
        if not files:
            return "No files uploaded."

        results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_obj in files:
                filename = os.path.basename(file_obj.name)
                try:
                    # Make a temporary copy
                    temp_path = os.path.join(temp_dir, filename)
                    with open(temp_path, 'wb') as f:
                        f.write(open(file_obj.name, 'rb').read())

                    # Check if it's a zip
                    if temp_path.lower().endswith('.zip'):
                        with tempfile.TemporaryDirectory() as zip_temp_dir:
                            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                                zip_ref.extractall(zip_temp_dir)

                            file_count = 0
                            for root, _, extracted_files in os.walk(zip_temp_dir):
                                for extracted_filename in extracted_files:
                                    if extracted_filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                                        extracted_path = os.path.join(root, extracted_filename)
                                        # Import each file inside the zip
                                        result = import_plain_text_file(
                                            extracted_path,
                                            author,
                                            keywords,
                                            system_prompt,
                                            user_prompt,
                                            auto_summarize,
                                            api_name,
                                            api_key
                                        )
                                        results.append(f"📄 {extracted_filename}: {result}")
                                        file_count += 1

                            results.append(f"📦 {filename} => Processed {file_count} file(s).")

                    else:
                        # Single file import
                        result = import_plain_text_file(
                            temp_path,
                            author,
                            keywords,
                            system_prompt,
                            user_prompt,
                            auto_summarize,
                            api_name,
                            api_key
                        )
                        results.append(f"📄 {filename}: {result}")

                except Exception as e:
                    logging.exception(f"❌ Error with file: {filename}")
                    results.append(f"❌ {filename} => {str(e)}")

        return "\n".join(results)

    except Exception as e:
        logging.exception(f"❌ Error during import: {str(e)}")
        return f"❌ Error during import: {str(e)}"


#
# End of Plaintext_Files.py
#######################################################################################################################

