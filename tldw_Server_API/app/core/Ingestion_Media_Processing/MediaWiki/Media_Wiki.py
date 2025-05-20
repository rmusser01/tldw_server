# Media_Wiki.py
# Description: This file contains the functions to import MediaWiki dumps into the media_db and Chroma databases.
#######################################################################################################################
#
# Imports
import json
import os
import re
import sys
import traceback
from typing import List, Dict, Any, Iterator, Optional, Union
from datetime import datetime, timezone  # Added for default ingestion_date
#
# 3rd-Party Imports
from loguru import logger
import mwparserfromhell
import mwxml
import yaml
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import process_and_store_content
from tldw_Server_API.app.core.Utils.Utils import logging
#
#######################################################################################################################
#
# Functions:
# Load configuration
def load_mediawiki_import_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Config_Files', 'mediawiki_import_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


media_wiki_import_config = load_mediawiki_import_config()


def setup_media_wiki_logger(name: str, level: Union[int, str] = "INFO", log_file: Optional[str] = None) -> None:
    """Set up the logger with the given name and level."""
    logger.remove()
    logger.add(sys.stdout, format="{time} - {name} - {level} - {message}", level=level)
    if log_file:
        logger.add(log_file, format="{time} - {name} - {level} - {message}", level=level)


setup_media_wiki_logger('mediawiki_import', log_file='./Logs/mediawiki_import.log')


#
#######################################################################################################################
#
# Functions:

def parse_mediawiki_dump(file_path: str, namespaces: List[int] = None, skip_redirects: bool = False) -> Iterator[
    Dict[str, Any]]:
    dump = mwxml.Dump.from_file(open(file_path, encoding='utf-8'))
    for page in dump.pages:
        if skip_redirects and page.redirect:
            continue
        if namespaces and page.namespace not in namespaces:
            continue

        for revision in page:  # mwxml revisions are an iterator
            wikicode = mwparserfromhell.parse(revision.text or "")  # Ensure text is not None
            plain_text = wikicode.strip_code()
            # Ensure timestamp is timezone-aware or consistently formatted if naive
            timestamp_obj = revision.timestamp
            if timestamp_obj:
                # If naive, assume UTC or make it configurable
                # For simplicity, if naive, we'll assume it's UTC. Best practice is for mwxml to provide tz-aware.
                if timestamp_obj.tzinfo is None:
                    timestamp_obj = timestamp_obj.replace(tzinfo=timezone.utc)
            else:  # Fallback timestamp if revision has none
                timestamp_obj = datetime.now(timezone.utc)

            yield {
                "title": page.title,
                "content": plain_text,
                "namespace": page.namespace,
                "page_id": page.id,
                "revision_id": revision.id,
                "timestamp": timestamp_obj  # Store as datetime object
            }
        logging.debug(f"Yielded page: {page.title}")


def optimized_chunking(text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Using simple newline splitting for sections as an example.
    # Your original implementation used re.split(r'\n==\s*(.*?)\s*==\n', text)
    # which is good for MediaWiki section syntax.
    # This function should produce a list of dictionaries, e.g.,
    # [{"text": "chunk text 1", "metadata": {"section_title": "Introduction", ...}}, ...]

    max_size = chunk_options.get('max_size', media_wiki_import_config.get('chunking', {}).get('default_size', 1000))
    # Fallback to simple splitting if no section-based logic is defined or needed
    # For this example, we'll just split by paragraphs if no section logic from original needed.
    # Your original `optimized_chunking` was section-aware. Let's keep that spirit.
    # If `text` is None or empty, handle gracefully
    if not text:
        return []

    sections = re.split(r'(\n==\s*[^=]+?\s*==\n)', text)  # Keep delimiters
    chunks = []
    current_chunk_text = ""
    current_section_title = "Introduction"  # Default for content before first heading

    # If the text doesn't start with a section, the first part is 'Introduction'
    if sections and not sections[0].startswith("\n=="):
        first_content = sections.pop(0).strip()
        if first_content:
            # If current_chunk_text + first_content is too large, start a new chunk
            if len(current_chunk_text) + len(first_content) > max_size and current_chunk_text:
                chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})
                current_chunk_text = first_content
            else:
                current_chunk_text += ("\n" if current_chunk_text else "") + first_content
    else:  # Text might start with a delimiter or be empty
        if sections and sections[0].strip() == "":  # Handle empty first element from split
            sections.pop(0)

    for i in range(0, len(sections), 2):
        header_part = sections[i].strip() if i < len(sections) else ""
        content_part = sections[i + 1].strip() if i + 1 < len(sections) else ""

        if header_part.startswith("==") and header_part.endswith("=="):
            new_section_title = header_part.strip("= \n")
            if current_chunk_text.strip():  # If there's content for the previous section, store it
                chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})
            current_chunk_text = ""  # Reset for new section
            current_section_title = new_section_title

        if content_part:  # Add content to current section's chunk
            # Further split content_part if it alone exceeds max_size (simplified here)
            if len(current_chunk_text) + len(content_part) > max_size and current_chunk_text.strip():
                chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})
                current_chunk_text = content_part
            else:
                current_chunk_text += ("\n" if current_chunk_text else "") + content_part

    # Add any remaining text
    if current_chunk_text.strip():
        chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})

    # If no chunks were created (e.g. empty input text), return empty list
    if not chunks and text.strip():  # If text was not empty but no chunks (edge case)
        chunks.append({"text": text.strip(), "metadata": {"section": "Full Text"}})

    logging.debug(f"optimized_chunking: Created {len(chunks)} chunks.")
    return chunks


def process_single_item(
        content: str,
        title: str,
        wiki_name: str,
        chunk_options: Dict[str, Any],
        item: Dict[str, Any],  # Contains timestamp, page_id etc. from parse_mediawiki_dump
        store_to_db: bool = True,
        store_to_vector_db: bool = True,
        api_name_vector_db: Optional[str] = None,
        api_key_vector_db: Optional[str] = None
) -> Dict[str, Any]:
    try:
        logging.debug(
            f"process_single_item: Processing item: {title} (StoreDB: {store_to_db}, StoreVector: {store_to_vector_db})")

        # Ensure timestamp is a datetime object for strftime, or handle if it's already string
        timestamp_dt = item.get("timestamp")
        if isinstance(timestamp_dt, str):
            try:
                timestamp_dt = datetime.fromisoformat(timestamp_dt.replace('Z', '+00:00'))
            except ValueError:
                timestamp_dt = datetime.now(timezone.utc)  # Fallback
        elif not isinstance(timestamp_dt, datetime):
            timestamp_dt = datetime.now(timezone.utc)  # Fallback

        iso_timestamp_str = timestamp_dt.isoformat()

        processed_data = {
            "title": title,
            "content": content,
            "namespace": item.get("namespace"),
            "page_id": item.get("page_id"),
            "revision_id": item.get("revision_id"),
            "timestamp": iso_timestamp_str,
            "chunks": [],
            "media_id": None,
            "message": "",
            "status": "Pending"
        }

        chunks = optimized_chunking(content, chunk_options)
        processed_data["chunks"] = chunks

        media_id = None
        if store_to_db:
            encoded_title = title.replace(" ", "_").replace("/", "_")  # Sanitize for URL part
            url = f"mediawiki:{wiki_name}:{encoded_title}"
            logging.debug(f"Generated Media URL: {url}")

            # Ensure ingestion_date is a string in 'YYYY-MM-DD' format
            ingestion_date_str = timestamp_dt.strftime('%Y-%m-%d')

            result = MediaDatabase.add_media_with_keywords(  # This function is from App_Function_Libraries.DB.DB_Manager
                url=url,
                title=title,
                media_type="mediawiki_page",  # Adjusted type
                content=content,
                keywords=f"mediawiki,{wiki_name},page",
                prompt="",
                analysis_content="",  # Analysis/summary would be separate
                transcription_model="N/A",
                author="MediaWiki",  # Or parse from page if possible
                ingestion_date=ingestion_date_str
            )
            # Assuming add_media_with_keywords returns (media_id, message)
            # If it returns the full DB record or just ID, adapt here.
            # Let's assume it's (media_id, message) as per your original code.
            if isinstance(result, tuple) and len(result) == 2:
                media_id, message = result
            else:  # Fallback if structure is different
                media_id = result if isinstance(result, int) else None
                message = "DB operation status unknown" if media_id else "DB operation failed"

            processed_data["media_id"] = media_id
            processed_data["message"] = message
            logging.info(f"Media item DB result for '{title}': ID={media_id}, Msg='{message}'")

        if store_to_vector_db and media_id:
            if not api_name_vector_db:
                logging.warning(f"Vector DB API name not provided for '{title}', skipping vector storage.")
                processed_data["message"] += " Skipped vector storage (no API name)."
            else:
                for i, chunk_dict in enumerate(chunks):
                    logging.debug(f"Storing chunk {i + 1}/{len(chunks)} for item: {title} to vector DB.")
                    try:
                        # process_and_store_content(content: str, collection_name: str, media_id: int, file_name: str,
                        #                           create_embeddings: bool = False, create_summary: bool = False,
                        #                           api_name: str = None, api_key: str = None):
                        process_and_store_content(
                            chunk_dict['text'],
                            f"mediawiki_{wiki_name}",
                            media_id,
                            title,  # Use page title as file_name context for vector DB
                            create_embeddings=True,
                            create_summary=True,  # Set to True if you want summaries per chunk via LLM
                            api_name=api_name_vector_db,
                            api_key=api_key_vector_db  # Pass the API key
                        )
                    except Exception as e_vec:
                        logging.error(f"Failed to store chunk {i + 1} for '{title}' to vector DB: {e_vec}")
                        processed_data["message"] += f" Error storing chunk {i + 1} to vector DB."
                        # Decide if this makes the whole item an error or just a warning
        elif store_to_vector_db and not media_id:
            logging.warning(
                f"Cannot store to vector DB for '{title}': media_id is missing (store_to_db may be False or failed).")
            processed_data["message"] += " Skipped vector storage (media_id missing)."

        processed_data["status"] = "Success" if media_id or not store_to_db else "Error"
        if media_id is None and store_to_db:  # If we intended to store but failed
            processed_data["message"] = processed_data.get("message", "") + " Failed to store media item to primary DB."

        logging.info(f"Successfully processed item '{title}' (Status: {processed_data['status']})")
        return processed_data

    except Exception as e:
        logging.error(f"Error processing item {title}: {str(e)}")
        logging.error(f"Exception details: {traceback.format_exc()}")
        # Ensure all keys from 'processed_data' are present in error return
        timestamp_val = item.get("timestamp")
        if isinstance(timestamp_val, datetime):
            iso_timestamp_str_err = timestamp_val.isoformat()
        elif isinstance(timestamp_val, str):
            iso_timestamp_str_err = timestamp_val  # assume already iso
        else:
            iso_timestamp_str_err = datetime.now(timezone.utc).isoformat()

        return {
            "title": title, "status": "Error", "error_message": str(e), "chunks": [],
            "content": content,  # content might be available even if processing fails later
            "namespace": item.get("namespace"), "page_id": item.get("page_id"),
            "revision_id": item.get("revision_id"), "timestamp": iso_timestamp_str_err,
            "media_id": None, "message": f"Failed to process: {str(e)}"
        }


def load_checkpoint(file_path: str) -> int:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('last_processed_id', 0)
        except json.JSONDecodeError:
            logging.warning(f"Checkpoint file {file_path} is corrupted. Starting from beginning.")
            return 0
    return 0


def save_checkpoint(file_path: str, last_processed_id: int):
    with open(file_path, 'w') as f:
        json.dump({'last_processed_id': last_processed_id}, f)


def import_mediawiki_dump(
        file_path: str,
        wiki_name: str,
        namespaces: List[int] = None,
        skip_redirects: bool = False,
        chunk_options_override: Dict[str, Any] = None,
        progress_callback: Any = None,
        store_to_db: bool = True,
        store_to_vector_db: bool = True,
        api_name_vector_db: Optional[str] = None,
        api_key_vector_db: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    try:
        logging.info(
            f"Importing MediaWiki dump: {file_path} for wiki: {wiki_name}. StoreDB: {store_to_db}, StoreVector: {store_to_vector_db}")
        final_chunk_options = chunk_options_override if chunk_options_override else media_wiki_import_config.get(
            'chunking', {})

        checkpoint_file = f"{wiki_name}_import_checkpoint.json"
        last_processed_id = 0
        if store_to_db:  # Checkpoints only make sense if we are saving progress to DB
            last_processed_id = load_checkpoint(checkpoint_file)

        total_pages = count_pages(file_path, namespaces, skip_redirects)
        processed_pages_count = 0

        yield {"type": "progress_total", "total_pages": total_pages,
               "message": f"Found {total_pages} pages to process for '{wiki_name}'."}

        for item_dict in parse_mediawiki_dump(file_path, namespaces, skip_redirects):
            current_page_id = item_dict.get('page_id', 0)
            current_title = item_dict.get('title', 'Unknown Title')

            if store_to_db and current_page_id <= last_processed_id:
                processed_pages_count += 1
                if progress_callback:
                    progress_callback(processed_pages_count / total_pages if total_pages > 0 else 0,
                                      f"Skipped (checkpoint): {current_title}")
                yield {"type": "progress_item", "status": "skipped_checkpoint", "title": current_title,
                       "page_id": current_page_id,
                       "progress_percent": processed_pages_count / total_pages if total_pages > 0 else 0}
                continue

            processed_item_details = process_single_item(
                content=item_dict['content'],
                title=current_title,
                wiki_name=wiki_name,
                chunk_options=final_chunk_options,
                item=item_dict,  # Pass the full dict from parse_mediawiki_dump
                store_to_db=store_to_db,
                store_to_vector_db=store_to_vector_db,
                api_name_vector_db=api_name_vector_db,
                api_key_vector_db=api_key_vector_db
            )

            if store_to_db and processed_item_details.get("status") == "Success" and processed_item_details.get(
                    "media_id") is not None:
                save_checkpoint(checkpoint_file, current_page_id)

            processed_pages_count += 1
            current_progress_percent = processed_pages_count / total_pages if total_pages > 0 else 0
            if progress_callback:
                progress_callback(current_progress_percent, f"Processed page: {current_title}")

            # Yield detailed result for each page, including its processing status
            yield {"type": "item_result", "data": processed_item_details, "progress_percent": current_progress_percent}

        if store_to_db and os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                logging.info(f"Successfully removed checkpoint file: {checkpoint_file}")
            except OSError as e:
                logging.warning(f"Could not remove checkpoint file {checkpoint_file}: {e}")

        yield {"type": "summary",
               "message": f"Successfully processed MediaWiki dump: {wiki_name}. Processed {processed_pages_count}/{total_pages} pages."}

    except FileNotFoundError:
        logger.error(f"MediaWiki dump file not found: {file_path}")
        yield {"type": "error", "message": f"Error: File not found - {file_path}"}
    except PermissionError:
        logger.error(f"Permission denied when trying to read: {file_path}")
        yield {"type": "error", "message": f"Error: Permission denied - {file_path}"}
    except Exception as e:
        logger.exception(f"Error during MediaWiki import: {str(e)}")
        yield {"type": "error", "message": f"Error during import: {str(e)}"}


def count_pages(file_path: str, namespaces: List[int] = None, skip_redirects: bool = False) -> int:
    count = 0
    try:
        dump = mwxml.Dump.from_file(open(file_path, encoding='utf-8'))
        for page in dump.pages:
            if skip_redirects and page.redirect:
                continue
            if namespaces and page.namespace not in namespaces:
                continue
            count += 1
    except Exception as e:
        logger.error(f"Error counting pages in MediaWiki dump {file_path}: {str(e)}", exc_info=True)
        return 0  # Return 0 if counting fails
    return count

#
# End of Media_Wiki.py
#######################################################################################################################
