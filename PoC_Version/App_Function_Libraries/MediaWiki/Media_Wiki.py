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
# 3rd-Party Imports
from loguru import logger
import mwparserfromhell
import mwxml
import yaml
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
from PoC_Version.App_Function_Libraries.RAG.ChromaDB_Library import process_and_store_content
from PoC_Version.App_Function_Libraries.Utils.Utils import logging
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
    logger.remove()  # Remove the default logger
    logger.add(sys.stdout, format="{time} - {name} - {level} - {message}", level=level)

    if log_file:
        logger.add(log_file, format="{time} - {name} - {level} - {message}", level=level)

# Usage
setup_media_wiki_logger('mediawiki_import', log_file='./Logs/mediawiki_import.log')

# End of setup
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

        for revision in page:
            wikicode = mwparserfromhell.parse(revision.text)
            plain_text = wikicode.strip_code()
            yield {
                "title": page.title,
                "content": plain_text,
                "namespace": page.namespace,
                "page_id": page.id,
                "revision_id": revision.id,
                "timestamp": revision.timestamp
            }
        logging.debug(f"Yielded page: {page.title}")


def optimized_chunking(text: str, chunk_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections = re.split(r'\n==\s*(.*?)\s*==\n', text)
    chunks = []
    current_chunk = ""
    current_size = 0

    logging.debug(f"optimized_chunking: Processing text with {len(sections) // 2} sections")
    for i in range(0, len(sections), 2):
        section_title = sections[i] if i > 0 else "Introduction"
        section_content = sections[i + 1] if i + 1 < len(sections) else ""

        if current_size + len(section_content) > chunk_options['max_size']:
            if current_chunk:
                chunks.append({"text": current_chunk, "metadata": {"section": section_title}})
            current_chunk = section_content
            current_size = len(section_content)
        else:
            current_chunk += f"\n== {section_title} ==\n" + section_content
            current_size += len(section_content)

    if current_chunk:
        chunks.append({"text": current_chunk, "metadata": {"section": "End"}})

    return chunks





def process_single_item(content: str, title: str, wiki_name: str, chunk_options: Dict[str, Any],
                        is_combined: bool = False, item: Dict[str, Any] = None, api_name: str = None):
    try:
        logging.debug(f"process_single_item: Processing item: {title}")

        # Create a unique URL using the wiki name and article title
        encoded_title = title.replace(" ", "_")
        url = f"mediawiki:{wiki_name}:{encoded_title}"
        logging.debug(f"Generated URL: {url}")

        result = add_media_with_keywords(
            url=url,  # Use the generated URL here
            title=title,
            media_type="mediawiki_dump" if is_combined else "mediawiki_article",
            content=content,
            keywords=f"mediawiki,{wiki_name}" + (",full_dump" if is_combined else ",article"),
            prompt="",
            summary="",
            transcription_model="",
            author="MediaWiki",
            ingestion_date=item['timestamp'].strftime('%Y-%m-%d') if item else None
        )
        logging.debug(f"Result from add_media_with_keywords: {result}")

        # Unpack the result
        media_id, message = result
        logging.info(f"Media item result: {message}")
        logging.debug(f"Final media_id: {media_id}")

        chunks = optimized_chunking(content, chunk_options)
        for i, chunk in enumerate(chunks):
            logging.debug(f"Processing chunk {i + 1}/{len(chunks)} for item: {title}")

            # FIXME
            # def process_and_store_content(content: str, collection_name: str, media_id: int, file_name: str,
            #                               create_embeddings: bool = False, create_summary: bool = False,
            #                               api_name: str = None):
            if api_name:
                process_and_store_content(chunk['text'], f"mediawiki_{wiki_name}", media_id, title, True, True, api_name)
            else:
                process_and_store_content(chunk['text'], f"mediawiki_{wiki_name}", media_id, title)
        logging.info(f"Successfully processed item: {title}")
    except Exception as e:
        logging.error(f"Error processing item {title}: {str(e)}")
        logging.error(f"Exception details: {traceback.format_exc()}")


def load_checkpoint(file_path: str) -> int:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)['last_processed_id']
    return 0


def save_checkpoint(file_path: str, last_processed_id: int):
    with open(file_path, 'w') as f:
        json.dump({'last_processed_id': last_processed_id}, f)


def import_mediawiki_dump(
        file_path: str,
        wiki_name: str,
        namespaces: List[int] = None,
        skip_redirects: bool = False,
        chunk_options: Dict[str, Any] = None,
        single_item: bool = False,
        progress_callback: Any = None,
        api_name: str = None,
        api_key: str = None
) -> Iterator[str]:
    try:
        logging.info(f"Importing MediaWiki dump: {file_path}")
        if chunk_options is None:
            chunk_options = config['chunking']

        checkpoint_file = f"{wiki_name}_import_checkpoint.json"
        last_processed_id = load_checkpoint(checkpoint_file)

        total_pages = count_pages(file_path, namespaces, skip_redirects)
        processed_pages = 0

        yield f"Found {total_pages} pages to process."

        for item in parse_mediawiki_dump(file_path, namespaces, skip_redirects):
            if item['page_id'] <= last_processed_id:
                continue
            # FIXME - ensure this works...
            if api_name is not None:
                # FIXME - add API key to the call/params
                process_single_item(item['content'], item['title'], wiki_name, chunk_options, False, item, api_name)
            process_single_item(item['content'], item['title'], wiki_name, chunk_options, False, item)
            save_checkpoint(checkpoint_file, item['page_id'])
            processed_pages += 1
            if progress_callback is not None:
                progress_callback(processed_pages / total_pages, f"Processed page: {item['title']}")
            yield f"Processed page {processed_pages}/{total_pages}: {item['title']}"

        os.remove(checkpoint_file)  # Remove checkpoint file after successful import
        yield f"Successfully imported and indexed MediaWiki dump: {wiki_name}"
    except FileNotFoundError:
        logger.error(f"MediaWiki dump file not found: {file_path}")
        yield f"Error: File not found - {file_path}"
    except PermissionError:
        logger.error(f"Permission denied when trying to read: {file_path}")
        yield f"Error: Permission denied - {file_path}"
    except Exception as e:
        logger.exception(f"Error during MediaWiki import: {str(e)}")
        yield f"Error during import: {str(e)}"

def count_pages(file_path: str, namespaces: List[int] = None, skip_redirects: bool = False) -> int:
    """
    Count the number of pages in a MediaWiki XML dump file.

    Args:
    file_path (str): Path to the MediaWiki XML dump file.
    namespaces (List[int], optional): List of namespace IDs to include. If None, include all namespaces.
    skip_redirects (bool, optional): Whether to skip redirect pages.

    Returns:
    int: The number of pages in the dump file.
    """
    try:
        dump = mwxml.Dump.from_file(open(file_path, encoding='utf-8'))
        count = 0
        for page in dump.pages:
            if skip_redirects and page.redirect:
                continue
            if namespaces and page.namespace not in namespaces:
                continue
            count += 1
        return count
    except Exception as e:
        logger.error(f"Error counting pages in MediaWiki dump: {str(e)}")
        return 0

#
# End of Media_Wiki.py
#######################################################################################################################
