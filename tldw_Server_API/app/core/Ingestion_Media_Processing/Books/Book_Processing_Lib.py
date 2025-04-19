# Book_Ingestion_Lib.py
#########################################
# Library to hold functions for ingesting book files.#
#
####################
# Function List
#
# 1. ingest_text_file(file_path, title=None, author=None, keywords=None):
# 2.
#
#
####################
#
# Imports
import os
import re
import tempfile
import zipfile
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
#
# External Imports
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import html2text
#
# Import Local
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords, add_media_to_database
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import summarize
from tldw_Server_API.app.core.Utils.Chunk_Lib import chunk_ebook_by_chapters, improved_chunking_process
from tldw_Server_API.app.core.Metrics.metrics_logger import (log_counter, log_histogram)
from tldw_Server_API.app.core.Utils.Utils import logging
#
#######################################################################################################################
# Function Definitions
#

def extract_epub_metadata_from_text(content: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts Title/Author if specific headers exist in the text."""
    title_match = re.search(r'^Title:\s*(.*?)$', content, re.IGNORECASE | re.MULTILINE)
    author_match = re.search(r'^Author:\s*(.*?)$', content, re.IGNORECASE | re.MULTILINE)

    title = title_match.group(1).strip() if title_match else None
    author = author_match.group(1).strip() if author_match else None

    return title, author

def format_toc_item(item, level):
    """
    Formats a table of contents item into Markdown list format.

    Parameters:
        - item (epub.Link or epub.Section): TOC item.
        - level (int): Heading level for indentation.

    Returns:
        - str: Markdown-formatted TOC item.
    """
    try:
        if isinstance(item, epub.Link):
            title = item.title
        elif isinstance(item, epub.Section):
            title = item.title
        else:
            title = str(item)

        return f"{'  ' * (level - 1)}- [{title}](#{slugify(title)})\n"
    except Exception as e:
        logging.exception(f"Error formatting TOC item: {str(e)}")
        return ""


def slugify(text):
    """
    Converts a string into a slug suitable for Markdown links.

    Parameters:
        - text (str): The text to slugify.

    Returns:
        - str: Slugified text.
    """
    return re.sub(r'[\W_]+', '-', text.lower()).strip('-')


#
# End of Contents modification
############################################################


############################################################
#
# File Conversion Functions

def epub_to_markdown(epub_path) -> Tuple[str, Optional[epub.EpubBook]]:
    """
    Converts an EPUB file to Markdown format, including the table of contents and chapter contents.

    Parameters:
        - epub_path (str): Path to the EPUB file.

    Returns:
        - str: Markdown-formatted content of the EPUB.
    """
    book = None # Initialize book
    try:
        logging.info(f"Converting EPUB to Markdown from {epub_path}")
        book = epub.read_epub(epub_path)
        markdown_content = "# Table of Contents\n\n"
        chapters = []

        # Extract and format the table of contents
        toc = book.toc
        for item in toc:
            if isinstance(item, tuple):
                section, children = item
                level = 1
                markdown_content += format_toc_item(section, level)
                for child in children:
                    markdown_content += format_toc_item(child, level + 1)
            else:
                markdown_content += format_toc_item(item, 1)

        markdown_content += "\n---\n\n"

        # Process each chapter
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_content = item.get_content().decode('utf-8')
                soup = BeautifulSoup(chapter_content, 'html.parser')

                # Extract chapter title
                title = soup.find(['h1', 'h2', 'h3'])
                if title:
                    chapter_title = title.get_text()
                    markdown_content += f"# {chapter_title}\n\n"

                # Process chapter content
                for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol']):
                    if elem.name.startswith('h'):
                        level = int(elem.name[1])
                        markdown_content += f"{'#' * level} {elem.get_text()}\n\n"
                    elif elem.name == 'p':
                        markdown_content += f"{elem.get_text()}\n\n"
                    elif elem.name in ['ul', 'ol']:
                        for li in elem.find_all('li'):
                            prefix = '-' if elem.name == 'ul' else '1.'
                            markdown_content += f"{prefix} {li.get_text()}\n"
                        markdown_content += "\n"

                markdown_content += "---\n\n"

        logging.debug("EPUB to Markdown conversion completed.")
        return markdown_content, book # Return book object too

    except Exception as e:
        logging.exception(f"Error converting EPUB to Markdown: {str(e)}")
        # Still return None for the book object on error
        return f"# Error converting EPUB\n\n{e}", book # Return error message and potentially None book

def extract_epub_metadata_from_epub_obj(book: epub.EpubBook) -> Tuple[Optional[str], Optional[str]]:
    """Extracts title and author directly from the ebooklib book object metadata."""
    title = None
    author = None
    try:
        metadata = book.get_metadata('DC', 'title')
        if metadata:
            title = metadata[0][0]
    except Exception:
        logging.debug("Could not extract DC:title metadata.")

    try:
        metadata = book.get_metadata('DC', 'creator')
        if metadata:
            # Often creators are listed with attributes like {'role': 'aut'}
            if isinstance(metadata[0], tuple) and len(metadata[0]) > 0:
                 author = metadata[0][0]
            else: # Fallback if it's just a simple string
                 author = str(metadata[0])
    except Exception:
        logging.debug("Could not extract DC:creator metadata.")

    return title, author

#
# End of File Conversion Functions
############################################################


############################################################
#
# epub parsing Functions

def read_epub_filtered(epub_path) -> Tuple[str, Optional[epub.EpubBook]]:
    """
    Reads an EPUB by following the spine, skipping known front matter
    but keeping the Table of Contents (TOC). Returns a cleaned-up
    text string with minimal empty whitespace.

    :param epub_path: Path to the .epub file.
    :return: A cleaned-up text string of the book's content.
    """
    book = None
    try:
        book = epub.read_epub(epub_path)

        # Known front-matter filenames to skip, except we want to keep
        # the actual "toc" if it is meaningful. Adjust as needed.
        # NOTE: Filenames vary across publishers, so you may need to
        # add or remove items from this set.
        skip_front_matter = {
            "cover",
            "titlepage",
            "copy",
            "copyright",
            "colophon",
            "upgrade",
            # "toc",    # Do NOT skip if you want to keep the TOC
            "notice",
            "legal",
            "license",
            #"nav"
        }

        all_text_segments = []

        # The spine is the main reading order of the EPUB.
        for itemref in book.spine:
            # itemref is typically ('idref', {})
            item_id = itemref[0]
            item = book.get_item_with_id(item_id)

            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                # Not an HTML/xHTML document, skip
                continue

            # Check if filename suggests front matter we want to skip
            filename_lower = item.file_name.lower()
            if any(name in filename_lower for name in skip_front_matter):
                logging.debug(f"Skipping front matter: {item.file_name}")
                continue

            # Otherwise, parse and extract text
            content = item.get_content().decode('utf-8', errors='replace')
            soup = BeautifulSoup(content, 'html.parser')

            # You can adjust which tags to extract
            # (h1..h6, p, lists, etc.)
            # We'll gather them in reading order:
            text_chunks = []
            for elem in soup.find_all(['h1','h2','h3','h4','h5','h6','p','ul','ol']):
                # Clean up the text
                text = elem.get_text().strip()

                # Skip truly empty or whitespace-only text
                if not text:
                    continue

                # For headings:
                if elem.name in ['h1','h2','h3','h4','h5','h6']:
                    # You might format headings in some special way:
                    level = int(elem.name[1])  # e.g., h2 -> 2
                    text_chunks.append(("#" * level) + " " + text)
                # For paragraphs
                elif elem.name == 'p':
                    text_chunks.append(text)
                # For lists
                elif elem.name in ['ul','ol']:
                    # Distinguish bullet vs numbered list
                    bullet = "-" if elem.name == 'ul' else "1."
                    for li in elem.find_all('li'):
                        li_text = li.get_text().strip()
                        if li_text:
                            text_chunks.append(f"{bullet} {li_text}")

            # Join everything from this item with double newlines
            # (or single newline, whichever you prefer)
            item_text = "\n\n".join(text_chunks)
            # Skip adding if there's nothing left
            if item_text.strip():
                all_text_segments.append(item_text)

        # Combine all items in the spine
        full_text = "\n\n".join(all_text_segments)

        full_text = re.sub(r'[ \t]+', ' ', full_text)  # collapse multiple spaces
        full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)  # collapse multiple blank lines
        return full_text, book # Return book object

    except Exception as e:
        logging.exception(f"Failed to parse EPUB: {str(e)}")
        return "", book # Return empty string and potentially None book

def read_epub(file_path) -> Tuple[str, Optional[epub.EpubBook]]:
    """
    Reads and extracts text from an EPUB file, cleaning up messy spacing.
    Returns cleaned text and book object.
    """
    book = None
    try:
        logging.info(f"Reading EPUB file from {file_path}")
        book = epub.read_epub(file_path)

        all_paragraphs = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                html_content = item.get_content().decode('utf-8', errors='replace')
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract headings and paragraphs (no nested loop!)
                for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                    raw_text = ' '.join(elem.stripped_strings)
                    if not raw_text.strip():
                        continue

                    if elem.name.startswith('h'):
                        # e.g. 'h2' -> level=2
                        level = int(elem.name[-1])
                        cleaned = f"{'#' * level} {raw_text}"
                    else:
                        cleaned = raw_text

                    all_paragraphs.append(cleaned)

        # Join all paragraphs with two newlines
        text = "\n\n".join(all_paragraphs)

        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse multiple blank lines into just one
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        logging.debug("EPUB content extraction completed (cleaned).")
        return text, book # Return book object

    except Exception as e:
        logging.exception(f"Error reading EPUB file: {str(e)}")
        # Re-raise or return error indication
        raise # Or return "", book

#
# End of epub parsing Functions
############################################################


############################################################
#
# epub Processing Functions

def xml_to_markdown(element, level=0):
    # ... (keep existing implementation - seems DB free) ...
    """
    Recursively converts XML elements to markdown format.
    """
    markdown = ""
    tag = element.tag.split('}')[-1] # Clean namespace if present

    # Add element name as heading (maybe simplify this)
    if level > 0 and tag:
        markdown += f"{'#' * min(level + 1, 6)} {tag}\n\n" # Start headings at level 2

    # Add element text if it exists
    if element.text and element.text.strip():
        markdown += f"{element.text.strip()}\n\n"

    # Add attributes (optional, could be noisy)
    # if element.attrib:
    #     markdown += "Attributes:\n"
    #     for k, v in element.attrib.items():
    #         markdown += f"- **{k}:** {v}\n"
    #     markdown += "\n"

    # Process child elements
    for child in element:
        markdown += xml_to_markdown(child, level + 1)

    # Add tail text if exists (text after a child element)
    if element.tail and element.tail.strip():
        markdown += f"{element.tail.strip()}\n\n"

    return markdown

def opml_to_markdown(root):
    # ... (keep existing implementation - seems DB free) ...
    """
    Converts OPML structure to markdown format.
    """
    markdown = ""
    head = root.find("head")
    if head is not None:
         title_elem = head.find("title")
         if title_elem is not None and title_elem.text:
              markdown += f"# {title_elem.text.strip()}\n\n"

    markdown += "## Outline\n\n" # Changed from Table of Contents

    def process_outline(outline_element, current_level=0):
        result = ""
        # Find direct child 'outline' elements
        for item in outline_element.findall("./outline"):
            text = item.get("text", item.get("title", "")) # Prefer 'text', fallback to 'title'
            if text:
                result += f"{'  ' * current_level}- {text}\n"
            # Recursively process children of this item
            result += process_outline(item, current_level + 1)
        return result

    body = root.find("body")
    if body is not None:
        markdown += process_outline(body)

    return markdown


def process_epub(
    file_path: str,
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    perform_chunking: bool = True,
    chunk_options: Optional[Dict[str, Any]] = None,
    custom_chapter_pattern: Optional[str] = None, # Kept separate for clarity
    perform_analysis: bool = False, # Renamed from auto_analyze
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    summarize_recursively: bool = False, # Added for consistency
    extraction_method: str = 'markdown' # 'markdown' or 'filtered'
) -> Dict[str, Any]:
    """
    Processes an EPUB file: extracts content & metadata, chunks, and optionally summarizes.
    Returns a dictionary with processed data, status, and errors. *No DB interaction.*

    Parameters:
        - file_path (str): Path to the EPUB file.
        - title_override (str, optional): User-provided title.
        - author_override (str, optional): User-provided author.
        - keywords (List[str], optional): Keywords for the book.
        - custom_prompt (str, optional): Custom user prompt for summarization.
        - system_prompt (str, optional): System prompt for summarization.
        - perform_chunking (bool): Whether to chunk the content.
        - chunk_options (dict, optional): Options for chunking.
        - custom_chapter_pattern (str, optional): Custom regex pattern for chapter detection.
        - perform_analysis (bool): Whether to perform summarization.
        - api_name (str, optional): API name for summarization.
        - api_key (str, optional): API key for summarization.
        - summarize_recursively (bool): Whether to perform recursive summarization.
        - extraction_method (str): 'markdown' or 'filtered'.

    Returns:
        - Dict[str, Any]: Dictionary containing processing results:
            {
                "status": "Success" | "Error",
                "input_ref": str (file_path),
                "media_type": "ebook",
                "text_content": Optional[str],
                "metadata": Optional[Dict], # {'title': str, 'author': str, 'raw': dict}
                "chunks": Optional[List[Dict]], # [{'text': str, 'metadata': {...}}]
                "summary": Optional[str],
                "keywords": Optional[List[str]], # Keywords PASSED IN, not added by DB
                "error": Optional[str],
                "analysis_details": Optional[Dict], # Add details about analysis if performed
            }
    """
    start_time = datetime.now()
    result: Dict[str, Any] = {
        "status": "Pending",
        "input_ref": file_path, # Use input file path as reference
        "processing_source": file_path, # Same for now, might differ if copied
        "media_type": "ebook",
        "text_content": None,
        "metadata": {"title": None, "author": None, "raw": None}, # Init structure
        "chunks": None,
        "summary": None,
        "keywords": keywords or [],
        "error": None,
        "warnings": None
    }

    try:
        logging.info(f"Processing EPUB file from {file_path}")
        log_counter("epub_processing_attempt", labels={"file_path": file_path})

        # 1. Extract Content and Metadata
        extracted_text = ""
        ebook_obj = None
        extractor_func: Optional[callable] = None

        if extraction_method == 'markdown': extractor_func = epub_to_markdown
        elif extraction_method == 'filtered': extractor_func = read_epub_filtered
        else: extractor_func = read_epub # Default fallback

        try:
             extracted_text, ebook_obj = extractor_func(file_path)
        except Exception as extract_err:
             # If main extractor fails, try the basic one as fallback
             logging.warning(f"Extractor '{extraction_method}' failed: {extract_err}. Falling back to basic read_epub.")
             result["warnings"] = (result["warnings"] or []) + [f"Extraction method '{extraction_method}' failed, used fallback."]
             try:
                  extracted_text, ebook_obj = read_epub(file_path)
             except Exception as fallback_err:
                  # If fallback also fails, raise the error to stop processing
                  raise ValueError(f"Failed to extract text from EPUB even with fallback: {fallback_err}") from fallback_err


        if not extracted_text or ebook_obj is None:
            # This case should be caught by the exceptions above now
            raise ValueError("Failed to extract text or book object from EPUB.")

        result["text_content"] = extracted_text
        logging.debug(f"Extracted EPUB content using '{extraction_method}' (or fallback). Length: {len(extracted_text)}")

        # Extract metadata from ebooklib object
        meta_title, meta_author = extract_epub_metadata_from_epub_obj(ebook_obj)

        # Get filename stem for fallback title
        file_path_obj = Path(file_path)
        filename_stem = file_path_obj.stem

        # Prioritize overrides, then metadata, then filename/default
        final_title = title_override or meta_title or filename_stem
        final_author = author_override or meta_author or "Unknown"

        result["metadata"] = {
            "title": final_title,
            "author": final_author,
            "raw": ebook_obj.metadata, # Store raw metadata
            # Add source filename for potential use later
            "source_filename": file_path_obj.name
        }
        logging.debug(f"Final metadata - Title: {final_title}, Author: {final_author}")

        # 2. Chunking
        processed_chunks = None
        if perform_chunking:
            # Ensure default chunk options if needed
            effective_chunk_options = chunk_options or {}
            # Override/ensure chapter method for ebooks unless specified otherwise
            effective_chunk_options['method'] = effective_chunk_options.get('method', 'chapter')
            # Make sure custom pattern is passed if provided separately
            if custom_chapter_pattern:
                effective_chunk_options['custom_chapter_pattern'] = custom_chapter_pattern

            logging.info(f"Chunking ebook content with options: {effective_chunk_options}")
            try:
                processed_chunks = chunk_ebook_by_chapters(extracted_text, effective_chunk_options)

                if not processed_chunks:
                     logging.warning(f"Chunking produced no chunks for {file_path}. Using full text as one chunk.")
                     processed_chunks = [{'text': extracted_text, 'metadata': {'chunk_num': 0}}]
                     result["warnings"] = (result["warnings"] or []) + ["Chunking yielded no results; using full text."]
                else:
                     logging.info(f"Total chunks created: {len(processed_chunks)}")
                     log_histogram("epub_chunks_created", len(processed_chunks), labels={"file_path": file_path})

                result["chunks"] = processed_chunks

            except Exception as chunk_err:
                logging.error(f"Chunking failed for {file_path}: {chunk_err}", exc_info=True)
                result["warnings"] = (result["warnings"] or []) + [f"Chunking failed: {chunk_err}"]
                # Fallback: use full text as one chunk so analysis can potentially proceed
                processed_chunks = [{'text': extracted_text, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {chunk_err}"}}]
                result["chunks"] = processed_chunks

        else:
             # If not chunking, create a single chunk containing the whole text
             processed_chunks = [{'text': extracted_text, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks # Store the single chunk
             logging.info("Chunking disabled. Using full text as one chunk.")


        # 3. Summarization / Analysis
        final_summary = None
        if perform_analysis and api_name and api_key and processed_chunks:
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks.")
            log_counter("epub_summarization_attempt", value=len(processed_chunks), labels={"file_path": file_path, "api_name": api_name})
            chunk_summaries = []
            summarized_chunks_for_result = [] # Keep track of chunks with summaries added

            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {}) # Get existing metadata
                if chunk_text:
                    try:
                        summary_text = summarize(
                            api_name=api_name,
                            input_data=chunk_text, # Pass input_data instead of text_to_summarize
                            custom_prompt_arg=custom_prompt, # Pass custom_prompt_input
                            api_key=api_key,
                            recursive_summarization=False, # Summarize each chunk individually first
                            temp=None, # Optional temp
                            system_message=system_prompt
                        )
                        if summary_text and summary_text.strip():
                            chunk_summaries.append(summary_text)
                            chunk_metadata['summary'] = summary_text # Add summary to the chunk's metadata
                        else:
                            chunk_metadata['summary'] = None # Indicate summarization was attempted but failed/empty
                            logging.debug(f"Summarization yielded empty result for chunk {i} of {file_path}.")
                    except Exception as summ_err:
                        logging.warning(f"Summarization failed for chunk {i} of {file_path}: {summ_err}")
                        chunk_metadata['summary'] = f"[Summarization Error: {summ_err}]"
                        result["warnings"] = (result["warnings"] or []) + [f"Summarization failed for chunk {i}: {summ_err}"]

                chunk['metadata'] = chunk_metadata # Update metadata in the original chunk dict
                summarized_chunks_for_result.append(chunk) # Add chunk regardless of summary success


            result["chunks"] = summarized_chunks_for_result # Update chunks in result

            # Combine chunk summaries (potentially recursively)
            if chunk_summaries:
                if summarize_recursively and len(chunk_summaries) > 1:
                    logging.info("Performing recursive summarization on chunk summaries.")
                    try:
                        final_summary = summarize(
                            api_name=api_name,
                            input_data="\n\n---\n\n".join(chunk_summaries), # Join summaries clearly
                            custom_prompt_arg=custom_prompt or "Provide a concise overall summary of the following chapter summaries.", # Recursive prompt
                            api_key=api_key,
                            recursive_summarization=False, # Final pass
                            temp=None,
                            system_message=system_prompt
                        )
                        if not final_summary or not final_summary.strip():
                            logging.warning(f"Recursive summarization for {file_path} yielded empty result. Falling back.")
                            final_summary = "\n\n---\n\n".join(chunk_summaries) # Fallback
                            result["warnings"] = (result["warnings"] or []) + ["Recursive summarization yielded empty result."]
                        else:
                             log_counter("epub_recursive_summarization_success", labels={"file_path": file_path})

                    except Exception as rec_summ_err:
                         logging.error(f"Recursive summarization failed for {file_path}: {rec_summ_err}")
                         final_summary = f"[Recursive Summarization Error: {rec_summ_err}]\n\n" + "\n\n---\n\n".join(chunk_summaries)
                         result["warnings"] = (result["warnings"] or []) + [f"Recursive summarization failed: {rec_summ_err}"]
                         log_counter("epub_recursive_summarization_error", labels={"file_path": file_path, "error": str(rec_summ_err)})
                else:
                    # Simple join if not recursive or only one summary
                    final_summary = "\n\n---\n\n".join(chunk_summaries)

            result["summary"] = final_summary
            log_counter("epub_chunks_summarized", value=len(chunk_summaries), labels={"file_path": file_path})
            logging.info("Summarization processing completed.")

        # --- Log reasons if summarization was skipped ---
        elif not perform_analysis: logging.info("Summarization disabled.")
        elif not api_name or not api_key: logging.warning("Summarization skipped: API name or key not provided.")
        elif not processed_chunks: logging.warning("Summarization skipped: No chunks were generated/available.")
        else: logging.warning("Summarization skipped due to unknown condition.")

        # Final status determination
        if result["warnings"]:
            result["status"] = "Warning"
        else:
             logging.warning("Summarization skipped: API name or key not provided.")

        result["status"] = "Success"
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("epub_processing_duration", processing_time, labels={"file_path": file_path})
        logging.info(f"Ebook '{final_title}' by {final_author} processed with status: {result['status']}.")
        log_counter(f"epub_processing_{result['status'].lower()}", labels={"file_path": file_path})

    except Exception as e:
        logging.exception(f"Error processing ebook {file_path}: {str(e)}")
        result["status"] = "Error"
        result["error"] = str(e)
        log_counter("epub_processing_error", labels={"file_path": file_path, "error": str(e)})

    # Ensure warnings list is None if empty
    if isinstance(result.get("warnings"), list) and not result["warnings"]:
        result["warnings"] = None

    return result


def process_zip_of_epubs(
    zip_file_path: str,
    keywords: Optional[List[str]] = None,
    # Pass all other relevant options down to process_epub
    **epub_options
    ) -> List[Dict[str, Any]]:
    """
    Processes a ZIP file containing multiple EPUB files, extracts each one,
    and processes it using process_epub. Returns a list of result dictionaries.
    *No DB interaction.*

    Parameters:
        - zip_file_path (str): Path to the ZIP file.
        - keywords (List[str], optional): Base keywords to apply to all books.
        - **epub_options: Keyword arguments to pass down to process_epub
                         (e.g., custom_prompt, perform_analysis, api_name, etc.)

    Returns:
        - List[Dict[str, Any]]: A list where each item is the result dictionary
                                from processing a single EPUB file.
    """
    results = []
    temp_dir_path = None # Keep track of temp dir path
    try:
        with tempfile.TemporaryDirectory(prefix="epub_zip_") as temp_dir:
            temp_dir_path = temp_dir # Store path for logging
            logging.info(f"Extracting ZIP file {zip_file_path} to temporary directory {temp_dir_path}")
            log_counter("zip_processing_attempt", labels={"zip_path": zip_file_path})

            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)
            except zipfile.BadZipFile as zip_err:
                 logging.error(f"Invalid ZIP file: {zip_file_path} - {zip_err}")
                 # Return a single error result for the whole zip
                 return [{
                     "status": "Error", "input_ref": zip_file_path, "media_type": "zip",
                     "error": f"Invalid or corrupted ZIP file: {zip_err}"
                 }]
            except Exception as extract_err:
                 logging.error(f"Failed to extract ZIP file {zip_file_path}: {extract_err}", exc_info=True)
                 return [{
                     "status": "Error", "input_ref": zip_file_path, "media_type": "zip",
                     "error": f"Failed to extract ZIP file: {extract_err}"
                 }]

            epub_files = list(Path(temp_dir_path).rglob('*.epub')) # Recursive search
            log_histogram("epub_files_in_zip", len(epub_files), labels={"zip_path": zip_file_path})
            logging.info(f"Found {len(epub_files)} EPUB files in ZIP.")

            if not epub_files:
                 results.append({
                     "status": "Warning", "input_ref": zip_file_path, "media_type": "zip",
                     "error": "No .epub files found within the ZIP archive.", "message": "No EPUBs found."
                 })

            for epub_path in epub_files:
                epub_filename = epub_path.name
                logging.info(f"Processing EPUB file '{epub_filename}' from ZIP.")
                try:
                    # Call the refactored process_epub
                    result = process_epub(
                        file_path=str(epub_path),
                        keywords=keywords, # Pass the base keywords
                        **epub_options # Pass other options like analysis settings
                    )
                    # Add zip source info for clarity/traceability
                    result["source_zip"] = zip_file_path
                    result["original_filename_in_zip"] = epub_filename
                    # Ensure processing_source reflects the temp path
                    result["processing_source"] = str(epub_path)
                    results.append(result)
                except Exception as single_epub_err:
                     logging.exception(f"Error processing '{epub_filename}' from zip {zip_file_path}: {single_epub_err}")
                     results.append({
                         "status": "Error",
                         "input_ref": epub_filename, # Use filename within zip as ref
                         "processing_source": str(epub_path), # Path within temp dir
                         "media_type": "ebook",
                         "error": f"Failed processing from ZIP: {single_epub_err}",
                         "source_zip": zip_file_path,
                         "original_filename_in_zip": epub_filename
                     })

            logging.info(f"Completed processing all EPUB files in the ZIP: {zip_file_path}")
            log_counter("zip_processing_success", labels={"zip_path": zip_file_path})

    except Exception as e:
        logging.exception(f"Error processing ZIP file {zip_file_path}: {str(e)}")
        log_counter("zip_processing_error", labels={"zip_path": zip_file_path, "error": str(e)})
        # Return a single error result for the whole zip if setup fails
        return [{
            "status": "Error", "input_ref": zip_file_path, "media_type": "zip",
            "error": f"Error processing ZIP file itself: {str(e)}"
        }]

    return results


def _process_markup_or_plain_text(
    file_path: str,
    file_type: str, # 'html', 'xml', 'opml', 'text'
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None, # Expect list
    perform_chunking: bool = True,
    chunk_options: Optional[Dict[str, Any]] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    summarize_recursively: bool = False,
) -> Dict[str, Any]:
    """
    Internal helper to process HTML, XML, OPML, or plain text files.
    Extracts content, converts to Markdown (if applicable), chunks, summarizes.
    Returns a result dictionary. *No DB interaction.*
    """
    start_time = datetime.now()
    # Determine media type based on input file_type
    media_type = "document" # Default
    if file_type in ['html', 'xml', 'opml', 'text']:
        media_type = file_type # Be more specific

    result: Dict[str, Any] = {
        "status": "Pending",
        "input_ref": file_path, # Use input path as reference
        "processing_source": file_path, # Actual processed path
        "media_type": media_type,
        "source_format": file_type,
        "text_content": None,
        "metadata": {"title": None, "author": None, "raw": None}, # Init structure
        "chunks": None,
        "summary": None,
        "keywords": keywords or [],
        "error": None,
        "warnings": None
    }

    try:
        logging.info(f"Processing {file_type} file from {file_path}")
        log_counter(f"{file_type}_processing_attempt", labels={"file_path": file_path})

        markdown_content = ""
        extracted_title = None
        extracted_author = "Unknown" # Author extraction is less common for these types
        raw_metadata = {} # Store raw extracted info

        file_path_obj = Path(file_path)
        filename_stem = file_path_obj.stem

        # 1. Read and Convert Content
        if file_type == 'html':
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.body_width = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                markdown_content = h.handle(html_content)
                soup = BeautifulSoup(html_content, 'html.parser')
                title_tag = soup.find('title')
                extracted_title = title_tag.string.strip() if title_tag and title_tag.string else None
                # Add more metadata extraction if desired (e.g., meta tags)
                meta_author_tag = soup.find('meta', attrs={'name': 'author'})
                if meta_author_tag and meta_author_tag.get('content'):
                     extracted_author = meta_author_tag['content'].strip()
                raw_metadata = {'html_title': extracted_title, 'html_author': extracted_author}
            except Exception as html_err:
                 raise ValueError(f"Failed to parse HTML file {file_path}: {html_err}") from html_err

        elif file_type == 'xml':
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                markdown_content = xml_to_markdown(root) # Use helper
                # Try finding a common title element
                title_elem = root.find('.//title') # Adjust XPath as needed
                extracted_title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
                # Add more metadata extraction if desired
                raw_metadata = {'xml_root_tag': root.tag}
            except ET.ParseError as xml_err:
                 raise ValueError(f"Failed to parse XML file {file_path}: {xml_err}") from xml_err

        elif file_type == 'opml':
             try:
                 tree = ET.parse(file_path)
                 root = tree.getroot()
                 markdown_content = opml_to_markdown(root) # Use helper
                 title_elem = root.find("./head/title")
                 extracted_title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
                 raw_metadata = {'opml_title': extracted_title}
             except ET.ParseError as opml_err:
                 raise ValueError(f"Failed to parse OPML file {file_path}: {opml_err}") from opml_err

        elif file_type == 'text':
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read() # Plain text is already "markdown-like"
                # Try extracting metadata if it looks like a converted epub text dump
                epub_title, epub_author = extract_epub_metadata_from_text(markdown_content)
                extracted_title = epub_title
                if epub_author: extracted_author = epub_author
                raw_metadata = {'maybe_epub_title': epub_title, 'maybe_epub_author': epub_author}
            except Exception as text_err:
                 raise ValueError(f"Failed to read text file {file_path}: {text_err}") from text_err
        else:
            raise ValueError(f"Unsupported file type for processing: {file_type}")

        # Clean up common issues like excessive newlines
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content).strip()
        if not markdown_content:
            # Handle empty content after potential conversion
            raise ValueError("Content became empty after processing.")
        result["text_content"] = markdown_content

        # Finalize metadata
        final_title = title_override or extracted_title or filename_stem
        final_author = author_override or extracted_author # Keep 'Unknown' if not found/overridden
        result["metadata"] = {
            "title": final_title,
            "author": final_author,
            "source_filename": file_path_obj.name,
            "raw": raw_metadata # Store any extracted raw metadata
        }
        logging.debug(f"Final metadata - Title: {final_title}, Author: {final_author}")

        # 2. Chunking (Similar logic as EPUB, adjust defaults if needed)
        processed_chunks = None
        if perform_chunking:
            effective_chunk_options = chunk_options or {}
            # Default method for general documents
            effective_chunk_options.setdefault('method', 'recursive')
            # Set default size/overlap if not provided
            effective_chunk_options.setdefault('max_size', 1000)
            effective_chunk_options.setdefault('overlap', 200)

            logging.info(f"Chunking {file_type} content with options: {effective_chunk_options}")
            try:
                # Use a more generic chunking function
                processed_chunks = improved_chunking_process(markdown_content, effective_chunk_options)

                if not processed_chunks:
                     logging.warning(f"Chunking produced no chunks for {file_path}. Using full text as one chunk.")
                     processed_chunks = [{'text': markdown_content, 'metadata': {'chunk_num': 0}}]
                     result["warnings"] = (result["warnings"] or []) + ["Chunking yielded no results; using full text."]
                else:
                     logging.info(f"Total chunks created: {len(processed_chunks)}")
                     log_histogram(f"{file_type}_chunks_created", len(processed_chunks), labels={"file_path": file_path})

                result["chunks"] = processed_chunks

            except Exception as chunk_err:
                logging.error(f"Chunking failed for {file_path}: {chunk_err}", exc_info=True)
                result["warnings"] = (result["warnings"] or []) + [f"Chunking failed: {chunk_err}"]
                processed_chunks = [{'text': markdown_content, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {chunk_err}"}}]
                result["chunks"] = processed_chunks # Store fallback chunk
        else:
             processed_chunks = [{'text': markdown_content, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks
             logging.info("Chunking disabled. Using full text as one chunk.")

        # 3. Summarization / Analysis (Identical logic to process_epub)
        final_summary = None
        if perform_analysis and api_name and api_key and processed_chunks:
            # --- (Copy the summarization loop from process_epub here) ---
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks of {file_type}.")
            log_counter(f"{file_type}_summarization_attempt", value=len(processed_chunks), labels={"file_path": file_path, "api_name": api_name})
            chunk_summaries = []
            summarized_chunks_for_result = []

            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                if chunk_text:
                    try:
                        summary_text = summarize(api_name, chunk_text, custom_prompt, api_key, system_prompt, None, False, )
                        if summary_text and summary_text.strip():
                            chunk_summaries.append(summary_text)
                            chunk_metadata['summary'] = summary_text
                        else:
                            chunk_metadata['summary'] = None
                            logging.debug(f"Summarization yielded empty result for chunk {i} of {file_path}.")
                    except Exception as summ_err:
                        logging.warning(f"Summarization failed for chunk {i} of {file_path}: {summ_err}")
                        chunk_metadata['summary'] = f"[Summarization Error: {summ_err}]"
                        result["warnings"] = (result["warnings"] or []) + [f"Summarization failed for chunk {i}: {summ_err}"]

                chunk['metadata'] = chunk_metadata
                summarized_chunks_for_result.append(chunk)

            result["chunks"] = summarized_chunks_for_result

            if chunk_summaries:
                if summarize_recursively and len(chunk_summaries) > 1:
                    logging.info("Performing recursive summarization on chunk summaries.")
                    try:
                        final_summary = summarize(api_name, "\n\n---\n\n".join(chunk_summaries), custom_prompt or "Provide an overall summary.", api_key, system_prompt, None, False,)
                        if not final_summary or not final_summary.strip():
                            logging.warning(f"Recursive summarization for {file_path} yielded empty result. Falling back.")
                            final_summary = "\n\n---\n\n".join(chunk_summaries)
                            result["warnings"] = (result["warnings"] or []) + ["Recursive summarization yielded empty result."]
                        else:
                             log_counter(f"{file_type}_recursive_summarization_success", labels={"file_path": file_path})
                    except Exception as rec_summ_err:
                         logging.error(f"Recursive summarization failed for {file_path}: {rec_summ_err}")
                         final_summary = f"[Recursive Summarization Error: {rec_summ_err}]\n\n" + "\n\n---\n\n".join(chunk_summaries)
                         result["warnings"] = (result["warnings"] or []) + [f"Recursive summarization failed: {rec_summ_err}"]
                         log_counter(f"{file_type}_recursive_summarization_error", labels={"file_path": file_path, "error": str(rec_summ_err)})
                else:
                    final_summary = "\n\n---\n\n".join(chunk_summaries)

            result["summary"] = final_summary
            log_counter(f"{file_type}_chunks_summarized", value=len(chunk_summaries), labels={"file_path": file_path})
            logging.info("Summarization processing completed.")
        # --- (Log summarization skipped reasons) ---
        elif not perform_analysis: logging.info("Summarization disabled.")
        elif not api_name or not api_key: logging.warning("Summarization skipped: API name or key not provided.")
        elif not processed_chunks: logging.warning("Summarization skipped: No chunks were generated/available.")
        else: logging.warning("Summarization skipped due to unknown condition.")

        # Final status determination
        if result["warnings"]:
            result["status"] = "Warning"
        else:
            result["status"] = "Success"

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram(f"{file_type}_processing_duration", processing_time, labels={"file_path": file_path})
        logging.info(f"{file_type.capitalize()} file '{final_title}' processed with status: {result['status']}.")
        log_counter(f"{file_type}_processing_{result['status'].lower()}", labels={"file_path": file_path})

    except FileNotFoundError:
        logging.error(f"{file_type.capitalize()} file not found: {file_path}")
        result["status"] = "Error"
        result["error"] = "File not found"
        log_counter(f"{file_type}_processing_error", labels={"file_path": file_path, "error": "FileNotFoundError"})
    except ValueError as ve: # Catch value errors from parsing/conversion/empty content
         logging.error(f"Processing value error for {file_type} file {file_path}: {str(ve)}")
         result["status"] = "Error"
         result["error"] = str(ve)
         log_counter(f"{file_type}_processing_error", labels={"file_path": file_path, "error": "ValueError"})
    except Exception as e:
        logging.exception(f"Error processing {file_type} file {file_path}: {str(e)}")
        result["status"] = "Error"
        result["error"] = str(e)
        log_counter(f"{file_type}_processing_error", labels={"file_path": file_path, "error": type(e).__name__})

    # Ensure warnings list is None if empty
    if isinstance(result.get("warnings"), list) and not result["warnings"]:
        result["warnings"] = None

    return result



def _process_single_ebook(
    ebook_path: Path,
    perform_chunking: bool,
    chunk_options: Optional[Dict[str, Any]], # Accept optional dict
    summarize: bool, # Changed from perform_analysis for clarity
    summarize_recursively: bool,
    api_name: Optional[str],
    api_key: Optional[str],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    # Add title/author overrides if needed by caller
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    # Add keywords if they should be passed *during* processing (rare)
    keywords: Optional[List[str]] = None,
    # Add custom chapter pattern if passed separately
    custom_chapter_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    CPU‑bound worker: read EPUB, chunk (optional), summarise (optional).
    Calls refactored process_epub. Returns its result dictionary.
    *No DB interaction.*
    """
    try:
        # Call the main processing function
        result_dict = process_epub(
            file_path=str(ebook_path),
            title_override=title_override,
            author_override=author_override,
            keywords=keywords, # Pass keywords list
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            perform_chunking=perform_chunking,
            chunk_options=chunk_options, # Pass the dict
            custom_chapter_pattern=custom_chapter_pattern, # Pass separately
            perform_analysis=summarize, # Map parameter name
            api_name=api_name,
            api_key=api_key,
            summarize_recursively=summarize_recursively,
            extraction_method='filtered' # Or choose based on parameter
        )
        # Ensure the input ref matches the path object expected by caller
        # The called process_epub already sets input_ref and processing_source
        return result_dict

    except Exception as e:
        logging.error(f"_process_single_ebook error for {ebook_path}: {e}", exc_info=True)
        return {
            "status": "Error",
            "input_ref": str(ebook_path), # Use path as input ref for error
            "processing_source": str(ebook_path),
            "media_type": "ebook",
            "error": f"Worker processing failed: {str(e)}",
            # Ensure other keys are present if needed by consuming code
            "text_content": None, "metadata": None, "chunks": None, "summary": None,
            "keywords": keywords or [], "warnings": None
        }


# Ingest a text file into the database with Title/Author/Keywords
def extract_epub_metadata(content):
    title_match = re.search(r'Title:\s*(.*?)\n', content)
    author_match = re.search(r'Author:\s*(.*?)\n', content)

    title = title_match.group(1) if title_match else None
    author = author_match.group(1) if author_match else None

    return title, author


def ingest_text_file(file_path, title=None, author=None, keywords=None):
    """
    Ingests a plain text file into the database with optional metadata.

    Parameters:
        - file_path (str): Path to the text file.
        - title (str, optional): Title of the document.
        - author (str, optional): Author of the document.
        - keywords (str, optional): Comma-separated keywords.

    Returns:
        - str: Status message indicating success or failure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Check if it's a converted epub and extract metadata if so
        if 'epub_converted' in (keywords or '').lower():
            extracted_title, extracted_author = extract_epub_metadata(content)
            title = title or extracted_title
            author = author or extracted_author
            logging.debug(f"Extracted metadata for converted EPUB - Title: {title}, Author: {author}")

        # If title is still not provided, use the filename without extension
        if not title:
            title = os.path.splitext(os.path.basename(file_path))[0]

        # If author is still not provided, set it to 'Unknown'
        if not author:
            author = 'Unknown'

        # If keywords are not provided, use a default keyword
        if not keywords:
            keywords = 'text_file,epub_converted'
        else:
            keywords = f'text_file,epub_converted,{keywords}'

        # Add the text file to the database
        add_media_with_keywords(
            url="its_a_book",
            title=title,
            media_type='book',
            content=content,
            keywords=keywords,
            prompt='No prompt for text files',
            summary='No summary for text files',
            transcription_model='None',
            author=author,
            ingestion_date=datetime.now().strftime('%Y-%m-%d')
        )

        logging.info(f"Text file '{title}' by {author} ingested successfully.")
        return f"Text file '{title}' by {author} ingested successfully."
    except Exception as e:
        logging.error(f"Error ingesting text file: {str(e)}")
        return f"Error ingesting text file: {str(e)}"


def ingest_folder(folder_path, keywords=None):
    """
    Ingests all text files within a specified folder.

    Parameters:
        - folder_path (str): Path to the folder containing text files.
        - keywords (str, optional): Comma-separated keywords to add to each file.

    Returns:
        - str: Combined status messages for all ingested text files.
    """
    results = []
    try:
        logging.info(f"Ingesting all text files from folder {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                result = ingest_text_file(file_path, keywords=keywords)
                results.append(result)
        logging.info("Completed ingestion of all text files in the folder.")
    except Exception as e:
        logging.exception(f"Error ingesting folder: {str(e)}")
        return f"Error ingesting folder: {str(e)}"

    return "\n".join(results)

#
# End of Function Definitions
#######################################################################################################################
