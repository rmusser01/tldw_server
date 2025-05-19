# Book_Ingestion_Lib.py
#########################################
# Library to hold functions for ingesting and processing book files.
# This library provides capabilities for:
# - Extracting metadata and content from EPUB, HTML, XML, OPML, and plain text files.
# - Converting EPUB files to Markdown.
# - Chunking text content using various strategies.
# - Performing summarization/analysis on content or chunks (via external LLM calls).
# - Processing ZIP archives containing EPUB files.
# - Ingesting text files into a database (specific functions).
#
# Note: While many processing functions are designed to be database-agnostic,
# some older functions (`ingest_text_file`, `ingest_folder`) directly interact
# with a database.
####################
# Function List (Generated documentation covers all functions below)
#
# 1. extract_epub_metadata_from_text
# 2. format_toc_item
# 3. slugify
# 4. epub_to_markdown
# 5. extract_epub_metadata_from_epub_obj
# 6. read_epub_filtered
# 7. read_epub
# 8. xml_to_markdown
# 9. opml_to_markdown
# 10. process_epub
# 11. process_zip_of_epubs
# 12. _process_markup_or_plain_text
# 13. extract_epub_metadata
# 14. ingest_text_file
# 15. ingest_folder
#
####################
#
# Imports
import os
import re
import tempfile
import zipfile
from datetime import datetime
import defusedxml.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
#
# External Imports
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import html2text
from loguru import logger

#
# Import Local
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Utils.Chunk_Lib import Chunker, improved_chunking_process, ChunkingError, \
    InvalidChunkingMethodError
from tldw_Server_API.app.core.Metrics.metrics_logger import (log_counter, log_histogram)
from tldw_Server_API.app.core.Utils.Utils import logging
#
#######################################################################################################################
# Function Definitions
#

def extract_epub_metadata_from_text(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts Title and Author from a string if specific headers are present.

    Searches for lines starting with "Title:" and "Author:" (case-insensitive)
    and extracts the subsequent text as metadata.

    Args:
        content (str): The text content to search within.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the extracted
        title (or None if not found) and author (or None if not found).
    """
    title_match = re.search(r'^Title:\s*(.*?)$', content, re.IGNORECASE | re.MULTILINE)
    author_match = re.search(r'^Author:\s*(.*?)$', content, re.IGNORECASE | re.MULTILINE)

    title = title_match.group(1).strip() if title_match else None
    author = author_match.group(1).strip() if author_match else None

    return title, author

def format_toc_item(item: Union[epub.Link, epub.Section, Any], level: int) -> str:
    """
    Formats a table of contents item into Markdown list format.

    It attempts to extract the title from `epub.Link` or `epub.Section` objects.
    For other types, it uses the string representation of the item.

    Args:
        item (Union[epub.Link, epub.Section, Any]): The TOC item, typically
            an `epub.Link` or `epub.Section` object from `ebooklib`.
        level (int): The nesting level for the TOC item, used for indentation
            in the Markdown output (e.g., level 1 for `- item`, level 2 for `  - item`).

    Returns:
        str: A Markdown-formatted string representing the TOC item (e.g., "- [Title](#slug)").
             Returns an empty string if an error occurs during formatting.
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


def slugify(text: str) -> str:
    """
    Converts a string into a slug suitable for Markdown anchors or URLs.

    The process involves:
    1. Replacing sequences of non-alphanumeric characters (excluding underscore,
       which is also replaced) with a single hyphen.
    2. Converting the string to lowercase.
    3. Stripping leading and trailing hyphens.

    Args:
        text (str): The text to slugify.

    Returns:
        str: The slugified text.
    """
    return re.sub(r'[\W_]+', '-', text.lower()).strip('-')


#
# End of Contents modification
############################################################


############################################################
#
# File Conversion Functions

def epub_to_markdown(epub_path: str) -> Tuple[str, Optional[epub.EpubBook]]:
    """
    Converts an EPUB file to Markdown format.

    The output includes a generated Table of Contents (TOC) based on the EPUB's
    TOC structure, followed by the content of each document item (chapter)
    in the EPUB. HTML elements like headings, paragraphs, and lists are
    converted to their Markdown equivalents.

    Args:
        epub_path (str): Path to the EPUB file.

    Returns:
        Tuple[str, Optional[epub.EpubBook]]: A tuple containing:
            - str: The Markdown-formatted content of the EPUB. If an error occurs
                   during conversion, this string will contain an error message
                   (e.g., "# Error converting EPUB\n\nDetails...").
            - Optional[epub.EpubBook]: The `ebooklib.epub.EpubBook` object if the
                   EPUB was successfully read, otherwise None (e.g., if the file
                   is corrupted or parsing fails).
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
    """
    Extracts title and author directly from an `ebooklib.epub.EpubBook` object's metadata.

    It queries the Dublin Core (DC) metadata fields for 'title' and 'creator'.

    Args:
        book (epub.EpubBook): The `ebooklib` book object from which to extract metadata.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the extracted
        title (or None if not found or an error occurs) and author (or None
        if not found or an error occurs).
    """
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
    Reads an EPUB file by following its spine, attempting to skip known front matter.

    This function aims to extract the main content of the book by iterating
    through the EPUB's spine (reading order). It uses a predefined list of
    common front matter filenames (e.g., "cover", "titlepage", "copyright")
    to skip these sections. The Table of Contents (TOC) is generally preserved
    if not explicitly in the skip list.
    Extracted text from HTML content is cleaned to minimize empty whitespace and
    format headings and lists into a readable plain text/markdown-like structure.

    Args:
        epub_path (str): Path to the .epub file.

    Returns:
        Tuple[str, Optional[epub.EpubBook]]: A tuple containing:
            - str: The cleaned-up text string of the book's main content.
                   Returns an empty string if parsing fails or no content is extracted.
            - Optional[epub.EpubBook]: The `ebooklib.epub.EpubBook` object if the EPUB
                   was successfully read, otherwise None.
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

    This function iterates through all document items in the EPUB, extracts
    text from headings (h1-h6) and paragraphs (p), and formats them into
    a single string. Headings are prefixed with Markdown-style '#' characters.
    The resulting text undergoes whitespace normalization.

    Args:
        file_path (str): Path to the EPUB file.

    Returns:
        Tuple[str, Optional[epub.EpubBook]]: A tuple containing:
            - str: The cleaned text content extracted from the EPUB.
            - Optional[epub.EpubBook]: The `ebooklib.epub.EpubBook` object if
                   successfully read.

    Raises:
        ValueError: If the EPUB file is invalid, corrupted, or cannot be parsed
                    by `ebooklib` (this wraps `ebooklib.epub.EpubException`).
        RuntimeError: For other unexpected errors encountered during the EPUB
                      reading process.
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
        return text, book
    except ebooklib.epub.EpubException as epub_err: # Catch specific epub errors
         logging.error(f"Ebooklib error reading EPUB {file_path}: {epub_err}", exc_info=True)
         # Reraise as ValueError for process_epub to catch nicely
         raise ValueError(f"Invalid or corrupted EPUB file: {epub_err}") from epub_err
    except Exception as e:
        logging.exception(f"Error reading EPUB file: {str(e)}")
        # Re-raise or return error indication
        # Reraise as a generic error
        raise RuntimeError(f"Unexpected error reading EPUB: {e}") from e

#
# End of epub parsing Functions
############################################################


############################################################
#
# epub Processing Functions

def xml_to_markdown(element, level=0):
    # ... (keep existing implementation - seems DB free) ...
    """
    Recursively converts an XML element and its children to a Markdown string.

    Each element's tag name is converted to a Markdown heading (level increases
    with XML depth). Text content of elements and their tail text (text following
    a child element) are included. Attributes are currently commented out but
    can be enabled.

    Args:
        element (xml.etree.ElementTree.Element or defusedxml.ElementTree.Element):
            The XML element to convert.
        level (int, optional): The current depth in the XML tree, used to
            determine Markdown heading levels. Starts at 0 for the root.
            Defaults to 0.

    Returns:
        str: The Markdown representation of the XML element and its descendants.
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
    Converts an OPML (Outline Processor Markup Language) XML structure to Markdown.

    Extracts the main title from the `<head><title>` element and then processes
    each `<outline>` element in the `<body>` into a Markdown nested list.
    It prefers the 'text' attribute of an outline item, falling back to 'title'.

    Args:
        root (xml.etree.ElementTree.Element or defusedxml.ElementTree.Element):
            The root element of the OPML XML tree.

    Returns:
        str: The Markdown representation of the OPML outline.
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
    # Removed custom_chapter_pattern - should be inside chunk_options if needed
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    summarize_recursively: bool = False,
    extraction_method: str = 'filtered' # 'markdown', 'filtered', 'basic'
) -> Dict[str, Any]:
    """
    Processes an EPUB file: extracts content & metadata, chunks, and optionally summarizes.
    Returns a dictionary with processed data, status, and errors. *No DB interaction.*

    This function is designed to be database-agnostic. It handles file reading,
    metadata extraction (with overrides), content chunking using specified options,
    and optional summarization/analysis of the content or chunks via an external
    `analyze` function.

    Args:
        file_path (str): Path to the EPUB file.
        title_override (Optional[str], optional): User-provided title to override
            metadata extracted from the EPUB. Defaults to None.
        author_override (Optional[str], optional): User-provided author to override
            metadata extracted from the EPUB. Defaults to None.
        keywords (Optional[List[str]], optional): A list of keywords to associate
            with the book. Defaults to None, resulting in an empty list.
        custom_prompt (Optional[str], optional): Custom user prompt to be used
            for summarization if `perform_analysis` is True. Defaults to None.
        system_prompt (Optional[str], optional): System prompt (e.g., instructions
            for the LLM) to be used for summarization. Defaults to None.
        perform_chunking (bool, optional): If True, the extracted content is chunked.
            Defaults to True. If False, the entire content is treated as a single chunk.
        chunk_options (Optional[Dict[str, Any]], optional): Dictionary of options for
            the chunking process. Examples:
            `{'method': 'ebook_chapters', 'max_size': 1500, 'overlap': 200}`.
            If 'method' is 'ebook_chapters', it uses chapter-based chunking.
            Other methods like 'recursive', 'fixed_size' can also be specified.
            Defaults to `{'method': 'ebook_chapters', 'max_size': 1500, 'overlap': 200}`.
        perform_analysis (bool, optional): If True, summarization/analysis is performed
            on the chunks (or the whole content if not chunked). Requires `api_name`
            and `api_key`. Defaults to False.
        api_name (Optional[str], optional): Name of the API/model to use for summarization
            (e.g., "openai_gpt3.5_turbo"). Required if `perform_analysis` is True.
            Defaults to None.
        api_key (Optional[str], optional): API key for the summarization service.
            Required if `perform_analysis` is True. Defaults to None.
        summarize_recursively (bool, optional): If True, `perform_analysis` is True,
            and multiple chunks are generated and summarized, their individual summaries
            are combined and then summarized again to create a final overall summary.
            Defaults to False.
        extraction_method (str, optional): Method to use for extracting content from
            the EPUB. Options are:
            - 'markdown': Converts EPUB to full Markdown including TOC (uses `epub_to_markdown`).
            - 'filtered': Reads EPUB spine, skips front matter, cleans text (uses `read_epub_filtered`).
            - 'basic': Reads all document items, extracts Hx/P tags (uses `read_epub`).
            Defaults to 'filtered'. If the chosen method fails, it attempts a fallback to 'basic'.

    Returns:
        Dict[str, Any]: A dictionary containing the processing results. Keys include:
            - "status" (str): "Success", "Warning" (if non-critical issues occurred), or "Error".
            - "input_ref" (str): The input `file_path`.
            - "processing_source" (str): The `file_path` that was processed.
            - "media_type" (str): Always "ebook".
            - "content" (Optional[str]): The extracted text content from the EPUB.
            - "metadata" (Dict):
                - "title" (Optional[str]): Final title (override, extracted, or filename).
                - "author" (Optional[str]): Final author (override, extracted, or "Unknown").
                - "raw" (Optional[Dict]): Raw metadata from the `ebooklib` object.
                - "source_filename" (str): Original filename of the EPUB.
            - "chunks" (Optional[List[Dict]]): List of chunk dictionaries. Each chunk dict
              contains 'text' (str) and 'metadata' (dict, which may include chunk_index,
              analysis if summarized per chunk, etc.). `None` if chunking fails catastrophically
              or if no content was extracted.
            - "analysis" (Optional[str]): The final summary/analysis result for the entire book.
              `None` if `perform_analysis` was False or failed.
            - "keywords" (List[str]): The list of keywords passed in or an empty list.
            - "error" (Optional[str]): An error message if `status` is "Error".
            - "warnings" (Optional[List[str]]): A list of warning messages if `status` is "Warning".
              `None` if no warnings.
            - "analysis_details" (Dict): Information about the summarization process, like
              model used, if prompts were used, and if recursion was applied.
            - "parser_used" (Optional[str]): Name of the EPUB parser function actually used,
              especially if a fallback occurred (e.g., "read_epub (fallback)").
    """
    start_time = datetime.now()
    result: Dict[str, Any] = {
        "status": "Pending",
        "input_ref": file_path,
        "processing_source": file_path,
        "media_type": "ebook",
        "content": None,
        "metadata": {"title": None, "author": None, "raw": None},
        "chunks": None,
        "analysis": None, # Renamed from summary for consistency
        "keywords": keywords or [],
        "error": None,
        "warnings": [], # Initialize as list
        "analysis_details": { # Initialize
            "analysis_model": api_name if perform_analysis else None,
            "custom_prompt_used": custom_prompt if perform_analysis else None,
            "system_prompt_used": system_prompt if perform_analysis else None,
            "summarized_recursively": summarize_recursively if perform_analysis else False,
        }
    }
    log_counter("epub_processing_attempt", labels={"file_path": file_path, "extractor": extraction_method})

    extracted_text: Optional[str] = None
    ebook_obj: Optional[epub.EpubBook] = None

    try:
        logging.info(f"Processing EPUB file from {file_path} using extractor '{extraction_method}'")
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
        except ValueError as primary_extract_err: # Catch specific errors raised by readers
            logging.warning(f"Extractor '{extraction_method}' failed for {file_path}: {primary_extract_err}. Trying basic read_epub.")
            result["warnings"].append(f"Extraction method '{extraction_method}' failed, used basic fallback.")
            result["parser_used"] = "read_epub (fallback)" # Indicate fallback
            try:
                extracted_text, ebook_obj = read_epub(file_path) # Fallback attempt
            except (ValueError, RuntimeError) as fallback_err:
                 # If fallback also fails, it's a critical error
                 raise ValueError(f"Failed to extract text from EPUB '{Path(file_path).name}' even with fallback: {fallback_err}") from fallback_err
        except Exception as primary_extract_err: # Catch unexpected errors
            logging.warning(f"Unexpected error with extractor '{extraction_method}' for {file_path}: {primary_extract_err}. Trying basic read_epub.", exc_info=True)
            result["warnings"].append(f"Extraction method '{extraction_method}' failed (unexpected), used basic fallback.")
            result["parser_used"] = "read_epub (fallback)"
            try:
                 extracted_text, ebook_obj = read_epub(file_path) # Fallback attempt
            except (ValueError, RuntimeError) as fallback_err:
                 raise ValueError(f"Failed to extract text from EPUB '{Path(file_path).name}' even with fallback: {fallback_err}") from fallback_err


        if not extracted_text or ebook_obj is None:
            # This should ideally be caught by exceptions above, but double-check
            raise ValueError(f"EPUB extraction yielded no text or book object for '{Path(file_path).name}'.")

        result["content"] = extracted_text
        logging.debug(f"Extracted EPUB content. Length: {len(extracted_text)}")

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
            "raw": ebook_obj.metadata,
            "source_filename": file_path_obj.name
        }
        logging.debug(f"Final metadata - Title: {final_title}, Author: {final_author}")

        # 2. Chunking
        processed_chunks = None
        if perform_chunking:
            effective_chunk_options = chunk_options or {}
            # Default to ebook_chapters for EPUBs unless specified otherwise in chunk_options.
            # This method name must match one of the methods handled by Chunker.chunk_text in Chunk_Lib.py
            effective_chunk_options.setdefault('method', 'ebook_chapters') # UPDATED

            # Default size/overlap are used by 'ebook_chapters' for sub-chunking very large chapters,
            # or by other chunking methods if 'method' is overridden in chunk_options.
            effective_chunk_options.setdefault('max_size', 1500)
            effective_chunk_options.setdefault('overlap', 200)

            logging.info(f"Chunking ebook content with options: {effective_chunk_options}")
            try:
                # Use improved_chunking_process, which handles Chunker instantiation and metadata enrichment.
                # It will use the 'method' from effective_chunk_options to dispatch correctly.
                processed_chunks = improved_chunking_process(
                    text=extracted_text,
                    chunk_options_dict=effective_chunk_options
                    # llm_call_function_for_chunker and llm_api_config_for_chunker can be passed here
                    # if any chunking method requiring LLM calls (like 'rolling_summarize') was to be used.
                    # For 'ebook_chapters', they are not directly needed unless sub-chunking uses an LLM method.
                )

                if not processed_chunks and extracted_text.strip(): # If no chunks but text existed
                     logging.warning(f"Chunking (via improved_chunking_process) produced no chunks for {file_path}, though text was present. Using full text as one chunk.")
                     # Fallback to a single chunk with basic metadata if improved_chunking_process returns empty for non-empty text.
                     # This scenario should be rare if _chunk_ebook_by_chapters is robust.
                     processed_chunks = [{'text': extracted_text, 'metadata': {
                         'chunk_index': 1, 'total_chunks': 1,
                         'chunk_method': 'fallback_single_chunk', 'error': 'Original chunking yielded no results'
                         }}]
                     result["warnings"].append("Chunking yielded no results; using full text as a single chunk (fallback).")
                elif not processed_chunks: # No chunks and text was empty/whitespace
                     logging.info(f"Chunking produced no chunks for {file_path} (extracted text was empty or whitespace).")
                     # processed_chunks remains empty or None, which is appropriate.
                else:
                     logging.info(f"Total chunks created: {len(processed_chunks)}")
                     log_histogram("epub_chunks_created", len(processed_chunks), labels={"file_path": file_path})

                result["chunks"] = processed_chunks # Assign the list of chunk dictionaries

            # Catch specific exceptions from Chunker/improved_chunking_process
            except InvalidChunkingMethodError as icme:
                logging.error(f"Invalid chunking method specified for {file_path}: {icme}", exc_info=True)
                result["warnings"].append(f"Chunking failed (invalid method): {str(icme)}")
                processed_chunks = [{'text': extracted_text, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed (invalid method): {str(icme)}"}}]
                result["chunks"] = processed_chunks
            except ChunkingError as ce:
                logging.error(f"ChunkingError during chunking for {file_path}: {ce}", exc_info=True)
                result["warnings"].append(f"Chunking failed: {str(ce)}")
                processed_chunks = [{'text': extracted_text, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {str(ce)}"}}]
                result["chunks"] = processed_chunks
            except Exception as chunk_err: # Catch any other unexpected errors during chunking
                logging.error(f"Unexpected error during chunking for {file_path}: {chunk_err}", exc_info=True)
                result["warnings"].append(f"Chunking failed (unexpected): {str(chunk_err)}")
                # Fallback: use full text as one chunk
                processed_chunks = [{'text': extracted_text, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed (unexpected): {str(chunk_err)}"}}]
                result["chunks"] = processed_chunks

        else:
             # If not chunking, create a single chunk containing the whole text
             # The metadata structure should be consistent with what improved_chunking_process would provide for a single chunk.
             processed_chunks = [{'text': extracted_text, 'metadata': {
                 'chunk_index': 1,
                 'total_chunks': 1,
                 'chunk_method': 'none', # Indicate no method was applied
                 'relative_position': 0.0,
                 'language': 'unknown' # Or detect language if important even when not chunking
                 }}]
             result["chunks"] = processed_chunks
             logging.info("Chunking disabled. Using full text as one chunk.")


        # 3. Summarization / Analysis
        final_analysis = None # Renamed for consistency
        if perform_analysis and api_name and api_key and processed_chunks:
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks of EPUB '{final_title}'.")
            log_counter("epub_summarization_attempt", value=len(processed_chunks), labels={"file_path": file_path, "api_name": api_name})
            chunk_summaries = []
            summarized_chunks_for_result = [] # Keep track of chunks with summaries added

            for i, chunk_dict in enumerate(processed_chunks): # Iterate over list of dictionaries
                chunk_text = chunk_dict.get('text', '')
                # Preserve existing metadata from chunking, add analysis to it
                chunk_metadata = chunk_dict.get('metadata', {}).copy() # Work on a copy
                if chunk_text:
                    try:
                        # Match expected args for summarize function
                        analysis_text = analyze(
                            api_name=api_name,
                            input_data=chunk_text,
                            custom_prompt_arg=custom_prompt, # Use this name
                            api_key=api_key,
                            system_message=system_prompt, # Use this name
                            temp=None,
                            recursive_summarization=False, # Summarize chunk first
                        )
                        if analysis_text and analysis_text.strip():
                            chunk_summaries.append(analysis_text)
                            chunk_metadata['analysis'] = analysis_text # Add analysis to chunk metadata
                        else:
                            chunk_metadata['analysis'] = None
                            logging.debug(f"Summarization yielded empty result for chunk {i+1}/{len(processed_chunks)} of {file_path}.")
                    except Exception as summ_err:
                        logging.warning(f"Summarization failed for chunk {i+1}/{len(processed_chunks)} of {file_path}: {summ_err}", exc_info=True)
                        chunk_metadata['analysis'] = f"[Summarization Error: {str(summ_err)}]"
                        result["warnings"].append(f"Summarization failed for chunk {i+1}: {str(summ_err)}")

                # Update the chunk dictionary with new/updated metadata
                updated_chunk_dict = {'text': chunk_text, 'metadata': chunk_metadata}
                summarized_chunks_for_result.append(updated_chunk_dict)

            result["chunks"] = summarized_chunks_for_result # Update chunks in result with analysis metadata

            # Combine chunk summaries (potentially recursively)
            if chunk_summaries:
                if summarize_recursively and len(chunk_summaries) > 1:
                    logging.info(f"Performing recursive summarization on {len(chunk_summaries)} chunk summaries for EPUB '{final_title}'.")
                    try:
                        final_analysis = analyze(
                            api_name=api_name,
                            input_data="\n\n---\n\n".join(chunk_summaries),
                            custom_prompt_arg=custom_prompt or "Provide a concise overall summary of the following chapter summaries.",
                            api_key=api_key,
                            system_message=system_prompt,
                            temp=None,
                            recursive_summarization=False, # Final pass
                        )
                        if not final_analysis or not final_analysis.strip():
                            logging.warning(f"Recursive summarization for {file_path} yielded empty result. Falling back to joined summaries.")
                            final_analysis = "\n\n---\n\n".join(chunk_summaries) # Fallback
                            result["warnings"].append("Recursive summarization yielded empty result.")
                        else:
                             log_counter("epub_recursive_summarization_success", labels={"file_path": file_path})

                    except Exception as rec_summ_err:
                         logging.error(f"Recursive summarization failed for {file_path}: {rec_summ_err}", exc_info=True)
                         final_analysis = f"[Recursive Summarization Error: {str(rec_summ_err)}]\n\n" + "\n\n---\n\n".join(chunk_summaries)
                         result["warnings"].append(f"Recursive summarization failed: {str(rec_summ_err)}")
                         log_counter("epub_recursive_summarization_error", labels={"file_path": file_path, "error": str(rec_summ_err)})
                else:
                    # Simple join if not recursive or only one summary
                    final_analysis = "\n\n---\n\n".join(chunk_summaries)
                    if len(chunk_summaries) > 1 : logging.info(f"Combined {len(chunk_summaries)} chunk summaries (non-recursive).")
                    else: logging.info(f"Using single chunk analysis as final analysis.")

            result["analysis"] = final_analysis # Store final combined analysis
            log_counter("epub_chunks_summarized", value=len(chunk_summaries), labels={"file_path": file_path})
            logging.info(f"Summarization processing completed for EPUB '{final_title}'.")

        elif not perform_analysis: logging.info(f"Summarization disabled for EPUB '{final_title}'.")
        elif not api_name or not api_key: logging.warning(f"Summarization skipped for EPUB '{final_title}': API name or key not provided.")
        elif not processed_chunks: logging.warning(f"Summarization skipped for EPUB '{final_title}': No processable chunks available.")
        else: logging.warning(f"Summarization skipped for EPUB '{final_title}' due to unknown condition.")


        # Final status determination
        if result["error"]: # If an error was set before this point (e.g. extraction failed hard)
             result["status"] = "Error"
        elif result["warnings"]:
            result["status"] = "Warning"
        else:
            result["status"] = "Success"

        if result["status"] != "Error": # Don't log success/warning if it already failed
             log_counter(f"epub_processing_{result['status'].lower()}", labels={"file_path": file_path})

    except FileNotFoundError:
        logging.error(f"EPUB file not found: {file_path}")
        result["status"] = "Error"
        result["error"] = "File not found"
        log_counter("epub_processing_error", labels={"file_path": file_path, "error": "FileNotFoundError"})
    except ValueError as ve: # Catch errors from extraction/parsing
         logging.error(f"Value error processing EPUB {file_path}: {str(ve)}", exc_info=False) # Don't need full trace for expected parse errors
         result["status"] = "Error"
         result["error"] = str(ve)
         log_counter("epub_processing_error", labels={"file_path": file_path, "error": "ValueError"})
    except RuntimeError as rterr: # Catch critical runtime errors from extraction
         logging.error(f"RuntimeError processing EPUB {file_path}: {str(rterr)}", exc_info=True)
         result["status"] = "Error"
         result["error"] = str(rterr)
         log_counter("epub_processing_error", labels={"file_path": file_path, "error": "RuntimeError"})
    except Exception as e:
        logging.exception(f"Unexpected error processing EPUB {file_path}: {str(e)}")
        result["status"] = "Error"
        result["error"] = f"Unexpected processing error: {str(e)}"
        log_counter("epub_processing_error", labels={"file_path": file_path, "error": type(e).__name__})


    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    # Ensure status for logging duration is the final one
    log_histogram("epub_processing_duration", processing_time, labels={"file_path": file_path, "status": result["status"]})

    # Final logging
    if result["status"] == "Success":
        logging.info(f"Successfully processed EPUB: {result.get('metadata',{}).get('title', file_path)} in {processing_time:.2f}s")
    elif result["status"] == "Warning":
        logging.warning(f"Processed EPUB with warnings: {result.get('metadata',{}).get('title', file_path)} in {processing_time:.2f}s. Warnings: {result['warnings']}")
    else: # Error status
        logging.error(f"Failed to process EPUB: {file_path} in {processing_time:.2f}s. Error: {result.get('error', 'Unknown')}")

    # Ensure warnings list is None if empty for cleaner JSON output
    if not result["warnings"]:
        result["warnings"] = None

    return result

############################################################
# ZIP Processing Function (DB-Free)

def process_zip_of_epubs(
    zip_file_path: str,
    keywords: Optional[List[str]] = None,
    # Pass all other relevant options down to process_epub
    **epub_options # Collects title_override, author_override, perform_chunking, chunk_options etc.
    ) -> List[Dict[str, Any]]:
    """
    Processes a ZIP file containing multiple EPUB files.

    Extracts each EPUB file from the ZIP archive into a temporary directory
    and then processes it using the `process_epub` function, passing along
    any provided `keywords` and `**epub_options`.

    This function is designed to be database-agnostic.

    Args:
        zip_file_path (str): Path to the ZIP file.
        keywords (Optional[List[str]], optional): Base keywords to apply to all
            EPUBs found within the ZIP. These will be passed to `process_epub`.
            Defaults to None.
        **epub_options (Any): Arbitrary keyword arguments that will be passed
            directly to the `process_epub` function for each EPUB file.
            Examples: `title_override`, `author_override`, `perform_chunking`,
            `chunk_options`, `perform_analysis`, `api_name`, `api_key`, etc.

    Returns:
        List[Dict[str, Any]]: A list where each item is the result dictionary
        from `process_epub` for a single EPUB file.
        If the ZIP file itself is invalid, cannot be extracted, or an error occurs
        during the ZIP handling phase, a list containing a single error dictionary
        pertaining to the ZIP file operation will be returned.
        If no .epub files are found in the ZIP, a list with a warning result
        for the ZIP file is returned.
    """
    results = []
    temp_dir_path_obj = None
    try:
        # Use context manager for reliable cleanup
        with tempfile.TemporaryDirectory(prefix="epub_zip_") as temp_dir:
            temp_dir_path_obj = Path(temp_dir)
            logging.info(f"Extracting ZIP file {zip_file_path} to temporary directory {temp_dir_path_obj}")
            log_counter("zip_processing_attempt", labels={"zip_path": zip_file_path})

            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path_obj)
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

            epub_files = list(temp_dir_path_obj.rglob('*.epub'))
            log_histogram("epub_files_in_zip", len(epub_files), labels={"zip_path": zip_file_path})
            logging.info(f"Found {len(epub_files)} EPUB files in ZIP.")

            if not epub_files:
                 results.append({
                     "status": "Warning", "input_ref": zip_file_path, "media_type": "zip",
                     "warnings": ["No .epub files found within the ZIP archive."], "error": None
                 })

            for epub_path in epub_files:
                epub_filename = epub_path.name
                logging.info(f"Processing EPUB file '{epub_filename}' extracted from ZIP.")
                try:
                    # Call the refactored process_epub
                    # Pass keywords and unpack other options collected in epub_options
                    result = process_epub(
                        file_path=str(epub_path),
                        keywords=keywords,
                        **epub_options # Pass title_override, perform_analysis etc.
                    )
                    # Add zip source info for clarity
                    result["source_zip"] = zip_file_path
                    result["original_filename_in_zip"] = epub_filename
                    # processing_source is already set to epub_path by process_epub
                    results.append(result)
                except Exception as single_epub_err:
                     # Catch errors during the processing of a single epub from the zip
                     logging.exception(f"Error processing '{epub_filename}' from zip {zip_file_path}: {single_epub_err}")
                     results.append({
                         "status": "Error",
                         "input_ref": epub_filename, # Use filename within zip as ref
                         "processing_source": str(epub_path),
                         "media_type": "ebook",
                         "error": f"Failed processing from ZIP: {single_epub_err}",
                         "source_zip": zip_file_path,
                         "original_filename_in_zip": epub_filename,
                         # Add other default fields for consistency
                         "content": None, "metadata": None, "chunks": None, "analysis": None,
                         "keywords": keywords or [], "warnings": None, "analysis_details": None,
                     })

            logging.info(f"Completed processing all EPUB files in the ZIP: {zip_file_path}")
            log_counter("zip_processing_success", labels={"zip_path": zip_file_path})

    except Exception as e:
        # Catch errors related to Temp dir creation or other unexpected issues
        logging.exception(f"Error processing ZIP file {zip_file_path}: {str(e)}")
        log_counter("zip_processing_error", labels={"zip_path": zip_file_path, "error": str(e)})
        # Return a single error result for the whole zip if setup fails
        return [{
            "status": "Error", "input_ref": zip_file_path, "media_type": "zip",
            "error": f"Error processing ZIP file itself: {str(e)}"
        }]

    return results


############################################################
# Markup / Plain Text Processing Function (DB-Free - for next step)

def _process_markup_or_plain_text(
    file_path: str,
    file_type: str, # 'html', 'xml', 'opml', 'text'
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None,
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

    This function reads the specified file, converts its content to Markdown
    (if applicable, e.g., for HTML, XML, OPML), extracts basic metadata,
    optionally chunks the content, and optionally performs summarization/analysis.
    It is designed to be database-agnostic.

    Args:
        file_path (str): Path to the input file.
        file_type (str): The type of the file. Must be one of 'html', 'xml',
                         'opml', or 'text'.
        title_override (Optional[str], optional): User-provided title. Defaults to None.
        author_override (Optional[str], optional): User-provided author. Defaults to None.
        keywords (Optional[List[str]], optional): List of keywords. Defaults to None.
        perform_chunking (bool, optional): Whether to chunk the content. Defaults to True.
        chunk_options (Optional[Dict[str, Any]], optional): Options for chunking,
            e.g., `{'method': 'recursive', 'max_size': 1000, 'overlap': 200}`.
            Defaults to `{'method': 'recursive', ...}`.
        perform_analysis (bool, optional): Whether to perform summarization.
            Requires `api_name` and `api_key`. Defaults to False.
        api_name (Optional[str], optional): API name for summarization. Defaults to None.
        api_key (Optional[str], optional): API key for summarization. Defaults to None.
        custom_prompt (Optional[str], optional): Custom prompt for summarization. Defaults to None.
        system_prompt (Optional[str], optional): System prompt for summarization. Defaults to None.
        summarize_recursively (bool, optional): If True and multiple chunks are
            summarized, their summaries are combined and summarized again. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing processing results, structured similarly
        to `process_epub`'s output, with keys like "status", "input_ref", "content",
        "metadata", "chunks", "analysis", "error", "warnings".
        "media_type" will reflect the `file_type` (e.g., "html", "xml").
        "source_format" will also be set to `file_type`.

    Raises:
        ValueError: If `file_type` is unsupported, or if content extraction/parsing
                    fails for the given file type (e.g., malformed XML).
                    These are typically caught and returned in the result dict.
    """
    start_time = datetime.now()
    # Initialize title for logging, ensuring it's always defined
    # It will be refined later if processing is successful.
    final_title_for_logging = Path(file_path).name

    media_type = file_type if file_type in ['html', 'xml', 'opml', 'text'] else "document"
    result: Dict[str, Any] = {
        "status": "Pending", "input_ref": file_path, "processing_source": file_path,
        "media_type": media_type, "source_format": file_type, "content": None,
        "metadata": {"title": None, "author": None, "raw": None}, "chunks": None,
        "analysis": None, "keywords": keywords or [], "error": None, "warnings": [],
        "analysis_details": {
            "analysis_model": api_name if perform_analysis else None,
            "custom_prompt_used": custom_prompt if perform_analysis else None,
            "system_prompt_used": system_prompt if perform_analysis else None,
            "summarized_recursively": summarize_recursively if perform_analysis else False,
        }
    }
    log_counter(f"{file_type}_processing_attempt", labels={"file_path": file_path})

    markdown_content: Optional[str] = None
    extracted_title: Optional[str] = None
    extracted_author: str = "Unknown"
    raw_metadata: Dict[str, Any] = {}

    try:
        logging.info(f"Processing {file_type} file from {file_path}")
        file_path_obj = Path(file_path)
        filename_stem = file_path_obj.stem
        final_title_for_logging = filename_stem # Refine with stem if path is valid

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
                markdown_content = xml_to_markdown(root)
                title_elem = root.find('.//title')
                extracted_title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
                raw_metadata = {'xml_root_tag': root.tag}
            except ET.ParseError as xml_err:
                 raise ValueError(f"Failed to parse XML file {file_path}: {xml_err}") from xml_err

        elif file_type == 'opml':
             try:
                 tree = ET.parse(file_path)
                 root = tree.getroot()
                 markdown_content = opml_to_markdown(root)
                 title_elem = root.find("./head/title")
                 extracted_title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
                 raw_metadata = {'opml_title': extracted_title}
             except ET.ParseError as opml_err:
                 raise ValueError(f"Failed to parse OPML file {file_path}: {opml_err}") from opml_err

        elif file_type == 'text':
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read()
                epub_title, epub_author = extract_epub_metadata_from_text(markdown_content)
                extracted_title = epub_title
                if epub_author: extracted_author = epub_author
                raw_metadata = {'maybe_epub_title': epub_title, 'maybe_epub_author': epub_author}
            except Exception as text_err:
                 raise ValueError(f"Failed to read text file {file_path}: {text_err}") from text_err
        else:
            raise ValueError(f"Unsupported file type for processing: {file_type}")

        if markdown_content: # Check if None before regex
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content).strip()
        if not markdown_content: # Check after potential stripping
            raise ValueError(f"Content extraction failed or resulted in empty content for {file_path}.")
        result["content"] = markdown_content

        # Finalize metadata
        final_title = title_override or extracted_title or filename_stem
        final_author = author_override or extracted_author
        result["metadata"] = {
            "title": final_title,
            "author": final_author,
            "source_filename": file_path_obj.name,
            "raw": raw_metadata
        }
        final_title_for_logging = final_title # Update for accurate logging if processing was successful
        logging.debug(f"Final metadata - Title: {final_title_for_logging}, Author: {final_author}")

        # 2. Chunking
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
                     result["warnings"].append("Chunking yielded no results; using full text.")
                else:
                     logging.info(f"Total chunks created: {len(processed_chunks)}")
                     log_histogram(f"{file_type}_chunks_created", len(processed_chunks), labels={"file_path": file_path})

                result["chunks"] = processed_chunks

            except Exception as chunk_err:
                logging.error(f"Chunking failed for {file_path}: {chunk_err}", exc_info=True)
                result["warnings"].append(f"Chunking failed: {chunk_err}")
                processed_chunks = [{'text': markdown_content, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {chunk_err}"}}]
                result["chunks"] = processed_chunks # Store fallback chunk
        else:
             processed_chunks = [{'text': markdown_content, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks
             logging.info("Chunking disabled. Using full text as one chunk.")

        # 3. Summarization / Analysis
        final_analysis = None
        # `processed_chunks` is guaranteed to be non-empty list here if markdown_content was valid.
        if perform_analysis and api_name and api_key:
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks of {file_type}.")
            log_counter(f"{file_type}_summarization_attempt", value=len(processed_chunks), labels={"file_path": file_path, "api_name": api_name})
            chunk_summaries = []
            summarized_chunks_for_result = []

            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                if chunk_text:
                    try:
                        summary_text = analyze(api_name, chunk_text, custom_prompt, api_key, system_prompt, None, False, )
                        if summary_text and summary_text.strip():
                            chunk_summaries.append(summary_text)
                            chunk_metadata['summary'] = summary_text
                        else:
                            chunk_metadata['summary'] = None
                            logging.debug(f"Summarization yielded empty result for chunk {i} of {file_path}.")
                    except Exception as summ_err:
                        logging.warning(f"Summarization failed for chunk {i} of {file_path}: {summ_err}")
                        chunk_metadata['summary'] = f"[Summarization Error: {summ_err}]"
                        result["warnings"].append(f"Summarization failed for chunk {i}: {summ_err}")

                chunk['metadata'] = chunk_metadata
                summarized_chunks_for_result.append(chunk)

            result["chunks"] = summarized_chunks_for_result

            if chunk_summaries:
                if summarize_recursively and len(chunk_summaries) > 1:
                    logging.info("Performing recursive summarization on chunk summaries.")
                    try:
                        final_analysis = analyze(api_name, "\n\n---\n\n".join(chunk_summaries), custom_prompt or "Provide an overall summary.", api_key, system_prompt, None, False, )
                        if not final_analysis or not final_analysis.strip():
                            logging.warning(f"Recursive summarization for {file_path} yielded empty result. Falling back.")
                            final_analysis = "\n\n---\n\n".join(chunk_summaries)
                            result["warnings"].append("Recursive summarization yielded empty result.")
                        else:
                             log_counter(f"{file_type}_recursive_summarization_success", labels={"file_path": file_path})
                    except Exception as rec_summ_err:
                         logging.error(f"Recursive summarization failed for {file_path}: {rec_summ_err}")
                         final_analysis = f"[Recursive Summarization Error: {rec_summ_err}]\n\n" + "\n\n---\n\n".join(chunk_summaries)
                         result["warnings"].append(f"Recursive summarization failed: {rec_summ_err}")
                         log_counter(f"{file_type}_recursive_summarization_error", labels={"file_path": file_path, "error": str(rec_summ_err)})
                else:
                    final_analysis = "\n\n---\n\n".join(chunk_summaries)

            result["analysis"] = final_analysis
            log_counter(f"{file_type}_chunks_summarized", value=len(chunk_summaries), labels={"file_path": file_path})
            logging.info("Summarization processing completed.")
        # Log summarization skipped reasons
        elif not perform_analysis:
            logging.info("Summarization disabled.")
        elif not api_name or not api_key: # This implies perform_analysis was true
            logging.warning("Summarization skipped: API name or key not provided.")
        # The case for `not processed_chunks` is unreachable here if content was extracted.
        # If `processed_chunks` was empty for some reason, summarization loop above handles it gracefully.

        # Final status determination
        if result["warnings"]:
            result["status"] = "Warning"
        else:
            result["status"] = "Success"
        # Removed redundant log_counter here; the one at the end covers all statuses.

    except FileNotFoundError:
        logging.error(f"{file_type.capitalize()} file not found: {file_path}")
        result["status"] = "Error"
        result["error"] = "File not found"
        log_counter(f"{file_type}_processing_error", labels={"file_path": file_path, "error": "FileNotFoundError"})
    except ValueError as ve:
         logging.error(f"Processing value error for {file_type} file {file_path}: {str(ve)}")
         result["status"] = "Error"
         result["error"] = str(ve)
         log_counter(f"{file_type}_processing_error", labels={"file_path": file_path, "error": "ValueError"})
    except Exception as e:
        logging.exception(f"Error processing {file_type} file {file_path}: {str(e)}")
        result["status"] = "Error"
        result["error"] = str(e)
        log_counter(f"{file_type}_processing_error", labels={"file_path": file_path, "error": type(e).__name__})

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    log_histogram(f"{file_type}_processing_duration", processing_time, labels={"file_path": file_path})
    # Use final_title_for_logging which is guaranteed to be defined
    logging.info(f"{file_type.capitalize()} file '{final_title_for_logging}' processed with status: {result['status']}.")
    # This log_counter correctly reflects the final status (Success, Warning, or Error)
    log_counter(f"{file_type}_processing_{result['status'].lower()}", labels={"file_path": file_path})

    if not result["warnings"]: # Convert empty list to None for cleaner output if desired
        result["warnings"] = None
    return result


def extract_epub_metadata(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts Title and Author from a string based on "Title:" and "Author:" prefixes.

    Searches for lines starting with "Title:" and "Author:" (case-sensitive,
    expects newline after value) and extracts the subsequent text.

    Args:
        content (str): The text content to search within.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing the extracted
        title (or None if not found) and author (or None if not found).
    """
    title_match = re.search(r'Title:\s*(.*?)\n', content)
    author_match = re.search(r'Author:\s*(.*?)\n', content)

    title = title_match.group(1) if title_match else None
    author = author_match.group(1) if author_match else None

    return title, author


def ingest_text_file(file_path, title=None, author=None, keywords=None):
    """
    Ingests a plain text file into the database with optional metadata.

    This function reads a text file, attempts to determine its title and author
    (using provided values, extracting from content if 'epub_converted' keyword
    is present, or defaulting to filename/Unknown), and then adds it to a
    database using `add_media_with_keywords`.

    **Note:** This function directly interacts with a database.

    Args:
        file_path (str): Path to the text file.
        title (Optional[str], optional): Title of the document. If None, the function
            attempts to extract it (if 'epub_converted' in keywords) or uses the
            filename (without extension). Defaults to None.
        author (Optional[str], optional): Author of the document. If None, the function
            attempts to extract it (if 'epub_converted' in keywords) or defaults to
            'Unknown'. Defaults to None.
        keywords (Optional[str], optional): A comma-separated string of keywords.
            If None, defaults to 'text_file,epub_converted'.
            'text_file' and 'epub_converted' are always added if keywords are provided.
            Defaults to None.

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
    Ingests all text files (.txt) within a specified folder into the database.

    Iterates through all files in the given `folder_path`. For each file ending
    with '.txt' (case-insensitive), it calls `ingest_text_file` to process and
    add it to the database.

    **Note:** This function directly interacts with a database via `ingest_text_file`.

    Args:
        folder_path (str): Path to the folder containing text files.
        keywords (Optional[str], optional): A comma-separated string of keywords
            to be added to each text file ingested from this folder. These keywords
            are passed to `ingest_text_file`. Defaults to None.

    Returns:
        str: A string containing combined status messages from each `ingest_text_file`
             call, separated by newlines. If an error occurs while accessing or
             listing the folder, an error message for the folder operation is returned.
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
