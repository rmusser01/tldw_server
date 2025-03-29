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
import html2text
import csv
#
# External Imports
import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
#
# Import Local
from App_Function_Libraries.DB.DB_Manager import add_media_with_keywords, add_media_to_database
from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from App_Function_Libraries.Chunk_Lib import chunk_ebook_by_chapters
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from App_Function_Libraries.Utils.Utils import logging


#
#######################################################################################################################
# Function Definitions
#

def import_epub(file_path,
                title=None,
                author=None,
                keywords=None,
                custom_prompt=None,
                system_prompt=None,
                summary=None,
                auto_summarize=False,
                api_name=None,
                api_key=None,
                chunk_options=None,
                custom_chapter_pattern=None
                ):
    """
    Imports an EPUB file, extracts its content, chunks it, optionally summarizes it, and adds it to the database.

    Parameters:
        - file_path (str): Path to the EPUB file.
        - title (str, optional): Title of the book.
        - author (str, optional): Author of the book.
        - keywords (str, optional): Comma-separated keywords for the book.
        - custom_prompt (str, optional): Custom user prompt for summarization.
        - summary (str, optional): Predefined summary of the book.
        - auto_summarize (bool, optional): Whether to auto-summarize the chunks.
        - api_name (str, optional): API name for summarization.
        - api_key (str, optional): API key for summarization.
        - chunk_options (dict, optional): Options for chunking.
        - custom_chapter_pattern (str, optional): Custom regex pattern for chapter detection.

    Returns:
        - str: Status message indicating success or failure.
    """
    try:
        logging.info(f"Importing EPUB file from {file_path}")
        log_counter("epub_import_attempt", labels={"file_path": file_path})

        start_time = datetime.now()

        # Convert EPUB to Markdown
        markdown_content = epub_to_markdown(file_path)
        logging.debug("Converted EPUB to Markdown.")

        # Extract metadata if not provided
        if not title or not author:
            extracted_title, extracted_author = extract_epub_metadata(markdown_content)
            title = title or extracted_title or os.path.splitext(os.path.basename(file_path))[0]
            author = author or extracted_author or "Unknown"
            logging.debug(f"Extracted metadata - Title: {title}, Author: {author}")

        # Process keywords
        keyword_list = [kw.strip() for kw in keywords.split(',')] if keywords else []
        logging.debug(f"Keywords: {keyword_list}")

        # Set default chunk options if not provided
        if chunk_options is None:
            chunk_options = {
                'method': 'chapter',
                'max_size': 500,
                'overlap': 200,
                'custom_chapter_pattern': custom_chapter_pattern
            }
        else:
            # Ensure 'method' is set to 'chapter' when using chapter chunking
            chunk_options.setdefault('method', 'chapter')
            chunk_options.setdefault('custom_chapter_pattern', custom_chapter_pattern)

        # Chunk the content by chapters
        chunks = chunk_ebook_by_chapters(markdown_content, chunk_options)
        logging.info(f"Total chunks created: {len(chunks)}")
        log_histogram("epub_chunks_created", len(chunks), labels={"file_path": file_path})

        if chunks:
            logging.debug(f"Structure of first chunk: {chunks[0].keys()}")

        # Handle summarization if enabled
        if auto_summarize and api_name and api_key:
            logging.info("Auto-summarization is enabled.")
            summarized_chunks = []
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    summary_text = perform_summarization(api_name, chunk_text, custom_prompt, api_key,
                                                            recursive_summarization=False, temp=None,
                                                            system_message=system_prompt
                                                            )
                    chunk['metadata']['summary'] = summary_text
                    summarized_chunks.append(chunk)

            chunks = summarized_chunks
            logging.info("Summarization of chunks completed.")
            log_counter("epub_chunks_summarized", value=len(chunks), labels={"file_path": file_path})
        else:
            # If not summarizing, set a default summary or use provided summary
            if summary:
                logging.debug("Using provided summary.")
            else:
                summary = "No summary provided."

        # Create info_dict
        info_dict = {
            'title': title,
            'uploader': author,
            'ingestion_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Prepare segments for database
        segments = [{'Text': chunk.get('text', chunk.get('content', ''))} for chunk in chunks]
        logging.debug(f"Prepared segments for database. Number of segments: {len(segments)}")

        # Add to database
        result = add_media_to_database(
            url=file_path,
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=custom_prompt,
            whisper_model="Imported",
            media_type="ebook",
            overwrite=False
        )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("epub_import_duration", processing_time, labels={"file_path": file_path})

        logging.info(f"Ebook '{title}' by {author} imported successfully. Database result: {result}")
        log_counter("epub ingested into the DB successfully", labels={"file_path": file_path})
        return f"Ebook '{title}' by {author} imported successfully. Database result: {result}"

    except Exception as e:
        logging.exception(f"Error importing ebook: {str(e)}")
        log_counter("epub_import_error", labels={"file_path": file_path, "error": str(e)})
        return f"Error importing ebook: {str(e)}"


# FIXME
def process_zip_file(zip_file,
                     title,
                     author,
                     keywords,
                     custom_prompt,
                     system_prompt,
                     summary,
                     auto_summarize,
                     api_name,
                     api_key,
                     chunk_options
                     ):
    """
    Processes a ZIP file containing multiple EPUB files and imports each one.

    Parameters:
        - zip_file (file-like object): The ZIP file to process.
        - title (str): Title prefix for the books.
        - author (str): Author name for the books.
        - keywords (str): Comma-separated keywords.
        - custom_prompt (str): Custom user prompt for summarization.
        - summary (str): Predefined summary (not used in this context).
        - auto_summarize (bool): Whether to auto-summarize the chunks.
        - api_name (str): API name for summarization.
        - api_key (str): API key for summarization.
        - chunk_options (dict): Options for chunking.

    Returns:
        - str: Combined status messages for all EPUB files in the ZIP.
    """
    results = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = zip_file.name if hasattr(zip_file, 'name') else zip_file.path
            logging.info(f"Extracting ZIP file {zip_path} to temporary directory {temp_dir}")
            log_counter("zip_processing_attempt", labels={"zip_path": zip_path})

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            epub_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.epub')]
            log_histogram("epub_files_in_zip", len(epub_files), labels={"zip_path": zip_path})

            for filename in epub_files:
                file_path = os.path.join(temp_dir, filename)
                logging.info(f"Processing EPUB file {filename} from ZIP.")
                result = import_epub(
                    file_path=file_path,
                    title=title,
                    author=author,
                    keywords=keywords,
                    custom_prompt=custom_prompt,
                    summary=summary,
                    auto_summarize=auto_summarize,
                    api_name=api_name,
                    api_key=api_key,
                    chunk_options=chunk_options,
                    custom_chapter_pattern=chunk_options.get('custom_chapter_pattern') if chunk_options else None
                )
                results.append(f"File: {filename} - {result}")

            logging.info("Completed processing all EPUB files in the ZIP.")
            log_counter("zip_processing_success", labels={"zip_path": zip_path})
    except Exception as e:
        logging.exception(f"Error processing ZIP file: {str(e)}")
        log_counter("zip_processing_error", labels={"zip_path": zip_path, "error": str(e)})
        return f"Error processing ZIP file: {str(e)}"

    return "\n".join(results)


def import_html(file_path, title=None, author=None, keywords=None, **kwargs):
    """
    Imports an HTML file and converts it to markdown format.
    """
    try:
        logging.info(f"Importing HTML file from {file_path}")
        h = html2text.HTML2Text()
        h.ignore_links = False

        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        markdown_content = h.handle(html_content)

        # Extract title from HTML if not provided
        if not title:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.string if title_tag else os.path.basename(file_path)

        return process_markdown_content(markdown_content, file_path, title, author, keywords, **kwargs)

    except Exception as e:
        logging.exception(f"Error importing HTML file: {str(e)}")
        raise


def import_xml(file_path, title=None, author=None, keywords=None, **kwargs):
    """
    Imports an XML file and converts it to markdown format.
    """
    try:
        logging.info(f"Importing XML file from {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Convert XML to markdown
        markdown_content = xml_to_markdown(root)

        return process_markdown_content(markdown_content, file_path, title, author, keywords, **kwargs)

    except Exception as e:
        logging.exception(f"Error importing XML file: {str(e)}")
        raise


def import_opml(file_path, title=None, author=None, keywords=None, **kwargs):
    """
    Imports an OPML file and converts it to markdown format.
    """
    try:
        logging.info(f"Importing OPML file from {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract title from OPML if not provided
        if not title:
            title_elem = root.find(".//title")
            title = title_elem.text if title_elem is not None else os.path.basename(file_path)

        # Convert OPML to markdown
        markdown_content = opml_to_markdown(root)

        return process_markdown_content(markdown_content, file_path, title, author, keywords, **kwargs)

    except Exception as e:
        logging.exception(f"Error importing OPML file: {str(e)}")
        raise


def xml_to_markdown(element, level=0):
    """
    Recursively converts XML elements to markdown format.
    """
    markdown = ""

    # Add element name as heading
    if level > 0:
        markdown += f"{'#' * min(level, 6)} {element.tag}\n\n"

    # Add element text if it exists
    if element.text and element.text.strip():
        markdown += f"{element.text.strip()}\n\n"

    # Process child elements
    for child in element:
        markdown += xml_to_markdown(child, level + 1)

    return markdown


def opml_to_markdown(root):
    """
    Converts OPML structure to markdown format.
    """
    markdown = "# Table of Contents\n\n"

    def process_outline(outline, level=0):
        result = ""
        for item in outline.findall("outline"):
            text = item.get("text", "")
            result += f"{'  ' * level}- {text}\n"
            result += process_outline(item, level + 1)
        return result

    body = root.find(".//body")
    if body is not None:
        markdown += process_outline(body)

    return markdown


def process_markdown_content(markdown_content, file_path, title, author, keywords, **kwargs):
    """
    Processes markdown content and adds it to the database.
    """
    info_dict = {
        'title': title or os.path.basename(file_path),
        'uploader': author or "Unknown",
        'ingestion_date': datetime.now().strftime('%Y-%m-%d')
    }

    # Create segments (you may want to adjust the chunking method)
    segments = [{'Text': markdown_content}]

    # Add to database
    result = add_media_to_database(
        url=file_path,
        info_dict=info_dict,
        segments=segments,
        summary=kwargs.get('summary', "No summary provided"),
        keywords=keywords.split(',') if keywords else [],
        custom_prompt_input=kwargs.get('custom_prompt'),
        whisper_model="Imported",
        media_type="document",
        overwrite=False
    )

    return f"Document '{title}' imported successfully. Database result: {result}"


def import_file_handler(files,
                       author,
                       keywords,
                       system_prompt,
                       custom_prompt,
                       auto_summarize,
                       api_name,
                       api_key,
                       max_chunk_size,
                       chunk_overlap,
                       custom_chapter_pattern):
    try:
        if not files:
            return "No files uploaded."

        # Convert single file to list for consistent processing
        if not isinstance(files, list):
            files = [files]

        results = []
        for file in files:
            log_counter("file_import_attempt", labels={"file_name": file.name})

            # Handle max_chunk_size and chunk_overlap
            chunk_size = int(max_chunk_size) if isinstance(max_chunk_size, (str, int)) else 4000
            overlap = int(chunk_overlap) if isinstance(chunk_overlap, (str, int)) else 0

            chunk_options = {
                'method': 'chapter',
                'max_size': chunk_size,
                'overlap': overlap,
                'custom_chapter_pattern': custom_chapter_pattern if custom_chapter_pattern else None
            }

            file_path = file.name
            if not os.path.exists(file_path):
                results.append(f"❌ File not found: {file.name}")
                continue

            start_time = datetime.now()

            # Extract title from filename
            title = os.path.splitext(os.path.basename(file_path))[0]

            if file_path.lower().endswith('.epub'):
                status = import_epub(
                    file_path,
                    title=title,  # Use filename as title
                    author=author,
                    keywords=keywords,
                    custom_prompt=custom_prompt,
                    system_prompt=system_prompt,
                    summary=None,
                    auto_summarize=auto_summarize,
                    api_name=api_name,
                    api_key=api_key,
                    chunk_options=chunk_options,
                    custom_chapter_pattern=custom_chapter_pattern
                )
                log_counter("epub_import_success", labels={"file_name": file.name})
                results.append(f"📚 {file.name}: {status}")

            elif file_path.lower().endswith('.zip'):
                status = process_zip_file(
                    zip_file=file,
                    title=None,  # Let each file use its own name
                    author=author,
                    keywords=keywords,
                    custom_prompt=custom_prompt,
                    system_prompt=system_prompt,
                    summary=None,
                    auto_summarize=auto_summarize,
                    api_name=api_name,
                    api_key=api_key,
                    chunk_options=chunk_options
                )
                log_counter("zip_import_success", labels={"file_name": file.name})
                results.append(f"📦 {file.name}: {status}")
            else:
                results.append(f"❌ Unsupported file type: {file.name}")
                continue

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            log_histogram("file_import_duration", processing_time, labels={"file_name": file.name})

        return "\n\n".join(results)

    except ValueError as ve:
        logging.exception(f"Error parsing input values: {str(ve)}")
        return f"❌ Error: Invalid input for chunk size or overlap. Please enter valid numbers."
    except Exception as e:
        logging.exception(f"Error during file import: {str(e)}")
        return f"❌ Error during import: {str(e)}"



def read_epub(file_path):
    """
    Reads and extracts text from an EPUB file, cleaning up messy spacing.
    """
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
        return text

    except Exception as e:
        logging.exception(f"Error reading EPUB file: {str(e)}")
        raise


def read_epub_filtered(epub_path):
    """
    Reads an EPUB by following the spine, skipping known front matter
    but keeping the Table of Contents (TOC). Returns a cleaned-up
    text string with minimal empty whitespace.

    :param epub_path: Path to the .epub file.
    :return: A cleaned-up text string of the book's content.
    """
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
        return full_text

    except Exception as e:
        logging.exception(f"Failed to parse EPUB: {str(e)}")
        return ""


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


def epub_to_markdown(epub_path):
    """
    Converts an EPUB file to Markdown format, including the table of contents and chapter contents.

    Parameters:
        - epub_path (str): Path to the EPUB file.

    Returns:
        - str: Markdown-formatted content of the EPUB.
    """
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
        return markdown_content

    except Exception as e:
        logging.exception(f"Error converting EPUB to Markdown: {str(e)}")
        raise


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
# End of Function Definitions
#######################################################################################################################
