# PDF_Ingestion_Lib.py
#########################################
# Library to hold functions for ingesting PDF files.#
#
####################
# Function List
#
# 1. convert_pdf_to_markdown(pdf_path)
# 2. ingest_pdf_file(file_path, title=None, author=None, keywords=None):
# 3.
#
#
####################
# Import necessary libraries
import time
from datetime import datetime
import os
import re
import shutil
import tempfile
from typing import Dict, Any, Optional, List, Coroutine

#
# Import External Libs
import pymupdf
import pymupdf4llm
from docling.document_converter import DocumentConverter
#
# Import Local
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import perform_summarization
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process
from tldw_Server_API.app.core.Utils.Utils import logging
#
# Constants
MAX_FILE_SIZE_MB = 50
CONVERSION_TIMEOUT_SECONDS = 300
#
#######################################################################################################################
# Function Definitions
#

def extract_text_and_format_from_pdf(pdf_path):
    """
    Extract text from a PDF file and convert it to Markdown, preserving formatting.
    """
    try:
        log_counter("pdf_text_extraction_attempt", labels={"file_path": pdf_path})
        start_time = datetime.now()

        markdown_text = ""
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                markdown_text += f"## Page {page_num}\n\n"
                blocks = page.get_text("dict")["blocks"]
                current_paragraph = ""
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"]
                                font_size = span["size"]
                                font_flags = span["flags"]

                                # Apply formatting based on font size and flags
                                if font_size > 20:
                                    text = f"# {text}"
                                elif font_size > 16:
                                    text = f"## {text}"
                                elif font_size > 14:
                                    text = f"### {text}"

                                if font_flags & 2 ** 0:  # Bold
                                    text = f"**{text}**"
                                if font_flags & 2 ** 1:  # Italic
                                    text = f"*{text}*"

                                line_text += text + " "

                            # Remove hyphens at the end of lines
                            line_text = line_text.rstrip()
                            if line_text.endswith('-'):
                                line_text = line_text[:-1]
                            else:
                                line_text += " "

                            current_paragraph += line_text

                        # End of block, add paragraph
                        if current_paragraph:
                            # Remove extra spaces
                            current_paragraph = re.sub(r'\s+', ' ', current_paragraph).strip()
                            markdown_text += current_paragraph + "\n\n"
                            current_paragraph = ""
                    elif block["type"] == 1:  # Image block
                        markdown_text += "[Image]\n\n"
                markdown_text += "\n---\n\n"  # Page separator

        # Clean up hyphenated words
        markdown_text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', markdown_text)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("pdf_text_extraction_duration", processing_time, labels={"file_path": pdf_path})
        log_counter("pdf_text_extraction_success", labels={"file_path": pdf_path})

        return markdown_text
    except Exception as e:
        logging.error(f"Error extracting text and formatting from PDF: {str(e)}")
        log_counter("pdf_text_extraction_error", labels={"file_path": pdf_path, "error": str(e)})
        raise


def pymupdf4llm_parse_pdf(pdf_path):
    """
    Extract text from a PDF file and convert it to Markdown, preserving formatting.
    """
    try:
        log_counter("pdf_text_extraction_attempt", labels={"file_path": pdf_path})
        start_time = datetime.now()

        markdown_text = pymupdf4llm.to_markdown(pdf_path)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("pdf_text_extraction_duration", processing_time, labels={"file_path": pdf_path})
        log_counter("pdf_text_extraction_success", labels={"file_path": pdf_path})

        return markdown_text
    except Exception as e:
        logging.error(f"Error extracting text and formatting from PDF: {str(e)}")
        log_counter("pdf_text_extraction_error", labels={"file_path": pdf_path, "error": str(e)})
        raise


def extract_metadata_from_pdf(pdf_path):
    """
    Extract metadata from a PDF file using PyMuPDF.
    """
    try:
        log_counter("pdf_metadata_extraction_attempt", labels={"file_path": pdf_path})
        with pymupdf.open(pdf_path) as doc:
            metadata = doc.metadata
        log_counter("pdf_metadata_extraction_success", labels={"file_path": pdf_path})
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata from PDF: {str(e)}")
        log_counter("pdf_metadata_extraction_error", labels={"file_path": pdf_path, "error": str(e)})
        return {}


def process_and_ingest_pdf(file, title, author, keywords, parser='pymupdf4llm'):
    if file is None:
        log_counter("pdf_ingestion_error", labels={"error": "No file uploaded"})
        return "Please select a PDF file to upload."

    try:
        log_counter("pdf_ingestion_attempt", labels={"file_name": file.name})
        start_time = datetime.now()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(file.name, temp_path)

            if parser == 'pymupdf':
                # Extract text and convert to Markdown
                markdown_text = extract_text_and_format_from_pdf(temp_path)

            elif parser == 'pymupdf4llm':
                # Extract text and convert to Markdown
                markdown_text = pymupdf4llm_parse_pdf(temp_path)

            elif parser == 'docling':
                # Extract text and convert to Markdown using Docling
                converter = DocumentConverter()
                parsed_pdf = converter.convert(temp_path)
                markdown_text = parsed_pdf.document.export_to_markdown()

            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            if not title:
                title = metadata.get('title', os.path.splitext(os.path.basename(file.name))[0])
            if not author:
                author = metadata.get('author', 'Unknown')

            # If keywords are not provided, use a default keyword
            if not keywords:
                keywords = 'pdf_file,markdown_converted'
            else:
                keywords = f'pdf_file,markdown_converted,{keywords}'

            # Add metadata-based keywords
            if 'subject' in metadata:
                keywords += f",{metadata['subject']}"

            # Add the PDF content to the database
            add_media_with_keywords(
                url=file.name,
                title=title,
                media_type='document',
                content=markdown_text,
                keywords=keywords,
                prompt='No prompt for PDF files',
                summary='No summary for PDF files',
                transcription_model='None',
                author=author,
                ingestion_date=datetime.now().strftime('%Y-%m-%d')
            )

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("pdf_ingestion_duration", processing_time, labels={"file_name": file.name})
        log_counter("pdf_ingestion_success", labels={"file_name": file.name})

        return f"PDF file '{title}' by {author} ingested successfully and converted to Markdown."
    except Exception as e:
        logging.error(f"Error ingesting PDF file: {str(e)}")
        log_counter("pdf_ingestion_error", labels={"file_name": file.name, "error": str(e)})
        return f"Error ingesting PDF file: {str(e)}"


def process_and_cleanup_pdf(file, title, author, keywords, parser='pymupdf4llm'):
    if file is None:
        log_counter("pdf_processing_error", labels={"error": "No file uploaded"})
        return "No file uploaded. Please upload a PDF file."

    try:
        log_counter("pdf_processing_attempt", labels={"file_name": file.name})
        start_time = datetime.now()

        result = process_and_ingest_pdf(file, title, author, keywords, parser)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("pdf_processing_duration", processing_time, labels={"file_name": file.name})
        log_counter("pdf_processing_success", labels={"file_name": file.name})

        return result
    except Exception as e:
        logging.error(f"Error in processing and cleanup: {str(e)}")
        log_counter("pdf_processing_error", labels={"file_name": file.name, "error": str(e)})
        return f"Error: {str(e)}"


async def process_pdf_task(
    file_bytes: bytes,
    filename: str,
    parser: str = "pymupdf4llm",
    custom_prompt: Optional[str] = None,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_summarize: bool = False,
    keywords: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    perform_chunking: bool = True,
    chunk_method: str = "sentences",
    max_chunk_size: int = 500,
    chunk_overlap: int = 200
) -> dict[str, Any] | None:
    """
    Process a single PDF (provided as file bytes) and return its text + optional summary.

    :param file_bytes: The in-memory PDF file content.
    :param filename:    A string filename for reference.
    :param parser:      Which parser to use (e.g. 'pymupdf4llm', 'docling', 'pymupdf', etc.).
    :param custom_prompt:  Optional custom prompt for summarization.
    :param api_name:    Which LLM API to use for summarization (e.g. 'openai'). If not set, no summarization.
    :param api_key:     API key for the LLM.
    :param auto_summarize: If True, attempt to summarize the extracted text.
    :param keywords:    Optional list of keywords.
    :param system_prompt: Optional system-level prompt for the LLM.
    :param perform_chunking: If True, chunk the text before summarizing.
    :param chunk_method:  One of 'sentences', 'words', etc.
    :param max_chunk_size: Max chunk size for chunking.
    :param chunk_overlap: Overlap size for chunking.
    :return: A dictionary with:
             - text_content: the full extracted text
             - summary: a summary if auto_summarize=True & api_name is set
             - title: (from PDF metadata or inferred from filename)
             - author: (from PDF metadata or 'Unknown')
             - parser_used: the parser name used
             - keywords: the input keywords (if any)
    """

    start_time = time.time()
    log_counter("pdf_processing_attempt", labels={"file_name": filename})

    # Prepare a result dict
    result_data: Dict[str, Any] = {
        "filename": filename,
        "parser_used": parser,
        "text_content": "",
        "summary": "",
        "title": None,   # will set below from metadata or user param
        "author": None,  # same as above
        "keywords": keywords or [],
    }

    try:
        # 1) Write the file bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            local_path = tmp.name

        # 2) Extract text with chosen parser
        try:
            if parser == "pymupdf4llm":
                text_content = pymupdf4llm_parse_pdf(local_path)
            elif parser == "pymupdf":
                from . import extract_text_and_format_from_pdf
                text_content = extract_text_and_format_from_pdf(local_path)
            else:
                # If you have a docling fallback or something else
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                parsed_obj = converter.convert(local_path)
                text_content = parsed_obj.document.export_to_markdown()

            result_data["text_content"] = text_content

            # 3) Extract PDF metadata to guess a title/author
            metadata = extract_metadata_from_pdf(local_path)
            # If metadata has e.g. metadata["title"], set that:
            result_data["title"] = metadata.get("title") or os.path.splitext(filename)[0]
            result_data["author"] = metadata.get("author", "Unknown")

            # 4) If auto_summarize + we have an LLM, do chunking & summarization
            if auto_summarize and api_name and api_name.lower() != "none":
                if perform_chunking:
                    chunk_opts = {
                        "method": chunk_method,
                        "max_size": max_chunk_size,
                        "overlap": chunk_overlap
                    }
                    chunked_texts = improved_chunking_process(text_content, chunk_opts)
                    if not chunked_texts:
                        # fallback to single pass
                        sum_result = perform_summarization(api_name, text_content, custom_prompt, api_key, system_message=system_prompt)
                        result_data["summary"] = sum_result or ""
                    else:
                        chunk_summaries = []
                        for c in chunked_texts:
                            sum_result = perform_summarization(api_name, c["text"], custom_prompt, api_key, system_message=system_prompt)
                            if sum_result:
                                chunk_summaries.append(sum_result)
                        if chunk_summaries:
                            combined_summary = "\n\n".join(chunk_summaries)
                            result_data["summary"] = combined_summary
                else:
                    # Single pass summarization
                    sum_result = perform_summarization(api_name, text_content, custom_prompt, api_key, system_message=system_prompt)
                    result_data["summary"] = sum_result or ""

        finally:
            # Cleanup the temp file
            if os.path.exists(local_path):
                os.remove(local_path)

        # Mark success in logs
        end_time = time.time()
        processing_time = end_time - start_time
        log_histogram("pdf_processing_duration", processing_time, labels={"file_name": filename})
        log_counter("pdf_processing_success", labels={"file_name": filename})

        return result_data

    except Exception as e:
        log_counter("pdf_processing_error", labels={"file_name": filename, "error": str(e)})
        logging.error(f"Error in process_pdf_task for file {filename}: {str(e)}", exc_info=True)
        raise

#
# End of PDF_Ingestion_Lib.py
#######################################################################################################################
