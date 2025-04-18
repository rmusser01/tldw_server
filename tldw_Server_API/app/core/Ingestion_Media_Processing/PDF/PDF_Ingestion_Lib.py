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
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

#
# Import External Libs
import pymupdf
import pymupdf4llm
#
# Import Local
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


def docling_parse_pdf(pdf_path: str):
    """
    Extract text using the Docling library (if available).
    """
    parser_name = "docling"
    DOCLING_AVAILABLE = False
    try:
        from docling.document_converter import DocumentConverter
    except:
        DOCLING_AVAILABLE = False
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling library is not installed.")
    try:
        log_counter("pdf_text_extraction_attempt", labels={"file_path": pdf_path, "parser": parser_name})
        start_time = datetime.now()

        converter = DocumentConverter()
        parsed_pdf = converter.convert(pdf_path)
        markdown_text = parsed_pdf.document.export_to_markdown() # Or other formats if needed

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("pdf_text_extraction_duration", processing_time, labels={"file_path": pdf_path, "parser": parser_name})
        log_counter("pdf_text_extraction_success", labels={"file_path": pdf_path, "parser": parser_name})
        return markdown_text

    except Exception as e:
        logging.error(f"Error extracting text ({parser_name}) from PDF {pdf_path}: {str(e)}", exc_info=True)
        log_counter("pdf_text_extraction_error", labels={"file_path": pdf_path, "parser": parser_name, "error": str(e)})
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


def process_pdf(
    file_input: Union[str, bytes, Path], # Can be path, bytes, or Path object
    filename: str, # Original filename for reference and metadata fallback
    parser: str = "pymupdf4llm",
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    perform_chunking: bool = True,
    chunk_options: Optional[Dict[str, Any]] = None,
    perform_analysis: bool = False, # Renamed from auto_summarize
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    summarize_recursively: bool = False,
    write_to_temp_file: bool = False # Control if bytes should be written to disk
) -> dict[str, Any] | None:
    """
    Processes a single PDF (from path or bytes): extracts text & metadata, chunks, summarizes.
    Returns a dictionary with processed data, status, and errors. *No DB interaction.*

    Parameters:
      - file_input (Union[str, bytes, Path]): Path to the PDF file or bytes content.
      - filename (str): Original filename for reference.
      - parser (str): Parser to use ('pymupdf4llm', 'pymupdf', 'docling').
      - title_override (str, optional): User-provided title.
      - author_override (str, optional): User-provided author.
      - keywords (List[str], optional): Keywords.
      - perform_chunking (bool): Whether to chunk the content.
      - chunk_options (dict, optional): Options for chunking.
      - perform_analysis (bool): Whether to perform summarization.
      - api_name (str, optional): API name for summarization.
      - api_key (str, optional): API key for summarization.
      - custom_prompt (str, optional): Custom user prompt for summarization.
      - system_prompt (str, optional): System prompt for summarization.
      - summarize_recursively (bool): Whether to perform recursive summarization.
      - write_to_temp_file (bool): If True and input is bytes, write to a temp file
                                  (needed for parsers that only accept paths).

    Returns:
        - Dict[str, Any]: Dictionary containing processing results:
            {
                "status": "Success" | "Error" | "Warning",
                "input_ref": str (filename),
                "media_type": "pdf",
                "parser_used": str,
                "text_content": Optional[str],
                "metadata": Optional[Dict], # {'title': str, 'author': str, 'raw': dict}
                "chunks": Optional[List[Dict]],
                "summary": Optional[str],
                "keywords": Optional[List[str]],
                "error": Optional[str],
                "warnings": Optional[List[str]]
            }
    """
    start_time = time.time()
    # Initialize the result dictionary structure
    result: Dict[str, Any] = {
        "status": "Pending",
        "input_ref": filename,
        "media_type": "pdf",
        "parser_used": parser,
        "text_content": None,
        "metadata": None,
        "chunks": None,
        "summary": None,
        "keywords": keywords or [],
        "error": None,
        "warnings": None,
    }
    log_counter("pdf_processing_attempt", labels={"file_name": filename, "parser": parser})

    temp_pdf_path: Optional[str] = None
    input_for_processing: Union[str, bytes] = file_input
    input_for_metadata: Union[str, bytes] = file_input

    try:
        # --- Step 0: Handle Input Type and Temporary File ---
        # Decide if a temporary file is needed based on input type and parser requirements
        if isinstance(file_input, bytes):
            needs_path = parser in ['pymupdf', 'docling'] # Parsers requiring a file path
            if write_to_temp_file or needs_path:
                # Create a temporary file ONLY if needed
                # Use a try-finally block or context manager for robust cleanup
                try:
                    # Create tempfile.NamedTemporaryFile within the main try block
                    # to ensure its lifecycle is managed correctly.
                    # delete=False is crucial here so the file persists after the 'with' block
                    # if we need the path later. Cleanup happens in the finally block.
                    _temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                    _temp_file.write(file_input)
                    temp_pdf_path = _temp_file.name # Store the path
                    _temp_file.close() # Close the file handle immediately after writing
                    logging.debug(f"Input bytes written to temporary file: {temp_pdf_path}")
                    input_for_processing = temp_pdf_path
                    input_for_metadata = temp_pdf_path # Metadata extraction also uses path now
                except Exception as temp_err:
                    logging.error(f"Failed to create or write temporary file: {temp_err}", exc_info=True)
                    raise IOError(f"Failed to handle temporary file: {temp_err}") from temp_err
            else:
                # Parser can handle bytes (e.g., pymupdf4llm)
                input_for_processing = file_input
                input_for_metadata = file_input # Metadata can also handle bytes
        elif isinstance(file_input, Path):
             input_for_processing = str(file_input)
             input_for_metadata = str(file_input)
        elif isinstance(file_input, str):
             # Ensure the file exists if a path string is provided
             if not os.path.exists(file_input):
                 raise FileNotFoundError(f"Input file path does not exist: {file_input}")
             input_for_processing = file_input
             input_for_metadata = file_input
        else:
            raise TypeError(f"Unsupported file_input type: {type(file_input)}")

        # --- Step 1: Extract Text ---
        text_content = None
        try:
            logging.info(f"Attempting text extraction for {filename} using parser: {parser}")
            if parser == "pymupdf4llm":
                # This parser can handle bytes or path directly
                text_content = pymupdf4llm_parse_pdf(input_for_processing)
            elif parser == "pymupdf":
                 if not isinstance(input_for_processing, str):
                      raise ValueError("Internal Error: pymupdf parser needs a file path, but received bytes.")
                 text_content = extract_text_and_format_from_pdf(input_for_processing)
            elif parser == "docling":
                DOCLING_AVAILABLE = False
                try:
                    from docling.document_converter import DocumentConverter
                    DOCLING_AVAILABLE = True
                except ImportError:
                    DOCLING_AVAILABLE = False
                if not DOCLING_AVAILABLE:
                    raise ImportError("Docling parser selected, but library is not installed.")
                if not isinstance(input_for_processing, str):
                    raise ValueError("Internal Error: docling parser needs a file path, but received bytes.")
                text_content = docling_parse_pdf(input_for_processing)
            else:
                # This case should ideally be caught by Pydantic validation in the endpoint
                logging.warning(f"Unsupported PDF parser specified: {parser}. Attempting fallback to pymupdf4llm.")
                result["warnings"] = (result["warnings"] or []) + [f"Unsupported parser '{parser}', fallback to 'pymupdf4llm'"]
                result["parser_used"] = "pymupdf4llm" # Update the parser used in the result
                text_content = pymupdf4llm_parse_pdf(input_for_processing) # Try fallback

            result["text_content"] = text_content
            logging.info(f"Text extracted successfully for {filename} using {result['parser_used']}.")
        except Exception as parse_err:
             # If parsing fails, log the error and add a warning. Continue to metadata extraction.
             logging.error(f"Text extraction failed for {filename} using {parser}: {parse_err}", exc_info=True)
             result["warnings"] = (result["warnings"] or []) + [f"Text extraction failed ({parser}): {str(parse_err)}"]
             # Do not set status to Error yet, metadata might still work.

        # --- Step 2: Extract Metadata ---
        # Metadata extraction should work even if text extraction failed.
        try:
            logging.info(f"Attempting metadata extraction for {filename}.")
            raw_metadata = extract_metadata_from_pdf(input_for_metadata)
            logging.info(f"Metadata extracted for {filename}.")

            # Add subject and keywords from metadata to the provided keywords list
            pdf_keywords_str = raw_metadata.get('keywords', '')
            pdf_subject = raw_metadata.get('subject')
            # Use sets for efficient merging and deduplication
            combined_keywords = set(k.strip() for k in (keywords or []) if k.strip()) # Start with input keywords
            if pdf_keywords_str and isinstance(pdf_keywords_str, str):
                combined_keywords.update(k.strip() for k in pdf_keywords_str.split(',') if k.strip())
            if pdf_subject and isinstance(pdf_subject, str) and pdf_subject.strip():
                 combined_keywords.add(pdf_subject.strip())
            result["keywords"] = sorted(list(combined_keywords)) # Store unique, sorted keywords

            # Determine final title/author using overrides, then metadata, then filename
            final_title = title_override or raw_metadata.get('title') or Path(filename).stem
            final_author = author_override or raw_metadata.get('author') or "Unknown"
            result["metadata"] = {
                "title": final_title,
                "author": final_author,
                "creationDate": raw_metadata.get('creationDate'), # Include other useful metadata
                "modDate": raw_metadata.get('modDate'),
                "producer": raw_metadata.get('producer'),
                "creator": raw_metadata.get('creator'),
                "page_count": raw_metadata.get('page_count', 0), # Get page count if available from PyMuPDF
                "raw": raw_metadata # Store the raw extracted metadata dictionary
            }
            logging.debug(f"Final metadata for {filename} - Title: {final_title}, Author: {final_author}")

        except Exception as meta_err:
             logging.error(f"Metadata extraction failed for {filename}: {meta_err}", exc_info=True)
             result["warnings"] = (result["warnings"] or []) + [f"Metadata extraction failed: {str(meta_err)}"]
             # Still set default metadata structure if extraction fails completely
             result["metadata"] = {
                 "title": title_override or Path(filename).stem,
                 "author": author_override or "Unknown",
                 "raw": {"error": f"Metadata extraction failed: {str(meta_err)}"}
             }


        # --- Step 3: Chunking ---
        processed_chunks = None
        # Only proceed if text extraction was successful
        if text_content and perform_chunking:
            if chunk_options is None:
                # Provide sensible defaults if none are passed
                chunk_options = {'method': 'recursive', 'max_size': 500, 'overlap': 100}
            # Ensure a method is set, default to 'recursive' if missing
            chunk_options.setdefault('method', 'recursive')

            logging.info(f"Attempting chunking for {filename} with options: {chunk_options}")
            try:
                # Call the chunking function (ensure it handles potential errors)
                processed_chunks = improved_chunking_process(text_content, chunk_options)

                if not processed_chunks:
                     logging.warning(f"Chunking produced no chunks for {filename}. Using full text as one chunk.")
                     # Create a single chunk containing the entire text
                     processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0, 'start_index': 0, 'end_index': len(text_content)}}]
                     result["warnings"] = (result["warnings"] or []) + ["Chunking yielded no results; using full text."]
                else:
                     logging.info(f"Chunking successful for {filename}. Total chunks created: {len(processed_chunks)}")
                     log_histogram("pdf_chunks_created", len(processed_chunks), labels={"file_name": filename})

                result["chunks"] = processed_chunks # Store the list of chunks

            except Exception as chunk_err:
                 logging.error(f"Chunking failed for {filename}: {chunk_err}", exc_info=True)
                 result["warnings"] = (result["warnings"] or []) + [f"Chunking failed: {str(chunk_err)}"]
                 # If chunking fails, we might still want to proceed with summarization on the full text
                 # Create a single chunk containing the whole text to allow summarization attempt
                 processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {chunk_err}"}}]
                 result["chunks"] = processed_chunks # Store the single chunk with error info

        elif text_content:
             # If not chunking, but text exists, create a single chunk for consistency
             processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks
             logging.info(f"Chunking disabled for {filename}. Using full text as one chunk.")
        else:
             # If text extraction failed, chunking cannot proceed
             logging.warning(f"Chunking skipped for {filename}: Text content is missing.")


        # --- Step 4: Summarization / Analysis ---
        final_summary = None
        # Check if summarization should be performed: analysis enabled, API configured, and chunks exist
        if perform_analysis and api_name and api_key and processed_chunks:
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks of {filename} using API: {api_name}.")
            log_counter("pdf_summarization_attempt", value=len(processed_chunks), labels={"file_name": filename, "api_name": api_name})

            chunk_summaries = []  # Store summaries of individual chunks
            summarized_chunks_for_result = [] # Store chunk data including the generated summary

            # Iterate through each chunk generated earlier
            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '') # Get the text content of the chunk
                chunk_metadata: Dict[str, Any] = chunk.get('metadata', {}) # Get existing metadata

                # Only summarize if the chunk has actual text content
                if chunk_text:
                    try:
                        # Call the external summarization library function
                        summary_text = perform_summarization(
                            api_name=api_name,
                            input_data=chunk_text,
                            custom_prompt_input=custom_prompt, # User's custom prompt, if any
                            api_key=api_key,
                            recursive_summarization=False, # Summarize this single chunk first
                            temp=None, # Optional temperature parameter
                            system_message=system_prompt # Optional system prompt
                        )

                        # Check if the summarization returned a valid, non-empty string
                        if summary_text and isinstance(summary_text, str) and summary_text.strip():
                            chunk_summaries.append(summary_text)
                            # Add the generated summary to the chunk's metadata
                            chunk_metadata['summary'] = summary_text
                            logging.debug(f"Summarized chunk {i+1}/{len(processed_chunks)} for {filename}.")
                        else:
                            # Summarization returned empty or invalid result
                            chunk_metadata['summary'] = None # Indicate no summary available
                            logging.debug(f"Summarization yielded empty result for chunk {i+1} of {filename}.")

                    except Exception as summ_err:
                        # Handle errors during the API call or summarization process
                        logging.warning(f"Summarization failed for chunk {i+1} of {filename}: {summ_err}", exc_info=True)
                        # Store error information in the chunk's metadata
                        chunk_metadata['summary'] = f"[Summarization Error: {str(summ_err)}]"
                        # Add a warning to the overall result
                        result["warnings"] = (result["warnings"] or []) + [f"Summarization failed for chunk {i+1}: {str(summ_err)}"]
                else:
                    # Chunk had no text to summarize
                    chunk_metadata['summary'] = None
                    logging.debug(f"Skipping summarization for empty chunk {i+1} of {filename}.")

                # Update the chunk with potentially modified metadata
                chunk['metadata'] = chunk_metadata
                # Add the chunk (with or without summary metadata) to the list for the final result
                summarized_chunks_for_result.append(chunk)

            # Update the main result dictionary with the chunks containing summary metadata
            result["chunks"] = summarized_chunks_for_result

            # --- Combine chunk summaries (optional recursive step) ---
            if chunk_summaries: # Proceed only if at least one chunk was successfully summarized
                if summarize_recursively and len(chunk_summaries) > 1:
                    # If recursive summarization is enabled and there are multiple chunk summaries
                    logging.info(f"Performing recursive summarization on {len(chunk_summaries)} chunk summaries for {filename}.")
                    # Join the individual chunk summaries into one large text block
                    combined_summaries_text = "\n\n---\n\n".join(chunk_summaries) # Use a clear separator

                    try:
                        # Call perform_summarization again on the combined text
                        final_summary = perform_summarization(
                            api_name=api_name,
                            input_data=combined_summaries_text,
                            # Use the original custom prompt, or a default recursive prompt if none provided
                            custom_prompt_input=custom_prompt or "Provide a concise overall summary of the preceding text sections.",
                            api_key=api_key,
                            recursive_summarization=False, # This is the final summarization pass
                            temp=None,
                            system_message=system_prompt
                        )
                        if not final_summary or not final_summary.strip():
                             logging.warning(f"Recursive summarization for {filename} yielded empty result. Falling back to joined summaries.")
                             final_summary = combined_summaries_text # Fallback
                             result["warnings"] = (result["warnings"] or []) + ["Recursive summarization yielded empty result."]
                        else:
                             log_counter("pdf_recursive_summarization_success", labels={"file_name": filename})

                    except Exception as rec_summ_err:
                        # Handle errors during the recursive summarization step
                        logging.error(f"Recursive summarization failed for {filename}: {rec_summ_err}", exc_info=True)
                        # Fallback: Use the joined chunk summaries as the final summary, but mark the error
                        final_summary = f"[Recursive Summarization Error: {str(rec_summ_err)}]\n\n" + combined_summaries_text
                        result["warnings"] = (result["warnings"] or []) + [f"Recursive summarization failed: {str(rec_summ_err)}"]
                        log_counter("pdf_recursive_summarization_error", labels={"file_name": filename, "error": str(rec_summ_err)})

                else:
                    # Not recursive, or only one chunk summary: simply join them
                    final_summary = "\n\n---\n\n".join(chunk_summaries)
                    if len(chunk_summaries) > 1 :
                         logging.info(f"Combined {len(chunk_summaries)} chunk summaries for {filename} (non-recursive).")
                    else:
                         logging.info(f"Using single chunk summary as final summary for {filename}.")


            # Store the final generated summary (or None if none was generated)
            result["summary"] = final_summary
            log_counter("pdf_chunks_summarized", value=len(chunk_summaries), labels={"file_name": filename})
            logging.info(f"Summarization processing completed for {filename}.")

        # --- Log reasons if summarization was skipped ---
        elif not perform_analysis:
             logging.info(f"Summarization disabled by 'perform_analysis=False' for {filename}.")
        elif not api_name or not api_key:
             logging.warning(f"Summarization skipped for {filename}: API name or key not provided.")
        elif not processed_chunks:
             # This case covers both chunking disabled and chunking failed/yielded no results
             logging.warning(f"Summarization skipped for {filename}: No processable chunks available.")
        else:
             # Fallback case, should ideally not be reached
             logging.warning(f"Summarization skipped for {filename} due to an unknown condition.")


        # --- Step 5: Determine Final Status ---
        # Check if any errors occurred during the critical steps (parsing, file handling)
        # If text extraction failed, it's likely an Error, otherwise Warning for lesser issues.
        if not text_content and any("Text extraction failed" in w for w in (result["warnings"] or [])):
            result["status"] = "Error"
            result["error"] = result["error"] or "Text extraction failed." # Assign primary error
        elif result["warnings"]:
             # If there were warnings but text was extracted, status is Warning
             result["status"] = "Warning"
        else:
             # No errors or warnings encountered
             result["status"] = "Success"

    # --- Main Exception Handler ---
    except FileNotFoundError as fnf_err:
        logging.error(f"File not found error for {filename}: {fnf_err}", exc_info=True)
        result["status"] = "Error"
        result["error"] = str(fnf_err)
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": "FileNotFoundError"})
        # --- MODIFICATION START ---
        # Catch RuntimeError for password issues and other load problems,
        # FileDataError for specific corruption types.
        # Added EmptyFileError for robustness.

    except (RuntimeError, pymupdf.FileDataError, pymupdf.EmptyFileError) as pdf_lib_err:
        # --- MODIFICATION END ---
        # Check the message specifically for password errors if needed for logging differentiation
        err_msg = str(pdf_lib_err)
        # Distinguish error types for logging and user messages
        if "password" in err_msg.lower():
            log_msg = f"PDF password error for {filename}: {err_msg}"
            err_type_label = "PasswordError"  # Specific label for metrics
            result["error"] = f"PDF Error: Password required or invalid."  # User-friendly message
        elif isinstance(pdf_lib_err, pymupdf.EmptyFileError):
            log_msg = f"PDF empty file error for {filename}: {err_msg}"
            err_type_label = "EmptyFileError"
            result["error"] = f"PDF Error: Input file is empty."
        elif isinstance(pdf_lib_err, pymupdf.FileDataError):
            log_msg = f"PDF file data error for {filename}: {err_msg}"
            err_type_label = "FileDataError"
            result["error"] = f"PDF Error: Corrupted or invalid file data."
        else:  # General RuntimeError or other caught types
            log_msg = f"PDF library runtime error for {filename}: {err_msg}"
            err_type_label = type(pdf_lib_err).__name__  # Use 'RuntimeError' usually
            result["error"] = f"PDF Error: {err_msg}"  # General PDF error

        logging.error(log_msg, exc_info=True)
        result["status"] = "Error"
        # Use the determined err_type_label for consistent metrics
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": err_type_label})

    except Exception as e:
        # Catch any other unexpected exceptions during the process
        logging.error(f"Unexpected error processing PDF {filename}: {str(e)}", exc_info=True)
        result["status"] = "Error"
        # Ensure error field is populated, even if warnings occurred before the fatal error
        result["error"] = result["error"] or f"Unexpected error: {str(e)}"
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": type(e).__name__})

    # --- Finally Block: Cleanup ---
    finally:
        # Cleanup the temporary file ONLY if it was created
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logging.debug(f"Removed temporary PDF file: {temp_pdf_path}")
            except OSError as rm_err:
                # Log a warning if cleanup fails, but don't let it prevent returning results
                logging.warning(f"Could not remove temporary PDF file {temp_pdf_path}: {rm_err}")
                result["warnings"] = (result["warnings"] or []) + [f"Failed to cleanup temp file: {rm_err}"]
                # If status was Success, downgrade to Warning due to cleanup issue
                if result["status"] == "Success":
                    result["status"] = "Warning"


    # --- Final Logging and Return ---
    end_time = time.time()
    processing_time = end_time - start_time
    log_histogram("pdf_processing_duration", processing_time, labels={"file_name": filename, "parser": result['parser_used'], "status": result["status"]})
    # Log success or final error/warning status
    if result["status"] == "Success":
        log_counter("pdf_processing_success", labels={"file_name": filename, "parser": result['parser_used']})
        logging.info(f"Successfully processed PDF: {filename} (Parser: {result['parser_used']})")
    elif result["status"] == "Warning":
        logging.warning(f"Processed PDF with warnings: {filename} (Parser: {result['parser_used']}). Warnings: {result['warnings']}")
    else: # Error status
        logging.error(f"Failed to process PDF: {filename} (Parser: {result['parser_used']}). Error: {result['error']}")


    # Ensure warnings list is None if empty
    if isinstance(result.get("warnings"), list) and not result["warnings"]:
        result["warnings"] = None

    return result


async def process_pdf_task(
    file_bytes: bytes,
    filename: str,
    parser: str = "pymupdf4llm",
    title_override: Optional[str] = None, # Added
    author_override: Optional[str] = None, # Added
    keywords: Optional[List[str]] = None,
    perform_chunking: bool = True,
    chunk_method: str = "recursive", # Changed default to recursive for consistency
    max_chunk_size: int = 500,
    chunk_overlap: int = 100, # Reduced default overlap
    perform_analysis: bool = False, # Renamed
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    summarize_recursively: bool = False # Added
    # Removed chunk_options dict param, pass individual params now
) -> Dict[str, Any]:
    """
    Async wrapper task to process a single PDF (provided as bytes)
    using the core `process_pdf` function. Returns its result dictionary.
    *No DB interaction.*
    """
    try:
        logging.info(f"process_pdf_task started for {filename} using {parser}")

        # Prepare chunk options dictionary for process_pdf
        chunk_options_dict = None
        if perform_chunking:
            chunk_options_dict = {
                'method': chunk_method,
                'max_size': max_chunk_size,
                'overlap': chunk_overlap
            }

        # Call the synchronous core processing function
        # We don't need run_in_executor here as the caller endpoint already handles it.
        # If this task itself were long-running IO/CPU, it would need it.
        # The internal process_pdf handles logging and timing.
        result_dict = process_pdf(
            file_input=file_bytes,
            filename=filename,
            parser=parser,
            title_override=title_override,
            author_override=author_override,
            keywords=keywords,
            perform_chunking=perform_chunking,
            chunk_options=chunk_options_dict,
            perform_analysis=perform_analysis,
            api_name=api_name,
            api_key=api_key,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            summarize_recursively=summarize_recursively,
            # process_pdf decides whether to write temp file based on parser
        )

        logging.info(f"process_pdf_task completed for {filename} with status: {result_dict.get('status')}")
        return result_dict

    except Exception as e:
        # Catch errors specifically from the task wrapper itself (e.g., setup)
        logging.error(f"Error within process_pdf_task for {filename}: {str(e)}", exc_info=True)
        # Return a standard error dictionary
        return {
            "status": "Error",
            "input_ref": filename,
            "media_type": "pdf",
            "parser_used": parser,
            "error": f"Task-level error: {str(e)}",
            "text_content": None, "metadata": None, "chunks": None, "summary": None,
            "keywords": keywords or [], "warnings": None
        }

#
# End of PDF_Ingestion_Lib.py
#######################################################################################################################
