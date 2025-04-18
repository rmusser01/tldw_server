# Plaintext_Files.py
# Description: This file contains functions for reading and writing plaintext files.
#
# Import necessary libraries
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List

#
# External Imports
from docx2txt import docx2txt
from pypandoc import convert_file

from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_to_database
#
# Local Imports
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import perform_summarization
from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process
from tldw_Server_API.app.core.Utils.Utils import logging
#
#######################################################################################################################
#
# Function Definitions

def _read_text(path: Path) -> str:
    """Read file content as UTF‑8, fallback to latin‑1 if needed."""
    try:
        return path.read_text(encoding="utf-8") # Corrected encoding name
    except UnicodeDecodeError:
        logging.warning(f"UTF-8 decode failed for {path}, trying latin-1.")
        return path.read_text(encoding="latin-1") # Corrected encoding name
    except Exception as e:
        logging.error(f"Failed to read text file {path}: {e}")
        raise # Re-raise other read errors


def convert_to_plain_text(file_path: Path) -> str:
    """
    Converts various document formats (.docx, .rtf) to plain text.
    Returns original content for .txt/.md.
    Raises ValueError for unsupported types.
    """
    extension = file_path.suffix.lower()
    content = ""
    try:
        if extension == '.docx':
            content = docx2txt.process(str(file_path))
            log_counter("docx_conversion_success", labels={"file_path": str(file_path)})
        elif extension == '.rtf':
            # Use pandoc via pypandoc for RTF
            # Requires pandoc binary to be installed on the system
            content = convert_file(str(file_path), 'plain', format='rtf')
            log_counter("rtf_conversion_success", labels={"file_path": str(file_path)})
        elif extension in ['.txt', '.md']:
            content = _read_text(file_path) # Use robust reader
        else:
            raise ValueError(f"Unsupported file type for plain text conversion: {extension}")
    except ImportError as ie:
         # Handle missing optional dependency like pypandoc
         logging.error(f"Missing dependency for {extension} conversion: {ie}")
         raise ValueError(f"Cannot convert {extension}: Missing library ({ie})") from ie
    except Exception as e:
        logging.error(f"Error converting {file_path} to plain text: {e}", exc_info=True)
        log_counter(f"{extension.strip('.')}_conversion_error", labels={"file_path": str(file_path), "error": str(e)})
        raise ValueError(f"Failed to convert {extension}: {e}") from e

    return content


def _process_single_document(
    doc_path: Path,
    perform_chunking: bool,
    chunk_options: Dict[str, Any], # Changed from chunk_opts
    perform_analysis: bool, # Changed from summarize
    summarize_recursively: bool,
    api_name: Optional[str],
    api_key: Optional[str],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    # Add optional metadata overrides if needed
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None # Accept keywords
) -> Dict[str, Any]:
    """
    CPU‑bound worker: Reads/converts doc, chunks (optional), summarises (optional).
    Handles .txt, .md, .docx, .rtf. Returns a result dictionary.
    *No DB interaction.*

    Returns:
        - Dict[str, Any]: Dictionary containing processing results:
            {
                "status": "Success" | "Error",
                "input_ref": str (doc_path),
                "media_type": "document",
                "source_format": str (e.g., '.docx'),
                "text_content": Optional[str],
                "metadata": Optional[Dict],
                "chunks": Optional[List[Dict]],
                "summary": Optional[str],
                "keywords": Optional[List[str]], # Keywords PASSED IN
                "error": Optional[str],
                "analysis_details": Optional[Dict]
            }
    """
    start_time = time.time()
    result: Dict[str, Any] = {
        "status": "Pending",
        "input_ref": str(doc_path),
        "media_type": "document",
        "source_format": doc_path.suffix.lower(),
        "text_content": None,
        "metadata": None,
        "chunks": None,
        "summary": None,
        "keywords": keywords or [], # Store keywords passed in
        "error": None,
        "analysis_details": { # Initialize analysis details
            "summarization_model": api_name if perform_analysis else None,
            "custom_prompt_used": custom_prompt if perform_analysis else None,
            "system_prompt_used": system_prompt if perform_analysis else None,
            "summarized_recursively": summarize_recursively if perform_analysis else False,
        }
    }
    log_counter("document_processing_attempt", labels={"file_path": str(doc_path)})

    try:
        # 1. Read/Convert Content
        text_content = convert_to_plain_text(doc_path)
        if not text_content or not text_content.strip():
            raise ValueError("Empty or whitespace-only content after conversion.")
        result["text_content"] = text_content

        # 2. Prepare Metadata
        final_title = title_override or doc_path.stem
        final_author = author_override or "Unknown"
        result["metadata"] = {"title": final_title, "author": final_author}
        logging.debug(f"Document metadata - Title: {final_title}, Author: {final_author}")


        # 3. Chunking
        processed_chunks = None
        if perform_chunking and text_content:
            if chunk_options is None:
                chunk_options = {'method': 'recursive', 'max_size': 1000, 'overlap': 200}
            chunk_options.setdefault('method', 'recursive')

            logging.info(f"Chunking document content {doc_path} with options: {chunk_options}")
            from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process # Assuming exists
            processed_chunks = improved_chunking_process(text_content, chunk_options)

            if not processed_chunks:
                 logging.warning(f"Chunking produced no chunks for {doc_path}. Using full text.")
                 processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0}}]
            else:
                 logging.info(f"Total chunks created: {len(processed_chunks)}")
                 log_histogram("document_chunks_created", len(processed_chunks), labels={"file_path": str(doc_path)})

            result["chunks"] = processed_chunks
        elif text_content:
             processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks
             logging.info("Chunking disabled. Using full text as one chunk.")
        else:
             # This case should be caught by the ValueError above if content is empty
             logging.warning("Chunking skipped: No text content.")


        # 4. Summarization / Analysis
        final_summary = None
        if perform_analysis and api_name and api_key and processed_chunks:
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks of {doc_path}.")
            chunk_summaries: List[str] = []
            summarized_chunks_for_result = []

            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                if chunk_text:
                    try:
                        summary_text = perform_summarization(
                            api_name, chunk_text, custom_prompt, api_key,
                            recursive_summarization=False, temp=None, system_message=system_prompt
                        )
                        if summary_text and isinstance(summary_text, str) and summary_text.strip():
                            chunk_summaries.append(summary_text)
                            chunk_metadata['summary'] = summary_text
                        else:
                            chunk_metadata['summary'] = None
                    except Exception as summ_err:
                        logging.warning(f"Summarization failed for chunk {i} of {doc_path}: {summ_err}")
                        chunk_metadata['summary'] = f"[Summarization Error: {summ_err}]"

                chunk['metadata'] = chunk_metadata
                summarized_chunks_for_result.append(chunk)

            result["chunks"] = summarized_chunks_for_result

            # Combine summaries
            if chunk_summaries:
                if summarize_recursively and len(chunk_summaries) > 1:
                    logging.info(f"Performing recursive summarization on {len(chunk_summaries)} chunk summaries.")
                    try:
                         final_summary = perform_summarization(
                             api_name, "\n\n---\n\n".join(chunk_summaries), custom_prompt or "Overall summary:", api_key,
                             recursive_summarization=False, temp=None, system_message=system_prompt
                         )
                    except Exception as rec_summ_err:
                         logging.error(f"Recursive summarization failed for {doc_path}: {rec_summ_err}")
                         final_summary = f"[Recursive Summarization Error: {rec_summ_err}]\n\n" + "\n\n---\n\n".join(chunk_summaries)
                else:
                    final_summary = "\n\n---\n\n".join(chunk_summaries)

            result["summary"] = final_summary
            log_counter("document_chunks_summarized", value=len(chunk_summaries), labels={"file_path": str(doc_path)})
            logging.info("Summarization processing completed.")
        # ... (log skipped summarization reasons) ...
        elif not perform_analysis: logging.info(f"Summarization disabled for {doc_path}.")
        elif not api_name or not api_key: logging.warning(f"Summarization skipped for {doc_path}: API credentials missing.")
        elif not processed_chunks: logging.warning(f"Summarization skipped for {doc_path}: No chunks available.")

        result["status"] = "Success"
        log_counter("document_processing_success", labels={"file_path": str(doc_path)})

    except ValueError as ve: # Catch specific conversion/empty file errors
        logging.error(f"_process_single_document error for {doc_path}: {ve}", exc_info=True)
        result["status"] = "Error"
        result["error"] = str(ve)
        log_counter("document_processing_error", labels={"file_path": str(doc_path), "error": "ValueError"})
    except Exception as e:
        logging.error(f"_process_single_document unexpected error for {doc_path}: {e}", exc_info=True)
        result["status"] = "Error"
        result["error"] = f"Unexpected processing error: {str(e)}"
        log_counter("document_processing_error", labels={"file_path": str(doc_path), "error": type(e).__name__})

    end_time = time.time()
    processing_time = end_time - start_time
    log_histogram("document_processing_duration", processing_time, labels={"file_path": str(doc_path), "status": result["status"]})

    return result

#
# End of Plaintext_Files.py
#######################################################################################################################
