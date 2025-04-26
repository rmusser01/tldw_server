# Plaintext_Files.py
# Description: This file contains functions for reading and writing plaintext files.
#
# Import necessary libraries
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import xml.etree.ElementTree as ET
#
# External Imports
from bs4 import BeautifulSoup
from docx2txt import docx2txt
import html2text
from pypandoc import convert_file
#
# Local Imports
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import summarize
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
        elif extension == '.rtccf':
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


def _xml_to_text_simple(element):
    text = ''
    if element.text:
        text += element.text.strip()
    for child in element:
        text += ' ' + _xml_to_text_simple(child)
    if element.tail:
        text += ' ' + element.tail.strip()
    return text.strip()

# ───────────────────────────  Conversion Function ───────────────────────────
class PandocMissing:
    pass


def convert_document_to_text(file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
    """
    Converts various document formats to plain text and extracts basic metadata.

    Supported input formats: .txt, .md, .html, .htm, .xml, .docx, .rtf
    Output format: Plain text (Markdown for HTML/XML for structure).

    Returns:
        - Tuple[str, str, Dict[str, Any]]: (extracted_text, source_format_used, raw_metadata)
          Returns empty string and raises ValueError on critical failure.
          raw_metadata might contain format-specific info like original title.
    """
    extension = file_path.suffix.lower()
    content = ""
    source_format_used = extension.lstrip('.')
    raw_metadata = {}
    extracted_title = None
    extracted_author = "Unknown"

    try:
        logging.info(f"Attempting conversion for {file_path} (type: {extension})")
        if extension == '.docx':
            content = docx2txt.process(str(file_path))
            log_counter("docx_conversion_success", labels={"file_path": str(file_path)})
        elif extension == '.rtf':
            try:
                content = convert_file(str(file_path), 'plain', format='rtf')
                log_counter("rtf_conversion_success", labels={"file_path": str(file_path)})
            except Exception as e_rtf:
                # Now check the type of the caught exception 'e_rtf'
                if isinstance(e_rtf, PandocMissing):  # Use the imported PandocMissing here
                    logging.error(f"Pandoc dependency missing for RTF conversion: {e_rtf}")
                    raise ValueError(f"Cannot convert {extension}: Pandoc dependency missing.") from e_rtf
                elif isinstance(e_rtf, FileNotFoundError):
                    logging.error(f"File not found during RTF conversion: {e_rtf}")
                    raise ValueError(f"Cannot convert {extension}: File not found.") from e_rtf
                elif isinstance(e_rtf, ValueError):  # Catch the specific mock error by type check
                    logging.error(f"Pandoc conversion failed (ValueError) for RTF {file_path}: {e_rtf}", exc_info=False)
                    # Raise a NEW, clean ValueError containing the mock message
                    raise ValueError(f"RTF conversion failed: {str(e_rtf)}")
                else:  # Catch other unexpected errors during RTF conversion
                    logging.error(f"Unexpected Pandoc conversion error for RTF {file_path}: {e_rtf}", exc_info=True)
                    raise ValueError(f"Unexpected RTF conversion error: {str(e_rtf)}") from e_rtf
        elif extension in ['.txt', '.md']:
            content = _read_text(file_path) # Use robust reader
        elif extension in ['.html', '.htm']:
            source_format_used = 'html'
            h = html2text.HTML2Text()
            h.ignore_links = False # Keep links as text
            h.body_width = 0 # Don't wrap lines
            html_content = _read_text(file_path)
            content = h.handle(html_content) # Convert to Markdown
            # Try extracting title/author
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            extracted_title = title_tag.string.strip() if title_tag and title_tag.string else None
            meta_author_tag = soup.find('meta', attrs={'name': 'author'})
            if meta_author_tag and meta_author_tag.get('content'):
                extracted_author = meta_author_tag['content'].strip()
            raw_metadata = {'html_title': extracted_title, 'html_author': extracted_author}
            log_counter("html_conversion_success", labels={"file_path": str(file_path)})
        elif extension == '.xml':
            # Simple text extraction from XML - may need refinement based on XML structure
            try:
                tree = ET.parse(str(file_path))
                root = tree.getroot()
                # Basic text concatenation - consider xml_to_markdown if structure is important
                content = _xml_to_text_simple(root)
                # Try finding common title elements
                title_elem = root.find('.//title') or root.find('.//Title')
                extracted_title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
                raw_metadata = {'xml_root_tag': root.tag, 'xml_title': extracted_title}
                log_counter("xml_conversion_success", labels={"file_path": str(file_path)})
            except ET.ParseError as xml_err:
                 raise ValueError(f"Failed to parse XML file {file_path}: {xml_err}") from xml_err
        # Add other formats like OPML if needed, similar to XML/HTML handling
        # elif extension == '.opml': ...
        else:
            # Attempt reading as plain text as a last resort? Or fail? Let's fail for now.
            raise ValueError(f"Unsupported document file type: {extension}")

        # Basic cleanup
        content = re.sub(r'[ \t]+', ' ', content) # Collapse multiple spaces/tabs
        content = re.sub(r'\n\s*\n+', '\n\n', content) # Collapse multiple blank lines
        content = content.strip()

        if not content:
             logging.warning(f"Conversion resulted in empty content for {file_path}")
             # Don't raise error here, let _process handle empty content if needed

        raw_metadata['extracted_title'] = extracted_title
        raw_metadata['extracted_author'] = extracted_author

        return content, source_format_used, raw_metadata

    except (ValueError, ImportError) as specific_error:
        # Catch ValueErrors/ImportErrors raised intentionally above or by libraries
        logging.error(f"Conversion error for {file_path}: {specific_error}", exc_info=False) # Log cleanly
        log_counter(f"{source_format_used}_conversion_error", labels={"file_path": str(file_path), "error": type(specific_error).__name__})
        # Re-raise the specific error caught to be handled by process_document_content
        raise specific_error

    except Exception as unexpected_error:
        # Catch truly unexpected errors
        logging.exception(f"Unexpected error converting {file_path} to text: {unexpected_error}") # Use exception for full trace
        log_counter(f"{source_format_used}_conversion_error", labels={"file_path": str(file_path), "error": "UnexpectedError"})
        # Wrap in a consistent ValueError for process_document_content
        raise ValueError(f"Unexpected failure converting {extension} file '{file_path.name}': {unexpected_error}") from unexpected_error


# ───────────────────────────  Main Processing Function ───────────────────────────
def process_document_content( # Renamed from _process_single_document for clarity
    doc_path: Path,
    perform_chunking: bool,
    chunk_options: Optional[Dict[str, Any]], # Use Optional Dict
    perform_analysis: bool,
    summarize_recursively: bool,
    api_name: Optional[str],
    api_key: Optional[str],
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Reads/converts various document formats, chunks (optional), analyses (optional).
    Handles .txt, .md, .html, .xml, .docx, .rtf (requires pandoc).
    Returns a result dictionary aligned with MediaItemProcessResponse.
    *No DB interaction.*

    Returns: Dict aligned with MediaItemProcessResponse structure.
    """
    start_time = datetime.now()
    result: Dict[str, Any] = {
        "status": "Pending",
        "input_ref": str(doc_path), # Will be overwritten by endpoint with original ref
        "processing_source": str(doc_path), # Actual file processed
        "media_type": "document",
        "source_format": None, # Will be set after conversion
        "content": None, # Renamed from text_content
        "metadata": {}, # Initialize as dict
        "segments": None, # Add for compatibility, usually None for docs
        "chunks": None,
        "analysis": None, # Renamed from summary
        "analysis_details": { # Initialize analysis details
            "summarization_model": api_name if perform_analysis else None,
            "custom_prompt_used": custom_prompt if perform_analysis else None,
            "system_prompt_used": system_prompt if perform_analysis else None,
            "summarized_recursively": summarize_recursively if perform_analysis else False,
            "parser_used": None, # Will be set after conversion
        },
        "keywords": keywords or [], # Store keywords passed in
        "error": None,
        "warnings": [], # Initialize as list
        # --- DB Fields (ensure None) ---
        "db_id": None,
        "db_message": None,
        # -----------------------------
    }
    log_counter("document_processing_attempt", labels={"file_path": str(doc_path)})

    try:
        # 1. Read/Convert Content & Get Initial Metadata
        text_content, source_format_used, raw_metadata = convert_document_to_text(doc_path)
        result["content"] = text_content
        result["source_format"] = source_format_used
        result["analysis_details"]["parser_used"] = source_format_used # Track conversion method

        # Handle empty content after conversion
        if not text_content or not text_content.strip():
            # Don't error immediately, but log a warning. Analysis/chunking will skip.
             logging.warning(f"Document {doc_path} resulted in empty content after conversion.")
             result["warnings"].append("Content is empty after conversion.")
             # Set status to Warning if otherwise successful
             result["status"] = "Warning"
             # No further processing needed if content is empty
             return result


        # 2. Prepare Final Metadata
        extracted_title = raw_metadata.get('extracted_title')
        extracted_author = raw_metadata.get('extracted_author', 'Unknown')
        final_title = title_override or extracted_title or doc_path.stem
        final_author = author_override or extracted_author
        result["metadata"] = {
            "title": final_title,
            "author": final_author,
            "source_filename": doc_path.name,
            "raw": raw_metadata # Store format-specific extracted info
        }
        logging.debug(f"Document metadata - Title: {final_title}, Author: {final_author}")


        # 3. Chunking
        processed_chunks = None
        if perform_chunking and text_content:
            # Ensure chunk_options is a dict, provide defaults if None
            effective_chunk_options = chunk_options or {}
            # Sensible defaults for documents
            effective_chunk_options.setdefault('method', 'recursive')
            effective_chunk_options.setdefault('max_size', 1000) # Smaller default?
            effective_chunk_options.setdefault('overlap', 200)

            logging.info(f"Chunking document content {doc_path} with options: {effective_chunk_options}")
            try:
                # Use the generic chunking process
                processed_chunks = improved_chunking_process(text_content, effective_chunk_options)

                if not processed_chunks:
                     logging.warning(f"Chunking produced no chunks for {doc_path}. Using full text.")
                     result["warnings"].append("Chunking yielded no results; using full text as one chunk.")
                     # Ensure chunks list contains the single chunk
                     processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0}}]
                else:
                     logging.info(f"Total chunks created: {len(processed_chunks)}")
                     log_histogram("document_chunks_created", len(processed_chunks), labels={"file_path": str(doc_path)})

            except Exception as chunk_err:
                logging.error(f"Chunking failed for {doc_path}: {chunk_err}", exc_info=True)
                result["warnings"].append(f"Chunking failed: {chunk_err}")
                # Fallback: use full text as one chunk
                processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {chunk_err}"}}]

            result["chunks"] = processed_chunks

        elif text_content:
             # If not chunking, but content exists, create a single chunk
             processed_chunks = [{'text': text_content, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks # Store the single chunk
             logging.info("Chunking disabled. Using full text as one chunk.")
        else:
             # Content was empty or None (should have been handled earlier)
             logging.warning("Chunking skipped: No text content available.")


        # 4. Summarization / Analysis
        final_analysis = None
        if perform_analysis and api_name and api_key and processed_chunks:
            logging.info(f"Analysis enabled for {len(processed_chunks)} chunks of {doc_path}.")
            log_counter("document_analysis_attempt", value=len(processed_chunks), labels={"file_path": str(doc_path), "api_name": api_name})
            chunk_summaries: List[str] = []
            summarized_chunks_for_result = []

            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                if chunk_text:
                    try:
                        # Use standard summarize function signature
                        analysis_text = summarize(
                            api_name=api_name,
                            input_data=chunk_text,
                            custom_prompt_arg=custom_prompt,
                            api_key=api_key,
                            system_message=system_prompt,
                            temp=None, # Add default temp if needed
                            recursive_summarization=False, # Summarize chunk first
                        )
                        if analysis_text and isinstance(analysis_text, str) and analysis_text.strip():
                            chunk_summaries.append(analysis_text)
                            # Add analysis to chunk metadata for potential later use
                            chunk_metadata['analysis'] = analysis_text
                        else:
                            chunk_metadata['analysis'] = None
                            logging.debug(f"Analysis yielded empty result for chunk {i+1}/{len(processed_chunks)} of {doc_path}.")
                    except Exception as summ_err:
                        logging.warning(f"Analysis failed for chunk {i+1}/{len(processed_chunks)} of {doc_path}: {summ_err}", exc_info=False)
                        chunk_metadata['analysis'] = f"[Analysis Error: {str(summ_err)}]"
                        result["warnings"].append(f"Analysis failed for chunk {i+1}: {str(summ_err)}")

                chunk['metadata'] = chunk_metadata # Update chunk with metadata
                summarized_chunks_for_result.append(chunk)

            result["chunks"] = summarized_chunks_for_result # Update result with modified chunks

            # Combine summaries if generated
            if chunk_summaries:
                if summarize_recursively and len(chunk_summaries) > 1:
                    logging.info(f"Performing recursive analysis on {len(chunk_summaries)} chunk summaries for {doc_path}.")
                    try:
                         final_analysis = summarize(
                             api_name=api_name,
                             input_data="\n\n---\n\n".join(chunk_summaries), # Join summaries
                             custom_prompt_arg=custom_prompt or "Provide a concise overall summary of the following text sections.", # Recursive prompt
                             api_key=api_key,
                             system_message=system_prompt,
                             temp=None,
                             recursive_summarization=False, # Final pass
                         )
                         if not final_analysis or not final_analysis.strip():
                            logging.warning(f"Recursive analysis for {doc_path} yielded empty result. Falling back to joined summaries.")
                            final_analysis = "\n\n---\n\n".join(chunk_summaries) # Fallback
                            result["warnings"].append("Recursive analysis yielded empty result.")
                         else:
                             log_counter("document_recursive_analysis_success", labels={"file_path": str(doc_path)})

                    except Exception as rec_summ_err:
                         logging.error(f"Recursive analysis failed for {doc_path}: {rec_summ_err}", exc_info=True)
                         final_analysis = f"[Recursive Analysis Error: {str(rec_summ_err)}]\n\n" + "\n\n---\n\n".join(chunk_summaries)
                         result["warnings"].append(f"Recursive analysis failed: {str(rec_summ_err)}")
                         log_counter("document_recursive_analysis_error", labels={"file_path": str(doc_path), "error": str(rec_summ_err)})
                else:
                    # Simple join if not recursive or only one summary
                    final_analysis = "\n\n---\n\n".join(chunk_summaries)
                    if len(chunk_summaries) > 1 : logging.info(f"Combined {len(chunk_summaries)} chunk analyses (non-recursive).")
                    else: logging.info(f"Using single chunk analysis as final analysis.")

            result["analysis"] = final_analysis # Store final combined analysis
            log_counter("document_chunks_analyzed", value=len(chunk_summaries), labels={"file_path": str(doc_path)})
            logging.info(f"Analysis processing completed for document {doc_path}.")

        # Log skipped analysis reasons
        elif not perform_analysis: logging.info(f"Analysis disabled for {doc_path}.")
        elif not api_name or not api_key: logging.warning(f"Analysis skipped for {doc_path}: API credentials missing.")
        elif not processed_chunks: logging.warning(f"Analysis skipped for {doc_path}: No processable chunks available.")
        else: logging.warning(f"Analysis skipped for {doc_path} due to unknown condition.")


        # Determine final status (Success or Warning)
        if not result["warnings"]:
            result["status"] = "Success"
        else:
            # Already set to Warning if content was empty, otherwise set it now
            if result["status"] != "Warning":
                 result["status"] = "Warning"

        log_counter(f"document_processing_{result['status'].lower()}", labels={"file_path": str(doc_path)})

    except ValueError as ve: # Catch specific conversion/empty file errors from convert_document_to_text
        logging.error(f"Processing error for {doc_path}: {ve}", exc_info=False) # Log less verbosely
        result["status"] = "Error"
        result["error"] = str(ve)
        log_counter("document_processing_error", labels={"file_path": str(doc_path), "error": "ValueError"})
    except Exception as e:
        logging.exception(f"Unexpected error processing document {doc_path}: {str(e)}")
        result["status"] = "Error"
        result["error"] = f"Unexpected processing error: {str(e)}"
        log_counter("document_processing_error", labels={"file_path": str(doc_path), "error": type(e).__name__})

    # Ensure warnings list is None if empty for cleaner JSON output
    if not result["warnings"]:
        result["warnings"] = None

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    log_histogram("document_processing_duration", processing_time, labels={"file_path": str(doc_path), "status": result["status"]})

    logging.info(f"Document '{result.get('metadata',{}).get('title', doc_path.name)}' processed with status: {result['status']} in {processing_time:.2f}s")

    return result

#
# End of Plaintext_Files.py
#######################################################################################################################
