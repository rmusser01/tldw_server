# PDF_Processing_Lib.py
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
import asyncio
import gc
import importlib.util
import re
import sys
from datetime import datetime
from typing import Any, Optional, Union

#
# Import External Libs
import pymupdf
import pymupdf4llm

#
# Import Local
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend as _get_ocr_backend
from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
    effective_page_concurrency as _effective_ocr_page_concurrency,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.Utils.Utils import logging

_PDF_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

try:
    # Optional VLM module (vision backends)
    from tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry import (
        get_backend as _get_vlm_backend,
    )
except ImportError:
    def _get_vlm_backend(name=None):
        return None  # type: ignore
#
# Constants
# Get configuration values or use defaults
media_config = loaded_config_data.get('media_processing', {}) if loaded_config_data else {}
MAX_FILE_SIZE_MB = media_config.get('max_pdf_file_size_mb', 50)
CONVERSION_TIMEOUT_SECONDS = media_config.get('pdf_conversion_timeout_seconds', 300)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)")
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s*")
_CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_PAGE_MARKER_RE = re.compile(r"^\s*##\s+Page\s+\d+\s*$", re.IGNORECASE)
_HRULE_RE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[:\- ]+\|[:\-\| ]+\s*$")
_LOWERCASE_START_RE = re.compile(r"^[a-z]")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_LIST_CONTINUATION_RE = re.compile(r"^\s{2,}\S")
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_OCR_BACKEND_OUTPUT_DENYLIST = {"argv", "host", "port", "url", "prompt", "model"}
#
#######################################################################################################################
# Function Definitions
#


def _ocr_min_text_threshold(value: int) -> int:
    return max(value, 1)


def _should_replace_ocr_content(
    content_text_len: int,
    ocr_mode: Optional[str],
    ocr_min_page_text_chars: int,
) -> bool:
    return (ocr_mode or "fallback").lower() == "always" or content_text_len < _ocr_min_text_threshold(
        ocr_min_page_text_chars
    )


def _sanitize_ocr_backend_details_for_output(details: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in details.items():
        if key in _OCR_BACKEND_OUTPUT_DENYLIST:
            continue
        if isinstance(value, dict):
            sanitized[key] = _sanitize_ocr_backend_details_for_output(value)
            continue
        if isinstance(value, list):
            sanitized[key] = [
                _sanitize_ocr_backend_details_for_output(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
            continue
        sanitized[key] = value
    return sanitized


def _is_usable_torch_module_for_docling() -> bool:
    """Best-effort guard to distinguish real torch from lightweight test stubs."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return False

    spec = getattr(torch_mod, "__spec__", None)
    if spec is None or getattr(spec, "loader", None) is None:
        return False

    # Common stub modules in tests only provide Tensor/nn and omit runtime attrs.
    if not hasattr(torch_mod, "__version__"):
        return False
    if not hasattr(torch_mod, "cuda"):
        return False

    return True


def _is_table_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _TABLE_SEPARATOR_RE.match(stripped):
        return True
    return stripped.count("|") >= 2


def _is_structural_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING_RE.match(line):
        return True
    if _LIST_ITEM_RE.match(line):
        return True
    if _BLOCKQUOTE_RE.match(line):
        return True
    if _PAGE_MARKER_RE.match(line):
        return True
    if _HRULE_RE.match(line):
        return True
    if _is_table_like_line(line):
        return True
    # Preserve indented literal/code-style blocks.
    return line.startswith("    ") or line.startswith("\t")


def _is_list_item_line(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line))


def _collapse_inline_whitespace(value: str) -> str:
    return _INLINE_WHITESPACE_RE.sub(" ", value).strip()


def _first_non_space_char(value: str) -> str:
    for char in value:
        if not char.isspace():
            return char
    return ""


def _last_non_space_char(value: str) -> str:
    for char in reversed(value):
        if not char.isspace():
            return char
    return ""


def _should_join_without_space(prev_text: str, next_text: str) -> bool:
    prev_char = _last_non_space_char(prev_text)
    next_char = _first_non_space_char(next_text)
    if not prev_char or not next_char:
        return False
    return bool(_CJK_CHAR_RE.match(prev_char) and _CJK_CHAR_RE.match(next_char))


def _join_wrapped_line(prev_text: str, next_text: str) -> str:
    collapsed_next = _collapse_inline_whitespace(next_text)
    if not collapsed_next:
        return prev_text
    if prev_text.endswith("-") and _LOWERCASE_START_RE.match(collapsed_next):
        return prev_text[:-1] + collapsed_next
    if _should_join_without_space(prev_text, collapsed_next):
        return f"{prev_text}{collapsed_next}"
    return f"{prev_text} {collapsed_next}"


def _is_artifact_heavy_block(lines: list[str]) -> bool:
    joined = " ".join(lines)
    if len(joined) < 40:
        return False
    punctuation_count = len(_PUNCTUATION_RE.findall(joined))
    punctuation_density = punctuation_count / max(len(joined), 1)
    has_heavy_delimiters = any(line.count("|") >= 2 or line.count("\t") >= 2 for line in lines)
    return punctuation_density >= 0.32 and has_heavy_delimiters


def _reflow_paragraph_lines(lines: list[str]) -> str:
    cleaned_lines = [_collapse_inline_whitespace(line) for line in lines]
    cleaned_lines = [line for line in cleaned_lines if line]
    if not cleaned_lines:
        return ""
    if _is_artifact_heavy_block(cleaned_lines):
        return "\n".join(cleaned_lines)

    output = cleaned_lines[0]
    for line in cleaned_lines[1:]:
        output = _join_wrapped_line(output, line)
    return _collapse_inline_whitespace(output)


def normalize_pdf_text_for_storage(text: str) -> str:
    """
    Normalize extracted PDF text to paragraph-safe flowed text for storage.

    Single-line soft wraps are joined only inside non-structural paragraph
    blocks. Headings, lists, tables, code fences, and page markers are
    preserved as-is.
    """
    if not isinstance(text, str):
        return ""
    if not text:
        return ""

    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized_newlines.split("\n")
    output_lines: list[str] = []
    paragraph_buffer: list[str] = []
    in_code_fence = False
    in_list_context = False

    def _flush_paragraph_buffer() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        reflowed = _reflow_paragraph_lines(paragraph_buffer)
        if reflowed:
            output_lines.append(reflowed)
        paragraph_buffer = []

    for raw_line in raw_lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if _CODE_FENCE_RE.match(stripped):
            _flush_paragraph_buffer()
            output_lines.append(line)
            in_code_fence = not in_code_fence
            in_list_context = False
            continue

        if in_code_fence:
            _flush_paragraph_buffer()
            output_lines.append(line)
            in_list_context = False
            continue

        if not stripped:
            _flush_paragraph_buffer()
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            in_list_context = False
            continue

        if _is_structural_line(line):
            _flush_paragraph_buffer()
            output_lines.append(line)
            in_list_context = _is_list_item_line(line)
            continue

        # Preserve list continuation indentation when directly following a list item.
        if in_list_context and _LIST_CONTINUATION_RE.match(line):
            _flush_paragraph_buffer()
            output_lines.append(line)
            continue

        # Treat non-structural unindented lines after list items as wrapped continuations.
        if in_list_context and output_lines and output_lines[-1] != "":
            _flush_paragraph_buffer()
            output_lines[-1] = _join_wrapped_line(output_lines[-1], line)
            continue

        paragraph_buffer.append(line)
        in_list_context = False

    _flush_paragraph_buffer()
    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    # Only trim boundary newlines; preserve intentional leading indentation.
    return "\n".join(output_lines).strip("\n")


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
    except _PDF_NONCRITICAL_EXCEPTIONS as e:
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
        DOCLING_AVAILABLE = True  # Set to True if import succeeds
    except ImportError:
        DOCLING_AVAILABLE = False
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling library is not installed.")
    try:
        log_counter("pdf_text_extraction_attempt", labels={"file_path": pdf_path, "parser": parser_name})
        start_time = datetime.now()

        # Avoid OCR-dependent backends (EasyOCR -> torch) in constrained
        # environments; keep docling focused on native PDF text extraction.
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = False
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
            )
        except _PDF_NONCRITICAL_EXCEPTIONS:
            converter = DocumentConverter()
        parsed_pdf = converter.convert(pdf_path)
        markdown_text = parsed_pdf.document.export_to_markdown() # Or other formats if needed

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        log_histogram("pdf_text_extraction_duration", processing_time, labels={"file_path": pdf_path, "parser": parser_name})
        log_counter("pdf_text_extraction_success", labels={"file_path": pdf_path, "parser": parser_name})
        return markdown_text

    except _PDF_NONCRITICAL_EXCEPTIONS as e:
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
    except _PDF_NONCRITICAL_EXCEPTIONS as e:
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
    except _PDF_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error extracting metadata from PDF: {str(e)}")
        log_counter("pdf_metadata_extraction_error", labels={"file_path": pdf_path, "error": str(e)})
        return {}


# PDF_Ingestion_Lib.py
# Add these imports at the top if not already present
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

# ... other imports ...

def process_pdf(
    file_input: Union[str, bytes, Path], # Can be path, bytes, or Path object
    filename: str, # Original filename for reference and metadata fallback
    parser: str = "pymupdf4llm",
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    perform_chunking: bool = True,
    chunk_options: Optional[dict[str, Any]] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    summarize_recursively: bool = False,
    # OCR options
    enable_ocr: bool = False,
    ocr_backend: Optional[str] = None,
    ocr_lang: Optional[str] = "eng",
    ocr_dpi: int = 300,
    ocr_mode: Optional[str] = "fallback",
    ocr_min_page_text_chars: int = 40,
    ocr_output_format: Optional[str] = None,
    ocr_prompt_preset: Optional[str] = None,
    # VLM options
    enable_vlm: bool = False,
    vlm_backend: Optional[str] = None,
    vlm_detect_tables_only: bool = True,
    vlm_max_pages: Optional[int] = None,
    base_dir: Optional[Path] = None,
    # write_to_temp_file: bool = False # This param seems unused/obsolete now
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
      - base_dir (Optional[Path]): If provided, require file_input paths to resolve
                                   within this directory before processing.
      - write_to_temp_file (bool): If True and input is bytes, write to a temp file
                                  (needed for parsers that only accept paths).

    Returns:
        - Dict[str, Any]: Dictionary containing processing results:
            {
                "status": "Success" | "Error" | "Warning",
                "input_ref": str (filename),
                "media_type": "pdf",
                "parser_used": str,
                "content": Optional[str],
                "metadata": Optional[Dict], # {'title': str, 'author': str, 'raw': dict}
                "chunks": Optional[List[Dict]],
                "analysis": Optional[str],
                "keywords": Optional[List[str]],
                "error": Optional[str],
                "warnings": Optional[List[str]],
                "analysis_details": Optional[Dict] # Added
            }
    """
    start_time = datetime.now()
    # Initialize the result dictionary structure

    result: dict[str, Any] = {
        "status": "Pending",
        "input_ref": filename,
        "media_type": "pdf",
        "parser_used": parser,
        "content": None,
        "metadata": None,
        "chunks": None,
        "analysis": None,
        "keywords": keywords or [], # Store keywords passed in
        "error": None,
        "warnings": [], # Initialize as list for easier appending
        "analysis_details": {
            "analysis_model": api_name if perform_analysis else None,
            "custom_prompt_used": custom_prompt if perform_analysis else None,
            "system_prompt_used": system_prompt if perform_analysis else None,
            "summarized_recursively": summarize_recursively if perform_analysis else False,
            "vlm": None,
        }
    }
    log_counter("pdf_processing_attempt", labels={"file_name": filename, "parser": parser})

    if base_dir is not None and isinstance(file_input, (str, Path)):
        candidate_path = Path(file_input)
        safe_path = resolve_safe_local_path(candidate_path, base_dir)
        if safe_path is None:
            err_msg = "PDF path rejected outside allowed base directory."
            result["status"] = "Error"
            result["error"] = err_msg
            result["warnings"] = [err_msg]
            return result
        file_input = safe_path

    temp_dir_for_pdf: Optional[str] = None
    path_for_processing: Optional[str] = None
    content: Optional[str] = None

    try:
        # --- Step 0: Handle Input Type and Ensure File Path for Processing ---
        if isinstance(file_input, bytes):
            result["processing_source"] = f"bytes_input_({len(file_input)})"
            # ALWAYS write bytes to a temp file for consistent parser input
            try:
                # Create a unique temporary directory
                temp_dir_for_pdf = tempfile.mkdtemp(prefix="pdf_process_")
                # Create a filename inside (use UUID for uniqueness)
                temp_pdf_path_obj = Path(temp_dir_for_pdf) / f"{uuid.uuid4()}.pdf"
                path_for_processing = str(temp_pdf_path_obj)

                # Write bytes to the file
                with open(path_for_processing, "wb") as f_out:
                    f_out.write(file_input)

                logging.debug(f"Input bytes written to temporary file: {path_for_processing} in dir {temp_dir_for_pdf}")
                result["processing_source"] = path_for_processing # Update source info

            except _PDF_NONCRITICAL_EXCEPTIONS as temp_err:
                # Cleanup directory if creation failed partially
                if temp_dir_for_pdf and os.path.isdir(temp_dir_for_pdf):
                    try:
                        shutil.rmtree(temp_dir_for_pdf)
                    except _PDF_NONCRITICAL_EXCEPTIONS:
                        logging.error(f"Failed secondary cleanup of {temp_dir_for_pdf}")
                raise OSError(f"Failed to create or write temporary file/dir: {temp_err}") from temp_err

        elif isinstance(file_input, Path):
            path_str = str(file_input)
            if not file_input.exists():
                raise FileNotFoundError(f"Input file path does not exist: {path_str}")
            path_for_processing = path_str # Use original path
            result["processing_source"] = path_str
        elif isinstance(file_input, str):
            if not os.path.exists(file_input):
                raise FileNotFoundError(f"Input file path does not exist: {file_input}")
            path_for_processing = file_input # Use original path
            result["processing_source"] = file_input
        else:
            raise TypeError(f"Unsupported file_input type: {type(file_input)}")

        # --- Step 1: Extract Text (Now always uses path_for_processing) ---
        if not path_for_processing: # Should not happen, but defensive check
             raise RuntimeError("Internal logic error: path_for_processing not set")

        try:
            logging.info(f"Attempting text extraction for {filename} using parser: {parser} on path: {path_for_processing}")
            if parser == "pymupdf4llm":
                # Now correctly called with a path
                content = pymupdf4llm_parse_pdf(path_for_processing)
            elif parser == "pymupdf":
                 content = extract_text_and_format_from_pdf(path_for_processing)
            elif parser == "docling":
                DOCLING_AVAILABLE = importlib.util.find_spec("docling.document_converter") is not None
                if not DOCLING_AVAILABLE:
                    raise ImportError("Docling parser selected, but library is not installed.")
                # Docling currently pulls torch-backed layout models. Avoid
                # running docling when torch is missing or clearly stubbed.
                if not _is_usable_torch_module_for_docling():
                    logging.info(
                        "Docling parser requested but torch is unavailable/stubbed; "
                        "falling back to pymupdf4llm for %s",
                        filename,
                    )
                    result["parser_used"] = "pymupdf4llm"
                    content = pymupdf4llm_parse_pdf(path_for_processing)
                else:
                    try:
                        content = docling_parse_pdf(path_for_processing)
                    except _PDF_NONCRITICAL_EXCEPTIONS as docling_exc:
                        logging.warning(
                            "Docling parser failed for %s (%s); falling back to pymupdf4llm",
                            filename,
                            docling_exc,
                        )
                        result["parser_used"] = "pymupdf4llm"
                        content = pymupdf4llm_parse_pdf(path_for_processing)
            else:
                # This case should ideally be caught by Pydantic validation in the endpoint
                logging.warning(f"Unsupported PDF parser specified: {parser}. Attempting fallback to pymupdf4llm.")
                result["warnings"].append(f"Unsupported parser '{parser}', fallback to 'pymupdf4llm'")
                result["parser_used"] = "pymupdf4llm"
                content = pymupdf4llm_parse_pdf(path_for_processing) # Fallback also uses path

            result["content"] = content
            if content is not None: # Check if extraction actually yielded content
                 logging.info(f"Text extracted successfully for {filename} using {result['parser_used']}.")
            else:
                 # Handle cases where parsing succeeded but returned nothing (e.g., empty PDF)
                 logging.warning(f"Text extraction using {result['parser_used']} for {filename} yielded no content.")
                 result["warnings"].append(f"Text extraction yielded no content ({result['parser_used']}).")

            # --- Optional OCR step (always/fallback) ---
            try:
                content_text_len = len((content or "").strip())
                should_ocr = False
                if enable_ocr:
                    mode = (ocr_mode or "fallback").lower()
                    if mode == "always" or content_text_len < _ocr_min_text_threshold(ocr_min_page_text_chars):
                        should_ocr = True

                if should_ocr:
                    requested_backend = (ocr_backend or "").strip().lower()
                    if requested_backend == "mineru":
                        mineru_warnings: list[str] = []
                        if (ocr_lang or "eng") != "eng":
                            mineru_warnings.append("MinerU ignores ocr_lang in v1")
                        if ocr_dpi != 300:
                            mineru_warnings.append("MinerU ignores ocr_dpi in v1")

                        try:
                            mineru_result = _run_mineru_document_ocr(
                                pdf_path=Path(path_for_processing),
                                output_format=ocr_output_format,
                                prompt_preset=ocr_prompt_preset,
                                requested_lang=ocr_lang,
                                requested_dpi=ocr_dpi,
                            )
                            result.setdefault("analysis_details", {})
                            details = dict(mineru_result.get("details") or {})
                            details.setdefault("backend", "mineru")
                            details.setdefault("mode", (ocr_mode or "fallback").lower())
                            details.setdefault("lang", ocr_lang or "eng")
                            details.setdefault("dpi", ocr_dpi)
                            details.setdefault("output_format", ocr_output_format)
                            details.setdefault("prompt_preset", ocr_prompt_preset)
                            details["warnings"] = mineru_warnings + list(mineru_result.get("warnings") or [])

                            structured = mineru_result.get("structured")
                            if structured is not None:
                                details["structured"] = structured

                            result["analysis_details"]["ocr"] = details
                            result["warnings"] = (result.get("warnings") or []) + details["warnings"]

                            ocr_text = str(mineru_result.get("text") or "")
                            if ocr_text.strip():
                                result["content"] = ocr_text
                                result["parser_used"] = f"{result['parser_used']}+mineru"
                            else:
                                result["warnings"] = (result.get("warnings") or []) + ["MinerU produced no text"]
                        except _PDF_NONCRITICAL_EXCEPTIONS as _ocr_err:
                            logging.error(f"MinerU OCR error for {filename}: {_ocr_err}", exc_info=True)
                            result.setdefault("analysis_details", {})
                            result["analysis_details"]["ocr"] = {
                                "backend": "mineru",
                                "mode": (ocr_mode or "fallback").lower(),
                                "lang": ocr_lang or "eng",
                                "dpi": ocr_dpi,
                                "output_format": ocr_output_format,
                                "prompt_preset": ocr_prompt_preset,
                                "error": str(_ocr_err),
                                "warnings": mineru_warnings,
                            }
                            result["warnings"] = (result.get("warnings") or []) + mineru_warnings + [
                                f"OCR error: {_ocr_err}"
                            ]
                    else:
                        backend = _get_ocr_backend(ocr_backend if ocr_backend not in (None, "auto") else None)
                        if backend is None:
                            logging.warning("OCR requested but no available OCR backend found.")
                            result["warnings"] = (result.get("warnings") or []) + [
                                "OCR requested but no backend available"
                            ]
                        else:
                            backend_details: dict[str, Any] = {}
                            backend_page_concurrency: Optional[int] = None
                            try:
                                if hasattr(backend, "describe") and callable(backend.describe):
                                    extra = backend.describe() or {}
                                    if isinstance(extra, dict):
                                        backend_details = dict(extra)
                                        raw_backend_page_concurrency = backend_details.get("backend_concurrency_cap")
                                        if raw_backend_page_concurrency is None:
                                            raw_backend_page_concurrency = backend_details.get("max_page_concurrency")
                                        if raw_backend_page_concurrency is None:
                                            raw_backend_page_concurrency = backend_details.get("page_concurrency")
                                        if raw_backend_page_concurrency is not None:
                                            try:
                                                backend_page_concurrency = max(1, int(raw_backend_page_concurrency))
                                            except (TypeError, ValueError):
                                                backend_page_concurrency = None
                            except _PDF_NONCRITICAL_EXCEPTIONS:
                                backend_details = {}
                                backend_page_concurrency = None

                            # Determine OCR concurrency using the global cap and any backend-local cap.
                            try:
                                import os as _os
                                concurrency_env = _os.getenv("OCR_PAGE_CONCURRENCY")
                                if concurrency_env is not None:
                                    concurrency_env = int(concurrency_env)
                                else:
                                    # Fall back to config default if present
                                    cfg = loaded_config_data.get('OCR', {}) if loaded_config_data else {}
                                    concurrency_env = int(cfg.get('page_concurrency_default', 1))
                            except _PDF_NONCRITICAL_EXCEPTIONS:
                                concurrency_env = 1
                            effective_page_concurrency = _effective_ocr_page_concurrency(
                                concurrency_env,
                                backend_page_concurrency,
                            )

                            ocr_text, page_count, ocr_pages, structured_pages = _ocr_pdf_pages(
                                pdf_path=path_for_processing,
                                lang=ocr_lang or "eng",
                                dpi=ocr_dpi,
                                backend=backend,
                                per_page_min_text=ocr_min_page_text_chars,
                                per_page_check=True,
                                concurrency=effective_page_concurrency,
                                output_format=ocr_output_format,
                                prompt_preset=ocr_prompt_preset,
                            )
                            result.setdefault("analysis_details", {})
                            details = {
                                "backend": getattr(backend, "name", type(backend).__name__),
                                "mode": (ocr_mode or "fallback").lower(),
                                "dpi": ocr_dpi,
                                "lang": ocr_lang or "eng",
                                "total_pages": page_count,
                                "ocr_pages": ocr_pages,
                                "output_format": ocr_output_format,
                                "prompt_preset": ocr_prompt_preset,
                            }
                            refreshed_backend_details = backend_details
                            try:
                                if hasattr(backend, "describe") and callable(backend.describe):
                                    refreshed = backend.describe() or {}
                                    if isinstance(refreshed, dict):
                                        refreshed_backend_details = dict(refreshed)
                            except _PDF_NONCRITICAL_EXCEPTIONS:
                                refreshed_backend_details = backend_details

                            if refreshed_backend_details:
                                refreshed_backend_details = _sanitize_ocr_backend_details_for_output(
                                    dict(refreshed_backend_details)
                                )
                                if backend_page_concurrency is None:
                                    refreshed_backend_details.pop("backend_concurrency_cap", None)
                                    refreshed_backend_details.pop("max_page_concurrency", None)
                                else:
                                    refreshed_backend_details["backend_concurrency_cap"] = backend_page_concurrency

                            # Attach backend-specific metadata if available.
                            if refreshed_backend_details:
                                details.update(refreshed_backend_details)
                            details["page_concurrency"] = effective_page_concurrency
                            details["effective_page_concurrency"] = effective_page_concurrency
                            if backend_page_concurrency is not None:
                                details.setdefault("backend_concurrency_cap", backend_page_concurrency)
                            if structured_pages is not None:
                                try:
                                    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import (
                                        OCRResult,
                                        normalize_ocr_format,
                                    )
                                    fmt = normalize_ocr_format(ocr_output_format)
                                    if fmt == "unknown":
                                        fmt = "text"
                                    details["structured"] = OCRResult(
                                        text=ocr_text or "",
                                        format=fmt,
                                        pages=structured_pages,
                                        meta={
                                            "backend": details.get("backend"),
                                            "mode": details.get("mode"),
                                            "prompt_preset": ocr_prompt_preset,
                                            "output_format": ocr_output_format,
                                        },
                                    ).as_dict()
                                except _PDF_NONCRITICAL_EXCEPTIONS:
                                    pass
                            result["analysis_details"]["ocr"] = details

                            if ocr_text and ocr_text.strip():
                                if _should_replace_ocr_content(
                                    content_text_len,
                                    ocr_mode,
                                    ocr_min_page_text_chars,
                                ):
                                    result["content"] = ocr_text
                                    result["parser_used"] = f"{result['parser_used']}+ocr"
                                else:
                                    result["content"] = (content or "") + "\n\n" + ocr_text
                                    result["parser_used"] = f"{result['parser_used']}+ocr-appended"
                            else:
                                result["warnings"] = (result.get("warnings") or []) + ["OCR produced no text"]
            except _PDF_NONCRITICAL_EXCEPTIONS as _ocr_err:
                logging.error(f"OCR error for {filename}: {_ocr_err}", exc_info=True)
                result["warnings"] = (result.get("warnings") or []) + [f"OCR error: {_ocr_err}"]

        except (RuntimeError, pymupdf.FileDataError, pymupdf.EmptyFileError) as parse_lib_err:
             # --- CATCH PDF library errors during parsing specifically ---
             err_msg = str(parse_lib_err)
             if "password" in err_msg.lower():
                 log_msg = f"PDF password error during text extraction for {filename}: {err_msg}"
             elif isinstance(parse_lib_err, pymupdf.EmptyFileError):
                 log_msg = f"PDF empty file error during text extraction for {filename}: {err_msg}"
             elif isinstance(parse_lib_err, pymupdf.FileDataError):
                 log_msg = f"PDF file data error during text extraction for {filename}: {err_msg}"
             else:
                 log_msg = f"PDF library runtime error during text extraction for {filename}: {err_msg}"

             logging.error(log_msg, exc_info=True) # Log specifics
             result["warnings"].append(f"Text extraction failed ({parser}): {err_msg}")
             # Don't raise here, allow metadata extraction attempt

        except _PDF_NONCRITICAL_EXCEPTIONS as parse_err:
             # Catch other potential errors during parsing
             logging.error(f"Unexpected error during text extraction for {filename} using {parser}: {parse_err}", exc_info=True)
             result["warnings"].append(f"Unexpected text extraction error ({parser}): {str(parse_err)}")
             # Don't raise here


        # --- Step 2: Extract Metadata ---
        # Metadata extraction should work even if text extraction failed.
        try:
            logging.info(f"Attempting metadata extraction for {filename}.")
            # Use pymupdf directly for metadata, as it's generally robust
            raw_metadata = {}
            page_count = 0
            # No need for internal try/except around import pymupdf if it's at top level
            # Use filename argument directly with pymupdf.open
            with pymupdf.open(filename=path_for_processing) as doc: # Use filename= for path
                raw_metadata = doc.metadata
                page_count = doc.page_count
            logging.info(f"Metadata extracted for {filename}.")

            # Add subject and keywords from metadata to the provided keywords list
            pdf_keywords_str = raw_metadata.get('keywords', '')
            pdf_subject = raw_metadata.get('subject')
            # Use sets for efficient merging and deduplication
            combined_keywords = {k.strip() for k in (keywords or []) if k.strip()} # Start with input keywords
            if pdf_keywords_str and isinstance(pdf_keywords_str, str):
                combined_keywords.update(k.strip() for k in pdf_keywords_str.split(',') if k.strip())
            if pdf_subject and isinstance(pdf_subject, str) and pdf_subject.strip():
                 combined_keywords.add(pdf_subject.strip())
            result["keywords"] = sorted(combined_keywords) # Store unique, sorted keywords

            # Determine final title/author using overrides, then metadata, then filename
            final_title = title_override or raw_metadata.get('title') or Path(filename).stem
            final_author = author_override or raw_metadata.get('author') or "Unknown"
            result["metadata"] = {
                "title": final_title,
                "author": final_author,
                "page_count": page_count,
                "creationDate": raw_metadata.get('creationDate'),
                "modDate": raw_metadata.get('modDate'),
                "producer": raw_metadata.get('producer'),
                "creator": raw_metadata.get('creator'),
                "raw": raw_metadata
            }
            logging.debug(f"Final metadata for {filename} - Title: {final_title}, Author: {final_author}")

        except (RuntimeError, pymupdf.FileDataError, pymupdf.EmptyFileError) as meta_lib_err:
             # --- CATCH PDF library errors during metadata specifically ---
             err_msg = str(meta_lib_err)
             # Create user-friendly error message for metadata failure
             if "password" in err_msg.lower():
                 meta_fail_reason = "PDF Error: Password required or invalid."
             elif isinstance(meta_lib_err, pymupdf.EmptyFileError):
                 meta_fail_reason = "PDF Error: Input file is empty."
             elif isinstance(meta_lib_err, pymupdf.FileDataError):
                 meta_fail_reason = "PDF Error: Corrupted or invalid file data."
             else:
                 meta_fail_reason = f"PDF Library Error: {err_msg}" # General PDF error

             logging.error(f"Metadata extraction failed for {filename}: {meta_fail_reason}", exc_info=True)
             result["warnings"].append(f"Metadata extraction failed: {meta_fail_reason}")
             result["metadata"] = { # Provide default structure on failure
                 "title": title_override or Path(filename).stem, "author": author_override or "Unknown",
                 "page_count": 0, "raw": {"error": f"Metadata extraction failed: {meta_fail_reason}"}
             }

        except _PDF_NONCRITICAL_EXCEPTIONS as meta_err:
             logging.error(f"Unexpected metadata extraction error for {filename}: {meta_err}", exc_info=True)
             meta_fail_reason = f"Unexpected error: {str(meta_err)}"
             result["warnings"].append(f"Metadata extraction failed: {meta_fail_reason}")
             result["metadata"] = { # Provide default structure
                 "title": title_override or Path(filename).stem, "author": author_override or "Unknown",
                 "page_count": 0, "raw": {"error": f"Metadata extraction failed: {meta_fail_reason}"}
             }


        # --- Step 2.5: VLM (Vision) Analysis / Detections ---
        try:
            if enable_vlm:
                backend = _get_vlm_backend(vlm_backend if vlm_backend not in (None, "auto") else None)
                if backend is None:
                    result["warnings"].append("VLM requested but no backend available")
                else:
                    vlm_summary: dict[str, Any] = {
                        "backend": getattr(backend, "name", "unknown"),
                        "pages_scanned": 0,
                        "detections_total": 0,
                        "by_page": [],
                    }
                    extra_chunks: list[dict[str, Any]] = []

                    # Prefer document-level processing if backend exposes it (e.g., docling)
                    if hasattr(backend, "process_pdf"):
                        try:
                            res = backend.process_pdf(path_for_processing, max_pages=vlm_max_pages)
                            # res.extra may contain by_page
                            by_page = []
                            if isinstance(getattr(res, "extra", None), dict):
                                by_page = res.extra.get("by_page") or []

                            # Build summary + chunks from by_page if present
                            for entry in by_page:
                                page_no = entry.get("page")
                                dets = entry.get("detections") or []
                                vlm_summary["pages_scanned"] += 1
                                page_dets = []
                                for d in dets:
                                    label = str(d.get("label"))
                                    score = float(d.get("score", 0.0))
                                    bbox = d.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                                    if vlm_detect_tables_only and label.lower() != "table":
                                        continue
                                    page_dets.append({"label": label, "score": score, "bbox": bbox})
                                    # Extra chunk
                                    chunk_text = f"Detected {label} ({score:.2f}) on page {page_no} at {bbox}"
                                    extra_chunks.append({
                                        "text": chunk_text,
                                        "start_char": None,
                                        "end_char": None,
                                        "chunk_type": "vlm" if label.lower() != "table" else "table",
                                        "metadata": {"page": page_no, "bbox": bbox, "label": label, "score": score},
                                    })
                                    vlm_summary["detections_total"] += 1
                                vlm_summary["by_page"].append({"page": page_no, "detections": page_dets})

                            # If no by_page provided, fall back to top-level detections
                            if not by_page and getattr(res, "detections", None):
                                page_no = None
                                page_dets = []
                                for det in res.detections:
                                    if vlm_detect_tables_only and str(det.label).lower() != "table":
                                        continue
                                    page_dets.append({"label": det.label, "score": det.score, "bbox": det.bbox})
                                    chunk_text = f"Detected {det.label} ({det.score:.2f}) on page {page_no} at {det.bbox}"
                                    extra_chunks.append({
                                        "text": chunk_text,
                                        "start_char": None,
                                        "end_char": None,
                                        "chunk_type": "vlm" if str(det.label).lower() != "table" else "table",
                                        "metadata": {"page": page_no, "bbox": det.bbox, "label": det.label, "score": det.score},
                                    })
                                    vlm_summary["detections_total"] += 1
                                vlm_summary["by_page"].append({"page": page_no, "detections": page_dets})
                        except _PDF_NONCRITICAL_EXCEPTIONS as _pdf_vlm_err:
                            logging.warning(f"VLM document-level processing failed: {_pdf_vlm_err}")
                    else:
                        with pymupdf.open(path_for_processing) as doc:
                            total_pages = len(doc)
                            max_pages = min(vlm_max_pages or total_pages, total_pages)
                            for i, page in enumerate(doc, start=1):
                                if i > max_pages:
                                    break
                                # Render a medium-resolution bitmap
                                scale = 2.0  # ~144 DPI for detection
                                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                                img_bytes = pix.tobytes("png")
                                res = backend.process_image(img_bytes, context={"page": i, "pdf_path": path_for_processing})

                                page_dets = []
                                for det in res.detections:
                                    if vlm_detect_tables_only and str(det.label).lower() != "table":
                                        continue
                                    page_dets.append({
                                        "label": det.label,
                                        "score": det.score,
                                        "bbox": det.bbox,
                                    })
                                    # Add a compact text chunk for retrieval
                                    chunk_text = f"Detected {det.label} ({det.score:.2f}) on page {i} at {det.bbox}"
                                    extra_chunks.append({
                                        "text": chunk_text,
                                        "start_char": None,
                                        "end_char": None,
                                        "chunk_type": "vlm" if str(det.label).lower() != "table" else "table",
                                        "metadata": {"page": i, "bbox": det.bbox, "label": det.label, "score": det.score},
                                    })

                                vlm_summary["by_page"].append({"page": i, "detections": page_dets})
                                vlm_summary["pages_scanned"] += 1
                                vlm_summary["detections_total"] += len(page_dets)

                    result.setdefault("analysis_details", {})["vlm"] = vlm_summary
                    if extra_chunks:
                        result["extra_chunks"] = extra_chunks
        except _PDF_NONCRITICAL_EXCEPTIONS as vlm_err:
            logging.warning(f"VLM processing failed for {filename}: {vlm_err}")
            result["warnings"].append(f"VLM processing error: {vlm_err}")


        # Normalize the final extracted text (including OCR merge) before chunking.
        final_content = result.get("content")
        if isinstance(final_content, str):
            content = final_content
        if content and content.strip():
            try:
                normalized_content = normalize_pdf_text_for_storage(content)
                result["content"] = normalized_content
                content = normalized_content
                result.setdefault("analysis_details", {})["text_normalization"] = {
                    "applied": True,
                    "mode": "paragraph_safe",
                    "chars_before": len(final_content or ""),
                    "chars_after": len(normalized_content),
                    "line_breaks_before": (final_content or "").count("\n"),
                    "line_breaks_after": normalized_content.count("\n"),
                }
            except _PDF_NONCRITICAL_EXCEPTIONS as normalization_error:
                logging.warning(f"PDF text normalization failed for {filename}: {normalization_error}")
                result["warnings"] = (result.get("warnings") or []) + [
                    f"Text normalization failed: {normalization_error}"
                ]

        # --- Step 3: Chunking ---
        processed_chunks = None
        # Only proceed if text extraction was successful
        if content and perform_chunking:
            if chunk_options is None:
                # Provide sensible defaults if none are passed
                chunk_options = {'method': 'sentences', 'max_size': 500, 'overlap': 100}
            # Ensure a method is set, default to 'sentences' if missing
            chunk_options.setdefault('method', 'sentences')

            logging.info(f"Attempting chunking for {filename} with options: {chunk_options}")
            try:
                from tldw_Server_API.app.core.Chunking import improved_chunking_process
                processed_chunks = improved_chunking_process(content, chunk_options)

                if not processed_chunks:
                     logging.warning(f"Chunking produced no chunks for {filename}. Using full text as one chunk.")
                     # Create a single chunk containing the entire text
                     processed_chunks = [{'text': content, 'metadata': {'chunk_num': 0, 'start_index': 0, 'end_index': len(content)}}]
                     result["warnings"].append("Chunking yielded no results; using full text.")
                else:
                     logging.info(f"Chunking successful for {filename}. Total chunks created: {len(processed_chunks)}")
                     log_histogram("pdf_chunks_created", len(processed_chunks), labels={"file_name": filename})

                result["chunks"] = processed_chunks # Store the list of chunks

            except _PDF_NONCRITICAL_EXCEPTIONS as chunk_err:
                 logging.error(f"Chunking failed for {filename}: {chunk_err}", exc_info=True)
                 result["warnings"].append(f"Chunking failed: {str(chunk_err)}")
                 processed_chunks = [{'text': content, 'metadata': {'chunk_num': 0, 'error': f"Chunking failed: {chunk_err}"}}]
                 result["chunks"] = processed_chunks # Store the single chunk with error info

        elif content:
             # If not chunking, but text exists, create a single chunk for consistency
             processed_chunks = [{'text': content, 'metadata': {'chunk_num': 0}}]
             result["chunks"] = processed_chunks
             logging.info(f"Chunking disabled for {filename}. Using full text as one chunk.")
        else:
             # If text extraction failed, chunking cannot proceed
             logging.warning(f"Chunking skipped for {filename}: Text content is missing.")


        # --- Step 4: Summarization / Analysis ---
        # Use path_for_processing for logging context if needed
        logging.debug(f"PROCESS_PDF: Checking condition -> perform_analysis={perform_analysis}, api_name='{api_name}', api_key='{api_key}', chunks_exist={bool(processed_chunks)}") # Keep this log
        # Allow analysis to proceed without explicit api_key (resolved from server config)
        if perform_analysis and api_name and processed_chunks:
            logging.info(f"Summarization enabled for {len(processed_chunks)} chunks of {filename} using API: {api_name}.")
            log_counter("pdf_summarization_attempt", value=len(processed_chunks), labels={"file_name": filename, "api_name": api_name})

            chunk_summaries = []  # Store summaries of individual chunks
            summarized_chunks_for_result = [] # Store chunk data including the generated analysis

            # Iterate through each chunk generated earlier
            for i, chunk in enumerate(processed_chunks):
                chunk_text = chunk.get('text', '') # Get the text content of the chunk
                chunk_metadata: dict[str, Any] = chunk.get('metadata', {}) # Get existing metadata

                # Only summarize if the chunk has actual text content
                if chunk_text:
                    try:
                        # Call the external summarization library function
                        analysis_text = analyze(
                            api_name=api_name,
                            input_data=chunk_text,
                            custom_prompt_arg=custom_prompt, # User's custom prompt, if any
                            api_key=api_key,
                            recursive_summarization=False, # Summarize this single chunk first
                            temp=None, # Optional temperature parameter
                            system_message=system_prompt # Optional system prompt
                        )

                        # Check if the summarization returned a valid, non-empty string
                        if analysis_text and isinstance(analysis_text, str) and analysis_text.strip():
                            chunk_summaries.append(analysis_text)
                            # Add the generated analysis to the chunk's metadata
                            chunk_metadata['analysis'] = analysis_text
                            logging.debug(f"Summarized chunk {i+1}/{len(processed_chunks)} for {filename}.")
                        else:
                            # Summarization returned empty or invalid result
                            chunk_metadata['analysis'] = None # Indicate no analysis available
                            logging.debug(f"Summarization yielded empty result for chunk {i+1} of {filename}.")

                    except _PDF_NONCRITICAL_EXCEPTIONS as summ_err:
                        # Handle errors during the API call or summarization process
                        logging.warning(f"Summarization failed for chunk {i+1} of {filename}: {summ_err}", exc_info=True)
                        # Store error information in the chunk's metadata
                        chunk_metadata['analysis'] = f"[Summarization Error: {str(summ_err)}]"
                        # Add a warning to the overall result
                        result["warnings"] = (result["warnings"] or []) + [f"Summarization failed for chunk {i+1}: {str(summ_err)}"]
                else:
                    # Chunk had no text to summarize
                    chunk_metadata['analysis'] = None
                    logging.debug(f"Skipping summarization for empty chunk {i+1} of {filename}.")

                # Update the chunk with potentially modified metadata
                chunk['metadata'] = chunk_metadata
                # Add the chunk (with or without analysis metadata) to the list for the final result
                summarized_chunks_for_result.append(chunk)

            # Update the main result dictionary with the chunks containing analysis metadata
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
                        final_summary = analyze(
                            api_name=api_name,
                            input_data=combined_summaries_text,
                            # Use the original custom prompt, or a default recursive prompt if none provided
                            custom_prompt_arg=custom_prompt or "Provide a concise overall analysis of the preceding text sections.",
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

                    except _PDF_NONCRITICAL_EXCEPTIONS as rec_summ_err:
                        # Handle errors during the recursive summarization step
                        logging.error(f"Recursive summarization failed for {filename}: {rec_summ_err}", exc_info=True)
                        # Fallback: Use the joined chunk summaries as the final analysis, but mark the error
                        final_summary = f"[Recursive Summarization Error: {str(rec_summ_err)}]\n\n" + combined_summaries_text
                        result["warnings"] = (result["warnings"] or []) + [f"Recursive summarization failed: {str(rec_summ_err)}"]
                        log_counter("pdf_recursive_summarization_error", labels={"file_name": filename, "error": str(rec_summ_err)})

                else:
                    # Not recursive, or only one chunk analysis: simply join them
                    final_summary = "\n\n---\n\n".join(chunk_summaries)
                    if len(chunk_summaries) > 1 :
                         logging.info(f"Combined {len(chunk_summaries)} chunk summaries for {filename} (non-recursive).")
                    else:
                         logging.info(f"Using single chunk analysis as final analysis for {filename}.")


            # Store the final generated summary (or None if none was generated)
            result["analysis"] = final_summary
            log_counter("pdf_chunks_summarized", value=len(chunk_summaries), labels={"file_name": filename})
            logging.info(f"Summarization processing completed for {filename}.")

        # --- Log reasons if summarization was skipped ---
        elif not perform_analysis:
             logging.info(f"Summarization disabled by 'perform_analysis=False' for {filename}.")
        elif not api_name or not api_key:
             logging.warning(f"Summarization skipped for {filename}: API name or key not provided.")
        elif not processed_chunks:
             # This case covers both chunking disabled and chunking failed/yielded no results
             logging.warning(f"Summarization skipped for {filename}: No processable chunks available (text extraction failed or chunking failed/disabled).")
        else:
            logging.warning(f"Summarization skipped for {filename} due to an unknown condition.")


        # --- Step 5: Determine Final Status (Based on content and warnings) ---
        # Check if critical step (text extraction) failed. Check warnings for specific errors.
        extraction_failed = not content and any("Text extraction failed" in w for w in result["warnings"])
        # Treat metadata failures as warnings unless text extraction also failed

        if extraction_failed:
            result["status"] = "Error"
            # Set a primary error message if not already set by a later exception
            primary_error_msg = "PDF Extraction Error."
            result["error"] = result["error"] or primary_error_msg
            logging.warning(f"Setting status to Error for {filename} due to critical extraction failure.")
        elif result["warnings"]:
             # If there were warnings but text was extracted, status is Warning
             result["status"] = "Warning"
             logging.info(f"Setting status to Warning for {filename} due to non-critical warnings.")
        else:
             # No errors or warnings encountered
             result["status"] = "Success"
             logging.info(f"Setting status to Success for {filename}.")

    # --- Main Exception Handler ---
    except FileNotFoundError as fnf_err:
        logging.error(f"File not found error for {filename}: {fnf_err}", exc_info=True)
        result["status"] = "Error"
        result["error"] = str(fnf_err)
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": "FileNotFoundError"})
    except OSError as io_err: # Catch temp file creation errors
        logging.error(f"IO error during temp file handling for {filename}: {io_err}", exc_info=True)
        result["status"] = "Error"
        result["error"] = f"Temporary file error: {io_err}"
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": "IOError"})
    # --- Catch PDF library errors that indicate fundamental file issues (but weren't caught during specific steps) ---
    except (RuntimeError, pymupdf.FileDataError, pymupdf.EmptyFileError) as pdf_lib_err:
        # --- MODIFICATION END ---
        # Check the message specifically for password errors if needed for logging differentiation
        err_msg = str(pdf_lib_err)
        # Distinguish error types for logging and user messages
        if "password" in err_msg.lower():
            log_msg = f"PDF password error for {filename}: {err_msg}"
            err_type_label = "PasswordError"  # Specific label for metrics
            result["error"] = "PDF Error: Password required or invalid."  # User-friendly message
        elif isinstance(pdf_lib_err, pymupdf.EmptyFileError):
            log_msg = f"PDF empty file error for {filename}: {err_msg}"
            err_type_label = "EmptyFileError"
            result["error"] = "PDF Error: Input file is empty."
        elif isinstance(pdf_lib_err, pymupdf.FileDataError):
            log_msg = f"PDF file data error for {filename}: {err_msg}"
            err_type_label = "FileDataError"
            result["error"] = "PDF Error: Corrupted or invalid file data."
        else:  # General RuntimeError or other caught types
            log_msg = f"PDF library runtime error for {filename}: {err_msg}"
            err_type_label = type(pdf_lib_err).__name__  # Use 'RuntimeError' usually
            logging.error(f"PDF library error processing {filename}: {result['error']}", exc_info=True)


        logging.error(log_msg, exc_info=True)
        result["status"] = "Error"
        # Use the determined err_type_label for consistent metrics
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": err_type_label})
        current_status_before_cleanup = result["status"] # Store status before cleanup attempt

    except _PDF_NONCRITICAL_EXCEPTIONS as e:
        # Catch any other unexpected exceptions
        logging.error(f"Unexpected error processing PDF {filename}: {str(e)}", exc_info=True)
        result["status"] = "Error"
        # Ensure error field is populated
        result["error"] = result["error"] or f"Unexpected error: {str(e)}"
        current_status_before_cleanup = "Error" # Ensure this reflects the error
        log_counter("pdf_processing_error", labels={"file_name": filename, "parser": parser, "error": type(e).__name__})

    # --- Finally Block: Cleanup ---
    finally:
        current_status_before_cleanup = result["status"]

        if path_for_processing and temp_dir_for_pdf and os.path.exists(path_for_processing):
            try:
                # --- Optional: Explicitly close handles via garbage collection ---
                # This can sometimes help if objects holding handles are lingering.
                logging.debug(f"Triggering garbage collection before file removal for {path_for_processing}")
                gc.collect()
                time.sleep(0.1) # Short delay after GC

                logging.debug(f"Attempting to remove temporary file: {path_for_processing}")
                os.remove(path_for_processing)
                logging.debug(f"Successfully removed temporary file: {path_for_processing}")
                time.sleep(0.1) # Small delay AFTER file removal before dir removal

            except OSError as file_rm_err:
                 logging.warning(f"OSError removing temporary file {path_for_processing}: {file_rm_err}")
                 result["warnings"].append(f"Failed to cleanup temp file: {file_rm_err}")
            except _PDF_NONCRITICAL_EXCEPTIONS as file_rm_exc:
                 logging.error(f"Unexpected error removing temporary file {path_for_processing}: {file_rm_exc}", exc_info=True)
                 result["warnings"].append(f"Unexpected error cleaning up temp file: {file_rm_exc}")

        # --- Now attempt to remove the directory ---
        if temp_dir_for_pdf and os.path.isdir(temp_dir_for_pdf):
             max_retries = 4
             retry_delay = 0.5 # Slightly increase delay

             for attempt in range(max_retries):
                 try:
                     logging.debug(f"Attempting to remove temporary directory (Attempt {attempt + 1}/{max_retries}): {temp_dir_for_pdf}")
                     shutil.rmtree(temp_dir_for_pdf)
                     logging.debug(f"Successfully removed temporary directory: {temp_dir_for_pdf}")
                     break # Exit loop if successful

                 except OSError as rm_err:
                     logging.warning(f"OSError removing temporary directory (Attempt {attempt + 1}/{max_retries}) {temp_dir_for_pdf}: {rm_err}")
                     if attempt == max_retries - 1:
                         logging.error(f"Final attempt failed to remove {temp_dir_for_pdf}: {rm_err}", exc_info=False)
                         # --- Modify status handling ---
                         warning_msg = f"Failed to cleanup temp dir after {max_retries} attempts: {rm_err}"
                         result["warnings"].append(warning_msg)
                         # Use the correctly initialized variable here
                         if current_status_before_cleanup == "Success":
                            logging.warning(f"Downgrading status to Warning due to failed temp dir cleanup for {temp_dir_for_pdf}")
                            result["status"] = "Warning"
                         else:
                            logging.warning(f"Temp dir cleanup failed, but original status was already {current_status_before_cleanup}. Keeping status.")
                         # --- End modify status handling ---
                     else:
                         logging.info("Retrying temp dir removal after delay...")
                         time.sleep(retry_delay * (attempt + 1))

                 except _PDF_NONCRITICAL_EXCEPTIONS as rm_exc:
                      logging.error(f"Unexpected error removing temporary directory {temp_dir_for_pdf} (Attempt {attempt + 1}): {rm_exc}", exc_info=True)
                      warning_msg = f"Unexpected error cleaning up temp dir: {rm_exc}"
                      result["warnings"] = (result["warnings"] or []) + [warning_msg]
                      # Only downgrade if original status was Success
                      if current_status_before_cleanup == "Success":
                         logging.warning(f"Downgrading status to Warning due to unexpected cleanup error for {temp_dir_for_pdf}")
                         result["status"] = "Warning"
                      else:
                         logging.warning(f"Temp dir cleanup failed unexpectedly, but original status was already {current_status_before_cleanup}. Keeping status.")
                      break # Don't retry on unexpected errors
        elif temp_dir_for_pdf:
             # Log if dir path exists but isn't a dir (shouldn't happen often)
             if not os.path.exists(temp_dir_for_pdf):
                 logging.debug(f"Temporary directory {temp_dir_for_pdf} did not exist for cleanup.")
             else:
                 logging.warning(f"Temporary directory path {temp_dir_for_pdf} exists but is not a directory.")
        else:
             logging.debug("No specific temporary directory was created by process_pdf, no cleanup needed by process_pdf.")

    # --- Final Logging and Return ---
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds() # Calculate duration as seconds
    log_histogram("pdf_processing_duration", processing_time, labels={"file_name": filename, "parser": result['parser_used'], "status": result["status"]})
    # Log success or final error/warning status
    if result["status"] == "Success":
        log_counter("pdf_processing_success", labels={"file_name": filename, "parser": result['parser_used']})
        logging.info(f"Successfully processed PDF: {filename} (Parser: {result['parser_used']}) in {processing_time:.2f}s")
    elif result["status"] == "Warning":
        log_counter("pdf_processing_warning", labels={"file_name": filename, "parser": result['parser_used']}) # Add warning counter
        logging.warning(f"Processed PDF with warnings: {filename} (Parser: {result['parser_used']}) in {processing_time:.2f}s. Warnings: {result['warnings']}")
    else: # Error status
        # Error counter is logged within the except blocks where the error type is known
        logging.error(f"Failed to process PDF: {filename} (Parser: {result['parser_used']}) in {processing_time:.2f}s. Error: {result.get('error', 'Unknown')}")


    # Ensure warnings list is None if empty
    if isinstance(result.get("warnings"), list) and not result["warnings"]:
        result["warnings"] = None

    return result


async def process_pdf_task(
    file_bytes: bytes,
    filename: str,
    parser: str = "pymupdf4llm",
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    perform_chunking: bool = True,
    chunk_method: Optional[str] = None,
    max_chunk_size: Optional[int] = 500,
    chunk_overlap: Optional[int] = 100,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    summarize_recursively: bool = False,
    # OCR options
    enable_ocr: bool = False,
    ocr_backend: Optional[str] = None,
    ocr_lang: Optional[str] = "eng",
    ocr_dpi: int = 300,
    ocr_mode: Optional[str] = "fallback",
    ocr_min_page_text_chars: int = 40,
    ocr_output_format: Optional[str] = None,
    ocr_prompt_preset: Optional[str] = None,
    # VLM options
    enable_vlm: bool = False,
    vlm_backend: Optional[str] = None,
    vlm_detect_tables_only: bool = True,
    vlm_max_pages: Optional[int] = None,
) -> dict[str, Any]:
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
                # Add other chunk params if needed by process_pdf's chunk_options
            }

        # Call the synchronous core processing function
        # process_pdf now handles the byte input correctly by creating a temp file
        result_dict = await asyncio.to_thread(
            process_pdf,
            file_input=file_bytes,  # Pass bytes directly
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
            enable_ocr=enable_ocr,
            ocr_backend=ocr_backend,
            ocr_lang=ocr_lang,
            ocr_dpi=ocr_dpi,
            ocr_mode=ocr_mode,
            ocr_min_page_text_chars=ocr_min_page_text_chars,
            ocr_output_format=ocr_output_format,
            ocr_prompt_preset=ocr_prompt_preset,
            enable_vlm=enable_vlm,
            vlm_backend=vlm_backend,
            vlm_detect_tables_only=vlm_detect_tables_only,
            vlm_max_pages=vlm_max_pages,
            # No need to pass write_to_temp_file
        )

        logging.info(f"process_pdf_task completed for {filename} with status: {result_dict.get('status')}")
        return result_dict

    except _PDF_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error within process_pdf_task for {filename}: {str(e)}", exc_info=True)
        # Return a standard error dictionary matching process_pdf's structure
        return {
            "status": "Error",
            "input_ref": filename,
            "processing_source": "bytes_input_task_error",
            "media_type": "pdf",
            "parser_used": parser,
            "error": f"Task-level error: {str(e)}",
            "content": None, "metadata": None, "chunks": None, "analysis": None,
            "keywords": keywords or [], "warnings": None,
            # Add analysis_details field for consistency if needed
            "analysis_details": {}
        }

#
# End of PDF_Ingestion_Lib.py
#######################################################################################################################


def _ocr_pdf_pages(
    pdf_path: str,
    lang: str,
    dpi: int,
    backend,
    per_page_min_text: int = 40,
    per_page_check: bool = True,
    concurrency: int = 1,
    output_format: Optional[str] = None,
    prompt_preset: Optional[str] = None,
) -> tuple[str, int, int, Optional[list[dict[str, Any]]]]:
    """
    Render PDF pages to images and run OCR.

    Returns: (markdown_text, total_pages, ocr_pages_count, structured_pages)
    """
    text_by_index: list[str] = []
    ocr_pages = 0
    with pymupdf.open(pdf_path) as doc:
        page_count = len(doc)
        text_by_index = [""] * page_count
        structured_pages: Optional[list[dict[str, Any]]] = None
        supports_structured = False
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import (
                OCRBackend as _OCRBackend,
            )
            supports_structured = (
                getattr(backend.__class__, "ocr_image_structured", None)
                is not getattr(_OCRBackend, "ocr_image_structured", None)
            )
        except _PDF_NONCRITICAL_EXCEPTIONS:
            supports_structured = False

        # Persist structured OCR outputs whenever a backend provides them,
        # even if the caller did not explicitly request structured output.
        persist_structured = (
            supports_structured
            or bool(prompt_preset)
            or (
                output_format is not None
                and str(output_format).strip().lower() not in ("", "text", "auto")
            )
        )
        use_structured = persist_structured
        if persist_structured:
            structured_pages = [None] * page_count  # type: ignore[list-item]
        scale = max(dpi, 72) / 72.0

        # Render pages sequentially (PyMuPDF doc is not thread-safe),
        # but dispatch OCR requests in a small thread pool to overlap I/O.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        concurrency = max(1, int(concurrency))
        futures = []
        idx_map = {}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for idx, page in enumerate(doc, start=1):
                do_ocr = True
                if per_page_check:
                    try:
                        pre_text = page.get_text("text") or ""
                        if len(pre_text.strip()) >= max(per_page_min_text, 1):
                            text_by_index[idx - 1] = f"## Page {idx}\n\n{pre_text.strip()}\n\n---\n"
                            if use_structured:
                                try:
                                    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import OCRResult
                                    structured_pages[idx - 1] = OCRResult(  # type: ignore[index]
                                        text=pre_text.strip(),
                                        format="text",
                                        meta={"source": "pdf_text"},
                                    ).as_dict()
                                except _PDF_NONCRITICAL_EXCEPTIONS:
                                    pass
                            do_ocr = False
                    except _PDF_NONCRITICAL_EXCEPTIONS:
                        do_ocr = True

                if do_ocr:
                    mat = pymupdf.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img_bytes = pix.tobytes("png")
                    if use_structured and hasattr(backend, "ocr_image_structured"):
                        def _call_structured(b: bytes) -> Any:
                            try:
                                return backend.ocr_image_structured(b, lang, output_format, prompt_preset)
                            except TypeError:
                                return backend.ocr_image_structured(b, lang)

                        fut = pool.submit(_call_structured, img_bytes)
                    else:
                        fut = pool.submit(backend.ocr_image, img_bytes, lang)
                    futures.append(fut)
                    idx_map[fut] = idx

            for fut in as_completed(futures):
                result = fut.result()
                idx = idx_map.get(fut)
                page_text = ""
                if use_structured and result is not None:
                    try:
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import OCRResult
                        if isinstance(result, OCRResult):
                            page_text = result.text or ""
                            if structured_pages is not None:
                                structured_pages[idx - 1] = result.as_dict()  # type: ignore[index]
                        elif isinstance(result, tuple) and len(result) == 2:
                            page_text = str(result[0] or "")
                            if structured_pages is not None and isinstance(result[1], dict):
                                structured_pages[idx - 1] = result[1]  # type: ignore[index]
                        elif isinstance(result, dict):
                            page_text = str(result.get("text") or "")
                            if structured_pages is not None:
                                structured_pages[idx - 1] = result  # type: ignore[index]
                        else:
                            page_text = str(result or "")
                    except _PDF_NONCRITICAL_EXCEPTIONS:
                        page_text = str(result or "")
                else:
                    page_text = str(result or "")
                if page_text.strip():
                    ocr_pages += 1
                text_by_index[idx - 1] = f"## Page {idx}\n\n{page_text.strip()}\n\n---\n"

    return ("".join(text_by_index).strip(), page_count, ocr_pages, structured_pages)


def _run_mineru_document_ocr(
    *,
    pdf_path: Path,
    output_format: str | None = None,
    prompt_preset: str | None = None,
    requested_lang: str | None = None,
    requested_dpi: int | None = None,
) -> dict[str, Any]:
    """Run the optional MinerU adapter with a local import boundary for PDF OCR."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.mineru_adapter import (
        run_mineru_document_ocr,
    )

    return run_mineru_document_ocr(
        pdf_path=pdf_path,
        output_format=output_format,
        prompt_preset=prompt_preset,
        requested_lang=requested_lang,
        requested_dpi=requested_dpi,
    )
