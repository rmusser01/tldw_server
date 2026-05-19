# XML_Ingestion.py
# Description: This file contains functions for reading and writing XML files.
# Imports
from pathlib import Path
from typing import Optional

try:  # Prefer hardened XML parser
    from defusedxml import ElementTree as DET  # type: ignore
    _DEFUSED_AVAILABLE = True
except Exception:  # pragma: no cover - defusedxml is an optional dependency
    DET = None  # type: ignore
    _DEFUSED_AVAILABLE = False
#
# External Imports
#
# Local Imports
from tldw_Server_API.app.core.Chunking import improved_chunking_process
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_media_repository,
    managed_media_database,
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Utils.Utils import logging

#
#######################################################################################################################
#
# Functions:


def _read_xml_source(import_file) -> tuple[str, str]:
    """
    Return XML text and a display name from a variety of upload types.
    Supports file paths, FastAPI UploadFile objects, and generic file-like objects.
    """
    display_name: Optional[str] = getattr(import_file, "filename", None) or getattr(import_file, "name", None)

    if isinstance(import_file, (str, Path)):
        path = Path(import_file)
        display_name = display_name or path.name
        xml_text = path.read_text(encoding="utf-8", errors="ignore")
        return xml_text, display_name or "uploaded.xml"

    file_like = getattr(import_file, "file", None)
    if file_like is not None:
        try:
            current_pos = file_like.tell()
        except Exception:
            current_pos = None
        try:
            if current_pos is not None:
                file_like.seek(0)
            raw = file_like.read()
        finally:
            if current_pos is not None:
                file_like.seek(current_pos)
        xml_text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return xml_text, display_name or "uploaded.xml"

    if hasattr(import_file, "read"):
        raw = import_file.read()
        xml_text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return xml_text, display_name or "uploaded.xml"

    raise ValueError("Unsupported XML input type; expected path, file-like, or UploadFile.")


def _ensure_defusedxml():
    """
    Ensure a safe XML parser is available.

    Raises:
        RuntimeError: if defusedxml is not installed.
    """
    if not _DEFUSED_AVAILABLE:
        raise RuntimeError(
            "Secure XML ingestion requires the 'defusedxml' package. "
            "Install it to enable XML parsing."
        )


def _parse_xml_string(xml_text: str):
    """Parse XML safely, leveraging defusedxml."""
    _ensure_defusedxml()
    return DET.fromstring(xml_text)  # type: ignore[union-attr]


def _parse_xml_file(xml_file: str):
    """Parse XML file safely, leveraging defusedxml."""
    _ensure_defusedxml()
    return DET.parse(xml_file)  # type: ignore[union-attr]


def xml_to_text(xml_file):
    try:
        tree = _parse_xml_file(xml_file)
        root = tree.getroot()
        # Extract text content recursively
        text_content = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_content.append(elem.text.strip())
        return '\n'.join(text_content)
    except Exception:
        logging.exception("Error parsing XML file")
        return None


def import_xml_handler(import_file, title, author, keywords, system_prompt,
                       custom_prompt, auto_summarize, api_name, api_key):
    if not import_file:
        return "Please upload an XML file"

    try:
        xml_text, display_name = _read_xml_source(import_file)
        # Parse XML and extract text with structure
        root = _parse_xml_string(xml_text)

        # Create chunk options
        chunk_options = {
            'method': 'xml',
            'max_size': 1000,  # Adjust as needed
            'overlap': 200,  # Adjust as needed
            'language': 'english'  # Add language detection if needed
        }

        # Use improved_chunking_process with xml method to get structured chunks
        chunk_options['method'] = 'xml'
        chunks = improved_chunking_process(DET.tostring(root, encoding='unicode'), chunk_options)  # type: ignore[union-attr]

        # Convert chunks to segments format expected by add_media_with_keywords
        segments = []
        for chunk in chunks:
            segment = {
                'Text': chunk['text'],
                'metadata': chunk['metadata']  # Preserve XML structure metadata
            }
            segments.append(segment)

        # Create info_dict
        info_dict = {
            'title': title or 'Untitled XML Document',
            'uploader': author or 'Unknown',
            'file_type': 'xml',
            'structure': root.tag  # Save root element type
        }

        # Process keywords
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()] if keywords else []

        # Handle summarization
        if auto_summarize and api_name and api_key:
            # Combine all chunks for summarization
            full_text = '\n'.join(chunk['text'] for chunk in chunks)
            summary = analyze(api_name, full_text, custom_prompt, api_key)
        else:
            summary = "No summary provided"

        # Add to database (ensure we have a MediaDatabase instance)
        with managed_media_database(
            client_id="xml_import",
            initialize=False,
        ) as db_instance:
            result = get_media_repository(db_instance).add_media_with_keywords(
                db_instance=db_instance,
                url=display_name,  # Using filename as URL
                info_dict=info_dict,
                segments=segments,
                summary=summary,
                keywords=keyword_list,
                custom_prompt_input=custom_prompt,
                whisper_model="XML Import",
                media_type="xml_document",
                overwrite=False
            )

    except getattr(DET, "ParseError", ValueError):  # type: ignore[arg-type]
        logging.exception("XML parsing error")
        return "Error parsing XML file"
    except Exception as e:
        logging.error(f"Error processing XML file: {str(e)}")
        return "Error processing XML file"
    else:
        return f"XML file '{display_name}' import complete. Database result: {result}"

#
# End of XML_Ingestion_Lib.py
#######################################################################################################################
