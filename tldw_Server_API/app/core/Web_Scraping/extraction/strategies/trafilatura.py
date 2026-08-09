"""Direct Trafilatura article extraction strategy."""

import re
from typing import Any

import trafilatura

from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
from tldw_Server_API.app.core.Web_Scraping.extraction.metrics import emit_global_counter as log_counter

DEFAULT_BOILERPLATE_PATTERNS = [
    r"\bsubscribe\s+now\b",
    r"\bsubscribe\s+today\b",
    r"\bsign\s+up\b",
    r"\bshare\s+this\b",
    r"\bshare\s+on\s+(facebook|twitter|linkedin|reddit)\b",
    r"\bfollow\s+us\b",
    r"\bnewsletter\b",
    r"\bread\s+more\b",
    r"\bthanks\s+for\s+reading\b",
]

_BOILERPLATE_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in DEFAULT_BOILERPLATE_PATTERNS]


def _strip_boilerplate_sections(text: str) -> str:
    """Remove common boilerplate phrases from extracted article text."""
    if not text:
        return text

    def _is_boilerplate(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and any(regex.search(stripped) for regex in _BOILERPLATE_REGEXES)

    lines = [line for line in text.splitlines() if not _is_boilerplate(line)]
    collapsed: list[str] = []
    previous_blank = False
    for line in lines:
        if line.strip():
            collapsed.append(line)
            previous_blank = False
        elif not previous_blank:
            collapsed.append(line)
            previous_blank = True
    return "\n".join(collapsed)


def extract_with_trafilatura(html: str, url: str) -> dict[str, Any]:
    """Extract article metadata and body from raw HTML for the direct pipeline."""
    logging.info("Extracting article data from HTML")
    downloaded = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_images=False,
    )
    downloaded = _strip_boilerplate_sections(downloaded)
    metadata = trafilatura.extract_metadata(html)

    result: dict[str, Any] = {
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "url": url,
        "extraction_successful": False,
    }
    if downloaded:
        logging.info("Content extracted successfully")
        log_counter("article_extracted", labels={"success": "true"})
        result["content"] = ContentMetadataHandler.format_content_with_metadata(
            url=url,
            content=downloaded,
            pipeline="Trafilatura",
            additional_metadata={
                "extracted_date": metadata.date if metadata and metadata.date else "N/A",
                "author": metadata.author if metadata and metadata.author else "N/A",
            },
        )
        result["extraction_successful"] = True
    else:
        log_counter("article_extracted", labels={"success": "false"})
        logging.warning("Content extraction failed")

    if metadata:
        result.update(
            {
                "title": metadata.title if metadata.title else "N/A",
                "author": metadata.author if metadata.author else "N/A",
                "date": metadata.date if metadata.date else "N/A",
            }
        )
    else:
        logging.warning("Metadata extraction failed")
    return result
