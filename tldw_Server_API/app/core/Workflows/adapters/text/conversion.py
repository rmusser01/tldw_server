"""Text conversion adapters.

This module includes adapters for format conversion operations:
- html_to_markdown: Convert HTML to Markdown
- markdown_to_html: Convert Markdown to HTML
- csv_to_json: Convert CSV to JSON
- json_to_csv: Convert JSON to CSV
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.text._config import (
    CSVToJSONConfig,
    HTMLToMarkdownConfig,
    JSONToCSVConfig,
    MarkdownToHTMLConfig,
)


@registry.register(
    "html_to_markdown",
    category="text",
    description="Convert HTML to Markdown",
    parallelizable=True,
    tags=["text", "conversion"],
    config_model=HTMLToMarkdownConfig,
)
async def run_html_to_markdown_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Convert HTML content to Markdown format.

    Config:
      - html: str (templated) - HTML content to convert
      - strip_tags: list[str] (optional) - Tags to remove
      - preserve_links: bool = True - Keep hyperlinks
    Output:
      - {"markdown": str, "text": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    html_text = config.get("html") or config.get("text") or ""
    if isinstance(html_text, str):
        html_text = _tmpl(html_text, context) or html_text

    if not html_text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            html_text = prev.get("html") or prev.get("text") or prev.get("content") or ""

    try:
        from markdownify import markdownify as md
        markdown_text = md(html_text, strip=config.get("strip_tags") or ["script", "style"])
        return {"markdown": markdown_text, "text": markdown_text}
    except ImportError:
        # Fallback: basic tag stripping
        text = re.sub(r"<script[^>]*>.*?</script>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return {"markdown": text.strip(), "text": text.strip(), "warning": "markdownify_not_installed"}


@registry.register(
    "markdown_to_html",
    category="text",
    description="Convert Markdown to HTML",
    parallelizable=True,
    tags=["text", "conversion"],
    config_model=MarkdownToHTMLConfig,
)
async def run_markdown_to_html_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Convert Markdown content to HTML format.

    Config:
      - markdown: str (templated) - Markdown content to convert
      - extensions: list[str] (optional) - Markdown extensions to use
      - safe_mode: bool = True - Sanitize output
    Output:
      - {"html": str, "text": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    markdown_text = config.get("markdown") or config.get("text") or ""
    if isinstance(markdown_text, str):
        markdown_text = _tmpl(markdown_text, context) or markdown_text

    if not markdown_text:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            markdown_text = prev.get("text") or prev.get("content") or prev.get("markdown") or ""

    extensions = config.get("extensions") or ["tables", "fenced_code"]

    try:
        import markdown
        html = markdown.markdown(markdown_text, extensions=extensions)
        return {"html": html, "text": html}
    except ImportError:
        # Fallback: basic conversion
        html = markdown_text.replace("\n\n", "</p><p>").replace("\n", "<br>")
        html = f"<p>{html}</p>"
        return {"html": html, "text": html, "warning": "markdown_library_not_installed"}


@registry.register(
    "csv_to_json",
    category="text",
    description="Convert CSV to JSON",
    parallelizable=True,
    tags=["text", "conversion"],
    config_model=CSVToJSONConfig,
)
async def run_csv_to_json_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Convert CSV data to JSON format.

    Config:
      - csv: str (templated) - CSV content to convert
      - delimiter: str = "," - CSV delimiter
      - has_header: bool = True - First row is header
    Output:
      - {"json": list[dict], "rows": int, "columns": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    csv_data = config.get("csv_data") or config.get("data") or ""
    if isinstance(csv_data, str):
        csv_data = _tmpl(csv_data, context) or csv_data

    if not csv_data:
        prev = context.get("prev") or context.get("last") or {}
        csv_data = prev.get("text") or prev.get("content") or "" if isinstance(prev, dict) else ""

    if not csv_data:
        return {"error": "missing_csv_data", "records": [], "count": 0}

    delimiter = config.get("delimiter", ",")
    has_header = config.get("has_header", True)

    try:
        reader = csv.reader(io.StringIO(csv_data), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return {"records": [], "count": 0}

        if has_header:
            headers = rows[0]
            records = [dict(zip(headers, row)) for row in rows[1:]]
        else:
            records = [{"col_" + str(i): v for i, v in enumerate(row)} for row in rows]

        return {"records": records, "count": len(records), "columns": headers if has_header else None}
    except Exception:
        logger.exception("CSV to JSON error")
        return {"error": "csv_to_json_error", "records": [], "count": 0}


@registry.register(
    "json_to_csv",
    category="text",
    description="Convert JSON to CSV",
    parallelizable=True,
    tags=["text", "conversion"],
    config_model=JSONToCSVConfig,
)
async def run_json_to_csv_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON data to CSV format.

    Config:
      - json: list[dict] (templated) - JSON data to convert
      - delimiter: str = "," - CSV delimiter
      - include_header: bool = True - Include header row
    Output:
      - {"csv": str, "rows": int, "columns": int}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    records = config.get("records") or config.get("data")
    if records is None:
        prev = context.get("prev") or context.get("last") or {}
        records = prev.get("records") or prev.get("data") or (prev if isinstance(prev, list) else [])

    if not isinstance(records, list) or not records:
        return {"error": "missing_records", "csv": "", "count": 0}

    delimiter = config.get("delimiter", ",")
    include_header = config.get("include_header", True)

    try:
        output = io.StringIO()
        if records and isinstance(records[0], dict):
            headers = list(records[0].keys())
            writer = csv.DictWriter(output, fieldnames=headers, delimiter=delimiter)
            if include_header:
                writer.writeheader()
            writer.writerows(records)
        else:
            writer = csv.writer(output, delimiter=delimiter)
            writer.writerows(records)

        csv_data = output.getvalue()
        return {"csv": csv_data, "count": len(records)}
    except Exception:
        logger.exception("JSON to CSV error")
        return {"error": "json_to_csv_error", "csv": "", "count": 0}
