from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

HandlerFunc = Callable[[str, str], dict[str, Any]]


def handle_generic_html(html: str, url: str) -> dict[str, Any]:
    """Default handler: extract article metadata and convert content to Markdown."""
    from tldw_Server_API.app.core.Web_Scraping.content import convert_html_to_markdown
    from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_data_from_html

    data = extract_article_data_from_html(html, url)
    if data.get("extraction_successful") and data.get("content"):
        data["content"] = convert_html_to_markdown(data["content"])
    return data


def resolve_handler(handler_path: str) -> HandlerFunc:
    """Resolve a handler import string to a callable, falling back to the default."""
    if not handler_path or ":" not in handler_path:
        return handle_generic_html
    module_path, func_name = handler_path.split(":", 1)
    if not module_path or not func_name:
        return handle_generic_html
    try:
        module = import_module(module_path)
        func = getattr(module, func_name, None)
        if callable(func):
            return func
    except Exception as resolver_error:
        _ = resolver_error  # fallback to generic handler
    return handle_generic_html


__all__ = [
    "handle_generic_html",
    "resolve_handler",
    "HandlerFunc",
]
