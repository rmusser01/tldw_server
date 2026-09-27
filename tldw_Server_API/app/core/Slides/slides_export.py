"""Export utilities for Slides presentations."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from functools import partial
from html import escape
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import bleach  # type: ignore
except ImportError:  # pragma: no cover - bleach is a declared dependency
    bleach = None

try:
    from bleach.css_sanitizer import CSSSanitizer  # type: ignore
except ImportError:  # pragma: no cover - CSS sanitizer depends on optional tinycss2
    CSSSanitizer = None  # type: ignore

try:
    import tinycss2  # type: ignore
except ImportError:  # pragma: no cover - direct dependency is smoke tested elsewhere
    tinycss2 = None  # type: ignore

try:
    import markdown  # type: ignore
except Exception as exc:  # pragma: no cover - markdown is a declared dependency
    raise RuntimeError("markdown package is required for slides export") from exc

try:
    from playwright.sync_api import Error as PlaywrightError  # type: ignore
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # type: ignore
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover - playwright is an optional dependency
    sync_playwright = None
    PlaywrightError = Exception  # type: ignore
    PlaywrightTimeoutError = TimeoutError  # type: ignore

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.Slides.slides_assets import (
    MAX_RESOLVED_SLIDE_ASSET_BYTES,
    SlidesAssetError,
    resolve_slide_asset,
)
from tldw_Server_API.app.core.Slides.slides_images import SlidesImageError, validate_images_payload
from tldw_Server_API.app.core.Slides.visual_style_packs import render_pack_custom_css
from tldw_Server_API.app.core.testing import is_truthy


class SlidesExportError(Exception):
    """Base exception for slides export errors."""


class SlidesAssetsMissingError(SlidesExportError):
    """Raised when Reveal.js assets are missing."""


class SlidesExportInputError(SlidesExportError):
    """Raised for invalid export inputs."""


_REVEAL_THEME_TO_MARP = {
    "black": "default",
    "white": "default",
    "league": "gaia",
    "beige": "gaia",
    "sky": "uncover",
    "night": "uncover",
    "serif": "gaia",
    "simple": "default",
    "solarized": "gaia",
    "blood": "uncover",
    "moon": "uncover",
    "dracula": "uncover",
}

_ALLOWED_HTML_TAGS = [
    "a",
    "blockquote",
    "br",
    "code",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
]

_ALLOWED_HTML_ATTRS = {
    "a": ["href", "title", "rel", "target"],
    "code": ["class"],
    "pre": ["class"],
    "h1": ["id"],
    "h2": ["id"],
    "h3": ["id"],
    "h4": ["id"],
    "h5": ["id"],
    "h6": ["id"],
}

_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]

_ALLOWED_CSS_PROPERTIES = [
    "--accent",
    "--border",
    "--glow",
    "--grid",
    "--pixel",
    "--rule",
    "--shadow",
    "--surface",
    "--text",
    "--visual-style-pack",
    "align-content",
    "align-items",
    "align-self",
    "background",
    "background-color",
    "backdrop-filter",
    "border",
    "border-collapse",
    "border-color",
    "border-radius",
    "border-style",
    "border-width",
    "box-shadow",
    "color",
    "display",
    "flex",
    "flex-direction",
    "flex-grow",
    "flex-wrap",
    "font",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "height",
    "justify-content",
    "letter-spacing",
    "line-height",
    "margin",
    "margin-bottom",
    "margin-left",
    "margin-right",
    "margin-top",
    "max-height",
    "max-width",
    "min-height",
    "min-width",
    "opacity",
    "overflow",
    "padding",
    "padding-bottom",
    "padding-left",
    "padding-right",
    "padding-top",
    "position",
    "text-align",
    "text-decoration",
    "text-transform",
    "text-shadow",
    "top",
    "right",
    "bottom",
    "left",
    "width",
]
_CSS_FUNCTION_TYPE = "function"

_PDF_FORMAT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_PDF_DIMENSION_RE = re.compile(r"^\d+(\.\d+)?(px|in|cm|mm)$")
_PDF_MARGIN_RE = re.compile(r"^\d+(\.\d+)?(px|in|cm|mm)?$")

_DEFAULT_PDF_FORMAT = "A4"
_DEFAULT_PDF_MARGIN = "0.4in"
_DEFAULT_PDF_TIMEOUT_SECONDS = 60
_DEFAULT_PDF_MAX_SLIDES = 200
_DEFAULT_PDF_MAX_HTML_BYTES = 25 * 1024 * 1024
SlideAssetResolver = Callable[[str], dict[str, Any]]


def _get_slide_value(slide: Any, key: str, default: Any = None) -> Any:
    if isinstance(slide, dict):
        return slide.get(key, default)
    return getattr(slide, key, default)


def _sorted_slides(slides: Iterable[Any]) -> list[Any]:
    return sorted(slides, key=lambda s: int(_get_slide_value(s, "order", 0)))


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("slides export: invalid {} value {}", name, raw)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if is_truthy(value):
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("slides export: invalid {} value {}", name, raw)
    return default


def _normalize_pdf_dimension(value: Any, *, key: str) -> str:
    if not isinstance(value, str):
        raise SlidesExportInputError(f"pdf_{key}_invalid")
    cleaned = value.strip()
    if not cleaned or not _PDF_DIMENSION_RE.match(cleaned):
        raise SlidesExportInputError(f"pdf_{key}_invalid")
    return cleaned


def _normalize_pdf_margin_value(value: Any, *, key: str, default_value: str, default_unit: str = "in") -> str:
    if value is None:
        value = default_value
    if isinstance(value, (int, float)):
        return f"{value}{default_unit}"
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            cleaned = default_value
        if not _PDF_MARGIN_RE.match(cleaned):
            raise SlidesExportInputError(f"pdf_margin_{key}_invalid")
        if cleaned[-1].isdigit():
            return f"{cleaned}{default_unit}"
        return cleaned
    raise SlidesExportInputError(f"pdf_margin_{key}_invalid")


def _normalize_pdf_options(options: dict[str, Any] | None) -> dict[str, Any]:
    opts = options or {}
    if not isinstance(opts, dict):
        raise SlidesExportInputError("pdf_options_invalid")

    width = opts.get("width")
    height = opts.get("height")
    pdf_format = opts.get("format")
    landscape = opts.get("landscape")
    margin_opts = opts.get("margin") or {}

    if margin_opts and not isinstance(margin_opts, dict):
        raise SlidesExportInputError("pdf_margin_invalid")

    pdf_options: dict[str, Any] = {}
    if width is not None or height is not None:
        if width is None or height is None:
            raise SlidesExportInputError("pdf_width_height_required")
        pdf_options["width"] = _normalize_pdf_dimension(width, key="width")
        pdf_options["height"] = _normalize_pdf_dimension(height, key="height")
    else:
        if pdf_format is None or not str(pdf_format).strip():
            pdf_format = _env_str("SLIDES_PDF_FORMAT", _DEFAULT_PDF_FORMAT)
        pdf_format = str(pdf_format).strip()
        if not _PDF_FORMAT_RE.match(pdf_format):
            raise SlidesExportInputError("pdf_format_invalid")
        pdf_options["format"] = pdf_format

    margin_defaults = {
        "top": _env_str("SLIDES_PDF_MARGIN_TOP", _DEFAULT_PDF_MARGIN),
        "bottom": _env_str("SLIDES_PDF_MARGIN_BOTTOM", _DEFAULT_PDF_MARGIN),
        "left": _env_str("SLIDES_PDF_MARGIN_LEFT", _DEFAULT_PDF_MARGIN),
        "right": _env_str("SLIDES_PDF_MARGIN_RIGHT", _DEFAULT_PDF_MARGIN),
    }
    pdf_options["margin"] = {
        "top": _normalize_pdf_margin_value(margin_opts.get("top"), key="top", default_value=margin_defaults["top"]),
        "bottom": _normalize_pdf_margin_value(
            margin_opts.get("bottom"), key="bottom", default_value=margin_defaults["bottom"]
        ),
        "left": _normalize_pdf_margin_value(margin_opts.get("left"), key="left", default_value=margin_defaults["left"]),
        "right": _normalize_pdf_margin_value(
            margin_opts.get("right"), key="right", default_value=margin_defaults["right"]
        ),
    }
    if landscape is None:
        landscape = _env_bool("SLIDES_PDF_LANDSCAPE", False)
    pdf_options["landscape"] = bool(landscape)
    pdf_options["print_background"] = bool(opts.get("print_background", True))
    return pdf_options


def _resolve_pdf_timeout_ms() -> int:
    seconds = _env_int("SLIDES_PDF_TIMEOUT_SECONDS", _DEFAULT_PDF_TIMEOUT_SECONDS)
    if seconds <= 0:
        seconds = _DEFAULT_PDF_TIMEOUT_SECONDS
    return seconds * 1000


def _resolve_pdf_limits() -> tuple[int, int]:
    max_slides = _env_int("SLIDES_PDF_MAX_SLIDES", _DEFAULT_PDF_MAX_SLIDES)
    if max_slides < 0:
        max_slides = _DEFAULT_PDF_MAX_SLIDES
    max_bytes = _env_int("SLIDES_PDF_MAX_HTML_BYTES", _DEFAULT_PDF_MAX_HTML_BYTES)
    if max_bytes <= 0:
        max_bytes = _DEFAULT_PDF_MAX_HTML_BYTES
    return max_slides, max_bytes


def _resolve_image_asset(
    image: dict[str, Any],
    *,
    asset_resolver: SlideAssetResolver | None,
) -> dict[str, Any]:
    asset_ref = image.get("asset_ref")
    if not isinstance(asset_ref, str) or not asset_ref.strip():
        return image
    resolver = asset_resolver or partial(
        resolve_slide_asset,
        max_bytes=MAX_RESOLVED_SLIDE_ASSET_BYTES,
    )
    try:
        resolved = resolver(asset_ref.strip())
    except SlidesAssetError as exc:
        raise SlidesExportInputError(exc.code) from exc
    if not isinstance(resolved, dict):
        raise SlidesExportInputError("slide_asset_unresolved")
    candidate = {
        "id": image.get("id") or resolved.get("id"),
        "mime": resolved.get("mime") or image.get("mime"),
        "data_b64": resolved.get("data_b64"),
        "alt": image.get("alt") if image.get("alt") is not None else resolved.get("alt"),
        "width": image.get("width") if image.get("width") is not None else resolved.get("width"),
        "height": image.get("height") if image.get("height") is not None else resolved.get("height"),
    }
    try:
        return validate_images_payload([candidate])[0]
    except SlidesImageError as exc:
        raise SlidesExportInputError(exc.code) from exc


def _extract_images(
    slide: Any,
    *,
    asset_resolver: SlideAssetResolver | None = None,
) -> list[dict[str, Any]]:
    metadata = _get_slide_value(slide, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return []
    images = metadata.get("images")
    if images is None:
        return []
    try:
        normalized = validate_images_payload(images)
    except SlidesImageError as exc:
        raise SlidesExportInputError(exc.code) from exc
    return [_resolve_image_asset(image, asset_resolver=asset_resolver) for image in normalized]


def _escape_markdown_alt(text: str) -> str:
    value = text.replace("\\", "\\\\")
    value = value.replace("[", "\\[").replace("]", "\\]")
    return value


def _render_image_html(image: dict[str, Any]) -> str:
    alt = escape(str(image.get("alt") or ""))
    attrs = f' alt="{alt}"'
    width = image.get("width")
    if isinstance(width, int):
        attrs += f' width="{width}"'
    height = image.get("height")
    if isinstance(height, int):
        attrs += f' height="{height}"'
    src = f"data:{image['mime']};base64,{image['data_b64']}"
    return f'<img src="{src}"{attrs} />'


def _render_images_html(images: list[dict[str, Any]]) -> str:
    if not images:
        return ""
    rendered = "\n".join(_render_image_html(image) for image in images)
    return f'<div class="slide-images">\n{rendered}\n</div>'


def _render_image_markdown(image: dict[str, Any]) -> str:
    alt = str(image.get("alt") or "")
    alt = _escape_markdown_alt(alt.replace("\r", " ").replace("\n", " ").strip())
    src = f"data:{image['mime']};base64,{image['data_b64']}"
    return f"![{alt}]({src})"


def _sanitize_markdown(markdown_text: str) -> str:
    html = markdown.markdown(markdown_text or "", extensions=["extra", "sane_lists"], output_format="html5")
    if bleach is None:
        return escape(html)
    return bleach.clean(
        html,
        tags=_ALLOWED_HTML_TAGS,
        attributes=_ALLOWED_HTML_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
        strip_comments=True,
    )


def _json_for_inline_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _css_tokens_contain_url(tokens: Iterable[Any]) -> bool:
    """Reject URL values after tinycss2 has normalized CSS escapes."""

    for token in tokens:
        token_type = getattr(token, "type", None)
        if token_type in {"url", "bad-url"}:
            return True
        if token_type == _CSS_FUNCTION_TYPE:
            name = getattr(token, "lower_name", None) or getattr(token, "name", None)
            if isinstance(name, str) and name.casefold() == "url":
                return True
            arguments = getattr(token, "arguments", None)
            if isinstance(arguments, list) and _css_tokens_contain_url(arguments):
                return True
        content = getattr(token, "content", None)
        if isinstance(content, list) and _css_tokens_contain_url(content):
            return True
    return False


def _parse_css_rules_without_urls(css_text: str) -> list[Any]:
    """Parse a stylesheet and reject every normalized URL-bearing token."""

    rules = tinycss2.parse_stylesheet(
        css_text,
        skip_comments=False,
        skip_whitespace=False,
    )
    for rule in rules:
        if getattr(rule, "type", None) == "error":
            raise SlidesExportInputError("custom_css_rule_blocked")
        prelude = getattr(rule, "prelude", None)
        content = getattr(rule, "content", None)
        if (isinstance(prelude, list) and _css_tokens_contain_url(prelude)) or (
            isinstance(content, list) and _css_tokens_contain_url(content)
        ):
            raise SlidesExportInputError("custom_css_url_blocked")
    return rules


def _sanitize_custom_css(css_text: str | None) -> str | None:
    if not css_text:
        return None
    if CSSSanitizer is None or tinycss2 is None:
        return None
    if re.search(r"@import", css_text, flags=re.IGNORECASE):
        raise SlidesExportInputError("custom_css_import_blocked")
    if re.search(r"url\s*\(", css_text, flags=re.IGNORECASE):
        raise SlidesExportInputError("custom_css_url_blocked")
    cleaned = ""
    try:
        sanitizer = CSSSanitizer(allowed_css_properties=_ALLOWED_CSS_PROPERTIES)
        rules = _parse_css_rules_without_urls(css_text)
        sanitized_rules: list[str] = []
        for rule in rules:
            if rule.type in {"comment", "whitespace"}:
                sanitized_rules.append(tinycss2.serialize([rule]))
                continue
            if rule.type != "qualified-rule":
                raise SlidesExportInputError("custom_css_rule_blocked")
            declarations = sanitizer.sanitize_css(tinycss2.serialize(rule.content)).strip()
            if declarations:
                sanitized_rules.append(f"{tinycss2.serialize(rule.prelude).strip()} {{{declarations}}}")
        cleaned = "".join(sanitized_rules)
        _parse_css_rules_without_urls(cleaned)
    except SlidesExportInputError:
        raise
    except Exception:
        logger.warning("slides export: css sanitizer failed")
        cleaned = ""
    cleaned = cleaned.replace("\x00", "").strip()
    return cleaned or None


def _normalize_text_block(text: str | None) -> str:
    """Collapse a text block into comparable normalized lines."""

    if not text:
        return ""
    lines = [" ".join(line.split()) for line in str(text).splitlines()]
    normalized = "\n".join(line for line in lines if line)
    return normalized.strip()


def _extract_visual_blocks(slide: Any) -> list[dict[str, Any]]:
    """Extract normalized visual block metadata from a slide payload."""

    metadata = _get_slide_value(slide, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return []
    blocks = metadata.get("visual_blocks")
    if not isinstance(blocks, list):
        return []
    normalized: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, dict):
            normalized.append(dict(block))
    return normalized


def _compile_visual_block_fallback(block: dict[str, Any]) -> list[str]:
    """Compile a visual block into plain-text fallback lines for markdown exports."""

    block_type = str(block.get("type") or "").strip()
    if block_type == "timeline":
        items = block.get("items")
        if not isinstance(items, list):
            return []
        lines: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("date") or item.get("year") or "").strip()
            title = str(item.get("title") or item.get("name") or "").strip()
            description = str(item.get("description") or item.get("summary") or "").strip()
            headline = ": ".join(part for part in (label, title) if part)
            if not headline:
                headline = "Timeline event"
            line = f"- {headline}"
            if description:
                line += f" - {description}"
            lines.append(line)
        return lines
    if block_type == "comparison_matrix":
        rows = block.get("rows")
        if not isinstance(rows, list):
            return []
        lines = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or row.get("name") or row.get("topic") or "").strip()
            values = row.get("values")
            summary = str(row.get("summary") or row.get("description") or "").strip()
            value_text = ", ".join(str(item) for item in values) if isinstance(values, list) else ""
            details = summary or value_text
            headline = label or "Comparison"
            line = f"- {headline}"
            if details:
                line += f": {details}"
            lines.append(line)
        return lines
    if block_type == "process_flow":
        steps = block.get("steps")
        if not isinstance(steps, list):
            return []
        lines = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            title = str(step.get("title") or step.get("name") or f"Step {index}").strip()
            description = str(step.get("description") or step.get("summary") or "").strip()
            line = f"{index}. {title}"
            if description:
                line += f" - {description}"
            lines.append(line)
        return lines
    if block_type == "stat_group":
        items = block.get("items")
        if not isinstance(items, list):
            return []
        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or "Metric").strip()
            value = str(item.get("value") or item.get("stat") or "").strip()
            context = str(item.get("context") or item.get("description") or "").strip()
            line = f"- {label}"
            if value:
                line += f": {value}"
            if context:
                line += f" - {context}"
            lines.append(line)
        return lines
    return []


def _render_timeline_block(block: dict[str, Any]) -> str:
    """Render timeline metadata into HTML for Reveal exports."""

    items = block.get("items")
    if not isinstance(items, list):
        return ""
    rendered_items: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = escape(str(item.get("label") or item.get("date") or item.get("year") or "").strip())
        title = escape(str(item.get("title") or item.get("name") or "").strip())
        description = escape(str(item.get("description") or item.get("summary") or "").strip())
        if not (label or title or description):
            continue
        parts = ['<li class="visual-block-item">']
        if label:
            parts.append(f'<p class="visual-block-label">{label}</p>')
        if title:
            parts.append(f"<h3>{title}</h3>")
        if description:
            parts.append(f"<p>{description}</p>")
        parts.append("</li>")
        rendered_items.append("".join(parts))
    if not rendered_items:
        return ""
    return (
        '<div class="visual-block visual-block--timeline" data-visual-block-type="timeline">'
        '<ol class="visual-block-list">' + "".join(rendered_items) + "</ol></div>"
    )


def _render_comparison_matrix_block(block: dict[str, Any]) -> str:
    """Render comparison matrix metadata into HTML for Reveal exports."""

    rows = block.get("rows")
    if not isinstance(rows, list):
        return ""
    rendered_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = escape(str(row.get("label") or row.get("name") or row.get("topic") or "").strip())
        values = row.get("values")
        summary = escape(str(row.get("summary") or row.get("description") or "").strip())
        value_cells = []
        if isinstance(values, list):
            value_cells = [f"<td>{escape(str(value).strip())}</td>" for value in values if str(value).strip()]
        if summary:
            value_cells.append(f"<td>{summary}</td>")
        if not label and not value_cells:
            continue
        heading = f"<th>{label or 'Comparison'}</th>"
        rendered_rows.append(f"<tr>{heading}{''.join(value_cells)}</tr>")
    if not rendered_rows:
        return ""
    return (
        '<div class="visual-block visual-block--comparison" data-visual-block-type="comparison_matrix">'
        '<table class="visual-block-table"><tbody>' + "".join(rendered_rows) + "</tbody></table></div>"
    )


def _render_process_flow_block(block: dict[str, Any]) -> str:
    """Render process flow metadata into HTML for Reveal exports."""

    steps = block.get("steps")
    if not isinstance(steps, list):
        return ""
    rendered_steps: list[str] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        title = escape(str(step.get("title") or step.get("name") or f"Step {index}").strip())
        description = escape(str(step.get("description") or step.get("summary") or "").strip())
        parts = ['<li class="visual-block-item">', f"<h3>{title}</h3>"]
        if description:
            parts.append(f"<p>{description}</p>")
        parts.append("</li>")
        rendered_steps.append("".join(parts))
    if not rendered_steps:
        return ""
    return (
        '<div class="visual-block visual-block--process" data-visual-block-type="process_flow">'
        '<ol class="visual-block-list">' + "".join(rendered_steps) + "</ol></div>"
    )


def _render_stat_group_block(block: dict[str, Any]) -> str:
    """Render stat group metadata into HTML for Reveal exports."""

    items = block.get("items")
    if not isinstance(items, list):
        return ""
    rendered_items: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = escape(str(item.get("label") or item.get("name") or "Metric").strip())
        value = escape(str(item.get("value") or item.get("stat") or "").strip())
        context = escape(str(item.get("context") or item.get("description") or "").strip())
        if not (label or value or context):
            continue
        parts = ['<div class="visual-block-item">']
        if label:
            parts.append(f"<dt>{label}</dt>")
        if value:
            parts.append(f"<dd>{value}</dd>")
        if context:
            parts.append(f"<p>{context}</p>")
        parts.append("</div>")
        rendered_items.append("".join(parts))
    if not rendered_items:
        return ""
    return (
        '<div class="visual-block visual-block--stats" data-visual-block-type="stat_group">'
        '<dl class="visual-block-stats">' + "".join(rendered_items) + "</dl></div>"
    )


def _render_visual_blocks_html(blocks: list[dict[str, Any]]) -> tuple[str, bool, str]:
    """Render all supported visual blocks and return HTML plus fallback metadata."""

    if not blocks:
        return "", False, ""
    rendered_html: list[str] = []
    fallback_lines: list[str] = []
    rendered_count = 0
    for block in blocks:
        fallback_lines.extend(_compile_visual_block_fallback(block))
        block_type = str(block.get("type") or "").strip()
        if block_type == "timeline":
            html = _render_timeline_block(block)
        elif block_type == "comparison_matrix":
            html = _render_comparison_matrix_block(block)
        elif block_type == "process_flow":
            html = _render_process_flow_block(block)
        elif block_type == "stat_group":
            html = _render_stat_group_block(block)
        else:
            html = ""
        if html:
            rendered_html.append(html)
            rendered_count += 1
    return "\n".join(rendered_html), rendered_count == len(blocks) and bool(rendered_html), "\n".join(fallback_lines)


def _content_is_structured_fallback(content: Any, fallback_text: str) -> bool:
    """Return whether slide content already matches the generated block fallback text."""

    normalized_content = _normalize_text_block(str(content or ""))
    normalized_fallback = _normalize_text_block(fallback_text)
    if not normalized_fallback:
        return False
    return normalized_content == normalized_fallback


def _render_style_shell_attrs(visual_style_snapshot: dict[str, Any] | None) -> str:
    """Render shell-level data attributes for built-in visual style hooks."""

    if not isinstance(visual_style_snapshot, dict):
        return ""
    resolution = visual_style_snapshot.get("resolution")
    if not isinstance(resolution, dict):
        return ""
    style_id = str(visual_style_snapshot.get("id") or "").strip()
    style_pack = str(resolution.get("style_pack") or "").strip()
    attrs: list[str] = []
    if style_id:
        attrs.append(f'data-visual-style="{escape(style_id)}"')
    if style_pack:
        attrs.append(f'data-style-pack="{escape(style_pack)}"')
    if not attrs:
        return ""
    return " " + " ".join(attrs)


def _compose_export_css(
    *,
    custom_css: str | None,
    visual_style_snapshot: dict[str, Any] | None,
) -> str | None:
    """Compose shared export CSS with builtin pack CSS and custom deck CSS."""

    built_in_css = None
    if isinstance(visual_style_snapshot, dict):
        resolution = visual_style_snapshot.get("resolution")
        if isinstance(resolution, dict):
            style_id = str(visual_style_snapshot.get("id") or "").strip()
            style_pack = str(resolution.get("style_pack") or "").strip()
            token_overrides = resolution.get("token_overrides")
            if style_id and style_pack and isinstance(token_overrides, dict):
                built_in_css = render_pack_custom_css(
                    style_id=style_id,
                    pack_id=style_pack,
                    token_overrides=token_overrides,
                )

    shared_css = """
.reveal .content > * + * { margin-top: 1rem; }
.reveal .visual-block { margin-top: 1rem; padding: 1rem; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 1rem; }
.reveal .visual-block-table { width: 100%; }
.reveal .visual-block-table th,
.reveal .visual-block-table td { padding: 0.6rem; border: 1px solid rgba(148, 163, 184, 0.25); text-align: left; }
.reveal .visual-block-list { margin: 0; padding-left: 1.25rem; }
.reveal .visual-block-item + .visual-block-item { margin-top: 0.75rem; }
.reveal .visual-block-item h3,
.reveal .visual-block-item p,
.reveal .visual-block-label,
.reveal .visual-block-stats dt,
.reveal .visual-block-stats dd { margin: 0; }
.reveal .visual-block-stats .visual-block-item + .visual-block-item { margin-top: 0.75rem; }
.reveal .visual-block-stats dd { font-weight: 700; }
""".strip()

    parts = [shared_css]
    if built_in_css and not (custom_css and built_in_css.strip() in custom_css):
        parts.append(built_in_css)
    if custom_css:
        parts.append(custom_css)
    merged = "\n\n".join(part.strip() for part in parts if part and part.strip())
    return merged or None


def _resolve_assets_dir(assets_dir: Path | str | None) -> Path:
    if assets_dir:
        return Path(assets_dir).expanduser().resolve()
    env_path = os.getenv("SLIDES_REVEALJS_ASSETS_DIR")
    if env_path:
        base = Path(env_path).expanduser()
        if not base.is_absolute():
            project_root = settings.get("PROJECT_ROOT")
            if project_root:
                base = Path(project_root) / base
        return base.resolve()
    return (Path(__file__).resolve().parent / "revealjs").resolve()


def _find_license_file(assets_dir: Path) -> Path | None:
    candidates = [
        assets_dir / "LICENSE.revealjs.txt",
        assets_dir / "LICENSE",
        assets_dir / "LICENSE.txt",
        assets_dir / "LICENSE.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_notice_file(assets_dir: Path) -> Path | None:
    candidates = [
        assets_dir / "NOTICE.revealjs.txt",
        assets_dir / "NOTICE",
        assets_dir / "NOTICE.txt",
        assets_dir / "NOTICE.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_reveal_assets(assets_dir: Path, theme: str) -> None:
    required = [
        assets_dir / "reveal.js",
        assets_dir / "reveal.css",
        assets_dir / "plugin" / "notes" / "notes.js",
        assets_dir / "theme" / f"{theme}.css",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise SlidesAssetsMissingError(f"Reveal.js assets missing: {missing_list}")
    if _find_license_file(assets_dir) is None:
        raise SlidesAssetsMissingError("Reveal.js LICENSE file missing")


def _render_sections(
    slides: Iterable[Any],
    *,
    asset_resolver: SlideAssetResolver | None = None,
) -> str:
    sections: list[str] = []
    for slide in _sorted_slides(slides):
        layout = escape(str(_get_slide_value(slide, "layout", "content")))
        title = _get_slide_value(slide, "title")
        content = _get_slide_value(slide, "content", "")
        notes = _get_slide_value(slide, "speaker_notes")
        images = _extract_images(slide, asset_resolver=asset_resolver)
        visual_blocks = _extract_visual_blocks(slide)
        images_html = _render_images_html(images)
        visual_blocks_html, rendered_all_blocks, fallback_text = _render_visual_blocks_html(visual_blocks)

        title_html = f"<h2>{escape(str(title))}</h2>" if title else ""
        suppress_content = rendered_all_blocks and _content_is_structured_fallback(content, fallback_text)
        content_html = _sanitize_markdown(str(content or "")) if content and not suppress_content else ""
        body_html = "".join(part for part in (content_html, visual_blocks_html, images_html) if part)
        notes_html = f'<aside class="notes">{escape(str(notes))}</aside>' if notes else ""

        section = (
            f'      <section data-layout="{layout}">\n'
            f"        {title_html}\n"
            f'        <div class="content">{body_html}</div>\n'
            f"        {notes_html}\n"
            f"      </section>"
        )
        sections.append(section)
    return "\n".join(sections)


def _render_index_html(
    *,
    title: str,
    slides: Iterable[Any],
    theme: str,
    settings_json: str,
    include_custom_css: bool,
    visual_style_snapshot: dict[str, Any] | None = None,
    asset_resolver: SlideAssetResolver | None = None,
) -> str:
    css_link = '  <link rel="stylesheet" href="assets/custom.css">\n' if include_custom_css else ""
    title_html = escape(title or "Presentation")
    sections_html = _render_sections(slides, asset_resolver=asset_resolver)
    reveal_attrs = _render_style_shell_attrs(visual_style_snapshot)
    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{title_html}</title>\n"
        '  <link rel="stylesheet" href="assets/reveal/reveal.css">\n'
        f'  <link rel="stylesheet" href="assets/reveal/theme/{escape(theme)}.css">\n'
        f"{css_link}"
        "</head>\n"
        "<body>\n"
        f'  <div class="reveal"{reveal_attrs}>\n'
        '    <div class="slides">\n'
        f"{sections_html}\n"
        "    </div>\n"
        "  </div>\n"
        '  <script src="assets/reveal/reveal.js"></script>\n'
        '  <script src="assets/reveal/plugin/notes/notes.js"></script>\n'
        "  <script>\n"
        f"    const settings = {settings_json};\n"
        "    settings.plugins = [ RevealNotes ];\n"
        "    Reveal.initialize(settings);\n"
        "  </script>\n"
        "</body>\n"
        "</html>"
    )


def export_presentation_bundle(
    *,
    title: str,
    slides: Iterable[Any],
    theme: str,
    settings: dict[str, Any] | None,
    custom_css: str | None,
    visual_style_snapshot: dict[str, Any] | None = None,
    assets_dir: Path | str | None = None,
    asset_resolver: SlideAssetResolver | None = None,
) -> bytes:
    resolved_assets = _resolve_assets_dir(assets_dir)
    _validate_reveal_assets(resolved_assets, theme)
    settings_json = _json_for_inline_script(settings or {})
    sanitized_css = _sanitize_custom_css(
        _compose_export_css(custom_css=custom_css, visual_style_snapshot=visual_style_snapshot)
    )
    index_html = _render_index_html(
        title=title,
        slides=slides,
        theme=theme,
        settings_json=settings_json,
        include_custom_css=bool(sanitized_css),
        visual_style_snapshot=visual_style_snapshot,
        asset_resolver=asset_resolver,
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", index_html)
        if sanitized_css:
            zf.writestr("assets/custom.css", sanitized_css)

        license_file = _find_license_file(resolved_assets)
        if license_file:
            zf.write(license_file, "LICENSE.revealjs.txt")
        notice_file = _find_notice_file(resolved_assets)
        if notice_file:
            zf.write(notice_file, "NOTICE.revealjs.txt")

        for path in resolved_assets.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(resolved_assets)
            zf.write(path, (Path("assets") / "reveal" / rel).as_posix())

    return buffer.getvalue()


def export_presentation_markdown(
    *,
    title: str,
    slides: Iterable[Any],
    theme: str,
    marp_theme: str | None = None,
    asset_resolver: SlideAssetResolver | None = None,
) -> str:
    resolved_theme = marp_theme or _REVEAL_THEME_TO_MARP.get(theme, "default")
    lines: list[str] = [
        "---",
        "marp: true",
        f"theme: {resolved_theme}",
        "---",
        "",
    ]
    for slide in _sorted_slides(slides):
        layout = str(_get_slide_value(slide, "layout", "content"))
        slide_title = _get_slide_value(slide, "title")
        content = _get_slide_value(slide, "content", "")
        notes = _get_slide_value(slide, "speaker_notes")
        images = _extract_images(slide, asset_resolver=asset_resolver)
        if layout in {"title", "section"} and slide_title:
            header = "# " if layout == "title" else "## "
            lines.append(f"{header}{slide_title}")
            lines.append("")
        elif slide_title:
            lines.append(f"## {slide_title}")
            lines.append("")
        if content:
            lines.append(str(content))
            lines.append("")
        for image in images:
            lines.append(_render_image_markdown(image))
            lines.append("")
        if notes:
            lines.append("<!--")
            lines.append(str(notes))
            lines.append("-->")
            lines.append("")
        lines.append("---")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    if lines and lines[-1] == "---":
        lines.pop()
    return "\n".join(lines).strip() + "\n"


def export_presentation_pdf(
    *,
    title: str,
    slides: Iterable[Any],
    theme: str,
    settings: dict[str, Any] | None,
    custom_css: str | None,
    visual_style_snapshot: dict[str, Any] | None = None,
    assets_dir: Path | str | None = None,
    pdf_options: dict[str, Any] | None = None,
    asset_resolver: SlideAssetResolver | None = None,
) -> bytes:
    slides_list = list(slides)
    max_slides, max_html_bytes = _resolve_pdf_limits()
    if max_slides and len(slides_list) > max_slides:
        raise SlidesExportInputError("pdf_slides_too_many")

    resolved_assets = _resolve_assets_dir(assets_dir)
    _validate_reveal_assets(resolved_assets, theme)
    settings_json = _json_for_inline_script(settings or {})
    sanitized_css = _sanitize_custom_css(
        _compose_export_css(custom_css=custom_css, visual_style_snapshot=visual_style_snapshot)
    )
    index_html = _render_index_html(
        title=title,
        slides=slides_list,
        theme=theme,
        settings_json=settings_json,
        include_custom_css=bool(sanitized_css),
        visual_style_snapshot=visual_style_snapshot,
        asset_resolver=asset_resolver,
    )
    if max_html_bytes and len(index_html.encode("utf-8")) > max_html_bytes:
        raise SlidesExportInputError("pdf_html_too_large")

    if sync_playwright is None:
        raise SlidesExportError("playwright_unavailable")

    pdf_settings = _normalize_pdf_options(pdf_options)
    timeout_ms = _resolve_pdf_timeout_ms()

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        assets_target = base_dir / "assets" / "reveal"
        assets_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(resolved_assets, assets_target, dirs_exist_ok=True)

        (base_dir / "index.html").write_text(index_html, encoding="utf-8")
        if sanitized_css:
            (base_dir / "assets" / "custom.css").write_text(sanitized_css, encoding="utf-8")

        pdf_url = (base_dir / "index.html").resolve().as_uri() + "?print-pdf"

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.set_default_timeout(timeout_ms)
                    page.goto(pdf_url, wait_until="load", timeout=timeout_ms)
                    try:
                        page.emulate_media(media="print")
                    except PlaywrightError:
                        logger.debug("slides export: emulate_media not supported by playwright")
                    try:
                        page.wait_for_function(
                            "window.Reveal && window.Reveal.isReady && window.Reveal.isReady()",
                            timeout=timeout_ms,
                        )
                    except PlaywrightTimeoutError:
                        logger.warning("slides export: reveal did not signal ready before timeout")
                    pdf_bytes = page.pdf(timeout=timeout_ms, **pdf_settings)
                finally:
                    browser.close()
        except PlaywrightTimeoutError as exc:
            raise SlidesExportError("pdf_timeout") from exc
        except PlaywrightError as exc:
            raise SlidesExportError("pdf_render_failed") from exc
        except Exception as exc:
            raise SlidesExportError("pdf_render_failed") from exc

    return pdf_bytes


def export_presentation_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
