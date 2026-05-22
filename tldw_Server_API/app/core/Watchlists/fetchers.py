"""
Watchlists fetch helpers for RSS/Atom feeds and HTML scraping.

Highlights:
- RSS fetch: reuses workflows adapter logic for URL policy checks and XML parsing.
  TEST_MODE returns static items so offline unit tests stay deterministic.
- Site fetch: relies on the blocking article extractor plus optional rule-based
  list scraping informed by FreshRSS-style XPath/CSS selectors.

Returned item structure (normalized):
- RSS items: { 'title': str, 'url': str, 'summary': Optional[str], 'published': Optional[str] }
- Site articles: { 'title': str, 'url': str, 'content': str, 'author': Optional[str] }
- Scraped list items: { 'title': str, 'url': str, 'summary': Optional[str], 'content': Optional[str], ... }
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from typing import Any
from urllib.parse import urljoin

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
import regex
from loguru import logger
from lxml import html
from lxml.etree import XPath, XPathError
from lxml.html import HtmlElement

from tldw_Server_API.app.core.Security.egress import is_url_allowed, is_url_allowed_for_tenant
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode

_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
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
    ValueError,
    UnicodeDecodeError,
    ET.ParseError,
    DefusedXmlException,
    XPathError,
)

_SELECTOR_CACHE_MAX = 512
_XPATH_SELECTOR_CACHE: OrderedDict[str, Any] = OrderedDict()
_CSS_SELECTOR_CACHE: OrderedDict[str, Any] = OrderedDict()
_SELECTOR_CACHE_LOCK = Lock()
_DEFAULT_MAX_SELECTOR_EXPR_LEN = 512
_DEFAULT_MAX_XPATH_DESCENDANT_STEPS = 12
_DEFAULT_MAX_XPATH_PREDICATES = 10
_DEFAULT_MAX_XPATH_FUNCTION_CALLS = 8


def _parse_guardrail_int_env(
    env_name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        logger.warning("Invalid {}='{}'; using default {}", env_name, raw, default)
        return default
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


@lru_cache(maxsize=1)
def _selector_guardrail_limits() -> tuple[int, int, int, int]:
    """Return selector guardrail limits with environment overrides."""
    return (
        _parse_guardrail_int_env(
            "WATCHLIST_SELECTOR_MAX_EXPR_LEN",
            _DEFAULT_MAX_SELECTOR_EXPR_LEN,
            minimum=32,
            maximum=8192,
        ),
        _parse_guardrail_int_env(
            "WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS",
            _DEFAULT_MAX_XPATH_DESCENDANT_STEPS,
            minimum=1,
            maximum=200,
        ),
        _parse_guardrail_int_env(
            "WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES",
            _DEFAULT_MAX_XPATH_PREDICATES,
            minimum=1,
            maximum=200,
        ),
        _parse_guardrail_int_env(
            "WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS",
            _DEFAULT_MAX_XPATH_FUNCTION_CALLS,
            minimum=0,
            maximum=200,
        ),
    )


def reload_selector_guardrails_from_env() -> None:
    """Clear selector guardrail config cache so env changes take effect."""
    _selector_guardrail_limits.cache_clear()


def _count_xpath_function_calls(expr: str) -> int:
    """Count likely XPath function calls using a linear scan."""
    count = 0
    for idx, ch in enumerate(expr):
        if ch != "(":
            continue
        prev = idx - 1
        while prev >= 0 and expr[prev].isspace():
            prev -= 1
        if prev < 0:
            continue
        if not (expr[prev].isalnum() or expr[prev] in {"_", "-"}):
            continue
        start = prev
        while start >= 0 and (expr[start].isalnum() or expr[start] in {"_", "-"}):
            start -= 1
        token = expr[start + 1 : prev + 1]
        if token and (token[0].isalpha() or token[0] == "_"):
            count += 1
    return count


def _selector_safety_error(selector: str) -> str | None:
    (
        max_selector_expr_len,
        max_xpath_descendant_steps,
        max_xpath_predicates,
        max_xpath_function_calls,
    ) = _selector_guardrail_limits()
    expr = selector.strip()
    if not expr:
        return None
    if len(expr) > max_selector_expr_len:
        return f"selector_too_complex:length>{max_selector_expr_len}"

    if expr.startswith("css:"):
        css_expr = expr[4:].strip()
        if len(css_expr) > max_selector_expr_len:
            return f"selector_too_complex:length>{max_selector_expr_len}"
        return None

    # Keep XPath expressive enough for normal extraction, but block
    # patterns that are commonly abused for expensive evaluation.
    if "//*" in expr:
        return "selector_too_complex:wildcard_descendant_not_allowed"
    if "::" in expr:
        return "selector_too_complex:axis_not_allowed"
    if "|" in expr:
        return "selector_too_complex:union_not_allowed"
    if "$" in expr:
        return "selector_too_complex:variables_not_allowed"

    if expr.count("//") > max_xpath_descendant_steps:
        return f"selector_too_complex:descendants>{max_xpath_descendant_steps}"
    if expr.count("[") > max_xpath_predicates:
        return f"selector_too_complex:predicates>{max_xpath_predicates}"

    function_calls = _count_xpath_function_calls(expr)
    if function_calls > max_xpath_function_calls:
        return f"selector_too_complex:function_calls>{max_xpath_function_calls}"

    return None


def _in_test_mode() -> bool:
    return _is_test_mode()


def get_selector_cache_stats() -> dict[str, int]:
    with _SELECTOR_CACHE_LOCK:
        return {
            "selector_xpath_cache_size": len(_XPATH_SELECTOR_CACHE),
            "selector_css_cache_size": len(_CSS_SELECTOR_CACHE),
        }


def clear_selector_caches() -> None:
    with _SELECTOR_CACHE_LOCK:
        _XPATH_SELECTOR_CACHE.clear()
        _CSS_SELECTOR_CACHE.clear()


def _selector_cache_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    with _SELECTOR_CACHE_LOCK:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value


def _selector_cache_put(cache: OrderedDict[str, Any], key: str, value: Any) -> None:
    with _SELECTOR_CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _SELECTOR_CACHE_MAX:
            cache.popitem(last=False)


async def _close_response(resp: Any) -> None:
    if resp is None:
        return
    close = getattr(resp, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(resp, "close", None)
    if callable(close):
        close()


def _ensure_sequence(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if isinstance(v, str)]


def _contextualize_xpath(expr: str, node: Any) -> str:
    if not isinstance(node, HtmlElement):
        return expr
    if expr.startswith("."):
        return expr
    if expr.startswith("//"):
        match = re.match(r"^//([a-zA-Z0-9_*:-]+)", expr)
        if match:
            token = match.group(1)
            try:
                node_tag = node.tag
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                node_tag = None
            if node_tag and token == str(node_tag):
                return expr
        return f".{expr}"
    if expr.startswith("/"):
        try:
            root = node.getroottree().getroot()
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            root = None
        if root is not None:
            root_tag = getattr(root, "tag", None)
            if isinstance(root_tag, str) and expr.startswith(f"/{root_tag}"):
                return expr
        if root is not None and root is not node:
            return f".{expr}"
    return expr


def _parse_nth_token(token: str, allowed: set[str]) -> tuple[str | None, int | None]:
    match = re.fullmatch(r"(?P<tag>[a-zA-Z0-9_-]+)(?::nth-child\((?P<index>\d+)\))?", token)
    if not match:
        return None, None
    tag = match.group("tag").lower()
    if tag not in allowed:
        return None, None
    idx = match.group("index")
    return tag, int(idx) if idx else None


def _css_table_nth_xpath(css_expr: str) -> str | None:
    expr = re.sub(r"\s+", " ", css_expr.strip())
    if not expr.startswith("table "):
        return None
    tokens = expr.split(" ")
    if not tokens or tokens[0] != "table":
        return None
    idx = 1
    section = None
    if idx < len(tokens) and tokens[idx] in {"tbody", "thead", "tfoot"}:
        section = tokens[idx]
        idx += 1
    if idx >= len(tokens):
        return None
    tr_tag, tr_idx = _parse_nth_token(tokens[idx], {"tr"})
    if not tr_tag:
        return None
    idx += 1
    cell_tag = None
    cell_idx = None
    if idx < len(tokens):
        cell_tag, cell_idx = _parse_nth_token(tokens[idx], {"td", "th"})
        if not cell_tag:
            return None
        idx += 1
    if idx != len(tokens):
        return None
    parts = [".//table"]
    if section:
        parts.append(f"//{section}")
    tr_expr = f"//{tr_tag}"
    if tr_idx:
        tr_expr = f"{tr_expr}[position()={tr_idx}]"
    parts.append(tr_expr)
    if cell_tag:
        cell_expr = f"//{cell_tag}"
        if cell_idx:
            cell_expr = f"{cell_expr}[position()={cell_idx}]"
        parts.append(cell_expr)
    return "".join(parts)


def _select_nodes(node: HtmlElement, selector: str, *, context_sensitive: bool = False) -> list[Any]:
    expr = selector.strip()
    if not expr:
        return []
    safety_error = _selector_safety_error(expr)
    if safety_error:
        logger.debug(f"Selector rejected by safety guard ({safety_error}): '{expr[:160]}'")
        return []
    if expr.startswith("css:"):
        css_expr = expr[4:].strip()
        if not css_expr:
            return []
        fast_xpath = _css_table_nth_xpath(css_expr)
        if fast_xpath:
            expr = fast_xpath
        else:
            try:
                from lxml.cssselect import CSSSelector
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"CSS selector support unavailable for '{css_expr}': {exc}")
                return []
            try:
                compiled = _selector_cache_get(_CSS_SELECTOR_CACHE, css_expr)
                if compiled is None:
                    compiled = CSSSelector(css_expr)
                    _selector_cache_put(_CSS_SELECTOR_CACHE, css_expr, compiled)
                return list(compiled(node))
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"CSS selector evaluation failed for '{css_expr}': {exc}")
                return []
    if context_sensitive:
        expr = _contextualize_xpath(expr, node)
    compiled_xpath = _selector_cache_get(_XPATH_SELECTOR_CACHE, expr)
    if compiled_xpath is None:
        try:
            compiled_xpath = XPath(expr)
            _selector_cache_put(_XPATH_SELECTOR_CACHE, expr, compiled_xpath)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"XPath compilation failed for '{expr}': {exc}")
            return []
    try:
        result = compiled_xpath(node)
    except XPathError as exc:
        logger.debug(f"XPath evaluation failed for '{expr}': {exc}")
        return []
    if isinstance(result, list):
        return result
    return [result]


def _coerce_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="ignore").strip()
            return text or None
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return None
    if hasattr(value, "text_content"):
        try:
            text = value.text_content().strip()
            return text or None
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return None
    try:
        text = str(value).strip()
        return text or None
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        return None


def _reduce_matches(matches: Sequence[Any], join_with: str) -> str | None:
    parts: list[str] = []
    for match in matches:
        val = _coerce_value(match)
        if val:
            parts.append(val)
    if not parts:
        return None
    return join_with.join(parts).strip() or None


def _extract_value(
    node: HtmlElement,
    selectors: Sequence[str] | str | None,
    *,
    join: bool = False,
    join_with: str = " ",
) -> str | None:
    for expr in _ensure_sequence(selectors):
        matches = _select_nodes(node, expr, context_sensitive=True)
        if not matches:
            continue
        value = _reduce_matches(matches, join_with) if join else _coerce_value(matches[0])
        if value:
            return value
    return None


def _normalize_selector_expr(selector: str | None, *, css: str | None = None, xpath: str | None = None) -> str | None:
    """Normalize selector inputs into a single expression string."""
    if selector and str(selector).strip():
        return str(selector).strip()
    if css and str(css).strip():
        return f"css:{str(css).strip()}"
    if xpath and str(xpath).strip():
        return str(xpath).strip()
    return None


def _field_selector(field: dict[str, Any]) -> str | None:
    return _normalize_selector_expr(
        field.get("selector"),
        css=field.get("css"),
        xpath=field.get("xpath"),
    )


def _base_selector(rules: dict[str, Any]) -> str | None:
    return _normalize_selector_expr(
        rules.get("baseSelector") or rules.get("base_selector"),
        css=rules.get("baseCss") or rules.get("base_css"),
        xpath=rules.get("baseXpath") or rules.get("base_xpath"),
    )


def _normalize_field_definitions(fields: Any) -> list[dict[str, Any]]:
    if isinstance(fields, list):
        return [f for f in fields if isinstance(f, dict)]
    if isinstance(fields, dict):
        normalized: list[dict[str, Any]] = []
        for name, spec in fields.items():
            entry = dict(spec) if isinstance(spec, dict) else {"selector": spec}
            entry.setdefault("name", str(name))
            normalized.append(entry)
        return normalized
    return []


def _number_normalize(value: str) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def _apply_single_transform(value: str, transform: Any, base_url: str) -> str | None:
    if value is None:
        return None
    if isinstance(transform, str):
        name = transform.strip().lower()
        params: dict[str, Any] = {}
    elif isinstance(transform, dict):
        name = str(transform.get("name") or transform.get("type") or "").strip().lower()
        params = transform
    else:
        return value
    if name == "lowercase":
        return value.lower()
    if name == "uppercase":
        return value.upper()
    if name == "strip":
        return value.strip()
    if name == "regex_replace":
        pattern = params.get("pattern")
        repl = params.get("repl", "")
        if not isinstance(pattern, str):
            return value
        try:
            return re.sub(pattern, str(repl), value)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return value
    if name == "urljoin":
        try:
            return urljoin(base_url, value)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return value
    if name == "date_normalize":
        normalized = _normalize_datetime(value, params.get("format") if isinstance(params.get("format"), str) else None)
        return normalized or value
    if name == "number_normalize":
        normalized = _number_normalize(value)
        return normalized or value
    return value


def _apply_transforms(value: Any, transforms: Any, base_url: str) -> Any:
    if value is None or not transforms:
        return value
    transform_list = transforms if isinstance(transforms, list) else [transforms]
    if isinstance(value, list):
        return [v for v in (_apply_transforms(v, transform_list, base_url) for v in value) if v is not None]
    if isinstance(value, dict):
        return value
    result: str | None = str(value)
    for transform in transform_list:
        result = _apply_single_transform(result, transform, base_url)
        if result is None:
            break
    return result


def _safe_template_format(template: str, context: dict[str, Any]) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_SafeDict(context))


def _extract_text_from_node(node: Any) -> str | None:
    return _coerce_value(node)


def _extract_html_from_node(node: Any) -> str | None:
    if isinstance(node, HtmlElement):
        try:
            return html.tostring(node, encoding="unicode")
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return None
    return _coerce_value(node)


def _extract_attribute_from_node(node: Any, attr: str | None) -> str | None:
    if not attr:
        return None
    if isinstance(node, HtmlElement):
        value = node.get(attr)
        return value.strip() if isinstance(value, str) and value.strip() else None
    return None


def _extract_regex_from_text(text: str, field: dict[str, Any]) -> str | None:
    pattern = field.get("pattern") or field.get("regex")
    if not isinstance(pattern, str) or not text:
        return None
    flags = 0
    if field.get("ignore_case") is True:
        flags |= regex.IGNORECASE
    try:
        compiled = regex.compile(pattern, flags)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        return None
    try:
        match = compiled.search(text, timeout=1.0)
    except TimeoutError:
        logger.warning(f"Regex timeout for pattern: {pattern[:50]}")
        return None
    if not match:
        return None
    group = field.get("group")
    try:
        return match.group(group) if group is not None else match.group(0)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        return match.group(0)


def _extract_list_items(
    node: HtmlElement,
    field: dict[str, Any],
    *,
    base_url: str,
    context: dict[str, Any],
) -> list[Any] | None:
    selector = _field_selector(field)
    nodes = _select_nodes(node, selector, context_sensitive=True) if selector else []
    if not nodes:
        return None
    item_selector = _normalize_selector_expr(
        field.get("item_selector") or field.get("itemSelector"),
        css=field.get("itemCss") or field.get("item_css"),
        xpath=field.get("itemXpath") or field.get("item_xpath"),
    )
    item_type = str(field.get("itemType") or field.get("item_type") or "text").strip().lower()
    attr = field.get("attribute") or field.get("attr")
    join_with = str(field.get("join_with") or " ")
    values: list[Any] = []
    for match in nodes:
        target_nodes = [match]
        if item_selector and isinstance(match, HtmlElement):
            target_nodes = _select_nodes(match, item_selector, context_sensitive=True)
        if not target_nodes:
            continue
        if item_type == "attribute":
            value = _extract_attribute_from_node(target_nodes[0], str(attr) if attr else None)
        elif item_type == "html":
            value = _extract_html_from_node(target_nodes[0])
        elif item_type == "regex":
            base_text = _extract_text_from_node(target_nodes[0]) or ""
            value = _extract_regex_from_text(base_text, field)
        else:
            if len(target_nodes) > 1:
                value = _reduce_matches(target_nodes, join_with)
            else:
                value = _extract_text_from_node(target_nodes[0])
        if value:
            values.append(value)
    if not values:
        return None
    transforms = field.get("transforms")
    return _apply_transforms(values, transforms, base_url)


def _extract_fields_from_node(
    node: HtmlElement,
    fields: Sequence[dict[str, Any]],
    *,
    base_url: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    computed_fields: list[dict[str, Any]] = []
    ctx = dict(context or {})

    for field in fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        field_type = str(field.get("type") or "text").strip().lower()
        if field_type == "computed":
            computed_fields.append(field)
            continue

        if field_type == "nested":
            selector = _field_selector(field)
            matches = _select_nodes(node, selector, context_sensitive=True) if selector else [node]
            nested_fields = _normalize_field_definitions(field.get("fields") or {})
            value = None
            if matches and nested_fields:
                value = _extract_fields_from_node(matches[0], nested_fields, base_url=base_url, context=ctx)
        elif field_type == "nested_list":
            selector = _field_selector(field)
            matches = _select_nodes(node, selector, context_sensitive=True) if selector else []
            nested_fields = _normalize_field_definitions(field.get("fields") or {})
            value = None
            if matches and nested_fields:
                items: list[dict[str, Any]] = []
                for match in matches:
                    if not isinstance(match, HtmlElement):
                        continue
                    item = _extract_fields_from_node(match, nested_fields, base_url=base_url, context=ctx)
                    if item:
                        items.append(item)
                if items:
                    value = items
        elif field_type == "list":
            value = _extract_list_items(node, field, base_url=base_url, context=ctx)
        else:
            selector = _field_selector(field)
            matches = _select_nodes(node, selector, context_sensitive=True) if selector else []
            if not matches:
                value = None
            elif field_type == "attribute":
                attr = field.get("attribute") or field.get("attr")
                value = _extract_attribute_from_node(matches[0], str(attr) if attr else None)
            elif field_type == "html":
                value = _extract_html_from_node(matches[0])
            elif field_type == "regex":
                base_text = _extract_text_from_node(matches[0]) or ""
                value = _extract_regex_from_text(base_text, field)
            else:
                join_with = str(field.get("join_with") or " ")
                value = _reduce_matches(matches, join_with) if len(matches) > 1 else _extract_text_from_node(matches[0])

        value = _apply_transforms(value, field.get("transforms"), base_url)
        if value is not None:
            extracted[name] = value
            ctx[name] = value

    for field in computed_fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        value = None
        if "template" in field and isinstance(field.get("template"), str):
            value = _safe_template_format(field["template"], ctx)
        else:
            source = field.get("from")
            if isinstance(source, list):
                join_with = str(field.get("join_with") or " ")
                parts = [str(ctx.get(item, "")) for item in source]
                value = join_with.join(part for part in parts if part)
            elif isinstance(source, str):
                value = ctx.get(source)
            elif "value" in field:
                value = field.get("value")
        value = _apply_transforms(value, field.get("transforms"), base_url)
        if value is not None:
            extracted[name] = value
            ctx[name] = value

    return extracted


def _is_schema_dsl(rules: dict[str, Any]) -> bool:
    return isinstance(rules.get("fields"), list) or isinstance(rules.get("baseFields"), (list, dict))


def _has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_nonempty_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_nonempty_value(item) for item in value.values())
    return True


def _is_fragile_class_name(value: str) -> bool:
    if not value:
        return False
    if value.startswith("css-") and len(value) >= 8:
        return True
    if len(value) >= 12 and re.fullmatch(r"[A-Za-z0-9_-]+", value):
        digits = sum(ch.isdigit() for ch in value)
        letters = sum(ch.isalpha() for ch in value)
        return digits >= 2 and letters >= 4
    return False


def _fragile_css_classes(selector: str) -> list[str]:
    if not selector.strip().startswith("css:"):
        return []
    expr = selector.strip()[4:]
    classes = re.findall(r"\.([A-Za-z0-9_-]+)", expr)
    attr_classes = re.findall(r'class\s*[*^$]?=\s*["\']([^"\']+)["\']', expr)
    candidates = classes + attr_classes
    return [cls for cls in candidates if _is_fragile_class_name(cls)]

_SCHEMA_SELECTOR_KEYS = (
    "entry_xpath",
    "entry_selector",
    "item_xpath",
    "items_xpath",
    "base_xpath",
    "base_selector",
    "title_xpath",
    "title_selector",
    "summary_xpath",
    "summary_selector",
    "description_xpath",
    "content_xpath",
    "content_selector",
    "author_xpath",
    "author_selector",
    "published_xpath",
    "date_xpath",
    "date_selector",
    "link_xpath",
    "url_xpath",
    "guid_xpath",
    "id_xpath",
)
_PAGINATION_SELECTOR_KEYS = (
    "next_xpath",
    "next_selector",
    "next_link_xpath",
    "next_link_selector",
)

_WATCHLIST_MULTI_KEYS = {
    "summary_xpath",
    "summary_selector",
    "description_xpath",
    "content_xpath",
    "content_selector",
    "entry_xpath",
    "entry_selector",
    "item_xpath",
    "items_xpath",
}


def _iter_rule_selectors(rules: dict[str, Any]) -> list[tuple[str, str]]:
    selectors: list[tuple[str, str]] = []
    for key in _SCHEMA_SELECTOR_KEYS:
        value = rules.get(key)
        for expr in _ensure_sequence(value):
            selectors.append((key, expr))
    pagination = rules.get("pagination")
    if isinstance(pagination, dict):
        for key in _PAGINATION_SELECTOR_KEYS:
            value = pagination.get(key)
            for expr in _ensure_sequence(value):
                selectors.append((f"pagination.{key}", expr))
    alternates = rules.get("alternates")
    if isinstance(alternates, list):
        for idx, alt in enumerate(alternates):
            if not isinstance(alt, dict):
                continue
            for key, expr in _iter_rule_selectors(alt):
                selectors.append((f"alternates[{idx}].{key}", expr))
    return selectors


def _iter_watchlist_selector_specs(rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for key in _SCHEMA_SELECTOR_KEYS:
        value = rules.get(key)
        for expr in _ensure_sequence(value):
            specs.append(
                {
                    "key": key,
                    "selector": expr,
                    "allow_multiple": key in _WATCHLIST_MULTI_KEYS,
                    "expect_nonzero": True,
                    "check_html": True,
                }
            )
    pagination = rules.get("pagination")
    if isinstance(pagination, dict):
        for key in _PAGINATION_SELECTOR_KEYS:
            value = pagination.get(key)
            for expr in _ensure_sequence(value):
                specs.append(
                    {
                        "key": f"pagination.{key}",
                        "selector": expr,
                        "allow_multiple": True,
                        "expect_nonzero": False,
                        "check_html": True,
                    }
                )
    alternates = rules.get("alternates")
    if isinstance(alternates, list):
        for idx, alt in enumerate(alternates):
            if not isinstance(alt, dict):
                continue
            for spec in _iter_watchlist_selector_specs(alt):
                alt_spec = dict(spec)
                alt_spec["key"] = f"alternates[{idx}].{spec['key']}"
                specs.append(alt_spec)
    return specs


def _iter_schema_dsl_selector_specs(rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    base_selector = _base_selector(rules)
    if base_selector:
        specs.append(
            {
                "key": "baseSelector",
                "selector": base_selector,
                "allow_multiple": False,
                "expect_nonzero": True,
                "check_html": True,
            }
        )

    def _walk_fields(fields: Sequence[dict[str, Any]], prefix: str) -> None:
        for field in fields:
            name = str(field.get("name") or "").strip()
            if not name:
                continue
            field_type = str(field.get("type") or "text").strip().lower()
            selector = _field_selector(field)
            if selector:
                specs.append(
                    {
                        "key": f"{prefix}{name}",
                        "selector": selector,
                        "allow_multiple": field_type in {"list", "nested_list"},
                        "expect_nonzero": True,
                        "check_html": True,
                    }
                )
            item_selector = _normalize_selector_expr(
                field.get("item_selector") or field.get("itemSelector"),
                css=field.get("itemCss") or field.get("item_css"),
                xpath=field.get("itemXpath") or field.get("item_xpath"),
            )
            if item_selector:
                specs.append(
                    {
                        "key": f"{prefix}{name}.item_selector",
                        "selector": item_selector,
                        "allow_multiple": True,
                        "expect_nonzero": False,
                        "check_html": False,
                    }
                )
            if field_type in {"nested", "nested_list"}:
                nested_fields = _normalize_field_definitions(field.get("fields") or {})
                if nested_fields:
                    _walk_fields(nested_fields, f"{prefix}{name}.")

    base_fields = _normalize_field_definitions(rules.get("baseFields") or [])
    fields = _normalize_field_definitions(rules.get("fields") or [])
    _walk_fields(base_fields, "baseFields.")
    _walk_fields(fields, "fields.")
    return specs


def validate_selector_rules(
    rules: dict[str, Any],
    *,
    html_text: str | None = None,
    include_counts: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    selector_counts: dict[str, int] = {}
    compile_specs: list[dict[str, Any]] = [
        {"key": key, "selector": expr} for key, expr in _iter_rule_selectors(rules or {})
    ]
    dsl_specs = _iter_schema_dsl_selector_specs(rules or {}) if _is_schema_dsl(rules or {}) else []
    compile_specs.extend(dsl_specs)

    for spec in compile_specs:
        key = spec.get("key")
        expr = spec.get("selector")
        stripped = (expr or "").strip()
        if not stripped:
            continue
        safety_error = _selector_safety_error(stripped)
        if safety_error:
            errors.append({"key": key, "selector": stripped, "error": safety_error})
            continue
        if stripped.startswith("css:"):
            css_expr = stripped[4:].strip()
            if not css_expr:
                continue
            try:
                from lxml.cssselect import CSSSelector

                CSSSelector(css_expr)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
                errors.append({"key": key, "selector": stripped, "error": str(exc)})
            continue
        try:
            XPath(stripped)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
            errors.append({"key": key, "selector": stripped, "error": str(exc)})

    if html_text:
        try:
            document = html.fromstring(html_text)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
            warnings.append({"key": "document", "selector": "", "warning": "html_parse_failed", "detail": str(exc)})
            return {"errors": errors, "warnings": warnings}
        specs = _iter_watchlist_selector_specs(rules or {})
        specs.extend(dsl_specs)
        for spec in specs:
            if not spec.get("check_html", True):
                continue
            expr = spec.get("selector")
            stripped = (expr or "").strip()
            if not stripped:
                continue
            matches = _select_nodes(document, stripped)
            count = len(matches)
            if include_counts:
                selector_counts[str(spec.get("key"))] = count
            if spec.get("expect_nonzero", True) and count == 0:
                warnings.append({"key": spec.get("key"), "selector": stripped, "warning": "no_matches"})
            if not spec.get("allow_multiple", False) and count > 1:
                warnings.append(
                    {"key": spec.get("key"), "selector": stripped, "warning": "non_unique_selector", "count": count}
                )
            if stripped.startswith("css:"):
                for cls in _fragile_css_classes(stripped):
                    warnings.append(
                        {
                            "key": spec.get("key"),
                            "selector": stripped,
                            "warning": "fragile_selector",
                            "detail": f"fragile class '{cls}'",
                        }
                    )

    result = {"errors": errors, "warnings": warnings}
    if include_counts:
        result["selector_counts"] = selector_counts
    return result


def extract_schema_fields(html_text: str, base_url: str, rules: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": base_url,
        "extraction_successful": False,
    }
    if not html_text:
        return result
    if not isinstance(rules, dict) or not rules:
        return result
    try:
        document = html.fromstring(html_text)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        result["error"] = f"HTML parse failed: {exc}"
        return result

    with contextlib.suppress(_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS):
        document.make_links_absolute(base_url)

    if _is_schema_dsl(rules):
        schema_name = rules.get("name")
        base_selector = _base_selector(rules)
        nodes: list[HtmlElement] = []
        if base_selector:
            nodes.extend([n for n in _select_nodes(document, base_selector) if isinstance(n, HtmlElement)])
        base_node = nodes[0] if nodes else document

        schema_fields: dict[str, Any] = {}
        base_fields = _normalize_field_definitions(rules.get("baseFields") or [])
        fields = _normalize_field_definitions(rules.get("fields") or [])
        if base_fields:
            schema_fields.update(_extract_fields_from_node(base_node, base_fields, base_url=base_url, context={}))
        if fields:
            schema_fields.update(_extract_fields_from_node(base_node, fields, base_url=base_url, context=schema_fields))

        if isinstance(schema_name, str) and schema_name.strip():
            result["schema_name"] = schema_name.strip()
        result["schema_fields"] = schema_fields

        for key in ("title", "summary", "content", "author", "published", "published_raw", "date"):
            if key not in schema_fields:
                continue
            value = schema_fields.get(key)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                joined = "\n".join(item.strip() for item in value if item and item.strip())
                result[key] = joined if joined else value
            else:
                result[key] = value

        result["extraction_successful"] = any(_has_nonempty_value(value) for value in schema_fields.values())
        return result

    base_selectors = (
        rules.get("base_xpath")
        or rules.get("base_selector")
        or rules.get("entry_xpath")
        or rules.get("entry_selector")
        or rules.get("item_xpath")
        or rules.get("items_xpath")
    )
    nodes: list[HtmlElement] = []
    for selector in _ensure_sequence(base_selectors):
        nodes.extend([n for n in _select_nodes(document, selector) if isinstance(n, HtmlElement)])
    base_node = nodes[0] if nodes else document

    summary_join = str(rules.get("summary_join_with") or " ")
    content_join = str(rules.get("content_join_with") or "\n")

    title = _extract_value(
        base_node,
        rules.get("title_xpath") or rules.get("title_selector"),
        join=False,
    )
    summary = _extract_value(
        base_node,
        rules.get("summary_xpath")
        or rules.get("description_xpath")
        or rules.get("summary_selector"),
        join=True,
        join_with=summary_join,
    )
    content = _extract_value(
        base_node,
        rules.get("content_xpath") or rules.get("content_selector"),
        join=True,
        join_with=content_join,
    )
    author = _extract_value(
        base_node,
        rules.get("author_xpath") or rules.get("author_selector"),
        join=False,
    )
    published_raw = _extract_value(
        base_node,
        rules.get("published_xpath")
        or rules.get("date_xpath")
        or rules.get("date_selector"),
        join=False,
    )

    if title:
        result["title"] = title
    if summary:
        result["summary"] = summary
    if content:
        result["content"] = content
    if author:
        result["author"] = author
    if published_raw:
        result["published_raw"] = published_raw
        fmt = rules.get("published_format") or rules.get("date_format")
        parsed = _normalize_datetime(published_raw, fmt if isinstance(fmt, str) else None)
        if parsed:
            result["published"] = parsed

    result["extraction_successful"] = bool(content or summary or title)
    return result


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        return default


def _normalize_datetime(raw: str, fmt: str | None = None) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if fmt:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            pass
    # Try dateutil if available
    try:
        from dateutil import parser as dateutil_parser  # type: ignore

        dt = dateutil_parser.parse(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        pass
    # Fallback to email.utils
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        pass
    return text


def parse_scraped_items(html_text: str, base_url: str, rules: dict[str, Any]) -> dict[str, Any]:
    """Parse HTML into structured items using XPath/CSS rules.

    Returns dict: { "items": [...], "next_pages": [...] } to support pagination-aware callers.
    """
    result: dict[str, Any] = {"items": [], "next_pages": []}
    if not html_text:
        return result
    try:
        document = html.fromstring(html_text)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"parse_scraped_items HTML parse failed: {exc}")
        return result

    with contextlib.suppress(_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS):
        document.make_links_absolute(base_url)

    def _gather_items(rule_set: dict[str, Any], *, seen: set[str], items: list[dict[str, Any]], limit: int | None) -> None:
        entry_selectors = (
            rule_set.get("entry_xpath")
            or rule_set.get("item_xpath")
            or rule_set.get("items_xpath")
            or rule_set.get("entry_selector")
            or rule_set.get("item_selector")
            or rules.get("entry_xpath")
            or rules.get("item_xpath")
            or rules.get("items_xpath")
            or rules.get("entry_selector")
            or rules.get("item_selector")
        )
        nodes: list[HtmlElement] = []
        for selector in _ensure_sequence(entry_selectors) or ["//article", "//item"]:
            nodes.extend([n for n in _select_nodes(document, selector) if isinstance(n, HtmlElement)])
        if not nodes:
            nodes = [document]

        summary_join = str(rule_set.get("summary_join_with") or rules.get("summary_join_with") or " ")
        content_join = str(rule_set.get("content_join_with") or rules.get("content_join_with") or "\n")

        for node in nodes:
            link = _extract_value(
                node,
                rule_set.get("link_xpath")
                or rule_set.get("url_xpath")
                or rules.get("link_xpath")
                or rules.get("url_xpath"),
                join=False,
            )
            if not link:
                continue
            link = link.strip()
            if not link or link in seen:
                continue
            seen.add(link)

            item: dict[str, Any] = {"url": link}
            title = _extract_value(
                node,
                rule_set.get("title_xpath") or rule_set.get("title_selector") or rules.get("title_xpath") or rules.get("title_selector"),
                join=False,
            )
            if title:
                item["title"] = title
            summary = _extract_value(
                node,
                rule_set.get("summary_xpath")
                or rule_set.get("description_xpath")
                or rule_set.get("summary_selector")
                or rules.get("summary_xpath")
                or rules.get("description_xpath")
                or rules.get("summary_selector"),
                join=True,
                join_with=summary_join,
            )
            if summary:
                item["summary"] = summary
            content = _extract_value(
                node,
                rule_set.get("content_xpath")
                or rule_set.get("content_selector")
                or rules.get("content_xpath")
                or rules.get("content_selector"),
                join=True,
                join_with=content_join,
            )
            if content:
                item["content"] = content
            author = _extract_value(
                node,
                rule_set.get("author_xpath") or rule_set.get("author_selector") or rules.get("author_xpath") or rules.get("author_selector"),
                join=False,
            )
            if author:
                item["author"] = author
            guid = _extract_value(
                node,
                rule_set.get("guid_xpath") or rule_set.get("id_xpath") or rules.get("guid_xpath") or rules.get("id_xpath"),
                join=False,
            )
            if guid:
                item["guid"] = guid
            published_raw = _extract_value(
                node,
                rule_set.get("published_xpath")
                or rule_set.get("date_xpath")
                or rule_set.get("date_selector")
                or rules.get("published_xpath")
                or rules.get("date_xpath")
                or rules.get("date_selector"),
                join=False,
            )
            if published_raw:
                item["published_raw"] = published_raw
                fmt = rule_set.get("published_format") or rules.get("published_format") or rule_set.get("date_format") or rules.get("date_format")
                parsed = _normalize_datetime(published_raw, fmt if isinstance(fmt, str) else None)
                if parsed:
                    item["published"] = parsed

            items.append(item)
            if limit is not None and len(items) >= limit:
                break
        return

    limit = _coerce_int(rules.get("limit") or rules.get("max_items"))
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    _gather_items(rules, seen=seen_urls, items=items, limit=limit)

    alternates = rules.get("alternates")
    if isinstance(alternates, list):
        for alt in alternates:
            if not isinstance(alt, dict):
                continue
            if limit is not None and len(items) >= limit:
                break
            merged = {**rules, **alt}
            _gather_items(merged, seen=seen_urls, items=items, limit=limit)
            if limit is not None and len(items) >= limit:
                break

    pagination_cfg = rules.get("pagination") if isinstance(rules.get("pagination"), dict) else None
    next_pages: list[str] = []
    if pagination_cfg:
        candidate_selectors = _ensure_sequence(
            pagination_cfg.get("next_xpath")
            or pagination_cfg.get("next_selector")
            or pagination_cfg.get("next_link_xpath")
            or pagination_cfg.get("next_link_selector")
        )
        attr = pagination_cfg.get("next_attribute") or "href"
        for selector in candidate_selectors:
            matches = _select_nodes(document, selector)
            for match in matches:
                url = None
                if isinstance(match, HtmlElement):
                    url = match.get(attr)
                    if not url:
                        url = _coerce_value(match)
                else:
                    url = _coerce_value(match)
                if not url:
                    continue
                absolute = urljoin(base_url, url)
                if absolute not in next_pages:
                    next_pages.append(absolute)
    result["items"] = items
    result["next_pages"] = next_pages
    return result


async def fetch_rss_items(urls: list[str], *, limit: int = 10, tenant_id: str = "default") -> list[dict[str, Any]]:
    """Fetch RSS/Atom feed items for the given URLs.

    Uses the workflows adapter implementation for consistency with URL
    allowlisting and parsing heuristics. In TEST_MODE, returns a single fake
    item to keep tests offline.
    """
    urls = [u for u in (urls or []) if isinstance(u, str) and u.strip()]
    if not urls:
        return []

    # Offline mode for unit tests
    if _in_test_mode():
        return [{"title": "Test Item", "url": "https://example.com/x", "summary": "Test", "published": None}][:limit]

    try:
        from tldw_Server_API.app.core.Workflows.adapters import run_rss_fetch_adapter  # reuse existing parser
        cfg = {"urls": urls, "limit": limit, "include_content": True}
        ctx = {"tenant_id": tenant_id}
        res = await run_rss_fetch_adapter(cfg, ctx)
        items = []
        for r in (res.get("results") or []):
            items.append({
                "title": r.get("title") or "",
                "url": r.get("link") or r.get("url") or "",
                "summary": r.get("summary"),
                "published": r.get("published"),
                "guid": r.get("guid"),
            })
        return items[:limit]
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"fetch_rss_items failed: {e}")
        return []


def fetch_site_article(url: str) -> dict[str, Any] | None:
    """Fetch and extract a single site article.

    Uses the blocking path from Article_Extractor, which works without a
    Playwright runtime. Returns None on failure.
    """
    try:
        from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
            ContentMetadataHandler,  # type: ignore
            scrape_article_blocking,
        )
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Article extractor import failed: {e}")
        return None

    try:
        data = scrape_article_blocking(url)
        if not data:
            return None
        # Normalize
        title = data.get("title") or "Untitled"
        author = data.get("author") or None
        content = data.get("content") or ""
        try:
            content = ContentMetadataHandler.strip_metadata(content)  # type: ignore[attr-defined]
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            pass
        return {"title": title, "url": url, "content": content, "author": author}
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"fetch_site_article failed for {url}: {e}")
        return None


async def fetch_site_article_async(url: str) -> dict[str, Any] | None:
    """Async wrapper for blocking article extraction."""
    try:
        return await asyncio.to_thread(fetch_site_article, url)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"fetch_site_article_async failed for {url}: {exc}")
        return None


async def fetch_site_top_links(base_url: str, *, top_n: int = 10, method: str = "frontpage") -> list[str]:
    """Discover top-N content links from a site.

    - method="frontpage": fetch homepage and extract likely-article links
    - method="sitemap": try EnhancedWebScraper.scrape_sitemap to pull URLs only

    Returns a list of URLs (same-origin) deduplicated, up to top_n.
    TEST_MODE: returns [base_url] repeated to satisfy callers without network.
    """
    if top_n <= 0:
        return []

    if _in_test_mode():
        # Provide stable deterministic links
        return [base_url] * min(top_n, 3)

    # Try using EnhancedWebScraper when available
    try:
        from urllib.parse import urljoin, urlparse

        from bs4 import BeautifulSoup

        from tldw_Server_API.app.core.http_client import afetch
        from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import is_content_page
        from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        headers = {
            "User-Agent": "tldw-watchlist/0.1 (+https://github.com/your-org/tldw_server2)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async def _fetch_text(url: str, timeout: float) -> tuple[int, str]:
            resp = None
            try:
                resp = await afetch(method="GET", url=url, headers=headers, timeout=timeout)
                return int(resp.status_code), resp.text or ""
            finally:
                await _close_response(resp)

        # Auto-detect sitemap via robots.txt or common path when method='auto'
        async def _detect_sitemap(u: str) -> str | None:
            try:
                robots_url = urljoin(origin, "/robots.txt")
                status, txt = await _fetch_text(robots_url, timeout=6)
                if status // 100 == 2:
                    # Look for Sitemap lines
                    for line in txt.splitlines():
                        if line.lower().startswith("sitemap:"):
                            sitemap_url = line.split(":", 1)[1].strip()
                            return sitemap_url
                # Try common location
                common = urljoin(origin, "/sitemap.xml")
                status, _ = await _fetch_text(common, timeout=6)
                if status // 100 == 2:
                    return common
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                return None
            return None

        effective_method = method or "auto"
        sitemap_url_for_auto: str | None = None
        if effective_method == "auto":
            sitemap_url_for_auto = await _detect_sitemap(base_url)
            effective_method = "sitemap" if sitemap_url_for_auto else "frontpage"

        # Sitemap method
        if effective_method == "sitemap" or base_url.endswith(".xml") or "sitemap" in base_url:
            scraper = EnhancedWebScraper()
            # We only need URLs; call scrape_sitemap then pluck the url fields
            sitemap_to_use = sitemap_url_for_auto or base_url
            results = await scraper.scrape_sitemap(sitemap_to_use, filter_func=is_content_page, max_urls=top_n)
            urls = []
            for r in results:
                u = r.get("url") or r.get("source_url")
                if not u:
                    continue
                if urlparse(u).netloc == parsed.netloc:
                    urls.append(u)
            # Dedup while preserving order
            seen = set()
            uniq = []
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    uniq.append(u)
            return uniq[:top_n]

        # Frontpage method: fetch HTML and pull article-like links
        status, html = await _fetch_text(base_url, timeout=15)
        if status // 100 != 2:
            return [base_url]
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            href = urljoin(origin, href)
            # Same origin and looks like content
            try:
                if urlparse(href).netloc != parsed.netloc:
                    continue
                if not is_content_page(href):
                    continue
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                continue
            links.append(href)
        # Dedup preserve order
        seen = set()
        uniq = []
        for u in links:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq[:top_n] if uniq else [base_url]
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"fetch_site_top_links fallback: {e}")
        return [base_url]


async def fetch_site_items_with_rules(
    base_url: str,
    rules: dict[str, Any],
    *,
    tenant_id: str = "default",
    timeout: float = 10.0,
    fetch_diagnostics: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fetch a list page and extract items using scrape rules."""
    list_url = str(rules.get("list_url") or base_url or "").strip()
    if not list_url:
        return []

    limit = _coerce_int(rules.get("limit") or rules.get("max_items"), default=10)
    if limit is not None and limit < 0:
        limit = 0
    if limit == 0:
        return []

    pagination_cfg = rules.get("pagination") if isinstance(rules.get("pagination"), dict) else {}
    max_pages = _coerce_int(pagination_cfg.get("max_pages"), default=1)
    if max_pages is None or max_pages < 1:
        max_pages = 1

    if _in_test_mode():
        max_items = limit if limit is not None else 3
        max_items = min(max_items, 5)
        samples: list[dict[str, Any]] = []
        for idx in range(max_items):
            url = f"{list_url.rstrip('/')}/test-scrape-{idx + 1}"
            samples.append(
                {
                    "title": f"Test scraped item {idx + 1}",
                    "url": url,
                    "summary": "Test summary from scrape rules.",
                    "content": "Test content from scrape rules.",
                }
            )
        return samples

    headers = {
        "User-Agent": "tldw-watchlist/0.1 (+https://github.com/your-org/tldw_server2)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
    }

    queue: list[str] = [list_url]
    visited: set[str] = set()
    seen_items: set[str] = set()
    collected: list[dict[str, Any]] = []

    def emit_fetch_diagnostic(event: dict[str, Any]) -> None:
        if fetch_diagnostics is None:
            return
        with contextlib.suppress(*_WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS):
            fetch_diagnostics(event)

    try:
        from tldw_Server_API.app.core.http_client import afetch
        while queue and len(visited) < max_pages:
            page_url = queue.pop(0)
            if page_url in visited:
                continue
            visited.add(page_url)

            allowed = False
            try:
                allowed = is_url_allowed_for_tenant(page_url, tenant_id)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                allowed = is_url_allowed(page_url)
            if not allowed:
                logger.debug(f"Scrape rules blocked by URL policy: {page_url}")
                emit_fetch_diagnostic({"url": page_url, "error": "blocked_by_url_policy"})
                continue

            resp = None
            try:
                resp = await afetch(method="GET", url=page_url, headers=headers, timeout=timeout)
                status_code = int(resp.status_code)
                emit_fetch_diagnostic({"url": page_url, "status": status_code})
                if status_code // 100 != 2:
                    logger.debug(f"fetch_site_items_with_rules HTTP {status_code} for {page_url}")
                    continue
                parsed = parse_scraped_items(resp.text or "", page_url, rules)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"fetch_site_items_with_rules request failed ({page_url}): {exc}")
                emit_fetch_diagnostic({"url": page_url, "error": str(exc) or exc.__class__.__name__})
                continue
            finally:
                await _close_response(resp)

            page_items = parsed.get("items") or []
            for item in page_items:
                url = item.get("url")
                if not url or url in seen_items:
                    continue
                seen_items.add(url)
                collected.append(item)
                if limit is not None and len(collected) >= limit:
                    break
            if limit is not None and len(collected) >= limit:
                break

            next_pages = parsed.get("next_pages") or []
            for nxt in next_pages:
                if not nxt or nxt in visited or nxt in queue:
                    continue
                if len(visited) + len(queue) >= max_pages:
                    break
                queue.append(nxt)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"fetch_site_items_with_rules pagination failed: {exc}")

    if limit is not None:
        return collected[:limit]
    return collected


async def fetch_rss_feed(
    url: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
    timeout: float = 8.0,
    tenant_id: str = "default",
) -> dict[str, Any]:
    """Fetch a single RSS/Atom feed with conditional headers.

    Returns dict:
      - status: int HTTP code (200/304/429/other)
      - items: list[dict] when 200
      - etag: str|None (from response headers)
      - last_modified: str|None (from response headers)
      - retry_after: int seconds (only when 429 and header present)
    """
    try:
        if not (url.startswith("http://") or url.startswith("https://")):
            return {"status": 400, "items": []}
        allowed = False
        try:
            allowed = is_url_allowed_for_tenant(url, tenant_id)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            allowed = is_url_allowed(url)
        if not allowed:
            return {"status": 403, "items": []}

        headers = {
            "Accept": "application/atom+xml, application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "tldw-watchlist/0.1 (+https://github.com/your-org/tldw_server2)"
        }
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        from tldw_Server_API.app.core.http_client import afetch
        resp = None
        status = 0
        resp_headers: dict[str, Any] = {}
        text = ""
        try:
            resp = await afetch(method="GET", url=url, headers=headers, timeout=timeout)
            if resp is not None:
                status = int(resp.status_code)
                resp_headers = dict(getattr(resp, "headers", {}) or {})
                text = resp.text or ""
        finally:
            await _close_response(resp)

        if resp is None:
            return {"status": 500, "items": []}
        if status == 0:
            return {"status": 500, "items": []}
        # Retry-After handling
        if status == 429:
            ra = resp_headers.get("Retry-After")
            retry_after_secs = None
            if ra:
                ra = ra.strip()
                # Seconds or HTTP-date
                try:
                    retry_after_secs = int(ra)
                except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                    from email.utils import parsedate_to_datetime
                    try:
                        dt = parsedate_to_datetime(ra)
                        import datetime as _dt
                        retry_after_secs = max(0, int((dt - _dt.datetime.utcnow().replace(tzinfo=dt.tzinfo)).total_seconds()))
                    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                        retry_after_secs = None
            return {"status": 429, "items": [], "retry_after": retry_after_secs}

        if status == 304:
            return {
                "status": 304,
                "items": [],
                "etag": resp_headers.get("ETag"),
                "last_modified": resp_headers.get("Last-Modified"),
            }

        if status // 100 != 2:
            return {"status": status, "items": []}

        try:
            root = ET.fromstring(text)
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            return {
                "status": status,
                "items": [],
                "etag": resp_headers.get("ETag"),
                "last_modified": resp_headers.get("Last-Modified"),
            }

        def _find_text(node, names):
            for n in names:
                x = node.find(n)
                if x is not None and (x.text or "").strip():
                    return x.text.strip()
            return None

        items_nodes = root.findall('.//item')
        if not items_nodes:
            items_nodes = root.findall('.//{http://www.w3.org/2005/Atom}entry')
        items: list[dict[str, Any]] = []
        atom_link_tag = "{http://www.w3.org/2005/Atom}link"
        atom_title_tag = "{http://www.w3.org/2005/Atom}title"
        atom_summary_tag = "{http://www.w3.org/2005/Atom}summary"
        atom_content_tag = "{http://www.w3.org/2005/Atom}content"
        atom_updated_tag = "{http://www.w3.org/2005/Atom}updated"
        atom_published_tag = "{http://www.w3.org/2005/Atom}published"
        atom_id_tag = "{http://www.w3.org/2005/Atom}id"
        for it in items_nodes:
            title = _find_text(it, ["title", atom_title_tag]) or ""

            link = ""
            link_nodes = list(it.findall("link")) + list(it.findall(atom_link_tag))
            preferred_link = ""
            fallback_link = ""
            for node in link_nodes:
                candidate = (node.get("href") or (node.text or "")).strip()
                if not candidate:
                    continue
                rel = (node.get("rel") or "").lower()
                if rel == "alternate" and not preferred_link:
                    preferred_link = candidate
                elif rel not in {"self"} and not fallback_link:
                    fallback_link = candidate
            link = preferred_link or fallback_link or _find_text(it, ["link", atom_link_tag]) or ""

            summary = _find_text(it, ["description", atom_summary_tag, atom_content_tag]) or ""
            published = _find_text(it, ["pubDate", atom_updated_tag, atom_published_tag]) or None
            guid = _find_text(it, ["guid", atom_id_tag]) or None
            rec = {"title": title, "url": link or "", "summary": summary, "published": published}
            if guid:
                rec["guid"] = guid
            items.append(rec)
        # Atom RFC5005: collect top-level archive paging links when present
        atom_links: list[dict[str, str]] = []
        try:
            # Look for link rel="prev-archive"/"next-archive" on the root <feed>
            for ln in list(root.findall(atom_link_tag)) + list(root.findall("link")):
                href = (ln.get("href") or (ln.text or "")).strip()
                if not href:
                    continue
                rel = (ln.get("rel") or "").strip().lower()
                # Resolve relative IRIs against the feed URL to produce an absolute URL
                try:
                    resolved = urljoin(url, href)
                except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                    resolved = href  # fall back to original if resolution fails
                if rel in {"prev-archive", "next-archive", "current", "self"}:
                    atom_links.append({"rel": rel, "href": resolved})
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            atom_links = []

        return {
            "status": 200,
            "items": items,
            "etag": resp_headers.get("ETag"),
            "last_modified": resp_headers.get("Last-Modified"),
            "atom_links": atom_links,
        }
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"fetch_rss_feed error: {e}")
        return {"status": 500, "items": []}


def discover_hub_url(
    xml_text: str,
    response_headers: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Return (hub_url, self_url) from feed XML and/or HTTP Link headers.

    Parses both:
    - HTTP ``Link`` header: ``<https://hub.example.com>; rel="hub"``
    - Atom/RSS XML: ``<link rel="hub" href="..." />`` and ``<link rel="self" href="..." />``
    """
    hub_url: str | None = None
    self_url: str | None = None

    # Phase 1: Check HTTP Link headers
    if response_headers:
        link_header = response_headers.get("Link") or response_headers.get("link") or ""
        for part in link_header.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                url_part, *attrs = part.split(";")
                url_val = url_part.strip().strip("<>").strip()
                rel_val = ""
                for attr in attrs:
                    attr = attr.strip()
                    if attr.lower().startswith("rel="):
                        rel_val = attr.split("=", 1)[1].strip().strip('"').strip("'").lower()
                if rel_val == "hub" and not hub_url:
                    hub_url = url_val
                elif rel_val == "self" and not self_url:
                    self_url = url_val
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                continue

    # Phase 2: Parse XML for <link rel="hub"> and <link rel="self">
    try:
        root = ET.fromstring(xml_text)
        atom_link_tag = "{http://www.w3.org/2005/Atom}link"
        # Search both direct children and descendants (RSS puts atom:link inside <channel>)
        for ln in list(root.findall(atom_link_tag)) + list(root.findall(".//" + atom_link_tag)) + list(root.findall("link")) + list(root.findall(".//link")):
            href = (ln.get("href") or (ln.text or "")).strip()
            if not href:
                continue
            rel = (ln.get("rel") or "").strip().lower()
            if rel == "hub" and not hub_url:
                hub_url = href
            elif rel == "self" and not self_url:
                self_url = href
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        pass

    return hub_url, self_url


async def fetch_rss_feed_history(
    url: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
    timeout: float = 8.0,
    tenant_id: str = "default",
    strategy: str = "auto",
    max_pages: int = 1,
    per_page_limit: int | None = None,
    on_304: bool = False,
    stop_on_seen: bool = False,
    seen_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fetch a feed and optionally traverse history pages.

    - strategy: "auto" | "atom" | "wordpress" | "none"
    - max_pages: total pages to fetch including the first page
    - per_page_limit: trim items per page (None = keep all)
    - on_304: when True, still attempt history traversal when the first page returns 304
    """
    try:
        max_pages = int(max_pages)
    except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
        max_pages = 1
    if max_pages < 1:
        max_pages = 1

    # First page with conditional headers
    first = await fetch_rss_feed(
        url,
        etag=etag,
        last_modified=last_modified,
        timeout=timeout,
        tenant_id=tenant_id,
    )
    status = int(first.get("status", 0) or 0)
    agg_items: list[dict[str, Any]] = []
    etag_out = first.get("etag")
    last_mod_out = first.get("last_modified")
    pages_fetched = 0

    def _trim(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if per_page_limit is None or per_page_limit <= 0:
            return items
        return items[: per_page_limit]

    if status == 304 and not on_304:
        return {"status": 304, "items": [], "etag": etag_out, "last_modified": last_mod_out, "pages_fetched": 0, "strategy_used": strategy}

    if status == 429:
        # Pass through rate-limit signal
        out = {k: first.get(k) for k in ("status", "retry_after")}
        out.update({"items": [], "pages_fetched": 0})
        return out

    if status // 100 != 2:
        return {"status": status, "items": [], "pages_fetched": 0, "strategy_used": strategy}

    base_items = list(first.get("items") or [])
    agg_items.extend(_trim(base_items))
    pages_fetched += 1
    stop_triggered = False

    # Early exit
    if max_pages == 1:
        return {
            "status": 200,
            "items": agg_items,
            "etag": etag_out,
            "last_modified": last_mod_out,
            "pages_fetched": pages_fetched,
            "strategy_used": (strategy or "auto").lower(),
            "stop_on_seen_triggered": False,
        }

    # Helper: follow Atom RFC5005 prev-archive links
    async def _follow_atom_prev(href: str, remaining: int) -> int:
        nonlocal agg_items, pages_fetched
        current_url = href
        # Track aggregate keys and DB-seen keys separately
        agg_seen: set[str] = set()
        for it in agg_items:
            key = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
            if key:
                agg_seen.add(key)
        db_seen: set[str] = {k.strip() for k in (seen_keys or []) if isinstance(k, str)}
        fetched_here = 0
        while remaining > 0 and current_url:
            try:
                res = await fetch_rss_feed(current_url, timeout=timeout, tenant_id=tenant_id)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                break
            if int(res.get("status", 0) or 0) // 100 != 2:
                break
            items = list(res.get("items") or [])
            # Dedup across pages and check DB-seen condition
            new: list[dict[str, Any]] = []
            for it in items:
                key = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                if not key or key in agg_seen:
                    continue
                agg_seen.add(key)
                new.append(it)
            if not new:
                break
            if stop_on_seen:
                # If none of the items on this page are new relative to DB-seen keys, stop
                page_new_vs_db = [it for it in new if ((it.get("guid") or it.get("url") or it.get("link") or "").strip()) not in db_seen]
                if not page_new_vs_db:
                    nonlocal stop_triggered
                    stop_triggered = True
                    break
                # Update db_seen with truly-new keys so that further pages respect boundary condition
                for it in page_new_vs_db:
                    k = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                    if k:
                        db_seen.add(k)
            agg_items.extend(_trim(new))
            pages_fetched += 1
            fetched_here += 1
            remaining -= 1
            # Look for next prev-archive
            next_links = [ln for ln in (res.get("atom_links") or []) if ln.get("rel") == "prev-archive" and ln.get("href")]
            current_url = next_links[0]["href"] if next_links else None
        return fetched_here

    # Helper: try common WordPress paged feed patterns
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    def _wp_paged_urls(base: str, pages: int) -> list[str]:
        out: list[str] = []
        try:
            parsed = urlparse(base)
            qs = parse_qs(parsed.query)
            # Case 1: ?feed=rss2 or ?feed=atom → add paged param
            if "feed" in qs:
                for p in range(2, pages + 1):
                    new_qs = qs.copy()
                    new_qs["paged"] = [str(p)]
                    new_url = urlunparse(parsed._replace(query=urlencode(new_qs, doseq=True)))
                    out.append(new_url)
            # Case 2: path endswith /feed or /feed/ → append ?paged=N
            path = parsed.path or ""
            if path.rstrip("/").endswith("/feed"):
                for p in range(2, pages + 1):
                    q = urlencode({"paged": p})
                    base2 = urlunparse(parsed._replace(query=q))
                    out.append(base2)
            # Fallback: append ?paged=N generally
            if not out:
                for p in range(2, pages + 1):
                    q = urlencode({"paged": p})
                    out.append(urlunparse(parsed._replace(query=q)))
        except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
            pass
        # Dedup preserve order
        dedup: list[str] = []
        seen: set[str] = set()
        for u in out:
            if u not in seen:
                seen.add(u)
                dedup.append(u)
        return dedup

    pages_left = max(0, max_pages - 1)
    used_strategy = (strategy or "auto").lower()
    strategy_used = used_strategy

    # Try Atom prev-archive first when auto/atom
    if used_strategy in {"auto", "atom"} and pages_left > 0:
        prev_links = [ln for ln in (first.get("atom_links") or []) if ln.get("rel") == "prev-archive" and ln.get("href")]
        if prev_links:
            consumed = await _follow_atom_prev(prev_links[0]["href"], pages_left)
            pages_left -= consumed
            strategy_used = "atom"

    # Then try WordPress style when auto/wordpress and still have budget
    if used_strategy in {"auto", "wordpress"} and pages_left > 0 and not stop_triggered:
        wp_urls = _wp_paged_urls(url, pages_left + 1)
        # Track DB-seen and aggregate keys
        prior_keys = {(it.get("guid") or it.get("url") or it.get("link") or "").strip() for it in agg_items}
        db_seen_wp: set[str] = {k.strip() for k in (seen_keys or []) if isinstance(k, str)}
        for u in wp_urls:
            if pages_left <= 0:
                break
            try:
                res = await fetch_rss_feed(u, timeout=timeout, tenant_id=tenant_id)
            except _WATCHLISTS_FETCHERS_NONCRITICAL_EXCEPTIONS:
                continue
            if int(res.get("status", 0) or 0) // 100 != 2:
                continue
            items = list(res.get("items") or [])
            if not items:
                continue
            # Dedup vs aggregate and check DB-seen boundary condition
            new: list[dict[str, Any]] = []
            for it in items:
                key = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                if not key or key in prior_keys:
                    continue
                prior_keys.add(key)
                new.append(it)
            if not new:
                continue
            if stop_on_seen:
                page_new_vs_db = [it for it in new if ((it.get("guid") or it.get("url") or it.get("link") or "").strip()) not in db_seen_wp]
                if not page_new_vs_db:
                    stop_triggered = True
                    break
                for it in page_new_vs_db:
                    k = (it.get("guid") or it.get("url") or it.get("link") or "").strip()
                    if k:
                        db_seen_wp.add(k)
            agg_items.extend(_trim(new))
            pages_fetched += 1
            pages_left -= 1
        if pages_fetched > 1:
            strategy_used = "wordpress" if strategy_used == "auto" else strategy_used

    return {
        "status": 200,
        "items": agg_items,
        "etag": etag_out,
        "last_modified": last_mod_out,
        "pages_fetched": pages_fetched,
        "strategy_used": strategy_used,
        "stop_on_seen_triggered": stop_triggered,
    }
