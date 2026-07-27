"""Selector safety, compilation, contextualization, and node selection."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from loguru import logger
from lxml.etree import XPath, XPathError
from lxml.html import HtmlElement

from .caches import (
    _get_css_selector,
    _get_xpath_selector,
    _put_css_selector,
    _put_xpath_selector,
)

_SELECTOR_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    XPathError,
)
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
    except _SELECTOR_NONCRITICAL_EXCEPTIONS:
        logger.warning("Invalid selector guardrail value for {}; using default", env_name)
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


def _selector_validation_error(selector: str) -> str | None:
    """Compile one selector and return its stable public validation error."""
    stripped = selector.strip()
    safety_error = _selector_safety_error(stripped)
    if safety_error:
        return safety_error
    if stripped.startswith("css:"):
        css_expr = stripped[4:].strip()
        if not css_expr:
            return None
        try:
            from lxml.cssselect import CSSSelector

            CSSSelector(css_expr)
        except _SELECTOR_NONCRITICAL_EXCEPTIONS:
            return "selector_invalid"
        return None
    try:
        XPath(stripped)
    except _SELECTOR_NONCRITICAL_EXCEPTIONS:
        return "selector_invalid"
    return None


def _ensure_sequence(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if isinstance(item, str)]


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
            except _SELECTOR_NONCRITICAL_EXCEPTIONS:
                node_tag = None
            if node_tag and token == str(node_tag):
                return expr
        return f".{expr}"
    if expr.startswith("/"):
        try:
            root = node.getroottree().getroot()
        except _SELECTOR_NONCRITICAL_EXCEPTIONS:
            root = None
        if root is not None:
            root_tag = getattr(root, "tag", None)
            if isinstance(root_tag, str) and expr.startswith(f"/{root_tag}"):
                return expr
        if root is not None and root is not node:
            return f".{expr}"
    return expr


def _parse_nth_token(
    token: str,
    allowed: set[str],
) -> tuple[str | None, int | None]:
    match = re.fullmatch(
        r"(?P<tag>[a-zA-Z0-9_-]+)(?::nth-child\((?P<index>\d+)\))?",
        token,
    )
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


def _select_nodes_with_status(
    node: HtmlElement,
    selector: str,
    *,
    context_sensitive: bool = False,
) -> tuple[list[Any], bool]:
    """Select nodes and report sanitized compilation or evaluation failure."""
    expr = selector.strip()
    if not expr:
        return [], False
    safety_error = _selector_safety_error(expr)
    if safety_error:
        logger.debug("Selector rejected by safety guard ({})", safety_error)
        return [], False
    if expr.startswith("css:"):
        css_expr = expr[4:].strip()
        if not css_expr:
            return [], False
        fast_xpath = _css_table_nth_xpath(css_expr)
        if fast_xpath:
            expr = fast_xpath
        else:
            try:
                from lxml.cssselect import CSSSelector

                compiled = _get_css_selector(css_expr)
                if compiled is None:
                    compiled = CSSSelector(css_expr)
                    _put_css_selector(css_expr, compiled)
                return list(compiled(node)), False
            except _SELECTOR_NONCRITICAL_EXCEPTIONS:
                logger.debug("CSS selector compilation or evaluation failed")
                return [], True
    if context_sensitive:
        expr = _contextualize_xpath(expr, node)
    compiled_xpath = _get_xpath_selector(expr)
    if compiled_xpath is None:
        try:
            compiled_xpath = XPath(expr)
            _put_xpath_selector(expr, compiled_xpath)
        except _SELECTOR_NONCRITICAL_EXCEPTIONS:
            logger.debug("XPath selector compilation failed")
            return [], True
    try:
        result = compiled_xpath(node)
    except _SELECTOR_NONCRITICAL_EXCEPTIONS:
        logger.debug("XPath selector evaluation failed")
        return [], True
    if isinstance(result, list):
        return result, False
    return [result], False


def _select_nodes(
    node: HtmlElement,
    selector: str,
    *,
    context_sensitive: bool = False,
) -> list[Any]:
    matches, _failed = _select_nodes_with_status(
        node,
        selector,
        context_sensitive=context_sensitive,
    )
    return matches


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
        except _SELECTOR_NONCRITICAL_EXCEPTIONS:
            return None
    if hasattr(value, "text_content"):
        try:
            text = value.text_content().strip()
            return text or None
        except _SELECTOR_NONCRITICAL_EXCEPTIONS:
            return None
    try:
        text = str(value).strip()
        return text or None
    except _SELECTOR_NONCRITICAL_EXCEPTIONS:
        return None


def _reduce_matches(matches: Sequence[Any], join_with: str) -> str | None:
    parts: list[str] = []
    for match in matches:
        value = _coerce_value(match)
        if value:
            parts.append(value)
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
