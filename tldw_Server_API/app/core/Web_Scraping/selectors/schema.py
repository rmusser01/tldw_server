"""Schema selector normalization, validation, transforms, and extraction."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

from lxml import html
from lxml.html import HtmlElement

from ..safe_regex import search_untrusted, sub_untrusted
from .engine import (
    _coerce_value,
    _ensure_sequence,
    _extract_value,
    _reduce_matches,
    _select_nodes,
    _select_nodes_with_status,
    _selector_validation_error,
)

_SCHEMA_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)


def _normalize_selector_expr(
    selector: str | None,
    *,
    css: str | None = None,
    xpath: str | None = None,
) -> str | None:
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
        return [field for field in fields if isinstance(field, dict)]
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


def _normalize_datetime(raw: str, fmt: str | None = None) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if fmt:
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except _SCHEMA_NONCRITICAL_EXCEPTIONS:
            pass
    try:
        from dateutil import parser as dateutil_parser  # type: ignore

        parsed = dateutil_parser.parse(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except _SCHEMA_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
    except _SCHEMA_NONCRITICAL_EXCEPTIONS:
        pass
    return text


def _apply_single_transform(
    value: str,
    transform: Any,
    base_url: str,
) -> str | None:
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
        if not isinstance(pattern, str):
            return value
        result = sub_untrusted(pattern, str(params.get("repl", "")), value)
        return result.value if result.code is None and result.value is not None else value
    if name == "urljoin":
        try:
            return urljoin(base_url, value)
        except _SCHEMA_NONCRITICAL_EXCEPTIONS:
            return value
    if name == "date_normalize":
        fmt = params.get("format")
        normalized = _normalize_datetime(value, fmt if isinstance(fmt, str) else None)
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
        return [
            item for item in (_apply_transforms(item, transform_list, base_url) for item in value) if item is not None
        ]
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
        except _SCHEMA_NONCRITICAL_EXCEPTIONS:
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
    flags = re.IGNORECASE if field.get("ignore_case") is True else 0
    result = search_untrusted(pattern, text, flags=flags)
    if not result.matched or result.match is None:
        return None
    group = field.get("group")
    try:
        return result.match.group(group) if group is not None else result.match.group(0)
    except _SCHEMA_NONCRITICAL_EXCEPTIONS:
        return result.match.group(0)


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
            value = _extract_attribute_from_node(
                target_nodes[0],
                str(attr) if attr else None,
            )
        elif item_type == "html":
            value = _extract_html_from_node(target_nodes[0])
        elif item_type == "regex":
            base_text = _extract_text_from_node(target_nodes[0]) or ""
            value = _extract_regex_from_text(base_text, field)
        elif len(target_nodes) > 1:
            value = _reduce_matches(target_nodes, join_with)
        else:
            value = _extract_text_from_node(target_nodes[0])
        if value:
            values.append(value)
    if not values:
        return None
    return _apply_transforms(values, field.get("transforms"), base_url)


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
                value = _extract_fields_from_node(
                    matches[0],
                    nested_fields,
                    base_url=base_url,
                    context=ctx,
                )
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
                    item = _extract_fields_from_node(
                        match,
                        nested_fields,
                        base_url=base_url,
                        context=ctx,
                    )
                    if item:
                        items.append(item)
                if items:
                    value = items
        elif field_type == "list":
            value = _extract_list_items(
                node,
                field,
                base_url=base_url,
                context=ctx,
            )
        else:
            selector = _field_selector(field)
            matches = _select_nodes(node, selector, context_sensitive=True) if selector else []
            if not matches:
                value = None
            elif field_type == "attribute":
                attr = field.get("attribute") or field.get("attr")
                value = _extract_attribute_from_node(
                    matches[0],
                    str(attr) if attr else None,
                )
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
    attr_classes = re.findall(
        r'class\s*[*^$]?=\s*["\']([^"\']+)["\']',
        expr,
    )
    return [class_name for class_name in classes + attr_classes if _is_fragile_class_name(class_name)]


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
        for expr in _ensure_sequence(rules.get(key)):
            selectors.append((key, expr))
    pagination = rules.get("pagination")
    if isinstance(pagination, dict):
        for key in _PAGINATION_SELECTOR_KEYS:
            for expr in _ensure_sequence(pagination.get(key)):
                selectors.append((f"pagination.{key}", expr))
    alternates = rules.get("alternates")
    if isinstance(alternates, list):
        for index, alternate in enumerate(alternates):
            if not isinstance(alternate, dict):
                continue
            for key, expr in _iter_rule_selectors(alternate):
                selectors.append((f"alternates[{index}].{key}", expr))
    return selectors


def _iter_watchlist_selector_specs(rules: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for key in _SCHEMA_SELECTOR_KEYS:
        for expr in _ensure_sequence(rules.get(key)):
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
            for expr in _ensure_sequence(pagination.get(key)):
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
        for index, alternate in enumerate(alternates):
            if not isinstance(alternate, dict):
                continue
            for spec in _iter_watchlist_selector_specs(alternate):
                alternate_spec = dict(spec)
                alternate_spec["key"] = f"alternates[{index}].{spec['key']}"
                specs.append(alternate_spec)
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

    _walk_fields(_normalize_field_definitions(rules.get("baseFields") or []), "baseFields.")
    _walk_fields(_normalize_field_definitions(rules.get("fields") or []), "fields.")
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
    invalid_selectors: set[tuple[Any, str]] = set()

    for spec in compile_specs:
        key = spec.get("key")
        stripped = (spec.get("selector") or "").strip()
        if not stripped:
            continue
        error = _selector_validation_error(stripped)
        if error:
            errors.append({"key": key, "selector": stripped, "error": error})
            invalid_selectors.add((key, stripped))

    if html_text:
        try:
            document = html.fromstring(html_text)
        except _SCHEMA_NONCRITICAL_EXCEPTIONS as exc:
            warnings.append(
                {
                    "key": "document",
                    "selector": "",
                    "warning": "html_parse_failed",
                    "detail": str(exc),
                }
            )
            return {"errors": errors, "warnings": warnings}
        specs = _iter_watchlist_selector_specs(rules or {})
        specs.extend(dsl_specs)
        for spec in specs:
            if not spec.get("check_html", True):
                continue
            stripped = (spec.get("selector") or "").strip()
            if not stripped:
                continue
            matches, selection_failed = _select_nodes_with_status(document, stripped)
            selector_identity = (spec.get("key"), stripped)
            if selection_failed and selector_identity not in invalid_selectors:
                errors.append(
                    {
                        "key": spec.get("key"),
                        "selector": stripped,
                        "error": "selector_invalid",
                    }
                )
                invalid_selectors.add(selector_identity)
            count = len(matches)
            if include_counts:
                selector_counts[str(spec.get("key"))] = count
            if spec.get("expect_nonzero", True) and count == 0:
                warnings.append(
                    {
                        "key": spec.get("key"),
                        "selector": stripped,
                        "warning": "no_matches",
                    }
                )
            if not spec.get("allow_multiple", False) and count > 1:
                warnings.append(
                    {
                        "key": spec.get("key"),
                        "selector": stripped,
                        "warning": "non_unique_selector",
                        "count": count,
                    }
                )
            if stripped.startswith("css:"):
                for class_name in _fragile_css_classes(stripped):
                    warnings.append(
                        {
                            "key": spec.get("key"),
                            "selector": stripped,
                            "warning": "fragile_selector",
                            "detail": f"fragile class '{class_name}'",
                        }
                    )

    result = {"errors": errors, "warnings": warnings}
    if include_counts:
        result["selector_counts"] = selector_counts
    return result


def extract_schema_fields(
    html_text: str,
    base_url: str,
    rules: dict[str, Any],
) -> dict[str, Any]:
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
    except _SCHEMA_NONCRITICAL_EXCEPTIONS as exc:
        result["error"] = f"HTML parse failed: {exc}"
        return result

    with contextlib.suppress(_SCHEMA_NONCRITICAL_EXCEPTIONS):
        document.make_links_absolute(base_url)

    if _is_schema_dsl(rules):
        schema_name = rules.get("name")
        base_selector = _base_selector(rules)
        nodes: list[HtmlElement] = []
        if base_selector:
            nodes.extend(node for node in _select_nodes(document, base_selector) if isinstance(node, HtmlElement))
        base_node = nodes[0] if nodes else document

        schema_fields: dict[str, Any] = {}
        base_fields = _normalize_field_definitions(rules.get("baseFields") or [])
        fields = _normalize_field_definitions(rules.get("fields") or [])
        if base_fields:
            schema_fields.update(
                _extract_fields_from_node(
                    base_node,
                    base_fields,
                    base_url=base_url,
                    context={},
                )
            )
        if fields:
            schema_fields.update(
                _extract_fields_from_node(
                    base_node,
                    fields,
                    base_url=base_url,
                    context=schema_fields,
                )
            )

        if isinstance(schema_name, str) and schema_name.strip():
            result["schema_name"] = schema_name.strip()
        result["schema_fields"] = schema_fields

        for key in (
            "title",
            "summary",
            "content",
            "author",
            "published",
            "published_raw",
            "date",
        ):
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
        nodes.extend(node for node in _select_nodes(document, selector) if isinstance(node, HtmlElement))
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
        rules.get("summary_xpath") or rules.get("description_xpath") or rules.get("summary_selector"),
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
        rules.get("published_xpath") or rules.get("date_xpath") or rules.get("date_selector"),
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
        parsed = _normalize_datetime(
            published_raw,
            fmt if isinstance(fmt, str) else None,
        )
        if parsed:
            result["published"] = parsed

    result["extraction_successful"] = bool(content or summary or title)
    return result
