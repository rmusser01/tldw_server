"""Schema selector normalization, validation, transforms, and extraction."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Formatter
from typing import Any
from urllib.parse import urljoin

from lxml import html
from lxml.html import HtmlElement

from ..safe_regex import search_untrusted, sub_untrusted
from .engine import (
    _reduce_matches,
    _select_nodes_with_status,
    _selector_validation_error,
    coerce_value,
    ensure_sequence,
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


@dataclass(frozen=True, slots=True)
class _SchemaLimits:
    max_depth: int = 32
    max_total_fields: int = 256
    max_selector_evaluations: int = 512
    max_aggregate_matches: int = 10_000
    max_retained_output_chars: int = 1_000_000
    max_template_length: int = 4_096
    max_rendered_output_chars: int = 1_000_000


_DEFAULT_SCHEMA_LIMITS = _SchemaLimits()


class _SchemaBudgetExceeded(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(slots=True)
class _SchemaBudget:
    limits: _SchemaLimits
    selector_evaluations: int = 0
    aggregate_matches: int = 0
    retained_output_chars: int = 0

    def begin_selection(self) -> None:
        self.selector_evaluations += 1
        if self.selector_evaluations > self.limits.max_selector_evaluations:
            raise _SchemaBudgetExceeded(
                "selector_too_complex:selector_evaluations>" f"{self.limits.max_selector_evaluations}"
            )

    def retain_matches(self, count: int) -> None:
        self.aggregate_matches += max(0, count)
        if self.aggregate_matches > self.limits.max_aggregate_matches:
            raise _SchemaBudgetExceeded("selector_too_complex:selector_matches>" f"{self.limits.max_aggregate_matches}")

    def retain_output(self, value: Any) -> None:
        pending = [value]
        added = 0
        while pending:
            current = pending.pop()
            if isinstance(current, str):
                added += len(current)
            elif isinstance(current, dict):
                pending.extend(current.values())
            elif isinstance(current, (list, tuple)):
                pending.extend(current)
            elif current is not None:
                added += len(str(current))
            if self.retained_output_chars + added > self.limits.max_retained_output_chars:
                raise _SchemaBudgetExceeded(
                    "selector_too_complex:retained_output_chars>" f"{self.limits.max_retained_output_chars}"
                )
        self.retained_output_chars += added


@dataclass(frozen=True, slots=True)
class _SchemaFieldRecord:
    field: dict[str, Any]
    path: str
    depth: int


def _schema_limit_error(kind: str, limit: int) -> _SchemaBudgetExceeded:
    return _SchemaBudgetExceeded(f"selector_too_complex:{kind}>{limit}")


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


def _iter_normalized_field_definitions(fields: Any):
    if isinstance(fields, list):
        yield from (field for field in fields if isinstance(field, dict))
        return
    if isinstance(fields, dict):
        for name, spec in fields.items():
            entry = dict(spec) if isinstance(spec, dict) else {"selector": spec}
            entry.setdefault("name", str(name))
            yield entry


def _preflight_schema_fields(
    rules: dict[str, Any],
    limits: _SchemaLimits,
) -> list[_SchemaFieldRecord]:
    records: list[_SchemaFieldRecord] = []
    groups = (
        (rules.get("baseFields") or [], "baseFields."),
        (rules.get("fields") or [], "fields."),
    )
    for definitions, prefix in groups:
        stack = [(iter(_iter_normalized_field_definitions(definitions)), prefix, 1)]
        while stack:
            iterator, current_prefix, depth = stack[-1]
            try:
                field = next(iterator)
            except StopIteration:
                stack.pop()
                continue

            if depth > limits.max_depth:
                raise _schema_limit_error("schema_depth", limits.max_depth)
            if len(records) >= limits.max_total_fields:
                raise _schema_limit_error("schema_fields", limits.max_total_fields)

            name = str(field.get("name") or "").strip()
            path = f"{current_prefix}{name}" if name else current_prefix.rstrip(".")
            records.append(_SchemaFieldRecord(field=field, path=path, depth=depth))

            field_type = str(field.get("type") or "text").strip().lower()
            if field_type not in {"nested", "nested_list"}:
                continue
            nested_fields = field.get("fields") or {}
            stack.append(
                (
                    iter(_iter_normalized_field_definitions(nested_fields)),
                    f"{path}.",
                    depth + 1,
                )
            )
    return records


def _number_normalize(value: str) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def normalize_datetime(raw: str, fmt: str | None = None) -> str | None:
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
        normalized = normalize_datetime(value, fmt if isinstance(fmt, str) else None)
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


_TEMPLATE_FORMATTER = Formatter()


def _replacement_field_tokens(template: str) -> list[str] | None:
    tokens: list[str] = []
    index = 0
    while index < len(template):
        if template.startswith("{{", index) or template.startswith("}}", index):
            index += 2
            continue
        if template[index] != "{":
            index += 1
            continue
        end = template.find("}", index + 1)
        if end < 0 or "{" in template[index + 1 : end]:
            return None
        tokens.append(template[index + 1 : end])
        index = end + 1
    return tokens


def _parse_computed_template(
    template: str,
    limits: _SchemaLimits,
) -> tuple[list[tuple[str, str | None, str | None, str | None]] | None, str | None]:
    if len(template) > limits.max_template_length:
        return None, (f"selector_too_complex:template_length>{limits.max_template_length}")
    try:
        parsed = list(_TEMPLATE_FORMATTER.parse(template))
    except ValueError:
        return None, "selector_invalid"
    tokens = _replacement_field_tokens(template)
    field_parts = [part for part in parsed if part[1] is not None]
    if tokens is None or len(tokens) != len(field_parts):
        return None, "selector_invalid"
    for token, (_literal, field_name, format_spec, conversion) in zip(
        tokens,
        field_parts,
        strict=True,
    ):
        if ":" in token or "!" in token:
            return None, "selector_invalid"
        if not field_name or not field_name.isidentifier():
            return None, "selector_invalid"
        if conversion is not None or format_spec:
            return None, "selector_invalid"
    return parsed, None


def _render_computed_template(
    template: str,
    context: dict[str, Any],
    limits: _SchemaLimits,
) -> tuple[str | None, str | None]:
    parsed, error = _parse_computed_template(template, limits)
    if error or parsed is None:
        return None, error
    pieces: list[str] = []
    rendered_chars = 0
    for literal, field_name, _format_spec, _conversion in parsed:
        rendered_chars += len(literal)
        if rendered_chars > limits.max_rendered_output_chars:
            return None, ("selector_too_complex:rendered_output>" f"{limits.max_rendered_output_chars}")
        pieces.append(literal)
        if field_name is None:
            continue
        value = str(context.get(field_name, ""))
        rendered_chars += len(value)
        if rendered_chars > limits.max_rendered_output_chars:
            return None, ("selector_too_complex:rendered_output>" f"{limits.max_rendered_output_chars}")
        pieces.append(value)
    return "".join(pieces), None


def _join_computed_sources(
    sources: Sequence[Any],
    context: dict[str, Any],
    join_with: str,
    budget: _SchemaBudget,
    *,
    retain_output: bool,
) -> str:
    parts: list[str] = []
    rendered_chars = 0
    max_chars = budget.limits.max_retained_output_chars
    if retain_output:
        max_chars -= budget.retained_output_chars
    for source in sources:
        value = str(context.get(source, ""))
        if not value:
            continue
        if parts:
            rendered_chars += len(join_with)
        rendered_chars += len(value)
        if rendered_chars > max_chars:
            raise _schema_limit_error(
                "retained_output_chars",
                budget.limits.max_retained_output_chars,
            )
        parts.append(value)
    return join_with.join(parts)


def _template_validation_errors(
    records: Sequence[_SchemaFieldRecord],
    limits: _SchemaLimits,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for record in records:
        field = record.field
        field_type = str(field.get("type") or "text").strip().lower()
        template = field.get("template")
        if field_type != "computed" or not isinstance(template, str):
            continue
        _parsed, error = _parse_computed_template(template, limits)
        if error:
            errors.append(
                {
                    "key": record.path,
                    "selector": "",
                    "error": error,
                }
            )
    return errors


def _extract_text_from_node(node: Any) -> str | None:
    return coerce_value(node)


def _extract_html_from_node(node: Any) -> str | None:
    if isinstance(node, HtmlElement):
        try:
            return html.tostring(node, encoding="unicode")
        except _SCHEMA_NONCRITICAL_EXCEPTIONS:
            return None
    return coerce_value(node)


def _extract_attribute_from_node(node: Any, attr: str | None) -> str | None:
    if not attr:
        return None
    if isinstance(node, HtmlElement):
        value = node.get(attr)
        return value.strip() if isinstance(value, str) and value.strip() else None
    return None


def _select_nodes_with_budget_status(
    node: HtmlElement,
    selector: str,
    budget: _SchemaBudget,
    *,
    context_sensitive: bool = False,
) -> tuple[list[Any], bool]:
    budget.begin_selection()
    matches, failed = _select_nodes_with_status(
        node,
        selector,
        context_sensitive=context_sensitive,
    )
    budget.retain_matches(len(matches))
    return matches, failed


def _select_nodes_with_budget(
    node: HtmlElement,
    selector: str,
    budget: _SchemaBudget,
    *,
    context_sensitive: bool = False,
) -> list[Any]:
    matches, _failed = _select_nodes_with_budget_status(
        node,
        selector,
        budget,
        context_sensitive=context_sensitive,
    )
    return matches


def _extract_value_with_budget(
    node: HtmlElement,
    selectors: Sequence[str] | str | None,
    budget: _SchemaBudget,
    *,
    join: bool = False,
    join_with: str = " ",
) -> str | None:
    for expr in ensure_sequence(selectors):
        matches = _select_nodes_with_budget(
            node,
            expr,
            budget,
            context_sensitive=True,
        )
        if not matches:
            continue
        value = _reduce_matches(matches, join_with) if join else coerce_value(matches[0])
        if value:
            return value
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
    budget: _SchemaBudget,
    context: dict[str, Any],
) -> list[Any] | None:
    selector = _field_selector(field)
    nodes = (
        _select_nodes_with_budget(
            node,
            selector,
            budget,
            context_sensitive=True,
        )
        if selector
        else []
    )
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
            target_nodes = _select_nodes_with_budget(
                match,
                item_selector,
                budget,
                context_sensitive=True,
            )
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
    budget: _SchemaBudget,
    context: dict[str, Any] | None = None,
    retain_output: bool = True,
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
            matches = (
                _select_nodes_with_budget(
                    node,
                    selector,
                    budget,
                    context_sensitive=True,
                )
                if selector
                else [node]
            )
            nested_fields = _normalize_field_definitions(field.get("fields") or {})
            value = None
            if matches and nested_fields:
                value = _extract_fields_from_node(
                    matches[0],
                    nested_fields,
                    base_url=base_url,
                    budget=budget,
                    context=ctx,
                    retain_output=False,
                )
        elif field_type == "nested_list":
            selector = _field_selector(field)
            matches = (
                _select_nodes_with_budget(
                    node,
                    selector,
                    budget,
                    context_sensitive=True,
                )
                if selector
                else []
            )
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
                        budget=budget,
                        context=ctx,
                        retain_output=False,
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
                budget=budget,
                context=ctx,
            )
        else:
            selector = _field_selector(field)
            matches = (
                _select_nodes_with_budget(
                    node,
                    selector,
                    budget,
                    context_sensitive=True,
                )
                if selector
                else []
            )
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
            if retain_output:
                budget.retain_output(value)
            extracted[name] = value
            ctx[name] = value

    for field in computed_fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        value = None
        if "template" in field and isinstance(field.get("template"), str):
            value, template_error = _render_computed_template(
                field["template"],
                ctx,
                budget.limits,
            )
            if template_error and template_error.startswith("selector_too_complex:"):
                raise _SchemaBudgetExceeded(template_error)
        else:
            source = field.get("from")
            if isinstance(source, list):
                join_with = str(field.get("join_with") or " ")
                value = _join_computed_sources(
                    source,
                    ctx,
                    join_with,
                    budget,
                    retain_output=retain_output,
                )
            elif isinstance(source, str):
                value = ctx.get(source)
            elif "value" in field:
                value = field.get("value")
        value = _apply_transforms(value, field.get("transforms"), base_url)
        if value is not None:
            if retain_output:
                budget.retain_output(value)
            extracted[name] = value
            ctx[name] = value

    return extracted


def _is_schema_dsl(rules: dict[str, Any]) -> bool:
    return isinstance(rules.get("fields"), list) or isinstance(rules.get("baseFields"), (list, dict))


def _has_nonempty_value(value: Any) -> bool:
    pending = [value]
    while pending:
        current = pending.pop()
        if current is None:
            continue
        if isinstance(current, str):
            if current.strip():
                return True
            continue
        if isinstance(current, list):
            pending.extend(current)
            continue
        if isinstance(current, dict):
            pending.extend(current.values())
            continue
        return True
    return False


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
        for expr in ensure_sequence(rules.get(key)):
            selectors.append((key, expr))
    pagination = rules.get("pagination")
    if isinstance(pagination, dict):
        for key in _PAGINATION_SELECTOR_KEYS:
            for expr in ensure_sequence(pagination.get(key)):
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
        for expr in ensure_sequence(rules.get(key)):
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
            for expr in ensure_sequence(pagination.get(key)):
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


def _iter_schema_dsl_selector_specs(
    rules: dict[str, Any],
    records: Sequence[_SchemaFieldRecord],
) -> list[dict[str, Any]]:
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

    for record in records:
        field = record.field
        field_type = str(field.get("type") or "text").strip().lower()
        selector = _field_selector(field)
        if selector:
            specs.append(
                {
                    "key": record.path,
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
                    "key": f"{record.path}.item_selector",
                    "selector": item_selector,
                    "allow_multiple": True,
                    "expect_nonzero": False,
                    "check_html": True,
                }
            )
    return specs


def _record_validation_selection(
    observations: dict[tuple[Any, str], dict[str, Any]],
    spec: dict[str, Any],
    matches: Sequence[Any],
    failed: bool,
) -> None:
    identity = (spec.get("key"), str(spec.get("selector") or ""))
    observation = observations.setdefault(
        identity,
        {"spec": spec, "count": 0, "failed": False},
    )
    observation["count"] += len(matches)
    observation["failed"] = observation["failed"] or failed


def _evaluate_dsl_fields_for_validation(
    node: HtmlElement,
    fields: Sequence[dict[str, Any]],
    prefix: str,
    budget: _SchemaBudget,
    observations: dict[tuple[Any, str], dict[str, Any]],
) -> None:
    for field in fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        path = f"{prefix}{name}"
        field_type = str(field.get("type") or "text").strip().lower()
        selector = _field_selector(field)
        matches: list[Any]
        if selector:
            matches, failed = _select_nodes_with_budget_status(
                node,
                selector,
                budget,
                context_sensitive=True,
            )
            _record_validation_selection(
                observations,
                {
                    "key": path,
                    "selector": selector,
                    "allow_multiple": field_type in {"list", "nested_list"},
                    "expect_nonzero": True,
                },
                matches,
                failed,
            )
        elif field_type == "nested":
            matches = [node]
        else:
            matches = []

        item_selector = _normalize_selector_expr(
            field.get("item_selector") or field.get("itemSelector"),
            css=field.get("itemCss") or field.get("item_css"),
            xpath=field.get("itemXpath") or field.get("item_xpath"),
        )
        if item_selector:
            item_spec = {
                "key": f"{path}.item_selector",
                "selector": item_selector,
                "allow_multiple": True,
                "expect_nonzero": False,
            }
            for match in matches:
                if not isinstance(match, HtmlElement):
                    continue
                item_matches, failed = _select_nodes_with_budget_status(
                    match,
                    item_selector,
                    budget,
                    context_sensitive=True,
                )
                _record_validation_selection(
                    observations,
                    item_spec,
                    item_matches,
                    failed,
                )

        if field_type not in {"nested", "nested_list"}:
            continue
        nested_fields = _normalize_field_definitions(field.get("fields") or {})
        if not nested_fields:
            continue
        nested_nodes = matches[:1] if field_type == "nested" else matches
        for match in nested_nodes:
            if not isinstance(match, HtmlElement):
                continue
            _evaluate_dsl_fields_for_validation(
                match,
                nested_fields,
                f"{path}.",
                budget,
                observations,
            )


def _evaluate_validation_rules(
    rules: dict[str, Any],
    document: HtmlElement,
    budget: _SchemaBudget,
    *,
    is_dsl: bool,
) -> list[dict[str, Any]]:
    observations: dict[tuple[Any, str], dict[str, Any]] = {}
    if not is_dsl:
        for spec in _iter_watchlist_selector_specs(rules):
            if not spec.get("check_html", True):
                continue
            selector = str(spec.get("selector") or "").strip()
            if not selector:
                continue
            matches, failed = _select_nodes_with_budget_status(
                document,
                selector,
                budget,
            )
            _record_validation_selection(observations, spec, matches, failed)
        return list(observations.values())

    base_node = document
    base_selector = _base_selector(rules)
    if base_selector:
        matches, failed = _select_nodes_with_budget_status(
            document,
            base_selector,
            budget,
        )
        spec = {
            "key": "baseSelector",
            "selector": base_selector,
            "allow_multiple": False,
            "expect_nonzero": True,
        }
        _record_validation_selection(observations, spec, matches, failed)
        base_node = next(
            (match for match in matches if isinstance(match, HtmlElement)),
            document,
        )

    _evaluate_dsl_fields_for_validation(
        base_node,
        _normalize_field_definitions(rules.get("baseFields") or []),
        "baseFields.",
        budget,
        observations,
    )
    _evaluate_dsl_fields_for_validation(
        base_node,
        _normalize_field_definitions(rules.get("fields") or []),
        "fields.",
        budget,
        observations,
    )
    return list(observations.values())


def _complexity_validation_result(
    code: str,
    *,
    include_counts: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "errors": [{"key": "schema", "selector": "", "error": code}],
        "warnings": [],
    }
    if include_counts:
        result["selector_counts"] = {}
    return result


def validate_selector_rules(
    rules: dict[str, Any],
    *,
    html_text: str | None = None,
    include_counts: bool = False,
    _limits: _SchemaLimits | None = None,
) -> dict[str, Any]:
    limits = _limits or _DEFAULT_SCHEMA_LIMITS
    normalized_rules = rules or {}
    is_dsl = _is_schema_dsl(normalized_rules)
    records: list[_SchemaFieldRecord] = []
    if is_dsl:
        try:
            records = _preflight_schema_fields(normalized_rules, limits)
        except _SchemaBudgetExceeded as exc:
            return _complexity_validation_result(
                exc.code,
                include_counts=include_counts,
            )

    errors = _template_validation_errors(records, limits)
    warnings: list[dict[str, Any]] = []
    selector_counts: dict[str, int] = {}
    compile_specs: list[dict[str, Any]] = [
        {"key": key, "selector": expr} for key, expr in _iter_rule_selectors(normalized_rules)
    ]
    dsl_specs = _iter_schema_dsl_selector_specs(normalized_rules, records) if is_dsl else []
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
        budget = _SchemaBudget(limits)
        try:
            observations = _evaluate_validation_rules(
                normalized_rules,
                document,
                budget,
                is_dsl=is_dsl,
            )
        except _SchemaBudgetExceeded as exc:
            return _complexity_validation_result(
                exc.code,
                include_counts=include_counts,
            )
        for observation in observations:
            spec = observation["spec"]
            stripped = str(spec.get("selector") or "").strip()
            selection_failed = bool(observation["failed"])
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
            count = int(observation["count"])
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


def _extract_dsl_schema_fields(
    document: HtmlElement,
    base_url: str,
    rules: dict[str, Any],
    result: dict[str, Any],
    budget: _SchemaBudget,
) -> dict[str, Any]:
    schema_name = rules.get("name")
    base_selector = _base_selector(rules)
    nodes: list[HtmlElement] = []
    if base_selector:
        nodes.extend(
            node for node in _select_nodes_with_budget(document, base_selector, budget) if isinstance(node, HtmlElement)
        )
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
                budget=budget,
                context={},
            )
        )
    if fields:
        schema_fields.update(
            _extract_fields_from_node(
                base_node,
                fields,
                base_url=base_url,
                budget=budget,
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


def _extract_legacy_schema_fields(
    document: HtmlElement,
    rules: dict[str, Any],
    result: dict[str, Any],
    budget: _SchemaBudget,
) -> dict[str, Any]:
    base_selectors = (
        rules.get("base_xpath")
        or rules.get("base_selector")
        or rules.get("entry_xpath")
        or rules.get("entry_selector")
        or rules.get("item_xpath")
        or rules.get("items_xpath")
    )
    nodes: list[HtmlElement] = []
    for selector in ensure_sequence(base_selectors):
        nodes.extend(
            node for node in _select_nodes_with_budget(document, selector, budget) if isinstance(node, HtmlElement)
        )
    base_node = nodes[0] if nodes else document

    summary_join = str(rules.get("summary_join_with") or " ")
    content_join = str(rules.get("content_join_with") or "\n")
    title = _extract_value_with_budget(
        base_node,
        rules.get("title_xpath") or rules.get("title_selector"),
        budget,
        join=False,
    )
    summary = _extract_value_with_budget(
        base_node,
        rules.get("summary_xpath") or rules.get("description_xpath") or rules.get("summary_selector"),
        budget,
        join=True,
        join_with=summary_join,
    )
    content = _extract_value_with_budget(
        base_node,
        rules.get("content_xpath") or rules.get("content_selector"),
        budget,
        join=True,
        join_with=content_join,
    )
    author = _extract_value_with_budget(
        base_node,
        rules.get("author_xpath") or rules.get("author_selector"),
        budget,
        join=False,
    )
    published_raw = _extract_value_with_budget(
        base_node,
        rules.get("published_xpath") or rules.get("date_xpath") or rules.get("date_selector"),
        budget,
        join=False,
    )

    for key, value in (
        ("title", title),
        ("summary", summary),
        ("content", content),
        ("author", author),
    ):
        if value:
            budget.retain_output(value)
            result[key] = value
    if published_raw:
        budget.retain_output(published_raw)
        result["published_raw"] = published_raw
        fmt = rules.get("published_format") or rules.get("date_format")
        parsed = normalize_datetime(
            published_raw,
            fmt if isinstance(fmt, str) else None,
        )
        if parsed:
            budget.retain_output(parsed)
            result["published"] = parsed

    result["extraction_successful"] = bool(content or summary or title)
    return result


def extract_schema_fields(
    html_text: str,
    base_url: str,
    rules: dict[str, Any],
    *,
    _limits: _SchemaLimits | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": base_url,
        "extraction_successful": False,
    }
    if not html_text:
        return result
    if not isinstance(rules, dict) or not rules:
        return result

    limits = _limits or _DEFAULT_SCHEMA_LIMITS
    is_dsl = _is_schema_dsl(rules)
    try:
        records = _preflight_schema_fields(rules, limits) if is_dsl else []
        for entry in _template_validation_errors(records, limits):
            error = str(entry.get("error") or "")
            if error.startswith("selector_too_complex:"):
                raise _SchemaBudgetExceeded(error)
    except _SchemaBudgetExceeded as exc:
        result["error"] = exc.code
        return result

    try:
        document = html.fromstring(html_text)
    except _SCHEMA_NONCRITICAL_EXCEPTIONS as exc:
        result["error"] = f"HTML parse failed: {exc}"
        return result

    with contextlib.suppress(_SCHEMA_NONCRITICAL_EXCEPTIONS):
        document.make_links_absolute(base_url)

    budget = _SchemaBudget(limits)
    try:
        if is_dsl:
            return _extract_dsl_schema_fields(
                document,
                base_url,
                rules,
                result,
                budget,
            )
        return _extract_legacy_schema_fields(
            document,
            rules,
            result,
            budget,
        )
    except _SchemaBudgetExceeded as exc:
        return {
            "url": base_url,
            "extraction_successful": False,
            "error": exc.code,
        }
