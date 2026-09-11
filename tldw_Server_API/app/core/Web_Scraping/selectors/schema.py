"""Schema selector normalization, validation, transforms, and extraction."""

from __future__ import annotations

import codecs
import contextlib
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from string import Formatter
from typing import Any
from urllib.parse import urljoin

from lxml import etree, html
from lxml.html import HtmlElement

from ..safe_regex import search_untrusted, sub_untrusted
from .engine import (
    _select_nodes_with_status,
    _selector_validation_error,
    ensure_sequence,
)
from .engine import (
    coerce_value as _engine_coerce_value,
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
    max_depth: int | None = None
    max_total_fields: int | None = None
    max_selector_evaluations: int | None = None
    max_aggregate_matches: int | None = None
    max_retained_output_chars: int | None = None
    max_template_length: int | None = None
    max_rendered_output_chars: int | None = None


_DEFAULT_SCHEMA_LIMITS = _SchemaLimits()


class _SchemaBudgetExceeded(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(slots=True)
class _OutputSlotIndexNode:
    subtree_chars: int = 0
    slot: tuple[Any, ...] | None = None
    slot_chars: int = 0
    children: dict[Any, _OutputSlotIndexNode] = dataclass_field(default_factory=dict)


class _OutputSlotSnapshotError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _OutputSlotSnapshot(Mapping[tuple[Any, ...], int]):
    __slots__ = ("__entries", "__index_subtree", "__prefix", "__used")

    def __init__(
        self,
        prefix: tuple[Any, ...],
        entries: Mapping[tuple[Any, ...], int],
        index_subtree: _OutputSlotIndexNode | None,
    ) -> None:
        self.__entries = dict(entries)
        self.__prefix = prefix
        self.__index_subtree = index_subtree
        self.__used = False

    def __getitem__(self, slot: tuple[Any, ...]) -> int:
        return self.__entries[slot]

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        return iter(self.__entries)

    def __len__(self) -> int:
        return len(self.__entries)

    @property
    def _owns_detached_subtree(self) -> bool:
        return self.__index_subtree is not None

    def _claim(self, prefix: tuple[Any, ...]) -> _OutputSlotIndexNode | None:
        if prefix != self.__prefix:
            raise _OutputSlotSnapshotError("output_slot_snapshot_prefix_mismatch")
        if self.__used:
            raise _OutputSlotSnapshotError("output_slot_snapshot_already_used")
        self.__used = True
        subtree = self.__index_subtree
        self.__index_subtree = None
        return subtree


@dataclass(slots=True)
class _SchemaBudget:
    limits: _SchemaLimits
    selector_evaluations: int = dataclass_field(default=0, init=False)
    aggregate_matches: int = dataclass_field(default=0, init=False)
    retained_output_chars: int = dataclass_field(default=0, init=False)
    output_slots: dict[tuple[Any, ...], int] = dataclass_field(
        default_factory=dict,
        init=False,
    )
    _output_slot_index: _OutputSlotIndexNode = dataclass_field(
        default_factory=_OutputSlotIndexNode,
        init=False,
        repr=False,
    )
    selection_observer: Callable[[dict[str, Any], Sequence[Any], bool], None] | None = dataclass_field(
        default=None,
        init=False,
    )
    validation_selection_cache: dict[tuple[int, Any, str], tuple[list[Any], bool]] | None = dataclass_field(
        default=None,
        init=False,
        repr=False,
    )
    enforce_output: bool = dataclass_field(default=True, init=False)

    def _output_index_path(
        self,
        slot: tuple[Any, ...],
        *,
        create: bool,
    ) -> list[_OutputSlotIndexNode] | None:
        node = self._output_slot_index
        nodes = [node]
        for part in slot:
            child = node.children.get(part)
            if child is None:
                if not create:
                    return None
                child = _OutputSlotIndexNode()
                node.children[part] = child
            node = child
            nodes.append(node)
        return nodes

    def _index_output_slot(self, slot: tuple[Any, ...], chars: int) -> None:
        nodes = self._output_index_path(slot, create=True)
        if nodes is None:
            raise RuntimeError("output slot index path creation failed")
        terminal = nodes[-1]
        replaced = terminal.slot_chars if terminal.slot is not None else 0
        terminal.slot = slot
        terminal.slot_chars = chars
        delta = chars - replaced
        for node in nodes:
            node.subtree_chars += delta

    def _indexed_subtree_chars(self, prefix: tuple[Any, ...]) -> int:
        nodes = self._output_index_path(prefix, create=False)
        return nodes[-1].subtree_chars if nodes is not None else 0

    def begin_selection(self) -> None:
        self.selector_evaluations += 1
        limit = self.limits.max_selector_evaluations
        if limit is not None and self.selector_evaluations > limit:
            raise _SchemaBudgetExceeded("selector_too_complex:selector_evaluations>" f"{limit}")

    def retain_matches(self, count: int) -> None:
        self.aggregate_matches += max(0, count)
        limit = self.limits.max_aggregate_matches
        if limit is not None and self.aggregate_matches > limit:
            raise _SchemaBudgetExceeded("selector_too_complex:selector_matches>" f"{limit}")

    def remaining_match_capacity(self) -> int | None:
        limit = self.limits.max_aggregate_matches
        return None if limit is None else max(0, limit - self.aggregate_matches)

    def remaining_output_chars(self, slot: tuple[Any, ...] | None = None) -> int | None:
        limit = self.limits.max_retained_output_chars
        if not self.enforce_output or limit is None:
            return None
        replaced = self._indexed_subtree_chars(slot) if slot is not None else 0
        return limit - self.retained_output_chars + replaced

    def retain_output(self, value: Any, *, slot: tuple[Any, ...] | None = None) -> None:
        limit = self.limits.max_retained_output_chars
        if not self.enforce_output or limit is None:
            return
        available = self.remaining_output_chars(slot)
        added = _output_chars(value, stop_after=available)
        if available is not None and added > available:
            raise _schema_limit_error(
                "retained_output_chars",
                limit,
            )
        if slot is not None:
            self.take_output_prefix(slot)
            self.output_slots[slot] = added
            self._index_output_slot(slot, added)
        self.retained_output_chars += added

    def take_output_prefix(
        self,
        prefix: tuple[Any, ...],
    ) -> _OutputSlotSnapshot:
        nodes = self._output_index_path(prefix, create=False)
        if nodes is None:
            return _OutputSlotSnapshot(prefix, {}, None)

        subtree = nodes[-1]
        removed_index_chars = subtree.subtree_chars
        replaced: dict[tuple[Any, ...], int] = {}
        pending = [subtree]
        while pending:
            node = pending.pop()
            if node.slot is not None:
                replaced[node.slot] = self.output_slots.pop(node.slot)
            pending.extend(node.children.values())

        if prefix:
            del nodes[-2].children[prefix[-1]]
            for node in nodes[:-1]:
                node.subtree_chars -= removed_index_chars
            for depth in range(len(nodes) - 2, 0, -1):
                node = nodes[depth]
                if node.slot is not None or node.children:
                    break
                del nodes[depth - 1].children[prefix[depth - 1]]
        else:
            self._output_slot_index = _OutputSlotIndexNode()

        self.retained_output_chars -= sum(replaced.values())
        return _OutputSlotSnapshot(prefix, replaced, subtree)

    def restore_output_prefix(
        self,
        prefix: tuple[Any, ...],
        snapshot: _OutputSlotSnapshot,
    ) -> None:
        if not isinstance(snapshot, _OutputSlotSnapshot):
            raise _OutputSlotSnapshotError("output_slot_snapshot_invalid")
        subtree = snapshot._claim(prefix)
        self.take_output_prefix(prefix)
        if subtree is not None:
            nodes = self._output_index_path(prefix, create=True)
            if nodes is None:
                raise RuntimeError("output slot index path creation failed")
            if prefix:
                nodes[-2].children[prefix[-1]] = subtree
                for node in nodes[:-1]:
                    node.subtree_chars += subtree.subtree_chars
            else:
                self._output_slot_index = subtree
        self.output_slots.update(snapshot)
        self.retained_output_chars += sum(snapshot.values())

    def ensure_output_chars(self, added: int) -> None:
        limit = self.limits.max_retained_output_chars
        if not self.enforce_output or limit is None:
            return
        if self.retained_output_chars + added > limit:
            raise _schema_limit_error(
                "retained_output_chars",
                limit,
            )

    def observe_selection(
        self,
        spec: dict[str, Any] | None,
        matches: Sequence[Any],
        failed: bool,
    ) -> None:
        if spec is not None and self.selection_observer is not None:
            self.selection_observer(spec, matches, failed)


@dataclass(frozen=True, slots=True)
class _SchemaFieldRecord:
    field: dict[str, Any]
    path: str
    depth: int


def _schema_limit_error(kind: str, limit: int) -> _SchemaBudgetExceeded:
    return _SchemaBudgetExceeded(f"selector_too_complex:{kind}>{limit}")


def _output_chars(value: Any, *, stop_after: int | None = None) -> int:
    pending = [value]
    total = 0
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            total += len(current)
        elif isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
        elif current is not None:
            total += len(str(current))
        if stop_after is not None and total > stop_after:
            return total
    return total


def _ensure_chars_within_limit(chars: int, limit: int | None, kind: str) -> None:
    if limit is not None and chars > limit:
        raise _schema_limit_error(kind, limit)


@dataclass(frozen=True, slots=True)
class _OutputBound:
    max_chars: int | None
    error_kind: str
    error_limit: int | None


_DEFAULT_OUTPUT_BOUND = _OutputBound(None, "retained_output_chars", None)
_ACTIVE_OUTPUT_BOUND: ContextVar[_OutputBound | None] = ContextVar(
    "selector_schema_output_bound",
    default=None,
)


def _current_output_bound() -> _OutputBound:
    return _ACTIVE_OUTPUT_BOUND.get() or _DEFAULT_OUTPUT_BOUND


def _bound_error(bound: _OutputBound) -> _SchemaBudgetExceeded:
    limit = bound.max_chars if bound.error_limit is None else bound.error_limit
    return _schema_limit_error(bound.error_kind, 0 if limit is None else limit)


@contextlib.contextmanager
def _construction_output_bound(
    max_chars: int | None,
    error_kind: str,
    error_limit: int | None,
):
    token = _ACTIVE_OUTPUT_BOUND.set(_OutputBound(max_chars, error_kind, error_limit))
    try:
        yield
    finally:
        _ACTIVE_OUTPUT_BOUND.reset(token)


def _ensure_active_output(value: str) -> str:
    bound = _current_output_bound()
    if bound.max_chars is not None and len(value) > bound.max_chars:
        raise _bound_error(bound)
    return value


def _bounded_element_text(node: HtmlElement) -> str | None:
    bound = _current_output_bound()
    pieces: list[str] = []
    trailing_parts: list[str] = []
    trailing_chars = 0
    trailing_overflow = False
    retained_chars = 0
    started = False

    for fragment in node.itertext():
        text = str(fragment)
        if not started:
            text = text.lstrip()
            if not text:
                continue
        body = text.rstrip()
        trailing = text[len(body) :]
        if body:
            added = trailing_chars + len(body)
            if trailing_overflow or (bound.max_chars is not None and retained_chars + added > bound.max_chars):
                raise _bound_error(bound)
            pieces.extend(trailing_parts)
            pieces.append(body)
            retained_chars += added
            trailing_parts = []
            trailing_chars = 0
            trailing_overflow = False
            started = True
        if started and trailing:
            trailing_chars += len(trailing)
            remaining = None if bound.max_chars is None else bound.max_chars - retained_chars
            if remaining is None or trailing_chars <= remaining:
                trailing_parts.append(trailing)
            else:
                trailing_overflow = True

    if not pieces:
        return None
    return "".join(pieces)


def coerce_value(value: Any) -> str | None:
    if isinstance(value, HtmlElement):
        return _bounded_element_text(value)
    coerced = _engine_coerce_value(value)
    return _ensure_active_output(coerced) if coerced is not None else None


def _bounded_join(
    values: Sequence[Any],
    join_with: str,
    max_chars: int | None,
    *,
    coerce: Callable[[Any], str | None] = coerce_value,
    strip_result: bool = False,
    error_kind: str = "retained_output_chars",
    error_limit: int | None = None,
) -> str | None:
    parts: list[str] = []
    total = 0
    for item in values:
        value = coerce(item)
        if not value:
            continue
        total += len(value) + (len(join_with) if parts else 0)
        if max_chars is not None and total > max_chars:
            raise _schema_limit_error(
                error_kind,
                max_chars if error_limit is None else error_limit,
            )
        parts.append(value)
    if not parts:
        return None
    joined = join_with.join(parts)
    return joined.strip() if strip_result else joined


def _compile_only_selector_error(selector: str) -> str | None:
    return _selector_validation_error(selector)


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

            if limits.max_depth is not None and depth > limits.max_depth:
                raise _schema_limit_error("schema_depth", limits.max_depth)
            if limits.max_total_fields is not None and len(records) >= limits.max_total_fields:
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
    max_rendered_output_chars: int | None,
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
        try:
            replacement = str(params.get("repl", ""))
        except _SCHEMA_NONCRITICAL_EXCEPTIONS:
            return value
        kwargs = {"max_output_chars": max_rendered_output_chars} if max_rendered_output_chars is not None else {}
        result = sub_untrusted(
            pattern,
            replacement,
            value,
            **kwargs,
        )
        if result.limit == "output" and max_rendered_output_chars is not None:
            raise _schema_limit_error(
                "rendered_output",
                max_rendered_output_chars,
            )
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


def _apply_transforms(
    value: Any,
    transforms: Any,
    base_url: str,
    max_rendered_output_chars: int | None,
) -> Any:
    if value is None or not transforms:
        return value
    transform_list = transforms if isinstance(transforms, list) else [transforms]
    _ensure_chars_within_limit(
        _output_chars(value, stop_after=max_rendered_output_chars),
        max_rendered_output_chars,
        "rendered_output",
    )
    if isinstance(value, list):
        transformed: list[Any] = []
        transformed_chars = 0
        for item in value:
            result = _apply_transforms(
                item,
                transform_list,
                base_url,
                max_rendered_output_chars,
            )
            if result is None:
                continue
            transformed_chars += _output_chars(result)
            _ensure_chars_within_limit(
                transformed_chars,
                max_rendered_output_chars,
                "rendered_output",
            )
            transformed.append(result)
        return transformed
    if isinstance(value, dict):
        return value
    result: str | None = str(value)
    for transform in transform_list:
        result = _apply_single_transform(
            result,
            transform,
            base_url,
            max_rendered_output_chars,
        )
        if result is None:
            break
        _ensure_chars_within_limit(
            len(result),
            max_rendered_output_chars,
            "rendered_output",
        )
    return result


_TEMPLATE_FORMATTER = Formatter()


class _SafeTemplateContext(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def _parse_computed_template(
    template: str,
    limits: _SchemaLimits,
) -> tuple[list[tuple[str, str | None, str | None, str | None]] | None, str | None]:
    if limits.max_template_length is not None and len(template) > limits.max_template_length:
        return None, (f"selector_too_complex:template_length>{limits.max_template_length}")
    try:
        parsed = list(_TEMPLATE_FORMATTER.parse(template))
    except ValueError:
        return None, "selector_invalid"
    for _literal, field_name, _format_spec, conversion in parsed:
        if field_name == "":
            return None, "selector_invalid"
        if conversion not in {None, "s", "r", "a"}:
            return None, "selector_invalid"
    return parsed, None


def _render_computed_template(
    template: str,
    context: dict[str, Any],
    limits: _SchemaLimits,
    max_chars: int | None,
    error_kind: str,
    error_limit: int | None,
) -> tuple[str | None, str | None]:
    parsed, error = _parse_computed_template(template, limits)
    if error or parsed is None:
        return None, error
    try:
        rendered = template.format_map(_SafeTemplateContext(context))
    except _SCHEMA_NONCRITICAL_EXCEPTIONS:
        return None, "selector_invalid"
    if max_chars is not None and len(rendered) > max_chars:
        public_limit = max_chars if error_limit is None else error_limit
        return None, f"selector_too_complex:{error_kind}>{public_limit}"
    return rendered, None


def _join_computed_sources(
    sources: Sequence[Any],
    context: dict[str, Any],
    join_with: str,
    max_chars: int | None,
    error_kind: str,
    error_limit: int | None,
) -> str:
    return (
        _bounded_join(
            sources,
            join_with,
            max_chars,
            coerce=lambda source: str(context.get(source, "")),
            error_kind=error_kind,
            error_limit=error_limit,
        )
        or ""
    )


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
    if isinstance(node, HtmlElement):
        return _bounded_element_text(node)
    return coerce_value(node)


class _BoundedHtmlWriter:
    def __init__(self, bound: _OutputBound) -> None:
        self._bound = bound
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._parts: list[str] = []
        self._chars = 0

    def _append(self, value: str) -> None:
        if not value:
            return
        if self._bound.max_chars is not None and self._chars + len(value) > self._bound.max_chars:
            raise _bound_error(self._bound)
        self._chars += len(value)
        self._parts.append(value)

    def write(self, value: bytes) -> None:
        self._append(self._decoder.decode(value))

    def finish(self) -> str:
        self._append(self._decoder.decode(b"", final=True))
        return "".join(self._parts)


def _extract_html_from_node(node: Any) -> str | None:
    if isinstance(node, HtmlElement):
        try:
            writer = _BoundedHtmlWriter(_current_output_bound())
            etree.ElementTree(node).write(
                writer,
                encoding="utf-8",
                method="html",
            )
            return writer.finish()
        except _SchemaBudgetExceeded:
            raise
        except _SCHEMA_NONCRITICAL_EXCEPTIONS:
            return None
    return coerce_value(node)


def _extract_attribute_from_node(node: Any, attr: str | None) -> str | None:
    if not attr:
        return None
    if isinstance(node, HtmlElement):
        value = node.get(attr)
        return _ensure_active_output(value.strip()) if isinstance(value, str) and value.strip() else None
    return None


def _select_nodes_with_budget_status(
    node: HtmlElement,
    selector: str,
    budget: _SchemaBudget,
    *,
    context_sensitive: bool = False,
    observation_spec: dict[str, Any] | None = None,
) -> tuple[list[Any], bool]:
    cache_key: tuple[int, Any, str] | None = None
    if observation_spec is not None and budget.validation_selection_cache is not None:
        cache_key = (
            id(node),
            observation_spec.get("key"),
            str(observation_spec.get("selector") or "").strip(),
        )
        if cache_key in budget.validation_selection_cache:
            return budget.validation_selection_cache[cache_key]
    budget.begin_selection()
    remaining_match_capacity = budget.remaining_match_capacity()
    matches, failed = _select_nodes_with_status(
        node,
        selector,
        context_sensitive=context_sensitive,
        max_results=None if remaining_match_capacity is None else remaining_match_capacity + 1,
    )
    budget.retain_matches(len(matches))
    budget.observe_selection(observation_spec, matches, failed)
    if cache_key is not None and budget.validation_selection_cache is not None:
        budget.validation_selection_cache[cache_key] = (matches, failed)
    return matches, failed


def _select_nodes_with_budget(
    node: HtmlElement,
    selector: str,
    budget: _SchemaBudget,
    *,
    context_sensitive: bool = False,
    observation_spec: dict[str, Any] | None = None,
) -> list[Any]:
    matches, _failed = _select_nodes_with_budget_status(
        node,
        selector,
        budget,
        context_sensitive=context_sensitive,
        observation_spec=observation_spec,
    )
    return matches


def _extract_value_with_budget(
    node: HtmlElement,
    selectors: Sequence[str] | str | None,
    budget: _SchemaBudget,
    *,
    key: str,
    slot: tuple[Any, ...],
    join: bool = False,
    join_with: str = " ",
) -> str | None:
    for expr in ensure_sequence(selectors):
        matches = _select_nodes_with_budget(
            node,
            expr,
            budget,
            context_sensitive=True,
            observation_spec={
                "key": key,
                "selector": expr,
                "allow_multiple": join,
                "expect_nonzero": True,
            },
        )
        if not matches:
            continue
        remaining = budget.remaining_output_chars(slot)
        with _construction_output_bound(
            remaining,
            "retained_output_chars",
            budget.limits.max_retained_output_chars,
        ):
            value = (
                _bounded_join(
                    matches,
                    join_with,
                    remaining,
                    coerce=coerce_value,
                    strip_result=True,
                    error_limit=budget.limits.max_retained_output_chars,
                )
                if join
                else coerce_value(matches[0])
            )
        if value:
            return value
    return None


def _extract_regex_from_text(text: str, field: dict[str, Any]) -> str | None:
    pattern = field.get("pattern") or field.get("regex")
    if not isinstance(pattern, str) or not text:
        return None
    flags = re.IGNORECASE if field.get("ignore_case") is True else 0
    result = search_untrusted(pattern, text, flags=flags, dialect="regex")
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
    path: str,
    max_output_chars: int | None,
    output_error_kind: str,
    output_error_limit: int | None,
) -> list[Any] | None:
    selector = _field_selector(field)
    nodes = (
        _select_nodes_with_budget(
            node,
            selector,
            budget,
            context_sensitive=True,
            observation_spec={
                "key": path,
                "selector": selector,
                "allow_multiple": True,
                "expect_nonzero": True,
            },
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
    value_chars = 0
    for match in nodes:
        target_nodes = [match]
        if item_selector and isinstance(match, HtmlElement):
            target_nodes = _select_nodes_with_budget(
                match,
                item_selector,
                budget,
                context_sensitive=True,
                observation_spec={
                    "key": f"{path}.item_selector",
                    "selector": item_selector,
                    "allow_multiple": True,
                    "expect_nonzero": False,
                },
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
            value = _bounded_join(
                target_nodes,
                join_with,
                max_output_chars,
                coerce=coerce_value,
                strip_result=True,
                error_kind=output_error_kind,
                error_limit=output_error_limit,
            )
        else:
            value = _extract_text_from_node(target_nodes[0])
        if value:
            value_chars += _output_chars(value)
            if max_output_chars is not None and value_chars > max_output_chars:
                raise _schema_limit_error(
                    output_error_kind,
                    max_output_chars if output_error_limit is None else output_error_limit,
                )
            values.append(value)
    if not values:
        return None
    return _apply_transforms(
        values,
        field.get("transforms"),
        base_url,
        budget.limits.max_rendered_output_chars if budget.enforce_output else None,
    )


def _extract_fields_from_node(
    node: HtmlElement,
    fields: Sequence[dict[str, Any]],
    *,
    base_url: str,
    budget: _SchemaBudget,
    context: dict[str, Any] | None = None,
    path_prefix: str = "",
    slot_prefix: tuple[Any, ...] = ("schema_fields",),
) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    computed_fields: list[dict[str, Any]] = []
    ctx = dict(context or {})

    def construction_bound(
        slot: tuple[Any, ...],
        transforms: Any,
    ) -> tuple[int | None, str, int | None]:
        if transforms and budget.enforce_output:
            rendered_limit = budget.limits.max_rendered_output_chars
            return rendered_limit, "rendered_output", rendered_limit
        return (
            budget.remaining_output_chars(slot),
            "retained_output_chars",
            budget.limits.max_retained_output_chars,
        )

    for field in fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        path = f"{path_prefix}{name}"
        field_type = str(field.get("type") or "text").strip().lower()
        if field_type == "computed":
            computed_fields.append(field)
            continue
        slot = slot_prefix + (name,)
        transforms = field.get("transforms")
        nested_snapshot = budget.take_output_prefix(slot) if field_type in {"nested", "nested_list"} else None
        field_limit, field_error_kind, field_error_limit = construction_bound(
            slot,
            transforms,
        )

        try:
            with _construction_output_bound(
                field_limit,
                field_error_kind,
                field_error_limit,
            ):
                if field_type == "nested":
                    selector = _field_selector(field)
                    matches = (
                        _select_nodes_with_budget(
                            node,
                            selector,
                            budget,
                            context_sensitive=True,
                            observation_spec={
                                "key": path,
                                "selector": selector,
                                "allow_multiple": False,
                                "expect_nonzero": True,
                            },
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
                            path_prefix=f"{path}.",
                            slot_prefix=slot,
                        )
                elif field_type == "nested_list":
                    selector = _field_selector(field)
                    matches = (
                        _select_nodes_with_budget(
                            node,
                            selector,
                            budget,
                            context_sensitive=True,
                            observation_spec={
                                "key": path,
                                "selector": selector,
                                "allow_multiple": True,
                                "expect_nonzero": True,
                            },
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
                                path_prefix=f"{path}.",
                                slot_prefix=slot + (len(items),),
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
                        path=path,
                        max_output_chars=field_limit,
                        output_error_kind=field_error_kind,
                        output_error_limit=field_error_limit,
                    )
                else:
                    selector = _field_selector(field)
                    matches = (
                        _select_nodes_with_budget(
                            node,
                            selector,
                            budget,
                            context_sensitive=True,
                            observation_spec={
                                "key": path,
                                "selector": selector,
                                "allow_multiple": False,
                                "expect_nonzero": True,
                            },
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
                        value = (
                            _bounded_join(
                                matches,
                                join_with,
                                field_limit,
                                coerce=coerce_value,
                                strip_result=True,
                                error_kind=field_error_kind,
                                error_limit=field_error_limit,
                            )
                            if len(matches) > 1
                            else _extract_text_from_node(matches[0])
                        )

                value = _apply_transforms(
                    value,
                    transforms,
                    base_url,
                    budget.limits.max_rendered_output_chars if budget.enforce_output else None,
                )
        except _SchemaBudgetExceeded:
            if nested_snapshot is not None:
                budget.restore_output_prefix(slot, nested_snapshot)
            raise
        if nested_snapshot is not None and value is None:
            budget.restore_output_prefix(slot, nested_snapshot)
        if value is not None:
            if field_type not in {"nested", "nested_list"}:
                budget.retain_output(value, slot=slot)
            extracted[name] = value
            ctx[name] = value

    for field in computed_fields:
        name = str(field.get("name") or "").strip()
        if not name:
            continue
        slot = slot_prefix + (name,)
        transforms = field.get("transforms")
        field_limit, field_error_kind, field_error_limit = construction_bound(
            slot,
            transforms,
        )
        if "template" in field and isinstance(field.get("template"), str):
            field_limit = budget.limits.max_rendered_output_chars
            field_error_kind = "rendered_output"
            field_error_limit = budget.limits.max_rendered_output_chars
        with _construction_output_bound(
            field_limit,
            field_error_kind,
            field_error_limit,
        ):
            value = None
            if "template" in field and isinstance(field.get("template"), str):
                value, template_error = _render_computed_template(
                    field["template"],
                    ctx,
                    budget.limits,
                    field_limit,
                    field_error_kind,
                    field_error_limit,
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
                        field_limit,
                        field_error_kind,
                        field_error_limit,
                    )
                elif isinstance(source, str):
                    value = ctx.get(source)
                elif "value" in field:
                    value = field.get("value")
            value = _apply_transforms(
                value,
                transforms,
                base_url,
                budget.limits.max_rendered_output_chars if budget.enforce_output else None,
            )
        if value is not None:
            budget.retain_output(value, slot=slot)
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
_LEGACY_MULTI_SELECTOR_KEYS = {
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


def _preflight_rule_selectors(
    rules: dict[str, Any],
    limits: _SchemaLimits,
    *,
    initial_fields: int = 0,
) -> list[tuple[str, str]]:
    selectors: list[tuple[str, str]] = []
    total_fields = initial_fields
    stack: list[tuple[dict[str, Any], str, int]] = [(rules, "", 0)]
    while stack:
        current, prefix, depth = stack.pop()
        if limits.max_depth is not None and depth > limits.max_depth:
            raise _schema_limit_error("schema_depth", limits.max_depth)
        for key in _SCHEMA_SELECTOR_KEYS:
            for expr in ensure_sequence(current.get(key)):
                selectors.append((f"{prefix}{key}", expr))
        pagination = current.get("pagination")
        if isinstance(pagination, dict):
            for key in _PAGINATION_SELECTOR_KEYS:
                for expr in ensure_sequence(pagination.get(key)):
                    selectors.append((f"{prefix}pagination.{key}", expr))
        alternates = current.get("alternates")
        if not isinstance(alternates, list):
            continue
        children: list[tuple[dict[str, Any], str, int]] = []
        for index, alternate in enumerate(alternates):
            if not isinstance(alternate, dict):
                continue
            total_fields += 1
            if limits.max_total_fields is not None and total_fields > limits.max_total_fields:
                raise _schema_limit_error("schema_fields", limits.max_total_fields)
            children.append(
                (
                    alternate,
                    f"{prefix}alternates[{index}].",
                    depth + 1,
                )
            )
        stack.extend(reversed(children))
    return selectors


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
                    "check_html": False,
                }
            )
    return specs


def _legacy_validation_specs(
    rule_selectors: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for key, selector in rule_selectors:
        selector_key = key.rsplit(".", 1)[-1]
        is_pagination = ".pagination." in f".{key}."
        specs.append(
            {
                "key": key,
                "selector": selector,
                "allow_multiple": is_pagination or selector_key in _LEGACY_MULTI_SELECTOR_KEYS,
                "expect_nonzero": not is_pagination,
                "check_html": True,
            }
        )
    return specs


def _validation_spec_identity(spec: Mapping[str, Any]) -> tuple[Any, str]:
    return spec.get("key"), str(spec.get("selector") or "").strip()


def _unique_validation_specs(specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for spec in specs:
        identity = _validation_spec_identity(spec)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(spec)
    return unique


def _record_validation_selection(
    observations: dict[tuple[Any, str], dict[str, Any]],
    spec: dict[str, Any],
    matches: Sequence[Any],
    failed: bool,
) -> None:
    identity = _validation_spec_identity(spec)
    observation = observations.setdefault(
        identity,
        {"spec": spec, "count": 0, "failed": False},
    )
    observation["count"] += len(matches)
    observation["failed"] = observation["failed"] or failed


def _evaluate_validation_rules(
    document: HtmlElement,
    budget: _SchemaBudget,
    *,
    configured_specs: Sequence[dict[str, Any]],
    invalid_selectors: set[tuple[Any, str]],
) -> list[dict[str, Any]]:
    observations: dict[tuple[Any, str], dict[str, Any]] = {}
    budget.validation_selection_cache = {}
    budget.selection_observer = lambda spec, matches, failed: _record_validation_selection(
        observations,
        spec,
        matches,
        failed,
    )
    for spec in configured_specs:
        identity = _validation_spec_identity(spec)
        if identity in invalid_selectors or not spec.get("check_html", True):
            continue
        selector = identity[1]
        if not selector:
            continue
        _select_nodes_with_budget_status(
            document,
            selector,
            budget,
            observation_spec=spec,
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
    try:
        if is_dsl:
            records = _preflight_schema_fields(normalized_rules, limits)
        rule_selectors = _preflight_rule_selectors(
            normalized_rules,
            limits,
            initial_fields=len(records),
        )
    except _SchemaBudgetExceeded as exc:
        return _complexity_validation_result(
            exc.code,
            include_counts=include_counts,
        )

    errors = _template_validation_errors(records, limits)
    warnings: list[dict[str, Any]] = []
    selector_counts: dict[str, int] = {}
    compile_specs = _legacy_validation_specs(rule_selectors)
    dsl_specs = _iter_schema_dsl_selector_specs(normalized_rules, records) if is_dsl else []
    compile_specs.extend(dsl_specs)
    compile_specs = _unique_validation_specs(compile_specs)
    invalid_selectors: set[tuple[Any, str]] = set()

    for spec in compile_specs:
        key = spec.get("key")
        stripped = (spec.get("selector") or "").strip()
        if not stripped:
            continue
        error = _compile_only_selector_error(stripped)
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
                document,
                budget,
                configured_specs=compile_specs,
                invalid_selectors=invalid_selectors,
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
            node
            for node in _select_nodes_with_budget(
                document,
                base_selector,
                budget,
                observation_spec={
                    "key": "baseSelector",
                    "selector": base_selector,
                    "allow_multiple": False,
                    "expect_nonzero": True,
                },
            )
            if isinstance(node, HtmlElement)
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
                path_prefix="baseFields.",
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
                path_prefix="fields.",
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
            root_slot = ("root", key)
            joined = _bounded_join(
                value,
                "\n",
                budget.remaining_output_chars(root_slot),
                coerce=lambda item: item.strip() or None,
                error_limit=budget.limits.max_retained_output_chars,
            )
            projected = joined if joined else value
        else:
            root_slot = ("root", key)
            projected = value
        budget.retain_output(projected, slot=root_slot)
        result[key] = projected

    result["extraction_successful"] = any(_has_nonempty_value(value) for value in schema_fields.values())
    return result


def _first_configured_rule(
    rules: dict[str, Any],
    keys: Sequence[str],
) -> tuple[str, Any]:
    for key in keys:
        value = rules.get(key)
        if value:
            return key, value
    return keys[0], None


def _extract_legacy_schema_fields(
    document: HtmlElement,
    rules: dict[str, Any],
    result: dict[str, Any],
    budget: _SchemaBudget,
) -> dict[str, Any]:
    base_key, base_selectors = _first_configured_rule(
        rules,
        (
            "base_xpath",
            "base_selector",
            "entry_xpath",
            "entry_selector",
            "item_xpath",
            "items_xpath",
        ),
    )
    nodes: list[HtmlElement] = []
    for selector in ensure_sequence(base_selectors):
        nodes.extend(
            node
            for node in _select_nodes_with_budget(
                document,
                selector,
                budget,
                observation_spec={
                    "key": base_key,
                    "selector": selector,
                    "allow_multiple": True,
                    "expect_nonzero": True,
                },
            )
            if isinstance(node, HtmlElement)
        )
    base_node = nodes[0] if nodes else document

    summary_join = str(rules.get("summary_join_with") or " ")
    content_join = str(rules.get("content_join_with") or "\n")
    title_key, title_selectors = _first_configured_rule(
        rules,
        ("title_xpath", "title_selector"),
    )
    title = _extract_value_with_budget(
        base_node,
        title_selectors,
        budget,
        key=title_key,
        slot=("root", "title"),
        join=False,
    )
    if title:
        budget.retain_output(title, slot=("root", "title"))
        result["title"] = title

    summary_key, summary_selectors = _first_configured_rule(
        rules,
        ("summary_xpath", "description_xpath", "summary_selector"),
    )
    summary = _extract_value_with_budget(
        base_node,
        summary_selectors,
        budget,
        key=summary_key,
        slot=("root", "summary"),
        join=True,
        join_with=summary_join,
    )
    if summary:
        budget.retain_output(summary, slot=("root", "summary"))
        result["summary"] = summary

    content_key, content_selectors = _first_configured_rule(
        rules,
        ("content_xpath", "content_selector"),
    )
    content = _extract_value_with_budget(
        base_node,
        content_selectors,
        budget,
        key=content_key,
        slot=("root", "content"),
        join=True,
        join_with=content_join,
    )
    if content:
        budget.retain_output(content, slot=("root", "content"))
        result["content"] = content

    author_key, author_selectors = _first_configured_rule(
        rules,
        ("author_xpath", "author_selector"),
    )
    author = _extract_value_with_budget(
        base_node,
        author_selectors,
        budget,
        key=author_key,
        slot=("root", "author"),
        join=False,
    )
    if author:
        budget.retain_output(author, slot=("root", "author"))
        result["author"] = author

    published_key, published_selectors = _first_configured_rule(
        rules,
        ("published_xpath", "date_xpath", "date_selector"),
    )
    published_raw = _extract_value_with_budget(
        base_node,
        published_selectors,
        budget,
        key=published_key,
        slot=("root", "published_raw"),
        join=False,
    )
    if published_raw:
        budget.retain_output(published_raw, slot=("root", "published_raw"))
        result["published_raw"] = published_raw
        fmt = rules.get("published_format") or rules.get("date_format")
        parsed = normalize_datetime(
            published_raw,
            fmt if isinstance(fmt, str) else None,
        )
        if parsed:
            budget.retain_output(parsed, slot=("root", "published"))
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
        _preflight_rule_selectors(
            rules,
            limits,
            initial_fields=len(records),
        )
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
