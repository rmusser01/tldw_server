"""Authoritative, parser-only validation for standalone HTML presentations.

The source is parsed as inert text. This module is neither a sanitizer nor a
renderer and deliberately exposes only derived scalar metadata.
"""

from __future__ import annotations

import contextlib
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

try:
    import html5lib
    import tinycss2
    from html5lib import _tokenizer as html5lib_tokenizer
    from html5lib import html5parser
except ImportError:  # pragma: no cover - direct dependencies are smoke tested
    html5lib = None  # type: ignore[assignment]
    tinycss2 = None  # type: ignore[assignment]
    html5lib_tokenizer = None  # type: ignore[assignment]
    html5parser = None  # type: ignore[assignment]

from .standalone_html_contracts import (
    StandaloneHtmlValidationError,
    StandaloneHtmlValidationResult,
)

DeliveryStyle = Literal["speaker-led", "self-guided"]

MAX_DOCUMENT_BYTES = 1_048_576
MAX_HTML_TOKENS = 50_000
MAX_HTML_TOKEN_BYTES = 65_536
MAX_HTML_ELEMENTS = 10_000
MAX_HTML_ATTRIBUTES = 20_000
MAX_HTML_DEPTH = 128
MAX_SLIDES = 30
MAX_STYLE_ELEMENTS = 64
MAX_CSS_BYTES = 524_288
MAX_CSS_TOKENS = 100_000
MAX_CSS_DECLARATIONS = 10_000
MAX_CSS_TOKEN_BYTES = 65_536
MAX_CSS_DEPTH = 64
MAX_CSS_ERRORS = 100
MAX_INDEXABLE_TEXT = 250_000
_UTF8_MEASUREMENT_CHARS = 16_384

_HTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
_XMLNS_NAMESPACE = "http://www.w3.org/2000/xmlns/"
_HTML_WHITESPACE = " \t\n\r\f"
_ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_FORBIDDEN_ELEMENTS = frozenset(
    {
        "applet",
        "base",
        "embed",
        "fencedframe",
        "form",
        "frame",
        "frameset",
        "iframe",
        "img",
        "input",
        "link",
        "object",
        "portal",
        "picture",
        "select",
        "source",
        "textarea",
        "track",
        "video",
        "audio",
    }
)
_URL_ATTRIBUTES = frozenset(
    {
        "action",
        "altimg",
        "about",
        "archive",
        "background",
        "base",
        "cite",
        "classid",
        "codebase",
        "data",
        "definitionurl",
        "dynsrc",
        "formaction",
        "href",
        "icon",
        "imagesrcset",
        "itemid",
        "itemtype",
        "longdesc",
        "lowsrc",
        "manifest",
        "ping",
        "poster",
        "prefix",
        "profile",
        "resource",
        "src",
        "srcset",
        "usemap",
        "vocab",
    }
)
_CSS_RESOURCE_ATTRIBUTES = frozenset(
    {
        "clip-path",
        "color-profile",
        "cursor",
        "fill",
        "filter",
        "marker-end",
        "marker-mid",
        "marker-start",
        "mask",
        "stroke",
    }
)
_UNMISTAKABLE_URL_SCHEMES = frozenset(
    {
        "about",
        "blob",
        "cid",
        "data",
        "file",
        "ftp",
        "geo",
        "git",
        "http",
        "https",
        "irc",
        "ircs",
        "javascript",
        "mailto",
        "news",
        "nntp",
        "sftp",
        "sms",
        "ssh",
        "tel",
        "urn",
        "webcal",
        "ws",
        "wss",
    }
)
_FOREIGN_ACTIVE_ELEMENTS = frozenset({"animate", "animatemotion", "animatetransform", "discard", "mpath", "set"})
_CSS_RESOURCE_FUNCTIONS = frozenset({"-webkit-image-set", "element", "image", "image-set", "paint", "src", "url"})
_SEMANTIC_ELEMENTS = frozenset(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "dt",
        "dd",
        "blockquote",
        "pre",
        "code",
        "caption",
        "th",
        "td",
        "figcaption",
    }
)
_EXCLUDED_ELEMENTS = frozenset({"script", "style", "template", "noscript"})
_EXCLUDED_CLASSES = frozenset(
    {
        "notes",
        "deck-header",
        "deck-footer",
        "slide-number",
        "progress",
        "navigation",
        "nav",
    }
)
_BIDI_FORMATTING = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\u206a",
        "\u206b",
        "\u206c",
        "\u206d",
        "\u206e",
        "\u206f",
    }
)
_SCRIPT_SINK_TOKEN_PATTERNS = (
    ("fetch", "("),
    ("import", "("),
    ("open", "("),
    ("new", "eventsource", "("),
    ("new", "sharedworker", "("),
    ("new", "websocket", "("),
    ("new", "worker", "("),
    ("new", "globalthis", ".", "worker", "("),
    ("new", "xmlhttprequest", "("),
    ("navigator", ".", "sendbeacon", "("),
    ("navigator", ".", "serviceworker", ".", "register", "("),
    ("document", ".", "location", "="),
    ("document", ".", "location", ".", "href"),
    ("document", ".", "location", ".", "assign", "("),
    ("document", ".", "location", ".", "replace", "("),
    ("document", ".", "location", ".", "reload", "("),
    ("location", "="),
    ("location", ".", "href"),
    ("location", ".", "assign", "("),
    ("location", ".", "replace", "("),
    ("location", ".", "reload", "("),
    ("history", ".", "back", "("),
    ("history", ".", "forward", "("),
    ("history", ".", "go", "("),
    ("history", ".", "pushstate", "("),
    ("history", ".", "replacestate", "("),
    ("navigation", ".", "back", "("),
    ("navigation", ".", "forward", "("),
    ("navigation", ".", "navigate", "("),
    ("navigation", ".", "reload", "("),
    ("navigation", ".", "traverseto", "("),
    ("document", ".", "cookie"),
    ("caches", ".", "open", "("),
    ("localstorage",),
    ("sessionstorage",),
    ("indexeddb",),
)
_SCRIPT_GLOBAL_QUALIFIERS = frozenset({"globalthis", "parent", "self", "top", "window"})
_SCRIPT_GLOBAL_NAMES = frozenset(
    {
        "caches",
        "document",
        "eventsource",
        "fetch",
        "history",
        "indexeddb",
        "localstorage",
        "location",
        "navigator",
        "navigation",
        "open",
        "sessionstorage",
        "sharedworker",
        "websocket",
        "worker",
        "xmlhttprequest",
    }
)
_SCRIPT_ALIASABLE_SINK_PATTERNS = (
    ("eventsource",),
    ("fetch",),
    ("sharedworker",),
    ("websocket",),
    ("worker",),
    ("xmlhttprequest",),
    ("open",),
    ("document", ".", "location", ".", "assign"),
    ("document", ".", "location", ".", "replace"),
    ("document", ".", "location", ".", "reload"),
    ("location", ".", "assign"),
    ("location", ".", "replace"),
    ("location", ".", "reload"),
    ("navigator", ".", "sendbeacon"),
    ("navigator", ".", "serviceworker", ".", "register"),
    ("history", ".", "back"),
    ("history", ".", "forward"),
    ("history", ".", "go"),
    ("history", ".", "pushstate"),
    ("history", ".", "replacestate"),
    ("navigation", ".", "back"),
    ("navigation", ".", "forward"),
    ("navigation", ".", "navigate"),
    ("navigation", ".", "reload"),
    ("navigation", ".", "traverseto"),
    ("caches", ".", "open"),
)


class _BudgetExceeded(RuntimeError):
    """Private source-free budget abort used across parser boundaries."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class _HtmlPreflight:
    doctype_start: int
    html_end: int


def _fail_invalid(
    reason: str,
    *,
    line: int | None = None,
    column: int | None = None,
) -> None:
    raise StandaloneHtmlValidationError(
        "standalone_html_invalid_document",
        status_code=422,
        reason=reason,
        line=line,
        column=column,
    )


def _fail_budget(reason: str) -> None:
    raise StandaloneHtmlValidationError(
        "standalone_html_validation_budget_exceeded",
        status_code=422,
        reason=reason,
    )


def _fail_unavailable() -> None:
    raise StandaloneHtmlValidationError("validator_unavailable", status_code=503)


def _ascii_lower(value: str) -> str:
    """Fold ASCII A-Z without changing source length or Unicode code points."""
    return str.translate(value, _ASCII_LOWER_TRANSLATION)


def _preflight_document_input(document: str | bytes) -> int:
    """Validate type, UTF-8, and byte size before full copies or IPC."""
    if isinstance(document, bytes):
        document_bytes = bytes.__len__(document)
        if document_bytes > MAX_DOCUMENT_BYTES:
            _fail_budget("document_bytes")
        encoding_failed = False
        try:
            bytes.decode(document, "utf-8", "strict")
        except UnicodeDecodeError:
            encoding_failed = True
        if encoding_failed:
            _fail_invalid("document_encoding")
        return document_bytes

    if not isinstance(document, str):
        _fail_invalid("document_type")

    character_count = str.__len__(document)
    if character_count > MAX_DOCUMENT_BYTES:
        _fail_budget("document_bytes")

    document_bytes = 0
    encoding_failed = False
    for start in range(0, character_count, _UTF8_MEASUREMENT_CHARS):
        chunk = str.__getitem__(
            document,
            slice(start, min(start + _UTF8_MEASUREMENT_CHARS, character_count)),
        )
        try:
            encoded = str.encode(chunk, "utf-8", "strict")
        except UnicodeEncodeError:
            encoding_failed = True
            break
        document_bytes += bytes.__len__(encoded)
        if document_bytes > MAX_DOCUMENT_BYTES:
            _fail_budget("document_bytes")
    if encoding_failed:
        _fail_invalid("document_encoding")
    return document_bytes


def _local_name(name: Any) -> str:
    value = str(name)
    if "}" in value:
        value = value.rsplit("}", 1)[1]
    if ":" in value:
        value = value.rsplit(":", 1)[1]
    return value.lower()


def _namespace(name: Any) -> str | None:
    value = str(name)
    if value.startswith("{") and "}" in value:
        return value[1 : value.index("}")]
    return None


def _is_namespace_declaration(name: Any) -> bool:
    value = str(name).lower()
    return _namespace(name) == _XMLNS_NAMESPACE or value == "xmlns" or value.startswith("xmlns:")


def _attribute_value_has_unmistakable_url_marker(value: str) -> bool:
    """Recognize explicit URL syntax without classifying fragments or paths."""
    lowered = value.lower()
    if "//" in lowered:
        return True

    scheme_characters = "+-."
    index = 0
    while index < len(lowered):
        character = lowered[index]
        previous_is_scheme_character = index > 0 and (
            lowered[index - 1].isascii() and (lowered[index - 1].isalnum() or lowered[index - 1] in scheme_characters)
        )
        if character.isascii() and character.isalpha() and not previous_is_scheme_character:
            end = index + 1
            while end < len(lowered):
                candidate = lowered[end]
                if not candidate.isascii() or not (candidate.isalnum() or candidate in scheme_characters):
                    break
                end += 1
            if end < len(lowered) and lowered[end] == ":" and lowered[index:end] in _UNMISTAKABLE_URL_SCHEMES:
                return True
            index = end
            continue
        index += 1

    search_from = 0
    while True:
        function_start = lowered.find("url", search_from)
        if function_start < 0:
            return False
        before_is_identifier = function_start > 0 and (
            lowered[function_start - 1].isascii()
            and (lowered[function_start - 1].isalnum() or lowered[function_start - 1] in {"_", "-"})
        )
        after = function_start + 3
        while after < len(lowered) and lowered[after] in _HTML_WHITESPACE:
            after += 1
        if not before_is_identifier and after < len(lowered) and lowered[after] == "(":
            return True
        search_from = function_start + 3


def _is_html_element(element: Any, local_name: str) -> bool:
    return _namespace(element.tag) == _HTML_NAMESPACE and _local_name(element.tag) == local_name


def _class_tokens(element: Any) -> frozenset[str]:
    value = str(element.attrib.get("class", ""))
    tokens: list[str] = []
    current: list[str] = []
    for character in value:
        if character in _HTML_WHITESPACE:
            if current:
                tokens.append("".join(current))
                current.clear()
        else:
            current.append(character)
    if current:
        tokens.append("".join(current))
    return frozenset(tokens)


def _is_forbidden_control(character: str) -> bool:
    codepoint = ord(character)
    return (codepoint < 0x20 and character not in {"\t", "\n", "\r"}) or 0x7F <= codepoint <= 0x9F


def _collapse_html_whitespace(value: str) -> str:
    chunks: list[str] = []
    current: list[str] = []
    for character in value:
        if character in _HTML_WHITESPACE:
            if current:
                chunks.append("".join(current))
                current.clear()
        else:
            current.append(character)
    if current:
        chunks.append("".join(current))
    return " ".join(chunks)


def _script_tokens(source: str) -> list[str]:
    """Return bounded ASCII diagnostic tokens without interpreting JavaScript."""
    tokens: list[str] = []
    index = 0
    contexts: list[list[Any]] = [["expression", 0, True]]

    def append_tokens(*values: str) -> None:
        tokens.extend(values)
        if len(tokens) > MAX_HTML_TOKENS:
            raise _BudgetExceeded("html_tokens")

    expression_prefixes = frozenset(
        {
            "await",
            "case",
            "delete",
            "in",
            "instanceof",
            "new",
            "of",
            "return",
            "throw",
            "typeof",
            "void",
            "yield",
        }
    )
    while index < len(source):
        context = contexts[-1]
        character = source[index]
        if context[0] == "template":
            if character == "\\":
                index = min(index + 2, len(source))
            elif character == "`":
                contexts.pop()
                contexts[-1][2] = False
                index += 1
            elif source.startswith("${", index):
                if len(contexts) >= MAX_HTML_DEPTH:
                    _fail_invalid("script_policy")
                contexts.append(["interpolation", 1, True])
                index += 2
            else:
                index += 1
            continue
        if character in _HTML_WHITESPACE:
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            close = source.find("*/", index + 2)
            index = len(source) if close < 0 else close + 2
            continue
        if character == "/" and context[2]:
            cursor = index + 1
            in_character_class = False
            regex_closed = False
            while cursor < len(source):
                candidate = source[cursor]
                if candidate == "\\":
                    cursor = min(cursor + 2, len(source))
                    continue
                if candidate in {"\n", "\r"}:
                    break
                if candidate == "[":
                    in_character_class = True
                elif candidate == "]":
                    in_character_class = False
                elif candidate == "/" and not in_character_class:
                    cursor += 1
                    while cursor < len(source) and source[cursor].isascii() and source[cursor].isalpha():
                        cursor += 1
                    regex_closed = True
                    break
                cursor += 1
            if regex_closed:
                context[2] = False
                index = cursor
                continue
        if character in {"'", '"'}:
            quote = character
            quote_start = index
            index += 1
            value_start = index
            value_end = index
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == quote:
                    value_end = index
                    index += 1
                    break
                else:
                    index += 1
            before = quote_start - 1
            while before >= 0 and source[before] in _HTML_WHITESPACE:
                before -= 1
            after = index
            while after < len(source) and source[after] in _HTML_WHITESPACE:
                after += 1
            property_name = source[value_start:value_end]
            if (
                before >= 0
                and source[before] == "["
                and after < len(source)
                and source[after] == "]"
                and property_name
                and property_name.isascii()
                and all(candidate.isalnum() or candidate in {"_", "$"} for candidate in property_name)
            ):
                append_tokens(".", property_name.lower())
            context[2] = False
            continue
        if character == "`":
            if len(contexts) >= MAX_HTML_DEPTH:
                _fail_invalid("script_policy")
            contexts.append(["template", 0, False])
            index += 1
            continue
        if context[0] == "interpolation":
            if character == "{":
                context[1] += 1
                if context[1] > MAX_HTML_DEPTH:
                    _fail_invalid("script_policy")
                context[2] = True
                index += 1
                continue
            if character == "}":
                context[1] -= 1
                index += 1
                if context[1] == 0:
                    contexts.pop()
                else:
                    context[2] = False
                continue
        if character.isascii() and (character.isalnum() or character in {"_", "$"}):
            start = index
            index += 1
            while index < len(source):
                candidate = source[index]
                if not candidate.isascii() or not (candidate.isalnum() or candidate in {"_", "$"}):
                    break
                index += 1
            token = source[start:index].lower()
            append_tokens(token)
            context[2] = token in expression_prefixes
            continue
        if character in {".", "(", "="}:
            append_tokens(character)
        if character in {
            "(",
            "=",
            "[",
            "{",
            ",",
            ":",
            "?",
            "!",
            "~",
            "+",
            "-",
            "*",
            "%",
            "&",
            "|",
            "^",
            "<",
            ">",
            ";",
            "/",
        }:
            context[2] = True
        elif character in {")", "]"}:
            context[2] = False
        index += 1
    return tokens


def _contains_token_pattern(tokens: list[str], pattern: tuple[str, ...]) -> bool:
    if len(pattern) > len(tokens):
        return False
    width = len(pattern)
    return any(
        tuple(tokens[index : index + width]) == pattern and (index == 0 or tokens[index - 1] != ".")
        for index in range(len(tokens) - width + 1)
    )


def _normalize_script_global_qualifiers(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            index + 2 < len(tokens)
            and tokens[index] in _SCRIPT_GLOBAL_QUALIFIERS
            and tokens[index + 1] == "."
            and tokens[index + 2] in _SCRIPT_GLOBAL_NAMES
        ):
            normalized.append(tokens[index + 2])
            index += 3
        else:
            normalized.append(tokens[index])
            index += 1
    return normalized


def _script_has_simple_sink_alias(tokens: list[str]) -> bool:
    aliases: set[str] = set()
    for index in range(len(tokens) - 3):
        alias = tokens[index + 1]
        rhs_start = index + 3
        if (
            tokens[index] in {"const", "let", "var"}
            and alias.isascii()
            and alias.replace("_", "a").replace("$", "a").isalnum()
            and tokens[index + 2] == "="
            and any(
                tuple(tokens[rhs_start : rhs_start + len(pattern)]) == pattern
                for pattern in _SCRIPT_ALIASABLE_SINK_PATTERNS
            )
        ):
            aliases.add(alias)
    return any(
        token in aliases
        and index + 1 < len(tokens)
        and tokens[index + 1] == "("
        and (index == 0 or tokens[index - 1] != ".")
        for index, token in enumerate(tokens)
    )


def _script_has_obvious_sink(source: str) -> bool:
    if "sourcemappingurl" in source.lower():
        return True
    tokens = _normalize_script_global_qualifiers(_script_tokens(source))
    return _script_has_simple_sink_alias(tokens) or any(
        _contains_token_pattern(tokens, pattern) for pattern in _SCRIPT_SINK_TOKEN_PATTERNS
    )


def _css_value_has_resource(value: str) -> bool:
    if tinycss2 is None:
        _fail_unavailable()
    parse_failed = False
    try:
        nodes = tinycss2.parse_component_value_list(value, skip_comments=False)
    except Exception:  # noqa: BLE001 - third-party CSS parser must fail closed
        parse_failed = True
        nodes = []
    if parse_failed:
        return True
    stack = list(nodes)
    while stack:
        node = stack.pop()
        node_type = getattr(node, "type", "")
        if node_type in {"error", "url"}:
            return True
        if node_type == "function":
            if getattr(node, "lower_name", "") in _CSS_RESOURCE_FUNCTIONS:
                return True
            stack.extend(node.arguments)
        elif node_type in {"{} block", "[] block", "() block"}:
            stack.extend(node.content)
    return False


def _check_token_size(source: str, start: int, end: int, reason: str) -> None:
    if len(source[start:end].encode("utf-8")) > MAX_HTML_TOKEN_BYTES:
        raise _BudgetExceeded(reason)


def _scan_tag_end(source: str, start: int) -> int:
    quote: str | None = None
    index = start
    while index < len(source):
        character = source[index]
        if quote is not None:
            if character == quote:
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == ">":
            return index + 1
        index += 1
    _fail_invalid("html_unterminated_tag")
    raise AssertionError("unreachable")


def _count_attributes(tag_source: str, name_end: int) -> int:
    count = 0
    index = name_end
    end = len(tag_source) - 1
    while index < end:
        while index < end and tag_source[index] in _HTML_WHITESPACE:
            index += 1
        if index >= end or tag_source[index] in {">", "/"}:
            index += 1
            continue
        if tag_source[index] in {"=", "'", '"'}:
            _fail_invalid("html_attribute_name")
        count += 1
        while index < end and tag_source[index] not in _HTML_WHITESPACE + "=/>\"'":
            index += 1
        while index < end and tag_source[index] in _HTML_WHITESPACE:
            index += 1
        if index < end and tag_source[index] == "=":
            index += 1
            while index < end and tag_source[index] in _HTML_WHITESPACE:
                index += 1
            if index < end and tag_source[index] in {"'", '"'}:
                quote = tag_source[index]
                index += 1
                while index < end and tag_source[index] != quote:
                    index += 1
                if index >= end:
                    _fail_invalid("html_unterminated_attribute")
                index += 1
            else:
                while index < end and tag_source[index] not in _HTML_WHITESPACE + ">":
                    index += 1
    return count


def _preflight_html(source: str) -> _HtmlPreflight:
    lower_source = _ascii_lower(source)
    token_count = 0
    start_tag_count = 0
    attribute_count = 0
    stack: list[str] = []
    starts: dict[str, list[int]] = {name: [] for name in ("html", "head", "title", "body")}
    ends: dict[str, list[int]] = {name: [] for name in ("html", "head", "title", "body")}
    doctype_positions: list[int] = []
    index = 0
    text_start = 0

    def add_token(
        start: int,
        end: int,
        reason: str,
        *,
        count_character_references: bool = False,
    ) -> None:
        nonlocal token_count
        token_count += 1
        if count_character_references:
            token_count += source.count("&", start, end)
        if token_count > MAX_HTML_TOKENS:
            raise _BudgetExceeded("html_tokens")
        _check_token_size(source, start, end, reason)

    while index < len(source):
        if source[index] != "<":
            index += 1
            continue
        if index > text_start:
            add_token(
                text_start,
                index,
                "html_text_token",
                count_character_references=True,
            )

        if source.startswith("<!--", index):
            close = source.find("-->", index + 4)
            if close < 0:
                _fail_invalid("html_unterminated_comment")
            end = close + 3
            add_token(index, end, "html_comment_token")
            index = end
            text_start = index
            continue

        if lower_source.startswith("<!doctype", index):
            end = _scan_tag_end(source, index + 2)
            add_token(index, end, "html_doctype_token")
            if _ascii_lower(source[index:end].strip()) != "<!doctype html>":
                _fail_invalid("html_doctype")
            doctype_positions.append(index)
            index = end
            text_start = index
            continue

        if source.startswith("<!", index) or source.startswith("<?", index):
            _fail_invalid("html_declaration")

        is_end = source.startswith("</", index)
        name_start = index + (2 if is_end else 1)
        name_end = name_start
        while name_end < len(source) and (source[name_end].isalnum() or source[name_end] in {"-", ":"}):
            name_end += 1
        if name_end == name_start:
            _fail_invalid("html_tag_name")
        name = _ascii_lower(source[name_start:name_end])
        end = _scan_tag_end(source, name_end)
        add_token(
            index,
            end,
            "html_tag_token",
            count_character_references=True,
        )

        if is_end:
            if source[name_end : end - 1].strip():
                _fail_invalid("html_end_tag")
            if name in ends:
                ends[name].append(end)
            if name in stack:
                while stack:
                    popped = stack.pop()
                    if popped == name:
                        break
        else:
            start_tag_count += 1
            if start_tag_count > MAX_HTML_ELEMENTS:
                raise _BudgetExceeded("html_elements")
            current_attributes = _count_attributes(source[index:end], name_end - index)
            attribute_count += current_attributes
            if attribute_count > MAX_HTML_ATTRIBUTES:
                raise _BudgetExceeded("html_attributes")
            if name in starts:
                starts[name].append(index)
            self_closing = source[index:end].rstrip().endswith("/>")
            if not self_closing and name not in _VOID_ELEMENTS:
                stack.append(name)
                if len(stack) > MAX_HTML_DEPTH:
                    raise _BudgetExceeded("html_depth")

        index = end
        text_start = index

        if not is_end and name in {"script", "style"}:
            close_marker = f"</{name}"
            close_start = lower_source.find(close_marker, index)
            if close_start < 0:
                _fail_invalid("html_unterminated_raw_text")
            if close_start > index:
                add_token(index, close_start, "html_raw_text_token")
            index = close_start
            text_start = index

    if text_start < len(source):
        add_token(
            text_start,
            len(source),
            "html_text_token",
            count_character_references=True,
        )

    if len(doctype_positions) != 1 or any(len(starts[name]) != 1 for name in starts):
        _fail_invalid("html_document_structure")
    if any(len(ends[name]) != 1 for name in ends):
        _fail_invalid("html_document_structure")

    doctype_start = doctype_positions[0]
    html_start = starts["html"][0]
    head_start = starts["head"][0]
    title_start = starts["title"][0]
    title_end = ends["title"][0]
    head_end = ends["head"][0]
    body_start = starts["body"][0]
    body_end = ends["body"][0]
    html_end = ends["html"][0]
    if not (
        doctype_start < html_start < head_start < title_start < title_end < head_end <= body_start < body_end < html_end
    ):
        _fail_invalid("html_document_order")
    if source[:doctype_start].strip() or source[html_end:].strip():
        _fail_invalid("html_document_boundary")
    return _HtmlPreflight(doctype_start=doctype_start, html_end=html_end)


def _node_depth(node: Any) -> int:
    depth = 0
    current = node
    while current is not None and getattr(current, "name", None) != "DOCUMENT_ROOT":
        depth += 1
        current = getattr(current, "parent", None)
    return depth


def _is_builder_element(node: Any) -> bool:
    return hasattr(node, "_namespace") and getattr(node, "name", None) not in {
        "DOCUMENT_ROOT",
        "DOCUMENT_FRAGMENT",
    }


def _subtree_height(node: Any) -> int:
    if not _is_builder_element(node):
        return 0
    maximum = 1
    stack = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        maximum = max(maximum, depth)
        for child in getattr(current, "childNodes", ()):
            if _is_builder_element(child):
                stack.append((child, depth + 1))
    return maximum


def _make_counting_tree_builder() -> type[Any]:
    if html5lib is None:
        _fail_unavailable()
    base_builder = html5lib.getTreeBuilder("etree")
    base_element = base_builder.elementClass
    counters = {"elements": 0, "attributes": 0}

    class CountingElement(base_element):  # type: ignore[misc, valid-type]
        def __init__(self, name: str, namespace: str | None = None) -> None:
            if counters["elements"] >= MAX_HTML_ELEMENTS:
                raise _BudgetExceeded("html_elements")
            counters["elements"] += 1
            self._bounded_attribute_count = 0
            super().__init__(name, namespace)

        def _get_bounded_attributes(self) -> Any:
            return base_element.attributes.fget(self)

        def _set_bounded_attributes(self, attributes: Any) -> None:
            new_count = len(attributes or {})
            delta = max(0, new_count - self._bounded_attribute_count)
            if counters["attributes"] + delta > MAX_HTML_ATTRIBUTES:
                raise _BudgetExceeded("html_attributes")
            counters["attributes"] += delta
            self._bounded_attribute_count = new_count
            base_element.attributes.fset(self, attributes)

        attributes = property(_get_bounded_attributes, _set_bounded_attributes)

        def _check_insertion(self, node: Any) -> None:
            subtree_height = _subtree_height(node)
            if subtree_height and _node_depth(self) + subtree_height > MAX_HTML_DEPTH:
                raise _BudgetExceeded("html_depth")

        def appendChild(self, node: Any) -> None:
            self._check_insertion(node)
            super().appendChild(node)

        def insertBefore(self, node: Any, refNode: Any) -> None:
            self._check_insertion(node)
            super().insertBefore(node, refNode)

        def cloneNode(self) -> Any:
            element = type(self)(self.name, self.namespace)
            element.attributes = dict(self.attributes)
            return element

    class CountingTreeBuilder(base_builder):  # type: ignore[misc, valid-type]
        elementClass = CountingElement

    return CountingTreeBuilder


def _parse_html(source: str) -> Any:
    if html5lib is None or html5lib_tokenizer is None or html5parser is None:
        _fail_unavailable()
    token_counter = {"count": 0}

    class CountingTokenizer(html5lib_tokenizer.HTMLTokenizer):  # type: ignore[misc]
        def __iter__(self):
            for token in super().__iter__():
                token_counter["count"] += 1
                if token_counter["count"] > MAX_HTML_TOKENS:
                    raise _BudgetExceeded("html_tokens")
                yield token

    class CountingParser(html5parser.HTMLParser):  # type: ignore[misc]
        def _parse(
            self,
            stream: str,
            innerHTML: bool = False,
            container: str = "div",
            scripting: bool = False,
            **kwargs: Any,
        ) -> None:
            self.innerHTMLMode = innerHTML
            self.container = container
            self.scripting = scripting
            self.tokenizer = CountingTokenizer(stream, parser=self, **kwargs)
            self.reset()
            try:
                self.mainLoop()
            except html5parser._ReparseException:  # type: ignore[attr-defined]
                token_counter["count"] = 0
                self.reset()
                self.mainLoop()

    parser: Any | None = None
    parse_failed = False
    line: int | None = None
    column: int | None = None
    try:
        parser = CountingParser(
            tree=_make_counting_tree_builder(),
            strict=True,
            namespaceHTMLElements=True,
        )
        return parser.parse(source, scripting=True)
    except _BudgetExceeded:
        raise
    except Exception:  # noqa: BLE001 - parser exceptions are source-redacted below
        parse_failed = True
        with contextlib.suppress(Exception):
            position = parser.tokenizer.stream.position() if parser is not None else None
            if isinstance(position, tuple) and len(position) == 2:
                line = int(position[0])
                column = int(position[1])
    if parse_failed:
        _fail_invalid("html_parse_error", line=line, column=column)
    raise AssertionError("unreachable")


def _css_is_name_start(character: str) -> bool:
    """Return whether one code point can start a CSS name."""
    return character == "_" or (character.isascii() and character.isalpha()) or ord(character) >= 0x80


def _css_is_name_character(character: str) -> bool:
    """Return whether one code point can continue a CSS name."""
    return _css_is_name_start(character) or (character.isascii() and character.isdigit()) or character == "-"


def _css_valid_escape(source: str, start: int) -> bool:
    """Recognize a CSS escape, including a terminal backslash."""
    return (
        start < len(source)
        and source[start] == "\\"
        and (start + 1 == len(source) or source[start + 1] not in "\n\r\f")
    )


def _css_escape_end(source: str, start: int) -> int:
    """Consume one CSS escape while always advancing past its backslash."""
    index = start + 1
    if index >= len(source) or source[index] in "\n\r\f":
        return index
    hex_start = index
    while index < len(source) and index - hex_start < 6 and source[index] in "0123456789abcdefABCDEF":
        index += 1
    if index > hex_start:
        if index < len(source) and source[index] in _HTML_WHITESPACE:
            index += 2 if source.startswith("\r\n", index) else 1
        return index
    return index + 1


def _css_name_end(source: str, start: int) -> int:
    """Return the end of one CSS name sequence."""
    index = start
    while index < len(source):
        if _css_is_name_character(source[index]):
            index += 1
        elif _css_valid_escape(source, index):
            index = _css_escape_end(source, index)
        else:
            break
    return index


def _css_would_start_identifier(source: str, start: int) -> bool:
    """Implement the CSS Syntax identifier-start lookahead."""
    if start >= len(source):
        return False
    first = source[start]
    if _css_is_name_start(first):
        return True
    if first == "\\":
        return _css_valid_escape(source, start)
    if first != "-" or start + 1 >= len(source):
        return False
    second = source[start + 1]
    return _css_is_name_start(second) or second == "-" or _css_valid_escape(source, start + 1)


def _css_would_start_number(source: str, start: int) -> bool:
    """Implement the CSS Syntax number-start lookahead."""

    def is_digit(index: int) -> bool:
        return index < len(source) and source[index].isascii() and source[index].isdigit()

    if start >= len(source):
        return False
    first = source[start]
    if is_digit(start):
        return True
    if first == ".":
        return is_digit(start + 1)
    if first in {"+", "-"}:
        return is_digit(start + 1) or (start + 1 < len(source) and source[start + 1] == "." and is_digit(start + 2))
    return False


def _css_numeric_token_end(source: str, start: int) -> int:
    """Consume a number and its optional percentage or dimension suffix."""

    def is_digit(index: int) -> bool:
        return index < len(source) and source[index].isascii() and source[index].isdigit()

    index = start
    if source[index] in {"+", "-"}:
        index += 1
    while is_digit(index):
        index += 1
    if index < len(source) and source[index] == "." and is_digit(index + 1):
        index += 1
        while is_digit(index):
            index += 1
    if index < len(source) and source[index] in {"e", "E"}:
        exponent = index + 1
        if exponent < len(source) and source[exponent] in {"+", "-"}:
            exponent += 1
        if is_digit(exponent):
            index = exponent + 1
            while is_digit(index):
                index += 1
    if _css_would_start_identifier(source, index):
        return _css_name_end(source, index)
    if index < len(source) and source[index] == "%":
        return index + 1
    return index


def _css_ascii_identifier_lower(source: str, start: int, end: int) -> str:
    characters: list[str] = []
    index = start
    while index < end and len(characters) <= 3:
        character = source[index]
        if character != "\\":
            if not character.isascii() or not (character.isalnum() or character in {"-", "_"}):
                return ""
            characters.append(character.lower())
            index += 1
            continue
        index += 1
        hex_start = index
        while index < end and index - hex_start < 6 and source[index] in "0123456789abcdefABCDEF":
            index += 1
        if index > hex_start:
            codepoint = int(source[hex_start:index], 16)
            if not 1 <= codepoint < 128:
                return ""
            characters.append(chr(codepoint).lower())
            if index < end and source[index] in _HTML_WHITESPACE:
                index += 2 if source.startswith("\r\n", index) else 1
        elif index < end:
            characters.append(source[index].lower())
            index += 1
        else:
            return ""
    return "".join(characters) if index == end else ""


def _css_url_token_end(source: str, start: int) -> int:
    """Return the end of an unquoted CSS URL token."""
    index = start
    while index < len(source):
        if _css_valid_escape(source, index):
            index = _css_escape_end(source, index)
            continue
        if source[index] == ")":
            return index + 1
        index += 1
    return index


def _check_css_token_size(source: str, start: int, end: int) -> None:
    if len(source[start:end].encode("utf-8")) > MAX_CSS_TOKEN_BYTES:
        raise _BudgetExceeded("css_token_bytes")


def _preflight_css(styles: list[str]) -> None:
    if len(styles) > MAX_STYLE_ELEMENTS:
        raise _BudgetExceeded("css_stylesheets")
    if sum(len(style.encode("utf-8")) for style in styles) > MAX_CSS_BYTES:
        raise _BudgetExceeded("css_bytes")

    token_count = 0
    declaration_count = 0
    stack: list[str] = []
    declaration_open: list[bool] = []
    matching = {"}": "{", "]": "[", ")": "("}
    for source in styles:
        index = 0
        while index < len(source):
            character = source[index]
            if character in _HTML_WHITESPACE:
                start = index
                while index < len(source) and source[index] in _HTML_WHITESPACE:
                    index += 1
            elif source.startswith("/*", index):
                start = index
                close = source.find("*/", index + 2)
                if close < 0:
                    _fail_invalid("css_unterminated_comment")
                index = close + 2
            elif character in {"'", '"'}:
                start = index
                quote = character
                index += 1
                while index < len(source):
                    if source[index] == "\\":
                        index += 2
                        continue
                    if source[index] == quote:
                        index += 1
                        break
                    if source[index] in {"\n", "\r"}:
                        _fail_invalid("css_unterminated_string")
                    index += 1
                else:
                    _fail_invalid("css_unterminated_string")
            elif (character == "@" and _css_would_start_identifier(source, index + 1)) or (
                character == "#"
                and index + 1 < len(source)
                and (_css_is_name_character(source[index + 1]) or _css_valid_escape(source, index + 1))
            ):
                start = index
                index = _css_name_end(source, index + 1)
            elif _css_would_start_number(source, index):
                start = index
                index = _css_numeric_token_end(source, index)
            elif _css_would_start_identifier(source, index):
                start = index
                index = _css_name_end(source, index)
                if index < len(source) and source[index] == "(":
                    argument_start = index + 1
                    while argument_start < len(source) and source[argument_start] in _HTML_WHITESPACE:
                        argument_start += 1
                    if _css_ascii_identifier_lower(source, start, index) == "url" and (
                        argument_start >= len(source) or source[argument_start] not in {"'", '"'}
                    ):
                        index = _css_url_token_end(source, index + 1)
                    else:
                        index += 1
                        stack.append("(")
                        if len(stack) > MAX_CSS_DEPTH:
                            raise _BudgetExceeded("css_depth")
            else:
                start = index
                index += 1
                if character in "{[(":
                    stack.append(character)
                    if character == "{":
                        declaration_open.append(False)
                    if len(stack) > MAX_CSS_DEPTH:
                        raise _BudgetExceeded("css_depth")
                elif character in "}])":
                    if not stack or stack[-1] != matching[character]:
                        _fail_invalid("css_unbalanced_block")
                    opening = stack.pop()
                    if opening == "{":
                        declaration_open.pop()
                elif character == ":" and stack and stack[-1] == "{":
                    if not declaration_open[-1]:
                        declaration_open[-1] = True
                        declaration_count += 1
                    if declaration_count > MAX_CSS_DECLARATIONS:
                        raise _BudgetExceeded("css_declarations")
                elif character == ";" and stack and stack[-1] == "{":
                    declaration_open[-1] = False

            token_count += 1
            if token_count > MAX_CSS_TOKENS:
                raise _BudgetExceeded("css_tokens")
            _check_css_token_size(source, start, index)
    if stack:
        _fail_invalid("css_unbalanced_block")


def _validate_css(styles: list[str]) -> None:
    if tinycss2 is None:
        _fail_unavailable()
    _preflight_css(styles)
    stack: list[tuple[Any, int]] = []
    for stylesheet in styles:
        parse_failed = False
        try:
            nodes = tinycss2.parse_stylesheet(
                stylesheet,
                skip_comments=False,
                skip_whitespace=False,
            )
        except Exception:  # noqa: BLE001 - third-party CSS parser must fail closed
            parse_failed = True
            nodes = []
        if parse_failed:
            _fail_invalid("css_parse_error")
        stack.extend((node, 0) for node in reversed(nodes))

    token_count = 0
    declaration_count = 0
    error_count = 0
    first_error_line: int | None = None
    first_error_column: int | None = None
    while stack:
        node, depth = stack.pop()
        token_count += 1
        if token_count > MAX_CSS_TOKENS:
            raise _BudgetExceeded("css_tokens")
        if depth > MAX_CSS_DEPTH:
            raise _BudgetExceeded("css_depth")
        node_type = getattr(node, "type", "")
        if node_type == "error":
            error_count += 1
            if first_error_line is None:
                first_error_line = getattr(node, "source_line", None)
                first_error_column = getattr(node, "source_column", None)
            if error_count > MAX_CSS_ERRORS:
                raise _BudgetExceeded("css_errors")
            continue
        if node_type == "url":
            _fail_invalid("css_resource")
        if node_type == "function":
            if getattr(node, "lower_name", "") in _CSS_RESOURCE_FUNCTIONS:
                _fail_invalid("css_resource")
            stack.extend((item, depth + 1) for item in reversed(node.arguments))
        elif node_type in {"{} block", "[] block", "() block"}:
            stack.extend((item, depth + 1) for item in reversed(node.content))
        elif node_type == "at-rule":
            if getattr(node, "lower_at_keyword", "") in {
                "import",
                "font-face",
                "namespace",
                "document",
            }:
                _fail_invalid("css_resource")
            stack.extend((item, depth) for item in reversed(node.prelude))
            if node.content is not None:
                parse_failed = False
                try:
                    nested = tinycss2.parse_blocks_contents(
                        node.content,
                        skip_comments=False,
                        skip_whitespace=False,
                    )
                except Exception:  # noqa: BLE001 - nested CSS must fail closed
                    parse_failed = True
                    nested = []
                if parse_failed:
                    _fail_invalid("css_parse_error")
                stack.extend((item, depth + 1) for item in reversed(nested))
        elif node_type == "qualified-rule":
            stack.extend((item, depth) for item in reversed(node.prelude))
            parse_failed = False
            try:
                nested = tinycss2.parse_blocks_contents(
                    node.content,
                    skip_comments=False,
                    skip_whitespace=False,
                )
            except Exception:  # noqa: BLE001 - nested CSS must fail closed
                parse_failed = True
                nested = []
            if parse_failed:
                _fail_invalid("css_parse_error")
            stack.extend((item, depth + 1) for item in reversed(nested))
        elif node_type == "declaration":
            declaration_count += 1
            if declaration_count > MAX_CSS_DECLARATIONS:
                raise _BudgetExceeded("css_declarations")
            stack.extend((item, depth) for item in reversed(node.value))

    if error_count:
        _fail_invalid(
            "css_parse_error",
            line=first_error_line,
            column=first_error_column,
        )


def _iter_element_text(root: Any) -> str:
    chunks: list[str] = []
    stack: list[tuple[Any, bool, bool]] = [(root, False, False)]
    while stack:
        node, exiting, include_tail = stack.pop()
        if exiting:
            if include_tail and node.tail:
                chunks.append(str(node.tail))
            continue
        if node.text:
            chunks.append(str(node.text))
        stack.append((node, True, include_tail))
        children = [child for child in node if isinstance(getattr(child, "tag", None), str)]
        stack.extend((child, False, True) for child in reversed(children))
    return "".join(chunks)


def _normalize_title(title_element: Any) -> str:
    title = unicodedata.normalize("NFC", _collapse_html_whitespace(_iter_element_text(title_element)))
    if not title:
        _fail_invalid("title_blank")
    if any(_is_forbidden_control(character) or character in _BIDI_FORMATTING for character in title):
        _fail_invalid("title_characters")
    if len(title) > 200 or len(title.encode("utf-8")) > 512:
        _fail_invalid("title_length")
    return title


def _semantic_text(slides: list[Any]) -> str:
    chunks: list[str] = []
    output_length = 0
    pending_space = False

    def separate_blocks() -> None:
        nonlocal pending_space
        if output_length:
            pending_space = True

    def append_text(value: str) -> None:
        nonlocal output_length, pending_space
        if output_length >= MAX_INDEXABLE_TEXT:
            return
        normalized: list[str] = []
        for character in value:
            if character in _HTML_WHITESPACE:
                if output_length or normalized:
                    pending_space = True
                continue
            if pending_space:
                if output_length + len(normalized) >= MAX_INDEXABLE_TEXT:
                    break
                normalized.append(" ")
                pending_space = False
            if output_length + len(normalized) >= MAX_INDEXABLE_TEXT:
                break
            normalized.append(character)
        if normalized:
            chunk = "".join(normalized)
            chunks.append(chunk)
            output_length += len(chunk)

    for slide in slides:
        stack: list[tuple[Any, bool, bool, bool]] = [(slide, False, False, False)]
        while stack:
            node, exiting, active, excluded = stack.pop()
            if exiting:
                if active and node.tail:
                    append_text(str(node.tail))
                continue
            is_element = isinstance(getattr(node, "tag", None), str)
            tag = _local_name(node.tag) if is_element else ""
            classes = _class_tokens(node) if is_element else frozenset()
            node_excluded = excluded or (
                is_element and (tag in _EXCLUDED_ELEMENTS or bool(classes & _EXCLUDED_CLASSES))
            )
            if is_element and node is not slide and "slide" in classes:
                node_excluded = True
            starts_semantic = is_element and _namespace(node.tag) == _HTML_NAMESPACE and tag in _SEMANTIC_ELEMENTS
            node_active = not node_excluded and (active or starts_semantic)
            if starts_semantic and not active and not node_excluded:
                separate_blocks()
            if is_element and node_active and node.text:
                append_text(str(node.text))
            stack.append((node, True, active and not excluded, False))
            children = list(node) if is_element else []
            stack.extend((child, False, node_active, node_excluded) for child in reversed(children))
    return "".join(chunks)


def _slide_has_excluded_ancestor(element: Any, parents: dict[int, Any | None]) -> bool:
    ancestor = parents.get(id(element))
    while ancestor is not None:
        tag = _local_name(ancestor.tag)
        classes = _class_tokens(ancestor)
        if tag in _EXCLUDED_ELEMENTS or classes & _EXCLUDED_CLASSES or "slide" in classes:
            return True
        ancestor = parents.get(id(ancestor))
    return False


def _validate_tree(root: Any, delivery_style: DeliveryStyle | None) -> tuple[str, list[Any], list[str]]:
    if delivery_style not in {None, "speaker-led", "self-guided"}:
        _fail_invalid("delivery_style")
    if not _is_html_element(root, "html"):
        _fail_invalid("html_root")

    all_elements: list[tuple[Any, Any | None]] = []
    stack: list[tuple[Any, Any | None]] = [(root, None)]
    while stack:
        element, parent = stack.pop()
        all_elements.append((element, parent))
        children = [child for child in element if isinstance(getattr(child, "tag", None), str)]
        stack.extend((child, element) for child in reversed(children))
    parents = {id(element): parent for element, parent in all_elements}

    heads = [element for element, _ in all_elements if _is_html_element(element, "head")]
    bodies = [element for element, _ in all_elements if _is_html_element(element, "body")]
    titles = [element for element, _ in all_elements if _is_html_element(element, "title")]
    if len(heads) != 1 or len(bodies) != 1 or len(titles) != 1:
        _fail_invalid("html_document_structure")
    title = _normalize_title(titles[0])
    body = bodies[0]

    scripts = [element for element, _ in all_elements if _local_name(element.tag) == "script"]
    if len(scripts) != 1 or not _is_html_element(scripts[0], "script") or scripts[0].attrib:
        _fail_invalid("script_structure")
    body_children = list(body)
    if not body_children or body_children[-1] is not scripts[0]:
        _fail_invalid("script_position")
    if scripts[0].tail and str(scripts[0].tail).strip(_HTML_WHITESPACE):
        _fail_invalid("script_position")
    script_text = _iter_element_text(scripts[0]).lower()
    if _script_has_obvious_sink(script_text):
        _fail_invalid("script_policy")

    slides: list[Any] = []
    total_slide_elements = 0
    styles: list[str] = []
    notes_by_slide: dict[int, int] = {}
    for element, parent in all_elements:
        tag = _local_name(element.tag)
        namespace = _namespace(element.tag)
        classes = _class_tokens(element)
        if namespace != _HTML_NAMESPACE and tag in {"script", "style"}:
            _fail_invalid("html_active_element")
        if tag in _FORBIDDEN_ELEMENTS:
            _fail_invalid("html_active_element")
        if namespace != _HTML_NAMESPACE and tag in _FOREIGN_ACTIVE_ELEMENTS:
            _fail_invalid("html_active_element")
        for attribute_name, attribute_value in element.attrib.items():
            local_attribute = _local_name(attribute_name)
            if local_attribute == "style" or local_attribute.startswith("on"):
                _fail_invalid("html_active_attribute")
            if local_attribute in _URL_ATTRIBUTES:
                _fail_invalid("html_resource_attribute")
            if not _is_namespace_declaration(attribute_name) and _attribute_value_has_unmistakable_url_marker(
                str(attribute_value)
            ):
                _fail_invalid("html_resource_attribute")
            if local_attribute in _CSS_RESOURCE_ATTRIBUTES and _css_value_has_resource(str(attribute_value)):
                _fail_invalid("html_resource_attribute")
        if tag == "meta" and str(element.attrib.get("http-equiv", "")).lower() == "refresh":
            _fail_invalid("html_refresh")
        if "slide" in classes:
            if namespace != _HTML_NAMESPACE or tag != "section":
                _fail_invalid("slide_structure")
            total_slide_elements += 1
            if not classes & _EXCLUDED_CLASSES and not _slide_has_excluded_ancestor(element, parents):
                slides.append(element)
                notes_by_slide[id(element)] = 0
        if "notes" in classes and namespace == _HTML_NAMESPACE:
            if parent is None or not _is_html_element(parent, "section") or "slide" not in _class_tokens(parent):
                _fail_invalid("notes_structure")
            notes_by_slide[id(parent)] = notes_by_slide.get(id(parent), 0) + 1
            if notes_by_slide[id(parent)] > 1:
                _fail_invalid("notes_structure")
        if namespace == _HTML_NAMESPACE and tag == "style":
            styles.append(_iter_element_text(element))

    if total_slide_elements > MAX_SLIDES or not 1 <= len(slides) <= MAX_SLIDES:
        _fail_invalid("slide_count")
    for slide in slides:
        notes_count = notes_by_slide.get(id(slide), 0)
        if delivery_style == "speaker-led" and notes_count != 1:
            _fail_invalid("notes_delivery_style")
        if delivery_style == "self-guided" and notes_count != 0:
            _fail_invalid("notes_delivery_style")
    return title, slides, styles


def validate_standalone_html(
    document: str | bytes,
    *,
    delivery_style: DeliveryStyle | None = None,
) -> StandaloneHtmlValidationResult:
    """Validate inert source and return only frozen derived metadata.

    No source, parser exception, token, or traceback is placed in a public
    failure. The original document is never rewritten.
    """
    if html5lib is None or tinycss2 is None:
        _fail_unavailable()
    expected_bytes = _preflight_document_input(document)
    if isinstance(document, bytes):
        source_bytes = bytes.__getitem__(document, slice(None))
        source = bytes.decode(source_bytes, "utf-8", "strict")
    else:
        source = str.__getitem__(document, slice(None))
        source_bytes = str.encode(source, "utf-8", "strict")
    if bytes.__len__(source_bytes) != expected_bytes:  # pragma: no cover - immutable input invariant
        _fail_invalid("document_encoding")
    if any(_is_forbidden_control(character) for character in source):
        _fail_invalid("document_controls")

    budget_reason: str | None = None
    try:
        _preflight_html(source)
        root = _parse_html(source)
        title, slides, styles = _validate_tree(root, delivery_style)
        _validate_css(styles)
    except _BudgetExceeded as exc:
        budget_reason = exc.reason
    if budget_reason is not None:
        _fail_budget(budget_reason)

    return StandaloneHtmlValidationResult(
        title=title,
        slide_count=len(slides),
        html_bytes=len(source_bytes),
        html_sha256=hashlib.sha256(source_bytes).hexdigest(),
        indexable_text=_semantic_text(slides),
    )


__all__ = [
    "DeliveryStyle",
    "MAX_DOCUMENT_BYTES",
    "validate_standalone_html",
]
