from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from tldw_Server_API.app.core.Chunking.constants import FRONTMATTER_SENTINEL_KEY
from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.process_text.models import PreparedText


def prepare_frontmatter(
    text: str,
    options: dict[str, Any] | None,
    *,
    tokenizer_name_or_path: str | None,
) -> PreparedText:
    prepared, frontmatter_enabled, sentinel_key = _prepare_frontmatter_options(
        text,
        options,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    return _parse_frontmatter(
        prepared,
        frontmatter_enabled=frontmatter_enabled,
        sentinel_key=sentinel_key,
    )


def _prepare_frontmatter_options(
    text: str,
    options: dict[str, Any] | None,
    *,
    tokenizer_name_or_path: str | None,
) -> tuple[PreparedText, bool, str]:
    opts = dict(options or {})
    if tokenizer_name_or_path and "tokenizer_name_or_path" not in opts and "tokenizer_name" not in opts:
        opts["tokenizer_name_or_path"] = tokenizer_name_or_path

    frontmatter_enabled_opt = opts.pop("enable_frontmatter_parsing", None)
    frontmatter_enabled = True if frontmatter_enabled_opt is None else bool(frontmatter_enabled_opt)
    sentinel_key_raw = opts.pop("frontmatter_sentinel_key", FRONTMATTER_SENTINEL_KEY)
    sentinel_key = str(sentinel_key_raw or FRONTMATTER_SENTINEL_KEY)

    return (
        PreparedText(
            original_text=text,
            processed_text=text,
            prefix_offset=0,
            json_meta={},
            header_text="",
            options=opts,
        ),
        frontmatter_enabled,
        sentinel_key,
    )


def _parse_frontmatter(
    prepared: PreparedText,
    *,
    frontmatter_enabled: bool,
    sentinel_key: str,
) -> PreparedText:
    json_meta: dict[str, Any] = {}
    processed_text = prepared.processed_text
    prefix_offset = prepared.prefix_offset
    if frontmatter_enabled:
        try:
            stripped = processed_text.lstrip()
            if stripped.startswith("{"):
                decoder = json.JSONDecoder()
                try:
                    parsed_candidate, end_idx = decoder.raw_decode(stripped)
                except ValueError:
                    parsed_candidate = None
                    end_idx = 0
                if (
                    isinstance(parsed_candidate, dict)
                    and len(stripped[:end_idx]) <= 1_000_000
                    and sentinel_key in parsed_candidate
                    and bool(parsed_candidate.get(sentinel_key))
                ):
                    json_meta = {k: v for k, v in parsed_candidate.items() if k != sentinel_key}
                    leading_ws = len(processed_text) - len(stripped)
                    tail = stripped[end_idx:]
                    tail_trimmed = tail.lstrip("\n\r")
                    prefix_offset += leading_ws + end_idx + (len(tail) - len(tail_trimmed))
                    processed_text = tail_trimmed
        except CHUNKER_NONCRITICAL_EXCEPTIONS:
            pass

    return replace(
        prepared,
        processed_text=processed_text,
        prefix_offset=prefix_offset,
        json_meta=json_meta,
    )


def extract_header(prepared: PreparedText) -> PreparedText:
    processed_text = prepared.processed_text
    prefix_offset = prepared.prefix_offset
    header_text = prepared.header_text
    try:
        header_re = re.compile(
            r"^ (This[ ]text[ ]was[ ]transcribed[ ]using (?:[^\n]*\n)*?\n) ",
            re.MULTILINE | re.VERBOSE,
        )
        m = header_re.match(processed_text)
        if m:
            header_text = m.group(1)
            tail = processed_text[len(header_text):]
            tail_trimmed = tail.lstrip()
            prefix_offset += len(header_text) + (len(tail) - len(tail_trimmed))
            processed_text = tail_trimmed
    except CHUNKER_NONCRITICAL_EXCEPTIONS:
        pass

    return replace(
        prepared,
        processed_text=processed_text,
        prefix_offset=prefix_offset,
        header_text=header_text,
    )
