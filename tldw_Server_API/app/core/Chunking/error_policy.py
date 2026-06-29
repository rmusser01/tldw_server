"""Shared exception policy for noncritical chunking fallback paths."""

from __future__ import annotations

import json

from .exceptions import ChunkingError, InvalidChunkingMethodError, InvalidInputError

CHUNKER_NONCRITICAL_EXCEPTIONS = (
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
    json.JSONDecodeError,
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
)
