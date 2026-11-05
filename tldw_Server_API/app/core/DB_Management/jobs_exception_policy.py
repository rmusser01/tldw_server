"""Exact optional-driver exception policy shared with the Jobs facade."""

from __future__ import annotations

import json
import sqlite3

from tldw_Server_API.app.core.exceptions import BadRequestError

try:
    import psycopg  # type: ignore

    _PG_ERRORS: tuple[type[BaseException], ...] = (psycopg.Error,)
except ImportError:
    _PG_ERRORS = ()

_JOB_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    BadRequestError,
    ConnectionError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    sqlite3.Error,
    *_PG_ERRORS,
)
