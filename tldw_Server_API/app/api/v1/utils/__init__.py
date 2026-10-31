"""
Shared utility helpers for API v1.

This package hosts reusable concerns for endpoints such as:
- Caching and ETag handling (`cache.py`)
- HTTP error mapping (`http_errors.py`)
- Request parsing and normalization (`request_parsing.py`)

Utilities here are intentionally lightweight and free of FastAPI
router wiring so they can be imported from both endpoints and
core services without circular dependencies.
"""

from __future__ import annotations

__all__ = [
    "cache",
    "deprecation",
    "http_errors",
    "request_parsing",
]
