from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def file_uri_for_path(path: Path, *, directory: bool = False) -> str:
    uri = path.expanduser().resolve().as_uri()
    return f"{uri}/" if directory and not uri.endswith("/") else uri


def source_defaults_metadata(*, keywords: tuple[str, ...], collection_names: tuple[str, ...]) -> dict[str, list[str]]:
    return {
        "default_keywords": list(keywords),
        "default_collections": list(collection_names),
    }


def redacted_url_for_display(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))


def url_has_query(url: str) -> bool:
    return bool(urlsplit(url).query)
