from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, cast

from .models import AccessScope

SourceProfile = Literal["locked_down", "local_first", "online_capable"]
StalePolicy = Literal["report", "tombstone"]
DiscoveryApplyDefault = Literal["register", "ingest", "register_and_ingest"]

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}
_SOURCE_PROFILES = ("locked_down", "local_first", "online_capable")
_DEFAULT_URL_USER_AGENT = "tldw-mcp-docs/0.1"


def _coerce_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        raise ValueError(f"{field_name} must be a recognized boolean string")
    raise ValueError(f"{field_name} must be a boolean or recognized boolean string")


def _coerce_trusted_roots(value: object) -> tuple[Path, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, (str, PathLike)):
        items = (value,)
    else:
        items = tuple(value) if isinstance(value, Iterable) else (value,)
    return tuple(Path(item).expanduser().resolve() for item in items)


def _coerce_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be a positive integer")
        result = int(value)
    elif isinstance(value, int):
        result = value
    elif isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a positive integer") from exc
    else:
        raise ValueError(f"{field_name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _coerce_positive_float(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _coerce_non_empty_string(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items = (value,)
    else:
        items = tuple(value) if isinstance(value, Iterable) else (value,)
    return tuple(str(item).strip() for item in items if str(item).strip())


def _coerce_optional_scope_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_access_scope(value: object) -> AccessScope:
    if value is None or value == "":
        return AccessScope()
    if isinstance(value, AccessScope):
        return value
    if isinstance(value, Mapping):
        return AccessScope(
            owner_scope=_coerce_optional_scope_value(value.get("owner_scope")),
            profile_scope=_coerce_optional_scope_value(value.get("profile_scope")),
        )
    raise ValueError("default_scope must be a mapping or AccessScope")


def _coerce_source_profile(value: object, field_name: str) -> SourceProfile:
    if value not in _SOURCE_PROFILES:
        raise ValueError(f"{field_name} must be one of: {', '.join(_SOURCE_PROFILES)}")
    return cast(SourceProfile, value)


def _coerce_stale_policy(value: object, field_name: str) -> StalePolicy:
    text = "report" if value is None else str(value).strip().lower()
    if text not in {"report", "tombstone"}:
        raise ValueError(f"{field_name} must be report or tombstone")
    return cast(StalePolicy, text)


def _coerce_discovery_apply_default(value: object, field_name: str) -> DiscoveryApplyDefault:
    """Coerce the default discovery apply action from config."""

    text = "register" if value is None else str(value).strip().lower()
    if text not in {"register", "ingest", "register_and_ingest"}:
        raise ValueError(f"{field_name} must be register, ingest, or register_and_ingest")
    return cast(DiscoveryApplyDefault, text)


@dataclass(frozen=True)
class DocsSettings:
    db_path: Path
    trusted_roots: tuple[Path, ...] = ()
    max_import_file_bytes: int = 2_000_000
    default_scope: AccessScope = AccessScope()
    enable_web_acquisition: bool = False
    web_source_profile: SourceProfile = "locked_down"
    preapproved_domains: tuple[str, ...] = ()
    allowed_url_prefixes: tuple[str, ...] = ()
    denied_domains: tuple[str, ...] = ()
    max_url_redirects: int = 3
    max_url_body_bytes: int = 2_000_000
    url_request_timeout_seconds: float = 10.0
    allowed_content_types: tuple[str, ...] = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")
    url_user_agent: str = _DEFAULT_URL_USER_AGENT
    respect_robots: bool = False
    allow_arbitrary_public_domains: bool = False
    enable_source_sync: bool = True
    max_sync_documents: int = 500
    max_sync_pages: int = 25
    max_sync_run_items: int = 500
    default_stale_policy: StalePolicy = "report"
    sitemap_sync_enabled: bool = False
    persist_url_query_strings: bool = False
    enable_source_discovery: bool = False
    max_discovery_pages: int = 25
    max_discovery_depth: int = 1
    max_discovery_sitemaps: int = 3
    discovery_apply_default: DiscoveryApplyDefault = "register"
    discovery_same_origin_only: bool = True

    @classmethod
    def from_mapping(cls, values: dict) -> "DocsSettings":
        roots = _coerce_trusted_roots(values.get("trusted_roots"))
        return cls(
            db_path=Path(values.get("db_path", "Databases/mcp_docs.db")).expanduser(),
            trusted_roots=roots,
            default_scope=_coerce_access_scope(values.get("default_scope")),
            max_import_file_bytes=_coerce_positive_int(
                values.get("max_import_file_bytes", 2_000_000),
                "max_import_file_bytes",
            ),
            enable_web_acquisition=_coerce_bool(
                values.get("enable_web_acquisition", False),
                "enable_web_acquisition",
            ),
            web_source_profile=_coerce_source_profile(
                values.get("web_source_profile", "locked_down"),
                "web_source_profile",
            ),
            preapproved_domains=_coerce_string_tuple(values.get("preapproved_domains")),
            allowed_url_prefixes=_coerce_string_tuple(values.get("allowed_url_prefixes")),
            denied_domains=_coerce_string_tuple(values.get("denied_domains")),
            max_url_redirects=_coerce_positive_int(
                values.get("max_url_redirects", 3),
                "max_url_redirects",
            ),
            max_url_body_bytes=_coerce_positive_int(
                values.get("max_url_body_bytes", 2_000_000),
                "max_url_body_bytes",
            ),
            url_request_timeout_seconds=_coerce_positive_float(
                values.get("url_request_timeout_seconds", 10.0),
                "url_request_timeout_seconds",
            ),
            allowed_content_types=_coerce_string_tuple(
                values.get(
                    "allowed_content_types",
                    ("text/html", "application/xhtml+xml", "text/plain", "text/markdown"),
                )
            ),
            url_user_agent=_coerce_non_empty_string(
                values.get("url_user_agent"),
                _DEFAULT_URL_USER_AGENT,
            ),
            respect_robots=_coerce_bool(values.get("respect_robots", False), "respect_robots"),
            allow_arbitrary_public_domains=_coerce_bool(
                values.get("allow_arbitrary_public_domains", False),
                "allow_arbitrary_public_domains",
            ),
            enable_source_sync=_coerce_bool(values.get("enable_source_sync", True), "enable_source_sync"),
            max_sync_documents=_coerce_positive_int(
                values.get("max_sync_documents", 500),
                "max_sync_documents",
            ),
            max_sync_pages=_coerce_positive_int(
                values.get("max_sync_pages", 25),
                "max_sync_pages",
            ),
            max_sync_run_items=_coerce_positive_int(
                values.get("max_sync_run_items", 500),
                "max_sync_run_items",
            ),
            default_stale_policy=_coerce_stale_policy(
                values.get("default_stale_policy", "report"),
                "default_stale_policy",
            ),
            sitemap_sync_enabled=_coerce_bool(
                values.get("sitemap_sync_enabled", False),
                "sitemap_sync_enabled",
            ),
            persist_url_query_strings=_coerce_bool(
                values.get("persist_url_query_strings", False),
                "persist_url_query_strings",
            ),
            enable_source_discovery=_coerce_bool(
                values.get("enable_source_discovery", False),
                "enable_source_discovery",
            ),
            max_discovery_pages=_coerce_positive_int(
                values.get("max_discovery_pages", 25),
                "max_discovery_pages",
            ),
            max_discovery_depth=_coerce_positive_int(
                values.get("max_discovery_depth", 1),
                "max_discovery_depth",
            ),
            max_discovery_sitemaps=_coerce_positive_int(
                values.get("max_discovery_sitemaps", 3),
                "max_discovery_sitemaps",
            ),
            discovery_apply_default=_coerce_discovery_apply_default(
                values.get("discovery_apply_default", "register"),
                "discovery_apply_default",
            ),
            discovery_same_origin_only=_coerce_bool(
                values.get("discovery_same_origin_only", True),
                "discovery_same_origin_only",
            ),
        )
