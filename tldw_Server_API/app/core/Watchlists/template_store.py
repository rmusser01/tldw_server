"""
File-based template storage for Watchlists outputs.

Templates are persisted under Config_Files/templates/watchlists (or the directory
specified via WATCHLIST_TEMPLATE_DIR). Each template is stored as <name>.<format>
where format is 'md' or 'html'.

Metadata is stored in a sibling <name>.meta.json file and includes:
- description
- current_version
- history (older template snapshots)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.exceptions import (
    InvalidTemplateFormatError,
    InvalidTemplateNameError,
    InvalidTemplatePathError,
)

_SLUG_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_SUPPORTED_SUFFIXES = {".md", ".html"}
_TEMPLATE_PATH_ERROR = "Template path must stay within the watchlist template directory"
_INVALID_TEMPLATE_NAME_ERROR = "Template name must match ^[A-Za-z0-9_\\-]{1,64}$"
_INVALID_TEMPLATE_FORMAT_ERROR = "Template format must be 'md' or 'html'"


@dataclass
class TemplateRecord:
    name: str
    format: str
    content: str
    description: str | None
    updated_at: str
    version: int = 1
    history_count: int = 0
    available_versions: list[int] | None = None
    composer_ast: dict[str, Any] | None = None
    composer_schema_version: str | None = None
    composer_sync_hash: str | None = None
    composer_sync_status: str | None = None


@dataclass
class TemplateVersionRecord:
    version: int
    format: str
    description: str | None
    updated_at: str
    is_current: bool


class TemplateNotFoundError(FileNotFoundError):
    """Raised when a requested template does not exist."""


class TemplateExistsError(FileExistsError):
    """Raised when creating a template that already exists and overwrite=False."""


class TemplateVersionNotFoundError(LookupError):
    """Raised when a requested template version does not exist."""


def _resolved_dir() -> Path:
    configured = os.getenv("WATCHLIST_TEMPLATE_DIR") or settings.get("WATCHLIST_TEMPLATE_DIR")
    if configured:
        base = Path(configured)
    else:
        base = (
            Path(__file__).resolve().parent.parent.parent.parent / "Config_Files" / "templates" / "watchlists"
        )
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to ensure watchlist template directory {base}: {exc}")
        raise
    return base


def _assert_within_base(path: Path, base: Path) -> None:
    """Validate that a path stays within the base directory.

    Args:
        path: The path to validate (may be relative or absolute).
        base: The base directory that must contain the path.

    Raises:
        InvalidTemplatePathError: If the resolved path escapes the base directory.
    """
    # Ensure that any path derived from user input stays within the base directory.
    # Using only `path.name` prevents directory traversal via subdirectories.
    resolved_base = base.resolve()
    candidate = path.resolve()
    try:
        # Ensure the candidate path is within the resolved base directory
        candidate.relative_to(resolved_base)
    except ValueError as err:
        raise InvalidTemplatePathError(_TEMPLATE_PATH_ERROR) from err


def _template_path(name: str, fmt: str) -> Path:
    """Construct and validate a template file path.

    Args:
        name: Template name (sanitized).
        fmt: Format ("md" or "html").

    Returns:
        Validated path within the template directory.

    Raises:
        InvalidTemplateNameError: If the name is invalid.
        InvalidTemplatePathError: If the path escapes the base directory.
    """
    name = _sanitize_name(name)
    fmt = fmt.lower()
    suffix = ".md" if fmt == "md" else ".html"
    base = _resolved_dir()
    path = base / f"{name}{suffix}"
    _assert_within_base(path, base)
    return path


def _meta_path(name: str) -> Path:
    """Construct and validate a metadata file path.

    Args:
        name: Template name (sanitized).

    Returns:
        Validated path to the .meta.json file within the template directory.

    Raises:
        InvalidTemplateNameError: If the name is invalid.
        InvalidTemplatePathError: If the path escapes the base directory.
    """
    name = _sanitize_name(name)
    base = _resolved_dir()
    path = base / f"{name}.meta.json"
    _assert_within_base(path, base)
    return path


def _load_metadata(meta_file: Path) -> dict[str, Any]:
    if not meta_file.exists():
        return {}
    try:
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
        return {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug(f"Failed to read template metadata from {meta_file}: {exc}")
        return {}


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _load_history(meta: dict[str, Any]) -> list[dict[str, Any]]:
    raw = meta.get("history")
    if not isinstance(raw, list):
        return []
    history: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        version = _safe_int(entry.get("version"), default=0)
        fmt = str(entry.get("format") or "").lower()
        content = entry.get("content")
        if version <= 0 or fmt not in {"md", "html"} or not isinstance(content, str):
            continue
        history.append(
            {
                "version": version,
                "format": fmt,
                "content": content,
                "description": (
                    str(entry.get("description"))
                    if entry.get("description") is not None
                    else None
                ),
                "updated_at": (
                    str(entry.get("updated_at"))
                    if entry.get("updated_at") is not None
                    else datetime.now(timezone.utc).isoformat()
                ),
            }
        )
    history.sort(key=lambda item: item["version"])
    return history


def _save_metadata(
    meta_file: Path,
    *,
    description: str | None,
    current_version: int,
    history: list[dict[str, Any]],
    composer_ast: dict[str, Any] | None = None,
    composer_schema_version: str | None = None,
    composer_sync_hash: str | None = None,
    composer_sync_status: str | None = None,
) -> None:
    payload: dict[str, Any] = {}
    if description is not None:
        payload["description"] = description
    if current_version > 1:
        payload["current_version"] = current_version
    if history:
        payload["history"] = history
    if composer_ast is not None:
        payload["composer_ast"] = composer_ast
    if composer_schema_version is not None:
        payload["composer_schema_version"] = composer_schema_version
    if composer_sync_hash is not None:
        payload["composer_sync_hash"] = composer_sync_hash
    if composer_sync_status is not None:
        payload["composer_sync_status"] = composer_sync_status

    if payload:
        meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif meta_file.exists():
        meta_file.unlink()


def _available_versions(current_version: int, history: list[dict[str, Any]]) -> list[int]:
    versions = {current_version}
    for entry in history:
        version = _safe_int(entry.get("version"), default=0)
        if version > 0:
            versions.add(version)
    return sorted(versions)


def _sanitize_name(name: str) -> str:
    if not _SLUG_RE.fullmatch(name):
        raise InvalidTemplateNameError(_INVALID_TEMPLATE_NAME_ERROR)
    return name


_VALID_COMPOSER_SYNC_STATUSES = {"in_sync", "needs_repair", "recovered_from_code"}


def _load_composer_metadata(meta: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str | None, str | None]:
    composer_ast_raw = meta.get("composer_ast")
    composer_ast = composer_ast_raw if isinstance(composer_ast_raw, dict) else None

    composer_schema_version_raw = meta.get("composer_schema_version")
    composer_schema_version = (
        str(composer_schema_version_raw)
        if composer_schema_version_raw is not None
        else None
    )

    composer_sync_hash_raw = meta.get("composer_sync_hash")
    composer_sync_hash = (
        str(composer_sync_hash_raw)
        if composer_sync_hash_raw is not None
        else None
    )

    composer_sync_status_raw = meta.get("composer_sync_status")
    composer_sync_status = (
        str(composer_sync_status_raw)
        if str(composer_sync_status_raw) in _VALID_COMPOSER_SYNC_STATUSES
        else None
    )

    return composer_ast, composer_schema_version, composer_sync_hash, composer_sync_status


_BUILTIN_BRIEFING_MARKDOWN = """\
# {{ title }}

**Generated:** {{ generated_at }}
**Job:** {{ job_name }} | **Run:** {{ run_id }}

---

{% for item in items %}
## {{ item.title or 'Untitled' }}

{% if item.llm_summary %}{{ item.llm_summary }}{% elif item.summary %}{{ item.summary }}{% endif %}

{% if item.url %}[Read more]({{ item.url }}){% endif %}
{% if item.published_at %}*Published: {{ item.published_at }}*{% endif %}

---

{% endfor %}

*{{ items | length }} item(s) in this briefing.*
"""

_BUILTIN_NEWSLETTER_MARKDOWN = """\
# {{ title }}

> Curated newsletter — {{ generated_at }}

{% for item in items %}
### {{ loop.index }}. {{ item.title or 'Untitled' }}

{% if item.llm_summary %}{{ item.llm_summary }}{% elif item.summary %}{{ item.summary }}{% endif %}

{% if item.url %}[Link]({{ item.url }}){% endif %}

{% endfor %}
"""

_BUILTIN_MECE_MARKDOWN = """\
# {{ title }} — MECE Categorized

**Generated:** {{ generated_at }}

{% set categorized = {} %}
{% for item in items %}
{% set cat = item.tags[0] if item.tags else 'Uncategorized' %}
{% if cat not in categorized %}{% set _ = categorized.update({cat: []}) %}{% endif %}
{% set _ = categorized[cat].append(item) %}
{% endfor %}

{% for category, cat_items in categorized.items() %}
## {{ category }}

{% for item in cat_items %}
- **{{ item.title or 'Untitled' }}** {% if item.url %}([link]({{ item.url }})){% endif %}
  {% if item.llm_summary %}{{ item.llm_summary }}{% elif item.summary %}{{ item.summary }}{% endif %}
{% endfor %}

{% endfor %}
"""

_BUILTIN_NEWSLETTER_HTML = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{{ title }}</title></head>
<body>
<h1>{{ title }}</h1>
<p><em>Generated: {{ generated_at }}</em></p>
<hr>
{% for item in items %}
<h2>{{ item.title or 'Untitled' }}</h2>
{% if item.llm_summary %}<p>{{ item.llm_summary }}</p>{% elif item.summary %}<p>{{ item.summary }}</p>{% endif %}
{% if item.url %}<p><a href="{{ item.url }}">Read more</a></p>{% endif %}
{% if item.published_at %}<p><small>Published: {{ item.published_at }}</small></p>{% endif %}
<hr>
{% endfor %}
<p><em>{{ items | length }} item(s)</em></p>
</body>
</html>
"""

_BUILTIN_CTI_OSINT_REPORT_MARKDOWN = """\
# {{ title }}

**Generated:** {{ generated_at }}
**Job:** {{ job_name }} | **Run:** {{ run_id }}
{% set report_data = report | default({}) %}
{% set readiness = report_data.readiness | default({}) %}
{% set source_summary = report_data.source_summary | default({}) %}
{% set evidence_items = report_data.included_items | default(items) %}
{% set excluded_items = report_data.excluded_items | default([]) %}

## Executive summary

Report readiness: {{ readiness.state | default('unknown') }}{% if readiness.score is defined %} ({{ readiness.score }}/100){% endif %}

- Included evidence: {{ report_data.included_count | default(evidence_items | length) }}
- Alert matches: {{ report_data.alert_count | default(0) }}
- Unique sources: {{ source_summary.unique_source_count | default(0) }}
- Missing provenance: {{ source_summary.missing_source_count | default(0) }}

{% if readiness.warnings | default([]) %}
## Evidence caveats

{% for warning in readiness.warnings %}
- {{ warning.message | default(warning.code | default('Evidence warning')) }}
{% endfor %}
{% endif %}

## Key findings

{% for item in evidence_items %}
### {{ loop.index }}. {{ item.title or 'Untitled' }}

{% if item.summary %}{{ item.summary }}{% elif item.llm_summary %}{{ item.llm_summary }}{% endif %}

- Source: {{ item.source_name | default('Unknown source') }}
{% if item.published_at %}- Published: {{ item.published_at }}{% endif %}
{% if item.url %}- Link: {{ item.url }}{% endif %}

{% if item.alerts | default([]) %}
Alert evidence:
{% for alert in item.alerts %}
- {{ alert.rule_name | default('Alert') }} ({{ alert.severity | default('unknown') }}){% if alert.matched_text %}: {{ alert.matched_text }}{% endif %}
{% endfor %}
{% else %}
Alert evidence: none captured.
{% endif %}

{% endfor %}

## Evidence table

| Update | Source | Published | Alerts | Link |
| --- | --- | --- | --- | --- |
{% for item in evidence_items %}
| {{ item.title or 'Untitled' }} | {{ item.source_name | default('Unknown source') }} | {{ item.published_at | default('-') }} | {{ item.alerts | default([]) | length }} | {{ item.url | default('-') }} |
{% endfor %}

## Excluded trail

{% if excluded_items %}
{% for item in excluded_items %}
- {{ item.title or ('Update #' ~ item.id) }} - {{ item.reason | default('excluded') }}
{% endfor %}
{% else %}
No excluded updates captured.
{% endif %}
"""

_BUILTIN_NEWS_BRIEFING_MARKDOWN = """\
# {{ title }}

**Generated:** {{ generated_at }}
{% set report_data = report | default({}) %}
{% set readiness = report_data.readiness | default({}) %}
{% set source_summary = report_data.source_summary | default({}) %}
{% set evidence_items = report_data.included_items | default(items) %}
{% set excluded_items = report_data.excluded_items | default([]) %}

## What changed

{% for item in evidence_items %}
### {{ loop.index }}. {{ item.title or 'Untitled' }}

{% if item.summary %}{{ item.summary }}{% elif item.llm_summary %}{{ item.llm_summary }}{% endif %}

{% if item.published_at %}Published: {{ item.published_at }}{% endif %}
{% if item.url %}[Follow up]({{ item.url }}){% endif %}

{% endfor %}

## Timeline and recency

- Report readiness: {{ readiness.state | default('unknown') }}{% if readiness.score is defined %} ({{ readiness.score }}/100){% endif %}
- Included updates: {{ report_data.included_count | default(evidence_items | length) }}
- Generated at: {{ generated_at }}

{% if readiness.warnings | default([]) %}
## Evidence caveats

{% for warning in readiness.warnings %}
- {{ warning.message | default(warning.code | default('Evidence warning')) }}
{% endfor %}
{% endif %}

## Source diversity

- Unique sources: {{ source_summary.unique_source_count | default(0) }}
- Missing provenance: {{ source_summary.missing_source_count | default(0) }}

## Follow-up links

{% for item in evidence_items %}
{% if item.url %}- [{{ item.title or 'Untitled' }}]({{ item.url }}){% endif %}
{% endfor %}

## Excluded trail

{% if excluded_items %}
{% for item in excluded_items %}
- {{ item.title or ('Update #' ~ item.id) }} - {{ item.reason | default('excluded') }}
{% endfor %}
{% else %}
No excluded updates captured.
{% endif %}
"""

_BUILTIN_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "briefing_markdown": ("md", _BUILTIN_BRIEFING_MARKDOWN, "Default markdown briefing template"),
    "newsletter_markdown": ("md", _BUILTIN_NEWSLETTER_MARKDOWN, "Default markdown newsletter template"),
    "mece_markdown": ("md", _BUILTIN_MECE_MARKDOWN, "Default MECE categorized briefing template"),
    "newsletter_html": ("html", _BUILTIN_NEWSLETTER_HTML, "Default HTML newsletter template"),
    "cti_osint_report_markdown": (
        "md",
        _BUILTIN_CTI_OSINT_REPORT_MARKDOWN,
        "CTI/OSINT report template with readiness, alert evidence, provenance, and excluded trail",
    ),
    "news_briefing_markdown": (
        "md",
        _BUILTIN_NEWS_BRIEFING_MARKDOWN,
        "News briefing template with recency, source diversity, follow-up links, and evidence caveats",
    ),
}

_defaults_seeded = False


def _seed_defaults() -> None:
    global _defaults_seeded
    if _defaults_seeded:
        return
    _defaults_seeded = True
    directory = _resolved_dir()
    for name, (fmt, content, description) in _BUILTIN_TEMPLATES.items():
        suffix = ".md" if fmt == "md" else ".html"
        path = directory / f"{name}{suffix}"
        if path.exists():
            continue
        try:
            path.write_text(content, encoding="utf-8")
            meta_path = directory / f"{name}.meta.json"
            _save_metadata(meta_path, description=description, current_version=1, history=[])
        except (OSError, PermissionError) as exc:
            logger.warning(f"Failed to seed built-in template {name}: {exc}")


def list_templates() -> list[TemplateRecord]:
    _seed_defaults()
    directory = _resolved_dir()
    records: list[TemplateRecord] = []
    for path in sorted(directory.glob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        name = path.stem
        fmt = path.suffix.lower().lstrip(".")
        meta = _load_metadata(_meta_path(name))
        description = str(meta.get("description")) if meta.get("description") is not None else None
        history = _load_history(meta)
        version = _safe_int(meta.get("current_version"), default=1)
        composer_ast, composer_schema_version, composer_sync_hash, composer_sync_status = _load_composer_metadata(meta)
        available_versions = _available_versions(version, history)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        records.append(
            TemplateRecord(
                name=name,
                format=fmt,
                content="",
                description=description,
                updated_at=updated_at,
                version=version,
                history_count=len(history),
                available_versions=available_versions,
                composer_ast=composer_ast,
                composer_schema_version=composer_schema_version,
                composer_sync_hash=composer_sync_hash,
                composer_sync_status=composer_sync_status,
            )
        )
    return records


def load_template(name: str, *, version: int | None = None) -> TemplateRecord:
    _seed_defaults()
    name = _sanitize_name(name)
    directory = _resolved_dir()
    for suffix in sorted(_SUPPORTED_SUFFIXES):
        candidate = directory / f"{name}{suffix}"
        _assert_within_base(candidate, directory)
        if candidate.exists():
            fmt = suffix.lstrip(".")
            meta = _load_metadata(_meta_path(name))
            history = _load_history(meta)
            current_version = _safe_int(meta.get("current_version"), default=1)
            requested_version = _safe_int(version, default=current_version) if version is not None else current_version
            description = str(meta.get("description")) if meta.get("description") is not None else None
            composer_ast, composer_schema_version, composer_sync_hash, composer_sync_status = _load_composer_metadata(meta)
            available_versions = _available_versions(current_version, history)

            if requested_version == current_version:
                content = candidate.read_text(encoding="utf-8")
                updated_at = datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc).isoformat()
                selected_fmt = fmt
                selected_description = description
            else:
                selected = next((entry for entry in history if entry.get("version") == requested_version), None)
                if not selected:
                    raise TemplateVersionNotFoundError(f"{name}:{requested_version}")
                content = str(selected["content"])
                updated_at = str(selected.get("updated_at") or datetime.now(timezone.utc).isoformat())
                selected_fmt = str(selected.get("format") or fmt)
                selected_description = (
                    str(selected.get("description"))
                    if selected.get("description") is not None
                    else None
                )

            return TemplateRecord(
                name=name,
                format=selected_fmt,
                content=content,
                description=selected_description,
                updated_at=updated_at,
                version=requested_version,
                history_count=len(history),
                available_versions=available_versions,
                composer_ast=composer_ast,
                composer_schema_version=composer_schema_version,
                composer_sync_hash=composer_sync_hash,
                composer_sync_status=composer_sync_status,
            )
    raise TemplateNotFoundError(name)


def save_template(
    name: str,
    fmt: str,
    content: str,
    *,
    description: str | None = None,
    overwrite: bool = False,
    composer_ast: dict[str, Any] | None = None,
    composer_schema_version: str | None = None,
    composer_sync_hash: str | None = None,
    composer_sync_status: str | None = None,
) -> TemplateRecord:
    name = _sanitize_name(name)
    fmt = fmt.lower()
    if fmt not in {"md", "html"}:
        raise InvalidTemplateFormatError(_INVALID_TEMPLATE_FORMAT_ERROR)
    path = _template_path(name, fmt)
    directory = path.parent
    meta_path = _meta_path(name)

    # Determine if any variant exists
    existing_variants = []
    for suffix in sorted(_SUPPORTED_SUFFIXES):
        candidate = directory / f"{name}{suffix}"
        _assert_within_base(candidate, directory)
        if candidate.exists():
            existing_variants.append(candidate)
    if existing_variants and not overwrite:
        raise TemplateExistsError(name)

    existing_variants.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    existing_meta = _load_metadata(meta_path)
    existing_history = _load_history(existing_meta)
    current_version = _safe_int(existing_meta.get("current_version"), default=1)
    (
        existing_composer_ast,
        existing_composer_schema_version,
        existing_composer_sync_hash,
        existing_composer_sync_status,
    ) = _load_composer_metadata(existing_meta)

    composer_metadata_omitted = (
        composer_ast is None
        and composer_schema_version is None
        and composer_sync_hash is None
        and composer_sync_status is None
    )
    if composer_metadata_omitted:
        had_existing_composer_metadata = any(
            value is not None
            for value in (
                existing_composer_ast,
                existing_composer_schema_version,
                existing_composer_sync_hash,
                existing_composer_sync_status,
            )
        )
        effective_composer_ast = None
        effective_composer_schema_version = None
        effective_composer_sync_hash = None
        effective_composer_sync_status = "needs_repair" if had_existing_composer_metadata else None
    else:
        effective_composer_ast = composer_ast if composer_ast is not None else existing_composer_ast
        effective_composer_schema_version = (
            composer_schema_version
            if composer_schema_version is not None
            else existing_composer_schema_version
        )
        effective_composer_sync_hash = (
            composer_sync_hash
            if composer_sync_hash is not None
            else existing_composer_sync_hash
        )
        effective_composer_sync_status = (
            composer_sync_status
            if composer_sync_status in _VALID_COMPOSER_SYNC_STATUSES
            else existing_composer_sync_status
        )

    if existing_variants:
        current_file = existing_variants[0]
        previous_updated_at = datetime.fromtimestamp(current_file.stat().st_mtime, timezone.utc).isoformat()
        existing_history.append(
            {
                "version": current_version,
                "format": current_file.suffix.lower().lstrip("."),
                "content": current_file.read_text(encoding="utf-8"),
                "description": (
                    str(existing_meta.get("description"))
                    if existing_meta.get("description") is not None
                    else None
                ),
                "updated_at": previous_updated_at,
            }
        )
        # Keep one snapshot per version for deterministic lookup.
        deduped: dict[int, dict[str, Any]] = {}
        for entry in existing_history:
            version_key = _safe_int(entry.get("version"), default=0)
            if version_key > 0:
                deduped[version_key] = entry
        existing_history = [
            entry for version_key, entry in sorted(deduped.items(), key=lambda item: item[0]) if version_key > 0
        ]
        next_version = current_version + 1
    else:
        next_version = 1

    # Clean up other variants when overwriting
    for other in existing_variants:
        if other != path:
            other.unlink()

    path.write_text(content, encoding="utf-8")
    _save_metadata(
        meta_path,
        description=description,
        current_version=next_version,
        history=existing_history,
        composer_ast=effective_composer_ast,
        composer_schema_version=effective_composer_schema_version,
        composer_sync_hash=effective_composer_sync_hash,
        composer_sync_status=effective_composer_sync_status,
    )
    return load_template(name)


def list_template_versions(name: str) -> list[TemplateVersionRecord]:
    record = load_template(name)
    meta = _load_metadata(_meta_path(name))
    history = _load_history(meta)
    version_map: dict[int, TemplateVersionRecord] = {}

    for entry in history:
        entry_version = _safe_int(entry.get("version"), default=0)
        if entry_version <= 0:
            continue
        version_map[entry_version] = TemplateVersionRecord(
            version=entry_version,
            format=str(entry.get("format") or record.format),
            description=(
                str(entry.get("description"))
                if entry.get("description") is not None
                else None
            ),
            updated_at=str(entry.get("updated_at") or record.updated_at),
            is_current=False,
        )

    version_map[record.version] = TemplateVersionRecord(
        version=record.version,
        format=record.format,
        description=record.description,
        updated_at=record.updated_at,
        is_current=True,
    )

    return [version_map[key] for key in sorted(version_map.keys())]


def delete_template(name: str) -> None:
    name = _sanitize_name(name)
    directory = _resolved_dir()
    removed = False
    for suffix in sorted(_SUPPORTED_SUFFIXES):
        candidate = directory / f"{name}{suffix}"
        _assert_within_base(candidate, directory)
        if candidate.exists():
            candidate.unlink()
            removed = True
    meta = _meta_path(name)
    if meta.exists():
        meta.unlink()
    if not removed:
        raise TemplateNotFoundError(name)
