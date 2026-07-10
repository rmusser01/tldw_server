"""Common utilities shared across workflow adapters.

This module contains helper functions for file path handling, artifact management,
OpenAI response parsing, MCP policy extraction, and other shared functionality.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import resolve_user_id_value
from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_explicit_pytest_runtime,
    is_test_mode,
)

_WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)


def extract_openai_content(response: Any) -> str | None:
    """Extract text content from an OpenAI-style response.

    Args:
        response: OpenAI API response (dict or string)

    Returns:
        Extracted text content or None
    """
    if isinstance(response, dict):
        try:
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = response.get("content") or response.get("text")
            if isinstance(text, str):
                return text
        except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
            return None
    if isinstance(response, str):
        return response
    return None


def sanitize_path_component(value: str, default: str, max_len: int = 80) -> str:
    """Normalize a string for safe use as a single filesystem path component.

    Args:
        value: Raw input to sanitize
        default: Fallback value when the input normalizes to empty
        max_len: Maximum length of the returned component

    Returns:
        A sanitized component containing only ASCII letters, digits, dot,
        underscore, or dash.

    Security:
        Replaces any other character with "_" and strips leading/trailing
        dot/underscore/dash to reduce traversal-like components.
    """
    raw = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not cleaned:
        cleaned = default
    return cleaned[:max_len]


def is_subpath(parent: Path, child: Path) -> bool:
    """Return True if 'child' is located within 'parent' (after resolving both).

    This is a compatibility-safe equivalent of Path.is_relative_to.
    """
    try:
        parent_resolved = parent.resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        logger.debug("Failed to resolve workflow parent path")
        parent_resolved = parent
    try:
        child_resolved = child.resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        logger.debug("Failed to resolve workflow child path")
        child_resolved = child
    try:
        child_resolved.relative_to(parent_resolved)
        return True
    except ValueError:
        return False


def resolve_context_user_id(context: dict[str, Any]) -> str | None:
    """Resolve user ID from workflow context.

    Args:
        context: Workflow context dict

    Returns:
        Resolved user ID or None
    """
    raw = context.get("user_id") or context.get("inputs", {}).get("user_id")
    return resolve_user_id_value(raw, allow_none=True)


def watchlist_artifact_metadata(context: dict[str, Any]) -> dict[str, Any]:
    """Return Watchlists correlation metadata safe to copy onto artifacts."""
    metadata = context.get("workflow_metadata")
    if not isinstance(metadata, dict):
        return {}
    if metadata.get("source") != "watchlist_audio_briefing":
        return {}
    return {
        key: metadata[key]
        for key in (
            "source",
            "watchlist_job_id",
            "watchlist_run_id",
            "briefing_occurrence_id",
            "briefing_attempt_id",
            "watchlist_output_id",
            "audio_request_id",
        )
        if metadata.get(key) is not None
    }


_SENSITIVE_URL_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "code",
    "credential",
    "id_token",
    "jwt",
    "key_pair_id",
    "password",
    "policy",
    "refresh_token",
    "secret",
    "signature",
    "sig",
    "session_token",
    "token",
    "x_amz_credential",
    "x_amz_security_token",
    "x_amz_signature",
}
_SENSITIVE_URL_QUERY_KEY_TERMS = {"auth", "credential", "password", "secret", "signature", "token"}
_URL_VALUE_QUERY_KEYS = {
    "callback",
    "continue",
    "destination",
    "next",
    "redirect",
    "redirect_uri",
    "redirect_url",
    "return",
    "return_to",
    "return_url",
    "target",
    "url",
}
_URL_INSPECTION_MAX_CHARS = 2048
_URL_INSPECTION_MAX_DEPTH = 3
_URL_INSPECTION_MAX_QUERY_FIELDS = 100
_PRIVATE_LOCATION_RE = re.compile(
    r"(?i)(?:file:/{0,2}|(?:^|\s)[a-z]:[\\/]|\\\\|(?:^|\s)/(?:[^\s]+)|(?:^|\s)~[\\/])"
)
_TRAVERSAL_RE = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")
_PROGRAM_FORMATS = {
    "concise_briefing",
    "solo_update",
    "host_discussion",
    "sportscast",
    "culture_roundtable",
    "custom",
}
_PROGRAM_OUTCOME_NOUNS = {"briefing", "episode"}
_SPEAKER_MARKER_MAX_CHARS = 64


def canonical_speaker_markers(values: list[Any]) -> list[str]:
    """Return unique parser-safe ASCII speaker markers in input order."""
    markers: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(values, 1):
        ascii_value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
        base = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_").upper()
        base = base[:_SPEAKER_MARKER_MAX_CHARS] or f"SPEAKER_{position}"
        marker = base
        suffix = 2
        while marker in seen:
            suffix_text = f"_{suffix}"
            marker = f"{base[: _SPEAKER_MARKER_MAX_CHARS - len(suffix_text)]}{suffix_text}"
            suffix += 1
        seen.add(marker)
        markers.append(marker)
    return markers


def _normalized_query_key(value: Any) -> str:
    decoded = _fully_unquote(str(value or ""))
    snake_case = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", decoded)
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_case)
    return re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")


def _is_sensitive_query_key(key: str) -> bool:
    if key in _SENSITIVE_URL_QUERY_KEYS or key.startswith("x_amz_") or key.startswith("x_goog_"):
        return True
    padded = f"_{key}_"
    return any(
        key == term or key.endswith(f"_{term}") or key.startswith(f"{term}_") or f"_{term}_" in padded
        for term in _SENSITIVE_URL_QUERY_KEY_TERMS
    )


def _fully_unquote(value: str) -> str:
    decoded = value
    for _ in range(_URL_INSPECTION_MAX_DEPTH):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def _url_contains_sensitive_data(value: str, *, depth: int = 0) -> bool:
    """Inspect nested URL-valued query parameters for credentials and signatures."""
    if depth > _URL_INSPECTION_MAX_DEPTH or len(value) > _URL_INSPECTION_MAX_CHARS:
        return True
    decoded = value.strip() if depth == 0 else _fully_unquote(value.strip())
    if any(ord(char) < 32 for char in decoded):
        return True
    parse_value = f"https:{decoded}" if decoded.startswith("//") else decoded
    try:
        parsed = urlsplit(parse_value)
        _ = parsed.port
        query = parse_qsl(
            parsed.query.replace(";", "&"),
            keep_blank_values=True,
            max_num_fields=_URL_INSPECTION_MAX_QUERY_FIELDS,
        )
    except (ValueError, UnicodeError):
        return True
    if parsed.username or parsed.password or parsed.fragment:
        return True
    for raw_key, raw_value in query:
        key = _normalized_query_key(raw_key)
        if _is_sensitive_query_key(key):
            return True
        nested = _fully_unquote(raw_value.strip())
        looks_url_like = nested.casefold().startswith(("http://", "https://", "//", "/"))
        if (key in _URL_VALUE_QUERY_KEYS or looks_url_like) and nested:
            if depth >= _URL_INSPECTION_MAX_DEPTH or _url_contains_sensitive_data(nested, depth=depth + 1):
                return True
    return False


def safe_public_source_url(value: Any) -> str:
    """Return a bounded public HTTP(S) URL with no credentials or signatures."""
    raw = str(value or "").strip()
    if not raw or len(raw) > _URL_INSPECTION_MAX_CHARS or any(ord(char) < 32 for char in raw):
        return ""
    try:
        parsed = urlsplit(raw)
        _ = parsed.port
    except ValueError:
        return ""
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        return ""
    if _url_contains_sensitive_data(raw):
        return ""
    return raw


def _safe_public_text(value: Any, *, max_chars: int = 500) -> str:
    """Return compact public text while rejecting path and URI-shaped values."""
    text = " ".join(str(value or "").split()).strip()
    if not text or _PRIVATE_LOCATION_RE.search(text) or _TRAVERSAL_RE.search(text):
        return ""
    return text[:max_chars]


def _public_non_negative_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _public_non_negative_float(value: Any) -> float | None:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def public_program_artifact_metadata(value: Any, *, speech_ready: bool) -> dict[str, Any]:
    """Project program metadata through an explicit public allowlist."""
    if not isinstance(value, Mapping) or not value:
        return {}
    if not any(
        key in value
        for key in (
            "program_format",
            "outcome_noun",
            "show_name",
            "premise",
            "audience",
            "tone",
            "episode_title",
            "show_notes",
            "source_ids",
            "source_urls",
            "cast",
            "is_no_material_update",
        )
    ):
        return {}
    metadata: dict[str, Any] = {}

    program_format = str(value.get("program_format") or "")
    if program_format in _PROGRAM_FORMATS:
        metadata["program_format"] = program_format
    outcome_noun = str(value.get("outcome_noun") or "")
    if outcome_noun in _PROGRAM_OUTCOME_NOUNS:
        metadata["outcome_noun"] = outcome_noun

    for key in ("show_name", "premise", "audience", "tone", "episode_title"):
        text = _safe_public_text(value.get(key))
        if text:
            metadata[key] = text
    for key in ("analysis_allowed", "target_duration_guaranteed", "is_no_material_update"):
        if key in value:
            metadata[key] = bool(value.get(key))
    for key in ("source_count", "candidate_count", "included_count", "omitted_count"):
        number = _public_non_negative_int(value.get(key))
        if number is not None:
            metadata[key] = number
    for key in ("target_duration_minutes", "estimated_duration_minutes"):
        number = _public_non_negative_float(value.get(key))
        if number is not None:
            metadata[key] = number

    raw_source_ids = value.get("source_ids")
    if isinstance(raw_source_ids, list):
        source_ids: list[Any] = []
        for source_id in raw_source_ids[:5000]:
            safe_id = source_id if isinstance(source_id, (int, float)) else _safe_public_text(source_id, max_chars=100)
            if safe_id not in (None, "") and safe_id not in source_ids:
                source_ids.append(safe_id)
        metadata["source_ids"] = source_ids

    raw_source_urls = value.get("source_urls")
    if isinstance(raw_source_urls, list):
        source_urls: list[str] = []
        for raw_url in raw_source_urls[:5000]:
            url = safe_public_source_url(raw_url)
            if url and url not in source_urls:
                source_urls.append(url)
        metadata["source_urls"] = source_urls

    raw_show_notes = value.get("show_notes")
    raw_sources = raw_show_notes.get("sources") if isinstance(raw_show_notes, Mapping) else []
    sources: list[dict[str, Any]] = []
    if isinstance(raw_sources, list):
        for raw_source in raw_sources[:5000]:
            if not isinstance(raw_source, Mapping):
                continue
            source: dict[str, Any] = {}
            for key in ("item_id", "source_id"):
                raw_id = raw_source.get(key)
                safe_id = raw_id if isinstance(raw_id, (int, float)) else _safe_public_text(raw_id, max_chars=100)
                if safe_id not in (None, ""):
                    source[key] = safe_id
            for key in ("title", "published_at"):
                text = _safe_public_text(raw_source.get(key))
                if text:
                    source[key] = text
            url = safe_public_source_url(raw_source.get("url"))
            if url:
                source["url"] = url
            if source:
                sources.append(source)
    metadata["show_notes"] = {
        "sources": sources,
        "source_count": len(sources),
        "speech_disclosure": (
            "Synthetic AI-generated speech" if speech_ready else "Synthetic speech generation pending"
        ),
    }

    raw_cast = value.get("cast")
    cast: list[dict[str, str]] = []
    if isinstance(raw_cast, list):
        for raw_speaker in raw_cast[:4]:
            if not isinstance(raw_speaker, Mapping):
                continue
            speaker = {
                key: text
                for key in ("label", "role", "synthetic_voice")
                if (text := _safe_public_text(raw_speaker.get(key), max_chars=200))
            }
            if speaker:
                cast.append(speaker)
    metadata["cast"] = cast
    metadata["ai_generated_speech"] = speech_ready
    metadata["speech_disclosure"] = (
        "Synthetic AI-generated speech" if speech_ready else "Synthetic speech generation pending"
    )
    return metadata


def artifacts_base_dir() -> Path:
    """Resolve the base directory used for workflow artifacts.

    Returns:
        Absolute artifacts base when project root is available, otherwise
        a relative `Databases/artifacts` path.

    Security:
        Prefers anchoring to the project root to avoid CWD-dependent behavior.
        In test mode, uses the current working directory to keep fixtures
        isolated.
    """
    env_override = os.getenv("WORKFLOWS_ARTIFACTS_DIR") or os.getenv("WORKFLOWS_ARTIFACT_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    try:
        if is_explicit_pytest_runtime() or is_test_mode():
            return (Path.cwd() / "Databases" / "artifacts").resolve()
    except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
        logger.exception("Error checking TEST_MODE/PYTEST_CURRENT_TEST for artifacts base dir")
    try:
        from tldw_Server_API.app.core.Utils.Utils import get_project_root
        return (Path(get_project_root()) / "Databases" / "artifacts").resolve()
    except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
        logger.exception("Error getting project root for artifacts base dir")
        return Path("Databases") / "artifacts"


def resolve_artifacts_dir(step_run_id: str | None) -> Path:
    """Build a per-step artifact directory path under the artifacts base.

    Args:
        step_run_id: Optional step run identifier used as a folder name

    Returns:
        A resolved candidate artifact directory path.

    Security:
        Uses `sanitize_path_component` to limit characters and length.
    """
    base_dir = artifacts_base_dir()
    try:
        base_resolved = base_dir.resolve(strict=False)
    except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
        logger.debug("Artifacts base dir resolve failed. Using unresolved base dir.")
        base_resolved = base_dir
    safe_id = sanitize_path_component(step_run_id or "", f"artifact_{int(time.time() * 1000)}")
    safe_id = Path(safe_id).name or f"artifact_{int(time.time() * 1000)}"
    candidate = (base_resolved / safe_id).resolve(strict=False)
    if not is_subpath(base_resolved, candidate):
        fallback_id = f"artifact_{int(time.time() * 1000)}"
        fallback_id = Path(fallback_id).name
        candidate = (base_resolved / fallback_id).resolve(strict=False)
        if not is_subpath(base_resolved, candidate):
            raise AdapterError("artifact_dir_resolution_failed")
    return candidate


def resolve_artifact_filename(name: str, ext: str, default_stem: str = "artifact") -> str:
    """Produce a safe artifact filename with a fixed extension.

    Args:
        name: Original filename input, possibly containing paths
        ext: Extension to append (without leading dot)
        default_stem: Fallback stem when the name is empty or unsafe

    Returns:
        Sanitized filename with the requested extension.
    """
    raw_name = Path(name).name
    if raw_name in {"", ".", ".."}:
        raw_name = default_stem
    stem = Path(raw_name).stem or default_stem
    safe_stem = sanitize_path_component(stem, default_stem)
    return f"{safe_stem}.{ext}"


def unsafe_file_access_allowed(config: dict[str, Any] | None) -> bool:  # noqa: ARG001
    """Determine whether unsafe file access is explicitly enabled.

    Args:
        config: Ignored on purpose to prevent user-supplied overrides

    Returns:
        True when the server environment enables unsafe access.

    Security:
        Only honors the `WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS` environment
        variable so workflow configs cannot bypass path restrictions.
    """
    return env_flag_enabled("WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS")


def parse_workflows_file_allowlist(raw: str | None) -> list[str]:
    """Parse the allowlist env var into a list of non-empty path strings."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def resolve_workflows_file_allowlist_paths(paths: list[str]) -> list[Path]:
    """Resolve allowlist entries into absolute Paths anchored to the project root."""
    if not paths:
        return []
    project_root = None
    try:
        from tldw_Server_API.app.core.Utils.Utils import get_project_root
        project_root = Path(get_project_root())
    except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
        logger.debug("Workflow file allowlist: failed to resolve project root")
    resolved: list[Path] = []
    for raw in paths:
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                if project_root is not None:
                    candidate = (project_root / candidate).resolve(strict=False)
                else:
                    candidate = candidate.resolve(strict=False)
            else:
                candidate = candidate.resolve(strict=False)
            resolved.append(candidate)
        except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
            logger.debug("Workflow file allowlist: invalid path skipped")
    return resolved


def workflow_file_allowlist(context: dict[str, Any]) -> list[Path]:
    """Return the resolved allowlist for the current tenant, if configured."""
    tenant_id = str(context.get("tenant_id") or "default") if isinstance(context, dict) else "default"
    tenant_key = f"WORKFLOWS_FILE_ALLOWLIST_{tenant_id.upper().replace('-', '_')}"
    raw = os.environ.get(tenant_key) if tenant_key in os.environ else os.getenv("WORKFLOWS_FILE_ALLOWLIST")
    return resolve_workflows_file_allowlist_paths(parse_workflows_file_allowlist(raw))


def workflow_file_base_dir(context: dict[str, Any], config: dict[str, Any] | None) -> Path:  # noqa: ARG001
    """Resolve the base directory for workflow file access.

    Args:
        context: Workflow context, may include `user_id`
        config: Currently unused; reserved for parity

    Returns:
        A resolved base directory for allowed file access.

    Security:
        Only honors server-side `WORKFLOWS_FILE_BASE_DIR` overrides.
    """
    env_override = os.getenv("WORKFLOWS_FILE_BASE_DIR")
    if env_override:
        base = Path(str(env_override)).expanduser()
        if not base.is_absolute():
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                base = (Path(get_project_root()) / base).resolve()
            except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
                logger.debug("Workflow file base dir: failed to resolve relative override")
                base = base.resolve()
        else:
            base = base.resolve()
        return base
    try:
        from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
        raw_user_id = context.get("user_id") if isinstance(context, dict) else None
        try:
            user_id = int(raw_user_id) if raw_user_id is not None else DatabasePaths.get_single_user_id()
        except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
            logger.debug("Workflow file base dir: invalid user id; using single-user fallback")
            user_id = DatabasePaths.get_single_user_id()
        return DatabasePaths.get_user_base_directory(user_id)
    except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
        logger.debug("Workflow file base dir: failed to resolve per-user base dir; using Databases fallback")
        return Path("Databases").resolve()


def resolve_workflow_file_path(path_value: str, context: dict[str, Any], config: dict[str, Any] | None = None) -> Path:
    """Resolve a workflow file path relative to the allowed base directory.

    Args:
        path_value: User-supplied path or filename
        context: Workflow context used to derive base dir
        config: Optional config; only used to check the unsafe access flag

    Returns:
        A resolved filesystem path.

    Security:
        Enforces containment via `is_subpath`, raising
        `AdapterError("file_access_denied")` on violations.
    """
    base_dir = workflow_file_base_dir(context, config)
    try:
        base_resolved = base_dir.resolve(strict=False)
    except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to resolve workflow file base directory")
        base_resolved = base_dir
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (base_resolved / candidate).resolve(strict=False)
    if unsafe_file_access_allowed(config):
        allowed_bases = [base_resolved]
        try:
            allowed_bases.extend(workflow_file_allowlist(context))
        except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
            logger.debug("Workflow file allowlist: failed to resolve allowlist")
        if not any(is_subpath(base, resolved) for base in allowed_bases):
            raise AdapterError("file_access_denied")
        return resolved
    if not is_subpath(base_resolved, resolved):
        raise AdapterError("file_access_denied")
    return resolved


def resolve_workflow_file_uri(file_uri: str, context: dict[str, Any], config: dict[str, Any] | None = None) -> Path:
    """Resolve a `file://` URI to a safe local filesystem path.

    Args:
        file_uri: File URI to resolve (must start with `file://`)
        context: Workflow context used to derive base dir
        config: Optional config for unsafe access flag

    Returns:
        A resolved filesystem path.

    Security:
        Rejects non-file URIs with `AdapterError("missing_or_invalid_file_uri")`.
    """
    if not file_uri.startswith("file://"):
        raise AdapterError("missing_or_invalid_file_uri")
    raw_path = file_uri[len("file://"):]
    return resolve_workflow_file_path(raw_path, context, config)


def normalize_str_list(value: Any) -> list[str]:
    """Normalize a value to a list of strings.

    Args:
        value: Input value (str, list, tuple, set, or other)

    Returns:
        List of non-empty trimmed strings
    """
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(value).strip()]


def extract_mcp_policy(context: dict[str, Any]) -> dict[str, Any]:
    """Extract MCP policy from workflow context.

    Args:
        context: Workflow context

    Returns:
        MCP policy dict (empty dict if not found)
    """
    policy = context.get("workflow_mcp_policy")
    if not isinstance(policy, dict):
        policy = None
    if policy is None:
        meta = context.get("workflow_metadata")
        if isinstance(meta, dict):
            candidate = meta.get("mcp") or meta.get("mcp_policy")
            if isinstance(candidate, dict):
                policy = candidate
    return policy or {}


def tool_matches_allowlist(tool_name: str, allowlist: list[str]) -> bool:
    """Check if a tool name matches the allowlist.

    Args:
        tool_name: Name of the tool
        allowlist: List of allowed tool patterns (supports * wildcard)

    Returns:
        True if tool is allowed
    """
    if not allowlist:
        return True
    if "*" in allowlist:
        return True
    for entry in allowlist:
        if entry == tool_name:
            return True
        if entry.endswith("*") and tool_name.startswith(entry[:-1]):
            return True
    return False


def extract_tool_scopes(tool_def: dict[str, Any] | None) -> list[str]:
    """Extract scopes from a tool definition.

    Args:
        tool_def: Tool definition dict

    Returns:
        List of scope strings
    """
    if not isinstance(tool_def, dict):
        return []
    raw = tool_def.get("scopes") or tool_def.get("scope")
    if raw is None:
        meta = tool_def.get("metadata") or {}
        if isinstance(meta, dict):
            raw = meta.get("scopes") or meta.get("scope") or meta.get("capabilities") or meta.get("capability")
    return normalize_str_list(raw)


def format_time_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        SRT-formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_time_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        VTT-formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class AsyncFileWriter:
    """Minimal async file writer context manager for streaming to disk.

    Uses synchronous file I/O; keep payloads small or swap to aiofiles if needed.
    """

    def __init__(self, path: Path):
        self._path = path
        self._fp = None

    async def __aenter__(self):
        self._fp = open(self._path, "wb")
        return self

    async def write(self, data: bytes):
        self._fp.write(data)

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._fp:
                self._fp.flush()
                self._fp.close()
        except _WORKFLOW_COMMON_NONCRITICAL_EXCEPTIONS:
            pass


# Backward-compatible aliases with underscore prefix
_async_file_writer = AsyncFileWriter
_extract_openai_content = extract_openai_content
_sanitize_path_component = sanitize_path_component
_is_subpath = is_subpath
_resolve_context_user_id = resolve_context_user_id
_artifacts_base_dir = artifacts_base_dir
_resolve_artifacts_dir = resolve_artifacts_dir
_resolve_artifact_filename = resolve_artifact_filename
_unsafe_file_access_allowed = unsafe_file_access_allowed
_parse_workflows_file_allowlist = parse_workflows_file_allowlist
_resolve_workflows_file_allowlist_paths = resolve_workflows_file_allowlist_paths
_workflow_file_allowlist = workflow_file_allowlist
_workflow_file_base_dir = workflow_file_base_dir
_resolve_workflow_file_path = resolve_workflow_file_path
_resolve_workflow_file_uri = resolve_workflow_file_uri
_normalize_str_list = normalize_str_list
_extract_mcp_policy = extract_mcp_policy
_tool_matches_allowlist = tool_matches_allowlist
_extract_tool_scopes = extract_tool_scopes
_format_time_srt = format_time_srt
_format_time_vtt = format_time_vtt
