from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess  # nosec B404
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import (
    open_safe_local_path,
    resolve_safe_local_path,
)

from tldw_Server_API.app.core.Ingestion_Sources.local_directory import (
    validate_local_directory_source,
)

GIT_REPOSITORY_MODES: frozenset[str] = frozenset({"local_repo", "remote_github_repo"})
GIT_NOTES_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".markdown", ".md", ".txt"})
_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_API_READ_CHUNK_BYTES = 64 * 1024
_MAX_GITHUB_BLOB_BYTES = 5 * 1024 * 1024
_GITHUB_BLOB_RESPONSE_OVERHEAD_BYTES = 64 * 1024
_LOCAL_FILE_MAX_BYTES_ENV = "INGESTION_SOURCES_LOCAL_FILE_MAX_BYTES"
_GIT_LS_FILES_TIMEOUT_SECONDS_ENV = "INGESTION_SOURCES_GIT_LS_FILES_TIMEOUT_SECONDS"
_DEFAULT_LOCAL_FILE_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_GIT_LS_FILES_TIMEOUT_SECONDS = 30


def _positive_int_from_env(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return value


def _local_file_max_bytes() -> int:
    return _positive_int_from_env(_LOCAL_FILE_MAX_BYTES_ENV, _DEFAULT_LOCAL_FILE_MAX_BYTES)


def _git_ls_files_timeout_seconds() -> int:
    return _positive_int_from_env(
        _GIT_LS_FILES_TIMEOUT_SECONDS_ENV,
        _DEFAULT_GIT_LS_FILES_TIMEOUT_SECONDS,
    )


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_mode(config: dict[str, Any]) -> str:
    raw_mode = str(config.get("mode") or "").strip().lower()
    if raw_mode:
        if raw_mode not in GIT_REPOSITORY_MODES:
            allowed = ", ".join(sorted(GIT_REPOSITORY_MODES))
            raise ValueError(f"Git repository source mode must be one of: {allowed}")
        return raw_mode
    if str(config.get("repo_url") or "").strip():
        return "remote_github_repo"
    return "local_repo"


def _normalize_optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_path_subpath(value: Any) -> str | None:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return None
    normalized = raw.strip("/")
    if not normalized or normalized == ".":
        return None
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Git repository root_subpath must stay within the repository root")
    return candidate.as_posix()


def _normalize_glob_list(value: Any) -> list[str]:
    items: list[str]
    if value is None:
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(",", "\n").splitlines()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(part).strip() for part in value]
    else:
        raise ValueError("Git repository glob filters must be a string or list of strings")
    return [item for item in items if item]


def _normalize_account_id(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        normalized = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Git repository account_id must be an integer") from exc
    if normalized <= 0:
        raise ValueError("Git repository account_id must be a positive integer")
    return normalized


def _normalize_github_repo_url(value: Any) -> tuple[str, str, str]:
    repo_url = str(value or "").strip()
    if not repo_url:
        raise ValueError("Remote GitHub repository sources require a non-empty repo_url")

    parsed = urlparse(repo_url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        raise ValueError("Remote git repository sources currently support GitHub HTTPS URLs only")

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        raise ValueError("GitHub repository URL must include both owner and repository name")

    owner = segments[0].strip()
    repo_name = segments[1].strip()
    if not owner or not repo_name:
        raise ValueError("GitHub repository URL must include both owner and repository name")

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    normalized_url = f"https://github.com/{owner}/{repo_name}"
    return normalized_url, owner, repo_name


def validate_local_git_repository_source(config: dict[str, Any]) -> Path:
    repo_root = validate_local_directory_source({"path": config.get("path")})
    git_marker = repo_root / ".git"
    if not git_marker.exists():
        raise ValueError(
            f"Local git repository source path is not a git repository root or worktree: {repo_root}"
        )
    return repo_root


def validate_git_repository_source(config: dict[str, Any]) -> dict[str, Any]:
    mode = _normalize_mode(config)
    normalized: dict[str, Any] = {"mode": mode}

    label = _normalize_optional_string(config.get("label"))
    if label:
        normalized["label"] = label

    ref = _normalize_optional_string(config.get("ref"))
    if ref:
        normalized["ref"] = ref

    root_subpath = _normalize_path_subpath(config.get("root_subpath"))
    if root_subpath:
        normalized["root_subpath"] = root_subpath

    include_globs = _normalize_glob_list(config.get("include_globs"))
    if include_globs:
        normalized["include_globs"] = include_globs

    exclude_globs = _normalize_glob_list(config.get("exclude_globs"))
    if exclude_globs:
        normalized["exclude_globs"] = exclude_globs

    if mode == "local_repo":
        repo_root = validate_local_git_repository_source(config)
        normalized["path"] = str(repo_root)
        normalized["respect_gitignore"] = bool(
            True if config.get("respect_gitignore") is None else config.get("respect_gitignore")
        )
        return normalized

    repo_url, owner, repo_name = _normalize_github_repo_url(config.get("repo_url"))
    normalized["repo_url"] = repo_url
    normalized["repo_owner"] = owner
    normalized["repo_name"] = repo_name
    account_id = _normalize_account_id(config.get("account_id"))
    if account_id is not None:
        normalized["account_id"] = account_id
    return normalized


def _supported_suffixes_for_sink(sink_type: str) -> frozenset[str]:
    normalized = str(sink_type or "").strip().lower()
    if normalized == "notes":
        return GIT_NOTES_SUPPORTED_SUFFIXES
    return frozenset()


def _matches_globs(relative_path: str, patterns: list[str]) -> bool:
    path = PurePosixPath(relative_path)
    return any(path.match(pattern) for pattern in patterns)


def _path_is_included(
    relative_path: str,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
) -> bool:
    if include_globs and not _matches_globs(relative_path, include_globs):
        return False
    if exclude_globs and _matches_globs(relative_path, exclude_globs):
        return False
    return True


def _normalize_scan_root(
    repo_root: Path,
    root_subpath: str | None,
) -> tuple[Path, str | None]:
    if not root_subpath:
        return repo_root, None
    safe_scan_root = resolve_safe_local_path(repo_root / root_subpath, repo_root)
    if safe_scan_root is None:
        raise ValueError(f"Git repository root_subpath is outside the repository: {root_subpath}")
    if not safe_scan_root.exists():
        raise ValueError(f"Git repository root_subpath does not exist: {root_subpath}")
    if not safe_scan_root.is_dir():
        raise ValueError(f"Git repository root_subpath is not a directory: {root_subpath}")
    return safe_scan_root, root_subpath


def _read_local_text_file(path: Path, *, base_dir: Path) -> str:
    handle = open_safe_local_path(path, base_dir, mode="rb")
    if handle is None:
        raise ValueError(f"Git repository source path rejected: {path}")
    with handle:
        data = handle.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _build_snapshot_item(
    *,
    relative_path: str,
    text: str,
    source_format: str,
    raw_metadata: dict[str, Any] | None = None,
    modified_at: str | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    normalized_metadata = dict(raw_metadata or {})
    normalized_metadata.setdefault("source_format", source_format)
    return {
        "relative_path": relative_path,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "modified_at": modified_at or _utc_now_text(),
        "size": len(text.encode("utf-8")) if size is None else int(size),
        "source_format": source_format,
        "raw_metadata": normalized_metadata,
        "text": text,
    }


def _git_ls_files(
    *,
    repo_root: Path,
    root_subpath: str | None,
    respect_gitignore: bool,
) -> list[str]:
    command = ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others"]
    if respect_gitignore:
        command.append("--exclude-standard")
    if root_subpath:
        command.extend(["--", root_subpath])
    try:
        completed = subprocess.run(  # nosec B603
            command,
            check=False,
            capture_output=True,
            text=False,
            timeout=_git_ls_files_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Timed out enumerating git repository files for {repo_root}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(
            stderr or f"Failed to enumerate git repository files for {repo_root}"
        )
    return [
        entry.replace("\\", "/").strip().strip("/")
        for entry in completed.stdout.decode("utf-8", errors="ignore").split("\0")
        if entry.strip()
    ]


def _iter_local_repository_candidates(
    *,
    repo_root: Path,
    scan_root: Path,
    root_subpath: str | None,
    respect_gitignore: bool,
) -> list[tuple[str, Path]]:
    if not respect_gitignore:
        candidates: list[tuple[str, Path]] = []
        for file_path in sorted(scan_root.rglob("*")):
            if not file_path.is_file() or file_path.is_symlink():
                continue
            safe_path = resolve_safe_local_path(file_path, scan_root)
            if safe_path is None:
                continue
            candidates.append((safe_path.relative_to(scan_root).as_posix(), safe_path))
        return candidates

    git_paths = _git_ls_files(
        repo_root=repo_root,
        root_subpath=root_subpath,
        respect_gitignore=respect_gitignore,
    )
    normalized_root_prefix = f"{root_subpath.rstrip('/')}/" if root_subpath else ""
    candidates = []
    for repo_relative_path in git_paths:
        if repo_relative_path.startswith(".git/"):
            continue
        if normalized_root_prefix:
            if not repo_relative_path.startswith(normalized_root_prefix):
                continue
            relative_path = repo_relative_path[len(normalized_root_prefix):]
        else:
            relative_path = repo_relative_path
        if not relative_path:
            continue
        safe_path = resolve_safe_local_path(repo_root / repo_relative_path, repo_root)
        if safe_path is None or not safe_path.is_file():
            continue
        candidates.append((relative_path, safe_path))
    return sorted(candidates, key=lambda item: item[0])


def _validate_github_api_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "api.github.com":
        raise ValueError("Git repository remote sync only allows GitHub API HTTPS URLs")
    return url


def _github_request(url: str, *, access_token: str | None, accept: str) -> Request:
    headers = {
        "Accept": accept,
        "User-Agent": "tldw-server-ingestion-sources",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return Request(_validate_github_api_url(url), headers=headers)


def _github_api_get_json(
    url: str,
    *,
    access_token: str | None,
    accept: str = "application/vnd.github+json",
    max_response_bytes: int | None = None,
) -> dict[str, Any]:
    with urlopen(  # nosec B310
        _github_request(url, access_token=access_token, accept=accept),
        timeout=30,
    ) as response:
        if max_response_bytes is None:
            payload_bytes = response.read()
        else:
            chunks: list[bytes] = []
            total_read = 0
            while True:
                chunk = response.read(_GITHUB_API_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > max_response_bytes:
                    raise ValueError(
                        f"GitHub API response exceeded size limit of {max_response_bytes} bytes"
                    )
                chunks.append(chunk)
            payload_bytes = b"".join(chunks)
        payload = payload_bytes.decode("utf-8")
    return json.loads(payload)


def _github_repository_ref(
    *,
    owner: str,
    repo_name: str,
    requested_ref: str | None,
    access_token: str | None,
) -> str:
    if requested_ref:
        return requested_ref
    repo_payload = _github_api_get_json(
        f"{_GITHUB_API_BASE}/repos/{owner}/{repo_name}",
        access_token=access_token,
    )
    default_branch = str(repo_payload.get("default_branch") or "").strip()
    if not default_branch:
        raise ValueError(
            f"GitHub repository {owner}/{repo_name} does not expose a default branch"
        )
    return default_branch


def _max_github_blob_response_bytes(blob_size: int | None) -> int:
    target_size = _MAX_GITHUB_BLOB_BYTES if blob_size is None else max(int(blob_size), 0)
    encoded_bytes = int(math.ceil(target_size / 3) * 4)
    return encoded_bytes + _GITHUB_BLOB_RESPONSE_OVERHEAD_BYTES


def _github_fetch_blob_text(
    blob_url: str,
    *,
    access_token: str | None,
    blob_size: int | None,
    relative_path: str,
) -> str:
    if blob_size is not None and blob_size > _MAX_GITHUB_BLOB_BYTES:
        raise ValueError(
            f"Git repository blob '{relative_path}' is too large ({blob_size} bytes > {_MAX_GITHUB_BLOB_BYTES} bytes)"
        )
    payload = _github_api_get_json(
        blob_url,
        access_token=access_token,
        max_response_bytes=_max_github_blob_response_bytes(blob_size),
    )
    content = str(payload.get("content") or "")
    encoding = str(payload.get("encoding") or "").strip().lower()
    if encoding == "base64":
        decoded = base64.b64decode(content.encode("utf-8"), validate=False)
        if len(decoded) > _MAX_GITHUB_BLOB_BYTES:
            raise ValueError(
                f"Git repository blob '{relative_path}' is too large ({len(decoded)} bytes > {_MAX_GITHUB_BLOB_BYTES} bytes)"
            )
        try:
            return decoded.decode("utf-8")
        except UnicodeDecodeError:
            return decoded.decode("latin-1")
    content_bytes = len(content.encode("utf-8"))
    if content_bytes > _MAX_GITHUB_BLOB_BYTES:
        raise ValueError(
            f"Git repository blob '{relative_path}' is too large ({content_bytes} bytes > {_MAX_GITHUB_BLOB_BYTES} bytes)"
        )
    return content


def _relative_path_with_root_subpath(
    repo_relative_path: str,
    *,
    root_subpath: str | None,
) -> str | None:
    normalized = repo_relative_path.replace("\\", "/").strip().strip("/")
    if not normalized:
        return None
    if not root_subpath:
        return normalized
    prefix = f"{root_subpath.rstrip('/')}/"
    if not normalized.startswith(prefix):
        return None
    relative = normalized[len(prefix):]
    return relative or None


def build_git_repository_snapshot_with_failures(
    config: dict[str, Any],
    *,
    sink_type: str,
    access_token: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized_config = validate_git_repository_source(config)
    supported_suffixes = _supported_suffixes_for_sink(sink_type)
    if not supported_suffixes:
        raise ValueError(
            f"Git repository sources do not support sink type '{sink_type}'."
        )

    mode = str(normalized_config.get("mode") or "")
    include_globs = list(normalized_config.get("include_globs") or [])
    exclude_globs = list(normalized_config.get("exclude_globs") or [])
    root_subpath = (
        str(normalized_config.get("root_subpath") or "").strip() or None
    )
    snapshot_items: dict[str, dict[str, Any]] = {}
    failed_items: dict[str, dict[str, Any]] = {}

    if mode == "local_repo":
        repo_root = validate_local_git_repository_source(normalized_config)
        scan_root, normalized_root_subpath = _normalize_scan_root(repo_root, root_subpath)
        max_file_bytes = _local_file_max_bytes()
        candidates = _iter_local_repository_candidates(
            repo_root=repo_root,
            scan_root=scan_root,
            root_subpath=normalized_root_subpath,
            respect_gitignore=bool(normalized_config.get("respect_gitignore", True)),
        )
        for relative_path, file_path in candidates:
            suffix = file_path.suffix.lower()
            if suffix not in supported_suffixes:
                continue
            if not _path_is_included(
                relative_path,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            ):
                continue
            stat = file_path.stat()
            if int(stat.st_size) > max_file_bytes:
                failed_items[relative_path] = {
                    "relative_path": relative_path,
                    "source_format": suffix.lstrip(".") or "unknown",
                    "size": int(stat.st_size),
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "error": (
                        f"File '{relative_path}' exceeds local file size limit of "
                        f"{max_file_bytes} bytes"
                    ),
                }
                continue
            try:
                text = _read_local_text_file(file_path, base_dir=repo_root)
            except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                failed_items[relative_path] = {
                    "relative_path": relative_path,
                    "source_format": suffix.lstrip(".") or "unknown",
                    "size": int(stat.st_size),
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(exc),
                }
                continue
            snapshot_items[relative_path] = _build_snapshot_item(
                relative_path=relative_path,
                text=text,
                source_format=suffix.lstrip(".") or "text",
                raw_metadata={
                    "repo_mode": "local_repo",
                    "repo_path": str(repo_root),
                    "repo_relative_path": (
                        f"{normalized_root_subpath}/{relative_path}"
                        if normalized_root_subpath
                        else relative_path
                    ),
                },
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S"),
                size=int(stat.st_size),
            )
        return snapshot_items, failed_items

    owner = str(normalized_config.get("repo_owner") or "").strip()
    repo_name = str(normalized_config.get("repo_name") or "").strip()
    repo_url = str(normalized_config.get("repo_url") or "").strip()
    resolved_ref = _github_repository_ref(
        owner=owner,
        repo_name=repo_name,
        requested_ref=str(normalized_config.get("ref") or "").strip() or None,
        access_token=access_token,
    )
    tree_payload = _github_api_get_json(
        (
            f"{_GITHUB_API_BASE}/repos/{owner}/{repo_name}/git/trees/"
            f"{quote(resolved_ref, safe='')}?recursive=1"
        ),
        access_token=access_token,
    )
    if bool(tree_payload.get("truncated")):
        raise ValueError(
            f"GitHub repository tree response for {owner}/{repo_name}@{resolved_ref} was truncated"
        )
    for entry in list(tree_payload.get("tree") or []):
        if str(entry.get("type") or "").strip().lower() != "blob":
            continue
        repo_relative_path = str(entry.get("path") or "").strip()
        relative_path = _relative_path_with_root_subpath(
            repo_relative_path,
            root_subpath=root_subpath,
        )
        if not relative_path:
            continue
        suffix = PurePosixPath(relative_path).suffix.lower()
        if suffix not in supported_suffixes:
            continue
        if not _path_is_included(
            relative_path,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        ):
            continue
        try:
            text = _github_fetch_blob_text(
                str(entry.get("url") or ""),
                access_token=access_token,
                blob_size=int(entry.get("size") or 0) or None,
                relative_path=relative_path,
            )
        except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            failed_items[relative_path] = {
                "relative_path": relative_path,
                "source_format": suffix.lstrip(".") or "unknown",
                "size": int(entry.get("size") or 0),
                "modified_at": _utc_now_text(),
                "error": str(exc),
            }
            continue
        snapshot_items[relative_path] = _build_snapshot_item(
            relative_path=relative_path,
            text=text,
            source_format=suffix.lstrip(".") or "text",
            raw_metadata={
                "repo_mode": "remote_github_repo",
                "repo_url": repo_url,
                "repo_relative_path": repo_relative_path,
                "repo_ref": resolved_ref,
                "repo_blob_sha": str(entry.get("sha") or ""),
            },
            size=int(entry.get("size") or len(text.encode("utf-8"))),
        )
    return snapshot_items, failed_items
