#!/usr/bin/env python3
"""Generate immutable Phase 4 predecessor behavior fixtures."""

from __future__ import annotations

import argparse
import asyncio
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import sys
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None  # type: ignore[assignment]

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = (
    "article_orchestration_fakes",
    "content",
    "extraction",
    "metadata",
    "router",
    "selectors",
)
_PRIOR_CASE_NAMES = (
    "article_orchestration_fakes",
    "content",
    "extraction",
    "metadata",
    "selectors",
)
_STABLE_IDENTITY_ERROR = "Fixture filesystem does not provide stable identity"
_STAGING_CLEANUP_ERROR = "Fixture staging directory could not be cleaned up"
_STAGING_RETAINED_DIAGNOSTIC = "warning: fixture staging directory retained for manual cleanup"
_RECOVERY_ERROR = "Fixture publication recovery could not be completed safely; manual recovery is required"


@dataclass(frozen=True)
class _FixtureFileSnapshot:
    name: str
    identity: tuple[int, int, int]
    size: int
    sha256: str


@dataclass(frozen=True)
class _FixtureSetSnapshot:
    directory_identity: tuple[int, int, int]
    files: tuple[_FixtureFileSnapshot, ...]


@dataclass(frozen=True)
class _RecoveryRecord:
    path: Path
    identity: tuple[int, int, int]
    raw: bytes
    parent_identity: tuple[int, int, int]
    backup: Path
    output_snapshot: _FixtureSetSnapshot


def _report_staging_retained() -> None:
    try:
        print(_STAGING_RETAINED_DIAGNOSTIC, file=sys.stderr)
    except BaseException:  # noqa: BLE001 - diagnostics must never replace the active failure
        pass


def _report_backup_retained(backup: Path) -> None:
    try:
        print(
            "warning: fixture output committed; backup cleanup failed; "
            f"backup retained as {backup.name!r} for manual cleanup",
            file=sys.stderr,
        )
    except BaseException:  # noqa: BLE001 - committed output must remain successful
        pass


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Helper_Scripts.web_scraping_phase4.content_metadata import (  # noqa: E402
    build_content_cases,
    build_metadata_cases,
)
from Helper_Scripts.web_scraping_phase4.extraction import (  # noqa: E402
    build_extraction_cases,
)
from Helper_Scripts.web_scraping_phase4.orchestration import (  # noqa: E402
    build_article_cases,
)
from Helper_Scripts.web_scraping_phase4.router import build_router_cases  # noqa: E402
from Helper_Scripts.web_scraping_phase4.selectors import (  # noqa: E402
    build_selector_cases,
)
from Helper_Scripts.web_scraping_phase4.shared import FIXED_ENV  # noqa: E402


def build_manifest(predecessor_commit: str, case_files: dict[str, str]) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", predecessor_commit) is None:
        raise ValueError("predecessor_commit must be a full 40-character lowercase commit id")
    return {
        "schema_version": SCHEMA_VERSION,
        "predecessor_commit": predecessor_commit,
        "cases": dict(sorted(case_files.items())),
    }


def _run_git(source_root: Path, *args: str) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise OSError("git executable not found")
    completed = subprocess.run(  # nosec B603
        [git_executable, *args],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source_root(predecessor_commit: str, source_root: Path) -> Path:
    build_manifest(predecessor_commit, {})
    resolved_root = source_root.resolve(strict=True)
    top_level = Path(_run_git(resolved_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != resolved_root:
        raise ValueError(f"source-root must be the git worktree root: {resolved_root}")

    head = _run_git(resolved_root, "rev-parse", "HEAD")
    if predecessor_commit != head:
        raise ValueError(f"predecessor_commit {predecessor_commit} does not match source-root HEAD {head}")

    status = _run_git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
    )
    if status:
        raise ValueError(f"source-root is not clean: {resolved_root}\n{status}")
    return resolved_root


def _remove_loaded_production_modules() -> dict[str, Any]:
    removed: dict[str, Any] = {}
    for name in tuple(sys.modules):
        if name == "tldw_Server_API" or name.startswith("tldw_Server_API."):
            removed[name] = sys.modules.pop(name)
    return removed


def _assert_production_modules_under(source_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "tldw_Server_API" and not name.startswith("tldw_Server_API."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            resolved_locations = (Path(module_file).resolve(),)
        else:
            module_path = getattr(module, "__path__", None)
            if module_path is None:
                raise RuntimeError(f"Imported predecessor module {name} has no resolvable path")
            resolved_locations = tuple(Path(location).resolve() for location in module_path)
            if not resolved_locations:
                raise RuntimeError(f"Imported predecessor module {name} has an empty package path")
        outside_root = [location for location in resolved_locations if not location.is_relative_to(source_root)]
        if outside_root:
            raise RuntimeError(
                f"Imported predecessor module {name} resolved outside source-root: "
                f"{', '.join(map(str, outside_root))}"
            )


@contextmanager
def _predecessor_modules(source_root: Path) -> Iterator[tuple[Any, Any, Any, Any, Any]]:
    previous_modules = _remove_loaded_production_modules()
    previous_path = list(sys.path)
    sys.path.insert(0, str(source_root))
    try:
        from tldw_Server_API.app.core.Watchlists import fetchers
        from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
        from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, PolicyDecision
        from tldw_Server_API.app.core.Web_Scraping.scraper_router import ScraperRouter

        _assert_production_modules_under(source_root)
        yield article, fetchers, FetchResponse, PolicyDecision, ScraperRouter
        _assert_production_modules_under(source_root)
    finally:
        _remove_loaded_production_modules()
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


def build_case_payloads(source_root: Path) -> dict[str, dict[str, Any]]:
    with _predecessor_modules(source_root) as modules:
        article, fetchers, FetchResponse, PolicyDecision, ScraperRouter = modules
        try:
            with patch.dict(os.environ, FIXED_ENV, clear=False):
                fetchers.reload_selector_guardrails_from_env()
                payloads = {
                    "article_orchestration_fakes": {
                        "category": "article_orchestration_fakes",
                        "cases": asyncio.run(build_article_cases(article, FetchResponse, PolicyDecision)),
                    },
                    "content": {
                        "category": "content",
                        "cases": build_content_cases(article),
                    },
                    "extraction": {
                        "category": "extraction",
                        "cases": build_extraction_cases(article),
                    },
                    "metadata": {
                        "category": "metadata",
                        "cases": build_metadata_cases(article),
                    },
                    "router": {
                        "category": "router",
                        "cases": build_router_cases(ScraperRouter),
                    },
                    "selectors": {
                        "category": "selectors",
                        "cases": build_selector_cases(fetchers),
                    },
                }
        finally:
            fetchers.reload_selector_guardrails_from_env()

    if set(payloads) != set(CASE_NAMES):
        raise RuntimeError("Fixture category set is incomplete")
    return payloads


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _write_fixture_set(
    output: Path,
    predecessor_commit: str,
    payloads: Mapping[str, object],
) -> None:
    case_files: dict[str, str] = {}
    for category in sorted(payloads):
        filename = f"{category}.json"
        _write_json(output / filename, payloads[category])
        case_files[category] = filename

    _write_json(output / "manifest.json", build_manifest(predecessor_commit, case_files))


def _invalid_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON value")


def _validate_fixture_set(
    output: Path,
    predecessor_commit: str | None,
    *,
    case_names: tuple[str, ...] = CASE_NAMES,
) -> _FixtureSetSnapshot:
    case_files = {category: f"{category}.json" for category in case_names}
    expected_names = {"manifest.json", *case_files.values()}
    try:
        directory_before = os.stat(output, follow_symlinks=False)
        directory_identity = _stable_metadata_identity(directory_before)
        entries = list(output.iterdir())
    except OSError:
        raise RuntimeError("Fixture set could not be inspected") from None

    actual_names = {path.name for path in entries}
    if actual_names != expected_names:
        raise RuntimeError("Fixture set does not contain the exact required files")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise RuntimeError("Fixture set entries must be regular files")

    decoded: dict[str, object] = {}
    file_snapshots: list[_FixtureFileSnapshot] = []
    for path in sorted(entries):
        try:
            metadata_before = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(metadata_before.st_mode):
                raise RuntimeError("Fixture set entries must be regular files")
            file_identity = _stable_metadata_identity(metadata_before)
            raw = path.read_bytes()
            metadata_after = os.stat(path, follow_symlinks=False)
            if _stable_metadata_identity(metadata_after) != file_identity or metadata_after.st_size != len(raw):
                raise RuntimeError("Fixture set changed during validation")
            payload = json.loads(
                raw.decode("ascii"),
                parse_constant=_invalid_json_constant,
            )
        except RuntimeError:
            raise
        except (OSError, UnicodeError, ValueError):
            raise RuntimeError(f"Fixture set contains invalid JSON: {path.name}") from None
        canonical = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
        if raw != canonical:
            raise RuntimeError(f"Fixture set contains noncanonical JSON: {path.name}")
        decoded[path.name] = payload
        file_snapshots.append(
            _FixtureFileSnapshot(
                name=path.name,
                identity=file_identity,
                size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )

    manifest = decoded["manifest.json"]
    if type(manifest) is not dict or set(manifest) != {"schema_version", "predecessor_commit", "cases"}:
        raise RuntimeError("Fixture set manifest structure is invalid")
    manifest_commit = manifest["predecessor_commit"]
    if type(manifest_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", manifest_commit) is None:
        raise RuntimeError("Fixture set manifest predecessor is invalid")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("Fixture set manifest schema version is invalid")
    if type(manifest["cases"]) is not dict or manifest["cases"] != case_files:
        raise RuntimeError("Fixture set manifest cases are invalid")
    if manifest != build_manifest(manifest_commit, case_files):
        raise RuntimeError("Fixture set manifest structure is invalid")
    if predecessor_commit is not None and manifest_commit != predecessor_commit:
        raise RuntimeError("Fixture set manifest does not match the requested provenance")

    for category, filename in case_files.items():
        payload = decoded[filename]
        if type(payload) is not dict or set(payload) != {"category", "cases"}:
            raise RuntimeError(f"Fixture set category structure is invalid: {filename}")
        if type(payload["category"]) is not str or payload["category"] != category:
            raise RuntimeError(f"Fixture set category is invalid: {filename}")
        if type(payload["cases"]) is not list or not payload["cases"]:
            raise RuntimeError(f"Fixture set cases are invalid: {filename}")
        if any(type(case) is not dict for case in payload["cases"]):
            raise RuntimeError(f"Fixture set case entry is invalid: {filename}")

    try:
        if _stable_metadata_identity(os.stat(output, follow_symlinks=False)) != directory_identity:
            raise RuntimeError("Fixture set changed during validation")
    except OSError:
        raise RuntimeError("Fixture set changed during validation") from None
    return _FixtureSetSnapshot(
        directory_identity=directory_identity,
        files=tuple(file_snapshots),
    )


def _resolve_output_path(output: Path, source_root: Path) -> Path:
    try:
        candidate = output if output.is_absolute() else Path.cwd() / output
        missing_parts: list[str] = []
        while True:
            try:
                resolved_output = candidate.resolve(strict=True)
            except FileNotFoundError:
                if candidate.name == ".." or candidate.is_symlink() or candidate == candidate.parent:
                    raise ValueError from None
                missing_parts.append(candidate.name)
                candidate = candidate.parent
            else:
                resolved_output = resolved_output.joinpath(*reversed(missing_parts))
                break
    except (OSError, RuntimeError):
        raise ValueError("output path could not be resolved") from None
    except ValueError:
        raise ValueError("output path could not be resolved") from None

    if source_root.is_relative_to(resolved_output):
        raise ValueError("output must not be source-root or an ancestor of source-root")

    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError("existing output must be a directory")
        return resolved_output

    existing_parent = resolved_output.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise ValueError("output parent must be a directory")
    return resolved_output


def _validate_existing_output(output: Path) -> _FixtureSetSnapshot | None:
    try:
        metadata = os.stat(output, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise RuntimeError("Fixture output changed during validation") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("existing output must be a directory")
    identity = _stable_metadata_identity(metadata)
    try:
        snapshot = _validate_fixture_set(output, predecessor_commit=None)
    except RuntimeError as current_error:
        try:
            snapshot = _validate_fixture_set(
                output,
                predecessor_commit=None,
                case_names=_PRIOR_CASE_NAMES,
            )
        except RuntimeError:
            raise current_error from None
    if snapshot.directory_identity != identity:
        raise RuntimeError("Fixture output changed during validation")
    _require_path_identity(output, identity, "Fixture output changed during validation")
    return snapshot


def _lock_path_for_output(output: Path) -> Path:
    normalized_output = os.path.normcase(str(output.resolve()))
    identity = hashlib.sha256(normalized_output.encode("utf-8")).hexdigest()
    namespace = "tldw-phase4-fixture-locks"
    get_effective_uid = getattr(os, "geteuid", None)
    if callable(get_effective_uid):
        namespace = f"{namespace}-{get_effective_uid()}"
    lock_root = Path(tempfile.gettempdir()).resolve() / namespace
    return lock_root / f"{identity}.lock"


def _is_link_like(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    if reparse_point and file_attributes & reparse_point:
        return True
    if getattr(metadata, "st_reparse_tag", 0):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _effective_uid() -> int | None:
    get_effective_uid = getattr(os, "geteuid", None)
    if not callable(get_effective_uid):
        return None
    return get_effective_uid()


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int] | None:
    device = getattr(metadata, "st_dev", None)
    inode = getattr(metadata, "st_ino", None)
    if not isinstance(device, int) or not isinstance(inode, int) or inode == 0:
        return None
    return device, inode, stat.S_IFMT(metadata.st_mode)


def _stable_metadata_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    identity = _metadata_identity(metadata)
    if identity is None:
        raise RuntimeError(_STABLE_IDENTITY_ERROR)
    return identity


def _path_identity_or_none(path: Path, error_message: str) -> tuple[int, int, int] | None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise RuntimeError(error_message) from None
    return _stable_metadata_identity(metadata)


def _path_identity(path: Path, error_message: str) -> tuple[int, int, int]:
    identity = _path_identity_or_none(path, error_message)
    if identity is None:
        raise RuntimeError(error_message)
    return identity


def _require_path_identity(
    path: Path,
    expected: tuple[int, int, int],
    error_message: str,
) -> None:
    if _path_identity(path, error_message) != expected:
        raise RuntimeError(error_message)


def _require_real_directory_identity(
    path: Path,
    expected: tuple[int, int, int],
    error_message: str,
) -> None:
    try:
        link_like = _is_link_like(path)
    except (OSError, RuntimeError):
        raise RuntimeError(error_message) from None
    if expected[2] != stat.S_IFDIR or link_like:
        raise RuntimeError(error_message)
    _require_path_identity(path, expected, error_message)


def _require_path_absent(path: Path, error_message: str) -> None:
    if _path_identity_or_none(path, error_message) is not None:
        raise RuntimeError(error_message)


def _capture_fixture_set_snapshot(
    output: Path,
    error_message: str,
    *,
    expected_names: set[str] | None = None,
) -> _FixtureSetSnapshot:
    if expected_names is None:
        expected_names = {"manifest.json", *(f"{category}.json" for category in CASE_NAMES)}
    try:
        directory_identity = _path_identity(output, error_message)
        entries = list(output.iterdir())
        if {path.name for path in entries} != expected_names:
            raise RuntimeError(error_message)

        file_snapshots: list[_FixtureFileSnapshot] = []
        for path in sorted(entries):
            metadata_before = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(metadata_before.st_mode) or _is_link_like(path):
                raise RuntimeError(error_message)
            file_identity = _stable_metadata_identity(metadata_before)
            raw = path.read_bytes()
            metadata_after = os.stat(path, follow_symlinks=False)
            if _stable_metadata_identity(metadata_after) != file_identity or metadata_after.st_size != len(raw):
                raise RuntimeError(error_message)
            file_snapshots.append(
                _FixtureFileSnapshot(
                    name=path.name,
                    identity=file_identity,
                    size=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )

        _require_path_identity(output, directory_identity, error_message)
    except OSError:
        raise RuntimeError(error_message) from None
    return _FixtureSetSnapshot(
        directory_identity=directory_identity,
        files=tuple(file_snapshots),
    )


def _require_fixture_set_snapshot(
    output: Path,
    expected: _FixtureSetSnapshot,
    error_message: str,
) -> None:
    expected_names = {file.name for file in expected.files}
    if (
        _capture_fixture_set_snapshot(
            output,
            error_message,
            expected_names=expected_names,
        )
        != expected
    ):
        raise RuntimeError(error_message)


def _restore_output_backup(
    backup: Path,
    output: Path,
    parent_identity: tuple[int, int, int],
    output_snapshot: _FixtureSetSnapshot,
) -> None:
    recovery_error = "Fixture output rollback could not be completed safely; " "manual recovery is required"
    output_identity = output_snapshot.directory_identity
    try:
        _require_path_identity(output.parent, parent_identity, recovery_error)
        current_output = _path_identity_or_none(output, recovery_error)
        current_backup = _path_identity_or_none(backup, recovery_error)
        if current_output == output_identity and current_backup is None:
            _require_fixture_set_snapshot(output, output_snapshot, recovery_error)
            return
        if current_output is not None or current_backup != output_identity:
            raise RuntimeError(recovery_error)
        _require_real_directory_identity(backup, output_identity, recovery_error)
        _require_fixture_set_snapshot(backup, output_snapshot, recovery_error)
        backup.replace(output)
        try:
            _require_fixture_set_snapshot(output, output_snapshot, recovery_error)
            _require_path_absent(backup, recovery_error)
        except (OSError, RuntimeError):
            try:
                _require_path_identity(output, output_identity, recovery_error)
                _require_path_absent(backup, recovery_error)
                output.replace(backup)
                _require_path_absent(output, recovery_error)
                _require_path_identity(backup, output_identity, recovery_error)
            except (OSError, RuntimeError):
                pass
            raise RuntimeError(recovery_error) from None
    except (OSError, RuntimeError):
        raise RuntimeError(recovery_error) from None


def _valid_lock_root_metadata(metadata: os.stat_result) -> bool:
    if not stat.S_ISDIR(metadata.st_mode):
        return False
    effective_uid = _effective_uid()
    if effective_uid is not None and getattr(metadata, "st_uid", None) != effective_uid:
        return False
    return os.name != "posix" or stat.S_IMODE(metadata.st_mode) & 0o077 == 0


def _valid_lock_file_metadata(metadata: os.stat_result) -> bool:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        return False
    effective_uid = _effective_uid()
    if effective_uid is not None and getattr(metadata, "st_uid", None) != effective_uid:
        return False
    return os.name != "posix" or stat.S_IMODE(metadata.st_mode) == 0o600


def _recovery_path_for_output(output: Path) -> Path:
    return output.with_name(f".{output.name}.publication-recovery.json")


def _identity_to_json(identity: tuple[int, int, int]) -> list[int]:
    return list(identity)


def _identity_from_json(value: object) -> tuple[int, int, int]:
    if type(value) is not list or len(value) != 3 or any(type(part) is not int for part in value):
        raise ValueError
    return value[0], value[1], value[2]


def _snapshot_to_json(snapshot: _FixtureSetSnapshot) -> dict[str, object]:
    return {
        "directory_identity": _identity_to_json(snapshot.directory_identity),
        "files": [
            {
                "identity": _identity_to_json(file.identity),
                "name": file.name,
                "sha256": file.sha256,
                "size": file.size,
            }
            for file in snapshot.files
        ],
    }


def _snapshot_from_json(value: object) -> _FixtureSetSnapshot:
    if type(value) is not dict or set(value) != {"directory_identity", "files"}:
        raise ValueError
    directory_identity = _identity_from_json(value["directory_identity"])
    if directory_identity[2] != stat.S_IFDIR or type(value["files"]) is not list:
        raise ValueError

    files: list[_FixtureFileSnapshot] = []
    for item in value["files"]:
        if type(item) is not dict or set(item) != {"identity", "name", "sha256", "size"}:
            raise ValueError
        identity = _identity_from_json(item["identity"])
        if (
            identity[2] != stat.S_IFREG
            or type(item["name"]) is not str
            or type(item["size"]) is not int
            or item["size"] < 0
            or type(item["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValueError
        files.append(
            _FixtureFileSnapshot(
                name=item["name"],
                identity=identity,
                size=item["size"],
                sha256=item["sha256"],
            )
        )

    expected_names = sorted({"manifest.json", *(f"{category}.json" for category in CASE_NAMES)})
    if [file.name for file in files] != expected_names:
        raise ValueError
    return _FixtureSetSnapshot(
        directory_identity=directory_identity,
        files=tuple(files),
    )


def _fsync_directory(
    directory: Path,
    expected_identity: tuple[int, int, int],
    error_message: str,
) -> None:
    unsupported = {
        errno.EINVAL,
        getattr(errno, "ENOSYS", -1),
        getattr(errno, "ENOTSUP", -1),
        getattr(errno, "EOPNOTSUPP", -1),
    }
    _require_path_identity(directory, expected_identity, error_message)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        if os.name != "posix" or exc.errno in unsupported:
            return
        raise RuntimeError(error_message) from None
    descriptor_owner = _OwnedDescriptor(descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or _stable_metadata_identity(metadata) != expected_identity:
            raise RuntimeError(error_message)
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in unsupported:
                raise RuntimeError(error_message) from None
    except (OSError, RuntimeError):
        descriptor_owner.close_quietly()
        raise RuntimeError(error_message) from None
    except BaseException:
        descriptor_owner.close_quietly()
        raise
    else:
        try:
            descriptor_owner.close()
        except OSError:
            raise RuntimeError(error_message) from None
    _require_path_identity(directory, expected_identity, error_message)


def _read_recovery_file(path: Path) -> tuple[tuple[int, int, int], bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise RuntimeError(_RECOVERY_ERROR) from None
    descriptor_owner = _OwnedDescriptor(descriptor)
    try:
        metadata_before = os.fstat(descriptor)
        if not _valid_lock_file_metadata(metadata_before) or metadata_before.st_size > 131_072:
            raise RuntimeError(_RECOVERY_ERROR)
        identity = _stable_metadata_identity(metadata_before)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        metadata_after = os.fstat(descriptor)
        if _stable_metadata_identity(metadata_after) != identity or metadata_after.st_size != len(raw):
            raise RuntimeError(_RECOVERY_ERROR)
    except (OSError, RuntimeError):
        descriptor_owner.close_quietly()
        raise RuntimeError(_RECOVERY_ERROR) from None
    except BaseException:
        descriptor_owner.close_quietly()
        raise
    else:
        try:
            descriptor_owner.close()
        except OSError:
            raise RuntimeError(_RECOVERY_ERROR) from None
    _require_path_identity(path, identity, _RECOVERY_ERROR)
    return identity, raw


def _write_recovery_record(
    output: Path,
    backup: Path,
    parent_identity: tuple[int, int, int],
    output_snapshot: _FixtureSetSnapshot,
) -> _RecoveryRecord:
    path = _recovery_path_for_output(output)
    payload = {
        "backup_name": backup.name,
        "output_name": output.name,
        "output_snapshot": _snapshot_to_json(output_snapshot),
        "parent_identity": _identity_to_json(parent_identity),
        "schema_version": 1,
    }
    raw = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        _require_path_identity(output.parent, parent_identity, _RECOVERY_ERROR)
        descriptor = os.open(path, flags, 0o600)
    except (OSError, RuntimeError):
        raise RuntimeError(_RECOVERY_ERROR) from None
    descriptor_owner = _OwnedDescriptor(descriptor)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError
            offset += written
        metadata = os.fstat(descriptor)
        if not _valid_lock_file_metadata(metadata):
            raise RuntimeError(_RECOVERY_ERROR)
        record_identity = _stable_metadata_identity(metadata)
        os.fsync(descriptor)
    except (OSError, RuntimeError):
        descriptor_owner.close_quietly()
        raise RuntimeError(_RECOVERY_ERROR) from None
    except BaseException:
        descriptor_owner.close_quietly()
        raise
    else:
        try:
            descriptor_owner.close()
        except OSError:
            raise RuntimeError(_RECOVERY_ERROR) from None

    _require_path_identity(path, record_identity, _RECOVERY_ERROR)
    read_identity, read_raw = _read_recovery_file(path)
    if read_identity != record_identity or read_raw != raw:
        raise RuntimeError(_RECOVERY_ERROR)
    _fsync_directory(output.parent, parent_identity, _RECOVERY_ERROR)
    return _RecoveryRecord(
        path=path,
        identity=record_identity,
        raw=raw,
        parent_identity=parent_identity,
        backup=backup,
        output_snapshot=output_snapshot,
    )


def _load_recovery_record(output: Path) -> _RecoveryRecord | None:
    path = _recovery_path_for_output(output)
    if _path_identity_or_none(path, _RECOVERY_ERROR) is None:
        return None
    try:
        record_identity, raw = _read_recovery_file(path)
        payload = json.loads(raw.decode("ascii"), parse_constant=_invalid_json_constant)
        canonical = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
        if raw != canonical or type(payload) is not dict:
            raise ValueError
        if set(payload) != {
            "backup_name",
            "output_name",
            "output_snapshot",
            "parent_identity",
            "schema_version",
        }:
            raise ValueError
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["output_name"] != output.name
        ):
            raise ValueError
        backup_name = payload["backup_name"]
        if (
            type(backup_name) is not str
            or re.fullmatch(rf"\.{re.escape(output.name)}\.backup-[0-9a-f]{{32}}", backup_name) is None
        ):
            raise ValueError
        parent_identity = _identity_from_json(payload["parent_identity"])
        if parent_identity[2] != stat.S_IFDIR:
            raise ValueError
        output_snapshot = _snapshot_from_json(payload["output_snapshot"])
    except (OSError, UnicodeError, ValueError, RuntimeError):
        raise RuntimeError(_RECOVERY_ERROR) from None
    return _RecoveryRecord(
        path=path,
        identity=record_identity,
        raw=raw,
        parent_identity=parent_identity,
        backup=output.parent / backup_name,
        output_snapshot=output_snapshot,
    )


def _clear_recovery_record(record: _RecoveryRecord) -> None:
    _require_path_identity(record.path.parent, record.parent_identity, _RECOVERY_ERROR)
    identity, raw = _read_recovery_file(record.path)
    if identity != record.identity or raw != record.raw:
        raise RuntimeError(_RECOVERY_ERROR)
    try:
        record.path.unlink()
    except OSError:
        raise RuntimeError(_RECOVERY_ERROR) from None
    _require_path_absent(record.path, _RECOVERY_ERROR)
    _fsync_directory(record.path.parent, record.parent_identity, _RECOVERY_ERROR)


def _recover_interrupted_publication(output: Path) -> None:
    record = _load_recovery_record(output)
    if record is None:
        return
    try:
        _require_path_identity(output.parent, record.parent_identity, _RECOVERY_ERROR)
        output_identity = _path_identity_or_none(output, _RECOVERY_ERROR)
        backup_identity = _path_identity_or_none(record.backup, _RECOVERY_ERROR)
        if output_identity == record.output_snapshot.directory_identity and backup_identity is None:
            _require_fixture_set_snapshot(output, record.output_snapshot, _RECOVERY_ERROR)
            _clear_recovery_record(record)
            return
        if output_identity is not None or backup_identity != record.output_snapshot.directory_identity:
            raise RuntimeError(_RECOVERY_ERROR)
        _require_real_directory_identity(
            record.backup,
            record.output_snapshot.directory_identity,
            _RECOVERY_ERROR,
        )
        _require_fixture_set_snapshot(record.backup, record.output_snapshot, _RECOVERY_ERROR)
        record.backup.replace(output)
        _require_fixture_set_snapshot(output, record.output_snapshot, _RECOVERY_ERROR)
        _require_path_absent(record.backup, _RECOVERY_ERROR)
        _clear_recovery_record(record)
    except (OSError, RuntimeError):
        raise RuntimeError(_RECOVERY_ERROR) from None


def _close_descriptor_quietly(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except BaseException:  # noqa: BLE001 - cleanup must not replace an active failure
        pass


class _OwnedDescriptor:
    def __init__(self, descriptor: int) -> None:
        self._descriptor: int | None = descriptor

    def fileno(self) -> int:
        if self._descriptor is None:
            raise RuntimeError("Descriptor ownership has already been released")
        return self._descriptor

    def detach(self) -> int:
        descriptor = self.fileno()
        self._descriptor = None
        return descriptor

    def close(self) -> None:
        descriptor = self.detach()
        # An ambiguous close result must not leave ownership armed for a retry.
        os.close(descriptor)

    def close_quietly(self) -> None:
        try:
            self.close()
        except BaseException:  # noqa: BLE001 - cleanup must not replace an active failure
            pass


def _close_lock_file(lock_file: Any, descriptor_owner: _OwnedDescriptor) -> None:
    try:
        lock_file.close()
    except BaseException:  # noqa: BLE001 - preserve the file close failure
        descriptor_owner.close_quietly()
        raise
    descriptor_owner.close()


def _prepare_lock_root(lock_root: Path, source_root: Path) -> Path:
    try:
        if _is_link_like(lock_root):
            raise OSError
        lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if _is_link_like(lock_root) or not lock_root.is_dir():
            raise OSError
        resolved_lock_root = lock_root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise RuntimeError("Fixture publication lock root is invalid") from None

    if resolved_lock_root.is_relative_to(source_root):
        raise RuntimeError("Fixture publication lock root is invalid")
    return resolved_lock_root


def _open_lock_descriptor(lock_root: Path, lock_name: str) -> int:
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)

    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    use_directory_descriptor = (
        fcntl is not None
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and getattr(os, "O_DIRECTORY", 0) != 0
    )
    if not use_directory_descriptor:
        lock_path = lock_root / lock_name
        descriptor: int | None = None
        try:
            if _is_link_like(lock_root):
                raise RuntimeError("Fixture publication lock root is invalid")
            root_before = os.stat(lock_root, follow_symlinks=False)
            if not _valid_lock_root_metadata(root_before):
                raise RuntimeError("Fixture publication lock root is invalid")
            root_identity = _stable_metadata_identity(root_before)

            if _is_link_like(lock_path):
                raise RuntimeError("Fixture publication lock file is invalid")
            try:
                file_before = os.stat(lock_path, follow_symlinks=False)
            except FileNotFoundError:
                file_before = None
            file_before_identity = None
            if file_before is not None:
                if not _valid_lock_file_metadata(file_before):
                    raise RuntimeError("Fixture publication lock file is invalid")
                file_before_identity = _stable_metadata_identity(file_before)

            descriptor = os.open(lock_path, flags, 0o600)
            opened_file = os.fstat(descriptor)
            if not _valid_lock_file_metadata(opened_file):
                raise RuntimeError("Fixture publication lock file is invalid")
            opened_identity = _stable_metadata_identity(opened_file)
            if file_before_identity is not None and opened_identity != file_before_identity:
                raise RuntimeError("Fixture publication lock file is invalid")

            root_after = os.stat(lock_root, follow_symlinks=False)
            if _is_link_like(lock_root) or not _valid_lock_root_metadata(root_after):
                raise RuntimeError("Fixture publication lock root is invalid")
            if _stable_metadata_identity(root_after) != root_identity:
                raise RuntimeError("Fixture publication lock root is invalid")
            file_after = os.stat(lock_path, follow_symlinks=False)
            if _is_link_like(lock_path) or not _valid_lock_file_metadata(file_after):
                raise RuntimeError("Fixture publication lock file is invalid")
            if _stable_metadata_identity(file_after) != opened_identity:
                raise RuntimeError("Fixture publication lock file is invalid")
            return descriptor
        except RuntimeError:
            _close_descriptor_quietly(descriptor)
            raise
        except OSError:
            _close_descriptor_quietly(descriptor)
            raise RuntimeError("Fixture publication lock could not be opened") from None
        except BaseException:
            _close_descriptor_quietly(descriptor)
            raise

    try:
        root_owner = _OwnedDescriptor(os.open(lock_root, directory_flags))
    except OSError:
        raise RuntimeError("Fixture publication lock root is invalid") from None
    root_descriptor = root_owner.fileno()
    lock_owner: _OwnedDescriptor | None = None
    try:
        try:
            root_metadata = os.fstat(root_descriptor)
        except OSError:
            raise RuntimeError("Fixture publication lock root is invalid") from None
        if not _valid_lock_root_metadata(root_metadata):
            raise RuntimeError("Fixture publication lock root is invalid")
        _stable_metadata_identity(root_metadata)

        try:
            existing_lock = os.stat(
                lock_name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing_lock = None
        except OSError:
            raise RuntimeError("Fixture publication lock file is invalid") from None
        existing_identity = None
        if existing_lock is not None:
            if not _valid_lock_file_metadata(existing_lock):
                raise RuntimeError("Fixture publication lock file is invalid")
            existing_identity = _stable_metadata_identity(existing_lock)

        try:
            lock_owner = _OwnedDescriptor(os.open(lock_name, flags, 0o600, dir_fd=root_descriptor))
        except OSError:
            raise RuntimeError("Fixture publication lock could not be opened") from None
        lock_descriptor = lock_owner.fileno()

        try:
            opened_lock = os.fstat(lock_descriptor)
        except OSError:
            raise RuntimeError("Fixture publication lock file is invalid") from None
        if not _valid_lock_file_metadata(opened_lock):
            raise RuntimeError("Fixture publication lock file is invalid")
        opened_identity = _stable_metadata_identity(opened_lock)
        if existing_identity is not None and opened_identity != existing_identity:
            raise RuntimeError("Fixture publication lock file is invalid")
        try:
            current_lock = os.stat(
                lock_name,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            raise RuntimeError("Fixture publication lock file is invalid") from None
        if not _valid_lock_file_metadata(current_lock):
            raise RuntimeError("Fixture publication lock file is invalid")
        if _stable_metadata_identity(current_lock) != opened_identity:
            raise RuntimeError("Fixture publication lock file is invalid")
    except BaseException:
        if lock_owner is not None:
            lock_owner.close_quietly()
        root_owner.close_quietly()
        raise

    try:
        root_owner.close()
    except OSError:
        lock_owner.close_quietly()
        raise RuntimeError("Fixture publication lock root could not be closed") from None
    except BaseException:
        lock_owner.close_quietly()
        raise
    return lock_owner.detach()


def _acquire_file_lock(lock_file: Any) -> None:
    if fcntl is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError:
            raise RuntimeError("Fixture publication lock could not be acquired") from None
        return

    if msvcrt is not None:  # pragma: no cover - exercised on Windows
        try:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            while True:
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                except OSError as exc:
                    if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        continue
                    raise
                else:
                    return
        except OSError:
            raise RuntimeError("Fixture publication lock could not be acquired") from None

    raise RuntimeError("Interprocess fixture publication locking is unavailable")


def _release_file_lock(lock_file: Any) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no cover - exercised on Windows
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _publication_lock(output: Path, source_root: Path) -> Iterator[None]:
    """Serialize cooperating local publishers; this is not a same-user privilege boundary."""
    lock_path = _lock_path_for_output(output)
    lock_root = _prepare_lock_root(lock_path.parent, source_root)
    descriptor_owner = _OwnedDescriptor(_open_lock_descriptor(lock_root, lock_path.name))

    try:
        lock_file = os.fdopen(
            descriptor_owner.fileno(),
            "r+b",
            buffering=0,
            closefd=False,
        )
    except BaseException:
        descriptor_owner.close_quietly()
        raise

    try:
        _acquire_file_lock(lock_file)
        try:
            yield
        except BaseException:
            try:
                _release_file_lock(lock_file)
            except BaseException:  # noqa: BLE001 - preserve the exact body exception
                pass
            raise
        else:
            try:
                _release_file_lock(lock_file)
            except OSError:
                raise RuntimeError("Fixture publication lock could not be released") from None
    except BaseException:
        try:
            _close_lock_file(lock_file, descriptor_owner)
        except BaseException:  # noqa: BLE001 - preserve the active lock failure
            pass
        raise
    else:
        try:
            _close_lock_file(lock_file, descriptor_owner)
        except OSError:
            raise RuntimeError("Fixture publication lock file could not be closed") from None


@contextmanager
def fixture_publication_reader(
    output: Path,
    *,
    source_root: Path | None = None,
) -> Iterator[Path]:
    """Hold the publication lock while a cooperating reader loads a complete fixture set."""
    try:
        resolved_source_root = (source_root or REPO_ROOT).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("source-root could not be resolved") from None
    resolved_output = _resolve_output_path(output, resolved_source_root)
    with _publication_lock(resolved_output, resolved_source_root):
        yield resolved_output


def _replace_output_directory(
    staging: Path,
    output: Path,
    *,
    expected_parent_identity: tuple[int, int, int] | None = None,
    expected_staging_identity: tuple[int, int, int] | None = None,
) -> None:
    """Cooperatively publish with content checks, rollback, and durable crash recovery."""
    staging_snapshot = _validate_fixture_set(staging, predecessor_commit=None)
    publication_error = "Fixture output parent changed during publication"
    if expected_parent_identity is None:
        parent_identity = _path_identity(output.parent, publication_error)
    else:
        parent_identity = expected_parent_identity
        _require_path_identity(output.parent, parent_identity, publication_error)
    staging_error = "Fixture staging directory changed during publication"
    if expected_staging_identity is None:
        staging_identity = _path_identity(staging, staging_error)
    else:
        staging_identity = expected_staging_identity
        _require_path_identity(staging, staging_identity, staging_error)
    _require_real_directory_identity(staging, staging_identity, staging_error)
    output_snapshot = _validate_existing_output(output)
    output_identity = output_snapshot.directory_identity if output_snapshot is not None else None
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}") if output_snapshot is not None else None
    recovery_record: _RecoveryRecord | None = None
    rename_started = False
    staging_rename_completed = False
    try:
        if output_snapshot is not None and output_identity is not None and backup is not None:
            _require_path_identity(
                output.parent,
                parent_identity,
                "Fixture output parent changed during publication",
            )
            _require_path_identity(
                output,
                output_identity,
                "Fixture output changed during publication",
            )
            _require_fixture_set_snapshot(
                output,
                output_snapshot,
                "Fixture output changed during publication",
            )
            _require_path_absent(
                backup,
                "Fixture output backup changed during publication",
            )
            _require_path_identity(staging, staging_identity, staging_error)
            _require_fixture_set_snapshot(staging, staging_snapshot, staging_error)
            recovery_record = _write_recovery_record(
                output,
                backup,
                parent_identity,
                output_snapshot,
            )
            _require_path_identity(
                output.parent,
                parent_identity,
                "Fixture output parent changed during publication",
            )
            _require_path_absent(
                backup,
                "Fixture output backup changed during publication",
            )
            _require_path_identity(staging, staging_identity, staging_error)
            _require_fixture_set_snapshot(staging, staging_snapshot, staging_error)
            _require_path_identity(
                output,
                output_identity,
                "Fixture output changed during publication",
            )
            _require_fixture_set_snapshot(
                output,
                output_snapshot,
                "Fixture output changed during publication",
            )
            rename_started = True
            output.replace(backup)
            _require_path_identity(
                output.parent,
                parent_identity,
                "Fixture output parent changed during publication",
            )
            _require_path_absent(
                output,
                "Fixture output changed during publication",
            )
            _require_path_identity(
                backup,
                output_identity,
                "Fixture output backup changed during publication",
            )

        _require_path_identity(
            output.parent,
            parent_identity,
            "Fixture output parent changed during publication",
        )
        _require_path_absent(
            output,
            "Fixture output changed during publication",
        )
        if output_snapshot is not None and output_identity is not None and backup is not None:
            _require_path_identity(
                backup,
                output_identity,
                "Fixture output backup changed during publication",
            )
            _require_fixture_set_snapshot(
                backup,
                output_snapshot,
                "Fixture output backup changed during publication",
            )
        _require_path_identity(
            staging,
            staging_identity,
            "Fixture staging directory changed during publication",
        )
        _require_fixture_set_snapshot(staging, staging_snapshot, staging_error)
        staging.replace(output)
        staging_rename_completed = True
        _require_path_identity(
            output.parent,
            parent_identity,
            "Fixture output parent changed during publication",
        )
        _require_path_identity(
            output,
            staging_identity,
            "Fixture output changed during publication",
        )
        _require_fixture_set_snapshot(
            output,
            staging_snapshot,
            "Fixture output changed during publication",
        )
        if recovery_record is not None:
            _clear_recovery_record(recovery_record)
    except BaseException:
        if rename_started and output_identity is not None and backup is not None:
            _restore_output_backup(
                backup,
                output,
                parent_identity,
                output_snapshot,
            )
            if recovery_record is not None:
                _clear_recovery_record(recovery_record)
        elif recovery_record is not None:
            try:
                _clear_recovery_record(recovery_record)
            except (OSError, RuntimeError):
                raise RuntimeError(_RECOVERY_ERROR) from None
        elif staging_rename_completed:
            raise RuntimeError(_RECOVERY_ERROR) from None
        raise
    else:
        if output_identity is not None and backup is not None:
            try:
                _require_path_identity(
                    output.parent,
                    parent_identity,
                    "Fixture output parent changed during cleanup",
                )
                _require_path_identity(
                    output,
                    staging_identity,
                    "Fixture output changed during cleanup",
                )
                _require_path_identity(
                    backup,
                    output_identity,
                    "Fixture output backup changed during cleanup",
                )
            except RuntimeError:
                _report_backup_retained(backup)
                return
            try:
                shutil.rmtree(backup)
            except OSError:
                _report_backup_retained(backup)


def _cleanup_staging_directory(
    staging: Path,
    parent_identity: tuple[int, int, int],
    staging_identity: tuple[int, int, int],
) -> None:
    try:
        _require_path_identity(staging.parent, parent_identity, _STAGING_CLEANUP_ERROR)
        current_identity = _path_identity_or_none(staging, _STAGING_CLEANUP_ERROR)
        if current_identity is None:
            return
        if current_identity != staging_identity:
            raise RuntimeError(_STAGING_CLEANUP_ERROR)
        shutil.rmtree(staging)
    except (OSError, RuntimeError):
        raise RuntimeError(_STAGING_CLEANUP_ERROR) from None


def generate_fixtures(
    predecessor_commit: str,
    output: Path,
    *,
    source_root: Path | None = None,
) -> None:
    resolved_source_root = _validate_source_root(
        predecessor_commit,
        source_root or REPO_ROOT,
    )
    output = _resolve_output_path(output, resolved_source_root)
    with _publication_lock(output, resolved_source_root):
        if output.parent.exists():
            _recover_interrupted_publication(output)
        _validate_existing_output(output)
        try:
            payloads = build_case_payloads(resolved_source_root)
        finally:
            _validate_source_root(predecessor_commit, resolved_source_root)

        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            staging_parent_identity = _path_identity(output.parent, _STAGING_CLEANUP_ERROR)
        except (OSError, RuntimeError):
            raise RuntimeError(_STAGING_CLEANUP_ERROR) from None
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=output.parent,
            )
        )
        try:
            staging_identity = _path_identity(staging, _STAGING_CLEANUP_ERROR)
            _require_real_directory_identity(
                staging,
                staging_identity,
                _STAGING_CLEANUP_ERROR,
            )
        except (OSError, RuntimeError):
            _report_staging_retained()
            raise RuntimeError(_STAGING_CLEANUP_ERROR) from None
        except BaseException:
            _report_staging_retained()
            raise
        try:
            _write_fixture_set(staging, predecessor_commit, payloads)
            _validate_fixture_set(staging, predecessor_commit)
            _replace_output_directory(
                staging,
                output,
                expected_parent_identity=staging_parent_identity,
                expected_staging_identity=staging_identity,
            )
        except BaseException:
            cleanup_failed = False
            try:
                _cleanup_staging_directory(
                    staging,
                    staging_parent_identity,
                    staging_identity,
                )
            except BaseException:  # noqa: BLE001 - preserve any primary BaseException
                cleanup_failed = True
            if cleanup_failed:
                _report_staging_retained()
            raise
        else:
            _cleanup_staging_directory(
                staging,
                staging_parent_identity,
                staging_identity,
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        generate_fixtures(
            args.predecessor_commit,
            args.output,
            source_root=args.source_root,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
