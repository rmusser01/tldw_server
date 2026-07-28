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
import subprocess  # nosec B404
import sys
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
    "selectors",
)

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
def _predecessor_modules(source_root: Path) -> Iterator[tuple[Any, Any, Any, Any]]:
    previous_modules = _remove_loaded_production_modules()
    previous_path = list(sys.path)
    sys.path.insert(0, str(source_root))
    try:
        from tldw_Server_API.app.core.Watchlists import fetchers
        from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
        from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, PolicyDecision

        _assert_production_modules_under(source_root)
        yield article, fetchers, FetchResponse, PolicyDecision
        _assert_production_modules_under(source_root)
    finally:
        _remove_loaded_production_modules()
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


def build_case_payloads(source_root: Path) -> dict[str, dict[str, Any]]:
    with _predecessor_modules(source_root) as modules:
        article, fetchers, FetchResponse, PolicyDecision = modules
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


def _validate_fixture_set(output: Path, predecessor_commit: str | None) -> None:
    case_files = {category: f"{category}.json" for category in CASE_NAMES}
    expected_names = {"manifest.json", *case_files.values()}
    try:
        entries = list(output.iterdir())
    except OSError:
        raise RuntimeError("Fixture set could not be inspected") from None

    actual_names = {path.name for path in entries}
    if actual_names != expected_names:
        raise RuntimeError("Fixture set does not contain the exact required files")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise RuntimeError("Fixture set entries must be regular files")

    decoded: dict[str, object] = {}
    for path in sorted(entries):
        try:
            raw = path.read_bytes()
            payload = json.loads(
                raw.decode("ascii"),
                parse_constant=_invalid_json_constant,
            )
        except (OSError, UnicodeError, ValueError):
            raise RuntimeError(f"Fixture set contains invalid JSON: {path.name}") from None
        canonical = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
        if raw != canonical:
            raise RuntimeError(f"Fixture set contains noncanonical JSON: {path.name}")
        decoded[path.name] = payload

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


def _validate_existing_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError("existing output must be a directory")
        _validate_fixture_set(output, predecessor_commit=None)


def _lock_path_for_output(output: Path) -> Path:
    normalized_output = os.path.normcase(str(output.resolve()))
    identity = hashlib.sha256(normalized_output.encode("utf-8")).hexdigest()
    lock_root = Path(tempfile.gettempdir()).resolve() / "tldw-phase4-fixture-locks"
    return lock_root / f"{identity}.lock"


def _prepare_lock_root(lock_root: Path, source_root: Path) -> Path:
    try:
        lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if lock_root.is_symlink() or not lock_root.is_dir():
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
        fcntl is not None and os.open in os.supports_dir_fd and getattr(os, "O_DIRECTORY", 0) != 0
    )
    if not use_directory_descriptor:
        try:
            return os.open(lock_root / lock_name, flags, 0o600)
        except OSError:
            raise RuntimeError("Fixture publication lock could not be opened") from None

    try:
        root_descriptor = os.open(lock_root, directory_flags)
    except OSError:
        raise RuntimeError("Fixture publication lock root is invalid") from None
    try:
        try:
            return os.open(lock_name, flags, 0o600, dir_fd=root_descriptor)
        except OSError:
            raise RuntimeError("Fixture publication lock could not be opened") from None
    finally:
        try:
            os.close(root_descriptor)
        except OSError:
            pass


def _acquire_file_lock(lock_file: Any) -> None:
    if fcntl is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError:
            raise RuntimeError("Fixture publication lock could not be acquired") from None
        return

    if msvcrt is not None:  # pragma: no cover - exercised on Windows
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        while True:
            lock_file.seek(0)
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise RuntimeError("Fixture publication lock could not be acquired") from None
            else:
                return

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
    lock_path = _lock_path_for_output(output)
    lock_root = _prepare_lock_root(lock_path.parent, source_root)
    descriptor = _open_lock_descriptor(lock_root, lock_path.name)

    with os.fdopen(descriptor, "r+b", buffering=0) as lock_file:
        _acquire_file_lock(lock_file)
        try:
            yield
        except BaseException:
            try:
                _release_file_lock(lock_file)
            except OSError:
                pass
            raise
        else:
            try:
                _release_file_lock(lock_file)
            except OSError:
                raise RuntimeError("Fixture publication lock could not be released") from None


def _replace_output_directory(staging: Path, output: Path) -> None:
    _validate_fixture_set(staging, predecessor_commit=None)
    backup: Path | None = None
    if output.exists():
        _validate_existing_output(output)
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        output.replace(backup)

    try:
        staging.replace(output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    else:
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError:
                print(
                    "warning: fixture output committed; backup cleanup failed; "
                    f"backup retained as {backup.name!r} for manual cleanup",
                    file=sys.stderr,
                )


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
        _validate_existing_output(output)
        try:
            payloads = build_case_payloads(resolved_source_root)
        finally:
            _validate_source_root(predecessor_commit, resolved_source_root)

        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=output.parent,
            )
        )
        try:
            _write_fixture_set(staging, predecessor_commit, payloads)
            _validate_fixture_set(staging, predecessor_commit)
            _replace_output_directory(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


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
