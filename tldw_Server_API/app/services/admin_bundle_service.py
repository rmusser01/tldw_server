# admin_bundle_service.py
# Description: Service layer for admin backup bundle operations.
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Callable
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Backups import (
    PostgresToolTimeout,
    _sqlite_error_is_busy,
    validate_postgres_backup,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    get_current_media_schema_version,
)
from tldw_Server_API.app.core.exceptions import (
    BundleConcurrencyError,
    BundleDiskSpaceError,
    BundleExportError,
    BundleImportError,
    BundleNotFoundError,
    BundleRateLimitError,
    BundleSchemaIncompatibleError,
    UnsupportedBackupBackendError,
)
from tldw_Server_API.app.core.Utils.path_utils import safe_join
from tldw_Server_API.app.services.admin_data_ops_service import (
    _backup_base_dir,
    _postgresql_database_config,
    _require_supported_backup_storage,
    _resolve_dataset_db_path,
    _validate_backup_dataset,
    create_backup_snapshot,
    restore_backup_file,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MANIFEST_VERSION = 1
_ALL_DATASETS = ("media", "chacha", "prompts", "evaluations", "audit", "authnz")
_PER_USER_DATASETS = frozenset(_ALL_DATASETS) - {"authnz"}
_POSTGRESQL_COVERAGE = ["all_database_sql_objects_and_rows", "all_users", "jobs"]
_POSTGRESQL_EXCLUSIONS = ["local_sqlite", "uploaded_files", "models", "vector_services", "cluster_roles"]

# Concurrency: only one bundle operation at a time
_bundle_lock = threading.Lock()

# Rate limiting: 5 operations per hour per (user_id, op_type)
_rate_limit_windows: dict[tuple[int, str], list[float]] = {}
_RATE_LIMIT_MAX = 5
_RATE_LIMIT_WINDOW_SECONDS = 3600


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BundleMetadata:
    bundle_id: str
    user_id: int | None
    created_at: datetime
    size_bytes: int
    datasets: tuple[str, ...]
    schema_versions: MappingProxyType[str, int | None]
    app_version: str | None
    manifest_version: int
    notes: str | None
    recovery_scope: str | None = None

    @property
    def requires_platform_admin(self) -> bool:
        return is_postgresql_bundle(
            {
                "manifest_version": self.manifest_version,
                "datasets": list(self.datasets),
                "recovery_scope": self.recovery_scope,
            }
        )


# ---------------------------------------------------------------------------
# Schema version registry
# ---------------------------------------------------------------------------
def _get_schema_versions() -> dict[str, int | None]:
    """Collect current schema versions from DB modules (lazy imports)."""
    versions: dict[str, int | None] = {}
    try:
        versions["media"] = get_current_media_schema_version()
    except Exception:
        versions["media"] = None
    try:
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
        versions["chacha"] = int(CharactersRAGDB._CURRENT_SCHEMA_VERSION)
    except Exception:
        versions["chacha"] = None
    try:
        from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
        versions["prompts"] = int(PromptsDatabase._CURRENT_SCHEMA_VERSION)
    except Exception:
        versions["prompts"] = None
    # Evaluations, audit, authnz may not expose a version constant
    for name in ("evaluations", "audit", "authnz"):
        versions.setdefault(name, None)
    return versions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _preflight_backup_storage(
    datasets: list[str], error_type: type[BundleExportError] | type[BundleImportError]
) -> None:
    """Reject the whole request before an earlier dataset can be changed/pruned."""
    for dataset in datasets:
        try:
            _require_supported_backup_storage(_validate_backup_dataset(dataset))
        except UnsupportedBackupBackendError as exc:
            raise error_type(str(exc), error_code="unsupported_backup_backend") from exc


def _bundle_base_dir() -> str:
    base = _backup_base_dir()
    return os.path.join(base, "bundles")


def _check_rate_limit(user_id: int, op_type: str) -> None:
    """Verify the rate limit has not been exceeded (does NOT record usage)."""
    key = (user_id, op_type)
    now = time.monotonic()
    window = _rate_limit_windows.get(key, [])
    # Prune old entries
    window = [t for t in window if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    _rate_limit_windows[key] = window
    if len(window) >= _RATE_LIMIT_MAX:
        oldest = min(window)
        retry_after = int(_RATE_LIMIT_WINDOW_SECONDS - (now - oldest)) + 1
        exc = BundleRateLimitError("rate_limit_exceeded")
        exc.retry_after = retry_after  # type: ignore[attr-defined]
        raise exc


def _record_rate_limit(user_id: int, op_type: str) -> None:
    """Record a successful operation against the rate limit window."""
    key = (user_id, op_type)
    now = time.monotonic()
    window = _rate_limit_windows.get(key, [])
    window.append(now)
    _rate_limit_windows[key] = window


def _check_disk_space(path: str, required_bytes: int) -> None:
    os.makedirs(path, exist_ok=True)
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        raise BundleDiskSpaceError("insufficient_disk_space")


def _compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _check_import_disk_space(
    *,
    file_path: str,
    required_bytes: int,
    datasets: list[str],
    user_id: int | None,
) -> None:
    """Verify free disk space for all import write targets.

    Import writes can touch:
    - configured backup root (contained staging and safety snapshots),
    - system temp dir,
    - upload/temp file directory,
    - live DB directories for datasets in the manifest.
    """
    check_paths: set[str] = {
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath(_backup_base_dir()),
        os.path.realpath(os.path.dirname(file_path) or _bundle_base_dir()),
    }

    for ds in datasets:
        ds_user_id = None if ds == "authnz" else user_id
        try:
            live_db_path, _ = _resolve_dataset_db_path(ds, ds_user_id)
        except Exception as exc:
            logger.warning(
                "Skipping live DB disk-space precheck for dataset {}: {}",
                ds,
                exc,
            )
            continue
        # A PostgreSQL DSN is not a local write target or a server-disk check.
        if ds == "authnz" and str(live_db_path).startswith("postgres"):
            continue
        live_dir = os.path.dirname(live_db_path) or "."
        check_paths.add(os.path.realpath(live_dir))

    for path in sorted(check_paths):
        _check_disk_space(path, required_bytes)


def _get_app_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("tldw_Server_API")
    except Exception as metadata_error:
        logger.bind(error_type=type(metadata_error).__name__).debug(
            "Failed to resolve app version from package metadata"
        )
    try:
        import tomllib
        pyproject = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "pyproject.toml",
        )
        if os.path.isfile(pyproject):
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return data.get("project", {}).get("version")
    except Exception as pyproject_error:
        logger.bind(error_type=type(pyproject_error).__name__).debug(
            "Failed to resolve app version from pyproject"
        )
    return None


def _estimate_total_db_size(datasets: list[str], user_id: int | None) -> int:
    total = 0
    for ds in datasets:
        try:
            db_path, _ = _resolve_dataset_db_path(ds, user_id)
            if os.path.isfile(db_path):
                total += os.path.getsize(db_path)
        except Exception as size_error:
            logger.bind(error_type=type(size_error).__name__).debug(
                "Failed to estimate dataset DB size for {}",
                ds,
            )
    return max(total, 1024)  # at least 1 KB estimate


def _build_manifest(
    *,
    user_id: int | None,
    datasets: list[str],
    files: dict[str, dict[str, str]],
    schema_versions: dict[str, int | None],
    notes: str | None,
    retention_hours: int | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": _MANIFEST_VERSION,
        "app_version": _get_app_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "datasets": datasets,
        "files": files,
        "schema_versions": schema_versions,
        "notes": notes,
        "platform": {
            "os": platform.system(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
    }
    if retention_hours is not None:
        manifest["retention_hours"] = retention_hours
    return manifest


def _validate_bundle_id(bundle_id: str) -> str:
    """Sanitize bundle_id to prevent path traversal."""
    name = os.path.basename(str(bundle_id or "").strip())
    if not name or name != bundle_id or name.startswith("-"):
        raise BundleNotFoundError("bundle_not_found")
    if not name.endswith(".zip"):
        raise BundleNotFoundError("bundle_not_found")
    return name


def _resolve_bundle_path(bundle_id: str) -> str:
    name = _validate_bundle_id(bundle_id)
    path = os.path.join(_bundle_base_dir(), name)
    if not os.path.isfile(path):
        raise BundleNotFoundError("bundle_not_found")
    return path


def _sidecar_path(zip_path: str) -> str:
    """Return the path to the sidecar manifest cache for a bundle ZIP."""
    return zip_path + ".manifest.json"


def _write_sidecar_manifest(zip_path: str, manifest: dict[str, Any]) -> None:
    """Write manifest data to a sidecar JSON file for fast listing."""
    try:
        with open(_sidecar_path(zip_path), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        pass  # non-critical


def _read_manifest_cached(zip_path: str) -> dict[str, Any]:
    """Read manifest from sidecar cache first, falling back to the ZIP."""
    sc = _sidecar_path(zip_path)
    if os.path.isfile(sc):
        try:
            with open(sc, encoding="utf-8") as f:
                return json.loads(f.read())
        except Exception as sidecar_error:
            logger.bind(error_type=type(sidecar_error).__name__).debug(
                "Failed to read bundle sidecar manifest; falling back to ZIP"
            )
    return _read_manifest_from_zip(zip_path)


def _read_manifest_from_zip(zip_path: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            posix, windows = PurePosixPath(name), PureWindowsPath(name)
            if posix.is_absolute() or windows.drive or ".." in posix.parts or ".." in windows.parts:
                raise BundleImportError("path_traversal_detected", error_code="path_traversal_detected")
        if "manifest.json" not in zf.namelist():
            raise BundleImportError(
                "missing_manifest", error_code="missing_manifest"
            )
        with zf.open("manifest.json") as mf:
            return json.loads(mf.read())


def _prune_expired_bundles(
    *,
    retention_hours: int,
    user_id: int | None,
    exclude_bundle_ids: set[str] | None = None,
    recovery_scope: str | None = None,
) -> int:
    """Delete expired bundles for a specific bundle scope.

    Scope is limited to bundles whose manifest user_id matches the provided
    user_id (including ``None`` for global bundles). Bundles with unreadable
    manifests are skipped safely and left untouched.
    """
    if retention_hours <= 0:
        return 0

    base = _bundle_base_dir()
    if not os.path.isdir(base):
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    excluded = exclude_bundle_ids or set()
    removed = 0

    for entry in os.scandir(base):
        if (
            not entry.is_file(follow_symlinks=False)
            or entry.is_symlink()
            or not entry.name.endswith(".zip")
            or entry.name in excluded
        ):
            continue

        try:
            manifest = _read_manifest_from_zip(entry.path)
        except Exception as exc:
            logger.warning(
                "Skipping bundle retention cleanup for unreadable manifest {}: {}",
                entry.name,
                exc,
            )
            continue

        if manifest.get("user_id") != user_id or manifest.get("recovery_scope") != recovery_scope:
            continue

        created_raw = manifest.get("created_at")
        if not isinstance(created_raw, str):
            logger.warning(
                "Skipping bundle retention cleanup for {}: invalid created_at",
                entry.name,
            )
            continue

        try:
            created_at = datetime.fromisoformat(created_raw)
        except Exception as exc:
            logger.warning(
                "Skipping bundle retention cleanup for {}: invalid created_at ({})",
                entry.name,
                exc,
            )
            continue

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)

        if created_at >= cutoff:
            continue

        try:
            os.remove(entry.path)
            removed += 1
        except OSError as exc:
            logger.warning("Failed to prune bundle {}: {}", entry.path, exc)
            continue

        sc = _sidecar_path(entry.path)
        if os.path.isfile(sc):
            try:
                os.remove(sc)
            except OSError:
                pass

    return removed


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def create_bundle(
    *,
    datasets: list[str] | None,
    user_id: int | None,
    include_vector_store: bool = False,
    notes: str | None = None,
    max_backups: int | None = None,
    retention_hours: int | None = None,
    admin_user_id: int = 0,
) -> BundleMetadata:
    """Create a backup bundle ZIP containing snapshots of the requested datasets."""
    if include_vector_store:
        raise BundleExportError(
            "vector_store_export_not_supported",
            error_code="vector_store_export_not_supported",
        )

    effective_datasets = [_validate_backup_dataset(ds) for ds in (datasets or _ALL_DATASETS)]
    database_scope = effective_datasets == ["postgresql"]
    if any(ds not in _ALL_DATASETS for ds in effective_datasets) and not database_scope:
        raise BundleExportError("unsupported_bundle_dataset", error_code="unsupported_bundle_dataset")
    if database_scope and user_id is not None:
        raise BundleExportError("user_id_forbidden", error_code="user_id_forbidden")
    if len(set(effective_datasets)) != len(effective_datasets):
        raise BundleExportError("duplicate_datasets", error_code="duplicate_datasets")

    _preflight_backup_storage(effective_datasets, BundleExportError)

    # Validate user_id requirement
    per_user_requested = [ds for ds in effective_datasets if ds in _PER_USER_DATASETS]
    if per_user_requested and user_id is None:
        # Auto-resolve in single-user mode
        try:
            user_id = DatabasePaths.get_single_user_id()
        except Exception as exc:
            raise BundleExportError("user_id_required", error_code="user_id_required") from exc

    # Rate limit
    _check_rate_limit(admin_user_id, "export")

    # Estimate size and check disk space
    estimated = _estimate_total_db_size(effective_datasets, user_id)
    dest_dir = _bundle_base_dir()
    _check_disk_space(dest_dir, estimated * 2)

    staging_dir = tempfile.mkdtemp(prefix="tldw_bundle_")
    try:
        files_meta: dict[str, dict[str, str]] = {}

        for ds in effective_datasets:
            ds_user_id = None if ds in {"authnz", "postgresql"} else user_id
            try:
                backup_file = create_backup_snapshot(
                    dataset=ds,
                    user_id=ds_user_id,
                    backup_type="full",
                    max_backups=max_backups,
                )
            except Exception as exc:
                raise BundleExportError(f"backup_failed:{ds}: {exc}") from exc

            # Copy to staging dir
            dest = os.path.join(staging_dir, backup_file.filename)
            shutil.copy2(backup_file.path, dest)

            sha256 = _compute_sha256(dest)
            files_meta[backup_file.filename] = {
                "dataset": ds,
                "sha256": sha256,
                "hash_algorithm": "sha256",
                "size_bytes": str(backup_file.size_bytes),
            }

        schema_versions = _get_schema_versions()
        manifest = _build_manifest(
            user_id=user_id,
            datasets=effective_datasets,
            files=files_meta,
            schema_versions=schema_versions,
            notes=notes,
            retention_hours=retention_hours,
        )

        if database_scope:
            manifest.update(
                manifest_version=2,
                recovery_scope="postgresql_database",
                coverage=list(_POSTGRESQL_COVERAGE),
                exclusions=list(_POSTGRESQL_EXCLUSIONS),
            )

        manifest_path = os.path.join(staging_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Build ZIP filename
        user_label = str(user_id) if user_id else "global"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        zip_name = (
            f"tldw-backup-bundle-postgresql-{timestamp}-{uuid4().hex}.zip"
            if database_scope
            else f"tldw-backup-bundle-{user_label}-{timestamp}.zip"
        )
        zip_path = os.path.join(staging_dir, zip_name)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, "manifest.json")
            for filename in files_meta:
                zf.write(os.path.join(staging_dir, filename), filename)

        # Move to final location
        os.makedirs(dest_dir, exist_ok=True)
        final_path = os.path.join(dest_dir, zip_name)
        shutil.move(zip_path, final_path)

        # Write sidecar manifest cache for fast listing
        _write_sidecar_manifest(final_path, manifest)

        if retention_hours is not None:
            try:
                pruned = _prune_expired_bundles(
                    retention_hours=retention_hours,
                    user_id=user_id,
                    exclude_bundle_ids={zip_name},
                    recovery_scope="postgresql_database" if database_scope else None,
                )
                if pruned > 0:
                    logger.info(
                        "Pruned {} expired bundle(s) for user_id={}",
                        pruned,
                        user_id,
                    )
            except Exception as exc:
                logger.warning("Bundle retention cleanup failed: {}", exc)

        stat = os.stat(final_path)
        _record_rate_limit(admin_user_id, "export")
        return BundleMetadata(
            bundle_id=zip_name,
            user_id=user_id,
            created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            size_bytes=int(stat.st_size),
            datasets=tuple(effective_datasets),
            schema_versions=MappingProxyType(schema_versions),
            app_version=manifest.get("app_version"),
            manifest_version=manifest["manifest_version"],
            notes=notes,
            recovery_scope=manifest.get("recovery_scope"),
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


async def create_bundle_async(
    *,
    datasets: list[str] | None,
    user_id: int | None,
    include_vector_store: bool = False,
    notes: str | None = None,
    max_backups: int | None = None,
    retention_hours: int | None = None,
    admin_user_id: int = 0,
) -> BundleMetadata:
    """Async wrapper that runs create_bundle under an atomic non-blocking lock."""
    return await _run_bundle_operation_with_lock(
        create_bundle,
        datasets=datasets,
        user_id=user_id,
        include_vector_store=include_vector_store,
        notes=notes,
        max_backups=max_backups,
        retention_hours=retention_hours,
        admin_user_id=admin_user_id,
    )


def is_postgresql_bundle(manifest: dict[str, Any]) -> bool:
    """Treat every database-scope claim as platform-only, including malformed ones."""
    datasets = manifest.get("datasets")
    return (
        manifest.get("manifest_version") == 2
        or manifest.get("recovery_scope") is not None
        or (isinstance(datasets, list) and any(str(ds).strip().lower() == "postgresql" for ds in datasets))
    )


def list_bundles(
    *,
    user_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
    include_database: bool = True,
) -> tuple[list[BundleMetadata], int]:
    """List available bundles, optionally filtered by user_id."""
    base = _bundle_base_dir()
    if not os.path.isdir(base):
        return [], 0

    items: list[BundleMetadata] = []
    for entry in os.scandir(base):
        if not entry.is_file() or not entry.name.endswith(".zip"):
            continue
        try:
            manifest = _read_manifest_from_zip(entry.path)
        except Exception:
            logger.warning("Skipping unreadable bundle: {}", entry.name)
            continue

        if not include_database and is_postgresql_bundle(manifest):
            continue
        bundle_user_id = manifest.get("user_id")
        if user_id is not None and bundle_user_id != user_id:
            continue

        stat = entry.stat()
        items.append(
            BundleMetadata(
                bundle_id=entry.name,
                user_id=bundle_user_id,
                created_at=datetime.fromisoformat(manifest["created_at"])
                if manifest.get("created_at")
                else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                size_bytes=int(stat.st_size),
                datasets=tuple(manifest.get("datasets", [])),
                schema_versions=MappingProxyType(manifest.get("schema_versions", {})),
                app_version=manifest.get("app_version"),
                manifest_version=manifest.get("manifest_version", 0),
                notes=manifest.get("notes"),
                recovery_scope=manifest.get("recovery_scope"),
            )
        )

    items.sort(key=lambda m: m.created_at, reverse=True)
    total = len(items)
    return items[offset : offset + limit], total


def get_bundle_metadata(bundle_id: str) -> BundleMetadata:
    """Get metadata for a single bundle."""
    path = _resolve_bundle_path(bundle_id)
    manifest = _read_manifest_from_zip(path)
    stat = os.stat(path)
    return BundleMetadata(
        bundle_id=bundle_id,
        user_id=manifest.get("user_id"),
        created_at=datetime.fromisoformat(manifest["created_at"])
        if manifest.get("created_at")
        else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        size_bytes=int(stat.st_size),
        datasets=tuple(manifest.get("datasets", [])),
        schema_versions=MappingProxyType(manifest.get("schema_versions", {})),
        app_version=manifest.get("app_version"),
        manifest_version=manifest.get("manifest_version", 0),
        notes=manifest.get("notes"),
        recovery_scope=manifest.get("recovery_scope"),
    )


def get_bundle_path(bundle_id: str) -> str:
    """Resolve and return the filesystem path for a bundle."""
    return _resolve_bundle_path(bundle_id)


def delete_bundle(bundle_id: str) -> None:
    """Delete a bundle ZIP and its sidecar manifest from disk."""
    path = _resolve_bundle_path(bundle_id)
    os.remove(path)
    sc = _sidecar_path(path)
    if os.path.isfile(sc):
        try:
            os.remove(sc)
        except OSError:
            pass


@contextmanager
def _validated_postgresql_archive(
    file_path: str,
    manifest: dict[str, Any],
    user_id: int | None,
    *,
    staging_parent: str | None = None,
    deadline: float | None = None,
):
    """Stage and inspect the exact database dump; retain files if a child cannot stop."""
    deadline = deadline if deadline is not None else time.monotonic() + 600
    if user_id is not None:
        raise BundleImportError("user_id_forbidden", error_code="user_id_forbidden")
    try:
        files = manifest["files"]
        if (
            manifest.get("manifest_version") != 2
            or manifest.get("recovery_scope") != "postgresql_database"
            or manifest.get("user_id") is not None
            or manifest.get("datasets") != ["postgresql"]
            or manifest.get("coverage") != _POSTGRESQL_COVERAGE
            or manifest.get("exclusions") != _POSTGRESQL_EXCLUSIONS
            or not isinstance(files, dict)
            or len(files) != 1
        ):
            raise ValueError("invalid_database_scope")
        filename, metadata = next(iter(files.items()))
        if (
            not isinstance(filename, str)
            or not filename.endswith(".dump")
            or PurePosixPath(filename).name != filename
            or PureWindowsPath(filename).name != filename
            or not isinstance(metadata, dict)
            or metadata.get("dataset") != "postgresql"
            or metadata.get("hash_algorithm") != "sha256"
        ):
            raise ValueError("invalid_database_artifact")
        size = int(metadata["size_bytes"])
        expected_hash = metadata["sha256"]
        if size < 32 or not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("invalid_database_integrity")
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleImportError("invalid_bundle_manifest", error_code="invalid_bundle_manifest") from exc
    try:
        _postgresql_database_config()
    except UnsupportedBackupBackendError as exc:
        raise BundleImportError(str(exc), error_code="unsupported_backup_backend") from exc
    staging = tempfile.mkdtemp(prefix="tldw_pg_bundle_", dir=staging_parent)
    preserve_staging = False
    dump_path = os.path.join(staging, filename)
    try:
        try:
            with zipfile.ZipFile(file_path) as archive:
                if (
                    sorted(archive.namelist()) != sorted(["manifest.json", filename])
                    or archive.getinfo(filename).file_size != size
                ):
                    raise ValueError("invalid_database_members")
                digest = hashlib.sha256()
                total = 0
                with archive.open(filename) as source, open(dump_path, "wb") as target:
                    header = source.read(32)
                    if not header.startswith(b"PGDMP") or len(header) < 32:
                        raise ValueError("invalid_database_dump")
                    target.write(header)
                    digest.update(header)
                    total += len(header)
                    while chunk := source.read(1024 * 1024):
                        if time.monotonic() >= deadline:
                            raise ValueError("archive_validation_timeout")
                        target.write(chunk)
                        total += len(chunk)
                        digest.update(chunk)
                if total != size or digest.hexdigest() != expected_hash:
                    raise ValueError("database_integrity_mismatch")
        except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
            raise BundleImportError("invalid_bundle_manifest", error_code="invalid_bundle_manifest") from exc
        status = validate_postgres_backup(dump_path, timeout_seconds=deadline - time.monotonic())
        if status != "ok":
            raise BundleImportError("invalid_postgresql_archive", error_code="invalid_postgresql_archive")
        yield dump_path
    except PostgresToolTimeout as exc:
        preserve_staging = exc.process_running
        if preserve_staging:
            logger.warning("Retaining PostgreSQL recovery staging for an unconfirmed child: {}", staging)
        raise
    finally:
        if not preserve_staging:
            shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def import_bundle(
    *,
    file_path: str,
    user_id: int | None,
    dry_run: bool = False,
    allow_downgrade: bool = False,
    admin_user_id: int = 0,
) -> dict[str, Any]:
    """Import a bundle ZIP, restoring datasets.

    Returns a dict compatible with BundleImportResponse.
    """
    _check_rate_limit(admin_user_id, "import")

    if not os.path.isfile(file_path):
        raise BundleImportError("file_not_found", error_code="file_not_found")

    # Estimate required import space from bundle size.
    zip_size = os.path.getsize(file_path)
    required = zip_size * 3

    try:
        manifest = _read_manifest_from_zip(file_path)
    except BundleImportError:
        raise
    except Exception as exc:
        raise BundleImportError(f"invalid_bundle: {exc}", error_code="invalid_bundle") from exc

    if not isinstance(manifest, dict):
        raise BundleImportError("invalid_bundle_manifest", error_code="invalid_bundle_manifest")
    m_version = manifest.get("manifest_version", 0)
    if not isinstance(m_version, int) or isinstance(m_version, bool) or m_version < 0:
        raise BundleImportError("invalid_bundle_manifest", error_code="invalid_bundle_manifest")
    if m_version == 2:
        with _validated_postgresql_archive(file_path, manifest, user_id):
            pass
        if not dry_run:
            raise BundleImportError(
                "offline_restore_required: use Helper_Scripts/pg_backup_restore.py restore-bundle with all API, Jobs and Scheduler clients stopped",
                error_code="offline_restore_required",
            )
        return {
            "status": "compatible",
            "datasets_restored": [],
            "safety_snapshots": {},
            "rollback_failures": [],
            "warnings": [
                "Whole database: all users, stored content and Jobs. Offline recovery only; uploaded files and external stores are excluded."
            ],
            "validations": [
                {"dataset": "postgresql", "compatible": True, "message": "Offline whole-database recovery required"}
            ],
        }
    if manifest.get("recovery_scope") is not None:
        raise BundleImportError("invalid_bundle_manifest", error_code="invalid_bundle_manifest")
    if m_version > _MANIFEST_VERSION:
        raise BundleImportError(
            "unsupported_manifest_version",
            error_code="unsupported_manifest_version",
        )

    manifest_versions = manifest.get("schema_versions", {})
    datasets = manifest.get("datasets")
    files_meta = manifest.get("files")
    try:
        if (
            not isinstance(datasets, list)
            or not datasets
            or not isinstance(files_meta, dict)
            or not files_meta
            or not isinstance(manifest_versions, dict)
        ):
            raise ValueError("missing_dataset_coverage")
        datasets = [_validate_backup_dataset(ds) for ds in datasets if isinstance(ds, str)]
        if len(datasets) != len(manifest["datasets"]) or len(set(datasets)) != len(datasets):
            raise ValueError("invalid_dataset_coverage")
        if any(ds not in _ALL_DATASETS for ds in datasets):
            raise ValueError("unsupported_bundle_dataset")
        file_datasets = []
        for filename, meta in files_meta.items():
            if (
                not isinstance(filename, str)
                or not filename
                or not isinstance(meta, dict)
                or not isinstance(meta.get("dataset"), str)
            ):
                raise ValueError("invalid_file_metadata")
            meta["dataset"] = _validate_backup_dataset(meta["dataset"])
            file_datasets.append(meta["dataset"])
        if sorted(file_datasets) != sorted(datasets):
            raise ValueError("incomplete_dataset_coverage")
    except ValueError as exc:
        raise BundleImportError("invalid_bundle_manifest", error_code="invalid_bundle_manifest") from exc
    _preflight_backup_storage(datasets, BundleImportError)
    current_versions = _get_schema_versions()

    # Per-user datasets need a user_id
    per_user_in_bundle = [ds for ds in datasets if ds in _PER_USER_DATASETS]
    if per_user_in_bundle and user_id is None:
        try:
            user_id = DatabasePaths.get_single_user_id()
        except Exception as exc:
            raise BundleImportError("user_id_required", error_code="user_id_required") from exc

    # Check disk space across all import write targets.
    _check_import_disk_space(
        file_path=file_path,
        required_bytes=required,
        datasets=datasets,
        user_id=user_id,
    )

    warnings: list[str] = []
    validations: list[dict[str, Any]] = []

    # App version check
    app_version = _get_app_version()
    manifest_app_version = manifest.get("app_version")
    if manifest_app_version and app_version:
        m_major = manifest_app_version.split(".")[0] if manifest_app_version else ""
        c_major = app_version.split(".")[0] if app_version else ""
        if m_major != c_major:
            warnings.append(f"Major version mismatch: bundle={manifest_app_version}, current={app_version}")

    # Schema compatibility checks
    for ds in datasets:
        m_ver = manifest_versions.get(ds)
        c_ver = current_versions.get(ds)
        validation: dict[str, Any] = {
            "dataset": ds,
            "manifest_version": m_ver,
            "current_version": c_ver,
            "compatible": True,
            "message": "ok",
        }
        if m_ver is not None and c_ver is not None:
            if m_ver > c_ver and not allow_downgrade:
                validation["compatible"] = False
                validation["message"] = f"Schema version {m_ver} is newer than current {c_ver}"
        validations.append(validation)

    incompatible = [v for v in validations if not v["compatible"]]
    if incompatible:
        if dry_run:
            return {
                "status": "incompatible",
                "datasets_restored": [],
                "warnings": warnings,
                "safety_snapshots": {},
                "validations": validations,
                "rollback_failures": [],
            }
        raise BundleSchemaIncompatibleError(f"schema_incompatible: {', '.join(v['dataset'] for v in incompatible)}")

    # Extract once to staging dir — verify checksums and path safety in
    # a single pass (avoids the previous double-extraction overhead).
    os.makedirs(_backup_base_dir(), exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix="tldw_bundle_import_", dir=_backup_base_dir())
    safety_snapshots: dict[str, str] = {}
    datasets_restored: list[str] = []

    try:
        # Extract with path traversal protection
        with zipfile.ZipFile(file_path, "r") as zf:
            for member in zf.infolist():
                target = os.path.realpath(os.path.join(staging_dir, member.filename))
                if not target.startswith(os.path.realpath(staging_dir) + os.sep) and target != os.path.realpath(
                    staging_dir
                ):
                    raise BundleImportError(
                        f"path_traversal_detected: {member.filename}",
                        error_code="path_traversal_detected",
                    )
                zf.extract(member, staging_dir)

        # Manifest file names are untrusted independently of ZIP members.
        validated_files: dict[str, str] = {}
        for filename, meta in files_meta.items():
            extracted = safe_join(staging_dir, filename)
            if extracted is None:
                raise BundleImportError("path_traversal_detected", error_code="path_traversal_detected")
            if not os.path.isfile(extracted):
                raise BundleImportError(
                    f"missing_file_in_bundle: {filename}",
                    error_code="missing_file_in_bundle",
                )
            actual_size = os.path.getsize(extracted)
            expected_size = meta.get("size_bytes")
            if expected_size is not None and actual_size != int(expected_size):
                raise BundleImportError(
                    f"size_verification_failed: {filename} (expected {expected_size}, got {actual_size})",
                    error_code="size_verification_failed",
                )
            actual_hash = _compute_sha256(extracted)
            expected_hash = meta.get("sha256", "")
            if actual_hash != expected_hash:
                raise BundleImportError(
                    f"checksum_verification_failed: {filename}",
                    error_code="checksum_verification_failed",
                )
            ds = meta.get("dataset")
            target_path, _ = _resolve_dataset_db_path(ds, None if ds == "authnz" else user_id)
            postgres_target = ds == "authnz" and str(target_path).startswith("postgres")
            with open(extracted, "rb") as source:
                header = source.read(16)
            expected = b"PGDMP" if postgres_target else b"SQLite format 3\0"
            if not header.startswith(expected):
                raise BundleImportError(
                    f"backup_format_mismatch:{ds}: expected {'PostgreSQL custom dump' if postgres_target else 'SQLite backup'}; restore requires the same database backend",
                    error_code="backup_format_mismatch",
                )
            validated_files[filename] = extracted

        if dry_run:
            return {
                "status": "compatible",
                "datasets_restored": [],
                "warnings": warnings,
                "safety_snapshots": {},
                "validations": validations,
                "rollback_failures": [],
            }

        for ds in datasets:
            ds_user_id = None if ds == "authnz" else user_id

            # Create safety snapshot before overwriting — if this fails we
            # must NOT overwrite the live DB because rollback would be
            # impossible.  Skip the dataset with a warning instead.
            try:
                safety = create_backup_snapshot(
                    dataset=ds,
                    user_id=ds_user_id,
                    backup_type="full",
                    max_backups=None,
                )
                safety_snapshots[ds] = safety.filename
            except Exception as exc:
                logger.warning("Safety snapshot failed for {}: {}", ds, exc)
                warnings.append(f"Skipped restoring '{ds}': safety snapshot failed ({exc})")
                continue

            # Find the matching file in the bundle
            ds_filename = None
            for filename, meta in files_meta.items():
                if meta.get("dataset") == ds:
                    ds_filename = filename
                    break

            if ds_filename is None:
                warnings.append(f"No file found in bundle for dataset '{ds}'")
                continue

            # Resolve live DB path and copy
            try:
                restore_backup_file(
                    dataset=ds,
                    user_id=ds_user_id,
                    backup_path=validated_files[ds_filename],
                )
                datasets_restored.append(ds)
            except Exception as exc:
                logger.error("Failed to restore dataset {}: {}", ds, exc)
                # Rollback already-restored datasets
                rollback_failures: list[str] = []
                for restored_ds in datasets_restored:
                    snap_id = safety_snapshots.get(restored_ds)
                    if snap_id:
                        try:
                            from tldw_Server_API.app.services.admin_data_ops_service import (
                                restore_backup_snapshot,
                            )

                            r_user_id = None if restored_ds == "authnz" else user_id
                            restore_backup_snapshot(
                                dataset=restored_ds,
                                user_id=r_user_id,
                                backup_id=snap_id,
                            )
                        except Exception as rollback_exc:
                            logger.error("Rollback failed for {}: {}", restored_ds, rollback_exc)
                            rollback_failures.append(f"{restored_ds}: rollback operation failed")
                detail = f"restore_failed:{ds}: restore operation failed"
                if isinstance(exc, sqlite3.Error) and _sqlite_error_is_busy(exc):
                    detail = f"restore_failed:{ds}: target database is busy/locked; stop active clients and retry"
                if rollback_failures:
                    detail += f"; rollback_failures: {'; '.join(rollback_failures)}"
                import_exc = BundleImportError(
                    detail,
                    error_code="restore_failed",
                )
                if rollback_failures:
                    import_exc.rollback_failures = rollback_failures  # type: ignore[attr-defined]
                raise import_exc from exc

        _record_rate_limit(admin_user_id, "import")
        return {
            "status": "imported",
            "datasets_restored": datasets_restored,
            "warnings": warnings,
            "safety_snapshots": safety_snapshots,
            "validations": validations,
            "rollback_failures": [],
        }
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


async def import_bundle_async(
    *,
    file_path: str,
    user_id: int | None,
    dry_run: bool = False,
    allow_downgrade: bool = False,
    admin_user_id: int = 0,
) -> dict[str, Any]:
    """Async wrapper that runs import_bundle under an atomic non-blocking lock."""
    return await _run_bundle_operation_with_lock(
        import_bundle,
        file_path=file_path,
        user_id=user_id,
        dry_run=dry_run,
        allow_downgrade=allow_downgrade,
        admin_user_id=admin_user_id,
    )


async def _run_bundle_operation_with_lock(
    operation: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Run an operation under the global bundle lock without queuing.

    Uses an atomic non-blocking acquire so overlapping requests fail fast with
    BundleConcurrencyError instead of waiting for the lock.
    """
    if not _bundle_lock.acquire(blocking=False):
        raise BundleConcurrencyError("bundle_operation_in_progress")

    work_task: asyncio.Task[Any] | None = None
    try:
        work_task = asyncio.create_task(asyncio.to_thread(operation, **kwargs))
        return await asyncio.shield(work_task)
    except asyncio.CancelledError:
        # Keep lock ownership until the background operation exits.
        if work_task is not None:
            try:
                await work_task
            except Exception as exc:
                logger.warning(
                    "Bundle operation ended with error after cancellation: {}",
                    exc,
                )
        raise
    finally:
        _bundle_lock.release()
