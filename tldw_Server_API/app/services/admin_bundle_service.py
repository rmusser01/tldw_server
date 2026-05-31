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
from dataclasses import dataclass
from types import MappingProxyType
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Backups import (
    _sqlite_error_is_busy,
    restore_sqlite_database_file,
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
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    _backup_base_dir,
    _resolve_dataset_db_path,
    _validate_backup_dataset,
    create_backup_snapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MANIFEST_VERSION = 1
_ALL_DATASETS = ("media", "chacha", "prompts", "evaluations", "audit", "authnz")
_PER_USER_DATASETS = frozenset(_ALL_DATASETS) - {"authnz"}

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
    - system temp dir (staging extraction),
    - upload/temp file directory,
    - live DB directories for datasets in the manifest.
    """
    check_paths: set[str] = {
        os.path.realpath(tempfile.gettempdir()),
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
            manifest = _read_manifest_cached(entry.path)
        except Exception as exc:
            logger.warning(
                "Skipping bundle retention cleanup for unreadable manifest {}: {}",
                entry.name,
                exc,
            )
            continue

        if manifest.get("user_id") != user_id:
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

    effective_datasets = list(datasets) if datasets else list(_ALL_DATASETS)

    # Validate each dataset name
    for ds in effective_datasets:
        _validate_backup_dataset(ds)

    # Validate user_id requirement
    per_user_requested = [ds for ds in effective_datasets if ds in _PER_USER_DATASETS]
    if per_user_requested and user_id is None:
        # Auto-resolve in single-user mode
        try:
            user_id = DatabasePaths.get_single_user_id()
        except Exception as exc:
            raise BundleExportError(
                "user_id_required", error_code="user_id_required"
            ) from exc

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
            ds_user_id = None if ds == "authnz" else user_id
            try:
                backup_file = create_backup_snapshot(
                    dataset=ds,
                    user_id=ds_user_id,
                    backup_type="full",
                    max_backups=max_backups,
                )
            except Exception as exc:
                raise BundleExportError(
                    f"backup_failed:{ds}: {exc}"
                ) from exc

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

        manifest_path = os.path.join(staging_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Build ZIP filename
        user_label = str(user_id) if user_id else "global"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        zip_name = f"tldw-backup-bundle-{user_label}-{timestamp}.zip"
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
            manifest_version=_MANIFEST_VERSION,
            notes=notes,
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


def list_bundles(
    *,
    user_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
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
            manifest = _read_manifest_cached(entry.path)
        except Exception:
            logger.warning("Skipping unreadable bundle: {}", entry.name)
            continue

        bundle_user_id = manifest.get("user_id")
        if user_id is not None and bundle_user_id != user_id:
            continue

        stat = entry.stat()
        items.append(BundleMetadata(
            bundle_id=entry.name,
            user_id=bundle_user_id,
            created_at=datetime.fromisoformat(manifest["created_at"]) if manifest.get("created_at") else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            size_bytes=int(stat.st_size),
            datasets=tuple(manifest.get("datasets", [])),
            schema_versions=MappingProxyType(manifest.get("schema_versions", {})),
            app_version=manifest.get("app_version"),
            manifest_version=manifest.get("manifest_version", 0),
            notes=manifest.get("notes"),
        ))

    items.sort(key=lambda m: m.created_at, reverse=True)
    total = len(items)
    return items[offset:offset + limit], total


def get_bundle_metadata(bundle_id: str) -> BundleMetadata:
    """Get metadata for a single bundle."""
    path = _resolve_bundle_path(bundle_id)
    manifest = _read_manifest_from_zip(path)
    stat = os.stat(path)
    return BundleMetadata(
        bundle_id=bundle_id,
        user_id=manifest.get("user_id"),
        created_at=datetime.fromisoformat(manifest["created_at"]) if manifest.get("created_at") else datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        size_bytes=int(stat.st_size),
        datasets=tuple(manifest.get("datasets", [])),
        schema_versions=MappingProxyType(manifest.get("schema_versions", {})),
        app_version=manifest.get("app_version"),
        manifest_version=manifest.get("manifest_version", 0),
        notes=manifest.get("notes"),
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
    except Exception as exc:
        raise BundleImportError(
            f"invalid_bundle: {exc}", error_code="invalid_bundle"
        ) from exc

    m_version = manifest.get("manifest_version", 0)
    if m_version > _MANIFEST_VERSION:
        raise BundleImportError(
            "unsupported_manifest_version",
            error_code="unsupported_manifest_version",
        )

    current_versions = _get_schema_versions()
    manifest_versions = manifest.get("schema_versions", {})
    datasets = manifest.get("datasets", [])
    files_meta = manifest.get("files", {})

    # Per-user datasets need a user_id
    per_user_in_bundle = [ds for ds in datasets if ds in _PER_USER_DATASETS]
    if per_user_in_bundle and user_id is None:
        try:
            user_id = DatabasePaths.get_single_user_id()
        except Exception as exc:
            raise BundleImportError(
                "user_id_required", error_code="user_id_required"
            ) from exc

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
            warnings.append(
                f"Major version mismatch: bundle={manifest_app_version}, current={app_version}"
            )

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
                validation["message"] = (
                    f"Schema version {m_ver} is newer than current {c_ver}"
                )
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
        raise BundleSchemaIncompatibleError(
            f"schema_incompatible: {', '.join(v['dataset'] for v in incompatible)}"
        )

    # Extract once to staging dir — verify checksums and path safety in
    # a single pass (avoids the previous double-extraction overhead).
    staging_dir = tempfile.mkdtemp(prefix="tldw_bundle_import_")
    safety_snapshots: dict[str, str] = {}
    datasets_restored: list[str] = []

    try:
        # Extract with path traversal protection
        with zipfile.ZipFile(file_path, "r") as zf:
            for member in zf.infolist():
                target = os.path.realpath(
                    os.path.join(staging_dir, member.filename)
                )
                if not target.startswith(os.path.realpath(staging_dir) + os.sep) and target != os.path.realpath(staging_dir):
                    raise BundleImportError(
                        f"path_traversal_detected: {member.filename}",
                        error_code="path_traversal_detected",
                    )
                zf.extract(member, staging_dir)

        # Verify checksums on the extracted files
        for filename, meta in files_meta.items():
            extracted = os.path.join(staging_dir, filename)
            if not os.path.isfile(extracted):
                raise BundleImportError(
                    f"missing_file_in_bundle: {filename}",
                    error_code="missing_file_in_bundle",
                )
            actual_size = os.path.getsize(extracted)
            expected_size = meta.get("size_bytes")
            if expected_size is not None and actual_size != int(expected_size):
                raise BundleImportError(
                    f"size_verification_failed: {filename} "
                    f"(expected {expected_size}, got {actual_size})",
                    error_code="size_verification_failed",
                )
            actual_hash = _compute_sha256(extracted)
            expected_hash = meta.get("sha256", "")
            if actual_hash != expected_hash:
                raise BundleImportError(
                    f"checksum_verification_failed: {filename}",
                    error_code="checksum_verification_failed",
                )

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
                warnings.append(
                    f"Skipped restoring '{ds}': safety snapshot failed ({exc})"
                )
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
                live_db_path, _ = _resolve_dataset_db_path(ds, ds_user_id)
                extracted_path = os.path.join(staging_dir, ds_filename)
                if not os.path.isfile(extracted_path):
                    warnings.append(f"Extracted file missing for dataset '{ds}'")
                    continue
                os.makedirs(os.path.dirname(live_db_path), exist_ok=True)
                restore_sqlite_database_file(
                    source_db_path=extracted_path,
                    target_db_path=live_db_path,
                    lock_timeout_seconds=0.5,
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
                            logger.error(
                                "Rollback failed for {}: {}", restored_ds, rollback_exc
                            )
                            rollback_failures.append(
                                f"{restored_ds}: rollback operation failed"
                            )
                detail = f"restore_failed:{ds}: restore operation failed"
                if isinstance(exc, sqlite3.Error) and _sqlite_error_is_busy(exc):
                    detail = (
                        f"restore_failed:{ds}: target database is busy/locked; "
                        "stop active clients and retry"
                    )
                if rollback_failures:
                    detail += (
                        f"; rollback_failures: {'; '.join(rollback_failures)}"
                    )
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
