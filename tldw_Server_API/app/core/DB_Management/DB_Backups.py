# Backup_Manager.py
#
# Imports:
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.parse
from datetime import datetime
from typing import Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseBackend

# Local Imports:
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.Utils.path_utils import safe_join
from tldw_Server_API.app.core.Utils.Utils import get_project_relative_path

#
# End of Imports

DB_BACKUP_RUNTIME_EXCEPTIONS = (
    sqlite3.Error,
    OSError,
    shutil.Error,
    subprocess.SubprocessError,
    ValueError,
    TypeError,
    AttributeError,
    RuntimeError,
    InvalidStoragePathError,
)


def _sqlite_error_is_busy(exc: BaseException) -> bool:
    """Return True when a SQLite exception indicates lock/busy contention."""
    text = str(exc or "").lower()
    return (
        "database is locked" in text
        or "database is busy" in text
        or "database is in use" in text
    )


def _copy_sqlite_database_via_backup(
    *,
    source_db_path: str,
    target_db_path: str,
    require_target_exclusive_lock: bool = False,
    lock_timeout_seconds: float = 0.5,
) -> None:
    """Copy SQLite database contents using SQLite's backup API.

    When ``require_target_exclusive_lock`` is True, this performs a preflight
    exclusive-lock check on the target so restore callers fail safely when
    other writers/readers are active.
    """
    source_path = str(source_db_path).strip()
    target_path = str(target_db_path).strip()
    if not source_path:
        raise sqlite3.OperationalError("source database path is empty")
    if not target_path:
        raise sqlite3.OperationalError("target database path is empty")

    source_is_uri = source_path.lower().startswith("file:")
    target_is_uri = target_path.lower().startswith("file:")

    # For backup artifact writes (non-restore mode), require target paths to stay
    # within the configured backup base directory. Restore-mode callers
    # intentionally target live database paths and should be validated by their
    # own DB-path validation before reaching this helper.
    if not target_is_uri:
        normalized_target_path = os.path.realpath(os.path.abspath(target_path))
        if not require_target_exclusive_lock:
            backup_base_dir = os.path.abspath(_get_backup_base_dir())
            try:
                common = os.path.commonpath([backup_base_dir, normalized_target_path])
            except ValueError:
                raise sqlite3.OperationalError("invalid target database path") from None
            if common != backup_base_dir:
                raise sqlite3.OperationalError("target database path is outside backup directory")
        # Use the normalized path from this point forward.
        target_path = normalized_target_path

        target_parent = os.path.dirname(target_path)
        if target_parent:
            os.makedirs(target_parent, exist_ok=True)

    timeout = max(0.0, float(lock_timeout_seconds))
    busy_timeout_ms = int(timeout * 1000)

    # Preflight lock check for restore callers: fail fast if the target DB is
    # currently busy instead of attempting any raw file replacement.
    if require_target_exclusive_lock:
        with sqlite3.connect(target_path, timeout=timeout, uri=target_is_uri) as guard:
            guard.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            guard.execute("BEGIN EXCLUSIVE")
            guard.rollback()

    with sqlite3.connect(source_path, timeout=timeout, uri=source_is_uri) as source:
        source.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        # Best-effort checkpoint to reduce WAL sidecar churn around backups.
        try:
            source.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.OperationalError:
            pass

        with sqlite3.connect(target_path, timeout=timeout, uri=target_is_uri) as target:
            target.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            busy_started_at: float | None = None

            def _backup_progress(status: int, _remaining: int, _total: int) -> None:
                nonlocal busy_started_at
                if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    if busy_started_at is None:
                        busy_started_at = time.monotonic()
                    elif timeout > 0 and (time.monotonic() - busy_started_at) >= timeout:
                        raise sqlite3.OperationalError("database is busy")
                else:
                    busy_started_at = None

            source.backup(target, pages=256, progress=_backup_progress, sleep=0.05)


def restore_sqlite_database_file(
    *,
    source_db_path: str,
    target_db_path: str,
    lock_timeout_seconds: float = 0.5,
) -> None:
    """Safely restore a target SQLite DB from a source SQLite file."""
    _copy_sqlite_database_via_backup(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        require_target_exclusive_lock=True,
        lock_timeout_seconds=lock_timeout_seconds,
    )


def _safe_join(base_dir: str, name: str) -> Optional[str]:
    """
    Safely join a base directory and a path component, preventing directory traversal
    and symlink-based escapes.

    Rejects empty components and absolute paths.

    Returns the normalized, real path on success, or None on failure.
    """
    return safe_join(base_dir, name)


def _get_backup_base_dir() -> str:
    """Return the base directory used for database backups."""
    return os.environ.get("TLDW_DB_BACKUP_PATH") or get_project_relative_path("tldw_DB_Backups")


def _ensure_within_base(base_dir: str, candidate_path: str) -> Optional[str]:
    """Validate that candidate_path resolves within base_dir and return a safe path."""
    base_abs = os.path.abspath(base_dir)
    candidate_abs = os.path.abspath(candidate_path)
    try:
        rel = os.path.relpath(candidate_abs, base_abs)
    except ValueError:
        return None
    return safe_join(base_abs, rel)


def _resolve_backup_dir(backup_dir: str) -> Optional[str]:
    """Resolve a backup directory under the configured backup base."""
    raw = str(backup_dir or "").strip()
    if not raw:
        return None
    base_dir = _get_backup_base_dir()
    if os.path.isabs(raw):
        return _ensure_within_base(base_dir, raw)
    candidate_abs = os.path.abspath(raw)
    resolved = _ensure_within_base(base_dir, candidate_abs)
    if resolved:
        return resolved
    return safe_join(os.path.abspath(base_dir), raw)


def _get_allowed_db_roots() -> list[str]:
    """Return a list of allowed database root directories."""
    roots: list[str] = []
    try:
        roots.append(str(DatabasePaths.get_user_db_base_dir(allow_legacy_alias=True)))
    except (InvalidStoragePathError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Failed to resolve user DB base dir: {}", exc)
    try:
        roots.append(get_project_relative_path("Databases"))
    except (InvalidStoragePathError, OSError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Failed to resolve project Databases path: {}", exc)

    extra = os.environ.get("TLDW_DB_ALLOWED_BASE_DIRS")
    if extra:
        for entry in extra.split(os.pathsep):
            candidate = entry.strip()
            if not candidate:
                continue
            try:
                expanded = os.path.expanduser(candidate)
            except (OSError, ValueError, TypeError) as exc:
                logger.debug("Failed to expand DB base dir {!r}: {}", candidate, exc)
                expanded = candidate
            if os.path.isabs(expanded):
                roots.append(os.path.abspath(expanded))
            else:
                roots.append(os.path.abspath(get_project_relative_path(expanded)))

    deduped: list[str] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _sqlite_uri_to_path(raw: str) -> Optional[str]:
    """Return filesystem path from a file: SQLite URI, or None if not applicable."""
    lowered = raw.lower()
    if not lowered.startswith("file:"):
        return None
    try:
        parsed = urllib.parse.urlparse(raw)
    except (ValueError, TypeError, AttributeError):
        return None
    if parsed.scheme != "file":
        return None
    query = (parsed.query or "").lower()
    path = urllib.parse.unquote(parsed.path or "")
    if not path or path in {":memory:", "/:memory:"} or "mode=memory" in query:
        return None
    return path


def _resolve_db_path(db_path: str) -> Optional[str]:
    """Resolve a database path within the allowed roots."""
    raw = str(db_path or "").strip()
    if not raw:
        return None
    uri_path = _sqlite_uri_to_path(raw)
    path_for_checks = uri_path if uri_path is not None else raw
    is_abs = os.path.isabs(path_for_checks)
    candidate_abs = os.path.abspath(path_for_checks)
    for base in _get_allowed_db_roots():
        resolved = _ensure_within_base(base, candidate_abs)
        if resolved:
            return resolved
        if not is_abs:
            resolved = safe_join(os.path.abspath(base), path_for_checks)
            if resolved:
                return resolved
    return None
#######################################################################################################################
#
# Functions:

_SQLITE_BACKUP_EXTS = (".db", ".sqlib")
_POSTGRES_BACKUP_EXTS = (".dump",)


def _sanitize_backup_label(label: str, fallback: str) -> str:
    """
    Sanitize a backup label so it is safe to use as a single path component.

    - Only allow alphanumerics, '-' and '_'.
    - Strip surrounding whitespace and '_' characters.
    - Truncate to a reasonable maximum length.
    - Fall back to the provided default if the result is empty or malformed.
    """
    raw = str(label or "").strip()
    # Allow only safe filename characters
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    # Trim leading/trailing underscores to avoid odd-looking names
    cleaned = cleaned.strip("_")
    # Enforce a maximum length to avoid pathological names
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    # Ensure we end up with a non-empty, sane label
    if not cleaned or not cleaned[0].isalnum():
        return fallback
    return cleaned


def _validate_backup_name(backup_name: str, allowed_exts: tuple[str, ...]) -> Optional[str]:
    """Validate a backup filename for safety and allowed extensions."""
    name = str(backup_name or "").strip()
    if not name:
        return None
    if os.path.basename(name) != name:
        return None
    if name.startswith("-"):
        return None
    if not name.endswith(allowed_exts):
        return None
    return name


def init_backup_directory(backup_base_dir: str, db_name: str) -> str:
    """Initialize backup directory for a specific database."""
    backup_dir = _safe_join(backup_base_dir, db_name) or ""
    if not backup_dir:
        raise InvalidStoragePathError("Invalid backup directory")
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


def create_backup(db_path: str, backup_dir: str, db_name: str) -> str:
    """Create a full backup of the database."""
    try:
        # Guard: in-memory databases cannot be backed up to disk
        raw_db_path = str(db_path).strip()
        mem = raw_db_path
        if mem == ":memory:" or mem.startswith("file::memory:"):
            return "Cannot create backup for in-memory database"
        resolved_db_path = _resolve_db_path(raw_db_path)
        if not resolved_db_path:
            error_msg = "Invalid database path"
            logger.error(f"{error_msg}: {db_path}")
            return error_msg
        resolved_backup_dir = _resolve_backup_dir(backup_dir)
        if not resolved_backup_dir:
            error_msg = "Invalid backup directory"
            logger.error(f"{error_msg}: {backup_dir}")
            return error_msg
        is_uri = raw_db_path.lower().startswith("file:")
        db_path = resolved_db_path
        backup_dir = resolved_backup_dir
        if not os.path.exists(db_path):
            error_msg = f"Database not found: {db_path}"
            logger.error(error_msg)
            return error_msg
        os.makedirs(backup_dir, exist_ok=True)

        logger.info("Creating backup:")
        logger.info(f"  DB Path: {db_path}")
        logger.info(f"  Backup Dir: {backup_dir}")
        safe_db_name = _sanitize_backup_label(db_name, "db")
        logger.info(f"  DB Name: {safe_db_name}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = _safe_join(backup_dir, f"{safe_db_name}_backup_{timestamp}.db")
        if not backup_file:
            error_msg = "Invalid backup path"
            logger.error(error_msg)
            return error_msg
        logger.info(f"  Full backup path: {backup_file}")

        # Create a backup using SQLite's backup API. Do not copy WAL/SHM sidecars:
        # sidecar replay against a different restore context can corrupt targets.
        source_connect_path = raw_db_path if is_uri else db_path
        _copy_sqlite_database_via_backup(
            source_db_path=source_connect_path,
            target_db_path=backup_file,
            require_target_exclusive_lock=False,
            lock_timeout_seconds=0.5,
        )

        logger.info(f"Backup created successfully: {backup_file}")
        return f"Backup created: {backup_file}"
    except DB_BACKUP_RUNTIME_EXCEPTIONS as e:
        error_msg = f"Failed to create backup: {str(e)}"
        logger.error(error_msg)
        return error_msg


def create_incremental_backup(db_path: str, backup_dir: str, db_name: str) -> str:
    """Create an incremental backup using VACUUM INTO."""
    try:
        raw_db_path = str(db_path).strip()
        mem = raw_db_path
        if mem == ":memory:" or mem.startswith("file::memory:"):
            return "Cannot create incremental backup for in-memory database"
        resolved_db_path = _resolve_db_path(raw_db_path)
        if not resolved_db_path:
            error_msg = "Invalid database path"
            logger.error(f"{error_msg}: {db_path}")
            return error_msg
        resolved_backup_dir = _resolve_backup_dir(backup_dir)
        if not resolved_backup_dir:
            error_msg = "Invalid backup directory"
            logger.error(f"{error_msg}: {backup_dir}")
            return error_msg
        is_uri = raw_db_path.lower().startswith("file:")
        db_path = resolved_db_path
        backup_dir = resolved_backup_dir
        if not os.path.exists(db_path):
            error_msg = f"Database not found: {db_path}"
            logger.error(error_msg)
            return error_msg
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_db_name = _sanitize_backup_label(db_name, "db")
        backup_file = _safe_join(
            backup_dir,
            f"{safe_db_name}_incremental_{timestamp}.sqlib",
        )
        if not backup_file:
            error_msg = "Invalid backup path"
            logger.error(error_msg)
            return error_msg

        source_connect_path = raw_db_path if is_uri else db_path
        with sqlite3.connect(source_connect_path, uri=is_uri) as conn:
            try:
                conn.execute("VACUUM INTO ?", (backup_file,))
            except sqlite3.OperationalError:
                escaped_path = backup_file.replace("'", "''")
                conn.execute(f"VACUUM INTO '{escaped_path}'")

        logger.info(f"Incremental backup created: {backup_file}")
        return f"Incremental backup created: {backup_file}"
    except DB_BACKUP_RUNTIME_EXCEPTIONS as e:
        error_msg = f"Failed to create incremental backup: {str(e)}"
        logger.error(error_msg)
        return error_msg


def list_backups(backup_dir: str) -> str:
    """List all available backups."""
    try:
        resolved_backup_dir = _resolve_backup_dir(backup_dir)
        if not resolved_backup_dir:
            error_msg = "Invalid backup directory"
            logger.error(f"{error_msg}: {backup_dir}")
            return error_msg
        backup_dir = resolved_backup_dir
        backups = [f for f in os.listdir(backup_dir)
                   if f.endswith(('.db', '.sqlib'))]
        backups.sort(reverse=True)  # Most recent first
        return "\n".join(backups) if backups else "No backups found"
    except DB_BACKUP_RUNTIME_EXCEPTIONS as e:
        error_msg = f"Failed to list backups: {str(e)}"
        logger.error(error_msg)
        return error_msg


def restore_single_db_backup(db_path: str, backup_dir: str, db_name: str, backup_name: str) -> str:
    """Restore database from a backup file."""
    try:
        safe_backup_name = _validate_backup_name(backup_name, _SQLITE_BACKUP_EXTS)
        if not safe_backup_name:
            error_msg = "Invalid backup name"
            logger.error(f"{error_msg}: {backup_name}")
            return error_msg
        resolved_db_path = _resolve_db_path(db_path)
        if not resolved_db_path:
            error_msg = "Invalid database path"
            logger.error(f"{error_msg}: {db_path}")
            return error_msg
        resolved_backup_dir = _resolve_backup_dir(backup_dir)
        if not resolved_backup_dir:
            error_msg = "Invalid backup directory"
            logger.error(f"{error_msg}: {backup_dir}")
            return error_msg
        db_path = resolved_db_path
        backup_dir = resolved_backup_dir
        logger.info(f"Restoring backup: {safe_backup_name}")
        backup_path = _safe_join(backup_dir, safe_backup_name)
        if not backup_path:
            error_msg = "Invalid backup path"
            logger.error(f"{error_msg}: {safe_backup_name}")
            return error_msg
        if not os.path.exists(backup_path):
            logger.error(f"Backup file not found: {backup_name}")
            return f"Backup file not found: {backup_name}"

        if os.path.exists(db_path):
            # Create a timestamp for the current db
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_db_name = _sanitize_backup_label(db_name, "db")
            current_backup = os.path.join(
                backup_dir, f"{safe_db_name}_pre_restore_{timestamp}.db"
            )

            # Backup current database before restore
            logger.info(f"Creating backup of current database: {current_backup}")
            _copy_sqlite_database_via_backup(
                source_db_path=db_path,
                target_db_path=current_backup,
                require_target_exclusive_lock=False,
                lock_timeout_seconds=0.5,
            )
        else:
            logger.info(
                f"No existing database at {db_path}; skipping pre-restore snapshot."
            )

        # Restore the backup. Fail safely if active clients hold DB locks.
        logger.info(f"Restoring database from {backup_name}")
        restore_sqlite_database_file(
            source_db_path=backup_path,
            target_db_path=db_path,
            lock_timeout_seconds=0.5,
        )

        logger.info(f"Database restored from {backup_name}")
        return f"Database restored from {backup_name}"
    except DB_BACKUP_RUNTIME_EXCEPTIONS as e:
        if _sqlite_error_is_busy(e):
            error_msg = (
                "Failed to restore backup: target database is busy/locked; "
                "stop active clients and retry"
            )
            logger.error(error_msg)
            return error_msg
        error_msg = f"Failed to restore backup: {str(e)}"
        logger.error(error_msg)
        return error_msg


def setup_backup_config(user_id: Optional[int] = None) -> dict[str, dict[str, str]]:
    """Setup configuration for database backups using centralized path utils.

    Returns a mapping of logical database names to their backup configuration.
    """
    # Standardized backup directory selection:
    # 1) TLDW_DB_BACKUP_PATH env var
    # 2) project-relative default ./tldw_DB_Backups/
    env_base = os.environ.get('TLDW_DB_BACKUP_PATH')
    backup_base_dir = env_base or get_project_relative_path('tldw_DB_Backups')
    os.makedirs(backup_base_dir, exist_ok=True)
    logger.info(f"Base backup directory: {os.path.abspath(backup_base_dir)}")

    uid = user_id if user_id is not None else DatabasePaths.get_single_user_id()

    # Resolve database paths
    db_paths = {
        'media': str(DatabasePaths.get_media_db_path(uid)),
        'chacha': str(DatabasePaths.get_chacha_db_path(uid)),
        'prompts': str(DatabasePaths.get_prompts_db_path(uid)),
        'evaluations': str(DatabasePaths.get_evaluations_db_path(uid)),
        'audit': str(DatabasePaths.get_audit_db_path(uid)),
    }

    configs: dict[str, dict[str, str]] = {}
    for name, path in db_paths.items():
        subdir = os.path.join(backup_base_dir, name)
        os.makedirs(subdir, exist_ok=True)
        logger.info(f"{name.capitalize()} backup directory: {os.path.abspath(subdir)}")
        configs[name] = {
            'db_path': path,
            'backup_dir': subdir,
            'db_name': name,
        }

    return configs


def _pg_dump_path() -> Optional[str]:
    """Return path to pg_dump binary if available, otherwise None."""
    path = shutil.which("pg_dump")
    if not path:
        logger.error("pg_dump not found on PATH. Install PostgreSQL client tools to enable backups.")
        return None
    return path


def create_postgres_backup(
    backend: DatabaseBackend,
    backup_dir: str,
    *,
    label: str = "content",
) -> str:
    """Create a PostgreSQL backup using pg_dump.

    Args:
        backend: A DatabaseBackend configured for PostgreSQL
        backup_dir: Target directory for the backup artifact
        label: Logical label to include in the backup filename

    Returns:
        str: Path to the backup file on success, or an error message on failure.
    """
    if backend.backend_type != BackendType.POSTGRESQL:
        msg = "create_postgres_backup requires a PostgreSQL backend"
        logger.error(msg)
        return msg

    pg_dump = _pg_dump_path()
    if not pg_dump:
        return "pg_dump not found on PATH"

    # Extract connection parameters from the backend configuration
    config = getattr(backend, "config", None)
    if not config:
        msg = "PostgreSQL backend missing configuration; cannot perform backup"
        logger.error(msg)
        return msg

    host = config.pg_host or "localhost"
    port = str(config.pg_port or 5432)
    dbname = config.pg_database or "tldw"
    user = config.pg_user or "postgres"
    password = config.pg_password or None

    backup_dir_real = _resolve_backup_dir(backup_dir)
    if not backup_dir_real:
        msg = "Invalid backup directory"
        logger.error(msg)
        return msg
    os.makedirs(backup_dir_real, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = _sanitize_backup_label(label, "content")
    out_file = _safe_join(backup_dir_real, f"{safe_label}_pgdump_{timestamp}.dump")
    if not out_file:
        msg = "Invalid backup path"
        logger.error(msg)
        return msg

    # Build pg_dump command
    cmd = [
        pg_dump,
        "-h", host,
        "-p", port,
        "-U", user,
        "-F", "c",            # custom format (compressed, pg_restore-compatible)
        "--no-owner",
        "--no-privileges",
        "-f", out_file,
        "--",
        dbname,
    ]

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = str(password)

    try:
        logger.info(f"Running pg_dump → {out_file}")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"pg_dump failed ({proc.returncode}): {proc.stderr.strip()}")
            return f"pg_dump failed: {proc.stderr.strip()}"
        logger.info(f"PostgreSQL backup created: {out_file}")
        return out_file
    except FileNotFoundError:
        msg = "pg_dump executable not found; ensure PostgreSQL client tools are installed"
        logger.error(msg)
        return msg
    except Exception:  # noqa: BLE001
        logger.error("pg_dump error")
        return "pg_dump error"


def _get_postgres_backup_base_dir(config) -> str:
    """
    Determine the base directory where PostgreSQL backups are stored for
    the given database configuration. This keeps restores confined to the
    expected backup root on disk instead of trusting user input paths.
    """
    # Reuse the same project-relative backup root used elsewhere.
    base_dir = _get_backup_base_dir()
    db_name = getattr(config, "pg_database", None) or "postgres"
    # Use a sanitized database name to avoid introducing path separators.
    safe_db_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(db_name))
    return os.path.join(base_dir, safe_db_name)


def restore_postgres_backup(
    backend: DatabaseBackend,
    dump_file: str,
    *,
    drop_first: bool = True,
) -> str:
    """Restore a PostgreSQL backup created with pg_dump (custom format).

    Args:
        backend: A DatabaseBackend configured for PostgreSQL
        dump_file: Path to a pg_dump custom-format .dump file
        drop_first: If True, use pg_restore -c to drop objects before restore

    Returns:
        str: "ok" on success or an error message on failure.
    """
    if backend.backend_type != BackendType.POSTGRESQL:
        msg = "restore_postgres_backup requires a PostgreSQL backend"
        logger.error(msg)
        return msg

    pg_restore = shutil.which("pg_restore")
    if not pg_restore:
        msg = "pg_restore not found on PATH"
        logger.error(msg)
        return msg

    # Accept either a backup basename or an explicit dump path.
    backup_id = str(dump_file or "").strip()
    backup_name = _validate_backup_name(os.path.basename(backup_id), _POSTGRES_BACKUP_EXTS)
    if not backup_name:
        msg = "Invalid dump file name"
        logger.error(f"{msg}: {dump_file}")
        return msg

    config = getattr(backend, "config", None)
    if not config:
        msg = "PostgreSQL backend missing configuration; cannot perform restore"
        logger.error(msg)
        return msg

    # Support both:
    # - explicit dump paths returned by create_postgres_backup/admin flows
    # - legacy basename-only lookup under the default postgres backup directory
    backup_base_dir = _get_backup_base_dir()
    candidate_paths: list[str] = []

    has_separator = (
        os.path.sep in backup_id
        or (os.altsep is not None and os.altsep in backup_id)
        or os.path.isabs(backup_id)
    )
    if backup_id and has_separator:
        resolved_input = _ensure_within_base(backup_base_dir, backup_id)
        if resolved_input:
            candidate_paths.append(resolved_input)

    legacy_backup_dir = _get_postgres_backup_base_dir(config)
    legacy_path = _safe_join(legacy_backup_dir, backup_name)
    if legacy_path:
        candidate_paths.append(legacy_path)

    if backup_id == backup_name:
        base_path = _safe_join(backup_base_dir, backup_name)
        if base_path:
            candidate_paths.append(base_path)

    deduped_candidates: list[str] = []
    for candidate in candidate_paths:
        if candidate not in deduped_candidates:
            deduped_candidates.append(candidate)

    safe_dump_path = next((path for path in deduped_candidates if os.path.exists(path)), None)
    if not safe_dump_path:
        msg = "dump not found"
        logger.error(f"{msg}: {backup_id} (searched={deduped_candidates})")
        return f"{msg}: {backup_id}"

    host = config.pg_host or "localhost"
    port = str(config.pg_port or 5432)
    dbname = config.pg_database or "tldw"
    user = config.pg_user or "postgres"
    password = config.pg_password or None

    cmd = [
        pg_restore,
        "-h", host,
        "-p", port,
        "-U", user,
        "-d", dbname,
        "-1",              # single transaction
    ]
    if drop_first:
        cmd.append("-c")    # clean (drop) database objects before recreating
    cmd.append("--")
    cmd.append(safe_dump_path)

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = str(password)

    try:
        logger.info(f"Running pg_restore on {safe_dump_path} into database '{dbname}'")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"pg_restore failed ({proc.returncode}): {proc.stderr.strip()}")
            return f"pg_restore failed: {proc.stderr.strip()}"
        logger.info("PostgreSQL restore completed successfully")
        return "ok"
    except FileNotFoundError:
        msg = "pg_restore executable not found; ensure PostgreSQL client tools are installed"
        logger.error(msg)
        return msg
    except Exception:  # noqa: BLE001
        logger.error("pg_restore error")
        return "pg_restore error"
