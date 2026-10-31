#!/usr/bin/env python3
"""
PostgreSQL content DB backup/restore helper.

Usage examples:

  # Backup to ./tldw_DB_Backups/postgres with label 'content'
  python Helper_Scripts/pg_backup_restore.py backup \
    --backup-dir ./tldw_DB_Backups/postgres \
    --label content

  # Restore from a .dump file (drops objects first)
  python Helper_Scripts/pg_backup_restore.py restore \
    --dump-file ./tldw_DB_Backups/postgres/content_20240101_000000.dump

Notes:
- The script reads the configured content backend via DB_Manager. Ensure your
  environment/config.txt selects PostgreSQL for the content DB.
- Requires pg_dump/pg_restore on PATH.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from loguru import logger


def _get_pg_backend():
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Manager import (
            get_content_backend_instance,
        )
        backend = get_content_backend_instance()
        return backend
    except Exception as exc:
        logger.error(f"Unable to resolve content backend: {exc}")
        return None


def _backup(args) -> int:
    backup_dir = Path(args.backup_dir or "./tldw_DB_Backups/postgres").resolve()
    label = args.label or "content"

    backend = _get_pg_backend()
    if backend is None:
        logger.error("No content backend is configured. Ensure PostgreSQL mode is enabled.")
        return 2

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    if getattr(backend, "backend_type", None) != BackendType.POSTGRESQL:
        logger.error("Content backend is not PostgreSQL. Switch content DB to PostgreSQL to use this helper.")
        return 2

    from tldw_Server_API.app.core.DB_Management.DB_Backups import create_postgres_backup

    out = create_postgres_backup(backend, backup_dir=str(backup_dir), label=label)
    if not out or not os.path.exists(out):
        logger.error(f"Backup failed: {out}")
        return 1
    logger.info(f"Backup created: {out}")
    print(out)
    return 0


def _restore(args) -> int:
    dump_file = Path(args.dump_file).resolve()
    if not dump_file.exists():
        logger.error(f"Dump file not found: {dump_file}")
        return 2

    backend = _get_pg_backend()
    if backend is None:
        logger.error("No content backend is configured. Ensure PostgreSQL mode is enabled.")
        return 2

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    if getattr(backend, "backend_type", None) != BackendType.POSTGRESQL:
        logger.error("Content backend is not PostgreSQL. Switch content DB to PostgreSQL to use this helper.")
        return 2

    from tldw_Server_API.app.core.DB_Management.DB_Backups import restore_postgres_backup

    status = restore_postgres_backup(backend, dump_file=str(dump_file), drop_first=(not args.no_drop))
    if status != "ok":
        logger.error(f"Restore failed: {status}")
        return 1
    logger.info("Restore completed successfully")
    print("ok")
    return 0


def _restore_bundle(args) -> int:
    """Recover an explicitly acknowledged database bundle with no application pool."""
    from tldw_Server_API.app.core.DB_Management import DB_Backups
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.exceptions import BundleError, UnsupportedBackupBackendError
    from tldw_Server_API.app.services import admin_bundle_service as bundles
    from tldw_Server_API.app.services import admin_data_ops_service as data_ops

    if not args.confirm_offline or args.no_drop or not math.isfinite(args.timeout) or not 0 < args.timeout <= 600:
        logger.error("Database recovery requires stopped clients, clean restore and a timeout in (0, 600] seconds")
        return 2
    try:
        config = data_ops._postgresql_database_config()
    except UnsupportedBackupBackendError:
        logger.error("Database recovery requires one explicit co-located AuthNZ/content/Jobs PostgreSQL target")
        return 2
    if args.confirm_database != config.pg_database:
        logger.error("The confirmed database name must exactly match the configured target")
        return 2
    deadline = time.monotonic() + args.timeout
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, config=config)
    try:
        DB_Backups.assert_postgres_database_offline(config, timeout_seconds=deadline - time.monotonic())
        manifest = bundles._read_manifest_from_zip(args.bundle_file)
        if not isinstance(manifest, dict):
            raise ValueError("invalid_bundle_manifest")
        base = data_ops._backup_base_dir()
        os.makedirs(base, exist_ok=True)
        with bundles._validated_postgresql_archive(
            args.bundle_file, manifest, None, staging_parent=base, deadline=deadline
        ) as dump_file:
            backup_backend = SimpleNamespace(
                backend_type=BackendType.POSTGRESQL, config=data_ops._postgresql_backup_config()
            )
            safety = DB_Backups.create_postgres_backup(
                backup_backend,
                backup_dir=os.path.join(base, "postgresql"),
                label="postgresql-safety",
                timeout_seconds=deadline - time.monotonic(),
            )
            if not safety or not os.path.isfile(safety):
                raise RuntimeError("safety_snapshot_failed")
            DB_Backups.assert_postgres_database_offline(config, timeout_seconds=deadline - time.monotonic())
            status = DB_Backups.restore_postgres_backup(
                backend, dump_file=dump_file, drop_first=True, timeout_seconds=deadline - time.monotonic()
            )
            if status != "ok":
                raise RuntimeError("database_restore_failed")
        if time.monotonic() >= deadline:
            raise RuntimeError("database_recovery_deadline_exceeded")
        print(
            "Database restored. Keep clients stopped; verify data, review queued/leased Jobs and explicitly decide resume/cancel before cold restart and normal sign-in."
        )
        print(f"Safety snapshot: {safety}")
        return 0
    except DB_Backups.PostgresToolTimeout as exc:
        logger.error("{}", str(exc))
        return 1
    except (BundleError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        logger.bind(error_type=type(exc).__name__).error(
            "PostgreSQL database recovery failed; keep clients stopped and inspect the target before retrying"
        )
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL content DB backup/restore helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_backup = sub.add_parser("backup", help="Create a PostgreSQL backup using pg_dump")
    p_backup.add_argument("--backup-dir", default="./tldw_DB_Backups/postgres", help="Output directory for backups")
    p_backup.add_argument("--label", default="content", help="Label prefix for backup filename")
    p_backup.set_defaults(func=_backup)

    p_restore = sub.add_parser("restore", help="Restore a PostgreSQL backup using pg_restore")
    p_restore.add_argument("--dump-file", required=True, help="Path to .dump file created by pg_dump -F c")
    p_restore.add_argument("--no-drop", action="store_true", help="Do not drop objects before restore (omit -c)")
    p_restore.set_defaults(func=_restore)

    p_bundle = sub.add_parser("restore-bundle", help="Offline whole-database recovery, including all users and Jobs")
    p_bundle.add_argument("--bundle-file", required=True, help="Trusted explicit PostgreSQL database bundle ZIP")
    p_bundle.add_argument("--confirm-database", required=True, help="Exact configured target database name")
    p_bundle.add_argument(
        "--confirm-offline",
        action="store_true",
        help="Acknowledge all API/Jobs/Scheduler clients are stopped and prevented from reconnecting",
    )
    p_bundle.add_argument(
        "--timeout", type=float, default=600, help="Whole operation deadline in seconds (maximum 600)"
    )
    p_bundle.add_argument(
        "--no-drop",
        action="store_true",
        help="Unsupported for database bundles; supplied only to report an explicit refusal",
    )
    p_bundle.set_defaults(func=_restore_bundle)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
