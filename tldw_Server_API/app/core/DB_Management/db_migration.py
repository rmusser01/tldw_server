# db_migration.py - Database Migration System
"""
Database migration system for tldw_server.

This module provides:
- Schema version tracking
- Migration execution (up/down)
- Backup before migration
- Rollback capabilities
- Migration history tracking

Based on ADR-002: Alembic-style migrations within SQLite constraints.
"""

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.distributed_lock import acquire_migration_lock

from tldw_Server_API.app.core.DB_Management.DB_Backups import (
    _sqlite_error_is_busy,
    restore_sqlite_database_file,
)
from tldw_Server_API.app.core.Utils.path_utils import resolve_path


class MigrationError(Exception):
    """Base exception for migration errors"""
    pass


class Migration:
    """Represents a single database migration"""

    def __init__(
        self,
        version: int,
        name: str,
        up_sql: str,
        down_sql: Optional[str] = None,
        description: str = "",
        idempotent: bool = False
    ):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.description = description
        self.idempotent = idempotent
        self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum for migration integrity"""
        import hashlib
        content = f"{self.version}:{self.name}:{self.up_sql}:{self.down_sql or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert migration to dictionary"""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "checksum": self.checksum,
            "up_sql": self.up_sql,
            "down_sql": self.down_sql,
            "idempotent": self.idempotent,
        }

    @classmethod
    def from_file(cls, filepath: Path) -> 'Migration':
        """Load migration from file"""
        with open(filepath) as f:
            data = json.load(f)

        return cls(
            version=data["version"],
            name=data["name"],
            up_sql=data["up_sql"],
            down_sql=data.get("down_sql"),
            description=data.get("description", ""),
            idempotent=bool(data.get("idempotent", False)),
        )


class DatabaseMigrator:
    """Handles database migrations for on-disk SQLite databases"""

    _ADD_COLUMN_RE = re.compile(
        r"^\s*ALTER\s+TABLE\s+(?P<table>[A-Za-z0-9_]+)\s+ADD\s+COLUMN\s+(?P<column>[A-Za-z0-9_]+)\b",
        re.IGNORECASE,
    )

    def __init__(self, db_path: str, migrations_dir: Optional[str] = None):
        if self._is_memory_db_path(db_path):
            raise MigrationError("DatabaseMigrator does not support in-memory database paths")
        db_path_resolved = resolve_path(Path(db_path))
        self.db_path = str(db_path_resolved)
        self._db_dir = db_path_resolved.parent

        package_migrations_dir = resolve_path(
            Path(__file__).resolve().parent / "migrations"
        )
        self._migration_roots = (self._db_dir, package_migrations_dir)
        if migrations_dir is not None:
            chosen_dir = self._validate_migrations_dir(Path(migrations_dir))
            chosen_dir.mkdir(parents=True, exist_ok=True)
        elif package_migrations_dir.exists():
            chosen_dir = package_migrations_dir
        else:
            fallback_dir = self._db_dir / "migrations"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            chosen_dir = fallback_dir

        self.migrations_dir = str(chosen_dir)

        # Backup directory
        self.backup_dir = str(self._db_dir / "backups")
        os.makedirs(self.backup_dir, exist_ok=True)

    @staticmethod
    def _sqlite_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except sqlite3.Error:
            return False
        for row in rows:
            name = row["name"] if isinstance(row, sqlite3.Row) else row[1]
            if name and str(name).lower() == column.lower():
                return True
        return False

    @staticmethod
    def _is_memory_db_path(db_path: str) -> bool:
        """Return True if db_path refers to an in-memory SQLite database."""
        raw = (db_path or "").strip()
        if raw == ":memory:":
            return True
        if raw.lower().startswith("file:"):
            lowered = raw.lower()
            return "mode=memory" in lowered or ":memory:" in lowered
        return False

    @staticmethod
    def _is_within_directory(path: Path, base_dir: Path) -> bool:
        """Return True if path is within base_dir."""
        try:
            path.relative_to(base_dir)
            return True
        except ValueError:
            return False

    def _validate_migrations_dir(self, migrations_dir: Path) -> Path:
        """Ensure migrations_dir stays within an approved root."""
        resolved = resolve_path(migrations_dir)
        if not any(
            self._is_within_directory(resolved, root) for root in self._migration_roots
        ):
            raise MigrationError(
                "Migrations directory must be within the database directory "
                f"or the packaged migrations path: {resolved}"
            )
        return resolved

    def _validate_backup_path(self, backup_path: str) -> Path:
        """Ensure backup_path exists, is a file, and is scoped to the backup directory."""
        resolved = resolve_path(Path(backup_path))
        backup_dir = resolve_path(Path(self.backup_dir))
        if not self._is_within_directory(resolved, backup_dir):
            raise MigrationError(
                f"Backup path is outside the backup directory: {resolved}"
            )
        if not resolved.exists():
            raise MigrationError(f"Backup not found: {resolved}")
        if not resolved.is_file():
            raise MigrationError(f"Backup path is not a file: {resolved}")
        return resolved

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize_migration_table(self):
        """Create migration tracking table if it doesn't exist"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TIMESTAMP NOT NULL,
                    execution_time REAL NOT NULL,
                    success BOOLEAN NOT NULL DEFAULT 1,
                    error_message TEXT
                )
            """)
            conn.commit()
            logger.info("Migration tracking table initialized")

    def get_current_version(self) -> int:
        """Get current schema version from database"""
        self.initialize_migration_table()

        with self._get_connection() as conn:
            # Try new migration table first
            result = conn.execute("""
                SELECT MAX(version) as version
                FROM schema_migrations
                WHERE success = 1
            """).fetchone()

            if result and result["version"] is not None:
                return result["version"]

            # Fall back to old schema_version table if exists
            try:
                result = conn.execute("""
                    SELECT version FROM schema_version LIMIT 1
                """).fetchone()
                if result:
                    return result["version"]
            except sqlite3.OperationalError:
                pass

            return 0

    def get_applied_migrations(self) -> list[dict[str, Any]]:
        """Get list of applied migrations"""
        self.initialize_migration_table()

        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT version, name, checksum, applied_at, execution_time
                FROM schema_migrations
                WHERE success = 1
                ORDER BY version
            """)

            return [dict(row) for row in cursor]

    def load_migrations(self) -> list[Migration]:
        """Load all migrations from the migrations directory"""
        migrations = []

        logger.info(f"Loading migrations from directory: {self.migrations_dir}")

        # Check if directory exists
        if not os.path.exists(self.migrations_dir):
            logger.warning(f"Migrations directory does not exist: {self.migrations_dir}")
            return migrations

        seen_versions: dict[int, Path] = {}

        for filepath in sorted(Path(self.migrations_dir).glob("*")):
            suffix = filepath.suffix.lower()
            try:
                if suffix == ".json":
                    migration = Migration.from_file(filepath)
                elif suffix == ".sql":
                    migration = self._load_sql_migration(filepath)
                else:
                    continue
            except Exception as e:
                raise MigrationError(
                    f"Invalid migration set: {filepath.name}"
                ) from e

            if not migration:
                continue

            previous = seen_versions.get(migration.version)
            if previous:
                raise MigrationError(
                    "Duplicate migration version "
                    f"{migration.version}: {previous.name} and {filepath.name}"
                )

            migrations.append(migration)
            seen_versions[migration.version] = filepath
            logger.debug(f"Loaded migration: {migration.name} (v{migration.version})")

        # Sort by version to ensure deterministic execution order
        migrations.sort(key=lambda m: m.version)
        return migrations

    @staticmethod
    def _strip_sql_comments(sql: str) -> str:
        lines = []
        for line in sql.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_version_from_sql(filepath: Path, sql: str) -> int:
        match = re.match(r"(\d+)", filepath.stem)
        if match:
            return int(match.group(1))

        for line in sql.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("-- version:"):
                try:
                    return int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    break

        raise ValueError(f"Unable to determine migration version for {filepath.name}")

    @staticmethod
    def _extract_name_from_sql(filepath: Path) -> str:
        match = re.match(r"\d+_(.+)", filepath.stem)
        if match:
            return match.group(1)
        return filepath.stem

    @staticmethod
    def _extract_description_from_sql(sql: str) -> str:
        for line in sql.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("-- description:"):
                return stripped.split(":", 1)[1].strip()
        return ""

    def _load_sql_migration(self, filepath: Path) -> Optional[Migration]:
        try:
            sql_text = filepath.read_text()
        except OSError as exc:
            raise MigrationError(
                f"Unable to read migration file: {filepath.name}"
            ) from exc

        executable_sql = self._strip_sql_comments(sql_text)
        if not executable_sql:
            logger.debug(f"Skipping SQL migration with no executable statements: {filepath.name}")
            return None

        version = self._extract_version_from_sql(filepath, sql_text)
        name = self._extract_name_from_sql(filepath)
        description = self._extract_description_from_sql(sql_text)

        return Migration(
            version=version,
            name=name,
            up_sql=sql_text,
            down_sql=None,
            description=description,
        )

    def create_backup(self, description: str = "") -> str:
        """Create a backup of the database before migration"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version = self.get_current_version()

        backup_filename = f"backup_v{version}_{timestamp}.db"
        if description:
            safe_desc = "".join(c for c in description if c.isalnum() or c in "-_")[:50]
            backup_filename = f"backup_v{version}_{timestamp}_{safe_desc}.db"

        backup_path = os.path.join(self.backup_dir, backup_filename)

        # Create a consistent snapshot even when WAL journaling is enabled.
        try:
            with sqlite3.connect(self.db_path) as source, sqlite3.connect(backup_path) as target:
                try:
                    source.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.OperationalError:
                    # Likely not in WAL mode; ignore.
                    pass
                source.backup(target)
        except sqlite3.Error as exc:
            raise MigrationError(f"Failed to create SQLite backup: {exc}") from exc

        # Create backup metadata
        metadata = {
            "original_path": self.db_path,
            "backup_path": backup_path,
            "timestamp": timestamp,
            "version": version,
            "description": description,
            "size": os.path.getsize(backup_path)
        }

        metadata_path = backup_path + ".json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Created backup: {backup_filename}")
        return backup_path

    def execute_migration(self, migration: Migration, direction: str = "up") -> float:
        """Execute a single migration"""
        sql = migration.up_sql if direction == "up" else migration.down_sql
        if direction == "down":
            if sql is None or not str(sql).strip():
                raise MigrationError(
                    f"Migration {migration.name} (v{migration.version}) does not define down_sql; downgrade is not supported."
                )
        if sql is None:
            raise MigrationError(
                f"Migration {migration.name} (v{migration.version}) does not define executable SQL."
            )
        start_time = datetime.now(timezone.utc)

        with self._get_connection() as conn:
            try:
                # Clean up any prior failed attempt for this version so retries work
                if direction == "up":
                    conn.execute(
                        "DELETE FROM schema_migrations WHERE version = ? AND success = 0",
                        (migration.version,),
                    )
                    conn.commit()

                # Execute migration SQL
                if direction == "up" and migration.idempotent:
                    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
                    for statement in statements:
                        match = self._ADD_COLUMN_RE.match(statement)
                        if match:
                            table = match.group("table")
                            column = match.group("column")
                            if self._sqlite_column_exists(conn, table, column):
                                logger.info(
                                    "Skipping duplicate column {}.{} for migration {}",
                                    table,
                                    column,
                                    migration.name,
                                )
                                continue
                        conn.execute(statement)
                else:
                    if ";" in sql:
                        # Multiple statements
                        conn.executescript(sql)
                    else:
                        # Single statement
                        conn.execute(sql)

                conn.commit()

                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                # Record successful migration
                if direction == "up":
                    conn.execute("""
                        INSERT INTO schema_migrations
                        (version, name, checksum, applied_at, execution_time, success)
                        VALUES (?, ?, ?, ?, ?, 1)
                    """, (
                        migration.version,
                        migration.name,
                        migration.checksum,
                        datetime.now(timezone.utc),
                        execution_time
                    ))
                else:
                    # Remove migration record on rollback
                    conn.execute("""
                        DELETE FROM schema_migrations WHERE version = ?
                    """, (migration.version,))

                conn.commit()

                # Update schema_version table if it exists (for compatibility with MediaDB)
                try:
                    if direction == "up":
                        conn.execute("UPDATE schema_version SET version = ? WHERE 1=1", (migration.version,))
                    else:
                        # On downgrade, set to previous version
                        conn.execute("UPDATE schema_version SET version = ? WHERE 1=1", (migration.version - 1,))
                    conn.commit()
                except sqlite3.OperationalError:
                    # Table doesn't exist, that's fine
                    pass

                logger.info(
                    f"Executed migration {migration.name} ({direction}) "
                    f"in {execution_time:.2f}s"
                )

                return execution_time

            except Exception as e:
                conn.rollback()

                # Record failed migration
                if direction == "up":
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO schema_migrations
                            (version, name, checksum, applied_at, execution_time,
                             success, error_message)
                            VALUES (?, ?, ?, ?, ?, 0, ?)
                        """, (
                            migration.version,
                            migration.name,
                            migration.checksum,
                            datetime.now(timezone.utc),
                            (datetime.now(timezone.utc) - start_time).total_seconds(),
                            str(e)
                        ))
                        conn.commit()
                    except Exception as log_err:
                        logger.debug(f"Failed to log migration failure: migration={migration.name}, error={log_err}")

                raise MigrationError(
                    f"Migration {migration.name} ({direction}) failed: {e}"
                ) from e

    def migrate_to_version(
        self,
        target_version: Optional[int] = None,
        create_backup: bool = True
    ) -> dict[str, Any]:
        """Migrate database to target version"""
        redis_url = os.getenv("REDIS_URL")
        lock_dir = str(Path(self.db_path).parent)

        with acquire_migration_lock(
            lock_dir=lock_dir,
            lock_name="db_migration",
            redis_url=redis_url,
            timeout=60,
        ):
            return self._migrate_to_version_locked(target_version, create_backup)

    def _migrate_to_version_locked(
        self,
        target_version: Optional[int] = None,
        create_backup: bool = True,
    ) -> dict[str, Any]:
        """Inner migration logic, called while holding the distributed lock."""
        current_version = self.get_current_version()
        migrations = self.load_migrations()
        available_versions = [migration.version for migration in migrations]

        if target_version is None:
            # Migrate to latest
            target_version = migrations[-1].version if migrations else 0

        logger.info(
            f"Migrating from version {current_version} to {target_version}"
        )

        # Determine migrations to apply
        if target_version > current_version:
            # Upgrade
            to_apply = [
                m for m in migrations
                if current_version < m.version <= target_version
            ]
            direction = "up"
        elif target_version < current_version:
            # Downgrade
            to_apply = [
                m for m in reversed(migrations)
                if target_version < m.version <= current_version
            ]
            direction = "down"
        else:
            # Already at target version
            return {
                "status": "no_change",
                "current_version": current_version,
                "target_version": target_version,
                "migrations_applied": []
            }

        self._validate_migration_plan(
            current_version=current_version,
            target_version=target_version,
            available_versions=available_versions,
            direction=direction,
            to_apply=to_apply,
        )

        if not to_apply:
            return {
                "status": "no_migrations",
                "current_version": current_version,
                "target_version": target_version,
                "migrations_applied": [],
                "migrations_dir": self.migrations_dir,
                "available_versions": available_versions,
                "missing_versions": [],
            }

        # Create backup before migration
        backup_path = None
        if create_backup:
            backup_path = self.create_backup(
                f"before_migration_to_v{target_version}"
            )

        # Apply migrations
        applied = []
        total_time = 0.0

        try:
            for migration in to_apply:
                execution_time = self.execute_migration(migration, direction)
                applied.append({
                    "version": migration.version,
                    "name": migration.name,
                    "direction": direction,
                    "execution_time": execution_time
                })
                total_time += execution_time

            new_version = self.get_current_version()

            return {
                "status": "success",
                "previous_version": current_version,
                "current_version": new_version,
                "target_version": target_version,
                "migrations_applied": applied,
                "total_execution_time": total_time,
                "backup_path": backup_path
            }

        except Exception as e:
            logger.error(f"Migration failed: {e}")

            # Attempt rollback to original version
            if backup_path and create_backup:
                logger.info(f"Migration failed. Backup available at: {backup_path}")

            raise MigrationError(f"Migration failed: {e}") from e

    @staticmethod
    def _get_missing_versions(
        start_version: int,
        end_version: int,
        available_versions: list[int],
    ) -> list[int]:
        available_set = set(available_versions)
        return [
            version
            for version in range(start_version, end_version + 1)
            if version not in available_set
        ]

    def _validate_migration_plan(
        self,
        *,
        current_version: int,
        target_version: int,
        available_versions: list[int],
        direction: str,
        to_apply: list[Migration],
    ) -> None:
        if direction == "up":
            missing_versions = self._get_missing_versions(
                current_version + 1,
                target_version,
                available_versions,
            )
        else:
            missing_versions = self._get_missing_versions(
                target_version + 1,
                current_version,
                available_versions,
            )

        if missing_versions:
            raise MigrationError(f"Missing migration versions: {missing_versions}")

        if direction == "down":
            missing_down_sql = [
                migration.version
                for migration in to_apply
                if migration.down_sql is None or not str(migration.down_sql).strip()
            ]
            if missing_down_sql:
                raise MigrationError(
                    "Cannot downgrade; missing down_sql for migration versions: "
                    f"{missing_down_sql}"
                )

    def rollback_to_backup(self, backup_path: str):
        """Restore database from backup"""
        backup_path_resolved = self._validate_backup_path(backup_path)
        backup_path_str = str(backup_path_resolved)

        # Create a safety backup of current state
        safety_backup = self.create_backup("before_rollback")

        try:
            # Restore via SQLite backup API to avoid unsafe raw file replacement.
            restore_sqlite_database_file(
                source_db_path=backup_path_str,
                target_db_path=self.db_path,
                lock_timeout_seconds=0.5,
            )
            logger.info(f"Rolled back to backup: {backup_path}")

            return {
                "status": "success",
                "restored_from": backup_path,
                "safety_backup": safety_backup
            }

        except sqlite3.Error as exc:
            if _sqlite_error_is_busy(exc):
                raise MigrationError(
                    "Rollback failed: target database is busy/locked; stop active clients and retry"
                ) from exc
            raise MigrationError(f"Rollback failed: {exc}") from exc
        except Exception as e:
            raise MigrationError(f"Rollback failed: {e}") from e

    def verify_migrations(self) -> list[dict[str, Any]]:
        """Verify integrity of applied migrations"""
        issues = []
        applied = self.get_applied_migrations()
        available = {m.version: m for m in self.load_migrations()}
        applied_versions = {migration_record["version"] for migration_record in applied}

        if available:
            available_versions = sorted(available)
            expected_versions = range(available_versions[0], available_versions[-1] + 1)
            missing_versions = [
                version
                for version in expected_versions
                if version not in available and version not in applied_versions
            ]

            for version in missing_versions:
                issues.append(
                    {
                        "version": version,
                        "issue": "migration_version_gap",
                        "message": (
                            f"Migration file for version {version} "
                            "is missing from the available set"
                        ),
                    }
                )

        for migration_record in applied:
            version = migration_record["version"]

            if version not in available:
                issues.append(
                    {
                        "version": version,
                        "issue": "migration_file_missing",
                        "message": f"Migration file for version {version} not found",
                    }
                )
                continue

            migration = available[version]
            if migration.checksum != migration_record["checksum"]:
                issues.append(
                    {
                        "version": version,
                        "issue": "checksum_mismatch",
                        "message": f"Checksum mismatch for migration {version}",
                        "expected": migration.checksum,
                        "actual": migration_record["checksum"],
                    }
                )

        return issues


# Convenience functions
def get_migrator(db_path: str) -> DatabaseMigrator:
    """Get a configured migrator instance"""
    return DatabaseMigrator(db_path)


def migrate_database(
    db_path: str,
    target_version: Optional[int] = None,
    create_backup: bool = True
) -> dict[str, Any]:
    """Migrate database to target version"""
    migrator = get_migrator(db_path)
    return migrator.migrate_to_version(target_version, create_backup)


__all__ = [
    'Migration',
    'MigrationError',
    'DatabaseMigrator',
    'get_migrator',
    'migrate_database'
]
