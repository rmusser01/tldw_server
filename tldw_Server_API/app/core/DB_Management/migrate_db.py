# migrate_db.py - Database Migration CLI Tool
"""
Command-line tool for managing database migrations.

Usage:
    python migrate_db.py status              # Show current version and pending migrations
    python migrate_db.py migrate             # Migrate to latest version
    python migrate_db.py migrate --version N # Migrate to specific version
    python migrate_db.py rollback N          # Rollback to version N
    python migrate_db.py verify              # Verify migration integrity
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def get_default_db_path() -> str:
    """Get default database path from environment or fallback"""
    # Check environment variable
    db_path = os.environ.get("TLDW_DB_PATH")
    if db_path:
        return db_path

    # Default to per-user Media DB v2 path using centralized utils
    try:
        return str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
    except Exception as exc:
        raise RuntimeError(
            "Failed to resolve default Media DB path via DatabasePaths. "
            "Set TLDW_DB_PATH or configure USER_DB_BASE_DIR."
        ) from exc


def show_status(db_path: str):
    """Show current database version and migration status"""
    migrator = DatabaseMigrator(db_path)

    current_version = migrator.get_current_version()
    print(f"\nDatabase: {db_path}")
    print(f"Current version: {current_version}")

    # Show applied migrations
    applied = migrator.get_applied_migrations()
    if applied:
        print(f"\nApplied migrations ({len(applied)}):")
        for m in applied:
            print(f"  v{m['version']}: {m['name']} (applied {m['applied_at']})")
    else:
        print("\nNo migrations applied yet.")

    # Show available migrations
    available = migrator.load_migrations()
    pending = [m for m in available if m.version > current_version]

    if pending:
        print(f"\nPending migrations ({len(pending)}):")
        for m in pending:
            print(f"  v{m.version}: {m.name} - {m.description}")
    else:
        print("\nNo pending migrations.")

    # Verify integrity
    issues = migrator.verify_migrations()
    if issues:
        print(f"\n⚠️  Migration integrity issues found ({len(issues)}):")
        for issue in issues:
            print(f"  - v{issue['version']}: {issue['message']}")


def migrate(
    db_path: str,
    target_version: Optional[int] = None,
    create_backup: bool = True,
) -> None:
    """Run database migrations"""
    migrator = DatabaseMigrator(db_path)

    current = migrator.get_current_version()
    target = target_version

    if target is None:
        # Get latest version
        migrations = migrator.load_migrations()
        if not migrations:
            print("No migrations found.")
            return
        target = migrations[-1].version

    print(f"\nMigrating database from version {current} to {target}...")

    try:
        result = migrator.migrate_to_version(target, create_backup=create_backup)

        if result["status"] == "success":
            print("\n✅ Migration successful!")
            print(f"   Previous version: {result['previous_version']}")
            print(f"   Current version: {result['current_version']}")

            if result["migrations_applied"]:
                print("\n   Migrations applied:")
                for m in result["migrations_applied"]:
                    print(f"   - v{m['version']}: {m['name']} ({m['direction']}) in {m['execution_time']:.2f}s")

            print(f"\n   Total time: {result['total_execution_time']:.2f}s")

            if result.get("backup_path"):
                print(f"   Backup saved: {result['backup_path']}")

        elif result["status"] == "no_change":
            print(f"\nDatabase already at version {target}. No changes needed.")

        else:
            print(f"\n❌ Migration failed: {result}")

    except MigrationError as e:
        print(f"\n❌ Migration error: {e}")
        sys.exit(1)


def rollback(db_path: str, target_version: int):
    """Rollback database to a previous version"""
    migrator = DatabaseMigrator(db_path)

    current = migrator.get_current_version()

    if target_version >= current:
        print(f"Cannot rollback to version {target_version} (current: {current})")
        return

    print(f"\nRolling back database from version {current} to {target_version}...")

    try:
        result = migrator.migrate_to_version(target_version)

        if result["status"] == "success":
            print("\n✅ Rollback successful!")
            print(f"   Rolled back from: {result['previous_version']}")
            print(f"   Current version: {result['current_version']}")

            if result["migrations_applied"]:
                print("\n   Migrations rolled back:")
                for m in result["migrations_applied"]:
                    print(f"   - v{m['version']}: {m['name']} (down) in {m['execution_time']:.2f}s")

    except MigrationError as e:
        print(f"\n❌ Rollback error: {e}")
        sys.exit(1)


def verify(db_path: str):
    """Verify migration integrity"""
    migrator = DatabaseMigrator(db_path)

    print(f"\nVerifying migration integrity for: {db_path}")

    issues = migrator.verify_migrations()

    if not issues:
        print("\n✅ All migrations verified successfully!")
    else:
        print(f"\n❌ Found {len(issues)} integrity issues:")
        for issue in issues:
            print(f"\n  Version {issue['version']}:")
            print(f"    Issue: {issue['issue']}")
            print(f"    Message: {issue['message']}")
            if "expected" in issue:
                print(f"    Expected: {issue['expected']}")
                print(f"    Actual: {issue['actual']}")


def main():
    parser = argparse.ArgumentParser(
        description="Database migration tool for tldw_server"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=get_default_db_path(),
        help="Path to database file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show migration status")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument(
        "--version",
        type=int,
        help="Target version (default: latest)"
    )
    migrate_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup"
    )

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument(
        "version",
        type=int,
        help="Target version to rollback to"
    )

    # Verify command
    subparsers.add_parser("verify", help="Verify migration integrity")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check if database exists
    if not os.path.exists(args.db_path) and args.command != "migrate":
        print(f"Database not found: {args.db_path}")
        sys.exit(1)

    # Execute command
    if args.command == "status":
        show_status(args.db_path)
    elif args.command == "migrate":
        migrate(args.db_path, args.version, create_backup=not args.no_backup)
    elif args.command == "rollback":
        rollback(args.db_path, args.version)
    elif args.command == "verify":
        verify(args.db_path)


if __name__ == "__main__":
    main()
