"""
Rotate BYOK encryption keys in AuthNZ databases.

Usage:
  # Set BYOK_ENCRYPTION_KEY to the new key and BYOK_SECONDARY_ENCRYPTION_KEY to the old key.
  python -m Helper_Scripts.AuthNZ.rotate_byok_keys

  # Dry-run (no writes) with a smaller batch size
  python -m Helper_Scripts.AuthNZ.rotate_byok_keys --dry-run --batch-size 100
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from tldw_Server_API.app.core.Provider_Credentials.byok_rotation import rotate_byok_secrets


def _format_stats(label: str, stats) -> None:
    logger.info(
        "{}: processed={} updated={} skipped={} failed={}",
        label,
        stats.processed,
        stats.updated,
        stats.skipped,
        stats.failed,
    )


async def _run(dry_run: bool, batch_size: int) -> int:
    summary = await rotate_byok_secrets(dry_run=dry_run, batch_size=batch_size)
    for table, stats in summary.tables.items():
        _format_stats(table, stats)
    _format_stats("total", summary.total)

    if summary.total.failed:
        logger.warning("BYOK rotation completed with {} failure(s)", summary.total.failed)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate BYOK encryption keys")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview updates without writing to the database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of rows to process per batch",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(_run(dry_run=args.dry_run, batch_size=args.batch_size))
    except ValueError as exc:
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
