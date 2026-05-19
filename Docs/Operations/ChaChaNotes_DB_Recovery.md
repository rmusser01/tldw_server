# ChaChaNotes DB Recovery

ChaChaNotes stores character cards, character chat conversations, notes, and related per-user data in each user's `ChaChaNotes.db` file. If the SQLite integrity preflight reports corruption, the API keeps starting and reports a degraded `chacha_notes` check from `/api/v1/health`.

The server does not automatically repair or replace a corrupted ChaChaNotes database. Automatic mutation could destroy the only copy of a user's character chat history, so recovery is operator-driven.

## Health Signal

When corruption is detected, `/api/v1/health` includes a sanitized failure payload:

```json
{
  "checks": {
    "chacha_notes": {
      "status": "degraded",
      "last_error": "sqlite_corruption",
      "last_failure": {
        "reason_code": "sqlite_corruption",
        "affected_db": "user:42/ChaChaNotes.db",
        "recovery": {
          "automatic_repair": false,
          "documentation": "Docs/Operations/ChaChaNotes_DB_Recovery.md"
        }
      }
    }
  }
}
```

`affected_db` intentionally avoids absolute host paths. Resolve it by checking the configured `USER_DB_BASE_DIR` and the user id in the health payload.

## Recovery Steps

1. Stop the API process or block new character chat traffic for the affected user.
2. Locate the affected file under `USER_DB_BASE_DIR/<user_id>/ChaChaNotes.db`.
3. Copy the corrupted DB and any sidecar files (`-wal`, `-shm`) to a safe backup location before making changes.
4. Restore `ChaChaNotes.db` and its sidecar files from the newest known-good backup.
5. If no backup exists and partial salvage is acceptable, attempt SQLite recovery into a new file:

```bash
sqlite3 /path/to/ChaChaNotes.db ".recover" | sqlite3 /path/to/ChaChaNotes.recovered.db
sqlite3 /path/to/ChaChaNotes.recovered.db "PRAGMA integrity_check;"
```

6. Only replace the live `ChaChaNotes.db` with the recovered DB after validation succeeds and a backup copy of the original exists.
7. If recovery is not possible, move the corrupted DB and sidecar files aside. The server will create a fresh empty ChaChaNotes DB on next access, but previous character chat data will not be present.
8. Restart the API or retry character chat access for that user.
9. Confirm `/api/v1/health` no longer reports a new `sqlite_corruption` failure for `chacha_notes`.
10. If the restored or recovered DB fails validation or causes new errors, roll back by stopping the API and restoring the backup copy made in step 3.

## Verification Commands

Use SQLite's integrity check before putting a restored DB back into service:

```bash
sqlite3 /path/to/ChaChaNotes.db "PRAGMA quick_check;"
sqlite3 /path/to/ChaChaNotes.db "PRAGMA integrity_check;"
```

Both commands should report `ok` for a healthy SQLite file.

## Notes

- Do not delete a corrupted DB until a backup copy has been made.
- Keep `ChaChaNotes.db`, `ChaChaNotes.db-wal`, and `ChaChaNotes.db-shm` together when copying or restoring.
- The degraded health status is a release blocker for first-class Character Chat because role-play sessions depend on this database for characters and conversation state.
