# UAT173 / TASK-13260.110 — conversation quota count rows

## Bounded design

The actual PostgreSQL chat quota count SELECT aliases its aggregate as `cnt`, but reads `row[0]`. PostgreSQL returns a dictionary, so this raises KeyError. The actual create-chat route correctly fails closed with HTTP503 when quota enforcement cannot read the count.

Change only the aggregate accessor to `row["cnt"]`, supported by PostgreSQL dictionaries and SQLite Row. Preserve SQL predicates, count meaning, optional scopes, authentication, quota thresholds, and fail-closed handling. Adjacent by-character and paged-search counts already support mapping rows; inspect/control them without unrelated refactoring.

## Verification

1. Add one focused backend test file using temporary SQLite and official `pg_database_config`/`pg_temp_db` fixtures, never AuthNZ `test_db_pool` against the runner's administrative DSN. Exercise empty counts and owner/global/workspace/deleted/character filters with literal expected values, plus adjacent per-character counts.
2. Exercise the actual create-chat FastAPI route and real quota limiter: successful character creation below the configured cap, rejection at the cap without another write, and a controlled count-storage exception still producing503 without a write. Override authentication/DB dependencies and unrelated sync capture; disable request-frequency enforcement only, retaining actual quota checks. No provider inference or application runtime is started.
3. Preserve actual PostgreSQL RED, apply the one-line repair, run required PostgreSQL/SQLite GREEN with no skips, adjacent existing regressions, Ruff/Bandit and independent review. Root owns native character creation/retry acceptance, runtime, tracker and commits.

## Owned paths

- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
- `tldw_Server_API/tests/DB_Management/test_conversation_quota_count_backends.py`
- Private evidence under this directory only.

No edits to ChaChaNotes_DB.py or note_store.py (UAT171 ownership).
