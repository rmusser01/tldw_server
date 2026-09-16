# UAT177 / TASK-13260.114

## Result

Frozen minimal analytics repair: PostgreSQL uses an explicit UTC date expression for review streaks; SQLite keeps its existing substring expression. Exactly three pure analytics SELECTs opt into the existing caller-preserving read scope. No DB schema, global translator, write, or caller transaction behavior changed.

## Verification

- Required official PostgreSQL fixture run `uat177-valid-red`: 12 expected PG failures, 3 SQLite passes, zero skips.
- Same 15-case suite `uat177-green`: **15 passed, zero skips** (25.05s).
- Permanent suite covers real analytics HTTP and subsequent saved-deck queue/history reads, empty/populated history, daily retention/lapse/latency, UTC streaks under Honolulu/Kiritimati session timezones with offset inputs, workspace/global/deck/deleted visibility, real SQL failure at each of the three SELECT boundaries with recovery, and implicit/explicit caller write commit/rollback preservation.
- Ruff touched source/test: 0 findings; source baseline 0. Bandit full touched Python source: 0 findings, 0 scan errors. Existing nosec-comment warnings are tool diagnostics, not findings. `git diff --check` clean.
- Commands: source `.venv/bin/activate`, set `TLDW_UAT_EVIDENCE_LABEL`, then `node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_flashcard_analytics_backends.py -q --tb=short`. Official fixture runner enforces PostgreSQL required/no Docker autostart. No auth test_db_pool or raw database provisioning.

## Diagnosis / scope limits

See DESIGN.md and the original `.tmp/uat177-diagnosis-20260916` probe packet. The first draft permanent suite separately discovered `soft_delete_flashcard` positional-row KeyError on PostgreSQL and contained an erroneous nonexistent soft_delete_deck setup call. Both are retained in `uat177-permanent-red`; corrected analytics fixture seeds historical deleted flags directly in a transaction. That log is not the final valid RED receipt.

Distinct response datetime defects are tracked as UAT179/TASK-13260.116 and excluded here. Native acceptance remains parent-owned and pending. This change does not claim to repair every Study500, a currently poisoned live connection, or unrelated timestamp/read paths.

## Files

Exact copies, hashes, and diff: review-snapshot/, owned-manifest.json, owned.patch. New permanent test is included in the patch as an added file. No git, tracker, browser, live database, provider, or runtime modifications performed.
