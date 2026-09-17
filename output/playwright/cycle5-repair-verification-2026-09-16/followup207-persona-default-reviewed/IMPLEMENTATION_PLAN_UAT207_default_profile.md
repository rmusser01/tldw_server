# TASK13260.145 / UAT207 stages

## Stage1: Diagnose actual failure
Goal: real two-owner route and driver causality.
Success: exact collision/aborted-retry evidence; SQLite controls.
Tests: private20 / permanent34 RED.
Status: Complete.

## Stage2: Approve and implement bounded owner/default transaction contract
Goal: shared core default resolution and local PostgreSQL insert ownership.
Success: tombstones/legacy/race/caller controls pass; three paths only.
Tests: permanent36.
Status: Complete.

## Stage3: Verify and freeze
Goal: focused plus adjacent tests/static checks and immutable review snapshot.
Success:36+121+26 pass; zero skips; no new Ruff/Bandit findings.
Status: Complete.

## Stage4: Independent review and native acceptance
Goal: parent-coordinated independent review and real browser retry.
Success: independent approval plus retained native acceptance.
Status: Pending parent gate; author source is frozen.
