## Task 255: PostgreSQL world-book creation and endpoint readback

**Backlog:** TASK13260.197. **Status:** In Progress. **Baseline:**6f6983b0620aae1f0892c6b0d3ae3bebfc105e02.

**Requirements:** Repair the observed normal Create World Book500 on actual PostgreSQL using existing supported DB lifecycle interfaces. Native Alice POST22:37:24.025UTC on a7d3155 returns Failed to create world book; backend15:37:24PDT confirms BackendConnectionWrapper context-manager error. The creation method uses an unsupported connection context; endpoint subsequently calls get_world_book and get_entries. Investigate and cover that complete endpoint path, avoiding unrelated CRUD rewrites.

**Owned scope:** tldw_Server_API/app/core/Character_Chat/world_book_manager.py and narrowly necessary tests under tldw_Server_API/tests/DB_Management/. Existing adjacent world-book tests and WorldBookService are references. Parent owns all Git, Backlog, tracker, native browser/runtime, frozen archives and model services; author must not edit these or commit. No subagents from implementer.

**Steps:**
1. Read actual manager, service, endpoint and existing portable transaction/read helpers; reproduce before implementation with official PostgreSQL fixtures plus SQLite controls.
2. Use supported write transaction and read-only lifecycle boundaries. Preserve outer caller rollback/commit ownership, standalone durability, duplicate-name conflict mapping, owner isolation, flags/defaults, soft-deletion filters and entry-count readback. No raw SQL outside existing DB abstraction, no schema/RLS/role relaxation.
3. Make the smallest repair for the actual create/readback path; run focused and relevant adjacent SQLite/PostgreSQL tests with zero skips. Record causal RED and final GREEN with commands, evidence files and exit codes.
4. Run .venv Ruff/compile/Bandit on touched Python scope, compare any existing findings rather than silently excluding new ones; self-review, write source hashes and full report. Parent dispatches independent spec/code review and original native acceptance before closure.

**Mandatory test runner:** activate .venv first, then use `TLDW_UAT_EVIDENCE_LABEL=worldbook255-<unique> node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs <test paths> -q --tb=short`. This runner uses official fixtures with explicit required PostgreSQL provisioning. Do not set no-Docker or create substitute databases. Never print/cat private credentials/configs/raw runtime logs. Retain sanitized outputs under .tmp/uat-repairs-231-246/worldbook255/. Actual browser failure is .tmp/uat-repairs-231-246/native-targeted/pg-multi/worldbook239-created.txt; safe cause extraction is worldbook255-error-cause-v2.json in that directory.

**Success criteria:** Real create+endpoint readback succeeds with persisted unique ID/metadata/entries on both DB backends; caller transactions and other-owner controls remain valid; no added static/security findings; independent review and later native Create/catalog/Character-editor readback accepted. Existing catalog239 and initialization112/Character-read153 repairs remain intact. No native runtime/source archive changes by implementer.
