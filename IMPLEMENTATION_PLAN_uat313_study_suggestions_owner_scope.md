# UAT313 Study Suggestions ownership repair

Backlog: TASK13260.251. Related worker startup/identity and test-fixture work: TASK13260.248–250. Continue the authorized repair stage before another full UAT matrix.

## Stage 1: Reproduce the owner boundary failures
**Goal**: Establish actual restricted PostgreSQL foreign reads/writes and SQLite controls.
**Success Criteria**: Retained causal failures without artificial policy bypass; positive owned lifecycle controls remain meaningful.
**Tests**: Snapshot detail/list, parent refresh, generation-link read/create/replace/finalize/delete/release, cold worker factory.
**Status**: Complete

## Stage 2: Apply existing database ownership patterns
**Goal**: Scope snapshot and generation-link operations to the canonical PostgreSQL owner, retaining SQLite per-user device compatibility.
**Success Criteria**: Minimal parameterized owner clauses and parent validation; no RLS weakening, new storage layer or queue redesign.
**Tests**: Causal cases pass; authenticated endpoint boundaries, full Study Suggestions regressions, Ruff and scoped Bandit; independent review.
**Status**: Complete

## Stage 3: Verify native fresh-install completion and isolation
**Goal**: Complete ordinary Study reviews in fresh SQLite and official restricted PostgreSQL profiles, including foreign snapshot denial.
**Success Criteria**: Visible suggestions finish, owner can reopen, foreign user cannot see/read/act on snapshots; manifests and cleanup verified.
**Tests**: Native browser flow with real API/database, read-only canonical controls; retain all harness/product failures in the running tracker.
**Status**: In Progress


Stage1 evidence: actual restricted PostgreSQL10causal failures/12controls pass, followed by22/22 after scoping. Cold worker factory2fail2warm controls pass, then4/4. Broader fixtures3fail127pass reproduce on cd06243f42; corrected independent expectations/stub. HTTP owner-positive exposes314 timestamp response500 after foreign denials pass. Related TASK13260.252 adds existing datetime-to-ISO schema normalization before native acceptance. Tests and private evidence remain under .tmp/uat310-repair.

Stage2 verification:208focused tests pass/0skips, Ruff0, Bandit0findings/errors, independent review clear. Test setup corrections and temporary-role cleanup are recorded in the tracker. Native frozen-source acceptance is next.

### Native gate finding UAT315
Both workers start on frozen488fc0f661. SQLite is ready; restricted PostgreSQL snapshot insertion fails because the shared Media-first sync_log has entity_uuid. TASK13260.253 covers extending existing fixed-column selection to snapshot/link triggers. Reproduce with real shared schema and restricted tenant, verify lifecycle/rollback and reopen, then rerun fresh native acceptance before closing Stage3.

UAT315/316 source follow-up:293backend and27frontend tests pass; causal shared-worker2fail/6pass and explicit old-schema failure retained. v72 trigger-only upgrade is transactional/serialized; current read-only opens still pass. Restricted Media-first lifecycle and foreign sync visibility verified. TopicBuilder uses supported orientation props. Ruff/Bandit clean; independent review clear. Native rerun will use a new immutable source/archive and official holder.
