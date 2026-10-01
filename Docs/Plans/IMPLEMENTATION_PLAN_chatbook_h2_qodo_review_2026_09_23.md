# Chatbook H2 Qodo Review Implementation Plan

**Goal:** Resolve verified Qodo findings on PR #3002 while preserving its internal native-fork storage scope and its dependency on H1.

**ADR check:** ADR required: yes. `Docs/ADR/050-native-chat-fork-storage-lifecycle.md` records the durable operation-receipt and workspace-admission rule in this PR; `Docs/ADR/049-chat-history-selection-ownership.md` governs its inherited H1 selection boundary.

## Stage 1: Reproduce substantive findings
**Goal:** Confirm snapshot-bound assistant projection and caller-owned PostgreSQL transaction behavior against real storage paths.
**Success Criteria:** Failing focused tests demonstrate both reported defects before fixes.
**Tests:** Native fork projection and workspace lifecycle tests on the supported backends.
**Status:** Complete

## Stage 2: Repair native fork correctness and contracts
**Goal:** Fix snapshot binding and transaction ownership; apply appropriate typing, documentation, test classification, and domain-error improvements.
**Success Criteria:** The original chat stays unchanged, fork projection recognizes its saved assistant, and workspace deletion cannot commit unrelated writes.
**Tests:** Focused projection, migration, transaction, and workspace lifecycle suites.
**Status:** Complete

## Stage 3: Verify, review, and integrate
**Goal:** Answer every Qodo thread with evidence, run relevant lint/Bandit/tests, restack on the final H1 head, and merge only after required PR gates pass.
**Success Criteria:** No actionable P1/P2 issue remains and GitHub confirms final checks and merge ancestry.
**Tests:** `git diff --check`, Ruff, Bandit, targeted SQLite/PostgreSQL tests, required GitHub checks.
**Status:** In Progress

All new functions in the three H2 migration, transaction, and workspace-lifecycle
test modules now have parameter and return annotations. The focused suite passed
64 SQLite cases; 68 PostgreSQL parametrizations skipped because PostgreSQL was
unavailable locally. Ruff and `git diff --check` passed. Final PostgreSQL and
required CI evidence is still pending.

On 2026-09-24, H2 was restacked onto H1 `a7a89a80db`, which descends server
`dev` `91c32e3126`. The focused post-restack run passed 172 tests, with 68
PostgreSQL-dependent parametrizations skipped locally; `git diff --check`
passed. The current PR head is `0f308f49b3` before this documentation update.
H2 was subsequently restacked onto H1's workflow-only shard fix
`aabca0478f`; `range-diff` shows all five H2 commits unchanged. Required
GitHub checks and final H1 merge ancestry remain pending.
After server `dev` advanced to `0db48866a5`, H2 was restacked onto H1
`f3bbacf367`. All six existing H2 commits are unchanged by range-diff,
and the stacked diff check passes. Required CI is pending on the new head.
Server `dev` then advanced to `a2f5e1b816` and H2 was restacked onto H1
`fa3a36997c`. All seven prior H2 commits are unchanged by range-diff;
the stacked diff check passes. Required CI must rerun on the new head.
H1 later corrected an `unavailable` history-owner TypeScript narrowing error
and moved to `63e95039db`. H2 restacked cleanly; all eight existing H2
commits are unchanged by range-diff, and the stacked diff check passes.

On 2026-09-26, H2 was restacked onto H1 `b789d5567d`, based on server dev `f5fa1f3a41`. Upstream claimed ADR-048, so inherited history selection is now ADR-049 and native storage is ADR-050. Only ADR identifiers, cross-references, indexes and tracking changed; accepted rationale and native source remain unchanged. The first nine H2 patches are identical by range-diff. Native projection/migration/transaction/workspace regressions passed 155 tests; 68 PostgreSQL parametrizations skipped because the local fixture reported PostgreSQL unavailable. ADR source/published mirrors, unique identifiers and diff checks pass. Prior touched-source security qualification remains applicable. Fresh CI and final H1 merge ancestry remain pending.

2026-09-27: all five stacked checks passed on H2 33822b212e. Restacked onto H1 dd03c1cc13, based on server dev a6e51f60d5 after MCP filesystem/build/license and license-audit workflow repairs. All ten prior H2 patches are identical by range-diff. The native projection/migration/transaction/workspace suite passed 155 cases, with 68 PostgreSQL cases skipped because the local fixture reported PostgreSQL unavailable; diff check passes. No native source changed, so prior touched-source security qualification remains applicable. Fresh stacked CI and final H1 merge ancestry remain pending; required dev gates must pass after retargeting.

2026-09-27 08:04 integration: all stacked checks passed on d64adfb911. H2 restacked cleanly onto H1 c09caf0a58, based on latest server dev 8b25dc729c and its license-first CI ordering. All ten H2 patches are identical by range-diff. Fresh native projection/migration/transaction/workspace suite passed 155 cases with 68 PostgreSQL cases skipped because the local fixture reported PostgreSQL unavailable; diff check passes. No native source changed, so prior touched-source security qualification remains applicable. Fresh stacked CI, H1 merge ancestry and required dev gates after retargeting remain pending.

2026-09-27 09:04 integration: all stacked checks passed on 23995c11a6. H2 restacked cleanly onto H1 4b54b63e28, based on latest server dev bfa343a608 and its VN recipe snapshot merge. H1 resolved the generated OpenAPI conflict by regenerating the combined schema; no native source changed. All ten H2 patches are identical by range-diff. Fresh native projection/migration/transaction/workspace suite passed 155 cases with 68 PostgreSQL cases skipped because the local fixture reported PostgreSQL unavailable; diff check passes. Prior scoped Bandit remains applicable. Fresh stacked CI, H1 merge ancestry and required dev checks after retargeting remain pending.

2026-09-27 10:04 integration: prior stacked checks passed on 5422478fa0. Restacked cleanly onto H1 f17909a7f0, rebased on server dev 0727e9ee32 and its Chat Macros authoring/output-profile merge. H1 regenerated and verified the complete merged OpenAPI fingerprint. All ten prior H2 patches are identical by range-diff before this tracking update. Fresh native projection/migration/transaction/workspace verification passes 155 tests; 68 PostgreSQL-dependent cases skip because the shared local fixture reports PostgreSQL unavailable. Diff check passes. Native source is unchanged, so prior scoped Bandit remains applicable. Fresh stacked CI must qualify this head; all required dev gates must run after H1 merges and H2 is retargeted. The human-written Change summary remains unchanged.

2026-09-27 11:04 Persona integration: restacked H2 onto published H1 68350734e3 based on server dev 056d9adbb3. Preserved upstream immutable owner/startup provenance and locked Sync replacement/local update behavior while rejecting protected native Sync rows under the same lock. Moved native schema to SQLite 72/PostgreSQL 76 after Persona and H1 history. Catalog recognition adopts the previously published SQLite70/PostgreSQL74 native schema without replaying duplicate DDL, preserving operation receipts, history projections, native child binding/bundle and workspace admission closure. Partial native catalogs fail with rollback of the original marker and receipt. Added real historical-DDL and partial-schema tests, updated H1-baseline migration/rollback tests and the Persona PostgreSQL concurrency hook for the expanded locking read. Independent integration review found no confirmed P1/P2. Initial combined native/history/Persona suite: 191 passed, 111 PostgreSQL shared-fixture skips. Final replay qualification is recorded below after completion. Fresh scoped Ruff, migration AST duplicate-method check and diff check passed; Bandit on changed DB/conversation/native migration/operation scopes reported zero findings/errors. Shard coverage passes with 813 shards, 4812 files and no new uncovered files. Current design and plan schema versions updated; historical records retained. Human Change summary remains unchanged. Fresh stacked CI is required, followed by all required dev gates after H1 merges and H2 is retargeted; no merge performed.

Final Persona integration replay verification completed successfully: 193 tests passed, 114 PostgreSQL-dependent tests skipped solely because the shared fixture reported PostgreSQL unavailable; command exited 0 (121.84 seconds). This covers final Qodo transaction/binding fixes plus native migration/receipt/asset/workspace/projection, H1 historical migration and Persona startup/Sync behavior.

2026-09-27 12:19 integration: H1 #2968 merged into dev as 46db4688c1 after all seven enforced checks passed at head 68350734e3. H2 rebased cleanly onto the actual merge commit; all 11 patches remain identical by range-diff and application/test/workflow trees are byte-identical to the previously qualified ce264f3df0 candidate. H1 Qodo task TASK-13261.10 is finalized here as dependency integration tracking; its broader plan integration stage stays open until H2 merges. H2 is being retargeted from the H1 branch to dev; fresh enforcement of all seven dev gates is required before its separate merge. The H2 human Change summary remains unchanged. No additional native feature or byte lifecycle implementation was introduced. Fresh shard coverage passes with 813 shards, 4812 files and no new uncovered files; diff check passes. Previous scoped Bandit remains applicable because production source is unchanged.

2026-09-27 12:49 CI trigger correction: H2 targets dev at 85445806cd and the dev license policy check passes, but retargeting emitted a base-change event that the six required CI workflows do not subscribe to. Only pre-commit/SBOM/smoke/review workflows ran from the prior stacked synchronization event; the enforced backend/frontend/coverage/E2E/security/container gates were absent. Push this tracking update as a normal synchronize event after retargeting to start the required dev workflows. Do not dispatch alternative checks or bypass protection. Server dev remains H1 merge 46db4688c1; source/tests/workflows remain unchanged, so prior native/Persona/security qualification is applicable. Qodo has zero open findings and the human Change summary remains preserved.
