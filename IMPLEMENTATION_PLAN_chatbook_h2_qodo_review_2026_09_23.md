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
