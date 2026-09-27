# Chatbook H1 Qodo Review Implementation Plan

**Goal:** Resolve verified Qodo findings on PR #2968 without weakening history ownership, and keep stacked PR #3002 aligned for review and merge.

**ADR check:** ADR required: yes. `Docs/ADR/049-chat-history-selection-ownership.md` records the durable owner-bound history and local-copy rule; existing `Docs/ADR/011-audio-api-semantics.md` covers the audio-auth alignment in this PR.

**Baseline:** Server `dev` `158287db30a28450e79fdbc38253168a468be88f`; H1 PR #2968 at `1f5a2016f3758014f88c20d6144d8fe3bcfb73c6` before review fixes; H2 PR #3002 stacked at `454635c33cf298ff10dde7e558231209c951e255`. Use only the isolated H1/H2 worktrees. Do not touch the main/UAT worktree or H2's untracked production evidence.

## Stage 1: Classify and reproduce Qodo findings
**Goal:** Read all inline comments, confirm behavior against source and existing patterns, and record a fix or reasoned rejection for each finding.
**Success Criteria:** Every H1 Qodo comment is mapped to a root cause and test path; H2 comments are collected when posted.
**Tests:** Focused existing selected-history, ordinary-chat, character-message, and route tests to establish baseline; new behavioral regressions must fail before fixes.
**Status:** Complete

## Stage 2: Repair confirmed behavior and security regressions
**Goal:** Fix idle-controller send/edit/delete, source-switch review state, server deletion display reconciliation, unresolved skill-directory admission, and any other confirmed high-impact H1 defects.
**Success Criteria:** Selected history keeps owner fences while ordinary chat paths remain usable; selected-history admission fails closed when skill visibility cannot be established.
**Tests:** Targeted Vitest for both chat hooks and review component, plus focused Python endpoint/security tests, each with red/green evidence.
**Status:** Complete

## Stage 3: Resolve remaining review items and stacked H2
**Goal:** Address validated async database, typing, documentation, equality, fixture, and cross-repository-contract findings with the smallest safe changes; process H2 Qodo feedback against its internal-storage scope.
**Success Criteria:** Every Qodo thread is fixed or answered technically; no P1/P2 review issue remains. H2 descends the final H1 head and stays scoped to groundwork.
**Tests:** Affected Python/frontend suites, SQLite/PostgreSQL fixtures when needed, Ruff/format and Bandit for touched Python.
**Status:** In Progress

The live cockpit smoke exposed one more H1 mismatch: the selected-history
regeneration handler correctly refuses a sibling-producing retry, but the runtime
rail still enabled its button and the old gate expected a provider request. The
button now reflects the same selection condition and shows a specific reason.
The focused real-server gate still selects five tests and verifies the visible
boundary after a real send.

The cross-repository audio finding also exposed a role-only Chatbook gate that
would reject wildcard-authorized server principals. The existing current-user
capabilities route now evaluates the same `RequireRole("admin")` guard as both
diagnostic endpoints and reports `can_run_audio_diagnostics`. Seventeen focused
server capability tests, Ruff, touched-source Bandit, and diff checks pass.
Chatbook PR #2822 consumes this decision; its focused tests cover wildcard,
older-server fallback, and post-check 403 behavior.

The first required CI verdict on H1 found a contradictory UI test expectation:
the H1 table rejected `PUT /api/v1/chats/{id}` even though the saved-title
scope policy intentionally permits it. The exact-base replay passed because
the contradictory row is new in H1. The repaired table and title policy pass
83 focused tests together. Backend CI also found that the new capability
field needs a refreshed OpenAPI fingerprint and generated client types. An
isolated Python 3.12 environment reproduced CI's exact schema hash
`9ca1d044cc33…`; the regenerated fingerprint passes its drift check, and
the generated TypeScript declaration builds. Seventeen capability tests,
touched-source Ruff, Bandit, and diff checks pass after the latest-dev rebase.
The range-diff exposed one overlap with newer `dev`: both the upstream change
and H1 added the owner-scope chunking test to the same shard. Removing H1's
duplicate leaves the two intended platform matrix entries. The shard coverage
guard passes with zero newly-unshared files.

## Stage 4: Verify and integrate
**Goal:** Recheck latest `origin/dev`, required CI and PR review status, then merge H1 followed by H2 only when both are qualified.
**Success Criteria:** PR heads and ancestry are verified, required checks pass, human-written Change summaries remain intact, and GitHub confirms both merge commits.
**Tests:** Fresh focused suites on final heads, `git diff --check`, touched-source Bandit, and GitHub required-check results.
**Status:** In Progress

The branch was rebased onto server `dev` `91c32e3126baad20aba2b74399927bd63004a5f7` on 2026-09-24.
The Chatbook compatibility PR was independently rebased onto its current
`dev` `f1ffa17d9e22d21c875e2b744605020ee8eb8b6e`.
Server `dev` advanced again to `0db48866a5bcb57cdfaff6b570edfe611583942a`.
H1 rebased cleanly, and range-diff showed all eight H1 commits unchanged.
The updated shard guard covers 811 shards and 4798 files with no new gaps.
Required CI is pending on the new head.
Server `dev` subsequently advanced to `a2f5e1b816cfe189db7f553a1ccf8d481dc2edbe`.
H1 rebased cleanly; all nine prior commits remain unchanged by range-diff.
The 811-shard coverage guard and diff check pass. Required CI must rerun.
The latest H1 `frontend-required` gate found two TypeScript errors in
`HistorySelectionReview`: the review identity accessed conversation fields on
the `unavailable` owner variant. The CI error reproduced locally with `tsc`.
The owner identity now excludes that unavailable variant, allowing its review
state to reset. The same full typecheck passed with CI's 8 GB Node heap;
14 focused review tests and `git diff --check` pass. The required remote gate
must rerun on the repaired head.

On 2026-09-26 the UX smoke artifact showed an early abort before the first
assistant token: only the user message remained, and regeneration correctly
reported that an assistant response is required. The test's unconditional
selected-history explanation was therefore timing-dependent. Keep the product
behavior and strict no-flaky gate. The test now verifies early-stop cleanup,
completes a real follow-up when needed, and always requires a visible assistant
message before asserting the selected-history regeneration boundary.
Local strict Stop verification also showed that awaiting response delivery can
let the request finish before Stop is clicked. After comparing existing
request-dispatch waits in the real-backend suites, this gate now waits for
the outbound request and validates its payload even when aborted before headers.
Three runs against the fresh production build pass with Stop evidence required,
retries disabled, and zero skips or flaky outcomes.

The 2026-09-26 rebase onto server dev `f5fa1f3a41` preserves all production and
test patches. Dev's Sync upload-expiry ADR now owns number 048, so this PR's
history ADR is renumbered to 049 with its accepted body unchanged. Both indexes
retain the upstream decision and the published mirror matches. The 21 new
Sync expiry tests, 14 history-review tests and updated 4800-file shard guard
pass. Fresh remote checks must qualify the rebased head.

2026-09-27: all seven required gates passed on H1 b789d5567d. Server dev advanced through MCP filesystem/build/license repairs and the license-audit checkout-depth fix to a6e51f60d5. Rebased cleanly; all 13 H1 patches are identical by range-diff. Fresh verification: 14 history-review Vitest cases pass; shard coverage passes with 811 shards, 4800 files and no new uncovered files; diff check passes. No H1 production or test source changed, so prior scoped security and real-server/browser qualification remain applicable. Fresh required CI must qualify the new latest-dev head before merge.

2026-09-27 08:04 integration: all seven required gates passed on dd03c1cc13. Server dev advanced to 8b25dc729c with license-first workflow ordering and negative license verdict propagation. H1 rebased cleanly; all 13 prior patches are identical by range-diff. Fresh history-review tests passed 14 cases and shard coverage passed with 811 shards, 4800 files and zero new uncovered files. Diff check passes. Only inherited CI changed; prior source/security/browser qualification remains applicable. Fresh required CI must qualify the latest-dev head before merge.

2026-09-27 09:04 integration: all seven required gates passed on c09caf0a58. Dev advanced to bfa343a608 through VN generation recipe snapshots. Rebase conflicts were confined to the generated OpenAPI fingerprint. Retained upstream pins during replay, reproduced contract drift, then regenerated the complete merged schema and ignored TypeScript declaration with the established Python 3.12 CI dependency environment. The fresh fingerprint is 4311385d0b9e (2099 paths, 3234 schemas); its check and generated declaration compile pass. Inspection confirms both VN recipe fields and H1 history/effective-audio-capability contracts. All prior H1 file blobs except the fingerprint remain identical to c09caf0a58; range-diff differences are only the superseded fingerprint patches. Fresh tests: 17 audio capability cases and 14 history review cases pass. Shard coverage passes with 811 shards, 4801 files and zero new uncovered files. Diff check passes. No Python source changed, so prior scoped Bandit remains applicable. Fresh required CI must qualify this latest-dev head before merge.

2026-09-27 10:04 integration: all seven required gates passed on 4b54b63e28. Server dev advanced to 0727e9ee32 with Chat Macros authoring and output profiles. Rebased all 14 H1 commits; the only conflict and changed prior file blob is the generated OpenAPI fingerprint. Reproduced contract drift and regenerated with the established Python 3.12 CI environment. Fingerprint 800ab8bead16 represents 2100 paths and 3235 schemas; its fresh check and generated declaration compile pass. Canonical schema inspection retains macro output profiles, H1 history selection, effective audio diagnostic capability, and VN recipe fields. Fresh verification: 61 macro repository/service unit tests, 16 macro API integration tests, and 14 history review tests pass; shard coverage passes with 811 shards, 4801 files and no new gaps. Diff check passes. H1 application and test source remain byte-identical, so prior scoped Bandit and browser qualification remain applicable. Fresh required CI must qualify this latest-dev head before merge.
