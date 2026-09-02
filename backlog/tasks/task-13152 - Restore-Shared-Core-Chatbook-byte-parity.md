---
id: TASK-13152
title: Restore Shared Core Chatbook byte parity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 00:05'
updated_date: '2026-09-02 00:12'
labels: []
dependencies: []
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - tldw_Server_API/tests/Personalization/test_personal_context_contract.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the server's vendored tldw-profile-core source snapshot to the exact current Chatbook contract bytes so the pinned cross-application parity guard passes without changing behavior, schemas, or the expected digest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server packages/tldw_profile_core/src/tldw_profile_core/models.py is byte-for-byte identical to the current Chatbook origin/dev counterpart.
- [x] #2 The pinned exact parity test, the full server Personal Context contract module, the Shared Core public contract tests, and the accessible Chatbook canonical contract suite pass without changing the expected digest.
- [x] #3 No runtime behavior, public API, canonical serialization, or schema changes are introduced.
- [x] #4 The implementation diff is limited to the parity-only models.py edit plus task tracking, with compile, lint, Bandit, diff, and security scope checks recorded.
- [x] #5 ADR required: no; existing backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs exact shared-contract parity.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm current server origin/dev and current Chatbook origin/dev provenance, reproduce the existing pinned parity test RED, and record both digests.
2. Apply the smallest byte-only edit to server packages/tldw_profile_core/src/tldw_profile_core/models.py so it exactly matches Chatbook; do not modify tests or the pinned digest.
3. Run the exact GREEN test, full server contract module, Shared Core public contract tests, accessible Chatbook canonical suite, byte comparison, compile, lint, Bandit, and diff/security checks.
4. Commit the code fix, independently review the narrow diff, complete task evidence and ADR disposition, then commit closeout.
5. Push codex/personal-context-parity-fix, open a PR to dev, wait for required checks/reviews, and leave merge blocked until the requester supplies the human-authored Change summary.

ADR required: no
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-002 already defines the peer contract and exact conformance boundary; this repair restores its pinned byte parity without changing architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the server Shared Core snapshot by reverting only the server-only import-order and blank-line drift in packages/tldw_profile_core/src/tldw_profile_core/models.py. The resulting file is byte-for-byte identical to current Chatbook origin/dev fce1774490, with matching SHA-256 e6ced5cd8ae3ab5d4d458d30567635c4ddd2cdf76d456d9844e0bb2dd4df8fdb. No test, digest, schema, API, serialization, dependency, or runtime behavior was changed.

Root cause and TDD: untouched server origin/dev abdc60ae89 reproduced test_server_pins_exact_chatbook_profile_core_contract RED with actual digest 6bfc0521da2646ff55d00f92f5562db54aea616cfe84f718ed11fd6d0ef1883e versus pinned 421672c5cc0e43481280b3cf5a5a63fe01f44bf33255353e1cd9a6dbc2f2e7d0. Server-only commit f51f2ac2bd had changed exactly these bytes. After the minimal edit, the unchanged exact test passed.

Fresh exact-head verification at code commit 25e81b9ff5: exact pinned parity 1 passed; full server Personal Context contract module 3 passed; Shared Core public contract 4 passed; current Chatbook origin/dev canonical package suite 151 passed from a clean temporary clone; byte comparison passed; compileall passed; Ruff passed every applicable rule with only I001 excluded because applying that import-sort rule recreates the proven byte-parity defect; Bandit scanned 256 LOC with zero findings, errors, nosec suppressions, or skipped tests; origin/dev ancestry, git diff --check, scope review, and whole-range self-review passed. The default Ruff I001 diagnostic is a deliberate exact-contract exception, not an unreviewed skip; server CI lints tldw_Server_API rather than the vendored packages tree. No required verification was skipped; a full unrelated repository sweep was not run because the requested complete contract and canonical suites were executed.

ADR required: no. Existing backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md applies. No documentation beyond this task record was needed. The future PR must remain unmerged until the requester supplies the repository-required human-authored Change summary explaining what changed and why.

Delivery: opened PR #2856, https://github.com/rmusser01/tldw_server/pull/2856, against dev with an explicit merge block pending the requester-authored Change summary.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored exact Chatbook/server Shared Core byte parity with a two-line source-only correction, preserved all behavior and pinned digests, and completed the required contract, canonical, compile, lint, security, diff, and self-review evidence. ADR-002 applies; no new ADR was required. Merge remains subject to the requester-authored Change summary gate.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
