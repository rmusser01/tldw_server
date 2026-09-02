---
id: TASK-13152
title: Restore Shared Core Chatbook byte parity
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 00:05'
updated_date: '2026-09-02 00:05'
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
- [ ] #1 Server packages/tldw_profile_core/src/tldw_profile_core/models.py is byte-for-byte identical to the current Chatbook origin/dev counterpart.
- [ ] #2 The pinned exact parity test, the full server Personal Context contract module, the Shared Core public contract tests, and the accessible Chatbook canonical contract suite pass without changing the expected digest.
- [ ] #3 No runtime behavior, public API, canonical serialization, or schema changes are introduced.
- [ ] #4 The implementation diff is limited to the parity-only models.py edit plus task tracking, with compile, lint, Bandit, diff, and security scope checks recorded.
- [ ] #5 ADR required: no; existing backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs exact shared-contract parity.
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
