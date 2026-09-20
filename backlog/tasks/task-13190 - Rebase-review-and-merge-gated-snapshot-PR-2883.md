---
id: TASK-13190
title: Rebase review and merge gated snapshot PR 2883
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 16:44'
updated_date: '2026-09-05 18:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Land the approved manual snapshot implementation after current-dev verification and the requested reviewer feedback, retaining the documented production support gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR is rebased on current dev and required CI passes on its final head.
- [ ] #2 Requested dodo review is identified and all actionable PR feedback is resolved with evidence.
- [ ] #3 PR is merged into dev without opening the production runtime allowlist.
- [x] #4 The six colliding snapshot task IDs and their feature references are migrated without changing unrelated dev tasks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Fetch/rebase dev and regenerate conflicted API fingerprint; verify targeted runtime and UI tests and CI; identify dodo review and address verified findings; resolve Backlog ID collisions only with direct-edit exception approval; retain human summary and explicit acceptance limits; merge only after checks/review clear. ADR required: no new ADR; existing Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md remains authoritative.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased onto dev dc0b7455f2, resolving the single generated fingerprint conflict by rerunning canonical export/type generation. Published rebased head 54802df88b with exact previous-head force-with-lease and marked PR ready for review. Fresh targeted backend matrix: 253 passed, one opt-in live skip, six warnings. Awaiting dodo identity and approval for direct renumbering of six snapshot task files with IDs reused by dev; no merge until requested review and checks clear.

Fresh rebased shared Admin/runtime UI run: 64 tests passed across four files. Fresh canonical OpenAPI drift check passed. Current user authorized ready review/merge, not production enablement; human summary preserved verbatim in PR. Background follow-up must keep unchanged states quiet and must not merge before requested reviewer identity/review and required checks are clear.

User explicitly approved the limited direct-file exception for six-task ID migration. Checked local tasks/drafts and fetched origin/dev; next available IDs are 13184–13189. Mapping: 13159→13184, 13160→13185, 13161→13186, 13162→13187, 13163→13188, 13174→13189. Preserve existing statuses and history; change only snapshot references, preserving Personal Context references to original IDs. Verify filenames/frontmatter/dependencies, byte-identical source/published ADR, and unrelated tasks against HEAD before publishing.

Migration verified: six new IDs 13184–13189 unique across local/remote inventory, filename/frontmatter match, dependency targets exist, statuses preserved, ten unrelated colliding task files byte-identical to HEAD, unrelated Personal Context lesson text unchanged, source/published ADR byte-identical, whitespace clean. User approval is recorded in thread and scheduled workflow updated to avoid asking again. Documentation-only change; runtime tests and Bandit not applicable.

Post-migration published-docs refresh test module passed: 33 tests, seven baseline warnings. No runtime files changed.

Qodo posted 14 inline findings on PR 2883; read all complete bodies. Remediation plan under existing ADR-043: independent API boundary/types/docstrings fixes; storage root-identity, orphan publication recovery and conservative retained-state detection; bounded stable-identity fingerprint cache to remove repeated model reads; controller launch/pre-dispatch restore cleanup and public-lifecycle regression improvements; classify new tests. Require failing behavioral reproductions for bugs, preserve unknown-outcome artifacts and ownership fences, run combined targeted tests/security, regenerate API contract and respond inline with evidence before merge.

Qodo remediation integrated with RED/GREEN evidence: 338 targeted backend/docs tests passed, one explicitly opt-in live skip, ten environment/fork warnings. All 18 touched Python files pass Ruff formatting and lint (four pre-existing runner BLE001 exceptions retained); touched production Bandit has zero issues/errors. API schemas/docs regenerated. Existing ADR-043 amended with root inode fencing, conservative pending-publication journal recovery, staged cleanup and bounded fingerprint-cache rationale. Production tested-build set remains empty. Independent final review and exact-head CI remain required before merge.

Independent final review found one additional staged-filename ownership gap. A controller-level collision regression first failed with deleted evidence, then passed after cleanup required a successful staging result retained across disk-call cancellation. Updated operations module: 42 tests passed. Shared Admin/runtime UI: 64 tests passed. OpenAPI drift check and generated TypeScript typecheck passed. Strict docs build passed with existing temporary serial git-plugin config (host multiprocessing constraint; canonical repository config unchanged). No new review requests beyond the same 14 Qodo inline findings. Latest dev advanced with email-summary PR2887; rebase again before publishing.

Final review P2 re-review confirmed resolved. Rebased all 25 commits onto latest dev 69c96ef715 (PR2887); only generated fingerprint conflicts, both resolved by canonical export/type generation. Rebased review fix commit 152d8ea8af. Fresh post-rebase 13-file matrix: 339 passed, one opt-in live skip, ten warnings; generated TypeScript check passed. Await exact-head CI and any follow-up review before merge; user human Change summary preserved verbatim.

Requester explicitly approved direct renumbering of the two new Buddy collisions after latest-dev rebase: snapshot review/merge record 13183→13190 and snapshot design record 13184→13191. Buddy records retain 13183/13184 unchanged. Rebased onto dev86eb9e517c; preserved both independent entries in the only lessons-document conflict. No snapshot runtime/UI code changed during this rebase. Fresh matrix339 passed/one opt-in live skip/ten warnings; canonical fingerprint matches. Await final-head CI after publishing.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
