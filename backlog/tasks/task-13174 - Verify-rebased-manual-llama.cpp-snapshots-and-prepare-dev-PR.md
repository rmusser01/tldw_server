---
id: TASK-13174
title: Verify rebased manual llama.cpp snapshots and prepare dev PR
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 14:30'
updated_date: '2026-09-05 15:11'
labels: []
dependencies: []
documentation:
  - Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the rebased manual snapshot implementation before opening a dev PR, retaining explicit evidence boundaries and outstanding acceptance work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Targeted backend and frontend tests and browser/runtime UAT are run with exact outcomes and remaining acceptance gaps recorded.
- [ ] #2 PR targets current dev and clearly distinguishes verified behavior from production support and client-routing limitations.
- [x] #3 Ordinary profiles without snapshot state remain deletable on unsupported snapshot platforms; retained snapshots still block profile deletion even when snapshots disabled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Verify rebase preserves dev changes; run targeted tests/static checks; exercise disposable real managed snapshot lifecycle with actual Admin browser where available and verify Pause/Resume semantics; record evidence and limitations; independent review; create PR against dev only after testing/UAT. ADR required: no new ADR. Existing Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md governs; verification does not change boundaries. Do not mutate production profiles or open support gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased on dev 2742468a19. Fixed reviewed missing-fcntl ordinary-profile deletion regression in 3e35a31eff; scoped re-review approved. Fresh targeted verification: 232 backend passed, 1 opt-in live skip, 6 warnings; 64 UI passed; earlier separate profile/cache run 21 passed. Actual disposable Admin/native UAT completed Save, Stop/Start, Restore, reload recovery, Pause/Resume cold control and Delete; warm reuse 1266 tokens plus 10 new versus cold 0 plus 1276. Light/dark 390px layout and confirmation focus checked. Evidence and explicit fixture/production/client limits: Docs/Guides/llamacpp-snapshots-uat-2026-09-05.md. Existing ADR-043 applies. Original model/client acceptance remains open; production allowlist empty. Backlog ID collisions await direct-renumbering approval. Draft dev PR pending.
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
