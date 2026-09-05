---
id: TASK-13163
title: Add manual snapshot Admin workflow and live reuse evidence
status: To Do
assignee: []
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 02:27'
labels: []
dependencies:
  - TASK-13162
documentation:
  - Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md
  - Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
  - Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make manual cache preservation understandable and accessible to first-time administrators and repeat operators.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Admin panel implements enablement without implicit restart, slot inspection, save, compatibility reasons, confirmed restore and confirmed deletion.
- [ ] #2 Operation status survives page reload, keyboard and narrow-screen flows work, and unsupported or unknown outcomes give explicit recovery guidance.
- [ ] #3 Pinned-build live save-stop-start-restore demonstrates cache reuse against a cold control; conversation and Pause/Resume semantics remain unchanged.
- [ ] #4 Operator documentation records privacy, quiescence, retention, limitations and tested compatibility.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
