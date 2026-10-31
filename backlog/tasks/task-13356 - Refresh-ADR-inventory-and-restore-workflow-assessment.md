---
id: TASK-13356
title: Refresh ADR inventory and restore workflow assessment
status: Done
assignee: []
created_date: '2026-09-25 16:18'
updated_date: '2026-09-25 16:30'
labels:
  - docs
  - process
  - adr
dependencies: []
references:
  - Docs/ADR/inventory/2026-06-03-decision-inventory.md
  - Docs/ADR/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recheck the June ADR inventory against current dev, reconcile proposed records and index links, and make ADR assessment explicit in AGENTS.md and repo-local Superpowers guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record a dated evidence-based disposition for unresolved inventory rows and proposed ADRs without silently accepting decisions.
- [x] #2 Add a concise ADR assessment step to AGENTS.md and reusable repo-local Superpowers guidance.
- [x] #3 Repair ADR index ordering and missing entries, and verify links and documentation consistency.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit current dev inventory, proposed ADRs, and code evidence. 2. Add dated inventory findings and repair index. 3. Wire AGENTS.md and repo-local Superpowers guidance. 4. Review diff, verify documentation checks, finalize task, and prepare PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR check: ADR required: no new workflow ADR. ADR-001 already governs ADR adoption; this task corrects guidance, index, and inventory only. TASK-13357 owns ADR-047 for the separate implemented audio preset decision.

Verification 2026-09-25: ADR index has 47 ordered rows with matching source statuses and valid local links; source/published ADR README and June/September inventories compare byte-for-byte; git diff --check passed. Docs/process-only changes, so pytest and Bandit are not applicable; no .venv exists in the isolated worktree. Unrelated tokenizer published-doc drift from refresh was removed.

PR: https://github.com/rmusser01/tldw_server/pull/3014. Merge gate remains: requester must provide a human-written Change summary.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled the June ADR inventory against current dev, corrected ADR-040/ADR-043 index entries, and restored explicit ADR assessment in AGENTS.md and the repo-local Superpowers skill. Published ADR mirrors match. PR #3014. Docs-only: pytest/Bandit not applicable; no blockers to documentation work.
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
