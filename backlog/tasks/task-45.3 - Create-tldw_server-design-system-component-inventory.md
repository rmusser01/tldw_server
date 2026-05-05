---
id: TASK-45.3
title: Create tldw_server design-system component inventory
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 03:22'
updated_date: '2026-05-05 03:29'
labels:
  - design-system
  - frontend
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1286'
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Stage 2 lightweight design-system inventory from the tldw WebUI design-system contract after the proof-surface PR merged. Scope is documentation and migration planning only: inventory current primitives/near-duplicates, canonical owners, proof-surface consumers, migration targets, and the first prioritized migration queue slice without broad component moves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inventory document covers the contract-required categories: Button, StatusBadge, Badge, Alert, FeatureEmptyState, EmptyState, loading states, recovery banners, page shells, modal footers, and admin/health panels.
- [x] #2 Inventory records canonical owners, near-duplicates, proof-surface consumers, and migration targets based on current repository files.
- [x] #3 Inventory explicitly marks non-goals and later migrations so the slice does not trigger broad component churn.
- [x] #4 Inventory prioritizes the first migration queue slice with concrete scope, risks, and suggested verification.
- [x] #5 Contract links to the inventory from the Stage 2 rollout section.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Start from the merged design-system proof-surface branch on a clean worktree from origin/dev.
2. Inspect current shared UI, Common, WebUI-local and Chat/Playground component surfaces with targeted file scans.
3. Add a Stage 2 inventory document with canonical ownership, near-duplicates, proof-surface consumers, migration targets, non-goals and first migration queue guidance.
4. Link the inventory from the design-system contract.
5. Run docs-focused verification, update this task, and commit the documentation slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1272 merged into dev at merge commit 5aa53a31d277a0b4126161aa8053fd8c322e5316. Work is isolated in .worktrees/tldw-design-system-inventory on branch codex/tldw-design-system-inventory from origin/dev.

Created Docs/Design/tldw_web_design_system_inventory.md. The inventory records canonical owners, near-duplicates, proof-surface consumers, migration targets, non-goals, and the first Chat/Playground migration queue. Updated Docs/Design/tldw_web_design_system_contract.md to link the inventory from Stage 2.

Verification: rg checks confirmed all required inventory categories and the contract link. awk line-length scan reported no lines over 120 characters in the inventory. git diff --check passed. Bandit skipped because this task touched documentation and Backlog metadata only; no Python code was touched.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1286
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 2 tldw Web Design System inventory and linked it from the design-system contract. The inventory documents current ownership for Button, StatusBadge, Badge, Alert, FeatureEmptyState, EmptyState, loading states, recovery banners, page shells, modal footers, and admin/health panels; it also records proof-surface consumers, non-goals, and a bounded first Chat/Playground migration queue with suggested focused verification.
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
