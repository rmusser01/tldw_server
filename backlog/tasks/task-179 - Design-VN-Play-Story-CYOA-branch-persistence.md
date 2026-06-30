---
id: TASK-179
title: Design VN Play Story/CYOA branch persistence
status: Done
assignee: []
created_date: '2026-05-09 18:58'
updated_date: '2026-05-09 19:16'
labels:
  - vn-play
  - design
  - story-mode
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1434'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/superpowers/specs/2026-05-01-vn-play-runtime-design.md
  - Docs/API-related/VN_PLAY_API.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for issue #1434: persist Story/CYOA choice selection as branch metadata in VN Play before implementation planning. Scope is design-only: branch creation timing, choice validation, idempotency, retry, replay/checkpoint behavior, docs, and test plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec captures the approved Story/CYOA branch persistence behavior.
- [x] #2 Spec defines error, idempotency, retry, and checkpoint/replay semantics before implementation.
- [x] #3 Spec includes API documentation and focused test-plan updates.
- [x] #4 Spec is reviewed locally and committed on the isolated worktree branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created design draft in Docs/superpowers/specs/2026-05-09-vn-play-story-branch-persistence-design.md inside the isolated worktree. Local review tightened the design to preserve existing custom-action behavior, require atomic branch+choice-selected persistence, and require retry-last-turn to reuse existing choice branches.

Verification before commit: git diff --check in the isolated worktree exited 0. This is a docs/task-only spec change, so Bandit is not applicable until implementation touches Python code.

Committed the design spec on codex/vn-play-story-branch-persistence. No Python code was touched; Bandit is recorded as not applicable for this doc-only design commit.

Reopened after design review. Follow-up fixes needed: keep branch_path API-compatible, persist accepted choice scene state before model work, pin retry source-of-truth, scope parent choice lookup, make Story custom-action behavior explicit, and require a repository helper for atomic accepted-choice persistence.

Addressed design review findings in the spec: branch_path_json remains list-shaped for VNPlayBranchResponse compatibility; accepted choices persist replay-derived scene state before model work; retry-last-turn is failure-retry only and uses the failed turn request input_event_id; parent_event_id lookup is bounded to the active replay window; Story custom_action is explicitly non-branching; implementation must use a repository helper for atomic branch+choice_selected+turn-request+scene-state persistence. Verification: git diff --check exited 0 in the isolated worktree.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the VN Play Story/CYOA branch persistence spec after design review. The spec now resolves the branch_path API mismatch, pre-model scene-state persistence gap, retry source-of-truth ambiguity, parent-choice lookup ambiguity, Story custom-action behavior, and accepted-choice atomicity implementation hook. This remains design-only; Bandit is not applicable until Python implementation work starts.
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
