---
id: TASK-368.7
title: Address PR 1727 llama.cpp register_model_path review feedback
status: In Progress
assignee: []
created_date: '2026-05-15 19:15'
updated_date: '2026-05-15 19:17'
labels:
  - llamacpp
  - pr-review
  - security
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1727#discussion_r3250409696'
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the unresolved PR #1727 review finding that register_model_path persists paths before validating them against configured llama.cpp allowlist boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 register_model_path rejects paths outside configured models_dir/allowed_paths before persistence
- [x] #2 Focused inventory tests cover allowed and rejected registration paths
- [ ] #3 PR branch is pushed with the review-fix commit and the unresolved review thread is addressed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the live PR feedback and check current registration/inventory behavior. The unresolved review item is register_model_path persisting paths outside the configured llama.cpp allowlist.
2. Update register_model_path so it reads saved config under the config write lock, canonicalizes the requested path, builds allowed bases from models_dir plus allowed_paths, and rejects outside-allowlist paths before setup_manager.update_config is called.
3. Update focused inventory tests so a valid registration path is explicitly allowed, and add a regression that an outside path returns a sanitized error and writes nothing.
4. Run focused inventory/backend tests plus Bandit on the touched Python source, then push the fix and address the GitHub review thread.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Live PR sweep for #1727 found one unresolved actionable thread: Gemini's comment on register_model_path persisting paths before allowlist validation. CodeRabbit skipped review, Qodo posted summary/status only, and gh pr checks did not show failing CI.

Implemented fail-closed allowlist validation before register_model_path writes registered_model_paths. Focused inventory test passed (12 passed), broader llama.cpp backend slice passed (124 passed, 6 warnings), git diff --check passed, and Bandit on llamacpp_inventory_service.py reported zero findings. Pytest still emits existing post-success Loguru closed-stream cleanup warnings.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
