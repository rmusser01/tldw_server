---
id: TASK-514
title: 'Address PR #2055 follow-up review comments'
status: Done
labels:
- research-workspace
- webui
- extension
- backend
- review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2055
modified_files:
- .github/workflows/ui-research-workspace-nightly.yml
- apps/extension/tests/e2e/research-workspace.real-backend.spec.ts
- apps/packages/ui/src/assets/locale/ko/option.json
- apps/packages/ui/src/assets/locale/ml/option.json
- apps/packages/ui/src/assets/locale/zh-TW/option.json
- apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
- tldw_Server_API/app/api/v1/endpoints/research_workspace.py
- tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/app/core/DB_Management/media_db/api.py
- tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py
- tldw_Server_API/app/core/DB_Management/media_db/repositories/media_lookup_repository.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/query_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/validation.py
- tldw_Server_API/app/core/Workspaces/status_projection.py
- tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the follow-up CodeRabbit and Qodo comments on PR #2055 after rebasing onto the latest dev. Scope includes workflow CI auth envs, extension real-backend response guards, locale labels, ChatSidebar test fallback, workspace endpoint typing/docstrings/logging, and workspace status/job projection correctness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active non-outdated CodeRabbit and Qodo PR #2055 review comments are verified against the rebased code and fixed or answered with technical rationale.
- [x] #2 Research Workspace nightly workflows avoid hardcoded API key literals while preserving CI self-containment.
- [x] #3 Extension real-backend test response parsing uses unknown plus shape guards before nested field access.
- [x] #4 Locale/test quick wins are fixed without broad unrelated i18n churn.
- [x] #5 Workspace backend endpoint modules/functions have the requested docstrings/type hints and no silent broad exception swallowing.
- [x] #6 Workspace status projection avoids loading full media content blobs for readiness checks, and workspace active jobs do not surface unrelated legacy media jobs.
- [x] #7 Focused backend/frontend verification is run and recorded; Bandit applicability is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Addressed PR #2055 follow-up review comments after rebasing the worktree onto latest dev. Fixed dynamic CI auth envs, extension API response guards, i18n labels, ChatSidebar fallback, backend docstrings/return annotations, job-list logging/fail-open behavior, active job workspace scoping, and lightweight media status reads for workspace source readiness. Verification: workspace source status/context pytest suites passed; media DB API targeted tests passed; ChatSidebar Vitest passed; extension compile and Playwright list passed; locale JSON parsed; workflow YAML parsed; git diff --check passed; Bandit on touched Python scope reported 0 results. actionlint was unavailable locally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2055 follow-up review feedback is addressed in the rebased branch. The remaining PR validation will run after pushing the new commit to the PR branch.
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
