---
id: TASK-12001
title: Harden Sharing module review findings
status: Done
assignee: []
created_date: 2026-06-24 00:00
updated_date: 2026-06-25 02:20
labels:
- sharing
- security
- review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated review findings in the Sharing module and its API wiring: resource ownership for token creation, atomic token use claiming, password-protected import flow, workspace deletion cleanup, clone chunk preservation and copy counts, and audit failure behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace and chatbook share tokens require owner validation before creation.
- [x] #2 Public token imports consume `max_uses` atomically and cannot exceed configured limits.
- [x] #3 Password-protected non-prototype imports have a usable verified path.
- [x] #4 Workspace deletion revokes active workspace shares and tokens through the Sharing cleanup hook.
- [x] #5 Workspace clone preserves media chunks or records reprocessing status, and reports successful copy counts.
- [x] #6 Sharing mutation endpoints handle audit-write failures without returning misleading partial failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing tests for token ownership, import claiming/password verification, deletion cleanup, clone chunks/counts, and audit failure behavior.
2. Patch Sharing endpoint/service code with minimal behavior changes that match existing patterns.
3. Run focused Sharing tests, compile checks for touched Python files, Bandit on touched scope, and diff hygiene.
4. Update this task with verification results and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task created manually because Backlog MCP was unavailable and the installed Backlog CLI hung on search/list/create operations in this workspace. User approved the temporary manual fallback.

RED verification before implementation:
- `.venv/bin/python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::TestWorkspaceSharing::test_share_workspace_succeeds_when_audit_write_fails tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::TestShareTokens::test_create_workspace_token_requires_owned_workspace tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::TestShareTokens::test_create_chatbook_token_requires_owned_chatbook tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::TestPublicEndpoints::test_public_import_accepts_password_for_protected_token tldw_Server_API/tests/Sharing/test_sharing_endpoints.py::TestPublicEndpoints::test_public_import_uses_atomic_claim_for_use_limit tldw_Server_API/tests/Sharing/test_clone_service.py::test_copy_media_deep_copies_unvectorized_chunks tldw_Server_API/tests/Sharing/test_clone_service.py::test_clone_skipped_source_log_is_sanitized_when_media_copy_fails tldw_Server_API/tests/Sharing/test_clone_service.py::test_clone_source_failure_log_is_sanitized tldw_Server_API/tests/Sharing/test_clone_service.py::test_clone_note_failure_log_is_sanitized tldw_Server_API/tests/Sharing/test_clone_service.py::test_clone_artifact_failure_log_is_sanitized tldw_Server_API/tests/Sharing/test_clone_service.py::test_clone_skips_source_when_media_copy_fails tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_delete_workspace_invokes_sharing_cleanup_hook -q` failed with 12 expected failures.

Implementation:
- Added best-effort Sharing audit logging so successful mutations are not reported as failed solely due to audit persistence errors.
- Added ownership checks before workspace/chatbook token creation and preserved prototype ownership checks.
- Updated public import to verify password-protected non-prototype links inline and consume `max_uses` through atomic `claim_token_use`.
- Wired workspace deletion to `on_workspace_deleted` after the local delete succeeds.
- Updated clone logic to pass source unvectorized chunks into target media creation and report attempted/copied/failed counts separately.

Verification:
- `.venv/bin/python -m compileall -q tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/Sharing` passed.
- Focused regression pytest command passed: 12 passed.
- `.venv/bin/python -m pytest tldw_Server_API/tests/Sharing tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_delete_workspace_invokes_sharing_cleanup_hook -q` passed: 152 passed, 360 warnings.
- `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sharing tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/endpoints/workspaces.py -f json -o /tmp/bandit_sharing_review_fixes.json` passed. Report totals: 0 results, 0 high/medium/low severity findings.

PR #2495 review follow-up:
- Rebasing on latest `origin/dev` completed cleanly.
- Moved chatbook file ownership checks and sync ChatbookService job lookup into a threadpool-backed helper before returning from the async request path.
- Added `PublicImportRequest` with optional `password` so unprotected public imports accept `{}` request bodies while protected imports still require a password.
- Fixed `workspace_deletion_hook.on_workspace_deleted` to await `get_db_pool()` before constructing `SharedWorkspaceRepo`.
- Added contextual workspace cleanup failure logging with `workspace_id` and `owner_user_id`.
- Added regressions for empty JSON public import, awaited cleanup hook revocation flow, and contextual cleanup logging.

Review follow-up verification:
- `.venv/bin/python -m compileall -q tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/sharing_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/core/Sharing` passed.
- Targeted review-fix pytest command passed: 6 passed.
- `.venv/bin/python -m pytest tldw_Server_API/tests/Sharing tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_delete_workspace_invokes_sharing_cleanup_hook tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_delete_workspace_cleanup_failure_log_includes_context -q` passed: 155 passed, 363 warnings.
- `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sharing tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/sharing_schemas.py -f json -o /tmp/bandit_sharing_review_fixes_comments.json` passed. Report totals: 0 results, 0 high/medium/low severity findings.

Modified files:
- `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
- `tldw_Server_API/app/core/Sharing/clone_service.py`
- `tldw_Server_API/app/core/Sharing/workspace_deletion_hook.py`
- `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`
- `tldw_Server_API/tests/Sharing/test_clone_service.py`
- `tldw_Server_API/tests/Sharing/test_workspace_deletion_hook.py`
- `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- `IMPLEMENTATION_PLAN_sharing_review_fixes_12001.md`
- `backlog/tasks/task-12001 - Harden-Sharing-module-review-findings.md`
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Sharing token creation/import paths, workspace deletion cleanup, clone fidelity/count reporting, and audit failure behavior. Regression tests, compile check, and Bandit touched-scope scan all passed; no skipped blockers remain.
Second PR review pass rebased onto latest dev and addressed all new CodeRabbit comments: plan markdown formatting, chatbook storage user id ownership checks, production cleanup-hook log context, and sparse chunk clone preservation. Focused and broader regression tests, compile check, diff check, and Bandit all passed; markdownlint-cli2 was unavailable locally.
Final push state is rebased on the latest fetched `origin/dev` and verified after that rebase: compile passed, 172 targeted/broad regression tests passed, and Bandit reported 0 findings.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Second PR review follow-up:
- Latest `origin/dev` advanced by 71 commits; rebased `codex/sharing-review-fixes-12001` cleanly on top of it.
- New CodeRabbit actionable comments to address: implementation plan markdownlint formatting, chatbook ownership storage user id, workspace deletion hook contextual production failure logging, and sparse unvectorized chunk copying.
Second PR review follow-up implementation:
- Added H1 and blank-line spacing to `IMPLEMENTATION_PLAN_sharing_review_fixes_12001.md` for markdownlint MD041/MD022.
- Updated chatbook ownership verification to resolve the storage user id once and use `user.id_int` when available for the ChaCha DB lookup and threadpool-backed ownership scan.
- Moved contextual cleanup failure logging into `workspace_deletion_hook.on_workspace_deleted`, so the production swallowed-error path includes `workspace_id` and `owner_user_id` without exception details.
- Added `get_unvectorized_max_chunk_index` to the Media DB read API/runtime surface and used it in `CloneService` so sparse active chunk indexes are included during clone.
- Added regressions covering numeric-string chatbook owner ids, contextual hook failure logging, sparse chunk clone range selection, and the new Media DB max-index helper.

Second follow-up verification:
- RED run before implementation failed the three new behavior regressions as expected.
- `.venv/bin/python -m pytest ...` focused DB/Sharing set passed: 18 passed, 58 warnings.
- `.venv/bin/python -m pytest tldw_Server_API/tests/Sharing tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_delete_workspace_invokes_sharing_cleanup_hook tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_delete_workspace_cleanup_failure_log_includes_context -q` passed: 157 passed, 367 warnings.
- `.venv/bin/python -m compileall -q ...` for touched Python modules passed.
- `.venv/bin/python -m bandit -r ... -f json -o /tmp/bandit_sharing_review_fixes_second_followup.json` passed with 0 findings.
- `git diff --check` passed.
- `markdownlint-cli2 IMPLEMENTATION_PLAN_sharing_review_fixes_12001.md` could not be run locally because `markdownlint-cli2` is not installed in the worktree environment.
Final post-rebase verification before push:
- Fetched latest `origin/dev`, rebased again after dev advanced by 5 commits, and confirmed branch state `0 1` against `origin/dev`.
- `.venv/bin/python -m compileall -q ...` for touched Python modules passed after final rebase.
- `.venv/bin/python -m pytest tldw_Server_API/tests/Sharing ... tldw_Server_API/tests/MediaDB2/test_unvectorized_chunk_count.py -q` passed after final rebase: 172 passed, 408 warnings.
- `.venv/bin/python -m bandit -r ... -f json -o /tmp/bandit_sharing_review_fixes_second_followup_final.json` passed after final rebase with 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
