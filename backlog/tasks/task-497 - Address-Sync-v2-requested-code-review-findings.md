---
id: TASK-497
title: Address Sync v2 requested code review findings
status: Done
labels:
- sync-v2
- code-review
priority: high
references:
- 'PR #2030'
- Review agent 019e57d2-7eb7-7ea0-811a-8940f3abfaca
- 'PR #2043 https://github.com/rmusser01/tldw_server/pull/2043'
modified_files:
- Docs/API/Sync_V2_M2.md
- Docs/Design/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-sync-v2-pr2030-review-fixes-plan.md
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/app/core/Sync/v2/blob_store.py
- tldw_Server_API/app/core/Sync/v2/factory.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py
- tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
- tldw_Server_API/tests/Sync/test_sync_v2_factory.py
- tldw_Server_API/tests/Sync/test_sync_v2_models.py
- tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py
- tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py
- tldw_Server_API/tests/Sync/test_sync_v2_retention.py
- tldw_Server_API/tests/Sync/test_sync_v2_workspace_blobs.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track fixes for the requested code review on PR #2030 after rebase onto latest dev. Scope covers restore preview completeness, API contract alignment, blob upload size/enablement behavior, workspace blob access/retention semantics, and blob completion retry safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify each reviewer finding against the current code and classify true fixes vs pushback.
2. Apply focused fixes with tests for restore completeness/API aliases/blob config and upload safety/workspace semantics.
3. Run targeted Sync tests and Bandit on touched production scope.
4. Commit and push review fixes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed requested PR #2030 code review findings and added regression coverage. Follow-up draft PR: https://github.com/rmusser01/tldw_server/pull/2043. Verification passed after rebase: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync` => 424 passed, 6 warnings; `git diff --check` => clean; Bandit on touched production Sync/API paths => 0 findings, results at `/tmp/bandit_sync_v2_review_fixes_pr2030.json`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
