---
id: TASK-12753
title: Reconcile stale Claude ACP docs after live certification merge
status: Done
labels:
- ACP
- docs
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2374
- https://github.com/rmusser01/tldw_server/pull/2376
- https://github.com/rmusser01/tldw_server/issues/1564
- https://github.com/rmusser01/tldw_server/issues/1532
modified_files:
- Docs/User_Guides/Integrations_Experiments/Anthropic_ClaudeCode_ClaudeSDK_Setup.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update duplicate/non-canonical Claude ACP setup wording that still reports claude_code as documented_unverified after PR #2374 merged, then reconcile or leave open the relevant ACP tracker issues based on whether this follow-up has landed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No current docs/setup surface claims claude_code is documented_unverified after live E2E certification.
- [x] #2 Issue #1564 remains open with this PR linked until the docs reconciliation lands.
- [x] #3 Issue #1532 remains open until #1564 is reconciled/closed after this follow-up lands.
- [x] #4 Verification includes targeted grep and git diff checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the duplicate Claude Code setup guide under Docs/User_Guides to match the live-E2E certification state merged in PR #2374. Verified no current user-facing setup/matrix surfaces still claim claude_code is documented_unverified/documented_only, ran git diff --check, opened PR #2376, and linked #1564/#1532 with comments. Bandit skipped because the PR is documentation-only.
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
