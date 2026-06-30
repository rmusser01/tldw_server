---
id: TASK-2226
title: Address PR 2224 ACP Codex review feedback
status: Done
labels:
- ACP
- review
- Codex
references:
- https://github.com/rmusser01/tldw_server/pull/2224
modified_files:
- apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts
- tools/tldw-agent/internal/acp/runner.go
- tools/tldw-agent/internal/acp/runner_test.go
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #2224 after rebasing the Codex ACP backend profile branch onto latest dev. Scope is limited to the reviewed ACP session modal default-agent update guard and Go runner passive blocked status messages, plus focused verification and PR thread resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Branch rebased onto latest origin/dev.
- [x] #2 Session-create modal avoids redundant agentType form writes when the target value is unchanged.
- [x] #3 Go runner passive blocked status has explicit messages for live_certification_required and entrypoint_strategy_missing.
- [x] #4 Focused UI/Go/diff verification recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased PR #2224 branch onto latest origin/dev; rebase completed without conflicts.
- Added an equality guard around the ACP session modal default-agent assignment so the form is not rewritten when the resolved target matches the current agentType.
- Added explicit passive blocker messages for `live_certification_required` and `entrypoint_strategy_missing` in the Go runner, plus a focused table test.
- Bandit skipped for this review-fix pass because no Python files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the PR #2224 review feedback after rebasing onto latest dev. The ACP session create modal now avoids redundant `agentType` writes, the Go runner returns actionable passive-readiness messages for live-certification and missing-entrypoint blockers, and focused UI/Go verification passed.
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
