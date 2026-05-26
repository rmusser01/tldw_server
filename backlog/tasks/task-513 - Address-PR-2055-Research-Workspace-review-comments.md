---
id: TASK-513
title: 'Address PR #2055 Research Workspace review comments'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-26 07:42'
labels:
  - research-workspace
  - webui
  - extension
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2055'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the active PR #2055 review comments after rebasing onto latest dev. Scope: workflow artifact action pinning, extension real-backend test hardening, Spanish locale labels, and Research Workspace chat composer typing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active non-outdated PR #2055 review comments are verified against codebase reality and addressed or answered with technical rationale.
- [x] #2 Workflow upload-artifact usage is pinned or otherwise handled according to the verified action availability/security requirement.
- [x] #3 Extension real-backend test helpers avoid hidden hangs, use typed connection-store access, and clean up browser contexts on skip paths.
- [x] #4 UI locale and Research Workspace ChatPane type-safety review comments are fixed.
- [x] #5 Focused verification is run and results are recorded; Bandit applicability is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Rebased PR #2055 branch onto latest origin/dev (d883054a05d05ee8c40ec599a4b77a0c6f822295) and force-pushed rebased head before review fixes.
- Verified actions/upload-artifact tag v7 resolves to upstream commit 043fb46d1a93c77aae656e7c1c64a875d1fc6a0a via git ls-remote; pinned the Research Workspace parity and nightly upload steps to that immutable SHA.
- Added bounded timeout handling to the extension real-backend apiFetch helper using an AbortController; caller-provided abort signals propagate into the timeout controller instead of being dropped.
- Replaced inline any connection-store window access in the extension real-backend spec with explicit local interfaces.
- Closed the launched extension browser context before the host-permission test.skip() path.
- Localized Spanish navigation labels for Research Workspace and Model Playground.
- Tightened Research Workspace ChatPane composer model state from any[] to ChatComposerModel[].
- Verification passed: git diff --check; jq . apps/packages/ui/src/assets/locale/es/option.json; bun run compile from apps/extension; bunx playwright test tests/e2e/research-workspace.real-backend.spec.ts --list from apps/extension; bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage3.test.tsx --maxWorkers=1 --no-file-parallelism from apps/packages/ui.
- Verification caveats: ad hoc bunx tsc over the extension E2E file still reports pre-existing helper typing issues in tests/e2e/utils/extension-build.ts and tests/e2e/utils/extension-id.ts; a broader ChatPane Vitest run still reports pre-existing failures in ChatPane.input-availability.guard.test.ts and ChatPane.stage4.lorebook-activity.test.tsx unrelated to this review slice.
- Bandit not applicable: no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the active PR #2055 review comments after rebasing onto latest dev: pinned Research Workspace workflow artifact uploads to the verified immutable `actions/upload-artifact` SHA, hardened the extension real-backend test helper and skip cleanup, localized the Spanish navigation labels, and tightened ChatPane composer model typing. Focused verification passed; unrelated baseline test/typecheck failures are documented in implementation notes.
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
