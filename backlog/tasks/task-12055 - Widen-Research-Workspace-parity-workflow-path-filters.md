---
id: TASK-12055
title: Widen Research Workspace parity workflow path filters
status: Done
labels:
- ci
- workflow
- frontend
priority: Medium
modified_files:
- .github/workflows/ui-research-workspace-parity.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and fix the UI Research Workspace Parity workflow paths filters so implementation changes in the WebUI, extension, and shared UI package trigger the parity jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The finding is verified against current code and only still-valid changes are made.
- [ ] #2 Both path filters in `.github/workflows/ui-research-workspace-parity.yml` include broader WebUI, extension, and UI package source globs.
- [ ] #3 Workflow YAML parses successfully and diff checks pass.
- [ ] #4 Backlog task records verification and final outcome.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify current workflow path filters and related path blocks.
2. Add minimal broader source-tree globs to both pull_request and push path filters.
3. Validate workflow YAML and diff hygiene.
4. Record results in Backlog.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified the review finding against current code. The `ui-research-workspace-parity` workflow still had narrow `pull_request.paths` and `push.paths` filters that only watched parity specs, targeted shared UI ResearchWorkspace files, package/config files, and the workflow itself. Fixed the still-valid issue by adding broader implementation-root globs to both path filters: current WebUI roots (`components`, `extension`, `hooks`, `lib`, `pages`, `styles`, `types`), current extension roots (`entrypoints`, `public`, `scripts`, `types`), and broader shared UI package coverage (`apps/packages/ui/src/**/*`, `index.ts`, package/config files). Skipped adding only literal `apps/tldw-frontend/src/**/*` and `apps/extension/src/**/*` because those directories do not exist in the current tree and would not trigger current implementation changes. Validation: YAML parsed successfully and both paths arrays contain the expected broader globs; `git diff --check` passed; targeted workflow contract test passed with `3 passed`; `actionlint` is not installed locally. Bandit was not applicable because no Python code was changed.
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
