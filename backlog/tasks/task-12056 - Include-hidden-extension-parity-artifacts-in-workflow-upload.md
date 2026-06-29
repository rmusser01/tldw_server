---
id: TASK-12056
title: Include hidden extension parity artifacts in workflow upload
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
Verify and fix the UI Research Workspace Parity extension artifact upload so the hidden `.workspace-parity-e2e-report.json` file is included when present.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Finding is verified against current workflow code.
- [ ] #2 Upload extension parity artifacts step includes `include-hidden-files: true` alongside existing upload-artifact inputs.
- [ ] #3 Workflow YAML and focused workflow contract validation pass.
- [ ] #4 Backlog task records validation and any skips.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify whether the extension parity upload step targets a hidden report without include-hidden-files.
2. Add the minimal upload-artifact input if still valid.
3. Validate workflow YAML and focused workflow contract tests.
4. Record the outcome in Backlog.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified the finding against current workflow code. The extension parity artifact upload step listed `apps/extension/.workspace-parity-e2e-report.json` but did not set `include-hidden-files: true`, so the hidden dotfile report would be omitted by `actions/upload-artifact` when present. Fixed the still-valid issue with the minimal workflow change: added `include-hidden-files: true` to the `Upload extension parity artifacts` step's `with` block, alongside the existing `name`, `if-no-files-found`, and `path` inputs. Validation: workflow YAML parsed successfully; a focused Python assertion confirmed the upload step has `include-hidden-files: true` and still includes `.workspace-parity-e2e-report.json`; `python -m pytest tldw_Server_API/tests/CI/test_research_workspace_workflow_contracts.py -q` passed with 3 passed and 4 warnings; `git diff --check` passed. Bandit was not applicable because no Python code changed.
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
