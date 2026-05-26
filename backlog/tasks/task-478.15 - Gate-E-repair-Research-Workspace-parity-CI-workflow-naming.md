---
id: TASK-478.15
title: 'Gate E: repair Research Workspace parity CI workflow naming'
status: Done
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
references:
- https://github.com/rmusser01/tldw_server/pull/2055
- https://github.com/rmusser01/tldw_server/actions/runs/26429491471/job/77799577506
modified_files:
- .github/workflows/ui-research-workspace-parity.yml
- .github/workflows/ui-research-workspace-nightly.yml
- .github/workflows/ui-workspace-playground-parity.yml
- .github/workflows/ui-workspace-playground-nightly.yml
- tldw_Server_API/tests/CI/test_research_workspace_workflow_contracts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the active GitHub Actions parity workflow after the Research Workspace route/script rename. Current PR #2055 CI failure: WebUI Workspace Playground Parity invokes removed script `bun run e2e:workspace-playground:parity`, causing `Script not found`. Scope is workflow/script metadata only: use current Research Workspace script names, remove stale workspace-playground active labels/artifact names/path filters where they refer to current parity gates, and verify the runnable script locally where feasible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Active Research Workspace parity/nightly workflows use current Research Workspace names, paths, and package scripts.
- [x] The WebUI parity workflow calls `bun run e2e:research-workspace:parity`, which exists and passes locally.
- [x] No active `.github/workflows` file contains stale `workspace-playground`, `Workspace Playground`, `WorkspacePlayground`, or `e2e:workspace-playground` references.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: PR #2055 CI was still invoking the removed WebUI script `bun run e2e:workspace-playground:parity` from the active workspace parity workflow. Replaced the active parity/nightly workflow files with Research Workspace-named workflow files, updated path filters/job ids/check names/artifact names/concurrency/server labels, and pointed WebUI parity/nightly jobs at existing `e2e:research-workspace:*` package scripts.

Added `tldw_Server_API/tests/CI/test_research_workspace_workflow_contracts.py` to prevent stale `workspace-playground` workflow names, paths, and commands from returning in these active workflows.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the failing WebUI parity CI gate caused by stale workspace-playground workflow metadata and command names. Verified the new Research Workspace workflow contract test, existing required workflow contracts, and the actual WebUI parity script locally. Bandit was run on the new CI test with pytest assert rule B101 skipped; no remaining findings.
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
