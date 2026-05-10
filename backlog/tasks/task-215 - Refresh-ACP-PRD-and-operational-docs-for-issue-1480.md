---
id: TASK-215
title: Refresh ACP PRD and operational docs for issue 1480
status: Done
assignee: []
created_date: '2026-05-10 03:20'
updated_date: '2026-05-10 03:33'
labels:
  - ACP
  - docs
  - PRD
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1480'
  - 'https://github.com/rmusser01/tldw_server/issues/1471'
documentation:
  - Docs/Plans/IMPLEMENTATION_PLAN_acp_docs_refresh_1480.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh the ACP PRD and operational documentation for GitHub issue #1480 in the isolated ACP productionization worktree. Align the draft PRD with the current implemented backend, runner, orchestration, sandbox, governance, schedules/triggers, and frontend surfaces; call out remaining work under epic #1471; and verify the doc set gives new contributors a clear overview-to-setup-to-troubleshooting path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP docs describe current backend runner orchestration sandbox governance schedules triggers and frontend surfaces accurately
- [x] #2 Remaining work is explicitly called out and linked to child issues under issue 1471
- [x] #3 Stale claims from the draft PRD are updated or marked superseded
- [x] #4 Doc set gives a new contributor a clear path from overview to setup to operational troubleshooting
- [x] #5 GitHub issue 1480 is updated with implementation status and verification evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Current-State Inventory
**Goal**: Compare the draft PRD, operational ACP docs, readiness matrix, and current issue map against the implemented ACP productionization slices.
**Success Criteria**: Identify stale PRD claims, authoritative operational docs, stable route contracts, and remaining linked work under #1471.
**Tests**: Readability and link review of referenced docs; no code tests yet.
**Status**: Complete

## Stage 2: PRD Truth Update
**Goal**: Convert Docs/Product/ACP_Agent_Orchestration_PRD.md from a draft-only proposal into a current product/design record.
**Success Criteria**: Shipped, partially shipped, superseded, and remaining items are explicit; route names and component responsibilities match the current implementation.
**Tests**: Targeted grep/read review for stale draft-only route names and pi-agent-only language.
**Status**: Complete

## Stage 3: Operational Doc Path
**Goal**: Make Docs/Development/Agent_Client_Protocol.md the contributor/operator entry point and link it to the readiness matrix for release checklist status.
**Success Criteria**: New contributors can move from overview to setup, route inventory, governance/sandbox/schedules/frontend troubleshooting, and closeout evidence without guessing which document is authoritative.
**Tests**: Targeted doc review for current route inventory and cross-links.
**Status**: Complete

## Stage 4: Verification And Issue Closeout
**Goal**: Verify docs formatting, update Backlog/GitHub, and record remaining production caveats in #1480.
**Success Criteria**: git diff --check is clean, docs-only security skip is recorded, Backlog TASK-215 is Done, and GitHub #1480 has implementation plus verification evidence.
**Tests**: git diff --check and targeted doc grep/read review.
**Status**: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Issue #1480 exact scope reviewed from GitHub. The authoritative operational doc will be Docs/Development/Agent_Client_Protocol.md, with Docs/Development/ACP_Production_Readiness.md serving as the release-readiness checklist. The PRD needs a truth-status preface, current endpoint inventory, implementation-status matrix, and explicit links to remaining productionization child issues.

Docs refreshed for #1480. Rewrote the PRD as a current implementation record with shipped, runtime-caveated, superseded, and remaining scope; added stable route contract families; made Agent_Client_Protocol.md the operational/contributor guide with documentation map and route inventory; updated ACP_Production_Readiness.md documentation row and #1480 closeout checklist entry. Verification recorded: git diff --check clean; targeted rg/read review found no escaped patch artifacts or draft status. Old route names only remain in the PRD superseded-claims section by design. Bandit is not applicable because this slice changed documentation and Backlog files only.

GitHub issue #1480 updated with implementation summary and verification evidence: https://github.com/rmusser01/tldw_server/issues/1480#issuecomment-4414346050. Known remaining production signoff caveats are intentionally left under #1472: live-backend ACP E2E, Go runner verification, artifact retention/redaction policy, and host-specific sandbox backend verification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
#1480 completed in the ACP productionization worktree. Rewrote the ACP PRD as a current implementation record, made Agent_Client_Protocol.md the explicit operational/contributor guide, updated the readiness documentation row and #1480 closeout item, and posted the GitHub issue evidence. Verification: git diff --check clean, targeted stale-artifact/docs-read review completed, and Bandit documented as not applicable for this docs-only slice. GitHub issue updated: https://github.com/rmusser01/tldw_server/issues/1480#issuecomment-4414346050.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Acceptance criteria completed
- [x] #8 Docs refreshed and reviewed against current implementation
- [x] #9 git diff --check and targeted doc verification recorded
- [x] #10 Bandit run for touched code when applicable or document docs-only skip
- [x] #11 Final summary added
<!-- DOD:END -->
