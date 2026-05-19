# ACP Docs Refresh Implementation Plan

## Stage 1: Current-State Inventory
**Goal**: Compare the draft PRD, operational ACP docs, readiness matrix, and current issue map against the implemented ACP productionization slices.
**Success Criteria**: Identify stale PRD claims, authoritative operational docs, stable route contracts, and remaining linked work under #1471.
**Tests**: Readability and link review of referenced docs; no code tests yet.
**Status**: Complete

## Stage 2: PRD Truth Update
**Goal**: Convert `Docs/Product/ACP_Agent_Orchestration_PRD.md` from a draft-only proposal into a current product/design record.
**Success Criteria**: Shipped, partially shipped, superseded, and remaining items are explicit; route names and component responsibilities match the current implementation.
**Tests**: Targeted grep/read review for stale draft-only route names and pi-agent-only language.
**Status**: Complete

## Stage 3: Operational Doc Path
**Goal**: Make `Docs/Development/Agent_Client_Protocol.md` the contributor/operator entry point and link it to the readiness matrix for release checklist status.
**Success Criteria**: New contributors can move from overview to setup, route inventory, governance/sandbox/schedules/frontend troubleshooting, and closeout evidence without guessing which document is authoritative.
**Tests**: Targeted doc review for current route inventory and cross-links.
**Status**: Complete

## Stage 4: Verification And Issue Closeout
**Goal**: Verify docs formatting, update Backlog/GitHub, and record remaining production caveats in #1480.
**Success Criteria**: `git diff --check` is clean, docs-only security skip is recorded, Backlog TASK-215 is Done, and GitHub #1480 has implementation plus verification evidence.
**Tests**: `git diff --check` and targeted doc grep/read review.
**Status**: Complete
