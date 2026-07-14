---
id: TASK-12082
title: Revise research discovery ingest boundary design on dev
status: Done
labels:
- design
- research
- media
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
modified_files:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Revise the research discovery chokepoint spec on dev to remove the stale research-owned ingest endpoint/service and make Media the sole public ingestion owner for discovery-selected candidates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec removes POST /api/v1/research/discovery/ingest as a planned public endpoint.
- [x] #2 Spec replaces ResearchIngestActionService with an internal resolver-only handoff boundary.
- [x] #3 Spec states Media owns ingestion, duplicate handling, egress checks, extraction, persistence, quotas, and response outcomes.
- [x] #4 Spec routes Phase 2 through the existing Media ingestion surface and existing PDF/web context extraction pipelines.
- [x] #5 Spec updates flows, tests, rollout, implementation scope, and acceptance criteria so no stale research-owned ingest service remains.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Revised the research discovery chokepoint spec around the corrected boundary: Discovery resolves selected candidates from server-owned snapshots, and Media remains the sole public ingestion owner. Removed the planned research-owned ingest endpoint/service, added Media-owned handoff details, closed Phase 2 to only `pdf` and `html_full_text`, added concrete synchronous bounds, and added Phase 2 acceptance criteria.

Verification: `rg` scan for stale Research-owned ingest service/endpoint references and `git diff --check` on the touched files. Bandit is not applicable because this is a Markdown-only design revision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the research discovery chokepoint spec on dev to remove the duplicate research-owned ingestion API plan and make Phase 2 a Media-owned ingest handoff through the existing Media ingestion surface. Added explicit resolver-only responsibilities, Media-owned duplicate/policy/extraction/persistence outcomes, HTML context-extraction-only handling, conservative synchronous caps, and Phase 2 acceptance criteria.
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
