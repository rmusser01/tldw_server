---
id: TASK-424
title: Create phased PRD for Watchlists digest and audio briefing workflow
status: Done
labels:
- watchlists
- prd
- ux
- design
references:
- apps/tldw-frontend/pages/watchlists.tsx
- apps/packages/ui/src/components/Option/Watchlists
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/Watchlists
modified_files:
- Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md
- backlog/tasks/task-424 - Create-phased-PRD-for-Watchlists-digest-and-audio-briefing-workflow.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a code-grounded phased PRD/spec for improving /watchlists to support source scraping setup, monitor cadence/rules/filters/dedupe ownership, digest/newsletter output, optional 1-4 speaker audio briefing generation, and operator/power-user observability without removing existing OSINT/news workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Created a code-grounded phased PRD/spec for /watchlists that reuses existing source settings, monitor filters, Scheduler/audio workflow, output artifacts, delivery metadata, and run observability. The PRD is phased across contract alignment, guided MVP, power-user throughput, and operator/admin reliability.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md. The PRD addresses source scraping setup, per-source extraction/dedupe ownership, per-monitor cadence/filter/output/delivery ownership, variable cadence, digest/newsletter output, optional 1-4 speaker audio briefing generation inside /watchlists, and preservation of existing OSINT/news workflows. Verification: git diff --check passed for touched spec/task files; Bandit skipped because this is documentation/task metadata only.
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
