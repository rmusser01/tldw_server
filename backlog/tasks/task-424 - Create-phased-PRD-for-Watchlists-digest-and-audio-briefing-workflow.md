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
- [x] PRD is grounded in verified `/watchlists` route, UI, API, scheduler, output, notification, and audio workflow behavior.
- [x] PRD separates existing capabilities from proposed backend/API/frontend improvements.
- [x] PRD includes phased ownership for source setup, monitor setup, digest/newsletter output, optional 1-4 speaker audio, power-user reuse, and operator observability.
- [x] Design review issues are incorporated before implementation planning, including dedupe configuration limits, source settings preservation, scheduled auto-output, email delivery status, and audio artifact persistence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Created a code-grounded phased PRD/spec for /watchlists that reuses existing source settings, monitor filters, Scheduler/audio workflow, output artifacts, delivery metadata, and run observability. The PRD is phased across contract alignment, guided MVP, power-user throughput, and operator/admin reliability.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design review follow-up: tightened the PRD so implementation cannot overpromise current code. Added explicit constraints for unknown source settings preservation, configurable dedupe identity as backend/API work, scheduled digest auto-output dependency, delivery status ownership on output artifacts, and persisting script/per-speaker/final audio artifacts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a design-review hardening pass to the Watchlists PRD before implementation planning. The PRD now explicitly separates verified current behavior from proposed backend/API work for configurable dedupe identity, preserves unknown source settings keys, distinguishes scheduled auto-output from manual test-run output creation, assigns delivery status to output artifacts, and requires persisting script/per-speaker/final audio artifacts for the optional briefing workflow. Verification: git diff --check passed for touched spec/task files; Bandit skipped because this is documentation/task metadata only.
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
