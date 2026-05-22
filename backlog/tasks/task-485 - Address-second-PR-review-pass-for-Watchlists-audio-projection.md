---
id: TASK-485
title: Address second PR review pass for Watchlists audio projection
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 21:59'
labels:
  - watchlists
  - audio
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable Qodo and CodeRabbit review findings on Watchlists durable audio projection: type annotations/docstrings, stale retry behavior, raw audio URI exposure, superseded response schema, metadata parsing/merging, Workflow migration error handling, Scheduler metadata validation, stricter audio request ID validation, and correlation-safe TTS artifact metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public run-audio responses expose download URLs and superseded metadata without raw file:// audio_uri paths.
- [x] #2 Retry and request-correlation paths do not mark old audio stale or select legacy artifacts unless a new queued request exists.
- [x] #3 Workflow metadata, migration, and TTS artifact metadata handling fail safely and preserve Watchlists correlation fields.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed second-pass Qodo and CodeRabbit findings. Added type annotations/docstrings on touched Watchlists helpers, preserved object-row output metadata through _parse_output_metadata, prevented retry state mutation when audio retry submission is not queued, removed raw file:// audio_uri exposure from public run-audio responses, surfaced superseded_by through the API response/schema, made Workflow run metadata extraction merge compatible sources, rejected legacy runs/artifacts when an active audio_request_id is present, tightened bare wla_ validation, made TTS artifact metadata preserve Watchlists correlation keys, and made SQLite v9 migration re-raise non-duplicate errors. Verification: targeted regressions first failed on old behavior and now pass; expanded backend suite passed with 341 passed, 1 skipped; Bandit on touched Python scope reported 0 results and 0 errors; git diff --check passed.
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
