---
id: TASK-264
title: Implement OpenWebUI non-image Media registration slice
status: Done
assignee: []
created_date: '2026-05-11 16:14'
updated_date: '2026-05-11 16:28'
labels:
  - chatbooks
  - openwebui
  - implementation
dependencies:
  - TASK-263
references:
  - >-
    Docs/superpowers/plans/2026-05-11-openwebui-attachment-hydration-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 of the OpenWebUI attachment hydration implementation plan: register referenced non-image attachments as durable tldw-owned Media DB records without automatic processing by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PDF/non-image hydration creates a Media row and MediaFiles original row under tldw-owned storage
- [x] #2 Registration records owner_user_id, personal visibility, byte checksum/source hash, and safe OpenWebUI source metadata
- [x] #3 Dedupe is owner-aware and does not collapse unrelated source-id-less files
- [x] #4 process_supported_files remains optional and processing failures do not remove registered MediaFiles rows
- [x] #5 Focused pytest, diff checks, and Bandit for touched backend code are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 4 of the implementation plan with TDD: inspect existing Media DB add/file helper contracts, add failing non-image registration tests, implement durable copy plus owner-aware registration in openwebui_hydration.py using existing Media DB APIs where possible, keep processing hook optional, verify, update tracking, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: service/media registration focus passed with 15 tests; combined OpenWebUI hydration/import + MediaFiles + SQLite MediaDB suite passed with 108 tests; git diff --check passed; Bandit on touched backend code wrote /tmp/bandit_openwebui_media_registration.json with 0 results and 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 4 non-image OpenWebUI attachment registration. Non-image refs now create owner-scoped Media rows, copy source bytes into tldw-owned per-user storage, insert MediaFiles original rows, record safe hydration metadata, preserve owner-aware dedupe without cross-user reuse, and keep optional processing behind process_supported_files with failed processing recorded without deleting the registered file.
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
