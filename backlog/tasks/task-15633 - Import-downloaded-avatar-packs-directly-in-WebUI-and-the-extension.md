---
id: TASK-15633
title: Import downloaded avatar packs directly in WebUI and the extension
status: In Progress
labels:
- buddy
- webui
- extension
references:
- Docs/Design/2026-11-24-independent-avatar-pack-import.md
- backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md
- backlog/decisions/006-buddy-artwork-credit-portability.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Install a downloaded native Buddy visual pack from Avatar settings without creating a Persona or using the SQLite-only Persona import jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WebUI and extension users can select and review a complete native archive, including available credits, before applying.
- [x] #2 Apply installs an independent avatar with the reviewed artwork and can attach it to the selected authorized conversation or workspace.
- [x] #3 The API validates and bounds archive upload and contents on both review and commit, detects changes between requests, and preserves existing attribution and storage behavior on SQLite and PostgreSQL.
- [x] #4 Cancellation and rejected imports leave stored artwork and attachments untouched. A failed attachment after a successful import can be retried using the saved avatar.
- [x] #5 User and API guides describe the new path and retain the Persona-specific workflow.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no new ADR. ADR-005 and ADR-006 already govern owned snapshots and credit provenance. The design records the implementation and verification sequence. Baseline: 72 relevant backend cases and 16 Avatar settings cases passed.
Implemented the shared Avatar settings file review and direct native upload path. Apply binds the reviewed SHA-256, preserves artwork credits, and saves to the existing independent Buddy store. Attachment retries retain the installed profile and refresh the attachment version. Independent review verified multipart bounds before parsing and Chromium-safe native file transport. Verification: 91 backend cases including real SQLite and PostgreSQL imports, 50 shared UI and transport cases, all 40 published pack previews, canonical OpenAPI drift check, clean Ruff and zero Bandit findings on the touched runtime. The full shared TypeScript invocation exceeded the local Node heap; the workflow typecheck remains pending. This task remains In Progress through PR review and protected merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
