---
id: TASK-15648
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
Implemented the shared Avatar settings file review and direct native upload path. Apply binds the reviewed SHA-256, preserves artwork credits, and saves to the existing independent Buddy store. Attachment retries retain the installed profile and refresh the attachment version. Independent review verified multipart bounds before parsing and Chromium-safe native file transport. Verification: 91 backend cases including real SQLite and PostgreSQL imports, 50 shared UI and transport cases, all 40 published pack previews, canonical OpenAPI drift check, clean Ruff and zero Bandit findings on the touched runtime. The full shared UI typecheck passes after rebasing onto the compiler heap and diagnostics fix in PR #3607. This task remains In Progress through PR review and protected merge.
Renumbered this Buddy upload task after TASK-15633 was assigned to legacy media source hashes on dev.

The rebase regenerated the OpenAPI fingerprint and path union from the combined backend. The multipart limit test now patches the archive limit in its defining module, preserving the existing transport size cap. The endpoint ratchet and all 89 focused backend cases passed. The 30 focused upload UI cases and all 1,091 tests in the previously failing coverage shard passed. Ruff, Bandit, canonical OpenAPI, and Backlog checks passed.

Live verification imported Cat in WebUI and Orbit in a production Chrome extension build against an isolated local Server and SQLite data store. Both clients displayed the archive credits, returned HTTP 200 for preview and HTTP 201 for import, stored an independent Buddy, and restored the attached avatar after reload. Preview left the stored profiles untouched. The extension test copy granted localhost access for the disposable Server.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Direct Buddy pack upload is implemented and verified in WebUI and Chrome extension settings. PR #3601 is pending CI and review.
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
