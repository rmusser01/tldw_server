---
id: TASK-490.1
title: 'Sync v2 M1: Lock implementation decisions and API docs'
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- docs
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- Docs/Design/Sync_V2_M1_Implementation_Decisions.md
- Docs/API/Sync_V2_M1.md
modified_files:
- Docs/Design/Sync_V2_M1_Implementation_Decisions.md
- Docs/API/Sync_V2_M1.md
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- backlog/tasks/task-490 - Plan-Sync-v2-completion-roadmap-for-Chatbook-clients.md
- backlog/tasks/task-490.1 - Sync-v2-M1-Lock-implementation-decisions-and-API-docs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the locked M1 implementation decisions and API contract before production code edits begin. This covers the per-user Sync DB location, ChaChaNotes projection boundary, explicit profile bootstrap contract, server_trusted_v1 at-rest encryption posture, and M1 public domains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs/Design/Sync_V2_M1_Implementation_Decisions.md records the planning gate decisions.
- [x] #2 Docs/API/Sync_V2_M1.md documents M1 profile, bootstrap, push, pull, restore preview, conflict resolution, envelope examples, tombstones, and attachment refs.
- [x] #3 Docs checks pass with no unresolved placeholders or M1/future-domain contradictions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-1-lock-m1-decisions-and-contract-docs
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/Design/Sync_V2_M1_Implementation_Decisions.md` to lock the five M1 planning gates: per-user `Sync_v2.db`, per-user `ChaChaNotes.db` projections, explicit bootstrap, `server_trusted_v1` deployment-level at-rest encryption attestation, and M1-only domains.
- Created `Docs/API/Sync_V2_M1.md` with profile, bootstrap, push, pull, restore preview, and conflict-resolution request/response shapes plus envelope examples for `notes.note`, `chat.conversation`, `chat.message`, `attachment.ref`, and tombstones.
- Verification:
  - `git diff --check` passed.
  - `rg -n "T[B]D|T[O]DO|FIX[M]E|client_private_v1.*M1|workspaces|source_cache|media" Docs/Design/Sync_V2_M1_Implementation_Decisions.md Docs/API/Sync_V2_M1.md` returned no matches; exit 1 is expected for no `rg` matches.
  - Bandit was not run because this task touched only documentation and Backlog records, with no production code.
- Spec compliance review follow-up:
  - Updated conflict actions to use `duplicate_rename` and documented `skip` as dismissing the conflict without applying either side.
  - Expanded restore preview to include optional attachment availability inventory, available datasets/domains, latest per-domain cursors, safe applies, tombstones, missing blobs, attachment-ref summaries with parent references, envelope ranges, and encryption/key status.
  - Updated the `attachment.ref` example and prose to use the required metadata fields: `attachment_id`, `parent_domain`, `parent_object_id`, `content_type`, `size_bytes`, `payload_hash`, and `availability`.
- Code-quality review follow-up:
  - Clarified that restore preview summary counts include all selected domains, including `attachment.ref` metadata, and corrected the sample total to match the per-domain counts.
  - Defined pull `next_cursor` as the highest scanned server cursor for the requested window, including echo-suppressed and domain-filter-skipped rows, and clarified that full-profile clients advance the global/profile cursor only after unfiltered pulls.
  - Added a `server_trusted_v1` not-ready capability example and documented profile/bootstrap fail-closed behavior.
- Re-review follow-up:
  - Replaced stale conflict action terms with the locked `overwrite`, `duplicate_rename`, and `skip` actions in the API contract, M1 implementation plan, and PRD spec.
  - Marked restore preview detail arrays as abbreviated examples and documented that production M1 responses include all selected detail rows unless a future pagination contract is explicitly added.
  - Added the exact docs-check `rg` command to the parent Backlog task notes.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Locked the Sync v2 M1 implementation-decision doc and API contract doc before production implementation work. The docs establish the per-user Sync v2 envelope store, ChaChaNotes materialized projection boundary, explicit bootstrap endpoint, `server_trusted_v1` at-rest encryption attestation, M1 public domain set, endpoint request/response contracts, restore preview contract, conflict-resolution actions, envelope examples, tombstone behavior, and M1 attachment-reference limitations. Required docs checks passed; Bandit was skipped because no code was touched.
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
