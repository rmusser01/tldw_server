---
id: TASK-490
title: Plan Sync v2 completion roadmap for Chatbook clients
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- planning
- chatbook
- local-first
priority: high
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- Docs/Design/Sync_V2_M1_Implementation_Decisions.md
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M1.md
- Docs/API/Sync_V2_M2.md
- Docs/API/Sync_V2_M3.md
- backlog/tasks/task-490 - Plan-Sync-v2-completion-roadmap-for-Chatbook-clients.md
- backlog/tasks/task-490.1 - Sync-v2-M1-Lock-implementation-decisions-and-API-docs.md
- backlog/tasks/task-490.2 - Sync-v2-M1-Align-envelope-models-and-storage.md
- backlog/tasks/task-490.3 - Sync-v2-M1-Add-profile-bootstrap-and-status.md
- backlog/tasks/task-490.4 - Sync-v2-M1-Materialize-Notes.md
- backlog/tasks/task-490.5 - Sync-v2-M1-Materialize-Chat.md
- backlog/tasks/task-490.6 - Sync-v2-M1-Sync-attachment-reference-metadata.md
- backlog/tasks/task-490.7 - Sync-v2-M1-Wire-push-pull-conflicts-API.md
- backlog/tasks/task-490.8 - Sync-v2-M1-Route-server-origin-Notes-and-Chat-through-Sync.md
- backlog/tasks/task-490.9 - Sync-v2-M1-Implement-restore-preview.md
- backlog/tasks/task-490.10 - Sync-v2-M1-Add-replay-and-repair.md
- backlog/tasks/task-490.11 - Sync-v2-M1-Verify-end-to-end-and-harden.md
- backlog/tasks/task-490.12 - Sync-v2-M2-Restore-completeness-and-blobs.md
- backlog/tasks/task-490.13 - Sync-v2-M3-Polished-multi-device-sync.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and track the complete Sync v2 roadmap for tldw_server as the sync authority for standalone tldw_chatbook clients when they opt into server connectivity. The roadmap should cover the full path to polished multi-device sync: personal dataset sync first, active server profile support, manual reliable sync as the first implementation milestone, later scheduled/background sync, new-device restore, selective sync, conflict review, user-private encryption and key recovery, workspace-scoped datasets, and broader domain coverage. Chatbook must remain usable as a fully standalone local-only application that never talks to tldw_server. When it does connect to a server, it must support both server-connected modes from the start: dumb front-end access to a server instance with no local sync, and offline-capable client sync to/from the server. Milestone 1 focuses on the authenticated user's personal dataset, with Notes and Chat included from the start. Chat sync must include the minimal conversation/session metadata needed to restore usable chat threads on a new device, not just message rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clarify the full roadmap to polished multi-device sync while keeping manual reliable sync as the first implementation milestone.
- [x] #2 Define Milestone 1 boundaries for personal dataset sync covering Notes and Chat, including minimal chat conversation/session metadata required for new-device restore.
- [x] #3 Identify server API/storage responsibilities, client integration expectations, encryption/key-recovery model, restore/selective-sync behavior, conflict handling, and verification strategy.
- [x] #4 Create and commit an approved PRD/design spec under Docs/superpowers/specs after brainstorming approval, then break implementation into Backlog child tasks during planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Brainstorming decisions captured:

- Chatbook must remain a fully standalone local-only application that can never interact with a tldw_server instance.
- When Chatbook does connect to tldw_server, Sync v2 must support both server-connected modes from the start: dumb server-front-end access with no local sync, and offline-capable client sync to/from the server.
- The full sync roadmap should target polished multi-device sync, but the first implementation milestone is manual reliable sync.
- Milestone 1 starts with the authenticated user's personal dataset before workspace-scoped datasets.
- Milestone 1 includes Notes and Chat from the start.
- Chat sync includes the minimal conversation/session metadata required to restore usable chat threads on a new device, not only message records.
- Milestone 1 uses server-unlocked encryption for user-private personal data, with the roadmap reserving a later path toward stricter client-only encryption for data that should remain opaque to the server.
- Milestone 1 supports sync/restore into an existing non-empty Chatbook profile, but first-time restore must require explicit conflict handling when the same note or chat thread already exists locally.
- Milestone 1 exposes sync as a profile-level Sync now / Restore area first, with per-domain status and details underneath for Notes and Chat.
- The existing `/api/v1/sync` endpoint may be replaced in place for Sync v2 because there are no active v1 sync clients; media sync should be subsumed into the new contract later rather than preserved as a compatibility API.
- Milestone 1 conflict handling is whole-object conflict review for Notes and conversation metadata, plus append-only non-duplicate Chat message merge by stable message ID.
- Milestone 1 stores both an append-only Sync v2 envelope log for restore/audit and materialized accepted Notes/Chat changes in the user's normal server-side state so dumb-front-end mode works immediately.
- Milestone 1 key unlock is tied to normal authenticated server use on trusted/self-hosted deployments. The PRD must reserve a later stricter mode for passphrase/device-key unlock and client-only encryption.
- Milestone 1 includes soft-delete/tombstone envelopes for Notes, conversation metadata, and Chat messages.
- Milestone 1 syncs attachment metadata/references only. Actual binary/blob transfer is deferred to Milestone 2.
- Selected approach: envelope log plus materialized projections. `/api/v1/sync` is replaced in place with Sync v2 endpoints that persist accepted envelopes and apply them to server Notes/Chat state.
- Design Section 1 approved after correction: product scope has three Chatbook modes: standalone local-only, server front-end, and offline sync client.
- Design Section 2 approved: replace `/api/v1/sync` in place with profile-aware envelope APIs, an append-only per-user/dataset log, domain materializers, conflict service, restore service, and an M1 auth-unlocked encryption boundary with later stricter modes.
- Design Section 3 approved: Sync v2 API uses profile, push, pull, restore preview, and conflict resolution endpoints; envelopes are versioned/domain-neutral; M1 domains are notes.note, chat.conversation, chat.message, and attachment.ref metadata only.
- Design Section 4 approved: M1 uses whole-object conflicts for notes/conversation metadata, append-only stable-id chat message merge, first-class tombstones, missing-blob restore warnings, and the same restore-preview path for clean and non-empty profiles.
- Design Section 5 approved: M1 uses trusted/self-hosted server-unlocked encryption, append-only envelopes with apply status and tombstone retention, replay/repair for materialized projections, and metadata to support later key recovery hardening, client-only encryption, retention/GC, and observability.
- Design Section 6 approved: split M1 into server schema/repository, API, Notes materializer, Chat materializer, restore/conflict flows, profile status, Chatbook client integration, and end-to-end verification tasks.
- Wrote superseding PRD/spec at Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md.
- Spec review loop iteration 1 approved with no blocking issues. Advisory: resolve implementation-planning open questions first, especially database placement, profile/device identity, and encryption primitive selection.
- Second-pass review fixes applied before implementation planning: explicit `POST /sync/profile/bootstrap`, base-state conflict metadata (`base_server_cursor`, `base_object_revision`, `base_object_hash`, `object_revision`), restore wording clarified so offline Chatbook applies plans/envelopes locally, and M1 encryption boundary now covers both envelope payload storage and materialized Notes/Chat projections.
- Spec review loop iteration 2 approved with no blocking issues. Advisory: treat DB location, profile/device identity, and at-rest encryption primitive as the first M1 planning gate.

Verification before first commit:

- `git diff --check` passed.
- Placeholder/stale contradiction scan over the spec passed with no matches.
- Bandit not run because this change is documentation/Backlog only and touches no production code.

M1 Task 1 completion:

- Locked the implementation decisions in `Docs/Design/Sync_V2_M1_Implementation_Decisions.md`.
- Locked the M1 API contract in `Docs/API/Sync_V2_M1.md`.
- `git diff --check` passed for the docs task.
- `rg -n "T[B]D|T[O]DO|FIX[M]E|client_private_v1.*M1|workspaces|source_cache|media" Docs/Design/Sync_V2_M1_Implementation_Decisions.md Docs/API/Sync_V2_M1.md` returned no matches; exit 1 is expected for no `rg` matches.
- Bandit not run because Task 1 touched documentation/Backlog records only and no production code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Sync v2 roadmap planning and server-side implementation tracking through the M1, M2, and M3 milestones. The approved PRD/spec and implementation plans define Chatbook local-only, server-front-end, and offline-sync modes; personal Notes/Chat M1 sync; M2 blob/restore completeness; and M3 device lifecycle, background status, workspace datasets, broader domains, stricter encryption, retention/GC, diagnostics, and closeout verification. Child Backlog tasks TASK-490.1 through TASK-490.13.11 record the implementation slices, verification, review follow-ups, and known deferrals. Verification for this closeout: `python -m pytest tldw_Server_API/tests/Sync -q` => 435 passed, 8 warnings; `git diff --check` => clean. Bandit is not applicable for this closeout because it only updates roadmap/spec and Backlog documentation. Chatbook client-side integration remains a separate downstream client effort, but the tracked tldw_server Sync v2 roadmap effort is complete.
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
