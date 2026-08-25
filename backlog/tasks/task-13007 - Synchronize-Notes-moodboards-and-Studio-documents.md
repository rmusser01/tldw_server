---
id: TASK-13007
title: Synchronize Notes moodboards and Studio documents
status: In Progress
assignee: []
created_date: '2026-08-08 20:25'
updated_date: '2026-08-25 15:39'
labels:
  - notes
  - sync-v2
  - parity
  - moodboards
  - studio
dependencies:
  - TASK-13006
references:
  - Docs/ADR/031-notes-capability-sync-domains.md
  - Docs/ADR/034-durable-server-origin-sync-mutation-batches.md
  - Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md
  - Docs/ADR/040-synchronized-moodboards-and-studio-authority.md
documentation:
  - Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Synchronize Notes moodboards, note placement, and persisted Studio document state so visual organization and accepted AI-assisted outputs survive offline and multi-device use without synchronizing transient generation requests.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Notes moodboard placement Studio and accepted-output suites pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 Transient-request exclusion provenance conflict and pagination scenarios have automated evidence.
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities advertise versioned notes.moodboard notes.moodboard_note and notes.studio_document domains with supported upsert and tombstone operations only after readiness
- [ ] #2 Moodboards and unique manual placements preserve portable identity canvas/layout state optimistic lineage and tenant scope while smart matches remain bounded derived local projections
- [ ] #3 Studio sidecars preserve accepted structured render state and provenance while notes.note remains the sole title and Markdown authority and legacy REST representation remains compatible
- [ ] #4 Server-origin and client-origin accepted mutations append complete canonical plans before product materialization when Sync v2 capture is active
- [ ] #5 AI title suggestion summarization and generation requests remain operations while only explicitly accepted persisted output enters synchronized state
- [ ] #6 Concurrent layout Studio note-lifecycle restore placement and capability-mismatch scenarios yield idempotent outcomes reviewable conflicts or stable fail-closed responses with bounded authorized queries
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Publish the reviewed proposed authority and lifecycle design in Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md and ADR-040 before implementation.
2. Run independent specification reviews and resolve every valid finding.
3. Obtain requester approval of the corrected independently reviewed written spec.
4. Use the writing-plans workflow to add concrete per-child implementation plans before code changes.
5. Execute TASK-13007.1 through TASK-13007.5 in dependency order with TDD review verification and live PostgreSQL gates.

ADR required: yes
ADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md
Reason: The work changes persistent schema tenant scope sync authority lifecycle conflict policy provenance and capability activation boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed proposed design decomposed into TASK-13007.1 through TASK-13007.5. Existing Chatbook reference paths are stale and will be replaced by the server-local authoritative spec and ADR-040.

Independent reviews corrected portable note time client compound-push semantics atomic task boundaries bounded smart matching closed Studio schemas encryption/source authorization REST compatibility placement ordering scope authority lifecycle barriers canonical serialization rollback conflict resolution mixed-version authority rollout and per-PR API drift ownership. ADR-039 now records the proposed per-graph scope-authority amendment. Final independent re-review found no remaining blocker high or medium production-readiness issue. Requester approval of the corrected written spec is the next gate. No implementation has started.
<!-- SECTION:NOTES:END -->
