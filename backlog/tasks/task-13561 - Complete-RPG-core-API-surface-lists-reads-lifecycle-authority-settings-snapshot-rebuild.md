---
id: TASK-13561
title: 'Complete RPG core API surface: lists, reads, lifecycle, authority settings,
  snapshot rebuild'
status: To Do
labels:
- rpg
- backend
- api
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The RPG module backend ships create/record/review mutations but almost no read or discovery surface: clients cannot list campaigns, sessions, events, or proposals, cannot read a session or snapshot over REST, cannot archive, and cannot ever change session authority settings (frozen at creation with model/mcp auto-commit disabled). This task adds the Phase 1 REST surface per the design spec: list/read endpoints, PATCH metadata + status transitions, PUT authority-settings, and POST snapshot rebuild, all following the module's established metadata-CAS + idempotency patterns. MCP tool parity, dice wiring, TASK-12031 answer-mode controls, and WebUI are subsequent phases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 List/read endpoints for campaigns, sessions, events, proposals, and snapshots exposed under /api/v1/rpg, owner-scoped, using existing rpg.* permission families
- [ ] #2 PATCH campaign/session metadata with status transitions (active/archived) using expected_version CAS, Idempotency-Key header, and exact replay on retry
- [ ] #3 PUT session authority-settings full-replace with strict bools and event-type whitelist validated against the event registry
- [ ] #4 POST snapshot rebuild replays the full event log through the pure reducer in one transaction; concurrent-commit races map to retryable 409
- [ ] #5 Resource Governor route map / route-auth ratchet artifacts and the permission catalog sync test updated for all new routes
- [ ] #6 RPG README route table and non-goals updated
- [ ] #7 Unit and integration tests for the new surface added; full existing RPG suite stays green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design decisions (2026-10-11 brainstorm): archive-status-only lifecycle (no delete, status columns already exist on both tables - zero schema change); REST-only scope (MCP parity deferred to the MCP-themed phase); approach A metadata updates - status/authority changes are version-bumped column updates following the rules-pack-refs replacement pattern, NOT new event types. ADR check: ADR required: no - additive endpoints extending established RPG module patterns (Docs/superpowers/specs/2026-06-25-rpg-*-design.md remain governing); no durable cross-module rule changed. Spec: Docs/superpowers/specs/2026-10-11-rpg-core-api-surface-completion-design.md
Second-pass spec review (2026-10-13): corrected Idempotency-Key header claim (bare header + service 409 idempotency_key_required; 8-128 bound is refs-body-only), pinned deterministic orderings (updated_at DESC, id DESC; proposals created_at DESC, id DESC), specified foreign campaign_id filter -> 404, status-filter omission returns all statuses, events pagination termination rule, mutation responses return updated entity (served by replay), adapter immutability as explicit non-goal, concrete title/description bounds, and confirmed no privilege_catalog.yaml change needed (scope reuse).
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
