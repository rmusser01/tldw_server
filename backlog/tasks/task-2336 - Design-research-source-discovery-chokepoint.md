---
id: TASK-2336
title: Design research source discovery chokepoint
status: In Progress
labels:
- research
- design
- discovery
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
modified_files:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a shared research discovery chokepoint seeded from Sourclip-style research sources, with open research graph source routing, OA resolution, and review-gated ingest support for standalone search and Deep Research.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming/spec phase only. The approved design documents a shared research discovery chokepoint, source catalog, normalized discovery contract, review-gated ingest, Deep Research integration, security/ops guardrails, tests, and rollout phases.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written at Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md. Initial spec review loop ran three passes: the first two found blocking ambiguity around discovery result identity and OA candidate ingest handoff; the spec was revised to use persisted discovery snapshots, snapshot-scoped result IDs, deterministic fingerprints, explicit candidate IDs, and `{ result_id, candidate_id }` ingest selections. Third review approved with no blocking issues. Post-user-review clarifications added Phase 1 planning scope, merged provenance entries for deduped results, over-cap source selection validation, and signed/token-bearing OA URL sanitization for API responses, snapshots, logs, candidate identity, and ingest re-resolution. Follow-up review approved with no blocking issues. Final user-requested review findings were addressed by splitting Phase 1 tests from later-phase tests, adding explicit signed/token-bearing URL sanitizer cases, and requiring the Phase 1 implementation plan to choose discovery snapshot storage/schema and default retention. Final spec review approved with no blocking issues. This is a documentation/spec-only change; Bandit is not applicable because no Python/code paths were modified.
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
