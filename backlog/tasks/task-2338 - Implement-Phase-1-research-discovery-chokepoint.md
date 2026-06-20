---
id: TASK-2338
title: Implement Phase 1 research discovery chokepoint
status: In Progress
labels:
- research
- implementation
- discovery
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Phase 1 research discovery chokepoint plan: source catalog, source router, standalone discovery API, normalized metadata/OA candidates, persisted discovery snapshots, and focused tests. Scope excludes standalone ingest, Deep Research migration, compatibility delegation, and fallback site-search rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Phase 1 endpoints are available at GET /api/v1/research/sources and POST /api/v1/research/discovery/search.
- [ ] #2 Discovery snapshots are persisted owner-scoped, sanitized, and short-lived in ResearchSessionsDB.
- [ ] #3 Over-cap category/source selections return validation errors instead of silent truncation.
- [ ] #4 Fallback site search remains disabled by default.
- [ ] #5 Raw signed/token-bearing URLs do not appear in API responses, snapshots, logs, or candidate ids.
- [ ] #6 Existing provider-specific paper-search endpoints are unchanged.
- [ ] #7 Focused pytest commands and adjacent research tests pass.
- [ ] #8 Bandit touched-scope scan has no new actionable findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Subagent readiness review found one plan contradiction: a successful empty-source-default service test used db_factory=None even though successful discovery must persist a snapshot. Patched the plan so that test uses ResearchSessionsDB(tmp_path / "research.db") and asserts the snapshot exists. Ready to dispatch Task 1 implementer after committing this plan hardening.
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
