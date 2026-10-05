---
id: TASK-13453
title: Implement Knowledge UX audit repairs and enhancements
status: In Progress
labels:
- knowledge
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the user-approved K01-K15 repairs and five enhancement themes from Docs/Reviews/KNOWLEDGE_NNG_UX_REVIEW_2026_10_04.md. Defaults approved: ask added items after ingest; continue in Research Workspace with sources attached. Coordinate four reviewable implementation units and browser verification on latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All K01-K15 have implemented fixes and regression evidence
- [ ] #2 Named source-set handoffs, readiness summaries, outcome recipes, saved-output review loop and extension parity are implemented
- [ ] #3 Research continuation preserves source identity, excerpts, trust and inspectable attachments
- [ ] #4 Both WebUI and native extension are verified; checks, review, security scope and owned-service cleanup recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User approved the audited solutions and both default choices on 2026-10-04. Work happens in the attached isolated checkout; original dirty checkout is untouched. ADR check: no new ownership or public-API rule; ADR-007 canonical ResearchWorkspace, ADR-008 split persistence and source lineage, and ADR-053 mixed-source retrieval govern. Reuse existing ingestion for non-media evidence snapshots and existing shared UI on both surfaces. Reassess ADR scope only if those existing contracts cannot express a required repair.
Approved design: Docs/Design/2026-10-04-knowledge-ux-remediation.md. Four-unit execution plan: IMPLEMENTATION_PLAN_knowledge_ux_remediation_20261004.md. All implementation is based on 75ab224081bf140ef52017c1a9b0a04f6878d488 plus the committed audit. Sequential implementation and independent task reviews; no new packages or public API.
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
