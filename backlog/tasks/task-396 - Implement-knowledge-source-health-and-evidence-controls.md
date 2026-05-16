---
id: TASK-396
title: Implement /knowledge source health and evidence controls
status: In Progress
assignee: []
created_date: '2026-05-16 00:51'
updated_date: '2026-05-16 01:25'
labels:
  - webui
  - knowledge
  - ux
  - feature
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-knowledge-source-health-evidence-controls-plan.md
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GET /api/v1/rag/source-health returns safe pre-query health for canonical Knowledge QA sources without altering search-response metadata.source_status.
- [ ] #2 Knowledge QA shows source health before search and keeps search usable when health loading fails.
- [ ] #3 Knowledge QA answer and evidence surfaces show compact trust/evidence controls without adding durable evidence persistence.
- [ ] #4 Focused backend, frontend, extension parity, diff-check, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 backend source-health contract implemented. Focused tests passed: pytest -q test_source_health.py test_rag_source_health_endpoint.py test_source_contract.py test_unified_pipeline.py -k 'source_status or source_health or source_contract' (8 passed, 23 deselected). git diff --check passed. Bandit touched backend scope passed with 0 findings at /tmp/bandit_knowledge_source_health_task1.json.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
