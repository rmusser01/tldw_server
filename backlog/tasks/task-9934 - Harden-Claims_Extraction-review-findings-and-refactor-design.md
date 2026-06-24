---
id: TASK-9934
title: Harden Claims_Extraction review findings and refactor design
status: In Progress
assignee: []
created_date: '2026-06-23 21:39'
updated_date: '2026-06-24 00:52'
labels:
  - claims
  - review-hardening
  - refactor
dependencies: []
references:
  - tldw_Server_API/app/core/Claims_Extraction
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated Claims_Extraction review findings, then capture a focused refactor design for the module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Validated review findings are covered by failing-first regression tests before production changes.
- [ ] #2 Rebuild storage failure cannot soft-delete existing claims and report success.
- [ ] #3 Claims cancellation propagates instead of being swallowed by noncritical exception handling.
- [ ] #4 LLM extraction timeout returns promptly without waiting on stuck worker shutdown.
- [ ] #5 Runtime limits, HTML escaping, analytics scoping, FVA metrics, and notification dispatch reliability are hardened.
- [ ] #6 A focused Claims_Extraction refactor design spec is written for follow-up modularization.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec follow-up review tightened implementation constraints for rebuild strictness, timeout worker bounds, analytics owner-scope SQL, and notification dispatcher saturation behavior.
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
