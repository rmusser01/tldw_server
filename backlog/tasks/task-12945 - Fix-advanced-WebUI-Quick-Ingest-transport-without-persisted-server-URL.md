---
id: TASK-12945
title: Fix advanced WebUI Quick Ingest transport without persisted server URL
status: In Progress
labels:
- frontend
- quick-ingest
- bug
- uat
priority: high
ordinal: 12105
references:
- 'PR #2702 release UAT'
- https://github.com/rmusser01/tldw_server/pull/2702
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT on release 0.1.40 found that Quick Ingest direct uploads fail before reaching the backend when advanced WebUI mode is configured via NEXT_PUBLIC_API_URL but tldwConfig.serverUrl is unset. Fix the shared request guard, correct the misleading configuration error classification, and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Direct browser upload requests use the resolved advanced NEXT_PUBLIC_API_URL origin when persisted serverUrl is unset.
- [ ] #2 Advanced requests still fail closed when neither persisted nor runtime API origin resolves to a valid HTTP origin.
- [ ] #3 A missing server configuration is not presented as an unsupported file format.
- [ ] #4 Focused frontend tests and typecheck pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
