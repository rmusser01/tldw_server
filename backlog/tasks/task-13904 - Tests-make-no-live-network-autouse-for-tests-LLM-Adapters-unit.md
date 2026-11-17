---
id: TASK-13904
title: 'Tests: make no_live_network autouse for tests/LLM_Adapters/unit'
status: To Do
labels:
- tests
- llm
dependencies:
- TASK-13903
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-13903, make the no_live_network fixture autouse for tests/LLM_Adapters/unit so the next config-default leak fails at once instead of in a DNS-less environment. LLM_Calls/test_provider_config_resolution.py also records slot-2.example (it fails closed) and needs handling before the fixture can cover tests/LLM_Calls. Only these two directories were measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 no_live_network is autouse for tests/LLM_Adapters/unit and the directory passes
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
