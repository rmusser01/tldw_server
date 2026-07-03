---
id: TASK-12105
title: Address PR 2573 code review placeholder and popup issues
status: Done
labels:
- webui
- auth
- chat
- code-review
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve independent code-review findings on PR #2573: runtime single-user API key placeholders should not be treated as valid auth, backend custom OpenAI readiness should reject known placeholder keys, and CharacterSelect should not use deprecated AntD popupClassName.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime single-user API key override rejects known placeholder keys before request auth and readiness checks use it.
- [x] #2 Backend custom OpenAI provider readiness rejects known placeholder API keys, including env-backed OpenAI-hosted custom endpoints.
- [x] #3 CharacterSelect no longer uses AntD Select popupClassName and has guard coverage for the replacement API.
- [x] #4 Focused regression tests cover the review findings and pass.
- [x] #5 Security/format verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Addressed PR #2573 review findings by sharing placeholder-key rejection through runtime auth normalization, request auth gating, connection readiness checks, and hosted frontend auth storage. Added backend placeholder filtering for custom OpenAI provider API keys so placeholder env/config values do not mark providers configured. Replaced the remaining CharacterSelect AntD Select popupClassName usage with classNames.popup.root and extended the source guard.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: UI package focused suite passed (8 files, 121 tests); frontend auth/runtime suite passed (2 files, 33 tests); backend provider/capability suite passed (22 tests); connection regression retest passed (30 tests); deprecated AntD prop scan found no non-test UI matches; git diff --check passed; Bandit on provider_config_resolution.py reported zero findings.
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
