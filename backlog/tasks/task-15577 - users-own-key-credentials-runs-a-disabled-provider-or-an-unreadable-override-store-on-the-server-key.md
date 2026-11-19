---
id: TASK-15577
title: users_own_key_credentials runs a disabled provider or an unreadable override
  store on the server key
status: To Do
labels:
- llm
- byok
- policy
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15571's plan review. The except clause of users_own_key_credentials (API_Deps/usage_quota_deps.py) catches every resolution failure and returns None, so the request keeps the server key, and the no-handle analyze path checks nothing. A provider the override disables, or an override store that cannot be read, therefore still runs an own-key user's analysis on the server key. That is unlike TASK-15571's dispatch rule, which refuses both (provider_disabled, credential_store_unavailable).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A disabled provider and an unreadable override store refuse an own-key analyze request at the boundary instead of running it on the server key, with tests
- [ ] #2 The other resolution failures keep their current server-key behaviour, or the owner decides otherwise
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
