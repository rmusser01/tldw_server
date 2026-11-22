---
id: TASK-15615
title: Chat refuses an off-list model on the server's key before input moderation
  runs
status: To Do
labels:
- llm
- byok
- policy
- chat
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minor M2 of PR #3544's final review (TASK-15582). Since #3544, chat's first provider-override check passes allowlist=False, because the credential source is not known yet. A request for a model off the override's allowed_models that runs on the server's key is therefore refused (model_not_allowed) only by the runtime check after credential resolution (tldw_Server_API/app/api/v1/endpoints/chat.py, about line 4671), after input moderation (about line 4602) has run. Topic monitoring, supervised-policy and audit side effects run for a request that is refused anyway, and a flagged input gets the moderation response instead of model_not_allowed. Before #3544 the 403 came before moderation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A chat request for a model off allowed_models that would run on the server's key is refused model_not_allowed before input moderation, topic monitoring or the audit side effects run, with a test
- [ ] #2 A user's own key (credential source user) still runs a model off allowed_models, and a disabled provider or a denied model is still refused before moderation, with tests
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
