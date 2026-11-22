---
id: TASK-15615
title: Chat refuses an off-list model on the server's key before input moderation
  runs
status: Done
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
- [x] #1 A chat request for a model off allowed_models that would run on the server's key is refused model_not_allowed before input moderation, topic monitoring or the audit side effects run, with a test
- [x] #2 A user's own key (credential source user) still runs a model off allowed_models, and a disabled provider or a denied model is still refused before moderation, with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Measured: t2red '8 failed, 2 passed' (plan: 7/3; the own-key row also failed because RR1's user-id assertion needs the lookup, which the old code never asked; the six helper rows failed on AttributeError and server-key on moderated ['input'] != []); t2green '10 passed'; RR1 mutation (None at the call site) fails the own-key row ('[None] and None not in [None]'); create_chat_completion 2904 lines; tests/Docs 214 passed; t2files 379 passed, 1 skipped, 0 failed, comm against baseline empty, netprobe 0. Bandit: not run (not installed). Known limits: a key the lookup counts as own that resolves off the vendor's host is still refused after moderation (credential runtime or adapter guard); a failed lookup counts as not own (fails closed); an off-list request makes one own-key lookup before moderation; no test combines a chat macro with an override, so the call site's macro flag is unpinned as on dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chat's first admin-override check (/chat/completions, before input moderation) refuses a model off allowed_models with model_not_allowed unless the user's own key serves the provider. It asks users_own_key_serves (the pre-resolution own-key test of ADR-083's quota exemption) for the user the credential runtime resolves for, and only when the allowlist alone would refuse, so other requests make no extra read; a chat macro, which runs later on the server's key, is always held to the list. A request the server's key would refuse is again refused before input moderation, topic monitoring, supervised policy and their audit records run; the user's own key still runs the model; a disabled provider or denied model is refused before moderation on every key; the credential runtime still checks the list on the key it resolves. ADR-086 gains an amendment.
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
