---
id: TASK-15583
title: The WebUI cannot pick a provider the server has no key for, even with the user's
  own key
status: To Do
labels:
- llm
- byok
- webui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15581's plan (2026-11-19). The owner's rule: a provider that is only 'not configured' (the server has no key or credits for it) must still be usable with a user's own key. The API serves it (/chat/completions resolves the user's key first). The WebUI does not offer it: GET /api/v1/llm/providers is server-config only, so a commercial provider the server has no key for is listed is_configured=false, provider_enabled=false, availability=not-configured, readiness_reason_code=provider_not_configured and models=[] (no inventory without a key), also when an override sets is_enabled=true. TldwModels.isSelectableChatModel (apps/packages/ui/src/services/tldw/TldwModels.ts:754) then drops its models, so a user who stored their own key cannot pick that provider in the chat model picker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user with a stored own key for a provider the server has no key for can pick that provider and its models in the WebUI's model pickers, with tests
- [ ] #2 A provider the admin explicitly disables (is_enabled=False), or a model off its allowed_models, stays unselectable for every user, own key or not
- [ ] #3 The listing for one user never reveals another user's keys or models, and a server-key-only user's listing is unchanged
- [ ] #4 The docs that describe the provider listing say how a user's own key affects it
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
