---
id: TASK-15583
title: The WebUI cannot pick a provider the server has no key for, even with the user's
  own key
status: In Progress
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
- [ ] #2 A provider the admin explicitly disables (is_enabled=False), or a model on its denied_models, stays unselectable for every user, own key or not; a model off its allowed_models is selectable only for a user whose own key serves that provider (owner's decision, 2026-11-21, TASK-15582)
- [ ] #3 The listing for one user never reveals another user's keys or models, and a server-key-only user's listing is unchanged
- [ ] #4 The docs that describe the provider listing say how a user's own key affects it
- [ ] #5 With no server key there is no model inventory, so the picker gets the provider's models from the user's own key (or accepts a typed model ID), never from another user's key
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC #2 amended per the owner's decision of 2026-11-21 (TASK-15582, chose 'Server key only'). It read: 'A provider the admin explicitly disables (is_enabled=False), or a model off its allowed_models, stays unselectable for every user, own key or not'. A model off allowed_models is now only not offered on the server's key and is usable with a user's own key; denied_models is the explicit model disallow, refused on every key. Plan: Docs/superpowers/plans/2026-11-21-webui-byok-provider.md.
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
