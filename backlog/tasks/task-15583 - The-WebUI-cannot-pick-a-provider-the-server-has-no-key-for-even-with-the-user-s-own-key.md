---
id: TASK-15583
title: The WebUI cannot pick a provider the server has no key for, even with the user's
  own key
status: Done
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
- [x] #1 A user with a stored own key for a provider the server has no key for can pick that provider and its models in the WebUI's model pickers, with tests
- [x] #2 A provider the admin explicitly disables (is_enabled=False), or a model on its denied_models, stays unselectable for every user, own key or not; a model off its allowed_models is selectable only for a user whose own key serves that provider (owner's decision, 2026-11-21, TASK-15582)
- [x] #3 The listing for one user never reveals another user's keys or models, and a server-key-only user's listing is unchanged
- [x] #4 The docs that describe the provider listing say how a user's own key affects it
- [x] #5 With no server key there is no model inventory, so the picker gets the provider's models from the user's own key (or accepts a typed model ID), never from another user's key
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC #2 amended per the owner's decision of 2026-11-21 (TASK-15582, chose 'Server key only'). It read: 'A provider the admin explicitly disables (is_enabled=False), or a model off its allowed_models, stays unselectable for every user, own key or not'. A model off allowed_models is now only not offered on the server's key and is usable with a user's own key; denied_models is the explicit model disallow, refused on every key. Plan: Docs/superpowers/plans/2026-11-21-webui-byok-provider.md.
Measured. Task 1: RED 12 failed (no listing_user_id); GREEN 202 passed (10 files), netprobe 8 pre-existing, 0 from the new file; mutations A 1 failed, B 4, C 2, D 1 (as planned). Task 2: RED 2 failed; GREEN 4 files, 117 tests passed (symlinked node_modules, removed). Sweep 20 files + new: 488 passed before, 500 after, no id differs, netprobe 17 both. tests/Docs 214 passed. tests/lint 331 passed. Gates: test_quality_triage --enforce, shard coverage, backlog ids, scope predicate all rc 0; ruff reports the same 2 findings as the base copy; git diff --check clean; OpenAPI fingerprint and paths match. Bandit not run (not installed; no download).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The provider listings (GET /llm/providers, /llm/providers/{provider_name}, /llm/models, /llm/models/metadata) read a signed-in user's credentials through a dependency that never refuses, so they stay public; a user's listing is marked private to caches. Each commercial provider the user's own stored key serves at its vendor (users_own_key_credentials) is listed on that key: configured and enabled, with the models that key lists, discovered as chat discovers them and cached by the key's digest, so a listing never shows another user's key or models. The admin override holds such an entry only to what every key refuses (a disabled provider, a denied model), applied by the listing itself, so a denied default model does not hide the provider; a model off allowed_models stays (owner's decision, 2026-11-21). Anonymous and keyless listings are unchanged. The WebUI's filters needed no change; the provider keys page refetches the pickers after a save or delete. AC #2 was amended for the owner's decision. The Providers API doc, the BYOK guide, llm-providers and the CHANGELOG say so.
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
