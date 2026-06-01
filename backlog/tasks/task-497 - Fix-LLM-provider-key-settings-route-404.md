---
id: TASK-497
title: Fix LLM provider key settings route 404
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 06:31'
labels:
  - webui
  - extension
  - settings
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-01-llm-provider-key-settings-route-design.md
  - >-
    Docs/superpowers/plans/2026-06-01-llm-provider-key-settings-route-implementation-plan.md
modified_files:
  - Docs/superpowers/specs/2026-06-01-llm-provider-key-settings-route-design.md
  - Docs/superpowers/plans/2026-06-01-llm-provider-key-settings-route-implementation-plan.md
  - apps/tldw-frontend/__tests__/pages/settings-provider-keys-route.test.tsx
  - apps/tldw-frontend/pages/settings/provider-keys.tsx
  - apps/tldw-frontend/__tests__/extension/route-registry.stability.test.ts
  - apps/tldw-frontend/extension/routes/route-registry.tsx
  - apps/packages/ui/src/routes/__tests__/deferred-options-route.test.tsx
  - apps/packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts
  - backlog/tasks/task-497 - Fix-LLM-provider-key-settings-route-404.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track design and implementation for making the LLM provider key management settings page reachable from the WebUI and extension settings navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /settings/provider-keys resolves in the hosted WebUI instead of showing a 404.
- [x] #2 /settings/provider-keys resolves in the extension/options route registry.
- [x] #3 Existing provider key management behavior is reused without conflating provider keys with tldw server authentication.
- [x] #4 Route coverage prevents regression for the provider key settings path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming approved: make /settings/provider-keys a first-class route in hosted WebUI and extension/options routing, reusing existing ProviderKeysSettings and adding route coverage.

Spec review completed by subagent on 2026-06-01. Status: Approved. No blocking issues found.

Follow-up design audit completed after user review request: clarified that the shared package shell resolves settings deep links via apps/packages/ui/src/routes/option-settings-route-registry.tsx and DeferredOptionsRoute, while the extension/options shell needs its own apps/tldw-frontend/extension/routes/route-registry.tsx entry. Added suggested route test locations/patterns.

Implementation plan written: Docs/superpowers/plans/2026-06-01-llm-provider-key-settings-route-implementation-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented in reviewed slices:
- Hosted WebUI now has a /settings/provider-keys page shim that lazy-loads ProviderKeysSettings instead of falling through to 404.
- Extension/options route registry now registers /settings/provider-keys with ProviderKeysSettings and the provider key nav token.
- Shared route tests now cover DeferredOptionsRoute resolution for /settings/provider-keys and source-contract alignment between option-settings-route-registry and settings-nav-config.

Verification recorded:
- bunx vitest run __tests__/pages/settings-provider-keys-route.test.tsx __tests__/extension/route-registry.stability.test.ts ../packages/ui/src/routes/__tests__/deferred-options-route.test.tsx ../packages/ui/src/routes/__tests__/option-settings-provider-keys-route.test.ts -> 4 files passed, 11 tests passed.
- rg -n "settings/provider-keys|ProviderKeysSettings|TldwSettings" over hosted settings pages, extension route registry, and shared option settings registry confirmed route presence in all expected surfaces.
- Bandit is not applicable to the touched implementation/test files because this change is TS/TSX-only frontend routing/test coverage; no Python code was modified.
- repo-wide git diff --check is currently blocked by unrelated pre-existing whitespace in Docs/Design/Agents.md:155; scoped checks for touched files passed during worker verification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the LLM provider key settings 404 by making /settings/provider-keys reachable in both hosted WebUI and extension/options routing. The implementation reuses the existing ProviderKeysSettings component for provider API key management, keeps it separate from tldw server authentication settings, and adds focused regression coverage for the hosted page, extension route registry, shared deferred route resolution, and shared registry/navigation alignment.
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
