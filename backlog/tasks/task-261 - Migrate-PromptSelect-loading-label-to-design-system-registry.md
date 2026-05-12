---
id: TASK-261
title: Migrate PromptSelect loading label to design-system registry
status: Done
assignee: []
created_date: '2026-05-11 15:50'
updated_date: '2026-05-12 00:07'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the frontend design-system product-state cleanup by replacing the Common PromptSelect modal loading fallback with the canonical design-system loading state label while preserving the existing modal behavior and translation lookup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PromptSelect loading fallback uses getDesignSystemState('loading').label instead of hardcoded Loading
- [x] #2 Existing PromptSelect modal behavior and translation key remain unchanged
- [x] #3 Focused coverage proves editor loading copy uses the design-system loading label fallback
- [x] #4 Matching PromptSelect Loading canonical-state-label baseline exception is removed and verifier passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red coverage for PromptSelect editor loading fallback using a mocked design-system loading label. 2. Route the common:loading fallback through getDesignSystemState('loading').label while preserving the translation key and modal flow. 3. Remove the PromptSelect canonical-state-label baseline exception and refresh current dev baseline drift needed for the verifier. 4. Verify focused PromptSelect test, product-state guard test, design-system verifier, diff checks, and touched-path typecheck output.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PromptSelect loading fallback registry lookup via getDesignSystemState('loading').label while keeping the common:loading translation key. Added focused coverage that first failed against the hardcoded Loading fallback and now passes with a mocked registry label.

Removed the PromptSelect canonical-state-label baseline exception. The full design-system verifier also required refreshing current origin/dev baseline IDs for AgentRegistry and ChatbooksPlaygroundPage AntD Alert findings after unrelated baseline drift.

Verification: PromptSelect focused Vitest passed; product-state guard Vitest passed; bun run verify:design-system-state passed; git diff --check passed; filtered UI tsc output for PromptSelect/baseline/AgentRegistry/Chatbooks returned no matches while full tsc remains on existing repo-wide baseline errors. Bandit not applicable because touched implementation files are TypeScript/JSON/Backlog markdown only.

PR: https://github.com/rmusser01/tldw_server/pull/1574

PR review follow-up: Gemini suggested optional chaining for the design-system loading label access. Applied getDesignSystemState('loading')?.label, preserving the registry fallback behavior when present while avoiding a throw if the registry is malformed.

Review follow-up verification: PromptSelect focused Vitest passed; product-state guard Vitest passed; bun run verify:design-system-state passed; git diff --check passed; filtered UI tsc output for PromptSelect/baseline/AgentRegistry/Chatbooks returned no matches.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PromptSelect now uses the design-system loading state label as the fallback for its system prompt editor loading copy with defensive optional chaining, preserving the existing translation key and modal behavior. Focused test coverage proves the registry fallback, the PromptSelect baseline exception is removed, and current dev baseline drift was refreshed so the product-state verifier passes.
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
