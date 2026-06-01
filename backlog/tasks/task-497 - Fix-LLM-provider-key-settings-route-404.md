---
id: TASK-497
title: Fix LLM provider key settings route 404
status: In Progress
labels:
- webui
- extension
- settings
documentation:
- Docs/superpowers/specs/2026-06-01-llm-provider-key-settings-route-design.md
modified_files:
- Docs/superpowers/specs/2026-06-01-llm-provider-key-settings-route-design.md
- backlog/tasks/task-497 - Fix-LLM-provider-key-settings-route-404.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track design and implementation for making the LLM provider key management settings page reachable from the WebUI and extension settings navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 /settings/provider-keys resolves in the hosted WebUI instead of showing a 404.
- [ ] #2 /settings/provider-keys resolves in the extension/options route registry.
- [ ] #3 Existing provider key management behavior is reused without conflating provider keys with tldw server authentication.
- [ ] #4 Route coverage prevents regression for the provider key settings path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming approved: make /settings/provider-keys a first-class route in hosted WebUI and extension/options routing, reusing existing ProviderKeysSettings and adding route coverage.

Spec review completed by subagent on 2026-06-01. Status: Approved. No blocking issues found. Advisory recommendation: optionally name expected test locations during implementation planning.
<!-- SECTION:PLAN:END -->

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
