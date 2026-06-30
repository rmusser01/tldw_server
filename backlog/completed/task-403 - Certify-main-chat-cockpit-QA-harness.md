---
id: TASK-403
title: Certify main /chat cockpit QA harness
status: Done
labels:
- chat
- cockpit
- webui
- qa
- e2e
- certification
priority: HIGH
references:
- Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR8 slice for the main WebUI /chat cockpit maturity roadmap. Keep scope to the main chat page cockpit: durable focused unit coverage, real-server Playwright proof, visual QA evidence, and a checkbox-by-checkbox certification artifact for merge readiness. Do not touch browser extension sidepanel/sidebar surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Durable unit coverage protects cockpit summaries, rails, accessibility, keyboard/focus behavior, and responsive transitions where gaps remain.
- [x] #2 Real-server Playwright proof covers prompt, persona, model settings, MCP populated/unavailable distinction, conversation send, mobile, focus/cockpit transitions, and screenshots without mocked routes.
- [x] #3 Certification artifact maps roadmap PR8 and remaining merge criteria to current evidence item-by-item.
- [x] #4 No sidepanel/sidebar/browser-extension files are modified for this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Plan: Docs/superpowers/plans/2026-05-16-chat-cockpit-qa-certification.md
- Certification: Docs/superpowers/specs/2026-05-16-main-chat-cockpit-merge-certification.md
- Added keyboard-specific cockpit control coverage to apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx.
- Real-server Playwright screenshot artifacts were generated under apps/tldw-frontend/test-results for desktop, focus, mobile, prompt/model/MCP, character, persona, and model-provider conversation states.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the PR8 QA/certification slice for the main WebUI /chat cockpit. Added a keyboard regression guard for rail/focus/mobile tab controls, created a merge certification checklist tied to concrete unit and real-server E2E evidence, and verified the suite against the running server without mocked backend routes.
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
