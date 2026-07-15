---
id: TASK-12115
title: Add first-class standalone HTML+JS presentation generation
status: In Progress
labels:
- slides
- presentation-studio
- backend
- frontend
- security
priority: High
documentation:
- Docs/superpowers/specs/2026-07-15-standalone-html-presentations-design.md
modified_files:
- Docs/superpowers/specs/2026-07-15-standalone-html-presentations-design.md
- backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a hardened standalone HTML+JavaScript presentation mode shared across existing Slides source types, with a form-first Presentation Studio flow, strict content-kind invariants, bounded LLM output, explicit-save editing, opt-in sandboxed execution, safe attachment export, compatibility guards, tests, documentation, and security verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approved design spec and implementation plan document the architecture, security boundary, compatibility behavior, and deferred scope.
- [ ] #2 The Slides backend supports structured_slides and standalone_html as explicit, validated content kinds without permitting split-brain records.
- [ ] #3 Standalone HTML generation uses one shared mode-aware service across supported source kinds and an allowlisted provider/model policy.
- [ ] #4 Presentation Studio exposes a form-first HTML generation flow and a dedicated code, static preview, opt-in interactive preview, save, conflict, and download experience.
- [ ] #5 HTML is never executed by server renderers or served inline; structured-only operations reject HTML with stable errors.
- [ ] #6 Legacy presentations and clients remain structured by default, database/version migrations are covered, and capabilities fail closed.
- [ ] #7 Focused backend, frontend, security, integration, and E2E tests pass, and Bandit reports no new findings in touched Python.
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
