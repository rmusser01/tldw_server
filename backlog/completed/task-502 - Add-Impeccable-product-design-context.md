---
id: TASK-502
title: Add Impeccable product design context
status: Done
labels:
- design-system
- docs
- impeccable
priority: medium
modified_files:
- PRODUCT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create PRODUCT.md for the Impeccable design workflow so future frontend design commands have project-specific strategic context. This records the default register, users, product purpose, brand personality modes, anti-references, design principles, and accessibility baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRODUCT.md exists at the repository root.
- [x] #2 PRODUCT.md follows the Impeccable teach format with Register, Users, Product Purpose, Brand Personality, Anti-references, Design Principles, and Accessibility & Inclusion sections.
- [x] #3 The captured context reflects the confirmed product register, multi-mode personality, and UI guardrails.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use the Impeccable teach flow: synthesize discovered project context and the user's confirmed answers, write PRODUCT.md, run the Impeccable context loader to refresh, then update this Backlog task with touched files and verification results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PRODUCT.md with default product register, inferred tldw user and purpose context, the user's confirmed multi-mode personality direction, anti-references against generic AI/SaaS gloss and toy-like UI, five strategic design principles, and WCAG AA plus reduced-motion accessibility baseline. Verification: `node /Users/macbook-dev/.codex/skills/impeccable/scripts/load-context.mjs` returned hasProduct=true and hasDesign=false. Bandit skipped because this was a Markdown-only context change.
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
