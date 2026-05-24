---
id: TASK-503
title: Add Impeccable visual design context
status: Done
labels:
- design-system
- docs
- impeccable
priority: medium
modified_files:
- DESIGN.md
- .impeccable/design.json
- .gitignore
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create DESIGN.md and the Impeccable design sidecar from the existing tldw WebUI tokens, shared UI primitives, and design-system documentation so future design commands have visual system context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DESIGN.md exists at the repository root and follows the Impeccable document command structure.
- [x] #2 DESIGN.md frontmatter captures existing color, typography, radius, spacing, and component tokens without inventing unrelated visual systems.
- [x] #3 .impeccable/design.json exists and records sidecar metadata for shadows, motion, breakpoints, and representative components.
- [x] #4 Verification with the Impeccable context loader reports hasProduct=true and hasDesign=true.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Scan existing CSS variables, Tailwind config, design-system documentation, and shared UI primitives. Synthesize DESIGN.md using the Google Stitch-compatible six-section structure and add .impeccable/design.json for extensions. Run the Impeccable context loader and record verification in this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added DESIGN.md with Stitch-compatible frontmatter and the six required sections: Overview, Colors, Typography, Elevation, Components, and Do's and Don'ts. Added .impeccable/design.json with color metadata, typography metadata, shadows, motion, breakpoints, rules, and representative component snippets. Added a narrow .gitignore exception so the generated sidecar is trackable despite the repo-wide *.json ignore rule. Verification: sidecar JSON parsed with Node; Impeccable loader returned hasProduct=true and hasDesign=true; checked DESIGN.md/PRODUCT.md sidecar for em dash characters. Bandit skipped because this was Markdown/JSON/.gitignore design documentation only.
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
