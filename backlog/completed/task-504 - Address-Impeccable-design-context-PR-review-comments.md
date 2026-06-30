---
id: TASK-504
title: Address Impeccable design context PR review comments
status: Done
labels:
- docs
- impeccable
- review
priority: medium
ordinal: 504
modified_files:
- DESIGN.md
- .impeccable/design.json
- backlog/completed/task-504 - Address-Impeccable-design-context-PR-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR #2058 review comments about invalid alert/recovery token semantics in DESIGN.md and .impeccable/design.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DESIGN.md no longer defines an alert component with identical background and text colors.
- [x] #2 DESIGN.md and .impeccable/design.json use consistent component token names and semantics for info alerts versus warning/recovery callouts.
- [x] #3 The Impeccable loader still reports hasProduct=true and hasDesign=true, and the sidecar JSON parses successfully.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Verify the review finding against the current design context, update the component token names/mappings in DESIGN.md and sidecar refersTo values, run JSON and context-loader verification, and push a follow-up commit to PR #2058.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added dedicated info alert surface/text tokens and dedicated warning recovery callout surface/text tokens in DESIGN.md. Updated .impeccable/design.json so the Recovery Callout component refers to recovery-callout instead of alert-info.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #2058 review comments by separating alert-info from recovery-callout semantics. Verification: sidecar JSON parsed with Node; Impeccable loader returned hasProduct=true and hasDesign=true; rg confirmed recovery-callout sidecar references and no alert token pair uses identical primary colors. Bandit skipped because this was Markdown/JSON design documentation only.
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
