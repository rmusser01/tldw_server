---
id: TASK-12128
title: Document external docs hosting links
status: In Progress
labels:
- docs
documentation:
- Docs/superpowers/specs/2026-07-04-external-docs-hosting-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure README and website copy point readers to the canonical external docs page at tldwproject.org/server/docs, and document how the MkDocs docs are built, updated, and administered for that external host.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 README links to https://tldwproject.org/server/docs/ as the public docs site.
- [ ] #2 Docs website landing page links to https://tldwproject.org/server/docs/.
- [ ] #3 Docs site guide explains external hosting at /server/docs and does not present GitHub Pages as the production target.
- [ ] #4 Existing MkDocs build path remains documented and locally verifiable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Canonical docs live at https://tldwproject.org/server/docs/. GitHub Pages remains enabled as a mirror from the same MkDocs source; do not disable the Pages deploy.
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
