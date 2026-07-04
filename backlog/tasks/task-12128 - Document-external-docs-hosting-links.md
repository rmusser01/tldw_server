---
id: TASK-12128
title: Document external docs hosting links
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-04 05:30'
labels:
  - docs
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-04-external-docs-hosting-design.md
  - >-
    Docs/superpowers/plans/2026-07-04-external-docs-hosting-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure README and website copy point readers to the canonical external docs page at tldwproject.com/server/docs, and document how the MkDocs docs are built, updated, mirrored, and administered for that external host.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 README links to https://tldwproject.com/server/docs/ as the public docs site.
- [x] #2 Docs website landing page links to https://tldwproject.com/server/docs/.
- [x] #3 Docs site guide explains external hosting at /server/docs, manual deploy, optional clone/pull/build automation, and GitHub Pages as a mirror.
- [x] #4 Existing MkDocs build path remains documented and locally verifiable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-07-04-external-docs-hosting-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Modified README.md, Docs/Website/index.html, Docs/mkdocs.yml, and Docs/Code_Documentation/Docs_Site_Guide.md.

Verification: bash Helper_Scripts/refresh_docs_published.sh exited 0; python3 -m mkdocs build -f Docs/mkdocs.yml exited 0 with existing baseline docs warnings.

Verification: canonical tldwproject.com/server/docs URL appears in README, website, MkDocs config, and docs guide; stale tldwproject.org/server/docs check returned no matches; MkDocs site_url check returned one match.

Bandit skipped: touched implementation files are Markdown, static HTML links, and YAML docs metadata only.

Known skip: broad Docs/Published refresh drift was not committed; CI and external deploy docs still run the refresh before building.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Linked README and the tldwproject.com website source to the canonical public docs URL, updated MkDocs site_url to tldwproject.com/server/docs, and revised the docs site guide to document manual external deployment, optional site-side clone/pull/build automation, and GitHub Pages as a mirror.
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
