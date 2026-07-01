---
id: TASK-12074
title: Fix post-PR-1982 MkDocs deploy verification failure
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-30 07:22'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/actions/runs/28421862122'
priority: high
modified_files:
  - .github/workflows/mkdocs.yml
  - Helper_Scripts/refresh_docs_published.sh
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the post-PR #1982 MkDocs Pages deploy verification failure by making the curated Docs/Published output satisfy the workflow's expected public-docs structure while preserving the current published-docs warning baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The docs refresh script creates Docs/Published/index.md.
- [x] #2 The docs refresh script publishes Evaluations from Docs/Evaluations or the current Docs/Evals source.
- [x] #3 The MkDocs workflow builds with Docs/mkdocs.yml and the curated-docs verification block passes.
- [x] #4 Known strict-mode baseline warnings are documented instead of re-breaking the deploy workflow.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed as part of PR #2557. The workflow now uses the real Docs/mkdocs.yml config path, and strict mode remains disabled because mkdocs build --strict -f Docs/mkdocs.yml currently aborts on 106 existing docs warnings unrelated to this deploy-verification fix. The evaluations-source fallback is now guarded in TASK-12075 follow-up work so missing source dirs fail fast.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the post-PR-1982 Deploy MkDocs to GitHub Pages verification failure by making the published docs refresh create Docs/Published/index.md, using Docs/Evals as the source for the expected Docs/Published/Evaluations output when Docs/Evaluations is absent, and updating the Pages workflow to build with the real Docs/mkdocs.yml config without strict mode because the current published docs have a broad baseline of link warnings. Verification: refresh script completed; the failed workflow Verify curated docs block passed locally with 223 Markdown files; check_public_private_boundary.py passed; mkdocs build -f Docs/mkdocs.yml passed; bash -n passed; pre-commit on workflow/script/task files passed; git diff --check --cached passed. Bandit skipped because this change only touches shell/YAML/docs task metadata.
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
