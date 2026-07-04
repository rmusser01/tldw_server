---
id: TASK-12138
title: Fix GitHub Pages docs deployment artifact path
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 07:16'
labels:
  - docs
  - deployment
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the GitHub Pages mirror to deploy the MkDocs docs by correcting the workflow artifact path and verifying a docs build/deploy artifact for tldwproject.com/server/docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GitHub Pages is enabled with workflow build source.
- [x] #2 MkDocs Pages workflow uploads the actual site output directory.
- [x] #3 A clean dev docs build artifact is produced for external tldwproject.com/server/docs deployment.
- [x] #4 Verification commands/results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
GitHub Pages was enabled via the GitHub API with build_type=workflow. Fixed .github/workflows/mkdocs.yml to run on dev pushes and upload Docs/site, which is the actual output directory when building with -f Docs/mkdocs.yml. Local verification from clean origin/dev worktree: bash Helper_Scripts/refresh_docs_published.sh exited 0; python3 Helper_Scripts/docs/check_public_private_boundary.py exited 0; python3 -m mkdocs build -f Docs/mkdocs.yml exited 0 with baseline docs warnings. External deploy artifact: /tmp/tldw-server-docs-site-06674d8931.tar.gz, SHA-256 d36e7e7263cd089148d12f4dfd35ab2dc15bd7d9673483d814c702bfc06974f6.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Bandit skipped: touched files are a GitHub Actions workflow YAML and Backlog task metadata only.

Review follow-up: kept dev in the workflow trigger for build verification only, added a deploy job branch guard for main/PG-Backend, and moved the Pages concurrency group from workflow scope to deploy-job scope so dev builds cannot cancel stable deploy runs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Enabled the GitHub Pages workflow source, corrected the MkDocs workflow artifact path to Docs/site, kept dev pushes as build-only verification, guarded Pages deployment to main/PG-Backend, and produced a verified local docs archive for copying to tldwproject.com/server/docs.
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
