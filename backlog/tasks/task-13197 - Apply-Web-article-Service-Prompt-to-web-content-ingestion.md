---
id: TASK-13197
title: Apply Web article Service Prompt to web-content ingestion
status: In Progress
assignee: []
created_date: '2026-09-06 01:34'
updated_date: '2026-09-06 01:52'
labels: []
dependencies: []
documentation:
  - Docs/Design/web-ingest-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the merged media.web.summarization setting to ingest-web-content using one authenticated-owner snapshot and recover real ephemeral crawl results. Scheduled workflows remain unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All four ingest scraping modes use one owner-scoped saved prompt snapshot.
- [x] #2 Explicit parts including empty strings, historical defaults, provider behavior, and disabled analysis remain compatible.
- [x] #3 URL-level and recursive ingestion return articles from real ephemeral task results.
- [x] #4 The existing Settings entry documents web-content ingestion; no new prompt setting is added.
- [x] #5 Focused tests, lint, security validation, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Record approved design and establish regression baseline. Stage 2: Add failing end-to-end prompt and crawl-result tests. Stage 3: Implement minimal resolver and orchestration wiring and update existing Settings metadata. Stage 4: Run compatibility checks, Bandit, review, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline: 40 passed. Targeted RED tests demonstrated missing owner prompt lookup and discarded enhanced URL-level results before implementation. Implemented shared explicit-parameter resolver, authenticated ingestion snapshot, minimal ephemeral result retrieval, and existing Settings copy update. Compatibility/registry/API tests: 136 passed. Shared Settings and WebUI Settings: 76 passed each. Extra crawl-result and lookup cleanup boundary selection: 17 passed. Bandit on five touched runtime files: zero findings. OpenAPI export/type generation and fingerprint check passed unchanged. Full prompt matrix and independent review in progress.

Independent reviewer found no blocking correctness/security/compatibility/over-engineering issues. Added the requested direct-ingestion unscoped regression (real model assembly and saved owner storage) to make that compatibility requirement explicit. Initial complete feature matrix: 75 passed. Final rerun includes this regression, connection cleanup cases, ephemeral boundaries, and strict forwarding contracts. Ruff is clean on changed code; existing B004 at process_web_scraping.py:41 is identical on dev and not modified. Python compilation passed. Temporary dependency symlinks removed without modifying shared dependencies.

Final verification: 87 feature/owner/cleanup/crawl/strict-contract tests passed; 136 registry/API/ingestion compatibility tests passed (223 focused backend total). Shared Settings 76 passed and WebUI Settings 76 passed (152 UI total). Both full feature runs eventually reported success; interpreter/session cleanup was slow, and a stack sample located finalization outside application request handling. No broad full-repository suite was run. All new lint issues fixed; one unchanged pre-existing B004 remains in the touched older endpoint. Bandit: zero findings. OpenAPI fingerprint unchanged. Independent review complete; direct-ingestion regression added. All four implementation stages complete; temporary plan removed per repository convention. Ready for integration choice; no PR created yet.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended the existing Web article summarization Service Prompt to authenticated web-content ingestion with one immutable owner snapshot across all four scraping modes. Preserved independent explicit overrides, disabled behavior, engine defaults and unscoped service calls. Recovered enhanced crawl articles from existing ephemeral storage while retaining legacy inline results. Updated the shared Settings scope description without adding another setting. Added real HTTP/storage/model-facing regressions and crawl-result boundary coverage. Verified 223 focused backend and 152 UI tests, clean Bandit, unchanged OpenAPI, and independent review.
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
