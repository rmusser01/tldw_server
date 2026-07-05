---
id: TASK-12869
title: Split published docs navigation into user and developer wiki entry points
status: Done
labels:
- docs
- mkdocs
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create audience-focused user and developer wiki entry points in the existing MkDocs site without moving existing source docs or breaking published links.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Published docs expose User Wiki and Developer Wiki top-level entry points.
- [x] #2 Docs/Wiki source pages are synced into Docs/Published by the refresh script.
- [x] #3 MkDocs navigation is organized around the audience split while preserving existing guide/reference links.
- [x] #4 README and docs-site guide explain which wiki to use and where new docs belong.
- [x] #5 Focused docs tests and MkDocs build pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use the existing single MkDocs site. Add Docs/Wiki landing pages, publish them through Helper_Scripts/refresh_docs_published.sh, reorganize Docs/mkdocs.yml nav around User Wiki and Developer Wiki, update README and Docs_Site_Guide, and add focused docs contract tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented one MkDocs/GitHub Pages docs site with audience-first entry points instead of a second documentation build. Added `Docs/Wiki` source landing pages, refreshed `Docs/Published/Wiki`, published contributor-oriented `Architecture.md` and `ADR` content, reorganized `Docs/mkdocs.yml` around `User Wiki` and `Developer Wiki`, updated README and docs-site guidance, and added a focused docs contract test.

Verification:
- `bash Helper_Scripts/refresh_docs_published.sh`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Docs/test_docs_audience_wikis.py` -> 3 passed
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Docs` -> 120 passed
- `.venv/bin/python Helper_Scripts/docs/check_public_private_boundary.py`
- `.venv/bin/python Helper_Scripts/docs/check_readme_docs_path_hygiene.py`
- `.venv/bin/python Helper_Scripts/docs/check_top_guides_docs_path_hygiene.py`
- `.venv/bin/python Helper_Scripts/docs/check_onboarding_command_boundaries.py`
- `.venv/bin/python Helper_Scripts/docs/check_onboarding_endpoint_drift.py`
- `.venv/bin/python -m mkdocs build -f Docs/mkdocs.yml` -> exit 0 with existing baseline warnings
- `.venv/bin/python -m bandit -r tldw_Server_API/tests/Docs/test_docs_audience_wikis.py tldw_Server_API/tests/Docs/conftest.py -f json -o /tmp/bandit_task_12119_docs.json` -> 0 findings
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added audience-focused User Wiki and Developer Wiki entry points to the existing MkDocs docs site without moving existing source docs. Updated the publish script, generated published docs, MkDocs navigation, README routing, docs-site guidance, CI curated-doc verification, and docs tests for the new structure.
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
