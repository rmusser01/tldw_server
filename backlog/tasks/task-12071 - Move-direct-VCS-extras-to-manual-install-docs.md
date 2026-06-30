---
id: TASK-12071
title: Move direct VCS extras to manual install docs
status: Done
assignee: []
created_date: '2026-06-29 19:53'
updated_date: '2026-06-30 01:06'
labels:
  - packaging
  - pypi
  - ocr
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove direct VCS dependencies from published package metadata so tldw-server can be uploaded to PyPI/TestPyPI. Preserve guidance for users who need optional source-only OCR/TTS backends by documenting manual installation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Published package metadata no longer contains direct git/VCS dependencies.
- [x] #2 Docs explain manual installation for dots-ocr, WePOINTS, and resemble-perth where relevant.
- [x] #3 Package build/check verification confirms the metadata is acceptable locally.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect package metadata and docs references for direct VCS extras. 2. Remove direct VCS dependency entries from pyproject metadata. 3. Add or update docs with manual installation guidance for affected optional backends. 4. Build/check the package and inspect metadata for direct URL dependency removal.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed direct VCS dependency entries from pyproject optional metadata for ocr_dots, ocr_points_transformers, and TTS_chatterbox while preserving normal PyPI-installable dependencies. Updated OCR and Chatterbox setup docs to direct users to manual source installs for dots.ocr, WePOINTS, and Perth. Verification: pyproject TOML parse passed; optional dependency scan reported zero direct URL deps; python -m build --no-isolation built wheel and sdist to /tmp/tldw_pypi_manual_ocr_dots_dist; twine check passed for both artifacts; wheel and sdist metadata inspection reported direct_url_requires=0. Isolated build without --no-isolation was skipped because sandboxed pip could not reach package indexes to install build-system packages.

PR review follow-up: PointsReaderBackend.available() now requires the manual WePOINTS module for transformers mode, logs a specific WePOINTS-missing warning, and import errors from optional/manual dependencies are caught as runtime failures instead of escaping OCR flow. Added focused unit coverage for missing WePOINTS availability and ModuleNotFoundError handling. Verification: pytest tldw_Server_API/tests/MediaIngestion_NEW/test_ocr_backend_points.py -q passed (4 passed); optional dependency scan still reports direct-url-deps 0; git diff --check passed.

Final review-fix verification: Bandit on points_reader.py passed with zero findings after the WePOINTS logging refinement.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved source-only direct VCS dependencies out of published package metadata and into manual install documentation. Local package build and twine validation pass, and generated wheel/sdist metadata contains no direct git/URL Requires-Dist entries. Bandit was skipped because the touched files are package metadata and Markdown docs, not Python code.
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
