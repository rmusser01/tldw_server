---
id: TASK-12071
title: Move direct VCS extras to manual install docs
status: Done
assignee: []
created_date: '2026-06-29 19:53'
updated_date: '2026-06-30 02:08'
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

Additional PR review follow-up: removed the backend-named extras ocr_dots, ocr_points_transformers, and TTS_chatterbox from published package metadata and from the aggregate all extra so manual-only backends no longer look like successful extras installs. Updated OCR/POINTS/Chatterbox docs, including published WebUI docs, to use explicit manual install steps. Pinned the Perth manual install to the previously used commit ce86c49d029f42272c1902eccb675556b9ed2330. Scoped ImportError/ModuleNotFoundError handling to the POINTS transformers path with _POINTS_TRANSFORMERS_EXCEPTIONS and added a regression test ensuring explicit SGLang import errors do not fall through to transformers. Verification: POINTS backend tests passed (5 passed); pyproject parse/direct URL scan passed; package build --no-isolation succeeded; twine check passed; wheel metadata has direct_url_requires=0 and extras_removed_present=[]; Bandit on points_reader.py passed with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved source-only direct VCS dependencies out of published package metadata and into manual install documentation. Removed backend-named extras that would otherwise install successfully without the manual backend, kept the SGLang client and Chatterbox language preprocessing extras, and validated generated wheel metadata has no direct URL requirements or removed extras. Runtime handling now requires WePOINTS for POINTS transformers mode and scopes import-error handling to transformers so SGLang errors are not masked. Local package build, twine validation, focused POINTS tests, and Bandit on touched Python passed.
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
