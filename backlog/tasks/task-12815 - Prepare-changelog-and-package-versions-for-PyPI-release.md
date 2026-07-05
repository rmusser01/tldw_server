---
id: TASK-12815
title: Prepare changelog and package versions for PyPI release
status: Done
labels:
- release
- packaging
- changelog
priority: High
modified_files:
- CHANGELOG.md
- pyproject.toml
- README.md
- Docs/mkdocs.yml
- tldw_Server_API/tests/Utils/test_release_helper.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update release changelog content and verify package version metadata is consistent for PyPI packaging/publishing while PR #1982 CI continues running.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CHANGELOG.md has a release-prep entry covering relevant PRs since the previous version push.
- [ ] #2 Python package version metadata is internally consistent for publishing.
- [ ] #3 Release packaging checks are run or documented if unavailable.
- [ ] #4 Changes are tracked in Backlog.md with verification notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current package/version sources and release changelog format.
2. Identify PRs/changes since the previous version push.
3. Update changelog and version metadata as needed.
4. Verify package metadata/build readiness and record results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared the release metadata for `tldw-server` 0.1.32. Promoted the changelog to a dated `0.1.32` entry, reset `Unreleased`, expanded rollup coverage from PR #2255 to PR #2544, and added grouped coverage for the 254 merged `dev` PRs from #2256 through #2544. Bumped the canonical package version in `pyproject.toml` to 0.1.32 and aligned README plus MkDocs visible release references. Updated the release-helper test so the remote-tag resume assertion derives the next patch version from `pyproject.toml` instead of hardcoding 0.1.32. Verification: pyproject metadata parse confirmed `tldw-server 0.1.32`; PyPI simple/index checks found no published `tldw-server` distribution/version collision; isolated sdist/wheel build succeeded after network escalation for build requirements; `twine check` passed; `Helper_Scripts/Packaging/check_pypi_artifacts.py` passed; built wheel metadata reports `Name: tldw-server` and `Version: 0.1.32`; release helper/docs tests passed with 52 passed and 4 warnings; `git diff --check` passed; Bandit on the touched Python test scope passed with 0 issues.
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
