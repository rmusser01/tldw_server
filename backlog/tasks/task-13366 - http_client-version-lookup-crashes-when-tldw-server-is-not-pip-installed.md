---
id: TASK-13366
title: http_client version lookup crashes when tldw-server is not pip-installed
status: Done
assignee: []
created_date: '2026-09-23 21:06'
updated_date: '2026-09-23 22:56'
labels:
  - bug
  - http
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_get_project_version caught _HTTPCLIENT_NONCRITICAL_EXCEPTIONS but not ImportError; importlib.metadata.PackageNotFoundError is an ImportError, so an uninstalled source checkout raised on every default-header build instead of falling back to pyproject.toml. Surfaced by the TASK-13330 agent: 69 tests/http_client failures without TLDW_VERSION.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Uninstalled checkout resolves the version from pyproject.toml
- [x] #2 Regression test simulates PackageNotFoundError
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 7039432940: except now includes ImportError. tests/http_client/test_http_client_project_version.py red on HEAD (PackageNotFoundError), green now. tests/http_client without TLDW_VERSION: 69 failed on HEAD -> 1 failed (test_sensitive_log_filter_does_not_hide_concurrent_public_request, also failing on HEAD). Bandit: one-line except change, no new calls. Docs: none.

Duplicate of TASK-13284 (the review task for the same defect); both closed by 7039432940.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Source checkouts without pip install now get the pyproject version instead of crashing every HTTP default-header build.
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
