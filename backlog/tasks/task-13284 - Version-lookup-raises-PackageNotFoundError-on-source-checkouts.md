---
id: TASK-13284
title: Version lookup raises PackageNotFoundError on source checkouts
status: Done
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-23 22:56'
labels:
  - backend
  - http-client
  - dev-experience
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The HTTP client builds a user-agent by reading installed package metadata. When the package is not pip-installed, which is the normal state of a source checkout, that lookup raises PackageNotFoundError. That class inherits from ImportError, which is not in the non-critical exception tuple the call is wrapped in, so the exception escapes, the client never builds, and every outbound provider call fails as a 502. The pyproject fallback immediately below handles the miss correctly but is never reached. Reported in the 2026-09-15 audit and still present 356 commits later. Reproduced at runtime on this commit. During UX testing this made the interface report a provider failure when the provider was fine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Running from a source checkout without the package installed builds the HTTP client successfully.
- [x] #2 The version falls back to the value declared in pyproject.
- [x] #3 A test covers the not-installed path.
- [x] #4 No environment variable workaround is needed to make provider calls succeed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-23: fixed in 7039432940 (tracked there as TASK-13366, a duplicate filed before this task was noticed). _get_project_version now catches ImportError (PackageNotFoundError), falling back to pyproject.toml. test_http_client_project_version.py covers the not-installed path (red on the pre-fix code). tests/http_client without TLDW_VERSION: 69 failed -> 1 (pre-existing, unrelated).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Uninstalled source checkouts get the pyproject version; no TLDW_VERSION workaround needed.
<!-- SECTION:FINAL_SUMMARY:END -->
