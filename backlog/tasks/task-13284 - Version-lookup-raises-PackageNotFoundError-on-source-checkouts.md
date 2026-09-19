---
id: TASK-13284
title: Version lookup raises PackageNotFoundError on source checkouts
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
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
- [ ] #1 Running from a source checkout without the package installed builds the HTTP client successfully.
- [ ] #2 The version falls back to the value declared in pyproject.
- [ ] #3 A test covers the not-installed path.
- [ ] #4 No environment variable workaround is needed to make provider calls succeed.
<!-- AC:END -->
