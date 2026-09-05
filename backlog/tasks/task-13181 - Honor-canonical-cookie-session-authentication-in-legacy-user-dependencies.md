---
id: TASK-13181
title: Honor canonical cookie-session authentication in legacy user dependencies
status: To Do
assignee: []
created_date: '2026-09-05 15:53'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real quickstart Migu UAT mints a single-user cookie session (200) and loads users/me/profile (200), but notifications and ingestion capability requests return401 despite the browser sending its cookie. User_DB_Handling.get_request_user only accepts existing request principal or header credentials; it does not call the cookie-aware canonical resolver. Persona and DB dependencies use this path, so quickstart Persona remains unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Valid single-user cookie sessions can access authorized Persona, ingestion capabilities, and notifications through their actual dependency chains without API-key headers.
- [ ] #2 Expired, revoked, absent, or invalid sessions remain rejected and existing permission, user-isolation, CSRF, and origin checks remain effective.
- [ ] #3 Real quickstart Migu builder and Buddy image UAT passes with cookie authentication; browser stream UAT uses an explicitly allowed origin.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
