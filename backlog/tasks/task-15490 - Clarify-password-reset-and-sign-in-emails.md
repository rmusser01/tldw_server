---
id: TASK-15490
title: Clarify password reset and sign-in emails
status: In Progress
labels:
- auth
- email
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give password reset and sign-in emails one clear browser action and a plain-text link. Preserve configured destinations and the copyable token for extension users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Password reset emails have a Choose a new password action in HTML and a working plain-text link.
- [x] #2 Sign-in emails have a Sign in action in HTML, a plain-text link, and a copyable extension token in both formats.
- [x] #3 Configured destinations, existing defaults, and mock token redaction continue to work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Template-only change using the existing mailer and link resolver. No architectural decision is required.
Both updated rendering checks failed on the original copy and passed with the templates. All 21 email service and URL tests passed, including mock token redaction. A disposable loopback SMTP receiver captured both multipart emails with configured URLs and the extension fallback. Ruff passed and Bandit reported zero production findings. External mailbox delivery and navigation through a real self-hosted frontend were outside the local check.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shared the layout and plain-text structure for password reset and sign-in. Kept the existing URL resolver and extension token fallback. Verified rendered content and local SMTP delivery.
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
