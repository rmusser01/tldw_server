---
id: TASK-12165
title: Design Ergo IRC community migration
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-06 17:47
labels:
- docs
- community
- infra
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-06-ergo-irc-community-migration-design.md
modified_files:
- Docs/superpowers/specs/2026-07-06-ergo-irc-community-migration-design.md
- backlog/tasks/task-12165 - Design-Ergo-IRC-community-migration.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a reviewed design spec for migrating tldw community communications from Discord-first to an Ergo IRC-first stack with Kiwi IRC, Matterbridge compatibility, and a public support archive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design captures selected launch architecture, moderation, bridge, history, archive, backup, and milestone decisions.
- [x] #2 Design calls out privacy, Discord API policy, public archive consent, redaction, noindex launch posture, bridge flood control, and IRC-first positioning.
- [x] #3 Spec is saved under Docs/superpowers/specs and linked from this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec drafted and review-hardened with the selected IRC-first Ergo architecture, Matterbridge compatibility, public #support archive, Discord policy considerations, retention/redaction controls, bridge flood controls, restricted #announcements write access, Discord-origin archive tagging, deploy-time domain placeholders, default archive route, and launch checks. Spec reviewer pass 1 found announcement write-access and archive origin-label gaps; both were patched. Spec reviewer pass 2 approved with no blocking issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the reviewed design spec for migrating tldw community communications from Discord-first to an Ergo IRC-first stack. The spec covers Ergo, Kiwi IRC, Caddy, Matterbridge, public #support history with 365-day retention, abuse controls, Discord policy constraints, redaction/backup operations, launch checks, and milestone 2 email verification. Verification: two spec-review subagent passes; the second approved with no issues. Tests/Bandit were not run because this was documentation and Backlog metadata only.
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
