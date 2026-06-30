---
id: TASK-443
title: Patch Persona Agent PRD with reconciled completion scope
status: Done
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md
- https://github.com/rmusser01/tldw_server/issues/1902
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the original Persona Agent PRD so it reflects current implementation status, current Persona Garden/live-session completion scope, and future PRD tracks moved to #1902.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Original Persona PRD distinguishes current completion scope from future PRD tracks.
- [x] #2 Stale shipped-status claims are updated using current code evidence.
- [x] #3 Future-scope buckets are explicitly labeled as not current completion blockers and linked to #1902.
- [x] #4 Transcript export, Scopes/Policies editing, tool discovery, memory controls, visual scope, and security/reliability hardening are captured.
- [x] #5 No design-system backlog tasks are touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the stale original Persona PRD with the reconciled Persona Garden/live Persona completion contract from `Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md`.
- Verified issue #1902 still tracks all moved-out future PRD buckets: Persona-backed Chat Startup, Workspace Persona Defaults, Persona Scheduled Work, Persona Expressive Avatar Runtime, Personalization Memory Layer, Persona Tool Administration, and Persona Collaboration / Multi-agent Workflows.
- Kept this slice docs-only and did not touch design-system backlog tasks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Patched `Docs/Product/Persona_Agent_Design.md` so the original Persona PRD now distinguishes current Persona Garden/live-session completion scope from future PRD tracks.
- Updated the current-status section from stale scaffold-era claims to code-evidence-backed shipped/gap language.
- Linked future scope to #1902 and labeled each moved-out bucket as not a current completion blocker.
- Recorded verification with `git diff --check` and live `gh issue view 1902`.
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
