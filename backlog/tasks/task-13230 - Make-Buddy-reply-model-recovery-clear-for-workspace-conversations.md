---
id: TASK-13230
title: Make Buddy reply model recovery clear for workspace conversations
status: Done
created_date: 2026-09-09 05:18
priority: high
references:
- TASK-13227
documentation:
- Docs/Reviews/2026-09-09-buddy-v1-qualification.md
assignee:
- '@codex'
updated_date: 2026-09-09 06:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-13227, a newly created workspace conversation generated successfully using the UI default model but its Buddy reply returned Choose a Chat provider and model before sending. Entering provider/model in collapsed Reply model settings recovered the reply. Clarify or repair the handoff of conversation model settings and preflight missing settings before Send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Buddy can reuse valid model/provider settings from a newly generated workspace conversation without requiring redundant manual identifiers.
- [x] #2 If no usable settings exist, required recovery controls and explanatory text are visible before sending and the user's draft is retained.
- [x] #3 Provider overrides remain explicit and cannot silently redirect the target conversation or cross server/account boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: repair existing Chat settings persistence and Buddy reply preflight; no new schema, credential store or provider authority. Trace actual Research Workspace generation through authoritative Chat creation/update and preserve the selected provider/model in the conversation's existing settings. Buddy preflight must use settings from the exact authorized conversation, show required model/provider recovery visibly when absent, retain drafts, and keep overrides explicit. Add failure-first contract regressions for persistence and scoped recovery, including target changes and unchanged account/server fences.
Expose an authenticated, attached-target-only reply-settings projection because the ordinary public Chat settings response intentionally omits roleplay resume state. Share its effective-completion resolver with Buddy turn acceptance; return only nullable provider/model, private/no-store. Keep the existing ADR-005 authority boundary.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Saved explicit provider/model from owned neutral workspaceChat with atomic merge; excludes Buddy overrides, omitted choices, global and tracked roleplay behavior. Added current-attachment-only reply-settings projection sharing acceptance resolver. UI shows effective settings, opens required fields beforeSend, keeps drafts and rejects unchecked/stale targets. Review fixed transport config race using pinned factory with deferred regression. Final backend58passed1Postgresunavailable skip; Persona compatibility12passed test-onlyCHAT_FORCE_MOCK; UI94passed; productionBandit0findings; focusedentrypointtypecheck0errors. Live explicit recovery completed exactturn156c528764bb46ffaa23d5a7f4e7c1cf; canonicalhandoff covered by realHTTP/SQLitetests, subsequentbrowserhandoff limited by disposableconnectionrecovery. ExistingADR005,user/API docs,sourceboundevidence updated. PR2934.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Owned neutral workspace Chat preserves explicit model selections; Buddy replies read the same effective settings through a pinned, authorized request. Missing fields are visible before Send, drafts remain intact, and temporary overrides preserve defaults. Final backend 58 passed/one unavailable-PostgreSQL skip, Persona compatibility 12 passed, UI 94 passed, production Bandit clean; independent review closed. Native/browser-handoff limits documented. ADR-005, API/user guide and evidence updated; PR #2934.
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
