---
id: TASK-13924
title: Redesign chat Context and Runtime rails
status: Done
labels:
- chat
- webui
- design
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep two independent /chat rails while clarifying their jobs and replacing the repeated nested-card summaries with calm task-oriented sections. User confirmed the two-rail direction after the separate composer redesign.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Context groups prompt and assistant identity, attached sources with existing open/remove/clear actions, conversation persistence, and an on-demand full request preview.
- [x] #2 Runtime brings status and stop/regenerate recovery together, shows provider/model once, and keeps scoped settings plus MCP controls accessible.
- [x] #3 Both rails retain independent persisted visibility, focus mode, keyboard access and mobile Context/Runtime access without changing composer behavior.
- [x] #4 Use existing theme tokens and flatter sections; suppress repeated idle/empty/default status decoration while preserving warnings and unavailable reasons.
- [x] #5 Focused behavior/integration tests, TypeScript, scoped lint and desktop/mobile light/dark browser/a11y checks pass; record review and security validation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User confirmed two separate rails, simplified and clarified, after the separately approved composer work. Branch codex/chat-rails-calm reuses the attached managed worktree and starts at composer commit 27527c937b. Search found older completed rail IA tasks (TASK-393/TASK-516), with no duplicate open redesign task.

Design: Docs/Design/chat-rails-calm.md. ADR required: no; this moves shared UI presentation and existing callbacks, without changing API, provider, auth, storage or ownership rules. Existing ADR-049 governs history ownership. The completed temporary implementation plan was removed per repository instructions.

Context now groups prompt and assistant identity, one source inventory with individual/bulk actions, conversation persistence and a disclosed request preview. Runtime groups run status/recovery and Stop/Regenerate, provider/model settings with disclosed defaults, and MCP controls. Flat sections replace repeated nested cards and idle badges. Mobile uses stable Context/Runtime tabs with one close-panel control; explicit null dismisses only the mobile panel, returns tab focus and preserves both saved rail flags. Existing draft, focus mode, readiness, warnings, scoped settings and assistant/source actions remain intact.

Verification: failing-first rail hierarchy and mobile dismissal tests were verified red then green. Final focused run: 128 passing tests, with 2 unchanged baseline failures for chat-placeholder fixtures. A final assistant/action recheck reported 32 passing and the same 2 failures. An unchanged-HEAD overlay run confirmed 49 existing failures across the wider 3-file integration group, involving stale effectiveAssistantState fixtures and mixed React dependencies. TypeScript noEmit exited 0. Scoped ESLint over 19 files found no new findings after comparing inherited warnings without shifted line numbers. Playwright passed all 5 cases: desktop 1440x1000 and mobile 390x844, light/dark, rail-scoped axe with zero violations, no horizontal viewport overflow, model-settings dialog, draft preservation, mobile close focus and independent visibility after reload. Six validated captures and their manifest are in .impeccable/review/rails/. The one-time Impeccable detector returned zero findings.

Fresh bounded code review: ship. Fresh visual review: ship for the supplied offline capture scope, with no material fixes; fallback reviewer instructions were used. The fallback documenter checked incumbent code/tokens and preserved global PRODUCT/DESIGN files and sidecar, recording the final evidence in the surface brief. Pre-existing context-loader scope/schema drift was reported and left outside this change.

Security/limits: Bandit was attempted against the touched TypeScript scope, producing 10 parse errors because it is a Python analyzer; no Python changed, so this records inapplicability rather than a successful security scan. Scoped lint and code review checked the changed UI. Live providers, populated-source visual states, microphone use and production bundles were not checked. Temporary React aliases needed by the local mixed dependency installation were removed and the exact original Next config restored. No new dependencies or global design-token changes.
Published as draft PR https://github.com/rmusser01/tldw_server/pull/3310 on clean branch codex/chat-rails-pr, stacked on composer PR #3309 (codex/chat-composer-pr). Only the reviewed rail commit and task documentation are included in this comparison; unrelated RPG ancestry is excluded. Fresh isolated checkout: 102 focused tests across eleven files and TypeScript noEmit passed. Previously documented integration baseline failures and live-provider/production validation limits remain. Merge composer first, then retarget rails to dev; the requester-authored Change summary remains required before merge. Original UI branches and active header workspace were left unchanged.
2026-10-23 follow-up: user requested both PRs rebased directly onto latest dev and fresh clean manual subagent reviews in place of Qodo. Move rails PR #3310 off its composer stack onto dev, preserving current dev behavior and the accepted rail presentation. Verify scoped regressions, typecheck and browser behavior, then publish with force-with-lease. ADR required: no; existing ownership rules remain unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Redesigned Context and Runtime as two calmer, independently controlled rails. Context owns reply inputs and sources; Runtime owns execution, model settings and MCP. Preserved existing actions and state gates, verified desktop/mobile light/dark behavior, resolved review findings, and recorded baseline test/environment limits. Design and validation: Docs/Design/chat-rails-calm.md.
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
