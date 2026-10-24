---
id: TASK-13924
title: Redesign chat Context and Runtime rails
status: In Progress
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
2026-10-23 independent rebase verification: based on dev 9acce4297594a56c3a186793859121ca18be0b68; composer ancestry dropped. Kept native disclosure buttons/toggle-close/null panel behavior and ownership-aware current callbacks. Preserved planned/null-footprint composition semantics and selected-context language; prompt/assistant deduplication does not restore uninserted quick prompts. 214 tests in 13 suites passed, frontend TypeScript passed, scoped production ESLint 0 errors/17 existing warnings (same as dev). Five desktop/mobile light/dark browser cases passed, including scoped axe, focus/draft preservation and saved independent visibility. Restored contrast for preview status text. Temporary React test/preview wiring restored before commit; no dependency/config changes shipped. Bandit applicability run cannot parse TypeScript and gives no security clearance. Fresh pinned manual reviewer is next; Qodo has not been requested. ADR required: no; ADR-049 preserved.
2026-10-23 clean manual review of 848527d00cc4de9b891dd7f88bf7940e5a88983b found no important or critical issues. Reviewer requested one nonblocking cleanup: restore the unrelated composer_advancedPreferenceUnavailable locale entry removed by the old patch. Restored the upstream key and verified locale mirroring. Rebasing onto newer dev 8e8d9f2dcac12ef9743f9bead4a4a16a65ca2353 adds backend-only TTS changes; reviewed UI content is otherwise unchanged. Reviewer will confirm final pinned head.
Review clarification: composer_advancedPreferenceUnavailable was already present at line 4058 in reviewed head 848527d00cc4de9b891dd7f88bf7940e5a88983b. The apparent deletion was JSON key movement; setdefault made no locale change. Locale mirror verification passed all 3 cases. No important/critical findings remain from the fresh manual review; the final latest-dev head is being checked before force-with-lease publication.
2026-10-24 serial landing follow-up: the requester authorized updating, fixing CI and merging PRs #3309, #3310 and #3311 in order, and explicitly replied summary waived. Composer #3309 actually merged at 2026-10-24T16:38:06Z with merge commit 17141ff73c3caee7998d93a290598d6c3ab4ec0c, verified identical to dev. Rails #3310 is now next: remote head d2b474ff9954a47fa80ded19b0653a0c28268ac2 was verified OPEN/draft and auto-merge disabled; MERGE_QUEUE is unset. Reuse the isolated chat-ui-merge worktree on local codex/chat-rails-merge-20261024, preserve original worktrees/evidence, reconcile approved rail behavior with current dev/composer/message views, verify affected contracts and browser behavior, obtain a fresh independent manual review, then publish with an exact lease and use protected CI/merge. Sidebar remains paused. Plan: IMPLEMENTATION_PLAN_chat_ui_serial_merge_20261024.md. ADR required: no; approved presentation integration preserves existing ADR-049 ownership and APIs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Independent Context/Runtime rail redesign is rebased onto dev 8e8d9f2dcac12ef9743f9bead4a4a16a65ca2353 for PR #3310. Retains current ownership, planned-preview and native mobile disclosure contracts. Validation: 214 tests in 13 affected suites, frontend TypeScript, unchanged scoped lint warnings, and all 5 responsive offline browser/axe cases passed. Fresh clean manual review replaced Qodo and found no important or critical issues. PR remains draft and needs the requester's own Change summary before merge.
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
