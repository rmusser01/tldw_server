# Character Library entry remediation — TASK-13504

Approved program: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Baseline f225562fc26 integrated153bc4e00d; compact checkpoint7f68a6633c.

## Stage 1: Pure Library entry intent
**Goal**: Current/new-tab Library actions pass the character route intent without changing source conversation or preferences; dismissing readiness stays local.
**Success Criteria**: Blocked model, cancel, accepted navigation and popup denial leave source identity, typed draft, H1/Core/Session and response settings unchanged.
**Tests**: Meaningful red-first mounted Library regressions for both entries and blocker cancel.
**Status**: Complete

## Stage 2: Guarded fresh destination acceptance
**Goal**: Resolve the owned card and delegate to the existing full Form native apply boundary with explicit fresh semantics.
**Success Criteria**: No destructive work before scoped card, shared preflight, exact draft retention, navigation and owned identity ACK; held work retires on newer route/view/account/input/settings changes. A missing card has Retry without a synthetic selection. Explicit fresh entry bypasses old automatic bookmarks.
**Tests**: Actual root/H1 routing plus actual Form/composer acceptance regressions for missing/held/failed/declined and supported fresh success, including same character and fresh source.
**Status**: Complete

## Stage 3: Review and checkpoint
**Goal**: Both clients remain compatible with latest dev and the bounded change is reviewable.
**Success Criteria**: Relevant suites and both client types pass; touched lint has no new diagnostics; two authorized independent reviews have no unresolved Important/Critical issue; limits recorded.
**Tests**: Relevant chat/catalog/owner/route/draft suites, WebUI types, extension compile, diff check and locale generator if keys change. Bandit N/A for TS/docs unless Python scope changes.
**Status**: In Progress

Progress: Library entry red/green seven regressions; targeted lifecycle green 24 tests across root route and actual Form/composer (actual draft hook over acknowledged Map IO). Reuse native destination acceptance; both independent reviews and broader coordinator/client/locales checks pending. No native popup, cold-tab private storage, focus, packaged-extension or live provider proof from these tests.

Stage 2 evidence: meaningful saved-source URL and held/automatic/transient-cancellation queue regressions red then green. Existing action revision retires held queue preparation permanently when setup blocking begins; pending or accepted replacements retain source queue items for explicit review. Actual root/H1/Core coordinator suite passed 75 tests; its destination Form remains an explicit stub. Full actual Form/composer/persistence coverage, both client gates, lint and latest-dev integration are the Stage 3 checks. Refreshed remote dev is 7226596cca; fetched, integration pending.
