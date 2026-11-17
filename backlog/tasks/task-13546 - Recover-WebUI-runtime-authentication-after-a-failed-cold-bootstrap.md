---
id: TASK-13546
title: Recover WebUI runtime authentication after a failed cold bootstrap
status: Done
labels:
- chat
- ux
- auth
- recovery
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A timed-out initial runtime configuration/session/profile bootstrap removes cookie configuration; account verification retries never repeat that bootstrap. Repair the shared recovery path on the accepted chat UX scope while preserving manual credentials, logout boundaries, current account authority and bounded requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failed initial WebUI bootstrap can be retried on the same page and establishes configuration only after runtime config, session and profile verification succeed.
- [x] #2 Concurrent retries share one bounded bootstrap and failed or stale attempts never publish cookie configuration across logout or a newer credential/configuration choice.
- [x] #3 Actual account/client retry callers recover through the shared path without per-button workarounds or reload requirements, with causal regressions and both client type gates.
- [x] #4 Native cold startup and preserved saved conversation evidence are recorded; independent review, static checks and explicit remaining limits are current.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Latest origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f is included in f4c6f60018. Searched existing runtime bootstrap tasks; previous fixes are Done and do not own same-module recovery. Remote/open max task13530, this worktree max13545. ADR required:no; existing cookie bootstrap and owner/request authority contracts remain unchanged, ADR049/050 protect chat selection and ADR059 controls task editing. New work is bounded transient recovery of existing bootstrap, not new auth transport.
Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_runtime_bootstrap_recovery_13546.md. Source root confirmed independently: bounded bootstrap settles without configuration, account Retry only initializes/profile-checks. Implement existing shared recovery seam with explicit logout/manual lifetime retirement; no timeout increase or per-button patch.
Three causal RED tests demonstrate null configuration after config/session/profile cold timeouts on the same module (chat-runtime-bootstrap-13546-red.log). Shared recovery now passes37 bootstrap cases including real useChatDraftOwner Retry preserving exact text, simultaneous initializer one session, logout/manual/foreign ABA and storage failure. Stage1/2 complete; review and final gates underway. Eligible failed bootstrap with no manual config only; no new auth event that would erase an unowned draft.
Native one-shot fault proxy attempt excluded: runtime-config returned available:false because its upstream port changed the default forwarded metadata. Source-confirmed sibling defect remains: connection.checkOnce normalizes null configuration through updateConfig and can retire failed runtime recovery. Adding a real bootstrap plus connection store regression before the shared guard.
Real bootstrap+connection regression RED: checkOnce persisted {serverUrl:currentOrigin}; smallest shared cfg guard GREEN (43 bootstrap cases). Upgrade recovery will accept only absent/JSON-null or inert exact-origin URL/cookie hints without any credential, token, manual metadata, foreign URL, or unknown fields; live session/profile/storage ACK and explicit authority/account lifetime fences remain required.
Final self-review found a needless eager facade dependency in WebUI bootstrap. Move the unchanged bounded recovery registration into a dependency-free focused module beside existing runtime-auth helpers; the facade retains client/lifecycle trigger ownership. This preserves the existing lazy-loading contract documented by connection.tsx without an auth/API/persistence change. No new ADR or generic framework; verify module and final bundle before final commit.
TASK13546 repairs failed same-page runtime auth recovery and the connection initializer URL-only write. A dependency-free shared recovery registration preserves existing lazy facade loading; direct bundle inspection confirms the API facade class leaves _app. Live capability/session/profile and cookie storage ACK remain required. Exact-origin inert legacy hints/JSON null can recover; manual/unknown/foreign authority is refused. Generation, account, initial/late ACK and locked scrub fences protect logout/new credentials. Final578/19 affected tests, both types0,6 source lint530/530/0 added,7 owned regions+1 whole module format and diff pass; two final independent source reviewers clear. Actual full-page held503→Retry preserves exact multiline/trailing-space draft without reload, removes Retry, enables Recover/Send and restores Online/Healthy. Corrected loopback fault fixture is stopped; final direct rebuilt page restores Mira3. All5 saved4/5/4/5/3 hashes match the original metadata-false messages serialization. Earlier mismatched proxy policy, stale bundle, encoding, temporary formatter and TS narrowing failures are excluded from passing evidence. TS-only authored scope: Bandit N/A. Current dev2c5f19d0328360d45bce99b8bae75ce252cc312f freshly fetched and included. Exact artifacts and limitations are recorded in the master design. Only this bounded task/owned plan completes; Stage5/all-ten and TASK13540 native discard AC remain In Progress. Next: compact title, ordinary scoped settings ACK/partial Retry/editor commitment, truthful planned/inherited lore, D7 client adoption and guarded server prerequisites; packaged/physical/AT/zoom/participant and complete native error journeys remain explicit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Failed WebUI bootstrap can recover on the same page through existing account/client callers, using shared bounded runtime verification and acknowledged cookie publication. Removed connection self-poisoning and protected current authority through initial, deferred storage and credential-lock races. Recovery registration stays outside the eagerly loaded API facade. Causal and full-page evidence, current final checks and precise remaining program limits are recorded in Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. All four bounded stages complete; only the owned temporary plan is retired. No PR, push or merge.
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
