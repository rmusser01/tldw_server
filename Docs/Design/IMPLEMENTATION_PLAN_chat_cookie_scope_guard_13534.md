# Canonical cookie owners in token-scope guards — TASK-13534

Parent: TASK-13458. Existing approved design: [Chat UX remediation](2026-10-04-chat-conversation-ux-remediation-design.md).

Native first-use evidence: a real cookie-session Chat saves and renames, while conversation search and header share-link reads return401. Identical requests with the disposable API key return200. The common required TokenScopeGuard checks only explicit JWT/API-key credentials in its missing-header branch. Reuse the existing canonical principal resolver at that boundary; retain the existing JWT/API-key restrictions and optional anonymous behavior. Do not patch each endpoint or add a credential registry.

ADR required: no. ADR-018/019/044 govern claim-based authentication and validated cookie-owner reuse; ADR-049/050 govern history lifetime. This restores those rules without changing scopes, quotas, session authority, or CSRF. ADR-059 governs task edits.

## Stage 1: Reproduce through real HTTP guards
**Goal**: Causal minted-session regression on actual Chat search/share routes and aliases.
**Success Criteria**: Cookie-only requests fail before the repair; no authentication dependency overrides; real isolated SQLite chat data.
**Tests**: Both search aliases and both share-link list aliases, plus missing/unusable/explicit-invalid/wrong-mode/store-failure controls.
**Status**: Complete

## Stage 2: Repair the shared boundary
**Goal**: Required scope guard resolves canonical owners before rejecting missing explicit credentials.
**Success Criteria**: Valid cookie owners succeed; rejected/revoked/expired sessions and invalid explicit headers remain rejected; existing scoped JWT/API-key quotas and constraints pass.
**Tests**: Cookie HTTP tests, scoped token and virtual-key suites, existing Chat search/share suites.
**Status**: Complete

## Stage 3: Verify native journeys and commit
**Goal**: The same native first-use session finds, reopens, and continues its saved Chat.
**Success Criteria**: Actual latest-dev API and WebUI, no browser interception; recorded server rows and screenshot; formatting/lint/Bandit and two existing reviewers clear.
**Tests**: Native search/reopen/second-turn/reload; static and touched-scope security checks.
**Status**: In Progress

The overall program stays In Progress; packaged extension, full interruption/transfer matrix, physical touch, VoiceOver and participant work are not closed by this repair.

Stage1 evidence: initial fixture signature error was corrected before diagnosis. Actual causal run /private/tmp/chat-cookie-scope-13534-red-causal.log: four HTTP401 versus expected200 failures, real minted sessions on both actual Chat aliases; no authentication override.

Stage2 evidence: 155 relevant backend cases pass with real minted-cookie route regressions and fail-closed controls; scoped JWT/API-key constraints and quotas stay intact. Owned formatting/AST/whitespace pass; Ruff zero additions; Bandit only 16 added LOW B101 pytest assertions, no production or non-assert additions. Two independent reviewers clear. Native ordinary and character greeting/two-turn/reload proofs recorded in TASK-13534. origin/dev advanced to2c5f19d032; integrate and verify affected code before retirement of this plan.
