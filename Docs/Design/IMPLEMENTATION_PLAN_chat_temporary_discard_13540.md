# Temporary conversation reset and selection — TASK-13540

Implements the accepted keep-or-discard rule in `2026-10-04-chat-conversation-ux-remediation-design.md`; on-device lifetime follows `2026-10-04-d7-conversation-model.md`. Compact tab deletion/bulk outcomes are the separate TASK-13541.

ADR check: no new durable rule; keep ADR-049/050 ownership and lifetime, ADR-059 tracking. No new persistence, dependency, async transition controller or native-character preflight relaxation. Reuse native synchronous confirmation used by existing settings/route guards; Cancel explicitly keeps the conversation, OK explicitly discards it. Do not claim DB export supports Temporary transcripts.

## Stage 1: Prove actual discard boundaries
**Goal**: Trace all reset and selection callers and write causal refusal/acceptance checks.
**Success Criteria**: Tests distinguish populated Temporary source loss from ordinary/empty transitions and retained compact tab switches.
**Tests**: Shared clear cancellation, saved selection cancellation, on-device prepared selection cancellation; existing source/settings/account fences.
**Status**: Complete

## Stage 2: Protect shared interactive transitions
**Goal**: Add one reusable native Temporary-content choice and honor refusal at reset/selection continuations.
**Success Criteria**: No source/view/draft/title/identity mutation after Cancel; explicit actions do not stack relevant confirmations; System Reset admission precedes deletion.
**Tests**: Causal checks green; confirmed/empty/saved positive controls.
**Status**: Complete

## Stage 3: Verify real journeys and client contracts
**Goal**: Run relevant neighbor suites, both client type gates, scoped static checks and native owned disposable Temporary Cancel/Discard flows.
**Success Criteria**: Native source and next draft survive Cancel/reload; explicit discard removes transcript; saved fixtures stay unchanged.
**Tests**: Real store/hooks and caller controls, actual browser UI, server fixture receipts, scoped lint/format. Bandit is N/A unless Python repository source is authored.
**Status**: In Progress

## Stage 4: Independent review and closeout
**Goal**: Reuse the two authorized reviewers, repair real findings, update parent gates and commit this bounded unit.
**Success Criteria**: Reviewed working commit; tracking and limitations recorded; entire NN/G program stays In Progress.
**Tests**: Final appropriate gates after any review repair, diff/whitespace review.
**Status**: In Progress

Review repairs: retain session and H1 selection ownership through held confirmations; reuse Form input/connection preparation guards; pre-admit linked Notes before identity writes; capture a requested normal linked-open mode before its saved load intent. Ordinary Temporary saved-history preview mode remains unchanged. Legacy sidebar cleanup qualifies live IDs.

Native evidence: real declined New preserved exact two-message transcript and unsent text (including trailing spaces) across reload. IAB exposed no active JS confirmation for positive acceptance; a saved-selection click subsequently timed out. Keep acceptance/packaged-browser validation open; do not infer it from unit controls. Both owned saved fixtures retained their exact hashes and Temporary content was absent from server history. Evidence lives under `/private/tmp/chat-native-temporary-*-13540-20261007.*`.
