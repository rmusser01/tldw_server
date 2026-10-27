# C04 — Carry the launched agent into Create ACP Session

Associated task: TASK-13260.278.18.83.45. Root recorded the distinct live failure and before-edit scope, then approved this single-modal design. Registry Launch reached `/acp-playground?agent=opencode`, but Create selected Goose; the actual modal was cancelled without a POST.

## Design and ADR check

ADR required: no. Reuse the existing WebUI/extension ACP search convention and structured catalog readiness. [ADR-016](../ADR/016-acp-session-and-orchestration-persistence.md) continues to govern session state; this change does not alter persistence, runner, ownership, workspace, permission or API contracts. The sole modal caller remains ACPSessionPanel. Capture a trimmed opaque agent query at each actual open using normal search or extension hash query. Apply it before catalog defaults; unknown or blocked requested agents stay refused with visible feedback. Existing ready-card manual selection wins over later catalog refresh. A URL change affects the next open, not an already edited form. No-query defaults and existing request fields stay unchanged.

## Stage 1: Source trace and causal mounted regression
**Goal**: Prove the query-to-form gap through real ACP parent, SessionPanel, creation modal, query hooks, stores and REST client.
**Success Criteria**: The real form submits Goose instead of requested OpenCode on unchanged source. Preserve pre-unit source and lint only in memory; no proof files.
**Tests**: Exact selected-agent POST and visible selection; normal/hash query parity; manual override/catalog refresh; absent/invalid/unavailable queries; reopen and deferred URL application; cwd/workspace/refusal controls.
**Status**: Complete

## Stage 2: Minimal modal correction and affected checks
**Goal**: Bind only initial agent selection and unavailable feedback to the captured query.
**Success Criteria**: New tests and genuinely affected existing modal/health/session tests pass in both normal configured clients; no assertion or readiness guard weakening. Node20.19.5, Vitest4.0.18, 8GB heap, 600s outer/300s case bounds. Maximum three issue attempts; preserve passes and distinguish fixture failures.
**Tests**: New ACPPlayground.launch-agent-selection.test.tsx plus existing modal prop guard, health-availability and SessionPanel modules. Scoped baseline/current ESLint and TypeScript syntax. Bandit N/A for TypeScript.
**Status**: Complete

## Stage 3: Source freeze and independent handoff
**Goal**: Review complete new bodies and connected lifecycle, then freeze exact scope for root review.
**Success Criteria**: No new static findings; root receives source/test hashes and full delta against retained in-memory bytes. Root owns final configured types/build/live query/form acceptance. No runtime/browser/API/provider/task/Git/install actions by this agent.
**Tests**: Independent root source review and later actual UI acceptance; do not claim live or all43 completion from transport doubles.
**Status**: In Progress

## Attempt ledger

1. Normal UI configuration on unchanged production: one selected causal case failed naturally (nine explicitly filtered), exit 1, Vitest 3.45s / outer 3.723s. Exact request retained cwd/name/workspace and sent `agent_type: goose` instead of requested `opencode`; no fixture failure. Production pre-unit bytes and scoped lint remain in memory. Minimal modal changes also reuse catalog readiness at form submission so Enter cannot dispatch an unavailable requested agent.
2. First affected UI green attempt: natural exit 1, 35 passing / six failing / 41 total, Vitest 22.15s / outer 22.466s. The exact launched request, trimmed hash query, ready manual override, absent/blank defaults, cwd and 29 existing controls passed. Five new checks observed title visibility before the real modal settled. The permission fixture returned a server session for a temporary local ID, creating an unrelated row. Correct only these fixture seams: await native dialog visibility and return the supported 404 for unknown IDs; retain all assertions and 35 passes.

Connected source finding, separately recorded before its repair: an explicitly chosen ready card could be replaced automatically if a later catalog refresh blocked it. Root confirmed this is within the original no-silent-fallback contract and approved a single real blocked-refetch causal control before the narrow card-click change. Test this distinct seam first, then retain its explicit intent in the existing requestedAgent ref. The final bounded qualification phase covers the six fixture failures, this new changed control and first Web scope; no unchanged UI passing controls are replayed.

Distinct manual blocked-refetch causal invocation: one selected new case failed naturally / 12 explicitly filtered, exit 1, Vitest 2.82s / outer 3.106s. After the actual query invalidation the manual selection had no required unavailable warning. This does not claim the UI had already completed its subsequent default effect or visibly switched agents; the fallback is source-backed. Preserve that failure and add the real blocked-card readiness precondition before green so the assertions require the refreshed catalog to be rendered. The narrow production repair sets requestedAgent.current to the exact chosen catalog ID in the existing ready-card click.

3. Final bounded qualification phase: corrected six UI failures plus the new blocked-refetch control passed naturally (seven selected / six explicitly filtered), exit 0, Vitest 10.60s / phase 10.821s. Retain the previous 35 passing cases: 42 distinct UI passes across runs, not one final full UI run. First full Web affected scope passed all 42 in four modules, no skips, exit 0, Vitest 21.70s / phase 21.982s. Combined declared phase outer 32.804s. The original query issue used three bounded phases; the separately authorized manual blocked-refetch issue used its causal red and this green, with no fourth unchanged replay.

## Frozen scope and review limits

One production file: ACPSessionCreateModal.tsx. One new test module: ACPPlayground.launch-agent-selection.test.tsx with 13 cases. Existing modal prop guard (six), canonical health availability (14) and SessionPanel (nine) remain byte-unchanged. TypeScript syntax passed for both changed TypeScript files. Scoped ESLint: zero errors; the source has one identical inherited exhaustive-deps warning (embedded target line moves with additions); the new test has no warnings. Bandit N/A: only TypeScript production/test changes.

Self-review read the complete new module and each production hunk against exact retained pre-unit memory bytes. Known query IDs are compared exactly after trimming; case variants/unknown values stay refused and are rendered as escaped React text. Normal and extension query modes share the existing precedence convention. Actual open resets capture intent, ready manual clicks replace that intent, and refetch cannot silently change it. Enter uses existing catalog readiness before local admission; cwd, name, tags, MCP, workspace, canonical connection/auth and backend request shape remain unchanged. Real parent, modal, Form, QueryClient, stores, readiness and REST client run in these tests; external configuration/storage/fetch/WebSocket and unrelated panels are doubles. Permission refusal coverage is client transport handling, not real backend authorization. Source/readiness behavior for removed catalog entries was inspected but not separately executed. Root owns independent review, final global types/build and actual Registry Launch/Create form acceptance; no live runner/session/all43 claim.
