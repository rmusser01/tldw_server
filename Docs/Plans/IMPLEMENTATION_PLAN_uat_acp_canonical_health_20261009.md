# C05 canonical ACP health implementation plan

**Associated task:** TASK-13260.278.18.83.45 (parent task is root-owned; its before-edit note authorizes this separate unit).
**Goal:** A successful canonical ACP health response exposes the existing Playground healthy state and prominent Create Session action.
**Approved design:** Normalize canonical overall `ok` to the existing client `healthy` state in normalizeACPHealthSnapshot. Preserve fallback precedence, other status strings, diagnostic fields, loading, canonical transport and modal agent/workspace checks. No runtime or configuration changes.
**ADR check:** ADR required: no. This repairs consumption of the existing ACPHealthResponse public contract without changing API, persistence, security or module rules; ADR-016 (Docs/ADR/016-acp-session-and-orchestration-persistence.md) remains unchanged.
**Scope:** apps/packages/ui/src/components/Option/ACPPlayground/ACPPlaygroundRecovery.tsx; new __tests__/ACPPlayground.health-availability.test.tsx; this plan only. AgentRegistry and RecipesTab remain separately owned.
**Qualification limits:** Node 20.19.5, existing Vitest 4.0.18, configured UI and Web, 8192 MB heap, outer cap 600 seconds, default case limits. At most three distinct causal attempts; setup failures are recorded separately. No Git, runtime, provider/model, private config/log/proof, install or parent task edits. Root owns combined types/build/live and independent review.

## Stage 1: Reproduce the actual consumer failure
**Goal:** Preserve pre-unit bytes and static baseline in memory; mount real ACP Playground/header/session panel/create modal with representative backend success and refusal/loading controls.
**Success Criteria:** Canonical overall ok reproduces the missing prominent Create Session action without mocking the health consumer or modal guards.
**Tests:** New mounted canonical ok success; healthy/degraded compatibility; unknown/malformed/unavailable/HTTP auth/errors/network failure; held health response; unavailable selected agent and missing workspace input.
**Status:** Complete

## Stage 2: Minimal shared normalization repair
**Goal:** Change only canonical overall ok to healthy at the existing normalization seam.
**Success Criteria:** Mounted causal case passes; all original status/fallback/modal guards remain intact.
**Tests:** New mounted cases plus genuinely connected ACP recovery/header/session/connection/readiness tests in both configured clients.
**Status:** Complete

## Stage 3: Static qualification, review and freeze
**Goal:** Complete source/caller/new-body self-review and scoped baseline/current ESLint and syntax validation, then report exact source/test hashes for root independent review.
**Success Criteria:** No new static findings; actual new and affected controls pass naturally under stated limits; frozen scope handed off with execution ledger and live limitation.
**Tests:** Configured UI/Web affected Vitest suites; configured scoped ESLint; TypeScript parser syntax check. Bandit is not applicable to TypeScript. Root performs combined types/build/live.
**Status:** In Progress

## Attempt ledger
Plan-wrapper setup: first Node heredoc had a template-string quoting syntax error before writing any file; excluded from causal/testing evidence. Replaced with a JSON string data literal. No qualification command executed yet. Pre-unit production bytes retained in functions memory before edits.

Stage 1: initial immediate missing-CTA run exited 1 with 1 failure/13 filtered at 1.33s, but lazy consumers were not settled; excluded as causal. With actual lazy consumers settled, exact backend-shaped success failed solely at missing prominent Create Session: 1 failure/13 filtered, natural 1, Vitest 1.64s/outer 1.834s. Production baseline hash ffa3cdb53dd600bd7e30134bd1300bd10d2a6eafb8a20fc53099934ba061a7b3 stayed unchanged until this repair. Real mounted header showed unhealthy and real Sessions consumer was present.
Static baseline setup: initial UI-local ESLint executable was absent; excluded. Correct apps-base installed Web ESLint on production baseline: natural 0, zero errors/warnings/fatals; inherited missing apps/pages advisory.

Stage 2 complete: one production line maps exact canonical overall ok to healthy after existing fallback precedence. No other production path changed. UI new module natural 0,14 passed/no skips,6.48s Vitest/6.698s outer; UI five connected existing modules natural 0,24 passed/no skips,4.36s/4.592s outer. UI coverage is 38 distinct cases over two green commands, not one combined 38-green command. Configured Web full six affected modules natural 0,38 passed/no skips,10.82s/11.105s outer. All used existing Node20.19.5/Vitest4.0.18/8192MB with600s outer cap and unchanged default case limits. No app/protocol guard or availability/modal mock. Real Antd/JSDOM emitted CSS stylesheet parse notices; retained as setup advisory, not hidden.
Stage 3 local qualification: baseline/current production ESLint0 errors/0 warnings/0 fatals; new test ESLint0 errors/0 warnings/0 fatals; inherited apps/pages advisory retained. Baseline/current TypeScript parser diagnostics0; test parser diagnostics0. Bandit N/A for TypeScript. Self-review read full production/new296-line test and all connected success/fallback/index/header/session-panel/modal/classifier/backend schema/endpoint/config contracts. Current source is exactly the retained baseline with the one approved line replacement. No new finding. No full types/build/browser/provider/live session acceptance claimed; root independent review, combined types/build/live remain open.
Frozen source SHA256 b461aceed1846d8407a296feb96d0c96b6c5c1ffd68c4fedf6123c7a68e764d9; new test SHA25637252da87ba6ee66bf03e21d904ea2d7e1865b0f5ff5f5275849536444758734. Baseline and final bytes are held in functions memory only, with no separate proof file/manifest. Production/test edits stop at this freeze while root reviews.

Exact existing affected module inventory (UI24; Web same plus new14): apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx9; ACPPlayground.connection.test.tsx3; ACPPlaygroundRecovery.test.tsx3; ACPPlaygroundHeader.test.tsx1; apps/packages/ui/src/services/acp/__tests__/readiness.test.ts8. These remain unchanged. No qualification rerun.
