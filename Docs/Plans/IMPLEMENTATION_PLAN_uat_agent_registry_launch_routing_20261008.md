# C04 Agent Registry Launch routing

Associated task: TASK-13260.278.18.83.45. The parent recorded the actual WebUI failure and authorized this bounded correction before edits: enabled OpenCode Launch changed `/agents` to `/agents#/acp-playground?agent=opencode` while leaving Registry mounted.

## Design and ADR check

Use the existing `react-router-dom` navigation primitive in the shared AgentCard. The extension host supplies its hash router; the Next host supplies the existing router shim through `next.config.mjs`. Preserve the selected agent query and the existing disabled readiness gate. No new host detection or navigation abstraction.

ADR required: no. This fixes a caller that bypasses the existing WebUI/extension adapter convention documented in `apps/tldw-frontend/AGENTS.md`; it changes no durable routing, persistence, runner, auth, provider, or workspace rule. ADR-016 (`Docs/ADR/016-acp-session-and-orchestration-persistence.md`) remains unchanged. Production scope is only `AgentRegistry/index.tsx`; C03 RecipesTab is outside this unit.

## Stage 1: Establish the host failure
**Goal**: Trace Registry callers and both host adapters; add mounted boundary regressions before production changes.
**Success Criteria**: Web regression fails because Launch bypasses canonical navigation; real extension hash destination and unavailable-agent controls remain meaningful.
**Tests**: Real Registry/AntD/ACP transport with mocked external fetch; actual HashRouter destination; actual Next shim with only Next router boundary doubled. Retain all existing connection tests/assertions, adding their router context fixture only when needed.
**Status**: Complete

## Stage 2: Correct the shared caller
**Goal**: Replace the manual hash mutation with the existing router navigation call.
**Success Criteria**: Canonical Web destination renders with the selected agent; extension hash destination still renders; unavailable Launch cannot navigate.
**Tests**: New modules plus genuinely affected Registry connection scope under the normal UI and Web Vitest configurations. Node 20.19.5, 8192MB heap, 600s outer command bound and 300s case limit; at most three bounded causal/green experiments per issue.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Read every changed hunk/new body and hand the exact candidate to independent review.
**Success Criteria**: Scoped syntax and baseline/current ESLint delta verified; no new static findings; source/test hashes sent directly without proof files. Root owns global types/build and actual live Launch acceptance.
**Tests**: Configured touched-scope ESLint and TypeScript syntax diagnostics; Bandit not applicable to this TypeScript-only unit. Tests prove mounted host boundary behavior, not actual browser/runner/session execution.
**Status**: In Progress

## Attempt ledger

1. Unchanged-production causal run under the normal Web config: 2 expected Web Launch failures (OpenCode and Codex did not render the ACP destination), 4 passing extension/disabled controls, 0 skips; natural exit 1, 3.784s. No fixture failure. Pre-unit production and connection-test bytes are retained in memory only. Baseline scoped ESLint: zero source findings.
2. Minimal three-hunk production change: import/use `useNavigate`, replace the hash assignment with navigation to the same selected-agent query. Existing connection module changes only add MemoryRouter fixture context; all twelve original test bodies/assertions are byte-unchanged. Normal configured UI run: 15/15 (3 new hash-host cases + 12 affected existing), natural exit 0/4.208s. Normal configured Web run: 18/18 (the same 15 + 3 new Next shim cases), natural exit 0/2.571s. No skips or further replay.

## Static and review qualification

- Scoped production baseline ESLint: zero findings. Current production plus all three test files: zero errors/warnings. The same inherited config-level missing `apps/pages` notice appears in both; it is not a changed-code finding.
- Four current TS/TSX files transpile with zero syntax diagnostics. This is syntax qualification, not global typing/build.
- Bandit: not applicable; no Python changes.
- Self-review read all three production hunks, both complete new mounted modules, retained connection fixture delta, route wrapper, production Next alias/shim, extension hash host, and ACP destination query consumers. Exact in-memory comparison confirms no unrelated production edits and no original connection assertion changes.
- New tests retain real Registry, AntD, ACP REST client/transport and host navigation primitive. Connection config and external fetch are fixtures; the Web module doubles the Next router boundary and renders a destination query reader. They do not run a real Next server/browser, certify an agent, create a session, or prove the ACP creation modal preselects this query. The existing destination currently consumes session/view deep links; its creation form independently chooses a ready/default profile. This unit preserves the selected-agent URL contract and repairs Launch routing only.
- Source/tests are frozen for independent review. Root owns final global types/build/live Launch acceptance; those remain open.
