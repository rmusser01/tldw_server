# Credit Workstreams — Coordination Index (2026-10-01)

Umbrella index for the four credit-funded execution plans drafted 2026-10-01. Context: two existing workstreams (console UX, performance, 60+ PRs) own `apps/**` changes; these batches are scoped to stay out of their way.

| # | Plan | Scope flavor | Size | Backlog coverage |
|---|------|--------------|------|------------------|
| 1 | [Due-debt sweep](2026-10-01-due-debt-sweep-implementation-plan.md) | Backend cleanup, security smalls, test debt | ~1 week | Mostly new tasks needed (titles proposed per stage); TASK-12113, TASK-13100 exist |
| 2 | [MCP agentic tools program](2026-10-01-mcp-agentic-tools-program-implementation-plan.md) | Greenfield backend features | Multi-week program | TASK-2281…2294, 12118, 12119, 2341 all exist |
| 3 | [DB parity & structural debt](2026-10-01-db-parity-structural-debt-implementation-plan.md) | Data layer correctness | ~1-2 weeks | New tasks needed (titles proposed per stage) |
| 4 | [UAT/E2E verification hardening](2026-10-01-uat-e2e-verification-hardening-implementation-plan.md) | Test infrastructure | ~1 week | TASK-13260.278.2-.5 exist; ratchet task to check/create |

## Recommended sequencing for credit spend

1. **Start Batch 4 Stage 1 (UAT392 collection fix) first** — until the Playwright collection actually runs cases, every other E2E claim is unverifiable, and it directly protects the 60-PR wave.
2. **Batch 1 stages run in parallel** (independent PRs; ideal subagent fan-out, one worktree per stage).
3. **Batch 2 Wave 1 (`web_fetch`/`web_search`) as the flagship slot** — read-only, highest-vision alignment. Coordinate with TASK-13139.3 in the design doc to avoid double-building the bounded reading view.
4. **Batch 3 Stage 1 (parity audit) is safe to start anytime** — it's read-only and gates the rest of Batch 3.
5. Sequence **TASK-13100 (cookie exposure)** from Batch 1 Stage 5 *before* any authenticated-retrieval work in the research program.

## Collision rules

- No batch modifies `apps/**` product code; Batch 4 touches `apps/**` **test files only**, with a `git log` freshness check per file against in-flight PRs first.
- Backend edits in Batches 1-3 should avoid files currently modified in the working tree (check `git status` per stage start).
- One Backlog task per stage before edits (repo rule §0).

## Known blocker (2026-10-01)

`backlog task create` crashes in CLI v1.44.0 with "Maximum call stack size exceeded" (4 attempts: full flags / simple / title-only / raised stack limit). `backlog search`, `task edit`, and `task list` work. Until fixed: retry `task create` once per session; if it still fails, get the user's explicit approval before creating task files manually (repo rule). Proposed umbrella task text is in the conversation record for retro-creation:

> Title: "Draft staged execution plans for credit-funded work batches 1-4" — Done once plans verified; links: the four plan files above.
