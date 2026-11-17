# MCP Agentic Tools Program Implementation Plan (Credit Batch 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MCP agentic tool family (largest untouched backend backlog: TASK-2281…2294, 12119, 2341) in capability-gated waves, read-only tools first.

**Architecture:** Each tool ships as an MCP module under `tldw_Server_API/app/core/MCP_unified/modules/`, registered through the existing `catalog_loader.py` / `module_surface.py` / `tool_execution/` machinery with JWT/RBAC capability gates from `security/` and `governance_packs/`. Every wave = design doc → TDD slices → release note. Waves are ordered so no wave depends on an unshipped predecessor.

**Tech Stack:** FastAPI, MCP Unified (`app/core/MCP_unified/`), pytest (`tldw_Server_API/tests/MCP_unified/` + in-module `tests/`), Backlog.md.

**Spec:** Backlog tasks `task-2281` … `task-2294*`, `task-12118`, `task-12119`, `task-2341` under `backlog/tasks/`; `Docs/Design/MCP_External_Server_Federation_Design.md` (draft, no owner — evaluate during Wave 5); this plan.

## Global Constraints

- `source .venv/bin/activate` before `python`/`pytest`.
- **Design first:** every wave opens with a design doc in `Docs/Design/` before code (repo rule).
- Every new tool must be capability-gated (RBAC), observable via `tool_observability.py`, and covered by tests in `tldw_Server_API/tests/MCP_unified/`.
- Existing backlog tasks already cover each tool — update them (status/notes/verification) instead of creating duplicates. `backlog task edit` worked when `task create` was crashing; verify.
- Template rule: before writing a new module, read the smallest existing module in `app/core/MCP_unified/modules/` and copy its structure, registration, and test layout.
- Backend scope; WebUI surfaces for new tools are out of scope for this program (console-UX workstream owns `apps/**`).
- Bandit per wave on touched paths.

---

## Stage 0: Baseline and 0.2.0 release (TASK-12118)

**Goal:** Establish a green baseline and finish the MCP Unified 0.2.0 package release.
**Success Criteria:** `python -m pytest tldw_Server_API/tests/MCP_unified tldw_Server_API/tests/MCP tldw_Server_API/tests/MCP_Hub -v` green; 0.2.0 released per TASK-12118 AC.
**Tests:** Existing MCP suites.
**Status:** Not Started

- [ ] Run the three MCP test suites; fix or quarantine failures before any new work (never disable tests).
- [ ] Complete TASK-12118 release steps from its task file; record evidence.
- [ ] Read `app/core/MCP_unified/README.md`, `module_surface.py`, `catalog_loader.py`, and one existing module; note the registration pattern in the Wave-1 design doc.

## Stage 1 (Wave 1): Read-only research tools — `web_fetch` + `web_search` (TASK-2291, TASK-2292)

**Goal:** Ship MCP tools exposing the existing research pipeline: `web_fetch` (bounded fetch with domain policy controls) and `web_search` (multi-provider search over tldw research providers).
**Success Criteria:** Both tools callable via MCP with JWT auth + capability gate; deny-by-default domain policy on `web_fetch`; provider errors surfaced as structured tool errors; tests cover allow/deny, authz, and happy paths.
**Tests:** New `tldw_Server_API/tests/MCP_unified/` test module per tool (authz denial, policy denial, happy path mocked against the research services).
**Status:** Not Started

Contract (freeze in the design doc):
- `web_fetch(url: str, max_bytes: int = 512_000, mode: "text"|"html" = "text")` → `{status, content_type, content, final_url, fetch_reason}`; domain allow/deny list from governance config.
- `web_search(query: str, providers: list[str] | None, max_results: int = 10)` → `{results: [{title, url, snippet, provider, rank}], fused_ranking_version}`.

- [ ] **Step 1:** Design doc `Docs/Design/2026-10-XX-mcp-web-fetch-web-search-design.md`: map to existing research services (`app/core/Search_and_Research/`, `Web_Scraping/`), define capability names (e.g. `mcp.tools.web_fetch`), policy source, and error envelope. Link TASK-2291/2292.
- [ ] **Step 2:** Write failing tests first (authz denial without capability; `web_fetch` deny on non-allowlisted domain; happy paths with mocked services).
- [ ] **Step 3:** Implement modules following the smallest-existing-module template; register via catalog; wire observability.
- [ ] **Step 4:** Integration test through the MCP endpoint (`GET /api/v1/mcp/status` shows the new tools; execute via tool-execution endpoint per existing tests).
- [ ] **Step 5:** Bandit; update TASK-2291/2292; commit per-tool (`feat(mcp): add web_fetch tool (TASK-2291)`).

Note: TASK-13139.3 (bounded reading views in MCP web.fetch, Credit Batch 2 sibling program) overlaps — coordinate in the design doc rather than building twice.

## Stage 2 (Wave 2): LSP-backed code-intelligence tools (TASK-2281)

**Goal:** MCP tools `code_definitions`, `code_references`, `code_hover`, `code_diagnostics` backed by an LSP server.
**Success Criteria:** Tools work against a workspace path with capability gating; LSP subprocess lifecycle managed (start/stop, timeout); tests use a tiny fixture project, no network.
**Tests:** Unit tests with a fixture workspace; lifecycle tests (spawn, timeout, shutdown).
**Status:** Not Started

- [ ] Design doc incl. LSP server choice (bundled vs system), sandboxing constraints (governance packs), and per-workspace scoping.
- [ ] Failing tests → implement → integration via MCP execute endpoint; update TASK-2281; commit.

## Stage 3 (Wave 3): Agentic task, scheduling, and notification tools (TASK-2285, 2286, 2287)

**Goal:** Task/checklist management, scheduling/wakeup, and notification tools for long-running agent sessions.
**Success Criteria:** Tasks persist (Jobs backend per the repo's Scheduler-vs-Jobs guide: user-facing → **Jobs**); wakeup schedules survive restart; notifications dispatch through existing channels; all capability-gated.
**Tests:** Persistence round-trips; restart-safe wakeup test; notification dispatch mocked.
**Status:** Not Started

- [ ] Design doc: choose Jobs vs Scheduler per repo decision guide (AGENTS.md "Scheduler vs Jobs"); define tool schemas and RBAC capabilities.
- [ ] TDD per tool; update the three tasks; commit per tool.

## Stage 4 (Wave 4): Subagent orchestration + plan-mode/worktree sessions (TASK-2288, 2289)

**Goal:** Effectful orchestration: subagent/agent-team execution tools and plan-mode/worktree session tools.
**Success Criteria:** Nested MCP execution gated per TASK-2294.4/.5 sequencing (read-only before effectful); sandbox containment proven by tests; session isolation via worktrees.
**Tests:** Containment tests (hostile tool cannot escape sandbox workspace); session lifecycle tests.
**Status:** Not Started

- [ ] Depends on Waves 1-3 patterns. Design doc must address the TASK-13130-13133 scheduled-execution security findings (identity, attestation) rather than rediscovering them.
- [ ] TDD; update tasks; commit.

## Stage 5 (Wave 5): Skills/workflow runners, `rag.*` module, prompt registry (TASK-2294 family, 12119, 2341)

**Goal:** Reusable agentic routines: skill & workflow runner tools (incl. capability-gated model completion adapter TASK-2294.3.2, strict snapshot loader 2294.3.3, nested execution 2294.4/.5), a `rag.*` MCP module, and a shared prompt registry.
**Success Criteria:** Skills execute from versioned snapshots with capability gates; `rag.*` module plan (TASK-12119) converted into shipped design + first endpoints; prompt registry service design reviewed.
**Tests:** Snapshot loader strictness tests; capability-gate tests; registry CRUD tests.
**Status:** Not Started

- [ ] Convert TASK-12119's plan into `Docs/Design/2026-10-XX-mcp-rag-module-design.md`; sequence `rag.search`/`rag.fetch` first.
- [ ] Evaluate `Docs/Design/MCP_External_Server_Federation_Design.md` (ownerless since 2026-02): adopt, revise, or archive with a decision note in the task.
- [ ] TDD slices per tool; update tasks; commit.

---

## Other flagship options (not planned here — see coordination index)

Web-research quality waves (TASK-13139.3-.5, roadmap `Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md`), scholarly source expansion (TASK-12968 family, ledger `Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json`), Personal Context lifecycle (TASK-13162/13164/13165, roadmap `Docs/Design/2026-09-13-second-brain-parity-roadmap.md`), Docker distribution WP1 (TASK-13343 AC2-4, review `Docs/Design/2026-09-20-complete-app-distribution-review.md`).
