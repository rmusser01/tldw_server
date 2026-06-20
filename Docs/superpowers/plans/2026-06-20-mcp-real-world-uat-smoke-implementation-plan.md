# MCP Real-World UAT Smoke Scenario Implementation Plan

> **For agentic workers:** Use `superpowers:test-driven-development` for
> production behavior changes and `superpowers:verification-before-completion`
> before claiming completion.

**Goal:** Add a `real-world` MCP smoke scenario with isolated artifacts,
realistic tool/action chaining, mounted-server configurability, and env-gated
real LLM API calls.

**Backlog:** `TASK-2394.6`

**Spec:** `Docs/superpowers/specs/2026-06-20-mcp-real-world-uat-smoke-design.md`

## Task 1: Scenario And Artifact Contract

**Files:**

- Modify: `mcp_unified/smoke/scenarios.py`
- Modify: `mcp_unified/smoke/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests**

Cover:

- CLI accepts `--scenario real-world`.
- Real-world scenario report redacts absolute artifact paths.
- Missing required artifact tool is a strict failure and best-effort skip.
- Same-host artifact setup records only root-relative names.

- [ ] **Step 2: Implement scenario skeleton**

Add `run_real_world_scenario()` with artifact setup, initialize, tools/list,
artifact-root redaction, and controlled missing-tool handling. Keep baseline
unchanged.

- [ ] **Step 3: Verify**

Run the focused tests for the new scenario.

## Task 2: Deterministic Fixture Artifact Tools

**Files:**

- Modify: `mcp_unified/smoke/fixtures.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests**

Cover:

- Fixture runtime exposes artifact tools only when configured with an artifact
  root.
- Fixture read -> summarize -> write -> stat chain succeeds.
- Attempts to escape the artifact root fail.
- Fixture runtime can discover the artifact root from
  `MCP_SMOKE_ARTIFACT_ROOT` for stdio/live fixture servers.

- [ ] **Step 2: Implement fixture tools**

Add deterministic `artifact.read`, `artifact.summarize`, `artifact.write`, and
`artifact.stat` tools scoped to the fixture artifact root. Support direct
constructor injection for in-process tests and env discovery for fixture server
processes.

- [ ] **Step 3: Verify**

Run focused fixture and scenario tests.

## Task 3: CLI Configuration For Mounted/Live Runs

**Files:**

- Modify: `mcp_unified/smoke/cli.py`
- Modify: `mcp_unified/smoke/scenarios.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests**

Cover:

- CLI forwards artifact dir and tool-name overrides.
- JSON object arguments are parsed and rejected if not objects.
- Reports contain root-relative artifact names only.
- Stdio fixture examples inherit `MCP_SMOKE_ARTIFACT_ROOT` when requested.

- [ ] **Step 2: Implement CLI options**

Add `--artifact-dir`, read/write/stat/summarize tool-name options, argument JSON
options, and same-host artifact setup controls needed by mounted/live runs.

- [ ] **Step 3: Verify**

Run focused CLI tests and at least one in-process `real-world` CLI smoke run.

## Task 4: Env-Gated Real LLM Step

**Files:**

- Add or modify: `mcp_unified/smoke/real_llm.py`
- Modify: `mcp_unified/smoke/scenarios.py`
- Modify: `mcp_unified/smoke/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py`

- [ ] **Step 1: Write failing tests**

Cover:

- LLM step is skipped when not requested.
- Missing env var is skip in best-effort and failure in strict when requested.
- Mocked OpenAI-compatible response produces a bounded structural success.
- API keys and full model output are redacted from reports.

- [ ] **Step 2: Implement gated provider call**

Use a small OpenAI-compatible request path with timeout and size bounds. Never
read env vars unless the step was explicitly requested.

- [ ] **Step 3: Verify**

Run mocked LLM tests. Do not run live LLM calls unless the environment is
explicitly configured by the operator.

## Task 5: Full Validation And PR Update

**Files:**

- Modify: `backlog/tasks/task-2394.6 - Add-real-world-MCP-tool-action-UAT-smoke-scenario.md`
- Modify PR notes if needed.

- [ ] **Step 1: Run verification**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -v
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit mcp_unified/smoke -f json -o /tmp/bandit_mcp_real_world_uat_smoke.json
```

- [ ] **Step 2: Run UAT smoke commands**

Run in-process real-world CLI. If the operator provides live LLM env, run the
LLM-enabled path and record the result. Otherwise record the intentional skip.

- [ ] **Step 3: Finalize tracking and commit**

Update Backlog task, commit the implementation, push the branch, and leave PR
#2415 as draft until the user explicitly asks to mark it ready.
