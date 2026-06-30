# ACP PR 2038 Review Fixes Implementation Plan

**Goal:** Resolve the remaining actionable Qodo and Gemini feedback on the open ACP Goose/OpenCode PR.

**Architecture:** Keep the fixes scoped to the existing ACP runner configuration path, Go runner agent selection, registry metadata, and compatibility docs. Preserve explicit invalid `agentType` failures while making omitted defaults choose a runnable registered agent.

**Tech Stack:** FastAPI/Python ACP config helpers, pytest, Go stdio ACP runner tests, Markdown/YAML compatibility metadata.

---

## Stage 1: Host Home Propagation

**Goal:** Preserve `TLDW_ACP_HOST_HOME` whenever an absolute operator home is available, including absolute `HOME` and `ACP_RUNNER_ENV` override paths.
**Success Criteria:** Focused pytest proves absolute and override envs keep their configured `HOME` while also exporting host home.
**Tests:** `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_cwd.py -q`
**Status:** Complete

## Stage 2: Runner Default Agent Selection

**Goal:** Make omitted `agentType` select the configured default only when it has a runnable command, otherwise fall back to the first runnable registered agent.
**Success Criteria:** Go runner tests prove default session creation skips an empty custom profile and explicit empty-profile requests still fail.
**Tests:** `go test ./internal/acp -count=1`
**Status:** Complete

## Stage 3: Review Metadata And Docs

**Goal:** Align support-state docs and evidence metadata with the merged Goose/OpenCode/Aider certification state.
**Success Criteria:** Registry tests cover commit/runner metadata, support-state docs no longer conflict, and new tests include return annotations.
**Tests:** `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py -q`
**Status:** Complete

## Stage 4: Verification And Task Closeout

**Goal:** Run focused verification and record final status in Backlog.md.
**Success Criteria:** Focused pytest, Go tests, Bandit on touched Python, and diff checks pass or documented blockers are recorded.
**Tests:** Focused pytest, Go runner tests, Bandit, `git diff --check`.
**Status:** Complete
