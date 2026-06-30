# ACP Aider And Continue Entrypoint Decisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve Aider and Continue ACP entrypoint/adapter decisions while keeping unsupported profiles conservative.

**Architecture:** Reuse the existing registry classification model instead of adding new runtime behavior. Aider becomes an external adapter candidate that blocks on missing/unverified `aider-acp`; Continue remains a documented candidate with the current `cn` package command and no ACP stdio entrypoint.

**Tech Stack:** YAML registry metadata, Python registry classifier tests, Python smoke-manifest tests, ACP Markdown docs.

---

### Task 1: Registry Decision Tests

**Files:**
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Modify: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [x] **Step 1: Write failing tests for seeded Aider and Continue rows**

Add tests asserting:
- `aider` uses `external_acp_adapter`, `acp_command == "aider-acp"`, adapter source/docs/package fields are present, and support remains `documented_unverified`.
- Aider classification with `aider` present and `aider-acp` absent returns `adapter_missing`.
- `continue_dev` uses display command `cn`, remains `documented_candidate`, has empty `acp_command`, and uses `entrypoint_strategy_missing`.
- Continue profile manifest stays documented-only and blocked from running.

- [x] **Step 2: Run tests to verify they fail**

Run:
```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q
```

Expected: FAIL on old Aider/Continue registry metadata.

### Task 2: Registry And Manifest Metadata

**Files:**
- Modify: `tldw_Server_API/Config_Files/agents.yaml`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Modify: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [x] **Step 1: Update seeded Aider registry row**

Set Aider to an external adapter candidate:
- `entrypoint_strategy: external_acp_adapter`
- `acp_command: aider-acp`
- adapter metadata for the third-party bridge
- support state remains `documented_unverified`
- compatibility notes say direct Aider prompting is not ACP certification

- [x] **Step 2: Update seeded Continue registry row**

Set display command to `cn`, keep `documented_candidate`, leave `acp_command` empty, and document that `continue` is a shell builtin locally while `@continuedev/cli` exposes `cn` without an ACP stdio server mode.

- [x] **Step 3: Run focused tests to verify green**

Run the same pytest command from Task 1.

### Task 3: Documentation Reconciliation

**Files:**
- Modify: `Docs/Development/ACP_Compatibility_Matrix.md`
- Modify: `Docs/Development/ACP_Certification_Checklist.md`
- Modify: `Docs/Development/ACP_OSS_Custom_Certification_2026_05_11.md`
- Modify: `backlog/tasks/task-2365 - Resolve-ACP-Aider-and-Continue-entrypoint-decisions.md`

- [x] **Step 1: Update matrix rows**

Document Aider as an external adapter candidate blocked on `aider-acp` live certification. Document Continue as current `cn` CLI with no ACP stdio entrypoint.

- [x] **Step 2: Update checklist and legacy note**

Clarify that direct one-shot/headless prompting does not satisfy ACP certification. Add explicit Aider adapter-candidate and Continue command-mismatch guidance.

- [x] **Step 3: Update Backlog task metadata**

Record modified files, implementation notes, and plan path for `TASK-2365`.

### Task 4: Final Verification And PR

**Files:**
- No new implementation files.

- [x] **Step 1: Run focused pytest**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q
```

- [x] **Step 2: Run Bandit on touched Python scope**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -s B101 -f json -o /tmp/bandit_acp_aider_continue_decisions.json
```

- [x] **Step 3: Run formatting/metadata checks**

```bash
git diff --check
git status --short --branch
```

- [x] **Step 4: Commit, push, and open PR**

Create a PR against `dev`, link #2050 and #2051, and update #1563 without closing it until the PR merges.
