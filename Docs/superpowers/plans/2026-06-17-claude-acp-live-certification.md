# Claude ACP Live Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the pinned Claude Code ACP adapter path for issue #1564 and update support claims only if direct and backend live evidence passes.

**Architecture:** Keep Claude Code modeled as an `external_acp_adapter` profile backed by `@agentclientprotocol/claude-agent-acp@0.40.0`. Run the adapter from a disposable local npm install by prepending its `node_modules/.bin` to `PATH`, then use the existing registry-backed smoke helper and backend REST E2E flow as the evidence source. Documentation and registry changes are evidence-driven: failed probes keep `documented_unverified`; passed probes may upgrade only the verified macOS host profile with explicit caveats.

**Tech Stack:** Python 3.11, pytest, FastAPI ACP endpoints, `Helper_Scripts/Testing-related/acp_certification_smoke.py`, Node/npm/npx, Claude Code CLI, Backlog.md, GitHub CLI.

---

### File Map

- Modify: `backlog/tasks/task-2368 - Certify-Claude-Code-ACP-adapter-live-path-for-issue-1564.md`
  - Track implementation notes, verification results, blockers, and final summary.
- Modify: `Docs/superpowers/plans/2026-06-17-claude-acp-live-certification.md`
  - Keep this plan current as tasks complete.
- Modify only if live evidence passes: `tldw_Server_API/Config_Files/agents.yaml`
  - Upgrade `claude_code` support state, verification level, and caveat text for the verified profile.
- Modify only if live evidence passes: `Docs/Development/ACP_Compatibility_Matrix.md`
  - Replace Claude row skip evidence with exact direct/backend evidence and narrow caveats.
- Modify only if live evidence passes: `Docs/Published/User_Guides/Integrations_Experiments/Anthropic_ClaudeCode_ClaudeSDK_Setup.md`
  - Update setup wording from documented candidate to verified-with-caveats for the exact host/profile.
- Modify only if live evidence passes: `Docs/Published/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
  - Keep release-facing wording aligned with matrix caveats.
- Modify only if live evidence passes: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
  - Assert the seeded Claude profile reflects the new evidence level.
- Modify only if live evidence passes: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`
  - Assert the registry-backed Claude manifest reflects support state and remains runnable only when its adapter is on `PATH`.

### Task 1: Baseline And Adapter Preflight

**Files:**
- Read: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Read: `tldw_Server_API/Config_Files/agents.yaml`
- Read: `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- Read: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Read: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [x] **Step 1: Run focused baseline tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q
```

Result: PASS, `82 passed, 6 warnings in 2.80s` before registry/docs edits.

- [x] **Step 2: Create disposable adapter workspace**

Run:

```bash
cert_dir="$(mktemp -d /tmp/tldw-claude-agent-acp-0.40.0.XXXXXX)"
npm --prefix "$cert_dir" install @agentclientprotocol/claude-agent-acp@0.40.0
```

Result: local adapter binary existed at `/tmp/tldw-claude-agent-acp-0.40.0.AjIJE4/node_modules/.bin/claude-agent-acp`.

- [x] **Step 3: Record toolchain versions without secrets**

Run bounded version/help probes for `node`, `npm`, `claude`, and `claude-agent-acp`; record only versions, binary paths, exit codes, and bounded output previews.

Result: recorded `node v26.0.0`, `npm 11.12.1`, `npx 11.12.1`, `claude 2.1.177 (Claude Code)`, and npm metadata for `@agentclientprotocol/claude-agent-acp@0.40.0`.

### Task 2: Direct ACP Stdio Certification Gate

**Files:**
- Read: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Potentially modify only if evidence passes: `Docs/Development/ACP_Compatibility_Matrix.md`

- [x] **Step 1: Emit the Claude profile manifest with the adapter on PATH**

Run:

```bash
PATH="$cert_dir/node_modules/.bin:$PATH" python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile claude_code --format json
```

Result: manifest reported `probe_state=ready_to_probe`, no blockers, and `acp_initialize_probe`.

- [x] **Step 2: Run the direct probe**

Run:

```bash
PATH="$cert_dir/node_modules/.bin:$PATH" python Helper_Scripts/Testing-related/acp_certification_smoke.py --agent-profile claude_code --run
```

Result: `PASS acp_initialize_probe`.

### Task 3: Backend Live E2E Certification Gate

**Files:**
- Read: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Read: `tools/tldw-agent/scripts/verify-local-build.sh`
- Potentially modify only if evidence passes: docs and seed registry listed in File Map.

- [x] **Step 1: Ensure backend E2E environment**

Use existing `TLDW_E2E_SERVER_URL` and `TLDW_E2E_API_KEY` if present. If absent, start a local loopback backend with an explicit temporary single-user API key and stop it after the run.

Result: local loopback backend on `127.0.0.1:18004` returned ACP health with runner `ok`, Claude status `available`, and the adapter on both server and runner `PATH`.

- [x] **Step 2: Run backend live E2E**

Run:

```bash
PATH="$cert_dir/node_modules/.bin:$PATH" ACP_AGENT_PROFILE=claude_code python Helper_Scripts/Testing-related/acp_certification_smoke.py --profile live-e2e --run
```

Result: exit 0. `PASS live_backend_acp_e2e` with session `44b71fdb-c014-41e1-8b56-14fa310039e6`, `stop_reason=end_turn`, `events_total=2`, `artifacts_total=0`, and `diagnostics_total=0`; `tools/tldw-agent/scripts/verify-local-build.sh` also passed.

### Task 4: Evidence-Driven Registry, Docs, And Tests

**Files:**
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Modify: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`
- Modify: `tldw_Server_API/Config_Files/agents.yaml`
- Modify: `Docs/Development/ACP_Compatibility_Matrix.md`
- Modify: `Docs/Published/User_Guides/Integrations_Experiments/Anthropic_ClaudeCode_ClaudeSDK_Setup.md`
- Modify: `Docs/Published/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`

- [x] **Step 1: Write failing tests for any support-state upgrade**

If both direct and backend live gates passed, change expected Claude seed support assertions to `supported_with_caveats` and `live_e2e_tested`, plus a manifest assertion that adapter metadata remains exact-pinned.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py::test_seeded_claude_code_profile_uses_current_external_acp_adapter_candidate tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q
```

Result: FAIL as expected before seed update: tests still observed `documented_unverified` / `documented_only`.

- [x] **Step 2: Update seed data and docs minimally**

Update only Claude Code release-facing fields and docs with the exact host, adapter version, command shape, date, evidence summary, and remaining caveats. Do not claim sandbox, artifacts, non-empty MCP injection, reviewer loop, or other hosts unless tested.

- [x] **Step 3: Re-run focused tests**

Run the same focused tests.

Result: PASS, `2 passed, 6 warnings in 1.79s`.

### Task 5: Verification, Issue Update, And PR

**Files:**
- Modify: `backlog/tasks/task-2368 - Certify-Claude-Code-ACP-adapter-live-path-for-issue-1564.md`

- [x] **Step 1: Run focused verification**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q
```

Result: PASS, `83 passed, 6 warnings in 2.12s`; ACP health coverage also passed with `21 passed, 6 warnings in 25.53s`.

- [x] **Step 2: Run Bandit for touched Python scope when applicable**

Run if Python files changed:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -f json -o /tmp/bandit_claude_acp_live_certification.json
```

Result: Full touched-test scan reported only existing pytest `B101 assert_used` findings (`452` low severity, no other test IDs). Re-run with `-s B101` exited 0 with `0` results in `/tmp/bandit_claude_acp_live_certification_no_b101.json`.

- [x] **Step 3: Update GitHub issue #1564**

Comment with exact evidence or blocker. Keep issue open unless all acceptance criteria are satisfied.

Result: Commented certification evidence and retained caveats at https://github.com/rmusser01/tldw_server/issues/1564#issuecomment-4730743066.

- [x] **Step 4: Open PR if repo files changed**

Push branch `codex/claude-acp-live-certification` and open a narrow PR. The PR body must include a human-readable change summary explaining what changed and why the support claim is or is not upgraded.

Result: Opened PR https://github.com/rmusser01/tldw_server/pull/2374.
