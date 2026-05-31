# Sandbox Cross-Runtime Session Reliability Contract Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable regression gate for sandbox session/recovery claims without changing runtime execution behavior.

**Architecture:** Treat `session_contract` as the source of static runtime posture and verify its projection through public discovery plus admin diagnostics. Keep real repair/recovery behavior scoped to `vz_linux`; this slice only hardens portable contract coverage and clarifies the remaining host-gated gaps.

**Tech Stack:** Python, pytest, Pydantic API schemas, sandbox runtime capability metadata.

---

## Design Review

- Keep the slice contract-only. Do not generalize repair, do not add warm reuse for host-local runtimes, and do not change helper or runner behavior.
- Avoid brittle tests that require Docker, Lima, Firecracker, or Apple Virtualization.framework availability. Use synthetic discovery rows and pure service projection helpers where possible.
- Documentation must say what is now covered by portable tests and what remains uncovered. Do not remove the Phase 4 recovery/repair gap entirely.

## Files

- Modify: `tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `backlog/tasks/task-124 - Add-sandbox-cross-runtime-session-reliability-contract-gate.md`

## Task 1: Add Portable Session/Admin Projection Regression Coverage

- [x] **Step 1: Write failing tests**

Add tests in `tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` that build synthetic runtime discovery rows and assert:

- every runtime row keeps `session_contract` details through `SandboxRuntimesResponse`
- `_runtime_diagnostics_item()` projects `session_reuse_model`, `requires_live_health_check`, and `repair_supported` from the same session contract
- only `supported` or `host_gated` repair states become `repair_supported=true`
- `seatbelt` and `worktree` remain `workspace_only`, do not require live health, and do not advertise repair

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q
```

Observed: failed on stale inventory wording because the new guard detected the old "beyond discovery-level `session_contract`" gap text. Admin projection assertions passed.

- [x] **Step 3: Implement only the minimum needed**

The current production projection already satisfied the contract, so no production code changed.

- [x] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q
```

Observed: pass with 8 tests.

## Task 2: Update Capability Inventory Wording

- [x] **Step 1: Update `Current Gaps`**

Change the session semantics gap so it no longer says portable contract checks are absent. Preserve the remaining gap for real recovery flows and host-gated repair ownership.

- [x] **Step 2: Add or adjust a maintenance rule**

Ensure the inventory tells future runtime authors to keep admin diagnostics aligned with `session_contract`.

- [x] **Step 3: Run docs/test guard**

Run the same focused capability gate after the docs edit:

```bash
python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q
```

Observed: pass with 8 tests, then 36 tests across the capability gate and runtime inventory contract.

## Task 3: Verification And Backlog Closeout

- [x] **Step 1: Run Python syntax check**

```bash
python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py
```

- [x] **Step 2: Run Bandit or record skip**

No production Python changed. Ran Bandit against the touched test file with pytest assert noise excluded; results were empty.

- [x] **Step 3: Run diff hygiene**

```bash
git diff --check
```

- [x] **Step 4: Update TASK-124**

Check acceptance criteria and definition-of-done items, record verification results, and add a final summary.
