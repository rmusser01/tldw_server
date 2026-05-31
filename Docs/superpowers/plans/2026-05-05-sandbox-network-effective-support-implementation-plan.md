# Sandbox Network Effective Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make sandbox network policy discovery and admission derive effective support from static policy contracts plus current readiness.

**Architecture:** Add one effective-support helper in `runtime_capabilities.py`, feed Docker/Firecracker readiness into shared preflight collection, and make policy admission plus discovery consume the helper. Keep unsupported/scaffold allowlist paths fail-closed and avoid adding new runtime allowlist implementations.

**Tech Stack:** Python dataclasses/helpers, pytest, existing SandboxService/SandboxPolicy contracts, Bandit.

---

### Task 1: Add Failing Coverage For Effective Network Support

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py`
- Modify: `tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py`

- [ ] **Step 1: Extend admission helper test setup**

Allow `_available_preflight()` in
`test_network_policy_contract_admission.py` to accept an `enforcement_ready`
override so tests can model ready and not-ready policy modes.

- [ ] **Step 2: Add Docker not-ready rejection test**

Add a test asserting Docker `allowlist` is rejected when the Docker preflight
has `{"deny_all": True, "allowlist": False}`.

- [ ] **Step 3: Preserve Docker ready admission**

Keep or add a test asserting Docker `allowlist` is admitted when the preflight
has `{"deny_all": True, "allowlist": True}`.

- [ ] **Step 4: Add discovery consistency tests**

In `test_runtime_inventory_contract.py`, assert each runtime's
`strict_deny_all_supported` and `strict_allowlist_supported` values match
`runtime_network_policy_effective_support(runtime, enforcement_ready)`.

- [ ] **Step 5: Confirm RED**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  -q --timeout=60
```

Expected: fail because admission still checks static support only and discovery
does not use the shared effective-support helper.

### Task 2: Implement Effective Support Helper And Admission

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/app/core/Sandbox/policy.py`

- [ ] **Step 1: Add readiness helpers**

Add Docker config readiness helpers to `runtime_capabilities.py` so
`collect_runtime_preflights()` can set Docker `enforcement_ready` for `deny_all`
and `allowlist`.

- [ ] **Step 2: Add effective-support helper**

Add `runtime_network_policy_effective_support(runtime, enforcement_ready)`.
Return false unless the static contract is `supported` or `host_gated`, strict
enforcement is possible, and current readiness is true.

- [ ] **Step 3: Wire preflight readiness**

Set Docker readiness to `deny_all=docker_available` and
`allowlist=docker_available and egress_enforcement and granular_enforcement`.
Set Firecracker readiness to `deny_all=firecracker_available` and
`allowlist=False`.

- [ ] **Step 4: Wire policy admission**

Update `_require_network_policy_supported()` to accept the selected runtime
preflight and use `runtime_network_policy_effective_support()`. Pass the
selected preflight from `apply_to_session()` and `apply_to_run()`.

- [ ] **Step 5: Confirm GREEN for admission tests**

Run the focused command from Task 1.

### Task 3: Align Discovery And Docs

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- Modify: `Docs/Sandbox/sandbox-security-policy-matrix.md`
- Modify: `backlog/tasks/task-54 - Stabilize-sandbox-network-policy-effective-support-reporting.md`

- [ ] **Step 1: Derive discovery booleans from the helper**

Use `runtime_network_policy_effective_support()` inside
`SandboxService.feature_discovery()` so `strict_deny_all_supported`,
`strict_allowlist_supported`, and `egress_allowlist_supported` are effective
support signals, not raw preflight/config echoes.

- [ ] **Step 2: Update Firecracker notes**

Ensure Firecracker discovery notes do not imply allowlist is usable today when
it remains scaffold/non-strict.

- [ ] **Step 3: Update docs**

Document that effective support requires the static contract and current
readiness to agree. Clarify that Docker's coarse `network=none` fallback is not
advertised as allowlist support.

- [ ] **Step 4: Run verification**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py \
  tldw_Server_API/tests/sandbox/test_lima_strict_admission.py \
  -q --timeout=60
python -m py_compile \
  tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
  tldw_Server_API/app/core/Sandbox/policy.py \
  tldw_Server_API/app/core/Sandbox/service.py
python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
  tldw_Server_API/app/core/Sandbox/policy.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  -f json -o /tmp/bandit_sandbox_network_effective_support.json
git diff --check
```

- [ ] **Step 5: Finalize task and commit**

Update `TASK-54`, stage all touched files, and commit:

```bash
git add \
  Docs/Sandbox/sandbox-runtime-capability-inventory.md \
  Docs/Sandbox/sandbox-security-policy-matrix.md \
  Docs/superpowers/specs/2026-05-05-sandbox-network-effective-support-design.md \
  Docs/superpowers/plans/2026-05-05-sandbox-network-effective-support-implementation-plan.md \
  "backlog/tasks/task-54 - Stabilize-sandbox-network-policy-effective-support-reporting.md" \
  tldw_Server_API/app/core/Sandbox/runtime_capabilities.py \
  tldw_Server_API/app/core/Sandbox/policy.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py \
  tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py
git commit -m "feat(sandbox): centralize network policy effective support"
```
