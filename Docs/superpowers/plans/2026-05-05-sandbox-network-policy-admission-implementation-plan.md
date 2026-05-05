# Sandbox Network Policy Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce sandbox runtime `network_policy_contract` during session and run admission.

**Architecture:** Add one static contract validation helper in `SandboxPolicy`, then call it from both session and run policy normalization after trust-profile defaults are applied. Keep dynamic host/runtime readiness in the existing service and runner preflight paths.

**Tech Stack:** Python, pytest, Backlog.md, Bandit.

---

## Files

- Modify: `tldw_Server_API/app/core/Sandbox/policy.py`
- Modify: `tldw_Server_API/tests/sandbox/test_lima_strict_admission.py` or create a nearby focused policy test file
- Create: `Docs/superpowers/specs/2026-05-05-sandbox-network-policy-admission-design.md`
- Create: `Docs/superpowers/plans/2026-05-05-sandbox-network-policy-admission-implementation-plan.md`
- Modify: `backlog/tasks/task-47 - Enforce-sandbox-network-policy-contract-during-admission.md`

## Task 1: Add Failing Policy Tests

**Files:**
- Test: `tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py`

- [x] **Step 1: Write failing tests**

Cover `SandboxPolicy.apply_to_run()` and `SandboxPolicy.apply_to_session()` directly:

```python
def test_apply_to_run_rejects_worktree_deny_all_from_standard_default():
    policy = SandboxPolicy(SandboxPolicyConfig(default_runtime=RuntimeType.worktree))
    spec = RunSpec(session_id=None, runtime=RuntimeType.worktree, base_image=None, command=["echo", "ok"])

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(spec, firecracker_available=False, runtime_preflights={RuntimeType.worktree: available_worktree_preflight()})

    assert exc.value.reasons == ["strict_deny_all_not_supported"]
```

Add equivalent cases for:

- `seatbelt` `deny_all`
- `worktree` `allowlist`
- invalid policy value
- `vz_linux` `deny_all` accepted at static contract layer
- `vz_linux` `allowlist` rejected
- `apply_to_session()` rejects the same host-local defaulted policy

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py -q --timeout=60
```

Expected: tests fail because `SandboxPolicy` does not yet validate the static network-policy contract.

## Task 2: Add Static Contract Admission Helper

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/policy.py`

- [x] **Step 1: Import runtime metadata accessor**

Import `runtime_network_policy_metadata` from `runtime_capabilities.py`.

- [x] **Step 2: Implement helper**

Add a helper to `SandboxPolicy`:

```python
@staticmethod
def _require_network_policy_supported(runtime: RuntimeType, network_policy: str | None) -> None:
    requested_policy = str(network_policy or "deny_all").strip().lower() or "deny_all"
    if requested_policy not in {"deny_all", "allowlist"}:
        raise SandboxPolicy.PolicyUnsupported(runtime, requirement=requested_policy, reasons=["unsupported_network_policy"])
    contract = runtime_network_policy_metadata(runtime)
    mode = contract.deny_all if requested_policy == "deny_all" else contract.allowlist
    if mode.support_state in {"unsupported", "not_applicable"} or not mode.strict_enforcement:
        reason = "strict_allowlist_not_supported" if requested_policy == "allowlist" else "strict_deny_all_not_supported"
        raise SandboxPolicy.PolicyUnsupported(runtime, requirement=requested_policy, reasons=[reason])
```

Wrap long lines to match project style.

- [x] **Step 3: Call helper from admission paths**

In `apply_to_session()` and `apply_to_run()`, call the helper after:

- runtime selection
- trust support validation
- profile/default `network_policy` assignment

- [x] **Step 4: Run tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py -q --timeout=60
```

Expected: new tests pass.

## Task 3: Regression Verification

**Files:**
- Existing tests under `tldw_Server_API/tests/sandbox/`

- [x] **Step 1: Run related sandbox tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_network_policy_contract_admission.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_lima_strict_admission.py -q --timeout=60
```

- [x] **Step 2: Run Bandit on touched production code**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/policy.py -f json -o /tmp/bandit_sandbox_network_policy_admission.json
```

- [x] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

- [x] **Step 4: Update Backlog task and commit**

Record verification in `TASK-47`, then commit the complete slice.
