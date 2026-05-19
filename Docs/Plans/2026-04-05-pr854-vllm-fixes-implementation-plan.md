# PR 854 Managed vLLM Follow-Up Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three blocking review issues in PR 854 so managed vLLM routing reflects real instance health, slow starts converge correctly, and provider metadata stops advertising a fake managed default.

**Architecture:** Keep the existing managed-vLLM structure intact. Tighten routing invariants in the resolver, make lifecycle state transitions reflect startup latency instead of immediate failure, and align provider-listing metadata with actual routing behavior rather than guessed defaults.

**Tech Stack:** FastAPI, Python 3.11, pytest, SQLite-backed repository, Jobs worker, managed vLLM routing helpers.

---

## File Map

- Verify: execution happens from the PR 854 branch or a worktree based on it
  Responsibility: prevent trying to apply vLLM follow-up fixes on a checkout that does not contain the managed-vLLM code.
- Modify: `tldw_Server_API/app/core/VLLM_Management/resolver.py`
  Responsibility: request-time instance selection and route validation.
- Modify: `tldw_Server_API/app/core/VLLM_Management/service.py`
  Responsibility: lifecycle state transitions for start/probe/restart flows.
- Modify: `tldw_Server_API/app/core/VLLM_Management/reconciler.py`
  Responsibility: background convergence of persisted desired state to observed health.
- Modify: `tldw_Server_API/app/main.py`
  Responsibility: startup wiring for managed-vLLM worker and reconciler loop.
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_metadata.py`
  Responsibility: provider-listing metadata exposed to UI and clients.
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
  Responsibility: top-level provider payload fields such as `default_model` and `endpoint`.
- Modify: `tldw_Server_API/tests/VLLM_Management/test_request_resolver.py`
  Responsibility: routing rejection coverage for unhealthy and stopped instances.
- Modify: `tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py`
  Responsibility: caller-level error mapping for managed-route rejection.
- Modify: `tldw_Server_API/tests/VLLM_Management/test_jobs_service.py`
  Responsibility: startup-state transition and readiness-window coverage.
- Modify: `tldw_Server_API/tests/VLLM_Management/test_reconciler.py`
  Responsibility: periodic reconciliation behavior.
- Modify: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`
  Responsibility: FastAPI lifespan wiring for reconciler startup and shutdown.
- Modify: `tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py`
  Responsibility: provider metadata contract when no default instance is configured.
- Modify: `Docs/User_Guides/Integrations_Experiments/Managed_vLLM.md`
  Responsibility: operator-facing behavior notes for health gating and reconciliation.

---

### Task 0: Preflight The Execution Branch

**Files:**
- Verify: git branch/worktree state only

- [ ] **Step 1: Verify this workspace contains PR 854 code**

Run:
`git branch --show-current`
`git rev-parse --abbrev-ref --symbolic-full-name @{upstream}`
`test -f tldw_Server_API/app/core/VLLM_Management/resolver.py && echo ok`

Expected:
- branch is the PR branch or a follow-up branch based on it
- upstream is `origin/pr-854-review` or equivalent
- `resolver.py` exists in `app/core/VLLM_Management`

- [ ] **Step 2: Stop immediately if preflight fails**

If the managed-vLLM files are absent, do not start implementation. Move to the PR worktree first.

- [ ] **Step 3: Commit**

No commit for preflight.

---

### Task 1: Enforce Runtime Health In The Resolver

**Files:**
- Modify: `tldw_Server_API/app/core/VLLM_Management/resolver.py`
- Test: `tldw_Server_API/tests/VLLM_Management/test_request_resolver.py`
- Test: `tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- unhealthy instance selected explicitly returns `ValueError`
- stopped instance selected explicitly returns `ValueError`
- missing default still returns the existing not-found error
- healthy instance still resolves normally
- chat and embeddings callers still surface managed-route rejection as user-facing request errors

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management/test_request_resolver.py tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py -v`
Expected: new unhealthy/stopped routing tests fail because resolver currently ignores `observed_state`, or caller-level mapping tests fail because the wrong error reaches chat or embeddings wiring.

- [ ] **Step 3: Write minimal implementation**

Update `resolve_vllm_instance_for_request()` to reject instances whose persisted runtime state is not usable for inference.

Suggested rule:
- allow `observed_state == "healthy"`
- reject `starting`, `stopped`, `stopping`, `failed`, `unhealthy`

Use specific messages:
- `instance not healthy`
- `instance not reachable`
- `instance not found`
- `instance lacks required capability`

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management/test_request_resolver.py tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/resolver.py tldw_Server_API/tests/VLLM_Management/test_request_resolver.py tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py
git commit -m "fix(vllm): reject unhealthy managed routes"
```

---

### Task 2: Make Slow Starts Converge Instead Of Flipping Directly To Unhealthy

**Files:**
- Modify: `tldw_Server_API/app/core/VLLM_Management/service.py`
- Modify: `tldw_Server_API/tests/VLLM_Management/test_jobs_service.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- `start_instance()` keeps instance in `starting` when the first probe fails with a transient startup error
- follow-up `probe_instance()` can promote that same instance to `healthy`
- hard start failures from `executor.start()` still mark `failed`

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management/test_jobs_service.py -v`
Expected: new startup-transition tests fail because first probe miss currently becomes `unhealthy`.

- [ ] **Step 3: Write minimal implementation**

Refactor `start_instance()` and `_apply_probe_result()` to distinguish:
- process launch failure
- startup still warming
- post-start health regression

Concrete shape:
- keep `observed_state="starting"` after spawn if the initial probe is not yet reachable
- persist `last_error` for startup diagnostics
- only transition to `unhealthy` from `probe_instance()` or later reconciliation when a previously running or starting instance remains unreachable on a later probe
- add a bounded startup rule so instances cannot remain `starting` forever

Suggested bound:
- use a simple startup timeout such as `VLLM_MANAGEMENT_STARTUP_TIMEOUT_SECONDS`
- prefer persisted `executor_handle["started_at"]` when available
- fall back to persisted timestamps already on the record if executor metadata is missing

Out of scope for this follow-up:
- no brand-new scheduler beyond the existing reconciler loop
- no large schema redesign for probe history
- no retry-policy expansion beyond what is needed for bounded startup convergence

Avoid broad redesign. Do not introduce a new scheduler if the existing reconciler loop can converge state.

- [ ] **Step 4: Run targeted tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management/test_jobs_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/service.py tldw_Server_API/tests/VLLM_Management/test_jobs_service.py
git commit -m "fix(vllm): preserve starting state during cold boot"
```

---

### Task 3: Run Periodic Reconciliation Instead Of One-Shot Startup Probe Only

**Files:**
- Modify: `tldw_Server_API/app/core/VLLM_Management/reconciler.py`
- Modify: `tldw_Server_API/app/main.py`
- Test: `tldw_Server_API/tests/VLLM_Management/test_reconciler.py`
- Test: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`

- [ ] **Step 1: Write the failing tests**

Add tests covering:
- reconciler loop probes instances whose `desired_state == "running"` and `observed_state == "starting"`
- FastAPI lifespan schedules a long-running reconciler loop, not only a one-shot probe
- FastAPI shutdown stops the reconciler cleanly

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management/test_reconciler.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v`
Expected: FAIL on the new lifecycle expectations if only one-shot startup probe exists or shutdown does not own the reconciler task.

- [ ] **Step 3: Write minimal implementation**

Prefer the simplest behavior: use the reconciler loop as the startup probe path because `run_loop()` already probes before its first sleep. Do not schedule both a one-shot startup probe task and a long-running loop from `main.py`.

Requirements:
- separate stop event
- startup flag for loop enablement
- graceful shutdown path mirroring other background services
- no duplicate probing storm on boot
- keep the change localized to `main.py` and `reconciler.py`

- [ ] **Step 4: Run targeted tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management/test_reconciler.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/VLLM_Management/reconciler.py tldw_Server_API/app/main.py tldw_Server_API/tests/VLLM_Management/test_reconciler.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py
git commit -m "fix(vllm): run periodic lifecycle reconciliation"
```

---

### Task 4: Stop Advertising A Fake Managed Default

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_metadata.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py`

- [ ] **Step 1: Write the failing test**

Add a provider-listing test where:
- one or more managed instances exist
- `default_instance_id` is `None`
- provider metadata should report `default_instance_id=None`, `default_model=None`, and `default_base_url=None`
- provider should still expose its managed instances list and aggregated models
- top-level `/llm/providers` payload for `vllm` should not fall back to the first model or first endpoint when no managed default exists

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py -v`
Expected: FAIL because code currently falls back to the first stored instance.

- [ ] **Step 3: Write minimal implementation**

In `get_managed_vllm_provider_metadata()`:
- only populate `default_model` and `default_base_url` from `default_route`
- do not synthesize a default from `ordered_records[0]`
- keep instance list and model aggregation behavior intact

In `llm_providers.py`:
- do not set top-level `default_model` from `models[0]` for `vllm` when there is no managed default
- do not synthesize `endpoint` from managed metadata unless a managed default base URL actually exists

- [ ] **Step 4: Run targeted tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/LLM_Calls/provider_metadata.py tldw_Server_API/app/api/v1/endpoints/llm_providers.py tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py
git commit -m "fix(vllm): remove implicit provider default metadata"
```

---

### Task 5: Full Regression Pass And Docs Alignment

**Files:**
- Modify: `Docs/User_Guides/Integrations_Experiments/Managed_vLLM.md`
- Verify: `tldw_Server_API/tests/VLLM_Management/test_jobs_service.py`
- Verify: `tldw_Server_API/tests/VLLM_Management/test_request_resolver.py`
- Verify: `tldw_Server_API/tests/VLLM_Management/test_reconciler.py`
- Verify: `tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py`
- Verify: `tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py`
- Verify: `tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py`
- Verify: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`

- [ ] **Step 1: Update docs**

Document:
- routing only targets healthy managed instances
- startup can remain `starting` during cold boot
- periodic reconciliation promotes or demotes observed health
- provider listing only exposes a managed default when explicitly configured

- [ ] **Step 2: Run focused regression suite**

Run:
`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VLLM_Management tldw_Server_API/tests/LLM_Calls/test_vllm_instance_routing.py tldw_Server_API/tests/LLM_Calls/test_vllm_provider_listing.py tldw_Server_API/tests/LLM_Local/test_vllm_management_api.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v`

Expected: PASS

- [ ] **Step 3: Run security check on touched scope**

Run:
`source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/VLLM_Management tldw_Server_API/app/core/LLM_Calls/provider_metadata.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_pr854_vllm_fixes.json`

Expected: no new findings in changed code

- [ ] **Step 4: Run diff hygiene**

Run: `git diff --check`
Expected: no whitespace or merge-marker issues

- [ ] **Step 5: Commit**

```bash
git add Docs/User_Guides/Integrations_Experiments/Managed_vLLM.md
git commit -m "docs(vllm): align lifecycle and routing behavior"
```

---

## Merge Criteria

- Managed routing rejects non-healthy instances deterministically.
- Initial cold boot does not flip directly from `starting` to `unhealthy` on the first missed probe.
- A periodic reconciler loop exists and shuts down cleanly.
- Provider listings do not fabricate a managed default when none is configured.
- Focused vLLM regression suite passes.
- Bandit shows no new issues in touched files.
