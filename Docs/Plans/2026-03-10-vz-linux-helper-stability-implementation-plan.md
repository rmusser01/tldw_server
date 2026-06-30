# VZ Linux Helper Stability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current stubbed `vz_linux` macOS helper path with a first-party, in-repo helper daemon and move runtime readiness, template truth, and session reuse onto helper-backed facts.

**Architecture:** Keep Python authoritative for sandbox policy, run/session persistence, artifacts, and ACP integration. Add a narrow native helper daemon for `vz_linux` over a Unix socket, then make diagnostics, preflight, and session reuse consume helper-backed host/template/VM truth instead of env-only scaffolding.

**Tech Stack:** Python, pytest, FastAPI admin diagnostics, SQLite-backed sandbox state, native macOS helper subproject, Unix domain sockets, `Virtualization.framework`, vsock guest agent contract.

---

### Task 1: Freeze The Helper Protocol

**Files:**
- Create: `tools/macos-vz-helper/README.md`
- Create: `tools/macos-vz-helper/PROTOCOL.md`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

**Step 1: Write the failing tests**

Add tests that expect the Python helper layer to parse and expose:

- helper `protocol_version`
- helper `helper_version`
- `ping`
- `validate_host`
- `get_vm_status`
- `list_vms`

Example expectation:

```python
def test_helper_ping_exposes_protocol_version():
    payload = {"protocol_version": "1", "helper_version": "0.1.0", "status": "ok"}
    result = parse_helper_ping(payload)
    assert result.protocol_version == "1"
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: failures for missing response models and/or parse helpers.

**Step 3: Write the minimal implementation**

- Add protocol-aware response models in `models.py`.
- Document the socket protocol in `tools/macos-vz-helper/PROTOCOL.md`.
- Document the native helper scope and non-goals in `tools/macos-vz-helper/README.md`.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm the new protocol tests pass.

**Step 5: Commit**

```bash
git add tools/macos-vz-helper/README.md tools/macos-vz-helper/PROTOCOL.md tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "test(vz_linux): freeze helper protocol surface"
```

### Task 2: Replace The Stub Client With A Real Socket Client

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

**Step 1: Write the failing tests**

Add tests that start a fake Unix-socket helper server and assert the Python client can:

- connect to the configured socket
- send `ping`
- send `validate_host`
- send `create_vm`
- send `exec_guest`
- send `terminate_vm`
- send `get_vm_status`
- map helper error payloads to custom exceptions

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -q
```

Expected: failures because the client still raises `macos_virtualization_helper_unavailable` outside `TEST_MODE`.

**Step 3: Write the minimal implementation**

- Add socket-path config to `helper_client.py`.
- Implement request/response transport over a Unix socket.
- Preserve deterministic fake transport only for explicit test-mode tests that still need it.
- Introduce narrow helper exceptions for unavailable, protocol mismatch, and helper-declared failure.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm all client transport tests pass.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "feat(vz_linux): add real macOS helper socket client"
```

### Task 3: Move Host Readiness Onto Helper Truth

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`

**Step 1: Write the failing tests**

Add tests that expect:

- `vz_linux` preflight to call helper `validate_host`
- helper version/protocol to appear in admin diagnostics
- env-only helper readiness to stop being sufficient when a real helper is configured

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py -q
```

Expected: failures because diagnostics and preflight still infer too much from env flags.

**Step 3: Write the minimal implementation**

- Make `vz_linux` preflight use helper-backed host validation.
- Update admin diagnostics to call helper `ping` and `validate_host`.
- Keep the non-macOS/Apple-silicon host facts in Python, but do not claim helper/template readiness without helper confirmation.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm the updated readiness path passes.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/app/core/Sandbox/macos_diagnostics.py tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py
git commit -m "feat(vz_linux): use helper-backed host readiness"
```

### Task 4: Add Helper-Owned Template Registration And Validation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/app/core/Sandbox/image_store.py`
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

**Step 1: Write the failing tests**

Add tests that expect:

- the helper client to support `register_template` and `validate_template`
- `vz_linux` to reject unknown or invalid templates based on helper truth
- diagnostics to report helper-backed template readiness

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: failures because template readiness is still env metadata and the client lacks template APIs.

**Step 3: Write the minimal implementation**

- Add helper operations for template registration and validation.
- Treat `spec.base_image` as a direct base-image path or helper-issued `template_id` in phase 1.
- Reduce `SandboxImageStore` to Python-side metadata support only; do not let it claim runnable-template truth.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm helper-backed template validation passes.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/app/core/Sandbox/macos_diagnostics.py tldw_Server_API/app/core/Sandbox/image_store.py tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "feat(vz_linux): move template truth into helper"
```

### Task 5: Make Session Reuse Health-Based

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/core/Sandbox/orchestrator.py`
- Modify: `tldw_Server_API/app/core/Sandbox/store.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py`

**Step 1: Write the failing tests**

Add tests that expect:

- session reuse to call helper `get_vm_status`
- stale `vm_id` rows to be treated as unhealthy
- `destroy_session()` to tolerate a helper response meaning "already gone"
- ACP-compatible session reuse to preserve existing sandbox semantics

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -q
```

Expected: failures because reuse currently trusts stored control metadata without helper health confirmation.

**Step 3: Write the minimal implementation**

- Query helper `get_vm_status` before reusing a session VM.
- Define helper result handling for healthy, missing, and unhealthy VMs.
- Keep Python authoritative for session identity and row deletion.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm the session-reuse health checks pass.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/core/Sandbox/orchestrator.py tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py
git commit -m "feat(vz_linux): gate session reuse on helper vm health"
```

### Task 6: Add Restart-Safe Reconciliation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py`
- Test: `tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py`

**Step 1: Write the failing tests**

Add tests that expect:

- helper `list_vms` results to be reconcilable with persisted `sandbox_vz_sessions`
- diagnostics to surface reconciliation mismatches clearly
- summarized runtime discovery to remain concise while still reflecting helper-backed truth

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py -q
```

Expected: failures because helper runtime state is not yet enumerable and reconciliation is absent.

**Step 3: Write the minimal implementation**

- Add helper `list_vms`.
- Add a reconciliation helper in `service.py` or a nearby sandbox support module.
- Use it in admin diagnostics and safe startup paths without introducing a second persistence layer.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm reconciliation behavior passes.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/core/Sandbox/macos_diagnostics.py tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py
git commit -m "feat(vz_linux): add helper reconciliation support"
```

### Task 7: Update Operator Docs And Real-Host E2E Expectations

**Files:**
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/Plans/2026-03-10-vz-linux-real-host-e2e-design.md`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

**Step 1: Write the failing tests**

Add or adjust the real-host smoke expectations so they assert:

- helper socket reachability
- helper protocol version availability
- helper-backed template validation instead of env-only template readiness

**Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q
```

Expected: failures or outdated skip reasons until the docs/test contract is updated.

**Step 3: Write the minimal implementation**

- Update operator documentation for the in-repo helper daemon.
- Update the real-host smoke module to expect helper-backed readiness.
- Remove stale references that imply env-only helper truth is the stable path.

**Step 4: Run the tests to verify they pass**

Run the same pytest command and confirm the smoke contract passes or skips with the new helper-specific reasons.

**Step 5: Commit**

```bash
git add Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md Docs/Plans/2026-03-10-vz-linux-real-host-e2e-design.md tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "docs(vz_linux): align operator guidance with first-party helper"
```

### Task 8: Run Verification And Security Checks

**Files:**
- Modify: none
- Test: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Test: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py`

**Step 1: Run the focused regression suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_runner.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py \
  tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -q
```

Expected: all targeted tests pass; real-host E2E may still skip on unprepared hosts with explicit helper/template reasons.

**Step 2: Run Bandit on the touched Python scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/macos_virtualization \
  tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py \
  tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/tests/sandbox \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py \
  -f json -o /tmp/bandit_vz_linux_helper_stability.json
```

Expected: no new findings in the touched implementation scope.

**Step 3: Commit the verification-aligned final state**

```bash
git add -A
git commit -m "chore(vz_linux): finalize helper stability slice"
```
