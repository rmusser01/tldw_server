# vz_linux Real Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn `vz_linux` into a real Apple-silicon macOS VM runtime with native-helper boot, vsock guest-agent command execution, `virtiofs` workspace sharing, and sandbox session reuse.

**Architecture:** Keep the existing Python sandbox service as the admission and bookkeeping layer, but make the macOS helper contract real for `vz_linux`. The native helper owns `Virtualization.framework`, the Linux guest agent owns in-guest exec, and the sandbox session path persists VM control metadata so later runs can reuse the same VM safely.

**Tech Stack:** FastAPI service layer, existing sandbox orchestrator/store, Python dataclasses, pytest, Apple `Virtualization.framework` helper contract, Linux guest agent over vsock, `virtiofs`.

**Implementation Note:** This repo currently contains the Python-side helper client and protocol seams, not a native helper source tree. Unless a helper source tree is added during execution, the repo work should integrate against an operator-installed helper binary/service and keep automated tests on a fake helper contract plus host-gated compatibility smoke tests.

---

### Task 1: Expand the macOS helper and guest-agent protocol contracts
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py`
- Modify: `tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py`
- Create: `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`

**Step 1: Write the failing test**

```python
def test_fake_helper_supports_vz_linux_vm_create_and_exec(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    created = client.create_vm({
        "runtime": "vz_linux",
        "vm_name": "vz-linux-run-1",
        "session_mode": True,
    })
    exec_reply = client.exec_guest(
        vm_id=created.vm_id,
        request={"argv": ["/bin/echo", "ok"], "cwd": "/workspace"},
    )

    assert created.state == "created"
    assert created.details["transport"] == "vsock"
    assert exec_reply.exit_code == 0
    assert exec_reply.stdout == b"ok\n"
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -v`
Expected: FAIL because helper exec/status/session protocol models do not exist yet.

**Step 3: Write minimal implementation**

```python
@dataclass(slots=True)
class HelperExecReply:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    details: dict[str, Any] = field(default_factory=dict)


class MacOSVirtualizationHelperClient:
    def exec_guest(self, *, vm_id: str, request: dict[str, Any]) -> HelperExecReply:
        if is_truthy(os.getenv("TEST_MODE")):
            return HelperExecReply(exit_code=0, stdout=b"ok\n", details={"vm_id": vm_id, "transport": "vsock"})
        raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_virtualization/models.py tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py
git commit -m "test(vz_linux): add helper protocol contracts for vm create and exec"
```

### Task 2: Replace env-only vz_linux preflight with helper/template validation
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_common.py`
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`

**Step 1: Write the failing test**

```python
def test_vz_linux_preflight_uses_helper_and_template_validation(monkeypatch) -> None:
    monkeypatch.setattr(vz_common.sys, "platform", "darwin")
    monkeypatch.setattr(vz_common.platform, "machine", lambda: "arm64")

    class _FakeHelper:
        def validate_vz_linux_host(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "helper_ready": True,
                "template_ready": True,
                "template_id": "vz_linux:ubuntu-24.04",
                "execution_mode": "real",
            }

    monkeypatch.setattr(vz_common, "MacOSVirtualizationHelperClient", lambda: _FakeHelper())

    result = VZLinuxRunner().preflight(network_policy="deny_all")

    assert result.available is True
    assert result.reasons == []
    assert result.host["apple_silicon"] is True
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py::test_vz_linux_preflight_uses_helper_and_template_validation -v`
Expected: FAIL because preflight still depends on `*_FAKE_EXEC` and env-only readiness.

**Step 3: Write minimal implementation**

```python
helper = MacOSVirtualizationHelperClient()
validation = helper.validate_vz_linux_host({"network_policy": requested_policy})
if not validation.get("helper_ready"):
    reasons.append("macos_virtualization_helper_unavailable")
if not validation.get("template_ready"):
    reasons.append(self.template_missing_reason)
execution_mode = str(validation.get("execution_mode") or "none")
if execution_mode != "real":
    reasons.append("vz_linux_real_execution_unavailable")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_common.py tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py
git commit -m "feat(vz_linux): validate real helper and template readiness in preflight"
```

### Task 3: Implement ephemeral vz_linux run execution through the helper
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_common.py`
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_runner.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py`

**Step 1: Write the failing test**

```python
def test_vz_linux_start_run_executes_real_ephemeral_vm_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeHelper:
        def create_vm(self, request: dict[str, object]):
            calls.append(("create_vm", request))
            return HelperVMReply(vm_id="vm-ephemeral-1", state="created", details={"transport": "vsock"})

        def exec_guest(self, *, vm_id: str, request: dict[str, object]):
            calls.append(("exec_guest", {"vm_id": vm_id, **request}))
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

        def terminate_vm(self, vm_id: str):
            calls.append(("terminate_vm", {"vm_id": vm_id}))
            return True

    monkeypatch.setattr(vz_linux_module, "MacOSVirtualizationHelperClient", lambda: _FakeHelper())

    status = VZLinuxRunner().start_run(
        run_id="vz-run-1",
        spec=RunSpec(session_id=None, runtime=RuntimeType.vz_linux, base_image="ubuntu-24.04", command=["/bin/echo", "ok"]),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert [name for name, _payload in calls] == ["create_vm", "exec_guest", "terminate_vm"]
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py::test_vz_linux_start_run_executes_real_ephemeral_vm_command -v`
Expected: FAIL because `start_run()` still only supports fake execution.

**Step 3: Write minimal implementation**

```python
vm = helper.create_vm({
    "runtime": self.runtime_type.value,
    "run_id": run_id,
    "session_mode": False,
    "workspace_path": session_workspace,
    "workspace_mount": "virtiofs",
    "template": spec.base_image,
})
reply = helper.exec_guest(
    vm_id=vm.vm_id,
    request={"argv": list(spec.command), "cwd": "/workspace", "env": dict(spec.env or {})},
)
helper.terminate_vm(vm.vm_id)
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_common.py tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py
git commit -m "feat(vz_linux): execute ephemeral runs through native helper"
```

### Task 4: Persist and clean up vz_linux session VM control metadata
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/store.py`
- Modify: `tldw_Server_API/app/core/Sandbox/orchestrator.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Create: `tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py`
- Create: `tldw_Server_API/tests/sandbox/test_vz_linux_session_cleanup.py`

**Step 1: Write the failing test**

```python
def test_store_persists_vz_linux_session_control_metadata(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=str(tmp_path / "sandbox.db"))

    store.put_vz_session_control(
        session_id="sess-1",
        runtime="vz_linux",
        vm_id="vm-session-1",
        template_id="vz_linux:ubuntu-24.04",
        workspace_mount="/tmp/ws",
        agent_ready=True,
    )

    row = store.get_vz_session_control("sess-1")

    assert row["vm_id"] == "vm-session-1"
    assert row["template_id"] == "vz_linux:ubuntu-24.04"
    assert row["agent_ready"] is True


def test_destroy_session_terminates_persisted_vz_linux_vm(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeHelper:
        def terminate_vm(self, vm_id: str) -> bool:
            terminated.append(vm_id)
            return True

    monkeypatch.setattr(service_module, "MacOSVirtualizationHelperClient", lambda: _FakeHelper())

    svc = SandboxService()
    svc._orch.put_vz_session_control(
        session_id="sess-1",
        runtime="vz_linux",
        vm_id="vm-session-1",
        template_id="vz_linux:ubuntu-24.04",
        workspace_mount="/tmp/ws",
        agent_ready=True,
    )

    assert svc.destroy_session("sess-1") is True
    assert terminated == ["vm-session-1"]
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py -v`
Expected: FAIL because there is no persisted vz session-control metadata API yet.

**Step 3: Write minimal implementation**

```python
def put_vz_session_control(self, *, session_id: str, runtime: str, vm_id: str, template_id: str | None, workspace_mount: str | None, agent_ready: bool) -> None:
    ...

def get_vz_session_control(self, session_id: str) -> dict[str, Any] | None:
    ...

def delete_vz_session_control(self, session_id: str) -> bool:
    ...

def _destroy_session_serialized(self, session_id: str) -> bool:
    control = self._orch.get_vz_session_control(session_id)
    if control and control.get("vm_id"):
        MacOSVirtualizationHelperClient().terminate_vm(str(control["vm_id"]))
        self._orch.delete_vz_session_control(session_id)
    ...
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_vz_linux_session_cleanup.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/app/core/Sandbox/orchestrator.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_vz_linux_session_cleanup.py
git commit -m "feat(vz_linux): persist and clean up session vm control metadata"
```

### Task 5: Reuse running vz_linux VMs for sandbox sessions
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py`
- Create: `tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py`

**Step 1: Write the failing test**

```python
def test_vz_linux_session_reuses_existing_vm_for_second_run(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class _FakeHelper:
        def create_vm(self, request: dict[str, object]):
            calls.append("create_vm")
            return HelperVMReply(vm_id="vm-session-1", state="created", details={"session_mode": True})

        def exec_guest(self, *, vm_id: str, request: dict[str, object]):
            calls.append(f"exec_guest:{vm_id}")
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

    monkeypatch.setattr(vz_linux_module, "MacOSVirtualizationHelperClient", lambda: _FakeHelper())

    runner = VZLinuxRunner()
    runner.start_run("run-1", first_spec, str(tmp_path))
    runner.start_run("run-2", second_spec, str(tmp_path))

    assert calls == ["create_vm", "exec_guest:vm-session-1", "exec_guest:vm-session-1"]
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py -v`
Expected: FAIL because session-scoped VM reuse does not exist yet.

**Step 3: Write minimal implementation**

```python
session_control = self._load_vz_session_control(spec.session_id)
if session_control and session_control.get("agent_ready"):
    vm_id = str(session_control["vm_id"])
else:
    created = helper.create_vm({... "session_mode": True})
    self._store_vz_session_control(spec.session_id, created.vm_id, ...)
    vm_id = created.vm_id
reply = helper.exec_guest(vm_id=vm_id, request=exec_request)
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py
git commit -m "feat(vz_linux): reuse session vms for sandbox sessions"
```

### Task 6: Verify ACP compatibility without redesigning the ACP manager
**Status:** Complete

**Files:**
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py`
- Modify only if required: `tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py`

**Step 1: Write the failing test**

```python
async def test_acp_sandbox_runner_vz_linux_contract_still_uses_stream_backed_session(monkeypatch) -> None:
    manager = ACPSandboxRunnerManager(config=_sandbox_config(runtime="vz_linux"))
    session_id = await manager.create_session(cwd="/workspace", user_id=7)
    control = await manager._get_session_control_record(session_id)

    assert control["sandbox_session_id"]
    assert control["run_id"]
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -v`
Expected: FAIL only if `vz_linux` real-exec integration accidentally breaks the existing ACP stream/session contract.

**Step 3: Write minimal implementation**

```python
# Prefer no ACP manager code changes.
# Only adjust ACP code if the real vz_linux runtime/control metadata contract requires it.
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py
git commit -m "test(vz_linux): preserve acp sandbox session contract"
```

### Task 7: Update diagnostics, operator docs, and host-gated smoke coverage
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
- Modify: `tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py`

**Step 1: Write the failing test**

```python
def test_admin_diagnostics_reports_real_vz_linux_execution_mode(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")

    data = collect_macos_diagnostics()

    assert data["runtimes"]["vz_linux"]["execution_mode"] == "real"
    assert data["runtimes"]["vz_linux"]["available"] is True
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_diagnostics_reports_real_vz_linux_execution_mode -v`
Expected: FAIL because diagnostics still describe `vz_linux` as fake/none only.

**Step 3: Write minimal implementation**

```python
runtime_statuses["vz_linux"] = {
    "available": True,
    "execution_mode": "real",
    "reasons": [],
    "remediation": "Install and start the macOS virtualization helper and register a vz_linux template.",
}
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/macos_diagnostics.py Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py
git commit -m "docs(vz_linux): publish real execution readiness and smoke coverage"
```

### Task 8: Run the final verification slice and prepare review
**Status:** Complete

**Files:**
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md` if verification reveals drift
- Run-only: sandbox, ACP, diagnostics, and macOS host-gated tests

**Step 1: Run the targeted verification suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py tldw_Server_API/tests/sandbox/test_vz_linux_session_control_store.py tldw_Server_API/tests/sandbox/test_vz_linux_session_lifecycle.py tldw_Server_API/tests/sandbox/test_macos_runtime_service_dispatch.py tldw_Server_API/tests/sandbox/test_macos_diagnostics.py tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sandbox_runner_client.py -v`
Expected: PASS.

**Step 2: Run the host-gated smoke test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py -v`
Expected: PASS on supported Apple silicon macOS hosts, SKIP elsewhere.

**Step 3: Run Bandit on the touched implementation scope**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sandbox/macos_virtualization tldw_Server_API/app/core/Sandbox/runners/vz_common.py tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py tldw_Server_API/app/core/Sandbox/runtime_capabilities.py tldw_Server_API/app/core/Sandbox/service.py tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py -f json -o /tmp/bandit_vz_linux_real_execution.json`
Expected: `results: []` for the newly touched implementation paths.

**Step 4: Commit any final doc drift fix**

```bash
git add Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "chore(vz_linux): finalize real execution verification notes"
```

**Step 5: Request review**

```bash
git status --short
```
