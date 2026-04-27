# vz_linux Real Host E2E Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in pytest smoke suite that proves real `vz_linux` ephemeral execution and real session VM reuse on a prepared Apple silicon macOS host.

**Architecture:** Reuse the existing sandbox service, runner, and VZ session-control store instead of inventing a new harness. The test module stays host-gated and opt-in, uses a temp SQLite sandbox store for isolation, and asserts session reuse via persisted `vm_id` metadata.

**Tech Stack:** pytest, existing sandbox service/orchestrator/store, temp SQLite store configuration, macOS host gating, real `vz_linux` runner preflight.

---

### Task 1: Add host-gating and isolated-store helpers for real vz_linux E2E
**Status:** Complete

**Files:**
- Create: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Reference: `tldw_Server_API/tests/sandbox/test_session_store_durability.py`
- Reference: `tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py`

**Step 1: Write the failing test**

```python
def test_vz_linux_real_host_e2e_requires_opt_in(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E", raising=False)

    with pytest.raises(pytest.skip.Exception, match="TLDW_SANDBOX_VZ_LINUX_E2E"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
```

**Step 2: Run test to verify it fails**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_host_e2e_requires_opt_in -v`
Expected: FAIL because the helper/gating fixture does not exist yet.

**Step 3: Write minimal implementation**

```python
def _configure_sqlite_store(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("SANDBOX_STORE_DB_PATH", str(tmp_path / "sandbox_store.db"))
    monkeypatch.setenv("SANDBOX_ROOT_DIR", str(tmp_path / "sandbox_root"))
    monkeypatch.setenv("SANDBOX_SNAPSHOT_PATH", str(tmp_path / "snapshots"))
    clear_config_cache()


def _require_vz_linux_real_host_e2e(monkeypatch, tmp_path: Path) -> str:
    _configure_sqlite_store(monkeypatch, tmp_path)
    if sys.platform != "darwin":
        pytest.skip("macOS host only")
    if platform.machine() != "arm64":
        pytest.skip("Apple silicon host only")
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_E2E=1 to enable this test")
    base_image = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE") or "").strip()
    if not base_image:
        pytest.skip("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE is required")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    return base_image
```

**Step 4: Run test to verify it passes**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_host_e2e_requires_opt_in -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "test(vz_linux): add real-host e2e gating helpers"
```

### Task 2: Add the real ephemeral vz_linux smoke test
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Reference: `tldw_Server_API/app/core/Sandbox/runners/vz_linux_runner.py`

**Step 1: Write the failing test**

```python
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_real_ephemeral_run_smoke(monkeypatch, tmp_path: Path) -> None:
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)

    status = VZLinuxRunner().start_run(
        "vz-linux-real-ephemeral",
        RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image=base_image,
            command=["/bin/echo", "vz-linux-e2e"],
            network_policy="deny_all",
        ),
        session_workspace=None,
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `source ../../.venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_E2E=1 TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=test-image python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_ephemeral_run_smoke -v`
Expected: SKIP on unprepared hosts or FAIL on a prepared host until the test asserts the real preflight contract correctly.

**Step 3: Write minimal implementation**

```python
runner = VZLinuxRunner()
preflight = runner.preflight(network_policy="deny_all")
if not preflight.available or preflight.execution_mode != "real":
    pytest.skip(f"vz_linux real execution unavailable: {preflight.reasons}")

hub = get_hub()
hub._buffers.pop(run_id, None)
status = runner.start_run(...)
frames = list(hub._buffers.get(run_id, []))
stdout_text = "".join(str(frame.get("data", "")) for frame in frames if frame.get("type") == "stdout")
assert "vz-linux-e2e" in stdout_text
```

**Step 4: Run test to verify it passes**

Run: `source ../../.venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_E2E=1 TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<real-base-image> python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_ephemeral_run_smoke -v`
Expected: PASS on a prepared Apple silicon macOS host; SKIP with a concrete reason otherwise.

**Step 5: Commit**

```bash
git add tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "test(vz_linux): add real ephemeral host smoke test"
```

### Task 3: Add the real session reuse and cleanup smoke test
**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Reference: `tldw_Server_API/app/core/Sandbox/service.py`
- Reference: `tldw_Server_API/app/core/Sandbox/orchestrator.py`

**Step 1: Write the failing test**

```python
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_real_session_reuse_smoke(monkeypatch, tmp_path: Path) -> None:
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    service = SandboxService()

    session = service.create_session(
        user_id="e2e-user",
        spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image=base_image, network_policy="deny_all"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"runtime": "vz_linux", "base_image": base_image, "spec_version": "1.0"},
    )

    first = service.start_run_scaffold(
        user_id="e2e-user",
        spec=RunSpec(session_id=session.id, runtime=RuntimeType.vz_linux, base_image=base_image, command=["/bin/echo", "first"], network_policy="deny_all"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"session_id": session.id, "runtime": "vz_linux", "command": ["/bin/echo", "first"]},
    )

    control_after_first = service._orch.get_vz_session_control(session.id)
    second = service.start_run_scaffold(
        user_id="e2e-user",
        spec=RunSpec(session_id=session.id, runtime=RuntimeType.vz_linux, base_image=base_image, command=["/bin/echo", "second"], network_policy="deny_all"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"session_id": session.id, "runtime": "vz_linux", "command": ["/bin/echo", "second"]},
    )

    control_after_second = service._orch.get_vz_session_control(session.id)
    assert first.phase == RunPhase.completed
    assert second.phase == RunPhase.completed
    assert control_after_first["vm_id"] == control_after_second["vm_id"]
```

**Step 2: Run test to verify it fails**

Run: `source ../../.venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_E2E=1 TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<real-base-image> python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_session_reuse_smoke -v`
Expected: FAIL until the test asserts foreground execution, persisted control metadata, and cleanup correctly.

**Step 3: Write minimal implementation**

```python
assert control_after_first is not None
assert control_after_second is not None
assert control_after_first["vm_id"] == control_after_second["vm_id"]

assert service.destroy_session(session.id) is True
assert service._orch.get_vz_session_control(session.id) is None
```

**Step 4: Run test to verify it passes**

Run: `source ../../.venv/bin/activate && TLDW_SANDBOX_VZ_LINUX_E2E=1 TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<real-base-image> python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_session_reuse_smoke -v`
Expected: PASS on a prepared host; SKIP with a concrete reason otherwise.

**Step 5: Commit**

```bash
git add tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py
git commit -m "test(vz_linux): add real session reuse host smoke test"
```

### Task 4: Document the opt-in contract and verify the new module
**Status:** Complete

**Files:**
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `Docs/Plans/2026-03-10-vz-linux-real-host-e2e-implementation-plan.md`
- Test: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

**Step 1: Write the failing doc/test expectation**

```python
def test_vz_linux_real_host_e2e_requires_base_image_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE", raising=False)

    with pytest.raises(pytest.skip.Exception, match="BASE_IMAGE"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
```

**Step 2: Run test to verify it fails**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py::test_vz_linux_real_host_e2e_requires_base_image_env -v`
Expected: FAIL until the skip message and docs are aligned.

**Step 3: Write minimal implementation**

```markdown
- Real host E2E smoke for `vz_linux` is opt-in only.
- Required env:
  - `TLDW_SANDBOX_VZ_LINUX_E2E=1`
  - `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<value>`
- The module requires a real native helper and a prepared Linux guest image.
```

**Step 4: Run test to verify it passes**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q`
Expected: PASS for helper-gating tests and SKIP for real-host tests on unprepared hosts.

**Step 5: Commit**

```bash
git add Docs/Sandbox/macos-runtime-operator-notes.md tldw_Server_API/app/core/Sandbox/README.md tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py Docs/Plans/2026-03-10-vz-linux-real-host-e2e-implementation-plan.md
git commit -m "docs(vz_linux): document real host e2e contract"
```

### Task 5: Final verification and handoff
**Status:** Complete

**Files:**
- Verify: `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`
- Verify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Verify: `tldw_Server_API/app/core/Sandbox/README.md`

**Step 1: Run the focused pytest module**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q`
Expected: PASS for gating tests; host-smoke tests PASS on a prepared host or SKIP with explicit reasons.

**Step 2: Run the existing macOS sandbox smoke slice**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_vz_runtime_macos_host_gated.py tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -q`
Expected: PASS/ SKIP only, with no unexpected failures.

**Step 3: Run Bandit on touched scope**

Run: `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py -f json -o /tmp/bandit_vz_linux_real_host_e2e.json`
Expected: `results: []` or no new findings in changed code.

**Step 4: Update task statuses**

```markdown
**Status:** Complete
```

**Step 5: Commit**

```bash
git add Docs/Plans/2026-03-10-vz-linux-real-host-e2e-implementation-plan.md
git commit -m "docs(vz_linux): finalize real host e2e plan"
```
