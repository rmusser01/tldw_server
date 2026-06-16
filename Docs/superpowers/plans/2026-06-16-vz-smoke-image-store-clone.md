# VZ Smoke Image-Store Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VZ Linux host smoke workflow use an image-store-backed disposable run bundle instead of mutating the operator-provided source bundle.

**Architecture:** Add a focused Python materializer CLI that uses `SandboxImageStore` to register the source bundle, prepare a per-run clone manifest, and materialize the run bundle. Update `run-host-e2e-smoke.sh` to treat `--bundle` as `SOURCE_BUNDLE_PATH` and pass the disposable run bundle to helper bundle smoke, real host smoke, and optional failure drills.

**Tech Stack:** Bash, Python 3.11, pytest, existing `SandboxImageStore`, macOS `clonefile(2)` via `ctypes` with `shutil.copy2()` fallback.

---

### Task 1: Materializer CLI

**Files:**
- Create: `tools/vz-linux-image/scripts/prepare-smoke-bundle.py`
- Test: `tools/vz-linux-image/tests/test_prepare_smoke_bundle.py`

- [ ] **Step 1: Write failing tests**

Add tests that create a minimal source bundle, run the CLI with `--source-bundle`, `--store-root`, and `--run-id`, then assert:

- stdout is the run bundle directory
- `runs/<run-id>/manifest.json` exists
- `kernel`, `rootfs.img`, and optional `initrd` exist in the run bundle
- source `rootfs.img` contents and mtime are unchanged after mutating the run copy
- metadata files `manifest.json` and `build-info.json` are copied when present

- [ ] **Step 2: Run the focused new tests and verify they fail**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py -q`

Expected: failure because `prepare-smoke-bundle.py` does not exist yet.

- [ ] **Step 3: Implement the materializer**

Implement an executable Python script that imports `SandboxImageStore`, registers the bundle, prepares a run clone, materializes clone items with macOS `clonefile(2)` fallback to `shutil.copy2()`, copies metadata files, chmods private directories where possible, and prints the run bundle path.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py -q`

Expected: all new materializer tests pass.

### Task 2: Smoke Wrapper Integration

**Files:**
- Modify: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
- Modify: `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`

- [ ] **Step 1: Write failing shell-wrapper tests**

Add/update tests proving:

- `--help` explains that `--bundle` is the canonical source bundle and VM stages use a disposable run bundle
- dry-run prints a materializer command
- dry-run helper/pytest commands use the run bundle path rather than the source bundle path
- a fake real run materializes a run bundle and leaves the source rootfs unchanged

- [ ] **Step 2: Run the focused shell-wrapper tests and verify failure**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q`

Expected: failure because the wrapper still passes the source bundle directly.

- [ ] **Step 3: Implement wrapper integration**

Add `SOURCE_BUNDLE_PATH`, `IMAGE_STORE_ROOT`, `SMOKE_RUN_ID`, and `PREPARE_SMOKE_BUNDLE_SCRIPT`. Validate the source bundle, prepare or dry-run the disposable run bundle before helper smoke, and pass the run bundle to all VM-executing stages.

- [ ] **Step 4: Run shell-wrapper tests and verify pass**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q`

Expected: all shell-wrapper tests pass.

### Task 3: Docs And Verification

**Files:**
- Modify: `tools/vz-linux-image/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `Docs/Sandbox/vz-linux-prepared-host-evidence.md`
- Modify: `backlog/tasks/task-2365 - Use-image-store-backed-disposable-clones-for-VZ-smoke.md`

- [ ] **Step 1: Update docs**

Document that `--bundle` is a source bundle, smoke creates a disposable image-store run bundle, and evidence should record source and run bundle hashes separately.

- [ ] **Step 2: Run verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_prepare_smoke_bundle.py tools/vz-linux-image/tests/test_host_e2e_smoke_script.py tldw_Server_API/tests/unit/test_sandbox_image_store.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
python -m bandit -r tools/vz-linux-image/scripts/prepare-smoke-bundle.py tldw_Server_API/app/core/Sandbox/image_store.py -f json -o /tmp/bandit_vz_smoke_image_store_clone.json
git diff --check
```

Expected: pytest, shell syntax, Bandit, and diff checks pass or any pre-existing/irrelevant warning is explicitly documented.

- [ ] **Step 3: Finalize task and commit**

Update `TASK-2365` with final summary, verification, known skips, and DoD. Commit with a message such as `feat(sandbox): smoke from disposable VZ image clones`.
