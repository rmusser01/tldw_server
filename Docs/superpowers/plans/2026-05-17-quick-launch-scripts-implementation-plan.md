# Quick-Launch Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add no-Docker quick-launch scripts for local single-user self-hosters on Linux, macOS, and Windows.

**Architecture:** Use OS-native wrappers that delegate to the existing local-single setup/start contract. The scripts own shell ergonomics only; setup behavior remains in the existing wizard and package install path.

**Tech Stack:** Bash, macOS `.command`, PowerShell, pytest contract tests, Markdown docs.

---

Task: `TASK-419`

### Task 1: Script Contract Tests

**Files:**
- Create: `tldw_Server_API/tests/Utils/test_quick_launch_scripts.py`

- [x] **Step 1: Write failing contract tests**

Add tests that require `quick-launch.sh`, `quick-launch.command`, and `quick-launch.ps1`; assert they use `local-single`, `uvicorn`, `.venv`, and `tldw_Server_API.cli.wizard.cli`; assert they do not use Docker, Make, `summarize.py`, or print API keys by default.

- [x] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_quick_launch_scripts.py -q`

Expected: FAIL because the scripts do not exist yet.

### Task 2: Quick-Launch Scripts

**Files:**
- Create: `quick-launch.sh`
- Create: `quick-launch.command`
- Create: `quick-launch.ps1`

- [x] **Step 1: Implement minimal launchers**

Create scripts that locate the repo root, create/install `.venv`, run the local-single setup wizard, then start `uvicorn` on `127.0.0.1:8000`.

- [x] **Step 2: Run contract tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_quick_launch_scripts.py -q`

Expected: PASS.

### Task 3: Documentation

**Files:**
- Modify: `README.md`
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`

- [x] **Step 1: Add docs tests if needed**

Extend existing onboarding doc tests only if the new docs wording needs a guard.

- [x] **Step 2: Update docs**

Mention the quick-launch scripts in the no-Docker local path without replacing the canonical Makefile workflow.

- [x] **Step 3: Run docs/onboarding tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_quick_launch_scripts.py tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Docs/test_onboarding_entrypoints.py -q`

Expected: PASS.

### Task 4: Final Verification

**Files:**
- Modify: `backlog/tasks/task-419 - Add-no-Docker-quick-launch-shortcut-scripts.md`

- [x] **Step 1: Run Bandit**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r quick-launch.sh quick-launch.command quick-launch.ps1 tldw_Server_API/tests/Utils/test_quick_launch_scripts.py -f json -o /tmp/bandit_quick_launch_scripts.json`

Expected: exit 0 or documented no-new-findings result.

- [x] **Step 2: Update Backlog task**

Record touched files, verification results, known skips, and final summary in `TASK-419`.
