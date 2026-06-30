# VZ Helper Launchd Operator Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit `vz-helperctl.py launchd` operator commands for inspecting and invoking helper LaunchAgent lifecycle actions.

**Architecture:** Keep launchd support inside the existing helper lifecycle wrapper and reuse current plist, path-hardening, and dry-run conventions. Mutating launchctl actions must be explicit subcommands, never side effects of `plist`, `status`, `smoke`, or server startup.

**Tech Stack:** Python 3 CLI, macOS `launchctl` command construction, existing pytest coverage in `tools/macos-vz-helper/Tests/test_vz_helperctl.py`.

---

### Task 1: Launchd Command Model

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Write failing tests for launchd target and argv construction**

Cover `gui/<uid>`, `gui/<uid>/<label>`, `bootstrap`, `bootout`, `kickstart`, and `status`/`print` command shapes.

- [x] **Step 2: Run focused test and verify red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'launchd' -q`

- [x] **Step 3: Implement minimal command construction helpers**

Add small helpers for launchd domain/target/argv and no subprocess execution yet.

- [x] **Step 4: Run focused test and verify green**

Run the same focused pytest command.

### Task 2: Launchd CLI Execution

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Write failing tests for CLI behavior**

Cover dry-run printing, bootstrap missing plist failure, explicit `--write-plist --create-dirs`, and JSON result output.

- [x] **Step 2: Run focused test and verify red**

Run the focused launchd pytest selection.

- [x] **Step 3: Implement `launchd` subcommand**

Add `launchd {status,bootstrap,kickstart,bootout}` with `--dry-run`, `--write-plist`, `--create-dirs`, helper/socket/log/plist/label/uid options, and an injectable command runner for tests.

- [x] **Step 4: Run full helperctl tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q`

### Task 3: Docs And Verification

**Files:**
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`
- Modify: `backlog/tasks/task-318 - Add-explicit-VZ-helper-launchd-operator-commands.md`

- [x] **Step 1: Update operator docs**

Document the explicit launchd flow, dry-run first expectation, and host reboot out-of-scope boundary.

- [x] **Step 2: Run verification**

Run helperctl tests, `git diff --check`, and Bandit on touched helper script/tests with expected test-file skips.

- [x] **Step 3: Close task and commit**

Update `TASK-318`, stage all touched files, and commit.
