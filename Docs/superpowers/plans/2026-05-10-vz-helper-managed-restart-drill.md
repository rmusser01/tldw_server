# VZ Helper Managed Restart Drill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an operator-owned `vz-helperctl.py restart-drill` command that validates managed helper stop/start/status recovery without launchd or host reboot automation.

**Architecture:** Keep the drill inside the existing `vz-helperctl.py` lifecycle wrapper so it reuses current socket, pid-file, log-directory, serial-directory, ping, and protocol checks. The drill only acts on helpers controlled by the managed pid-file path; unmanaged launchd/helper crash classes remain documented manual procedures.

**Tech Stack:** Python 3 CLI, existing `tools/macos-vz-helper/Tests/test_vz_helperctl.py` pytest coverage, existing sandbox operator docs.

---

### Task 1: Add Portable Restart-Drill Behavior

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Write failing tests for the drill helper**

Add tests proving a running managed helper is stopped, started, and status-checked after restart; an absent helper fails without mutation; and start failure is reported without running post-status.

- [x] **Step 2: Verify red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -k 'restart_drill' -q`

Expected: failures because restart-drill behavior does not exist yet.

- [x] **Step 3: Implement the minimal restart-drill helper**

Add a focused function that collects pre-status, requires `process=helper_pid_running` and successful `ping`, calls `stop_helper`, calls `start_helper`, then collects post-status and reports a final `restart_drill` result.

- [x] **Step 4: Verify green**

Run the same focused pytest command and confirm all restart-drill tests pass.

### Task 2: Add CLI Entry Point And Docs

**Files:**
- Modify: `tools/macos-vz-helper/scripts/vz-helperctl.py`
- Modify: `tools/macos-vz-helper/README.md`
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Test: `tools/macos-vz-helper/Tests/test_vz_helperctl.py`

- [x] **Step 1: Write failing CLI test**

Add coverage that `restart-drill` accepts the same helper/socket/pid/log/entitlement options as status/start/stop, passes them into the drill helper, and supports JSON output.

- [x] **Step 2: Verify red**

Run the focused restart-drill pytest selection and confirm the CLI test fails because the subcommand is not wired.

- [x] **Step 3: Implement CLI and document operator usage**

Add the `restart-drill` parser, print named results through existing `_print_results`, return non-zero when any result fails, and update docs to position it before launchd or host reboot validation.

- [x] **Step 4: Verify focused tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q`

### Task 3: Final Verification And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-206 - Add-managed-VZ-helper-restart-status-drill.md`

- [x] **Step 1: Run static and security checks**

Run `git diff --check` and Bandit on the touched helper script/tests with `B101` skipped for tests.

- [x] **Step 2: Update TASK-206**

Check completed acceptance criteria, record verification, and add a final summary.

- [x] **Step 3: Commit**

Stage the plan, implementation, docs, tests, and task file. Commit with a message such as `feat(sandbox): add VZ helper restart drill`.
