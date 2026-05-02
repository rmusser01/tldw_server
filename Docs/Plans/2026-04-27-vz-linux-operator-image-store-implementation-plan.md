# vz_linux Operator Workflow And Image Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable host-side `vz_linux` real-E2E smoke workflow and harden the sandbox image store into a durable bundle manifest catalog.

**Architecture:** Keep the helper as the source of truth for bootability while the Python image store owns local inventory, hashes, provenance, and clone-manifest planning. Add one shell operator command that composes existing build/sign/helper/pytest pieces instead of introducing a second runtime path.

**Tech Stack:** Python dataclasses and JSON manifests, pytest, POSIX shell, SwiftPM helper command invocation, existing macOS VZ helper and `vz_linux` E2E tests.

---

## Stage 1: Durable Image Store

**Goal:** Replace the in-memory image-store stub with a filesystem-backed manifest catalog.

**Success Criteria:** Template registration writes `manifest.json`; new store instances reload existing templates; hashes and provenance are exposed.

**Tests:**

- `tldw_Server_API/tests/sandbox/test_macos_image_store.py`

**Status:** Complete

- [x] Add failing tests for registering a canonical bundle and reloading it from disk.
- [x] Add failing tests for missing artifact paths and duplicate template handling.
- [x] Implement durable manifest write/read in `tldw_Server_API/app/core/Sandbox/image_store.py`.
- [x] Add SHA-256 and size metadata for registered artifacts.
- [x] Preserve compatibility with existing `register_template()` and `prepare_run_clone()` callers.
- [x] Run `python -m pytest tldw_Server_API/tests/sandbox/test_macos_image_store.py -q`.

## Stage 2: Provenance And GC Planning

**Goal:** Make image-store contents auditable and safely cleanable.

**Success Criteria:** Store captures optional `build-info.json`, labels, registration timestamp, and dry-run GC plans without deleting by default.

**Tests:**

- `tldw_Server_API/tests/sandbox/test_macos_image_store.py`

**Status:** Complete

- [x] Add failing tests for provenance capture from bundle `build-info.json`.
- [x] Add failing tests for `list_templates()`, `get_template()`, and GC dry-run planning.
- [x] Implement list/get APIs returning dataclass records.
- [x] Implement `plan_garbage_collection()` with explicit candidate records and no deletion side effects.
- [x] Run `python -m pytest tldw_Server_API/tests/sandbox/test_macos_image_store.py -q`.

## Stage 3: Host Smoke Operator Script

**Goal:** Add one command that runs the real helper and `vz_linux` E2E workflow on a prepared Apple silicon host.

**Success Criteria:** Script validates required inputs, supports dry-run, starts/stops helper safely, and runs helper-daemon plus real E2E pytest commands with the right env.

**Tests:**

- `tools/vz-linux-image/tests/test_host_e2e_smoke_script.py`

**Status:** Complete

- [x] Add failing tests for `--help`, missing `--bundle`, and `--dry-run` command output.
- [x] Create `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`.
- [x] Implement argument parsing and dry-run command printing.
- [x] Implement helper build/sign/start/stop orchestration with shell traps.
- [x] Wire pytest commands for `test_macos_virtualization_helper_daemon_host_gated.py` and `test_vz_linux_real_host_e2e.py`.
- [x] Run `python -m pytest tools/vz-linux-image/tests/test_host_e2e_smoke_script.py -q`.

## Stage 4: Operator Docs And Verification

**Goal:** Document the repeatable workflow and verify the touched scope.

**Success Criteria:** Operator notes explain the smoke command, image-store layout, signing/socket expectations, and deferred launchd/CI work.

**Tests:**

- targeted image-store and script tests
- existing helper client and bundle layout tests
- Bandit on touched Python sandbox files

**Status:** Complete

- [x] Update `Docs/Sandbox/macos-runtime-operator-notes.md`.
- [x] Update `tools/vz-linux-image/README.md`.
- [x] Update `tldw_Server_API/app/core/Sandbox/README.md` if new APIs need mention.
- [x] Run targeted pytest suite for image store, smoke script, helper client, and bundle layout.
- [x] Run Bandit on `tldw_Server_API/app/core/Sandbox/image_store.py`.
- [x] Commit and prepare PR against `dev`.
