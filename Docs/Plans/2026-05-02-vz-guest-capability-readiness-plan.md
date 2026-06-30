# VZ Guest Capability Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Surface `vz_linux` guest-agent version, workspace root, and optional capability metadata in helper VM readiness/status responses.

**Architecture:** Extend the existing guest handshake with optional capabilities while preserving protocol version `1` and old guest compatibility. Store the parsed metadata in the Swift helper session/registry path and expose it through the existing VM `details` maps used by `create_vm`, `get_vm_status`, and `list_vms`. Treat the metadata as diagnostics/readiness information only; do not use it for security enforcement in this slice.

**Tech Stack:** Swift helper daemon, Go `tools/tldw-agent` guest protocol, existing helper protocol JSON details, Swift Testing, Go tests.

---

## Stage 1: Lock Swift Session Metadata Behavior
**Goal:** Prove the helper records optional guest capabilities and handles old guests as unknown.
**Success Criteria:** `VSockSessionManager` exposes parsed guest version/workspace/capability state after handshake, and missing or malformed capabilities do not fail readiness.
**Tests:** `swift test --package-path tools/macos-vz-helper --filter VSockSessionManagerTests`
**Status:** Complete

- [x] Add failing tests for capability-present, capability-missing, and malformed capability handshakes.
- [x] Implement a small `GuestAgentInfo` model and session/manager accessors.
- [x] Run the focused Swift session tests until green.

## Stage 2: Surface Metadata Through VM Status
**Goal:** Include guest readiness metadata in VM `details` without changing the helper protocol shape.
**Success Criteria:** `create_vm`, `get_vm_status`, and `list_vms` include deterministic `guest_*` detail keys when guest info is known, and report capability state as unknown for old guests.
**Tests:** `swift test --package-path tools/macos-vz-helper --filter HelperServiceVMTests`
**Status:** Complete

- [x] Add failing helper service tests for guest details in create/status/list replies.
- [x] Persist guest info in `VMRecord` via `VZLinuxVMManager` after readiness.
- [x] Format details with stable strings: `guest_version`, `guest_workspace_root`, `guest_capabilities_known`, and `guest_capabilities`.

## Stage 3: Advertise Capabilities From the Guest Agent
**Goal:** Have new `tldw-agent` builds send explicit capabilities while old images remain compatible.
**Success Criteria:** The initial guest handshake includes a stable capability list; existing host validation still accepts older handshakes.
**Tests:** `cd tools/tldw-agent && go test ./internal/guest`
**Status:** Complete

- [x] Add failing Go test coverage for handshake capability emission.
- [x] Add `capabilities` to `HandshakeRequest`.
- [x] Send a sorted V1 capability list from `VSockClient.sendHandshake`.

## Stage 4: Document and Verify
**Goal:** Document the optional metadata contract and run focused verification.
**Success Criteria:** Protocol docs mention optional capabilities and status details; focused Swift and Go tests pass; whitespace/security checks are clean.
**Tests:** `git diff --check`; focused Swift/Go test commands above.
**Status:** Complete

- [x] Update guest protocol and helper docs.
- [x] Run focused tests and `git diff --check`.
- [x] Run Bandit on touched Python paths only if Python files are changed.
