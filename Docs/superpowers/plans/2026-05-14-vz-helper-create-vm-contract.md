# VZ Helper Create VM Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the macOS VZ helper `create_vm` request boundary so malformed or unsupported direct helper requests fail before VM boot.

**Architecture:** Keep Python as the sandbox policy owner and keep the helper as the live VM owner. Add matching Python client and Swift helper request-shape validation for `create_vm`, but do not introduce image-store root allowlisting, vmnet networking, automatic repair, or guest-user rewrites in this slice.

**Tech Stack:** Python helper client, Swift helper daemon, pytest, Swift Testing.

---

## Design Review

- Do not require templates or workspaces to live under a single root yet. Compatibility templates and session workspaces are intentionally flexible, so this slice validates absolute non-NUL paths and rejects symlink paths.
- Do not move policy admission into the helper. The helper should reject unsupported runtime/network values and malformed request shape, while trust-level and user policy decisions stay in Python.
- Do not change helper protocol version. The contract is a stricter interpretation of protocol v1 fields and remains compatible with the Python runner's existing request shape.
- Do not require live host VM smoke for this patch. Portable Python and Swift tests prove request validation; real VZ smoke remains host-gated.

## Stage 1: Python Client Contract

**Goal:** Validate `create_vm` requests before fake or real helper transport.

**Success Criteria:** Invalid Python client requests raise `MacOSVirtualizationHelperFailure` with stable codes and valid runner-shaped requests still serialize.

**Tests:** Focused pytest in `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py`.

**Status:** Complete

## Stage 2: Swift Helper Contract

**Goal:** Validate direct Unix-socket/service `create_vm` requests before VM manager boot and registry mutation.

**Success Criteria:** Invalid helper requests return stable error payloads and do not register a VM.

**Tests:** Swift helper tests in `tools/macos-vz-helper/Tests/HelperServiceVMTests.swift` and `UnixSocketServerTests.swift`.

**Status:** Complete

## Stage 3: Docs And Verification

**Goal:** Document the stricter protocol contract and verify the focused slice.

**Success Criteria:** Protocol/operator docs mention the `create_vm` validation boundary; focused Python/Swift tests, diff check, and Bandit pass or record host/tooling skips.

**Tests:** `pytest`, `swift test`, `git diff --check`, and Bandit over touched Python scope.

**Status:** Complete
