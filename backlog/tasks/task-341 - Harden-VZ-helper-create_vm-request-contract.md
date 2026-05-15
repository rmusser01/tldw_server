---
id: TASK-341
title: Harden VZ helper create_vm request contract
status: Done
assignee: []
created_date: '2026-05-14 16:52'
updated_date: '2026-05-14 17:28'
labels:
  - sandbox
  - vz_linux
  - security
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-security-policy-matrix.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - tools/macos-vz-helper/PROTOCOL.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused Phase 2 security hardening slice for the macOS VZ Linux helper. The helper and Python client should reject malformed or unsupported create_vm requests before VM boot while preserving Python as the source of policy admission and the helper as live VM truth. Keep the scope narrow: validate request shape, runtime/network support, bounded metadata, timeout, and basic path safety; do not add vmnet, OCI boot changes, automatic repair, or broader guest-user rewrites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Python helper client validates create_vm request shape before fake or real transport and returns stable helper failure codes for invalid requests.
- [x] #2 Swift helper validates create_vm requests before VM manager boot and maps denials to stable error payloads without registering a VM.
- [x] #3 Validation covers unsupported runtime/network policy, missing or invalid vm id, bounded text metadata, timeout range, and absolute non-NUL template/workspace/run-manifest paths with symlink rejection.
- [x] #4 Existing VZ Linux runner create_vm calls remain compatible and include required metadata for the hardened contract.
- [x] #5 Helper protocol/operator docs describe the create_vm request contract and its non-goals.
- [x] #6 Focused Python and Swift tests cover accepted requests and rejected malformed/unsupported requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-14-vz-helper-create-vm-contract.md

Stages:
1. Python helper client create_vm request validation with focused pytest red/green.
2. Swift helper create_vm validation before VM manager boot with Swift Testing red/green.
3. Protocol/operator docs plus verification: focused tests, git diff --check, Bandit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Working in /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vz-helper-create-vm-contract on branch codex/vz-helper-create-vm-contract from origin/dev. Main checkout is dirty/diverged and intentionally untouched.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_client.py tldw_Server_API/tests/sandbox/test_macos_helper_client.py tldw_Server_API/tests/sandbox/test_vz_linux_runner.py -q` passed: 69 passed, 2 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 101 passed, 1 skipped, 2 warnings.
- `env CLANG_MODULE_CACHE_PATH=/private/tmp/tldw-swift-module-cache swift test --package-path tools/macos-vz-helper` passed: 83 tests passed.
- `git diff --check` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/macos_virtualization/helper_client.py -f json -o /tmp/bandit_vz_helper_create_vm_contract.json` passed with 0 findings.

Known notes:
- SwiftPM still warns that `tools/macos-vz-helper/Tests/test_vz_helperctl.py` is an unhandled package resource; this is pre-existing because the package directory also hosts pytest tests.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the VZ Linux helper `create_vm` boundary by adding matching Python client and Swift daemon validation before fake transport, socket transport, VM boot, or registry mutation. The contract now rejects unsupported runtimes/network policy, invalid VM ids, malformed non-string request fields, unsafe path shape, symlink paths, oversized metadata, and invalid timeouts with stable helper error codes. Protocol/operator docs and focused tests were updated to make the boundary explicit.
<!-- SECTION:FINAL_SUMMARY:END -->
