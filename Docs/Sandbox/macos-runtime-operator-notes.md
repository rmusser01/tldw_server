# macOS Sandbox Runtime Operator Notes

## Scope

These notes cover the current macOS runtime scaffolding for the sandbox subsystem:

- `vz_linux`
- `vz_macos`
- `seatbelt`

This is not yet a guide for shipping the full macOS runtime roadmap. The current implementation exposes runtime identities, policy admission, discovery metadata, helper/image-store contracts, and one real VM-backed path:
- `vz_linux` now supports helper-backed ephemeral execution plus session VM reuse.
- `vz_macos` remains scaffold-only.
- `seatbelt` has a real trusted-workflow subprocess path on compatible macOS hosts.

## Host Assumptions

- Target host platform: Apple silicon macOS
- The VM-oriented runtimes assume Apple `Virtualization.framework`
- `vz_linux` and `vz_macos` preflights fail closed when the host is not macOS or not Apple silicon

## Trust-Level Policy

- `untrusted`:
  - must use a VM runtime
  - `seatbelt` is rejected
- `standard`:
  - allowed on VM runtimes
  - allowed on `seatbelt` only when `TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED=1`
- `trusted`:
  - allowed on all current runtime identities

## Helper And Template Readiness

The VM scaffolding is controlled by explicit readiness signals.

Required env flags today:

- `TLDW_SANDBOX_MACOS_HELPER_READY=1`
- `TLDW_SANDBOX_VZ_LINUX_AVAILABLE=1`
- `TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY=1`
- `TLDW_SANDBOX_VZ_MACOS_AVAILABLE=1`
- `TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC=1`
- `TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY=1`

`vz_linux` uses those readiness signals to expose a real helper-backed execution path.
`vz_macos` still stays unavailable without its fake execution flag and exposes
`real_execution_not_implemented` in preflight/discovery.

The helper contract lives under `tldw_Server_API/app/core/Sandbox/macos_virtualization/`.

The intended production shape is a native signed helper or service that owns `Virtualization.framework` lifecycle operations. The current Python-side helper client is a contract stub with fake transport in test mode.

## Template Preparation Flow

Current image handling is manifest-driven rather than real APFS cloning.

Expected operator flow later:

1. Prepare a sealed template image per runtime family.
2. Register the template in the sandbox image store.
3. Create run-scoped clone manifests from that template.
4. Hand clone metadata to the native helper for VM boot.
5. Destroy the run-scoped clone state after completion.

Today, the image store implements template registration plus deterministic run-clone manifests only.

## Networking

- `deny_all` is the intended strict baseline for the new macOS runtimes.
- `seatbelt` only offers a best-effort deny-all claim via seatbelt policy; discovery should not be read as VM-grade or firewall-backed network isolation.
- `strict_allowlist_not_supported` is still the expected result for:
  - `vz_linux`
  - `vz_macos`
  - `seatbelt`
- `vz_linux` session-mode reuse exists now and reuses a persisted VM for later commands in the same sandbox session.
- `vz_macos` still has no warm-session optimization.

## Discovery And ACP

`/api/v1/sandbox/runtimes` now exposes:

- runtime availability
- preflight reasons
- supported trust levels
- enforcement readiness
- host facts

`/api/v1/sandbox/admin/macos-diagnostics` is the operator-focused companion surface.
It is admin-only and returns:

- detailed host readiness, including macOS version
- helper readiness, with optional configured path and transport metadata
- template readiness for `vz_linux` and `vz_macos`, with optional template source metadata
- per-runtime execution mode and remediation hints

Use the admin endpoint when you are validating host setup or trying to explain why a
runtime is unavailable. Use `/api/v1/sandbox/runtimes` for client-facing discovery;
that payload stays summarized and does not expose helper/template internals.

ACP sandbox session creation now performs runtime preflight validation before calling the sandbox service, and converts failures into `ACPResponseError` instead of leaking raw sandbox exceptions.

## Current Limits

- `vz_linux` real guest command execution is available behind helper/template readiness on Apple silicon macOS hosts
- `vz_linux` session VM reuse persists VM control metadata and reuses the same VM for later sandbox-session runs
- `vz_macos` still has no real guest command execution
- `seatbelt` real execution is available for trusted workflows when `sandbox-exec` is present and not blocked by an enclosing sandbox
- `sandbox-exec` is deprecated and should be treated as a compatibility-gated bridge, not the long-term macOS isolation foundation
- `seatbelt` availability depends on `sandbox-exec` existing on the host, but its summarized discovery payload still keeps `strict_deny_all_supported=false`
- `seatbelt` runner-owned control files plus isolated `HOME` and temp directories are created outside the writable workspace and removed after each run
- No APFS clone execution path yet
- No allowlist networking for the new macOS runtimes
- No `vz_macos` warm-session VM reuse yet

Current diagnostics are still env-driven scaffolding:

- helper readiness is gated by `TLDW_SANDBOX_MACOS_HELPER_READY`
- helper path metadata is optional and comes from `TLDW_SANDBOX_MACOS_HELPER_PATH`
- template readiness is gated by `TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY` and `TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY`
- template source metadata is optional and comes from `TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE` and `TLDW_SANDBOX_VZ_MACOS_TEMPLATE_SOURCE`
- `vz_linux` reports `execution_mode=real` when helper/template readiness succeeds
- `vz_macos` reports `execution_mode=fake` only when `TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC=1`

## Real Host E2E Smoke

There is now an opt-in pytest smoke module for proving real `vz_linux`
execution on a prepared Apple silicon macOS host:

- `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

Required env for that module:

- `TLDW_SANDBOX_VZ_LINUX_E2E=1`
- `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<guest-template-id-or-base-image>`
- real helper/template readiness required by normal `vz_linux` preflight

The helper function in that test module also forces:

- `SANDBOX_ENABLE_EXECUTION=1`
- `SANDBOX_BACKGROUND_EXECUTION=0`

That keeps the smoke path synchronous and prevents it from silently using the
fake helper contract. On unprepared hosts, the module should skip with explicit
reasons instead of reporting a fake pass.
