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

`vz_linux` now treats the helper daemon as the source of truth for runtime readiness.
The expected control plane is:

- a local Unix-socket helper daemon
- a successful helper `ping` carrying `protocol_version` and `helper_version`
- helper-backed `validate_host` for runtime availability
- helper-backed `validate_template` for runnable-template truth
- successful template validation now reports `boot_mode` plus
  `validation_strength`, so operators can distinguish canonical bundles from
  raw-disk compatibility mode

Required real-helper config for `vz_linux`:

- `TLDW_SANDBOX_MACOS_HELPER_SOCKET=/path/to/helper.sock`

Test/scaffold env flags still exist, but they are no longer the stable readiness
path for real `vz_linux` execution. They remain relevant for `TEST_MODE` and
other scaffold-only paths:

- `TLDW_SANDBOX_MACOS_HELPER_READY=1`
- `TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY=1`
- `TLDW_SANDBOX_VZ_LINUX_AVAILABLE=1`
- `TLDW_SANDBOX_VZ_MACOS_AVAILABLE=1`
- `TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC=1`
- `TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY=1`

`vz_linux` uses helper-backed truth outside `TEST_MODE` and exposes a real
execution path only when the helper and template are both validated.
`vz_macos` still stays scaffold-only and exposes `real_execution_not_implemented`
in preflight/discovery unless its fake execution flags are enabled.

The helper contract lives under:

- `tldw_Server_API/app/core/Sandbox/macos_virtualization/`
- `tools/macos-vz-helper/`

The repo now contains:

- a real Unix-socket Python client
- a first-party Swift helper daemon subproject in `tools/macos-vz-helper/`
- a first guest-protocol bridge in `tools/tldw-agent/` for guest-mode `tldw-agent`

What is still incomplete is the actual `Virtualization.framework` boot path and
the real vsock transport binding. The protocol and daemon surfaces are now in-repo.

## Template Preparation Flow

Current image handling is manifest-driven rather than real APFS cloning.

There is now an in-repo reference-image/rootfs scaffold for `vz_linux` under:

- `tools/vz-linux-image/`

Its current role is narrow:

- build `tldw-agent-guest`
- install it into a rootfs-like directory
- stage and enable `workspace.mount` plus `tldw-agent-guest.service`
- verify the expected guest binary path before later VM/image automation lands
- provide the guest-mode binary that now serves the first request/response guest protocol

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
- helper readiness, including transport, protocol version, and helper version when reachable
- template readiness for `vz_linux` and `vz_macos`, with optional template source metadata
- per-runtime execution mode and remediation hints
- reconciliation data comparing persisted VZ session rows with live helper VM state

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

Current diagnostics are mixed-mode:

- outside `TEST_MODE`, `vz_linux` helper readiness comes from helper `ping`, `validate_host`, `validate_template`, and `list_vms`
- helper socket discovery for real `vz_linux` uses `TLDW_SANDBOX_MACOS_HELPER_SOCKET`
- helper path metadata is still optional and comes from `TLDW_SANDBOX_MACOS_HELPER_PATH`
- `vz_macos` readiness remains scaffolded through `TLDW_SANDBOX_VZ_MACOS_*`
- fake helper/template env flags still drive test-mode scaffolding
- `vz_linux` reports `execution_mode=real` only when the helper-backed path is reachable and the template validates
- `vz_macos` reports `execution_mode=fake` only when `TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC=1`

## Real Host E2E Smoke

There is now an opt-in pytest smoke module for proving real `vz_linux`
execution on a prepared Apple silicon macOS host:

- `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

Required env for that module:

- `TLDW_SANDBOX_VZ_LINUX_E2E=1`
- `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<guest-template-id-or-base-image>`
- `TLDW_SANDBOX_MACOS_HELPER_SOCKET=/path/to/helper.sock`
- real helper reachability required through helper `ping`
- helper-backed template validation required through `validate_template`

The helper function in that test module also forces:

- `SANDBOX_ENABLE_EXECUTION=1`
- `SANDBOX_BACKGROUND_EXECUTION=0`

That keeps the smoke path synchronous and prevents it from silently using the
fake helper contract. On unprepared hosts, the module should skip with explicit
helper-or-template reasons instead of reporting a fake pass.

## Helper Daemon Smoke

There is also an opt-in cross-language smoke test for the first-party helper daemon:

- `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py`

Required env for that module:

- `TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1`
- optional `TLDW_SANDBOX_MACOS_HELPER_BINARY=/abs/path/to/macos-vz-helper`

By default it looks for the built helper binary at:

- `tools/macos-vz-helper/.build/debug/macos-vz-helper`

What it proves today:

- the real Python helper client can talk to the real Swift daemon over a Unix socket
- helper `ping` returns the frozen protocol and helper versions
- helper-backed `validate_template` works against a real temporary file path and
  can now surface `boot_mode` and `validation_strength` when template resolution
  succeeds
- `create_vm` now goes through the real boot-driver path; concrete failures depend
  on template validity, host readiness, and guest readiness until the canonical
  bundle host smoke is wired
