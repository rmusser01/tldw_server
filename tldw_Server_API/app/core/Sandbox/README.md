# Sandbox

## Current Feature Set

- Purpose: isolated execution with sessions, queued runs, idempotency, artifact streaming, and capability-driven runtime admission.
- Core runtimes:
  - `docker`
  - `firecracker`
  - `lima`
  - `vz_linux`
  - `vz_macos`
  - `seatbelt`
- Capabilities:
  - Create and destroy sessions
  - Queue runs with TTL and capacity limits
  - Stream run events over WebSocket
  - Serve guarded artifact download URLs
  - Expose runtime discovery with preflight reasons, host facts, and supported trust levels

## Runtime Model

- `docker`: general-purpose default runtime with existing interactive support.
- `firecracker`: VM-oriented Linux isolation path.
- `lima`: strict macOS-host VM path with explicit deny-all readiness checks.
- `vz_linux`: Apple `Virtualization.framework` Linux guest runtime on Apple silicon macOS hosts, with real helper-backed ephemeral execution and session VM reuse.
- `vz_macos`: Apple `Virtualization.framework` macOS guest scaffold on Apple silicon macOS hosts.
- `seatbelt`: host-local process isolation runtime for conservative trusted macOS workflows, compatibility-gated by deprecated `sandbox-exec`.

Trust-level rules:

- `untrusted` requires a VM runtime.
- `seatbelt` is rejected for `untrusted`.
- `seatbelt` defaults to `trusted` only; `standard` requires `TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED=1`.
- `vz_linux` and `vz_macos` advertise `trusted`, `standard`, and `untrusted`.

## Technical Notes

- `SandboxOrchestrator` owns session/run lifecycle, queueing, idempotency, and artifact storage.
- `SandboxService` is the integration point for policy admission, runtime preflights, execution dispatch, and runtime discovery.
- Runtime capability snapshots are collected in `runtime_capabilities.py`.
- macOS scaffolding currently includes:
  - a Unix-socket helper client plus protocol models in `macos_virtualization/`
  - frozen helper contract docs in `tools/macos-vz-helper/`
  - a first-party Swift helper daemon that now serves the helper protocol over a real Unix socket
  - migrated `tools/tldw-agent/` guest mode that now serves the first request/response guest protocol over a generic stream transport
  - manifest/image-store contract in `image_store.py`
  - a real helper-backed `vz_linux` runner with ephemeral execution plus session VM reuse
  - a fake-backed `vz_macos` runner
  - a real trusted-only `seatbelt` runner that stages a run-local workspace and launches through `sandbox-exec`

Current limitations:

- `vz_macos` real `Virtualization.framework` execution is not implemented yet.
- `vz_linux` requires helper/template readiness and reports `execution_mode=real` when the helper-backed path is available.
- `vz_macos` still requires helper/template readiness plus `*_FAKE_EXEC=1`; otherwise discovery reports `real_execution_not_implemented`.
- Strict allowlist networking is not implemented for `vz_linux`, `vz_macos`, or `seatbelt`.
- `seatbelt` discovery may be `available=True` while `strict_deny_all_supported=False`; deny-all is a best-effort host policy claim, not a VM-grade guarantee.
- `seatbelt` control files and isolated `HOME`/temp dirs live outside the writable workspace and are removed after each run.
- `seatbelt` real execution still depends on deprecated `sandbox-exec` and may be blocked by an enclosing sandbox even on macOS hosts.
- `vz_linux` supports session VM reuse through persisted VZ session-control metadata; `vz_macos` does not.
- `vz_linux` admin diagnostics now include reconciliation data comparing persisted VZ session-control rows against live helper VM state.
- the helper daemon and guest protocol are now real at the socket/stream level, but the actual `Virtualization.framework` boot driver and vsock transport binding are still incomplete.
- helper-backed template validation now distinguishes canonical bundles from
  raw-disk compatibility mode through `boot_mode` and `validation_strength`.
- `seatbelt` is intentionally conservative and should not be treated as equivalent to a VM boundary.
- Real host `vz_linux` smoke coverage is opt-in through `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py` and requires `TLDW_SANDBOX_VZ_LINUX_E2E=1`, `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<value>`, `TLDW_SANDBOX_MACOS_HELPER_SOCKET=<socket>`, `SANDBOX_ENABLE_EXECUTION=1`, and `SANDBOX_BACKGROUND_EXECUTION=0`.
- Real helper-daemon smoke coverage is opt-in through `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py` and requires `TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1`.

## Operations And Development

- Main API surface: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Main schemas: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- ACP integration: `tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py`
- Recommended validation endpoints:
  - `/api/v1/sandbox/health`
  - `/api/v1/sandbox/runtimes`
  - `/api/v1/sandbox/admin/macos-diagnostics`
  - `/api/v1/sandbox/runs`

`/api/v1/sandbox/runtimes` is the summarized discovery surface used by clients and ACP.
`/api/v1/sandbox/admin/macos-diagnostics` is an admin-only diagnostics surface for
operator troubleshooting and exposes helper/template readiness details that are not
included in the public discovery payload, plus reconciliation data for persisted
`vz_linux` session-control rows versus live helper VM state.

Selected configuration knobs:

- Queue and idempotency:
  - `SANDBOX_QUEUE_MAX_LENGTH`
  - `SANDBOX_QUEUE_TTL_SEC`
  - `SANDBOX_IDEMPOTENCY_TTL_SEC`
- macOS scaffolding:
  - `TLDW_SANDBOX_MACOS_HELPER_SOCKET`
  - `TLDW_SANDBOX_MACOS_HELPER_READY`
  - `TLDW_SANDBOX_MACOS_HELPER_PATH`
  - `TLDW_SANDBOX_VZ_LINUX_AVAILABLE`
  - `TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY`
  - `TLDW_SANDBOX_VZ_LINUX_TEMPLATE_SOURCE`
  - `TLDW_SANDBOX_VZ_MACOS_AVAILABLE`
  - `TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC`
  - `TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY`
  - `TLDW_SANDBOX_VZ_MACOS_TEMPLATE_SOURCE`
  - `TLDW_SANDBOX_SEATBELT_AVAILABLE`
  - `TLDW_SANDBOX_SEATBELT_FAKE_EXEC`
  - `TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED`
