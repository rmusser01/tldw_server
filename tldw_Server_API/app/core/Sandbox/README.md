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
- `vz_linux`: Apple `Virtualization.framework` Linux guest runtime on prepared Apple silicon macOS hosts, with real helper-backed boot, guest command execution, and session VM reuse when helper/template readiness passes.
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
- Subsystem-wide sandbox rules now live in `Docs/Sandbox/sandbox-architecture-doctrine.md`.
  Future runtime work should reference that doctrine for:
  - layered readiness and source-of-truth ownership
  - canonical vs compatibility artifact paths
  - audit and provenance expectations
  - lifecycle and reconciliation rules
- macOS scaffolding currently includes:
  - a Unix-socket helper client plus protocol models in `macos_virtualization/`
  - frozen helper contract docs in `tools/macos-vz-helper/`
  - a first-party Swift helper daemon that now serves the helper protocol over a real Unix socket
  - migrated `tools/tldw-agent/` guest mode that now serves the first request/response guest protocol over a generic stream transport
  - a filesystem-backed manifest/image-store contract in `image_store.py`
  - a real helper-backed `vz_linux` runner with ephemeral execution plus session VM reuse
  - a fake-backed `vz_macos` runner
  - a real trusted-only `seatbelt` runner that stages a run-local workspace and launches through `sandbox-exec`

Current limitations:

- `vz_macos` real `Virtualization.framework` execution is not implemented yet.
- `vz_linux` requires helper/template readiness and reports `execution_mode=real` when the helper-backed boot and guest execution path is available.
- `vz_macos` still requires helper/template readiness plus `*_FAKE_EXEC=1`; otherwise discovery reports `real_execution_not_implemented`.
- Strict allowlist networking is not implemented for `vz_linux`, `vz_macos`, or `seatbelt`.
- `vz_linux` VM creation is fail-closed at both Python admission and helper
  protocol layers: only `network_policy=deny_all` is accepted, and the helper
  records the accepted policy in VM metadata/status details. The current
  Virtualization.framework configuration does not attach a network device.
- `vz_linux` passes `SANDBOX_MAX_LOG_BYTES` to helper `exec_guest` as
  `max_output_bytes`, clamped to the helper protocol ceiling of 256 MiB, and
  also uses the effective cap when publishing stdout/stderr frames. Rebuilt
  images with the updated `tldw-agent-guest` enforce the cap inside the guest
  and terminate noisy commands when observed output exceeds it. The helper still
  applies host-side response capping as defense in depth and fallback for older
  images. Guest-side streaming remains follow-up work.
- `vz_linux` helper VM status details include guest-agent readiness metadata
  when available: `guest_version`, `guest_workspace_root`,
  `guest_capabilities_known`, and `guest_capabilities`. These fields are
  diagnostic only; old images that omit capabilities remain compatible and are
  reported as capability-unknown.
- `vz_linux` artifact capture is bounded by `SANDBOX_MAX_ARTIFACT_FILE_BYTES`
  and `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES`. Oversized or over-budget artifacts
  are skipped without failing an otherwise successful run, and aggregate skip
  counters are recorded in `resource_usage` and audit metadata without raw
  artifact paths.
- `seatbelt` discovery may be `available=True` while `strict_deny_all_supported=False`; deny-all is a best-effort host policy claim, not a VM-grade guarantee.
- `seatbelt` control files and isolated `HOME`/temp dirs live outside the writable workspace and are removed after each run.
- `seatbelt` real execution still depends on deprecated `sandbox-exec` and may be blocked by an enclosing sandbox even on macOS hosts.
- `vz_linux` supports session VM reuse through persisted VZ session-control metadata; `vz_macos` does not.
- `vz_linux` admin diagnostics include reconciliation data comparing persisted VZ session-control rows against live helper VM state.
- `vz_linux` admin diagnostics also include a read-only image-store block that correlates persisted run manifests and dry-run GC classifications with reconciliation/helper state.
- `vz_linux` also exposes `GET /api/v1/sandbox/admin/macos-image-store/cleanup-plan` for read-only GC action planning and `POST /api/v1/sandbox/admin/macos-image-store/cleanup` for explicit admin cleanup, which defaults to `dry_run=true`; unfiltered mutating cleanup requires `confirm_all=true`.
- `vz_linux` admin diagnostics now also project an additive
  `startup_warning_summary` field from the app-owned startup warning registry;
  low-level diagnostics collection remains app-agnostic.
- `vz_linux` repair is explicit and admin-only through `POST /api/v1/sandbox/admin/macos-reconciliation/repair`; diagnostics do not mutate state.
- `vz_linux` repair defaults to dry-run, skips active sessions, can delete stale or unhealthy inactive persisted session-control rows when requested, and can terminate orphan helper VMs only when `terminate_orphaned_vms=true` is explicitly requested, helper metadata proves `owner=tldw` and `runtime=vz_linux`, and the ownership record remains eligibility-complete. Eligibility-complete means `run_id` and `created_at` are present, and `session_id` is also present when `session_mode=true`.
- `vz_linux` orphan VM diagnostics split live unreferenced helper VMs into `owned_orphaned_vm`, `unknown_orphaned_vm`, and `foreign_orphaned_vm`. Only ownership-eligible `owned_orphaned_vm` records can be terminated automatically; unknown, foreign, and legacy generic orphan records are reported but skipped by automated repair.
- Helper unavailable or protocol mismatch conditions fail closed and block mutating repair.
- Startup warning policy is narrower: helper protocol mismatch blocks startup,
  while helper unavailable and reconciliation drift remain warnings only.
- Orphan VM termination is not automatic repair behavior; operators should inspect the dry-run plan before running mutating repair.
- The generic admin startup warning endpoint is `GET /api/v1/admin/startup-warnings`.
  It exposes current-process warning records only; there is no
  cross-process aggregation or persistence in this slice.
- `tools/macos-vz-helper/scripts/vz-helperctl.py` is the preferred operator helper lifecycle command for `check`, `build`, `sign`, `start`, `status`, `stop`, `plist`, and `smoke`; it can generate launchd plist scaffolding but does not install or load services automatically.
- helper-backed template validation now distinguishes canonical bundles from
  raw-disk compatibility mode through `boot_mode` and `validation_strength`.
- `SandboxImageStore` persists template manifests under
  `<root>/templates/<runtime>/<template>/manifest.json`, records artifact
  size/SHA-256 metadata plus optional bundle provenance, persists run clone
  planning manifests under `<root>/runs/<run_id>/manifest.json`, and exposes
  dry-run run-directory GC planning with candidate reasons that distinguish
  planning-only manifests, fully materialized inactive runs, and legacy run
  directories without a persisted manifest.
- When `TLDW_SANDBOX_IMAGE_STORE_ROOT` is configured, `vz_linux` can also
  resolve `spec.base_image` as a registered image-store `template_id` instead
  of a raw path, provided the template record has a stored `source_path`.
- `seatbelt` is intentionally conservative and should not be treated as equivalent to a VM boundary.
- Real host `vz_linux` smoke coverage should normally be run through
  `tools/macos-vz-helper/scripts/vz-helperctl.py smoke`; the lower-level
  fallback remains `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`. The
  underlying pytest module remains opt-in through
  `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py` and requires
  `TLDW_SANDBOX_VZ_LINUX_E2E=1`,
  `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<value>`,
  `TLDW_SANDBOX_IMAGE_STORE_ROOT=<path>` when that value is a registered
  template id,
  `TLDW_SANDBOX_MACOS_HELPER_SOCKET=<socket>`, `SANDBOX_ENABLE_EXECUTION=1`,
  and `SANDBOX_BACKGROUND_EXECUTION=0`.
- Real helper-daemon smoke coverage is opt-in through `tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py` and requires `TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1`.

## Operations And Development

- Main API surface: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Main schemas: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- ACP integration: `tldw_Server_API/app/core/Agent_Client_Protocol/sandbox_runner_client.py`
- Recommended validation endpoints:
  - `/api/v1/sandbox/health`
  - `/api/v1/sandbox/runtimes`
  - `/api/v1/admin/startup-warnings`
  - `/api/v1/sandbox/admin/macos-diagnostics`
  - `/api/v1/sandbox/admin/macos-image-store/cleanup-plan`
  - `POST /api/v1/sandbox/admin/macos-image-store/cleanup`
  - `POST /api/v1/sandbox/admin/macos-reconciliation/repair`
  - `/api/v1/sandbox/runs`

`/api/v1/sandbox/runtimes` is the summarized discovery surface used by clients and ACP.
`/api/v1/sandbox/admin/macos-diagnostics` is an admin-only diagnostics surface for
operator troubleshooting and exposes helper/template readiness details that are not
included in the public discovery payload, plus reconciliation data for persisted
`vz_linux` session-control rows versus live helper VM state, and image-store
correlation for persisted run manifests and dry-run GC candidates. It is
read-only and now includes a compact `startup_warning_summary` field projected from the
current-process startup warning registry.
`/api/v1/admin/startup-warnings` is the generic admin-only companion surface for
the same startup records and returns full current-process warning items plus
grouped counts.
`POST /api/v1/sandbox/admin/macos-reconciliation/repair` is the separate
admin-only repair surface. Repair defaults to dry-run, skips active sessions,
can delete stale or unhealthy inactive persisted session-control rows when
requested, and can terminate orphan helper VMs only when
`terminate_orphaned_vms=true` is explicitly requested and the reconciliation item
is ownership-eligible. Ownership-eligible means helper metadata proves
`owner=tldw`, `runtime=vz_linux`, a non-empty `run_id`, a non-empty
`created_at`, and a non-empty `session_id` when `session_mode=true`.

Selected configuration knobs:

- Queue and idempotency:
  - `SANDBOX_QUEUE_MAX_LENGTH`
  - `SANDBOX_QUEUE_TTL_SEC`
  - `SANDBOX_IDEMPOTENCY_TTL_SEC`
- Output and artifacts:
  - `SANDBOX_MAX_LOG_BYTES`
  - `SANDBOX_MAX_ARTIFACT_FILE_BYTES`
  - `SANDBOX_MAX_ARTIFACT_TOTAL_BYTES`
- macOS scaffolding:
  - `TLDW_SANDBOX_MACOS_HELPER_SOCKET`
  - `TLDW_SANDBOX_MACOS_HELPER_READY`
  - `TLDW_SANDBOX_MACOS_HELPER_PATH`
  - `TLDW_SANDBOX_IMAGE_STORE_ROOT`
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
