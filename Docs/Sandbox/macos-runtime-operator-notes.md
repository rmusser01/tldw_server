# macOS Sandbox Runtime Operator Notes

## Scope

These notes cover the current macOS runtime scaffolding for the sandbox subsystem:

- `vz_linux`
- `vz_macos`
- `seatbelt`

This is not yet a guide for shipping the full macOS runtime roadmap. The current implementation exposes runtime identities, policy admission, discovery metadata, helper/image-store contracts, and one real VM-backed path:
- `vz_linux` supports helper-backed boot, guest command execution, ephemeral execution, and session VM reuse on prepared Apple silicon macOS hosts.
- `vz_macos` remains scaffold-only.
- `seatbelt` has a real trusted-workflow subprocess path on compatible macOS hosts.

## Host Assumptions

- Target host platform: Apple silicon macOS
- The VM-oriented runtimes assume Apple `Virtualization.framework`
- `vz_linux` and `vz_macos` preflights fail closed when the host is not macOS or not Apple silicon

## Apple Container Prior Art

Apple's [`container`](https://github.com/apple/container) CLI is relevant prior
art for `vz_linux` because it runs Linux containers as lightweight per-workload
VMs on Apple silicon and uses the same family of macOS building blocks this
subsystem targets:
`Virtualization.framework`, vmnet, `launchd`, helper services, unified logging,
OCI-compatible images, and guest control over vsock.

It is not an operator prerequisite for `tldw_server`. Operators should not need
to install or run Apple's `container` CLI for the current `vz_linux` path.
Near-term work should instead use it as a comparison point for:

- helper lifecycle and service decomposition
- image-store layout, digests, provenance, and future OCI compatibility
- optimized Linux kernel/rootfs choices for faster VM startup
- vmnet-backed networking only when a reviewed network policy needs it
- guest init/agent readiness contracts over vsock

Direct reuse of Apple's lower-level
[`containerization`](https://github.com/apple/containerization) Swift package
should be decided in a focused implementation plan, not introduced
incidentally. Any adoption must preserve the repo-owned helper protocol, admin
diagnostics, fail-closed policy admission, the intended macOS support window,
and the separate `seatbelt` and `vz_macos` runtime tracks.

See `Docs/Design/2026-05-02-apple-containerization-evaluation.md` for the
current adopt/defer/reject evaluation before changing image-store, helper,
networking, or guest-agent implementation.

See `Docs/Sandbox/sandbox-security-policy-matrix.md` before changing runtime
trust eligibility, network-policy semantics, workspace mounts, artifact
exposure, helper/request allowlisting, or audit behavior.

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
boot and guest execution path only when the helper and template are both
validated.
`vz_macos` still stays scaffold-only and exposes `real_execution_not_implemented`
in preflight/discovery unless its fake execution flags are enabled.

The helper contract lives under:

- `tldw_Server_API/app/core/Sandbox/macos_virtualization/`
- `tools/macos-vz-helper/`

The repo now contains:

- a real Unix-socket Python client
- a first-party Swift helper daemon subproject in `tools/macos-vz-helper/`
- a first guest-protocol bridge in `tools/tldw-agent/` for guest-mode `tldw-agent`

The `vz_linux` helper-backed boot and guest execution path is real when host,
helper, guest agent, and canonical bundle or compatibility template validation
all pass. `vz_macos` remains scaffold-only.

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
- expose a thin Linux-container wrapper around the same builder flow when
  direct Linux build tooling is not available on the host

Expected operator flow:

1. Prepare a sealed template image per runtime family.
2. Register the canonical bundle, or a weaker compatibility template, in the sandbox image store.
3. Create run-scoped clone manifests from that template.
4. Hand clone metadata to the native helper for VM boot.
5. Destroy the run-scoped clone state after completion.

The image store is now filesystem-backed at:

```text
<image-store-root>/
  templates/
    <runtime>/
      <template-name>/
        manifest.json
  runs/
    <run-id>/
      manifest.json
```

Template manifests include artifact paths, artifact size, SHA-256 hashes,
labels, registration time, source path, and optional build provenance from a
bundle `build-info.json`. Run clone manifests now persist deterministic
per-run clone planning under `runs/<run_id>/manifest.json`, so later helper
integration and dry-run GC can reason over store-owned run metadata. The store
remains an inventory and planning layer, not the bootability source of truth;
helper `validate_template` still owns that.

Minimal Python registration example:

```python
from tldw_Server_API.app.core.Sandbox.image_store import SandboxImageStore

store = SandboxImageStore(root_path="/var/lib/tldw/sandbox-images")
template_id = store.register_bundle(
    runtime="vz_linux",
    template_name="debian-bookworm-arm64",
    bundle_path="/var/lib/tldw/vz-linux/debian-bookworm-arm64/bundle",
    labels={"suite": "bookworm", "profile": "minimal"},
)
record = store.get_template(template_id)
gc_plan = store.plan_garbage_collection(active_run_ids=set())
```

`plan_garbage_collection()` is dry-run only. It returns candidate records for
inactive run directories and does not delete files. Candidate reasons now
differentiate:

- `planning_only_run_manifest`: a persisted run manifest exists but no clone
  artifacts were materialized under that run directory yet
- `inactive_run`: a persisted run manifest exists and the run directory also
  contains clone/runtime files
- `legacy_run_directory`: files exist under `runs/<run_id>/` without a
  persisted run manifest

The admin image-store surfaces now split planning from mutation:

- `GET /api/v1/sandbox/admin/macos-image-store/cleanup-plan`: read-only
  candidate planning derived from diagnostics correlation
- `POST /api/v1/sandbox/admin/macos-image-store/cleanup`: explicit admin action
  surface that defaults to `dry_run=true` and reuses the same candidate plan

When `dry_run=false`, the request must either include at least one filter
(`action_types` or `run_ids`) or set `confirm_all=true`. This keeps broad
cleanup of every planned candidate explicit.

Mutating cleanup only applies to already planned candidates that do not match a
live VM. It currently supports:

- deleting a planning-only manifest directory when it only contains
  `manifest.json`
- deleting a fully inactive run directory
- deleting a legacy run directory that has no persisted manifest

If a candidate still matches a live helper VM, cleanup fails closed for that
item and reports `live_vm_matches_blocked_cleanup` instead of deleting files.

## Networking

- `deny_all` is the intended strict baseline for the new macOS runtimes.
- `vz_linux` enforces that baseline in both layers: Python policy admission
  rejects unsupported policies before dispatch, and the Swift helper rejects
  direct `create_vm` requests unless `network_policy=deny_all`. The helper also
  echoes the accepted policy in VM metadata/status details for diagnostics.
  Direct `create_vm` requests are also shape-checked before boot: unsupported
  runtimes, invalid VM ids, non-absolute or NUL-bearing paths, symlink leaf
  paths or user-controlled symlinked parent components, oversized metadata, and
  out-of-range startup timeouts are rejected without registering a VM. Root-level
  macOS compatibility prefixes such as `/tmp` and `/var` are allowed and
  subsequent components are still checked.
- The current `vz_linux` Virtualization.framework configuration omits network
  device attachment; guest command transport uses vsock rather than guest
  network access.
- `seatbelt` only offers a best-effort deny-all claim via seatbelt policy; discovery should not be read as VM-grade or firewall-backed network isolation.
- `strict_allowlist_not_supported` is still the expected result for:
  - `vz_linux`
  - `vz_macos`
  - `seatbelt`
  - `worktree`
- `vz_linux` session-mode reuse exists now and reuses a persisted VM for later commands in the same sandbox session.
- `vz_macos` still has no warm-session optimization.

## Discovery And ACP

`/api/v1/sandbox/runtimes` now exposes:

- runtime availability
- preflight reasons
- normalized runtime reason codes
- isolation boundary metadata and host-local warnings
- session contract metadata
- supported trust levels
- enforcement readiness
- host facts

`/api/v1/sandbox/admin/runtime-diagnostics` is the cross-runtime operator
summary. It is admin-only, read-only, and derived from `/api/v1/sandbox/runtimes`
rather than a separate readiness source. Use it for dashboard-style triage
across Docker, Firecracker, Lima, `vz_linux`, `vz_macos`, `seatbelt`, and
`worktree`; use `/api/v1/sandbox/admin/macos-diagnostics` when you need
helper/template/image-store details for the macOS runtime family.

`/api/v1/sandbox/admin/macos-diagnostics` is the operator-focused companion surface.
It is admin-only and returns:

- detailed host readiness, including macOS version
- helper readiness, including transport, protocol version, and helper version when reachable
- template readiness for `vz_linux` and `vz_macos`, with optional template source metadata
- per-runtime execution mode and remediation hints
- reconciliation data comparing persisted VZ session rows with live helper VM state
- image-store correlation showing persisted run manifests, dry-run GC candidate
  classification, and any matching reconciliation/helper VM records
- read-only observability for `vz_linux` helper stdout/stderr log pointers,
  per-VM serial log pointers, guest readiness metadata, and helper-provided
  resource counters when available. Current `vz_linux` resource snapshots are
  configured VM facts (`cpu_count`, `memory_size_mb`) plus diagnostic uptime
  (`wall_time_sec`); they are not live CPU/RSS/I/O utilization telemetry.
- read-only `recovery_summary` metadata derived from reconciliation,
  image-store, and observability blocks, including recovery posture, stable
  issue codes, counts, and existing dry-run-first admin endpoint pointers
- an additive image-store cleanup plan endpoint and a default-dry-run cleanup
  mutation endpoint for explicit operator action
- additive `startup_warning_summary` showing whether current-process startup
  warnings were recorded, whether any were blocking, and which stable warning
  codes were emitted during boot

`GET /api/v1/admin/startup-warnings` is the generic app-level companion surface
for the same startup records. It is also admin-only and returns the full
current-process startup warning list plus grouped summary counts. The warning
scope is intentionally limited to the current API process and current boot; it
is not cluster-wide or persisted across restarts.

Use the admin endpoint when you are validating host setup or trying to explain why a
runtime is unavailable. Use `/api/v1/sandbox/runtimes` for client-facing discovery;
that payload stays summarized and does not expose helper/template internals.
The diagnostics observability block reports log paths, existence, and byte
sizes only; it does not read or return log contents.
The diagnostics `recovery_summary` block is also read-only. It summarizes
already-collected diagnostics as `healthy`, `action_recommended`, or
`unavailable`; it does not re-query the helper, scan the image store, or perform
repair.

ACP sandbox session creation now performs runtime preflight validation before calling the sandbox service, and converts failures into `ACPResponseError` instead of leaking raw sandbox exceptions.

## Reconciliation And Repair

`/api/v1/sandbox/admin/macos-diagnostics` is read-only. Its reconciliation block
compares persisted VZ session-control rows with helper live VM state and reports
healthy, stale, unhealthy, active, and orphan facts without changing either side.
Its `recovery_summary` block points operators at the existing repair and cleanup
plan endpoints when relevant, but the actual repair endpoint remains separate,
admin-only, explicit, and dry-run-first.

`POST /api/v1/sandbox/admin/macos-reconciliation/repair` is the explicit
admin-only repair endpoint. Repair defaults to dry-run, skips active sessions,
and can delete stale or unhealthy inactive persisted session-control rows when
requested. It can terminate orphan helper VMs only when
`terminate_orphaned_vms=true` is explicitly requested and helper metadata proves
the VM is owned by this `tldw` `vz_linux` sandbox control plane. Operators should
inspect the dry-run plan first. Helper unavailable or protocol mismatch
conditions fail closed and block mutating repair.

## Helper Crash, Restart, And Host Reboot Recovery

`vz_linux` recovery is generation-aware and helper-truth-driven. Persisted
session-control rows are useful control-plane state, but they are not proof that
a warm VM still exists. A row should be reused only when the current helper can
prove live VM health, `tldw/vz_linux` ownership, matching session metadata, and
matching helper generation.

If the helper crashes or is manually stopped and has not been restored:

- `vz_linux` runs that require helper truth fail closed.
- persisted session-control rows are preserved.
- `/api/v1/sandbox/admin/macos-diagnostics` reports recovery as unavailable.
- mutating reconciliation repair is blocked because helper truth is unavailable.
- the operator should restore helper readiness before retrying runs or repair.

If the helper is restarted directly or by a future launchd-managed workflow:

- treat the replacement helper as a new helper generation.
- run `tools/macos-vz-helper/scripts/vz-helperctl.py status` or `check` to
  verify socket, pid, log-directory, entitlements, ping, and protocol state.
- re-run macOS sandbox diagnostics.
- inspect `reconciliation` and `recovery_summary` before mutating anything.
- retrying a same-session run may clear stale control state and provision a
  fresh VM after reachable helper truth proves the old row is stale.
- if stale inactive rows remain, run
  `POST /api/v1/sandbox/admin/macos-reconciliation/repair` in dry-run mode
  before applying a mutating repair.

After a host reboot, assume helper process identity, helper in-memory VM state,
virtiofs state, and guest-agent readiness were lost until proven otherwise.
Durable image-store manifests and persisted session-control rows may still
exist, but they are provenance, not live VM proof. The recommended manual
procedure is:

1. Start or verify the helper through the managed operator workflow.
2. Run `vz-helperctl.py status` and confirm protocol-compatible helper ping.
3. Run `/api/v1/sandbox/admin/macos-diagnostics`.
4. Inspect stale, unhealthy, skipped-active, and orphan classifications.
5. Run reconciliation repair in dry-run mode if stale inactive rows are
   reported.
6. Apply mutating repair only after reviewing the dry-run plan.
7. Run the real host smoke to verify fresh ephemeral execution and same-session
   behavior.

Diagnostics, startup warnings, and host smoke must not delete session-control
rows or terminate VMs automatically. Host reboot is an operator procedure today,
not a scheduled CI action or hidden startup repair path.

Startup now also records bounded reconciliation/helper warnings during process
boot through the shared startup warning framework. That startup path is
read-only and never performs repair or VM termination. The current sandbox
startup warning policy is:

- `vz_helper_protocol_mismatch`: blocks startup
- `vz_helper_unavailable_at_startup`: warning only
- stale, unhealthy, skipped-active, and orphaned reconciliation findings:
  warning only

When startup succeeds with warnings, the warnings are visible both through
`GET /api/v1/admin/startup-warnings` and through the additive
`startup_warning_summary` field on `/api/v1/sandbox/admin/macos-diagnostics`.
When startup is blocked by helper protocol mismatch, logs are the guaranteed
surface because the API process never finishes booting.

Orphan VM classifications are:

- `owned_orphaned_vm`: metadata has `owner=tldw`, `runtime=vz_linux`, non-empty
  `run_id`, non-empty helper-created `created_at`, and a `session_id` when
  `session_mode=true`; repair may terminate these when explicitly requested.
- `unknown_orphaned_vm`: metadata is missing, legacy, or incomplete; repair
  reports and skips these.
- `foreign_orphaned_vm`: metadata exists but owner or runtime does not match this
  sandbox control plane; repair reports and skips these.

This metadata is local helper ownership metadata, not cryptographic proof. Unknown,
foreign, or legacy helper VMs may require manual operator cleanup outside the
automated repair endpoint.

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
- Managed helper lifecycle is available through `tools/macos-vz-helper/scripts/vz-helperctl.py`, including explicit `launchd` operator commands, but it remains operator-driven and does not install or load launchd services automatically
- No automatic orphan VM termination during diagnostics or repair; orphan termination is explicit repair-only behavior and is limited to owned `vz_linux` helper VMs

Current diagnostics are mixed-mode:

- outside `TEST_MODE`, `vz_linux` helper readiness comes from helper `ping`, `validate_host`, `validate_template`, and `list_vms`
- helper socket discovery for real `vz_linux` uses `TLDW_SANDBOX_MACOS_HELPER_SOCKET`
- helper path metadata is still optional and comes from `TLDW_SANDBOX_MACOS_HELPER_PATH`
- `vz_macos` readiness remains scaffolded through `TLDW_SANDBOX_VZ_MACOS_*`
- fake helper/template env flags still drive test-mode scaffolding
- `vz_linux` reports `execution_mode=real` only when the helper-backed path is reachable and the template validates
- `vz_macos` reports `execution_mode=fake` only when `TLDW_SANDBOX_VZ_MACOS_FAKE_EXEC=1`

## Cleanup Contract Coverage

Portable unit coverage asserts cleanup bookkeeping and temporary-directory
removal without requiring Docker, `sandbox-exec`, or a real VM host. The
hostless cleanup contract currently covers:

- Docker container lifecycle paths that create a container and then complete,
  hit startup timeout, or hit execution timeout; these must remove the container
  and clear active cancellation/egress bookkeeping.
- `seatbelt` and `worktree` cancellation paths; these must terminate the active
  process group, clear active process/run-directory bookkeeping, and remove the
  per-run directory.
- `vz_linux` helper-execution failure after VM creation; this must terminate
  the helper VM, clear active VM/run-directory bookkeeping, and remove the
  auto-created workspace.

Real Virtualization.framework cleanup remains host-gated. The portable tests do
not prove that a prepared Apple silicon host releases every VM process,
virtiofs resource, serial log handle, or helper-side run clone. Operators should
use the real host smoke below for VM process lifecycle coverage, including
ephemeral VM teardown and same-session VM reuse.

## Real Host E2E Smoke

The preferred operator entrypoint for proving real `vz_linux` execution on a
prepared Apple silicon macOS host is the managed helper wrapper:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py check
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements
```

Use `--dry-run` first to print the exact SwiftPM, codesign, helper, and pytest
commands without starting VMs:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --dry-run \
  --bundle /path/to/canonical/bundle
```

The wrapper delegates to `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
with the managed helper defaults. That lower-level script validates the bundle,
builds the Swift helper when the binary is missing, ad hoc signs it with the
supplied entitlements unless the helper is already signed with
`com.apple.security.virtualization`, runs the helper-daemon smoke, starts one
helper daemon for real `vz_linux` E2E, verifies ephemeral execution, verifies
same-session VM reuse, verifies recovery diagnostics plus dry-run
reconciliation repair planning, and stops the helper on exit.

`restart-drill` validates the direct `vz-helperctl.py`-managed lifecycle. Use it
after starting the helper through `vz-helperctl.py start` when you need to prove
that the managed pid-file/socket workflow can stop the helper, start a
replacement on the same paths, and reach healthy status again. It refuses
absent or unmanaged helpers, preserves existing helper logs under the
configured log directory, and does not run guest commands, mutate
reconciliation state, manage launchd, or validate host reboot behavior.

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py start
tools/macos-vz-helper/scripts/vz-helperctl.py restart-drill
tools/macos-vz-helper/scripts/vz-helperctl.py stop
```

`launchd` is available when operators want to manage the helper through a
LaunchAgent instead of the direct pid-file wrapper. Treat it as an explicit
procedure: run `--dry-run` first, write the plist only with `--write-plist`, and
create runtime/log directories only with `--create-dirs`.

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py launchd status --dry-run
tools/macos-vz-helper/scripts/vz-helperctl.py launchd bootstrap \
  --write-plist \
  --create-dirs \
  --dry-run
tools/macos-vz-helper/scripts/vz-helperctl.py launchd bootstrap \
  --write-plist \
  --create-dirs
tools/macos-vz-helper/scripts/vz-helperctl.py launchd kickstart
tools/macos-vz-helper/scripts/vz-helperctl.py launchd status
tools/macos-vz-helper/scripts/vz-helperctl.py status
tools/macos-vz-helper/scripts/vz-helperctl.py launchd bootout
```

`launchd-drill` is the opt-in validation path for LaunchAgent bootstrap,
kickstart, helper readiness, bootout, and optional real `vz_linux` smoke through
the launchd-managed helper. Use isolated labels and private plist/runtime paths
so the drill cannot take ownership of a user's existing service.

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-launchd-drill.XXXXXX")"
chmod 700 "$runtime_dir"
label="org.tldw.macos-vz-helper.drill.$$"
trap 'tools/macos-vz-helper/scripts/vz-helperctl.py launchd bootout --label "$label" --plist-output "$runtime_dir/${label}.plist" >/dev/null 2>&1 || true; rm -rf "$runtime_dir"' EXIT

tools/macos-vz-helper/scripts/vz-helperctl.py launchd-drill \
  --socket "$runtime_dir/helper.sock" \
  --log-dir "$runtime_dir/logs" \
  --plist-output "$runtime_dir/${label}.plist" \
  --label "$label" \
  --write-plist \
  --create-dirs \
  --skip-smoke
```

Omit `--skip-smoke` and add `--bundle /path/to/canonical/bundle` when the
prepared host should also run the real `vz_linux` smoke against the
launchd-managed helper. The default direct-helper smoke path remains unchanged.

The launchd path does not run from diagnostics, server startup, `status`,
`plist`, or `smoke`. It also does not validate host reboot behavior; reboot
testing remains a manual operator drill and must stay out of scheduled CI until
a prepared runner can tolerate disruptive host lifecycle changes and preserve
logs.

Manual failure drills are opt-in and remain disabled for default smoke and
scheduled host-gated runs. To include them, pass:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements \
  --include-failure-drills
```

The manual drills currently cover a drill-owned stale session VM after
helper-side VM termination and stale session-control VM state after the smoke
harness stops and restarts its helper process through a private restart lease.
They do not cover host reboot, launchd bootstrap/load behavior, broad helper
crash classes, networking changes, or destructive repair generalization.

The recovery smoke is non-destructive. It seeds an isolated test-store stale VZ
session-control row, verifies `/api/v1/sandbox/admin/macos-diagnostics` style
reconciliation and `recovery_summary` output through the service layer, then
verifies `repair_macos_reconciliation(dry_run=true)` produces a planned delete
without deleting state or terminating VMs.

The lower-level fallback remains available when operators need to bypass the
managed defaults:

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-helper-e2e.XXXXXX")"
chmod 700 "$runtime_dir"

tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  --bundle /path/to/canonical/bundle \
  --socket "$runtime_dir/helper.sock" \
  --serial-log-dir "$runtime_dir/serial" \
  --entitlements /path/to/helper.entitlements
```

The underlying opt-in pytest module is still available directly:

- `tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py`

Required env for that module:

- `TLDW_SANDBOX_VZ_LINUX_E2E=1`
- `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=<base-image-path-or-registered-template-id>`
- `TLDW_SANDBOX_MACOS_HELPER_SOCKET=/path/to/helper.sock`
- real helper reachability required through helper `ping`
- helper-backed template validation required through `validate_template`

If `TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE` is a registered template id instead of
an absolute bundle/path, also set:

- `TLDW_SANDBOX_IMAGE_STORE_ROOT=/var/lib/tldw/sandbox-images`

When that root is configured, the `vz_linux` runner resolves registered
template ids through the image store, uses the stored `source_path` for helper
validation/boot, and persists a run-scoped clone manifest before VM creation.

The helper function in that test module also forces:

- `SANDBOX_ENABLE_EXECUTION=1`
- `SANDBOX_BACKGROUND_EXECUTION=0`

That keeps the smoke path synchronous and prevents it from silently using the
fake helper contract. On unprepared hosts, the module should skip with explicit
helper-or-template reasons instead of reporting a fake pass.

## Host-Gated CI

Real `vz_linux` execution cannot run on the default hosted CI fleet because it
requires Apple silicon macOS plus a prepared Virtualization.framework helper and
canonical bundle. The repo therefore keeps normal CI portable and adds a
separate host-gated workflow:

- `.github/workflows/vz-linux-host-gated.yml`
- runner labels: `self-hosted`, `macOS`, `ARM64`, `vz-linux`
- manual trigger: `workflow_dispatch`
- scheduled trigger: present, but skipped unless
  `TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1`

Required host/repository configuration:

- a self-hosted Apple silicon macOS runner with the labels above
- SwiftPM and Xcode command line tools available to the runner
- a canonical `vz_linux` bundle already present on the runner
- workflow input `bundle_path` or repository variable
  `TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH`
- optional repository variable `TLDW_SANDBOX_VZ_HELPER_ENTITLEMENTS_PATH`
- optional repository variable `TLDW_SANDBOX_VZ_HELPER_SKIP_SIGN=true`

The self-hosted job is branch-gated to `main` and `dev` so a manual dispatch
does not accidentally execute arbitrary feature-branch code on the prepared
host. External actions in this workflow are pinned to immutable SHAs because
they execute on the self-hosted runner.

The acceptance policy for when this workflow should run, what counts as an
expected skip, and what counts as a blocking `vz_linux` regression lives in
`Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`.
Prepared-host run evidence and residual gaps should be recorded in
`Docs/Sandbox/vz-linux-prepared-host-evidence.md`.

The workflow calls the same operator smoke script documented above:

```bash
bash tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  --bundle "$TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH" \
  --socket "$RUNNER_TEMP/tldw-vz-helper-ci/helper.sock" \
  --serial-log-dir "$RUNNER_TEMP/tldw-vz-helper-ci/serial"
```

That keeps CI aligned with local operator behavior instead of creating a second
helper lifecycle path. The job uploads helper logs from the runner temp
directory even when the smoke fails.

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
