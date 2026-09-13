# macOS VZ Helper

This directory contains the first-party native macOS helper used by the
`vz_linux` sandbox runtime.

## Scope

The helper is intentionally narrow:

- `vz_linux` only
- local Unix-socket daemon
- owns `Virtualization.framework` lifecycle
- owns host readiness and runnable-template truth
- owns VM create, exec, status, list, and terminate operations
- stores per-VM ownership metadata supplied by the Python sandbox control plane

## Non-Goals

The first helper slice is not a generic macOS sandbox backend:

- no `vz_macos` support yet
- no `seatbelt` support here
- no second persistence layer for sandbox sessions
- no APFS clone execution path yet
- no automatic launchd installation or helper auto-upgrade yet
- no automatic orphan VM termination during admin repair; Python can request it only through explicit dry-run-first reconciliation repair, and only for helper VMs whose metadata proves `owner=tldw` and `runtime=vz_linux`

Python remains authoritative for sandbox admission, session identity, artifacts, and ACP
integration. The helper only owns runtime VM facts and control-plane operations.
Helper VM metadata is local control-plane metadata, not cryptographic attestation.
Legacy or manually created helper VMs without metadata are reported as unknown and skipped
by automated repair.

## Managed Helper Lifecycle

Use `tools/macos-vz-helper/scripts/vz-helperctl.py` for local operator workflows:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py check
tools/macos-vz-helper/scripts/vz-helperctl.py build
tools/macos-vz-helper/scripts/vz-helperctl.py start
tools/macos-vz-helper/scripts/vz-helperctl.py status
tools/macos-vz-helper/scripts/vz-helperctl.py restart-drill
tools/macos-vz-helper/scripts/vz-helperctl.py stale-socket-drill
tools/macos-vz-helper/scripts/vz-helperctl.py stop
tools/macos-vz-helper/scripts/vz-helperctl.py plist
tools/macos-vz-helper/scripts/vz-helperctl.py launchd status --dry-run
```

The command uses stable user-owned defaults under
`~/Library/Application Support/tldw/sandbox/macos-vz-helper/` and
`~/Library/Logs/tldw/macos-vz-helper/`.

`plist` prints LaunchAgent scaffolding by default and does not create runtime
directories unless `--create-dirs` is provided. It does not call `launchctl`,
install services, or auto-upgrade helpers.

`launchd` provides explicit operator actions for a LaunchAgent-managed helper.
Use dry-run first to inspect the `launchctl` command, then opt into plist
creation/loading only when intended:

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
tools/macos-vz-helper/scripts/vz-helperctl.py launchd bootout
```

### Launchd Validation Drill

`launchd-drill` validates the launchd-managed helper path without making
launchd the default smoke lifecycle. Use it when an operator wants proof of
LaunchAgent bootstrap, kickstart, helper readiness, and bootout on isolated
runtime paths.

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
  --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements \
  --write-plist \
  --create-dirs \
  --skip-smoke
```

Omit `--skip-smoke` and add `--bundle /path/to/canonical/bundle` to run the
real `vz_linux` host smoke through the launchd-managed helper. When
`--entitlements` is provided, the drill signs the helper after the
already-loaded preflight and before bootstrap so a failed signing step does not
leave a launchd service running. The checked-in entitlements template is the
least-privilege operator template and intentionally omits debugger attachment
entitlements such as `com.apple.security.get-task-allow`. Keep drill labels
isolated and plist paths private; the drill refuses to take over a service that
is already loaded before bootstrap because cleanup ownership would be
ambiguous.

These commands never run automatically from `plist`, `status`, `smoke`, or
server startup. They are operator-owned scaffolding and do not validate host
reboot behavior.

### Live-Session Launchd Restart

The manual host test below restarts its own LaunchAgent **after** a real guest
command. It requires a changed helper generation, replaces the stale session
VM, verifies guest stdout and exit status, then reuses the replacement VM for a
third command. This is separate from `launchd-drill`, whose initial kickstart
occurs before smoke execution. It does not reboot the host or restart an
operator's existing helper.

Run from the repository root on a prepared Apple Silicon macOS host with a
logged-in GUI user. Set `SOURCE_BUNDLE` to a canonical bundle containing the
current guest agent; never pass that source directly as the writable test image.
The test skips unless both real-E2E and launchd-restart opt-ins are set.

```bash
(
  set -eu
  umask 077
  source .venv/bin/activate
  swift build --package-path tools/macos-vz-helper
  python tools/macos-vz-helper/scripts/vz-helperctl.py sign \
    --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements

  evidence_root="$HOME/Library/Logs/tldw/vz-launchd-recovery"
  install -d -m 700 "$evidence_root"
  evidence_dir="$(mktemp -d "$evidence_root/run.XXXXXX")"
  printf 'Retaining evidence in %s\n' "$evidence_dir"
  bundle="$(python tools/vz-linux-image/scripts/prepare-smoke-bundle.py \
    --source-bundle "${SOURCE_BUNDLE:?Set SOURCE_BUNDLE to a prepared canonical bundle}" \
    --store-root "$evidence_dir/image-store" --run-id launchd-recovery)"

  TLDW_SANDBOX_VZ_LINUX_E2E=1 \
  TLDW_SANDBOX_VZ_LINUX_LAUNCHD_RESTART_DRILL=1 \
  TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE="$bundle" \
  TLDW_SANDBOX_MACOS_HELPER_BINARY="$PWD/tools/macos-vz-helper/.build/debug/macos-vz-helper" \
    python -m pytest \
      tldw_Server_API/tests/sandbox/test_vz_linux_launchd_recovery_host_gated.py \
      -q -rs --junitxml="$evidence_dir/pytest.xml" --basetemp="$evidence_dir/pytest"
)
```

Acceptance requires **1 passed, 0 skipped, 0 errors**, not merely pytest exit 0.
`launchd-recovery.json`, the generated plist, and helper/serial logs live under
the pytest artifact tree. The receipt includes helper generations, VM IDs,
command output, and cleanup checks. Only the short private socket directory is
removed automatically after service absence and socket unavailability are
confirmed. Evidence and disposable disks are retained deliberately; remove that
run's evidence directory after review when it is no longer needed. Hard host
termination can bypass Python cleanup; inspect the receipt's unique label and
plist before any manual cleanup. Do not add this opt-in to scheduled CI.

### Host Reboot Validation Drill

`host-reboot-drill` records bounded helper evidence before a manual host reboot
and validates the restored helper after the machine comes back. The evidence
directory must be durable across reboot and private to the operator; use a path
such as `~/Library/Logs/tldw/vz-host-reboot-drill/<run-id>`, not `/tmp`,
`$TMPDIR`, or another volatile root.
The pre phase records a host boot marker, helper lifecycle preflight results,
and bundle dry-run validation. The post phase fails if the host boot marker is
missing or unchanged, because that means the drill did not prove a reboot.

Direct helper mode uses the managed helper socket directly:

```bash
evidence_root="$HOME/Library/Logs/tldw/vz-host-reboot-drill"
mkdir -p "$evidence_root"
chmod 700 "$evidence_root"
run_id="manual-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$run_id" > "$evidence_root/latest-run-id"
chmod 600 "$evidence_root/latest-run-id"
evidence_dir="$evidence_root/$run_id"
socket_path="$HOME/Library/Application Support/tldw/sandbox/macos-vz-helper/helper.sock"

tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill pre \
  --evidence-dir "$evidence_dir" \
  --socket "$socket_path" \
  --pid-file "$HOME/Library/Application Support/tldw/sandbox/macos-vz-helper/helper.pid" \
  --bundle /path/to/canonical/bundle \
  --create-evidence-dir

# Manually reboot the host, then restore or verify the same helper socket path.

evidence_root="$HOME/Library/Logs/tldw/vz-host-reboot-drill"
run_id="$(cat "$evidence_root/latest-run-id")"
evidence_dir="$evidence_root/$run_id"
socket_path="$HOME/Library/Application Support/tldw/sandbox/macos-vz-helper/helper.sock"

tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill post \
  --evidence-dir "$evidence_dir" \
  --socket "$socket_path" \
  --pid-file "$HOME/Library/Application Support/tldw/sandbox/macos-vz-helper/helper.pid" \
  --bundle /path/to/canonical/bundle \
  --run-smoke
```

Launchd helper mode is explicit in both phases. Provide the same `--label` and
`--plist-output` before and after reboot so the manifests can bind evidence to
the intended LaunchAgent instead of an implicit default:

```bash
label="org.tldw.macos-vz-helper.manual-reboot"
plist="$HOME/Library/LaunchAgents/${label}.plist"
evidence_root="$HOME/Library/Logs/tldw/vz-host-reboot-drill"
mkdir -p "$evidence_root"
chmod 700 "$evidence_root"
run_id="manual-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$run_id" > "$evidence_root/latest-launchd-run-id"
chmod 600 "$evidence_root/latest-launchd-run-id"
evidence_dir="$evidence_root/$run_id"

tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill pre \
  --helper-mode launchd \
  --label "$label" \
  --plist-output "$plist" \
  --evidence-dir "$evidence_dir" \
  --bundle /path/to/canonical/bundle \
  --create-evidence-dir

# Manually reboot the host, then verify launchd restored the helper.

label="org.tldw.macos-vz-helper.manual-reboot"
plist="$HOME/Library/LaunchAgents/${label}.plist"
evidence_root="$HOME/Library/Logs/tldw/vz-host-reboot-drill"
run_id="$(cat "$evidence_root/latest-launchd-run-id")"
evidence_dir="$evidence_root/$run_id"

tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill post \
  --helper-mode launchd \
  --label "$label" \
  --plist-output "$plist" \
  --evidence-dir "$evidence_dir" \
  --bundle /path/to/canonical/bundle \
  --run-smoke
```

When `post --run-smoke` is used, the smoke targets the restored helper socket
through the host smoke path. It must not start a new helper process for the
post-reboot proof. Diagnostics and dry-run reconciliation repair remain
operator-reviewed follow-up steps and are separate from this drill. Scheduled
or nightly CI must not reboot hosts; this validation is manual or explicitly
operator-triggered only. Blocking post failures include unchanged host boot
marker, missing boot marker, failed lifecycle readiness, metadata mismatch,
helper ping/protocol failure, and requested post-smoke failure.

`restart-drill` is an operator-managed lifecycle drill for helpers already
started through `vz-helperctl.py start`. It verifies the current managed helper
status, stops it through the pid-file/socket lease, starts a replacement on the
same managed paths, and verifies status again. It does not manage launchd,
reboot the host, or take ownership of helpers started outside this wrapper.

`stale-socket-drill` is an operator-managed check for the helper socket recovery
path. It validates private runtime/log directories, creates or preserves only a
safe inactive Unix socket, starts the helper through the normal managed start
path, and verifies helper status afterward. It is manual only and must not be
wired into normal PR/push CI.

```bash
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-stale-socket.XXXXXX")"
chmod 700 "$runtime_dir"
trap 'rm -rf "$runtime_dir"' EXIT

tools/macos-vz-helper/scripts/vz-helperctl.py stale-socket-drill \
  --helper tools/macos-vz-helper/.build/debug/macos-vz-helper \
  --socket "$runtime_dir/helper.sock" \
  --pid-file "$runtime_dir/helper.pid" \
  --log-dir "$runtime_dir/logs"
```

For evidence, record the command output, runtime directory mode, socket path,
helper stdout/stderr paths under the log directory, and whether the drill
created a controlled stale socket or recovered a pre-existing inactive socket.

For real host E2E smoke, prefer the managed wrapper:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements tools/macos-vz-helper/macos-vz-helper.entitlements
```
