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
tools/macos-vz-helper/scripts/vz-helperctl.py stop
tools/macos-vz-helper/scripts/vz-helperctl.py plist
```

The command uses stable user-owned defaults under
`~/Library/Application Support/tldw/sandbox/macos-vz-helper/` and
`~/Library/Logs/tldw/macos-vz-helper/`.

`plist` prints LaunchAgent scaffolding by default and does not create runtime
directories unless `--create-dirs` is provided. It does not call `launchctl`,
install services, or auto-upgrade helpers.

`restart-drill` is an operator-managed lifecycle drill for helpers already
started through `vz-helperctl.py start`. It verifies the current managed helper
status, stops it through the pid-file/socket lease, starts a replacement on the
same managed paths, and verifies status again. It does not manage launchd,
reboot the host, or take ownership of helpers started outside this wrapper.

For real host E2E smoke, prefer the managed wrapper:

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py smoke \
  --bundle /path/to/canonical/bundle \
  --entitlements /path/to/helper.entitlements
```
