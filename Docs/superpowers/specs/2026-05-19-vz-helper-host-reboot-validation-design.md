# VZ Helper Host Reboot Validation Design

**Status:** Proposed implementation slice.
**Date:** 2026-05-19.
**Backlog:** TASK-438.

## Goal

Turn the current host-reboot recovery guidance for `vz_linux` into a repeatable
operator validation procedure without adding automatic reboot behavior to the
application, default smoke path, or scheduled host-gated CI.

The validation should prove that, after a real macOS host reboot, operators can
restore or verify helper readiness, inspect stale runtime state, run dry-run
repair planning, and prove fresh `vz_linux` execution plus same-session behavior
through existing smoke surfaces.

## Current State

The repo already has these building blocks:

- `vz-helperctl.py status`, `check`, `start`, `stop`, `restart-drill`, `launchd`,
  `launchd-drill`, and `smoke`.
- `launchd-drill` for explicit LaunchAgent bootstrap, kickstart, readiness,
  bootout, and optional real `vz_linux` smoke through a launchd-managed socket.
- host-gated smoke for real ephemeral execution, same-session VM reuse,
  recovery diagnostics, and dry-run reconciliation repair planning.
- generation-aware `vz_linux` session reuse that rejects stale helper/session
  control metadata before provisioning a fresh VM.
- read-only macOS diagnostics and explicit dry-run-first reconciliation repair.

The remaining gap is host reboot as an operator procedure. Reboot can invalidate
helper process identity, helper in-memory VM registry state, virtiofs resources,
guest-agent readiness, and existing warm VM assumptions. Persisted
session-control rows and image-store manifests survive reboot but are provenance,
not live runtime proof.

## Non-Goals

- Do not reboot the host from repo code, CLI, tests, CI, or server startup.
- Do not add scheduled host reboot CI.
- Do not make launchd the default helper lifecycle.
- Do not install, load, or unload launchd services implicitly from diagnostics
  or server startup.
- Do not automatically delete session-control rows or terminate VMs.
- Do not broaden orphan repair behavior beyond the existing explicit repair
  endpoint.
- Do not read raw serial logs into API responses or committed evidence.

## Proposed Operator Model

Add a small host-reboot validation layer around existing commands. The preferred
shape is a `vz-helperctl.py host-reboot-drill` command with two explicit phases
and a reboot-surviving evidence directory. Operators should use a durable path
such as `~/Library/Logs/tldw/vz-host-reboot-drill/<run-id>` or another
operator-owned directory, not `/tmp`, because temporary directories may be
purged across reboot.

```bash
tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill pre \
  --evidence-dir ~/Library/Logs/tldw/vz-host-reboot-drill/<run-id> \
  --bundle /path/to/canonical/bundle \
  --helper-mode direct

# Operator manually reboots the host.

tools/macos-vz-helper/scripts/vz-helperctl.py host-reboot-drill post \
  --evidence-dir ~/Library/Logs/tldw/vz-host-reboot-drill/<run-id> \
  --bundle /path/to/canonical/bundle \
  --helper-mode direct
```

`helper-mode` should support `direct` and `launchd`:

- `direct` means the operator restores helper readiness through
  `vz-helperctl.py start`, `status`, `smoke`, and `stop`.
- `launchd` means the operator restores helper readiness through explicit
  `launchd` or `launchd-drill` commands.

The command must never call `shutdown`, `reboot`, `launchctl bootstrap`, or
mutating repair unless the operator passed a specific phase/action that already
exists for that purpose. The host reboot step itself remains outside the repo.
If the post phase runs real smoke against an already-restored helper, it should
use the existing managed-helper smoke seam (`run_vz_linux_host_smoke`) that
points pytest at the selected socket. It should not call the default
`vz-helperctl.py smoke` path in a way that starts a second helper and validates
the wrong process.

## Pre-Reboot Phase

The pre phase should:

1. Require a durable private evidence directory (`0700`) or create it with
   owner-only permissions when explicitly requested. Warn or fail when the path
   is under volatile temp roots such as `/tmp`, `/private/tmp`, or `$TMPDIR`
   unless an explicit override is provided.
2. Record bounded metadata in a JSON manifest:
   - timestamp
   - hostname
   - helper mode
   - helper path
   - socket path
   - log directory
   - serial-log directory
   - bundle path or template id
   - helper ping result
   - helper protocol version
   - helper version
   - helper generation details when available
   - launchd label/plist path when helper mode is `launchd`
3. Run existing non-mutating preflight checks:
   - `check` or `status`
   - launchd service presence check when helper mode is `launchd`
   - bundle/template validation through existing helper smoke dry-run paths
4. Print the next manual step: reboot the host, then run the post phase.

The pre phase may optionally run the existing real smoke before reboot when
operators want a before/after comparison, but that should be explicit and
host-gated. It should not create a new bespoke VM/session mechanism.
The first implementation should not try to preserve a drill-owned warm VM across
reboot. That requires a separate persistent-session drill because the existing
host smoke owns and cleans up its own helper/VM lifecycle.

## Post-Reboot Phase

The post phase should:

1. Read and validate the pre-phase manifest.
2. Refuse world-readable evidence directories or malformed manifest data.
3. Verify helper readiness using the selected helper mode:
   - `direct`: explicit `status` or operator-requested `start` followed by
     `status`.
   - `launchd`: explicit `launchd status` or operator-requested
     `launchd-drill`; never infer ownership of a production LaunchAgent.
4. Record post-reboot helper ping, protocol, version, generation, socket, and
   log-path facts.
5. Tell the operator to inspect macOS sandbox diagnostics and dry-run repair.
6. Optionally run existing real smoke against the restored helper socket:
   - ephemeral command execution
   - same-session VM reuse
   - recovery diagnostics
   - dry-run reconciliation repair planning
7. Write a post-reboot evidence JSON next to the pre-reboot manifest.

The post phase should treat helper generation changes as expected after reboot.
It should fail only when helper readiness, protocol compatibility, diagnostics,
or real smoke cannot be proven after the operator restores the helper.
Stale pid files from before reboot should be reported as stale metadata and
should not be trusted as process ownership. Existing helperctl start/status
safety checks remain the source of truth for direct-helper process ownership.

## Diagnostics And Repair Boundary

The drill should not directly call admin APIs unless a later implementation adds
an explicit authenticated API option. The initial implementation should print
the existing API steps instead:

```text
GET  /api/v1/sandbox/admin/macos-diagnostics
POST /api/v1/sandbox/admin/macos-reconciliation/repair?dry_run=true
```

If a future authenticated option is added, it must remain dry-run by default and
must not apply mutating repair unless a separate explicit operator flag is used.

## Evidence Contract

Evidence files are local operator artifacts, not public API. They should be
small, structured, and safe to attach to a bug report after review.

Required files:

- `host-reboot-pre.json`
- `host-reboot-post.json`
- optional helper stdout/stderr paths as references only
- optional serial-log paths as references only

The JSON should store paths and summary facts, not raw serial logs, guest
stdout/stderr bodies, API keys, environment dumps, or user workspace contents.

## Safety Rules

- Evidence directory must be private (`0700`) and owned by the current user.
- The command must validate configured socket and log directories using the
  same helperctl path-hardening helpers as `status`, `check`, and
  `launchd-drill`.
- `launchd` mode must use an explicit label and plist path from the manifest or
  from current arguments; it must not infer ownership of unrelated user
  LaunchAgents.
- Post-reboot smoke must target the restored helper socket. It must not silently
  start a new helper and then claim that the preconfigured direct or launchd
  helper recovered.
- A post-reboot mismatch between pre and post helper generation is expected and
  should be reported as `helper_generation_changed`, not as failure.
- Missing helper readiness after reboot is a failure until the operator restores
  helper status.
- Mutating repair is outside the drill.

## Portable Tests

Normal CI should remain host-independent. Focused tests should cover:

- pre phase creates or validates a private evidence directory
- pre phase rejects volatile evidence directories by default
- pre phase writes bounded JSON and rejects unsafe directories
- post phase rejects missing, malformed, or world-readable evidence
- post phase treats helper-generation drift as expected after reboot
- direct helper mode constructs status/smoke steps without launchd mutation
- launchd helper mode constructs launchd status/drill steps without using a
  default production label accidentally
- post phase smoke targets the restored helper socket instead of invoking a
  helper-owning smoke path
- dry-run output is deterministic and JSON output is parseable

These tests should mock command runners and helper ping/status functions. They
must not require a real reboot, real launchd, or Virtualization.framework.

## Manual Host-Gated Validation

Manual validation on a prepared Apple silicon host should be documented as:

1. Run `host-reboot-drill pre` with a private evidence directory.
2. Reboot the host manually.
3. Restore or verify helper readiness through `direct` or `launchd` mode.
4. Run `host-reboot-drill post`.
5. Run `/api/v1/sandbox/admin/macos-diagnostics`.
6. Run reconciliation repair dry-run.
7. Run the real `vz_linux` smoke.
8. Attach bounded evidence files and helper log paths to the operator record.

This is not a scheduled CI gate until a dedicated prepared runner can tolerate
disruptive reboot testing and preserve logs across reboot.

## Design Risk Review

The main implementation risk is accidentally turning a manual host lifecycle
procedure into a hidden mutating repair path. The design avoids that by keeping
the actual reboot outside the tool, keeping repair dry-run and manual, and
reusing existing helperctl and smoke commands.

The second risk is creating a false sense of proof. A post-reboot status check
alone does not prove real VM execution. The drill therefore treats helper
readiness, diagnostics, dry-run repair planning, and real smoke as separate
evidence layers. The implementation must also avoid calling a helper-owning
smoke wrapper that would start a fresh helper and validate the wrong lifecycle.

The third risk is launchd ownership confusion. LaunchAgent labels and plist
paths must be explicit in launchd mode and must not allow the drill to bootout
an unrelated service.

The fourth risk is evidence leakage. The evidence contract must store bounded
metadata and path references only, not raw logs, environment dumps, or user
workspace data.

The fifth risk is losing the evidence directory during reboot. The design
requires a durable operator-owned path and rejects temporary roots by default.

## Open Questions For Implementation

- Should the first implementation add only `pre`/`post` evidence commands, or
  also a thin `--run-smoke` wrapper that invokes the existing host smoke during
  post?
- Should authenticated diagnostics/dry-run repair API calls stay out of
  `vz-helperctl.py` permanently, or be added later behind explicit API URL/key
  options?
- Should host reboot evidence become part of the prepared-host tracker once a
  maintainer has run the procedure successfully?
