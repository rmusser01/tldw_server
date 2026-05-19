# VZ Linux Helper Generation Session Recovery Design

**Date:** 2026-05-09
**Status:** Approved design; awaiting implementation-plan review
**Backlog:** TASK-160
**Scope:** `vz_linux` helper-generation metadata, persisted session-control reuse checks, focused host-independent tests, and minimal operator documentation.

## Summary

The `vz_linux` runner can already reuse a session VM by reading persisted
session-control metadata and asking the helper whether the stored VM is healthy.
That is not enough after a helper restart. The replacement helper owns a fresh
in-memory VM registry, so a stale session-control row can point at a VM ID that
is no longer meaningful to the current helper process. The runner should
distinguish a healthy same-helper reuse candidate from stale control state after
helper identity or generation drift, clear stale metadata only when helper truth
is available, and provision a fresh VM for the session.

This slice adds a helper-owned generation signal and makes session reuse depend
on that signal. It does not add host reboot automation, launchd lifecycle
management, network behavior, or broader reconciliation repair.

## Current Baseline

- `VZLinuxRunner._run_real()` loads `sandbox_vz_sessions` metadata when
  `RunSpec.session_id` is present.
- A persisted row is reusable when its runtime is `vz_linux`, `agent_ready` is
  true, it has a VM ID, and `helper.get_vm_status(vm_id).healthy` is true.
- If `get_vm_status()` returns unhealthy or no status, the runner deletes the
  session-control row and provisions a fresh VM.
- The store persists only `runtime`, `vm_id`, `template_id`,
  `workspace_mount`, `agent_ready`, and timestamps for VZ session control.
- The Swift helper returns `protocol_version`, `helper_version`, VM status,
  ownership metadata, and details, but it does not currently expose a helper
  instance or start-time generation.

## Goals

- Add a helper-owned generation signal that changes when the helper process
  restarts.
- Persist the helper generation alongside `vz_linux` session-control metadata.
- Require same-generation helper truth before reusing a stored session VM.
- Clear stale generation metadata and provision a fresh session VM only when the
  helper is reachable and protocol-compatible.
- Preserve persisted state when helper availability or protocol compatibility is
  ambiguous.
- Keep tests host-independent and focused on the runner/store/helper protocol
  contract.

## Non-Goals

- Do not add launchd install/load/unload behavior.
- Do not automate host reboot drills.
- Do not add network policy changes.
- Do not broaden admin repair mutation.
- Do not change public sandbox API shape.
- Do not require real VZ execution for this PR's normal verification.

## Approach Options

### Option A: Helper-Owned Generation In Protocol Details

The Swift helper creates a per-process `helper_instance_id` at service
initialization and records `helper_started_at`. It returns those values in
`ping.details` and VM status/details. Python parses the details as optional
strings and persists them with session-control metadata.

This is the recommended path. The generation truth belongs to the helper, which
matches the architecture doctrine: Python owns persisted identity, while the
runtime helper owns live VM truth. It also gives tests a small protocol surface
without changing the top-level helper response schema.

Tradeoff: it touches both Swift helper and Python store schema, but the change
is narrowly scoped and backward-compatible at the protocol level.

### Option B: Python-Synthesized Generation

Python could derive a generation key from helper protocol version, helper
version, socket path, and possibly process facts. This avoids Swift changes, but
it is not authoritative. The values can remain identical across helper restarts,
and Python should not infer runtime generation from filesystem or process
heuristics when the helper can report it directly.

This should not be used.

### Option C: Always Treat Session Rows As Stale After Restart Drills

The runner could skip generation metadata and rely on missing/unhealthy VM
status after helper restart to force reprovisioning. That is close to the
current behavior and does not encode the recovery invariant. It also cannot
distinguish "reachable helper with stale state" from "helper unavailable or
protocol mismatched", which leads to unsafe deletion of potentially recoverable
rows.

This is insufficient for long-term recovery hardening.

## Chosen Design

Use Option A. The helper is the source of truth for process generation:

- `helper_instance_id`: a random UUID generated once per helper process.
- `helper_started_at`: an ISO-8601 timestamp generated once per helper process.

The helper includes both keys in:

- `ping.details`
- `HelperVMResponse.details` for newly created VMs
- `HelperVMStatusResponse.details` for status/list replies

Python treats these detail fields as optional untrusted strings until they pass
simple non-empty normalization. The runner persists them when it stores session
control and compares them during later session reuse.

## Session-Control Schema

Add optional columns/fields to every VZ session-control store implementation:

- `helper_instance_id`
- `helper_started_at`

The in-memory store, SQLite store, Postgres store, and orchestrator facade must
accept and return the same fields. SQLite and Postgres initialization should
include the columns for new stores and backfill migrations for existing stores.

The fields are optional so older rows remain readable. Since this runtime is not
available in production yet, backward compatibility is not a product blocker,
but the implementation should still avoid breaking existing local operator
stores unnecessarily.

## Reuse Contract

For a session-mode `vz_linux` run, the runner should evaluate persisted
session-control rows in this order:

1. Validate the row is structurally reusable: runtime is `vz_linux`,
   `agent_ready` is true, and `vm_id` is non-empty.
2. Ask the helper for the stored VM status.
3. If helper reachability or protocol validation fails, abort the run through
   the existing failed status path and do not delete the session-control row.
4. If status is absent, unhealthy, or not owned by `tldw/vz_linux`, delete the
   stale row and provision a fresh VM.
5. If status metadata has a non-empty `session_id`, it must match the requested
   session ID. A mismatch is stale and should be replaced.
6. If stored generation and live generation are both present, they must match.
   A mismatch is stale and should be replaced.
7. If either stored generation or live generation is missing while helper truth
   is reachable, treat the row as stale unless the live VM metadata proves the
   same `tldw/vz_linux` session. This allows older local rows to be recovered
   conservatively without trusting metadata-free reuse.
8. Only after all checks pass should the runner reuse the VM and skip template
   validation/create.

For a newly provisioned session VM, the runner should persist the helper
generation returned by `create_vm.details`. If the creation response does not
include generation but a prior `ping()` did, the runner may persist the ping
generation. If no generation is available, store empty values and let the next
reuse treat the row conservatively.

## Failure Behavior

The important difference from the current baseline is ambiguous helper state:

- `MacOSVirtualizationHelperUnavailable`: fail closed, preserve row.
- `MacOSVirtualizationHelperProtocolError`: fail closed, preserve row.
- `None` status from a reachable helper: stale, delete row, provision fresh VM.
- `healthy=false` status from a reachable helper: stale, delete row, provision
  fresh VM.
- helper-generation mismatch: stale, delete row, provision fresh VM.
- metadata owner/runtime/session mismatch: stale, delete row, provision fresh VM.

This preserves recoverable control state when Python cannot establish helper
truth. It still repairs stale state once the helper is reachable and can speak
the expected protocol.

## Test Strategy

Host-independent tests should cover:

- Swift helper/service ping includes `helper_instance_id` and
  `helper_started_at` details.
- Python parser/client preserves helper generation details without promoting
  malformed fields to trusted top-level state.
- Store implementations persist and return helper generation fields for
  `put_vz_session_control()` and `get_vz_session_control()`.
- Healthy same-generation session reuse executes against the existing VM and
  does not call template validation or VM creation.
- Generation mismatch deletes the stale row, validates the template, creates a
  fresh VM, stores the new generation, and completes.
- Helper unavailable during reuse fails closed and does not delete or overwrite
  the row.
- Helper protocol mismatch during reuse fails closed and does not delete or
  overwrite the row.
- Legacy row behavior is explicit: when generation is missing, reachable helper
  truth plus matching ownership/session metadata is required before reuse.

Real VZ smoke remains covered by host-gated workflows and manual operator smoke.
This PR does not need a new real-host drill unless the implementation changes
host smoke behavior.

## Documentation

Update only the sandbox operator docs needed to explain the new contract:

- session reuse depends on helper-owned generation and live VM metadata
- helper unavailable/protocol mismatch preserves persisted state and fails
  closed
- stale generation or stale VM truth triggers fresh VM provisioning

Do not expand public API docs unless a user-visible response field changes.

## Plan And Design Review

### Potential Issue: Helper Generation Could Be Spoofed By Tests Or Malformed Payloads

Mitigation: generation fields are hints from a trusted local helper only after
the normal helper protocol/version checks pass. Python should normalize them as
strings and require matching live status plus ownership/session metadata before
reuse.

### Potential Issue: Missing Generation On Existing Rows Could Break Local Testing

Mitigation: support optional fields and an explicit legacy path. Reuse is
allowed only when reachable helper status proves the VM belongs to the same
`tldw/vz_linux` session. Otherwise, stale rows are replaced.

### Potential Issue: Fail-Closed Ambiguity Leaves Stale Rows Behind

Mitigation: this is intentional. Deleting rows when the helper cannot be
contacted or cannot speak the expected protocol can destroy recoverable state.
Operator reconciliation/repair remains the explicit cleanup path for stale
metadata.

### Potential Issue: Store Schema Drift Across Backends

Mitigation: update the abstract store, in-memory store, SQLite store, Postgres
store, and orchestrator facade together. Add tests for the backend-independent
contract and at least SQLite persistence.

### Potential Issue: This Becomes A Launchd Or Reboot Lifecycle PR

Mitigation: keep lifecycle ownership unchanged. The helper reports generation,
and the runner uses it. Process startup, launchd scaffolding, host reboot
recovery, and broader repair remain separate slices.

## Acceptance Mapping

- AC1: this spec defines the helper-generation/session-control contract and
  reviews lifecycle ownership and scope risks.
- AC2: reuse requires same-generation helper truth and live metadata checks.
- AC3: reachable stale state is replaced, while helper unavailable/protocol
  mismatch preserves rows.
- AC4: focused host-independent tests cover healthy reuse, generation mismatch,
  helper unavailable, and protocol mismatch.
- AC5: docs are limited to operator-facing session reuse/recovery expectations.
