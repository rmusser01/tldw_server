# Manual llama.cpp slot snapshots

Status: Approved by requester for implementation planning. No implementation in this change.
Task: TASK-13159. Baseline: origin/dev c5dfe0ff73.

ADR required: yes
ADR path: Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
Reason: Sensitive runtime-owned storage, restore semantics, and operation lifecycle.

## Scope and mental model

An administrator can manually save processed context from a server-managed
llama.cpp slot, list saved snapshots, restore a compatible snapshot into a chosen
slot, and delete saved snapshots. Automation comes later.

Snapshots are inference acceleration artifacts, not conversation backups. They
do not recover messages, tool state, or interrupted generation. Continuing in
Chatbook still requires the original conversation and a compatible prompt.
Cache reuse is not guaranteed by a successful restore or by selecting a provider.

Release one is admin-only, opt-in per managed profile, and restricted to a
single-model runtime on the supervisor's host. No external endpoints, router
mode, cross-profile restore, import/export, user-provided paths, scheduled saves,
automatic restore, or changes to existing Pause/Resume semantics. Pause continues
to stop the runtime; Resume starts it without restoring any snapshot.

## Evidence and gaps

The current supervisor in
`tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py` controls start,
stop, pause and resume with per-profile locks. Profiles have no per-user owner;
the existing profile API in `app/api/v1/endpoints/llamacpp.py` requires admin role.
The shared Admin entry is
`apps/packages/ui/src/routes/option-admin-llamacpp.tsx`.

Upstream exposes slot inspection and save/restore actions using a filename under
the launch-configured slot-save directory. tldw must provide its own catalog,
authorization, compatibility policy, and durable completion evidence. See
[official server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-slotsid_slotactionrestore-restore-the-prompt-cache-of-the-specified-slot-from-a-file)
(checked 2026-09-04). Implementation must record and test an exact upstream build;
support must not be inferred merely from the installed binary's version string.

This is a source-based design review, not a live usability test.

| Priority | Issue | Proposed solution / verifiable outcome |
| --- | --- | --- |
| P0 | “Resume” suggests session recovery | Keep process controls distinct; every restore explains that conversations are unchanged. |
| P0 | A slot can contain another user's context | All metadata and operations require admin; no prompt previews or downloadable binaries. |
| P0 | Stale UI can target a new process | Bind mutations to an opaque launch generation and reject stale requests. |
| P0 | Failed restore can leave uncertain slot state | Warn before mutation; never promise rollback; quarantine uncertain operations until the child exits. |
| P1 | First-time user cannot discover prerequisites | Inline enablement, restart-required state, and actionable capability diagnostics. |
| P1 | Users cannot judge whether a snapshot fits | Show Compatible, Incompatible, or Unknown with specific mismatch reasons. Only Compatible is actionable. |
| P1 | Large saves outlive an HTTP request or page visit | Durable operation receipt, bounded background work, reconnectable status. |
| P1 | Cache files consume disk invisibly | Show bytes and retention; prune only after a verified successful save. |
| P2 | Experts repeat discovery and diagnosis | Compact slot table, snapshot sorting, explicit refresh, stable IDs and sanitized diagnostics. |

## First-time workflow

1. Open Admin → llama.cpp → managed profile → Slot snapshots.
2. Read: “Save processed context to reuse later. Restoring does not change your
   conversations. Snapshots can contain sensitive context from this runtime.”
3. Enable manual snapshots. Show required launch changes and explain that an
   explicit restart interrupts inference. Do not restart as a toggle side effect.
4. After restart, capability checks show Ready or an actionable reason. A stopped
   runtime still permits catalog browsing and deletion.
5. Use Chatbook normally. Return to the Admin panel, refresh slot state, and choose
   an idle slot with processed context. Save creates a generated timestamp label.
   Slots are not labeled as conversations because that association is unproven.
6. After a later start of the same profile, select a compatible snapshot and an
   idle destination. Review replacement warning and confirm Restore explicitly.
7. On success show restored token count and “Open the original conversation in
   Chatbook to continue.” Do not automatically send a message or change a provider.

## Power-user workflow and wireframe

An operator uses the same surface without a wizard: inspect slots, save, compare
snapshot metadata, restore, and inspect an operation receipt. Timestamp labels
include timezone; exact IDs are copyable. Retention defaults to the newest 10
committed snapshots per managed profile, configurable from 1 to 1000.

```text
Admin / llama.cpp / Research model                    Running
[Stop] [Pause]                         existing process controls

Slot snapshots                              Enabled | Ready
Save processed context to reuse later. Conversations are unchanged.
Sensitive runtime context. Administrators only.

Slots                                       [Refresh]
Slot   State     Processed tokens             Action
0      Idle      8,192                        [Save snapshot]
1      Busy      2,048                        Save unavailable: busy

Saved snapshots             Keep newest [10]       Total: 1.2 GiB
Created (local timezone)     Tokens   Size     Compatibility   Actions
Sep 4, 14:32                8,192    600 MiB  Compatible      [Restore] [Delete]
Sep 3, 10:08                8,192    600 MiB  Model changed   [Details] [Delete]

Restore Sep 4, 14:32                              inline disclosure
Destination slot [0: Idle v]
This replaces the destination cache. Failure may also clear it.
Messages and tool state will not be restored.
[Restore into slot 0] [Cancel]

Latest operation: Restoring…  [Details]
Do not stop the runtime until this operation completes.
```

Use shared components, existing theme and density tokens. Preserve both themes:
operators may use this in bright daytime offices or dim overnight maintenance.
No new palette or decorative motion. Stack row fields on narrow screens, preserve
labels and keyboard access, announce status changes without moving focus, and
return focus to the initiating control after closing confirmation. Disabled
reasons must be readable without hovering. Confirmation controls name the target.
Deletion has an explicit confirmation naming the snapshot and permanent nature;
it deletes the saved copy, never erases an active slot.

## Proposed API

All routes live below `/api/v1/llamacpp/profiles/{profile_id}` and require the
existing admin and rate-limit dependencies, including reads and operation status.

| Method / suffix | Contract |
| --- | --- |
| GET `/slots` | Sanitized capability, launch generation, slot state and fresh signed request token; no prompt text. |
| GET `/snapshots` | Paginated committed metadata, total bytes and compatibility reasons. |
| POST `/snapshots` | Save: slot_id, expected_launch_generation, request_id. Returns 202 with operation_id. |
| POST `/snapshots/{snapshot_id}/restore` | Restore: destination slot_id, expected_launch_generation, request_id, explicit replace confirmation. Returns 202. |
| DELETE `/snapshots/{snapshot_id}` | Delete one saved artifact; reject if currently referenced by an operation. |
| GET `/snapshot-operations/{operation_id}` | Admin-only status, stage, stable error code and safe recovery action. |

Strict schemas reject extra fields and paths. IDs resolve within the selected
profile, never by filesystem path. Authentication precedes resource disclosure.
Use 409 for stale generation, busy profile or blocked lifecycle; 422 for invalid
input/incompatibility; 503 for unavailable owner/runtime. Read failures must not
masquerade as an empty catalog. A repeated request_id with identical input returns
the same receipt; different input returns 409. Reject expired keys rather than
silently replaying them. Record a retention window of at least 30 days for receipts
and refuse keys older than that window using a server-issued request token.
The signed token includes issuance time, profile and a random nonce; submit it as
request_id. GET `/slots` supplies fresh tokens without a filesystem write. Bind
receipt lookup to the authenticated admin boundary and profile, and check expiry
before treating a missing receipt as a new submission.

## Ownership, storage and lifecycle

Snapshots belong to a managed profile on one server installation, not to the
requesting administrator or Chatbook user. Audit actor IDs but never cache bodies,
raw runtime responses, credentials, or absolute storage paths.

Use a private service-owned snapshot root with a versioned per-profile catalog,
immutable generated snapshot IDs, and per-launch working directories. The child
receives only its working directory as its slot-save path, not the committed
catalog. Reject conflicting user launch arguments; do not silently override them.
Do not alter process-global umask. Verify restrictive directory/file permissions
or ACLs; unsupported confinement is a capability failure, not a security promise.
Local service-account/root access remains outside this boundary. No new claim of
encryption at rest; document disk-encryption and backup sensitivity.

Use atomic metadata publication after verified binary completion, file hashing,
flush and rename on the same filesystem. The catalog stores format version,
snapshot ID, source slot, UTC timestamp, monotonic commit sequence, byte/token
counts, SHA-256, compatibility fingerprint and creator audit ID. Crash recovery
ignores uncommitted files. Retention prunes oldest commit sequences only after a
new save commits; changing the limit alone does not delete files. Report partial
prune failures without pretending the successful save failed. Profile deletion
is blocked while snapshots exist; administrators explicitly delete snapshots
first. Disabling the feature preserves catalog browse/delete access.

Mutations are supervisor-owned bounded operations with durable receipts, not
replayable jobs (the proposed ADR records this narrow ADR-003 exception). A durable
dispatch marker is written before sending the upstream mutation. Recovery never
replays a dispatched request. Receipt creation and a per-profile reservation are
atomic under the owning supervisor; a filesystem/process ownership fence prevents
two application workers from owning one snapshot root. Non-owner workers reject
mutations rather than guessing a runtime. Multi-host snapshot transfer is excluded.

Introduce an opaque launch generation; PID, host and port alone are insufficient.
Reserve the profile against overlapping save/restore/delete and managed lifecycle
changes without holding a general lock across network I/O. Stop/Pause/Restart
return conflict during an operation; emergency process termination can still
occur and leads to interrupted/unknown status. Register operation shutdown with
the existing services lifecycle (ADR-021), not orphaned request tasks.

An idle observation is not a reservation against inference. Reject observed busy
slots and recheck immediately before dispatch. Do not claim snapshot isolation
from concurrent clients. Document that an operator must quiesce callers for a
predictable snapshot. No prompt/user attribution is inferred from a slot number.
Do not expose upstream slot-management routes directly to ordinary users.

Use the central checked egress transport with the captured server-owned runtime
origin, consistent with ADR-030. Never accept a URL in snapshot requests. Verify
generation and origin immediately before dispatch and before publishing success.

## Compatibility and failure contract

Fail closed if compatibility cannot be established. Fingerprint model contents,
projector contents when present, executable/build identity, effective cache/context
options, adapters and scales, and snapshot format. Mutable adapter state or model
files that cannot be proven stable make the runtime unsupported. Same filename,
model alias or matching token counts do not prove compatibility. No force-restore
override in release one. Multimodal restore is enabled only for explicitly tested
build/model/projector combinations; otherwise give an unsupported explanation.

Save sequence: validate → reserve generation → inspect slot → dispatch once →
validate acknowledgement and output file → hash → commit → prune → release.
Restore sequence: validate → reserve generation → copy committed file into launch
working directory while verifying its hash → recheck slot/generation → dispatch
once → verify acknowledgement → record result → release.

A pre-dispatch failure is safe to retry manually. A timeout, disconnect, shutdown,
or malformed response after dispatch yields Outcome unknown, never Success.
Do not publish or prune on an unacknowledged save. Do not retry a restore or promise
rollback. Quarantine the operation's working files and reject further mutations
on that launch until the child is confirmed dead; explicit stop is allowed in this
terminal unknown state and warns that inference will stop. On next launch reconcile
receipts without replay and clean only working files whose child is proven dead.

Use short capability probes (5 seconds) and a bounded 10-minute mutation deadline
including staging; connect timeout 5 seconds, write 30 seconds, read bounded by
remaining operation time. No fake progress percentage: show Validating, Saving,
Verifying, Restoring, Complete, Failed, or Outcome unknown. Closing the page does
not cancel execution. Disk exhaustion preserves previous committed snapshots.
Bound concurrent work to one mutation per profile and a server-wide configured
limit; check available space before copying without treating that check as a
guarantee. Filesystem-full errors remain handled at every write.

## Verification and delivery boundaries

Implementation planning follows review of this written design. Split reviewable
work into storage/compatibility, supervisor/API, and Admin UX with integration
evidence. These are proposed work packages, not completed or allocated task IDs.

Required evidence:

- Unit/property tests for path traversal/symlink rejection, metadata validation,
  hashes, fingerprints, retention order, and request-key deduplication.
- API tests for non-admin denial on every route, cross-profile IDs, rate limits,
  stale generation, wrong owner, duplicate submissions and lifecycle conflicts.
- Fault injection for crash before/after dispatch and commit, slow responses,
  partial writes, disk full, malformed acknowledgements and unknown outcomes.
- UI tests for first-use setup, stopped/unsupported/busy/error states, explicit
  confirmations, reload during work, keyboard focus and narrow-screen layout.
- Live save → stop → start → restore against a pinned supported llama.cpp build,
  followed by matching-prompt inference and a cold-cache control. Demonstrate
  actual cache reuse using runtime evidence, not HTTP 200, file existence or
  generated-answer similarity. Record text and any claimed multimodal coverage.
- Verify original Chatbook conversation and tools remain unchanged; verify
  Pause/Resume never silently save or restore.

Automatic checkpoint triggers, conversation-slot binding, cross-user isolation,
and scheduling are separate future designs. No hidden auto mode is shipped now.
