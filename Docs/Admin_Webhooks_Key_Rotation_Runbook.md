# Admin Webhooks Key Provisioning And Rotation Runbook

## Scope

Canonical webhook target URLs, signing secrets, protected event bodies, pending
incident markers, and secret-bearing idempotency replays use a dedicated AES-256
key ring. The ring is independent of BYOK, session, JWT, API-key, and general
application encryption credentials.

This runbook covers initial provisioning and forward-only rotation. It does not
rotate receiver signing secrets; use the canonical registration rotate-secret
action for those.

## Key-Ring Contract

`TLDW_ADMIN_WEBHOOK_KEYS_JSON` is a strict JSON object. Each key ID is 1-64
characters from `[A-Za-z0-9._-]`; each value is canonical base64 for exactly 32
random bytes. `TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID` names one key in that object.
Duplicate IDs, non-object JSON, malformed/noncanonical base64, wrong key sizes,
and an absent primary fail closed.

Schema example only. The placeholders intentionally fail key validation and
must be replaced from the deployment secret manager:

```bash
TLDW_ADMIN_WEBHOOK_KEYS_JSON='{"whk_2026_08":"<canonical-base64-of-32-random-bytes>","whk_2026_11":"<canonical-base64-of-32-different-random-bytes>"}'
TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID='whk_2026_08'
```

Generate production keys with a cryptographically secure 32-byte generator and
place them directly in the deployment secret manager. For example, on a trusted
operator workstation:

```bash
umask 077
openssl rand -base64 32
```

Do not commit, paste into tickets/chat, store in shell history, or log the key
ring. Secret-manager audit and access controls are the evidence source.

## Durable Primary Invariant

Migration state stores `active_primary_key_id`. Outside an active rotation,
every protected write requires the process's configured primary to equal this
durable ID. Reads may use any retained key in the ring; ordinary new writes use
only the configured primary.

A process with a missing ring returns the closed key-unavailable state. A
process whose local primary differs from durable state fails protected writes
with `admin_webhook_key_configuration_mismatch`. It never falls back to another
credential or plaintext.

## Initial Provisioning

1. Generate one 32-byte key in the approved secret manager and assign a stable,
   non-secret ID such as `whk_2026_08`.
2. Set `TLDW_ADMIN_WEBHOOK_KEYS_JSON` and
   `TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID` identically on every application and
   operator process.
3. Restart/drain processes according to the deployment runbook.
4. Read `/api/v1/admin/webhooks/status` directly from every node. Require
   `key_state=available`.
5. Complete the legacy migration. Migration completion binds the durable active
   primary to the configured key.
6. Preserve secret-manager version/audit references without recording key
   bytes.

Never initialize durable state by manually updating the database.

## Rotation Impact

Rotation is maintenance for every operation that writes protected values and
for any replay that would disclose a signing secret. In the complete canonical
runtime this blocks:

- create registration;
- destination replacement;
- signing-secret rotation;
- secret-bearing create/rotate replay;
- user event capture in its source database transaction;
- incident mutation/notification marker publication;
- automatic/manual event capture and delivery work that requires protected
  target, secret, or event-body access.

Metadata-only updates, disable, soft delete, redacted reads, and status remain
available where mode permits. In mode `on`, a protected-write key failure aborts
the source user/incident mutation rather than committing a domain change without
its canonical event.

Legacy import and key rotation are mutually exclusive. Start neither while the
other has an active durable phase.

## Rotation Procedure

The example rotates `whk_2026_08` to `whk_2026_11`. Keep both keys in the ring
for the entire procedure.

### 1. Preflight

Record:

- deployment, application SHA, database backend, and verified backup reference;
- every application/operator process;
- source and target key IDs, but no key bytes;
- unique operation ID, operator user ID, start time, and maintenance owner;
- current `rotation-status` output and per-node webhook status;
- current protected-write traffic and a plan to pause/retry it.

Require migration phase `complete`, no import in progress, no previous active
rotation, and no unexpected key/status errors.

Add the target key to the secret-manager ring on **every** process while keeping
the source as local primary:

```bash
TLDW_ADMIN_WEBHOOK_KEYS_JSON='{"whk_2026_08":"<source-base64>","whk_2026_11":"<target-base64>"}'
TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID='whk_2026_08'
```

Restart/drain all processes and verify each reports `key_state=available` before
starting. Do not change the primary yet.

Capture initial durable status:

```bash
tldw-admin-webhooks rotation-status
```

### 2. Start

Use one stable operation ID for the entire rotation:

```bash
tldw-admin-webhooks rotate-key start \
  --operation-id whkeyrot_2026_08_22_a \
  --source-key-id whk_2026_08 \
  --target-key-id whk_2026_11 \
  --operator-id 7
```

Start verifies that migration is complete, both keys are available, the durable
active primary and local primary equal the source, and no other rotation is
active. It durably enters `rewriting` before work proceeds.

### 3. Resume Rewriting

```bash
tldw-admin-webhooks rotate-key resume \
  --operation-id whkeyrot_2026_08_22_a \
  --operator-id 7
```

Resume processes bounded, committed batches across registration targets,
registration secrets, retained event bodies, unexpired idempotency replay
secrets, and pending incident-marker file entries. Database replacement and its
cursor commit together. File publication occurs before its cursor update, so a
crash may leave an already-target-encrypted marker that resume safely accounts
for once.

If start or resume is interrupted, keep both keys and the source primary, then
rerun `rotation-status` and the same `resume` command. Do not create a new
operation ID, reset durable cursors, rewrite the system-ops file manually, or
restore old ciphertext. Once any value has moved, the operation is
forward-resume only.

Resume returns immediately when durable phase has already advanced to
`verifying`, `awaiting_primary_cutover`, or `complete`.

### 4. Verify While Source Remains Primary

Confirm `rotation-status` reports `verifying`, then run:

```bash
tldw-admin-webhooks rotate-key verify \
  --operation-id whkeyrot_2026_08_22_a \
  --operator-id 7
```

Verification scans and context-decrypts the complete rotation-start inventory
in every protected database table and the locked pending-marker file. Every
envelope must name the target key, and the verified count must equal the durable
processed count. Success enters `awaiting_primary_cutover`.

Keep the local primary set to the source through this command. Keep both source
and target key bytes available. A count, context, file, or decrypt mismatch is a
hard stop; preserve evidence and resume/forward-fix instead of bypassing it.

### 5. Deploy The Target Primary Everywhere

While phase is `awaiting_primary_cutover`, protected writes remain blocked.
Change only the primary selector on every application, worker, and operator
process; retain both keys:

```bash
TLDW_ADMIN_WEBHOOK_KEYS_JSON='{"whk_2026_08":"<source-base64>","whk_2026_11":"<target-base64>"}'
TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID='whk_2026_11'
```

Drain/restart each process and prove directly that all of them received the
target primary. Do not rely only on a load-balanced sample. Do not admit
protected-write traffic before finalization.

At this boundary durable `active_primary_key_id` is still the source. That
temporary mismatch is expected only inside the active rotation. Finalize must be
run from an operator process whose local primary is the target. A source-primary
operator process cannot finalize.

### 6. Finalize

```bash
tldw-admin-webhooks rotate-key finalize \
  --operation-id whkeyrot_2026_08_22_a \
  --operator-id 7
```

Finalize requires `awaiting_primary_cutover`, local primary equal to the target,
and both keys present. It repeats the complete zero-source-envelope decrypt/
readback pass before atomically setting durable `active_primary_key_id` to the
target and phase to `complete`.

If finalize is interrupted, keep both keys and the target primary, inspect
`rotation-status`, and rerun the same finalize command. It is idempotent after
completion.

### 7. Verify Every Process And Remove The Source

Record final `rotation-status` and per-node `/webhooks/status`. Require:

- operation ID matches;
- phase `complete`;
- durable active primary is the target;
- processed and verified counts match;
- every node reports `key_state=available`;
- protected writes no longer return rotation/mismatch errors.

A lagging source-primary node now fails protected writes closed with
`admin_webhook_key_configuration_mismatch`; remove it from service and deploy
the target-primary configuration. It cannot reintroduce source ciphertext.

Only after finalization's complete zero-source-envelope verification and all-node
proof may the source key be removed from `TLDW_ADMIN_WEBHOOK_KEYS_JSON`. Roll the
target-only ring everywhere:

```bash
TLDW_ADMIN_WEBHOOK_KEYS_JSON='{"whk_2026_11":"<target-base64>"}'
TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID='whk_2026_11'
```

Recheck every node. Then retire the old secret-manager version according to the
approved key-retention policy. Retained database, file, snapshot, and backup
data encrypted under a removed key may become unrecoverable; verify backup and
legal-retention requirements before destruction.

Before retiring any source key, perform the coordinated incident-marker backup
and restore/readback proof in the migration runbook against a snapshot that
still contains both keys. Record which secret-manager versions are required by
each retained backup. A successful live zero-source-envelope scan does not prove
that an older retained backup no longer needs the source key.

## Failure Handling

| Condition | Required action |
| --- | --- |
| Key unavailable/invalid | Restore the exact approved ring; do not substitute BYOK/session/JWT/API-key material. |
| Key configuration mismatch before start | Reconcile local primary with durable source on every node. |
| Rotation already in progress | Use its recorded operation ID and resume; do not start a competitor. |
| Crash in `rewriting` | Keep source primary and both keys; rerun resume. |
| Crash in `verifying` | Keep source primary and both keys; rerun verify after resume reports rewriting complete. |
| Crash in `awaiting_primary_cutover` | Keep both keys; finish all-node target-primary rollout, then rerun finalize. |
| Lagging node after finalize | Remove it from service and deploy target primary; do not revert durable state. |
| Protected inventory/readback mismatch | Keep writes blocked, preserve both keys and evidence, and forward-fix. |

For a key loss while no rotation is active:

- leave mode `off` or move every node to `off` through the reviewed deployment
  path;
- preserve AuthNZ, Jobs, `system_ops.json`, and backup bytes unchanged;
- do not retry source mutations until the exact approved key version is
  restored and every node reports `key_state=available`;
- treat pending incident markers as durable encrypted work. The reconciler must
  remain failed closed and leave them byte-for-byte unchanged;
- after key recovery, run strict marker readback, `activation-check --phase
  predeploy`, one no-traffic live canary, and the reconciliation proof before
  admitting normal traffic.

Never write plaintext as an emergency measure, manually alter key IDs in stored
envelopes or migration state, delete pending file markers, or remove the source
key before final zero-envelope verification.
