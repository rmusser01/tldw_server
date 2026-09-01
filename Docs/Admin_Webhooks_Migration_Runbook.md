# Admin Webhooks Migration And Rollback Runbook

## Purpose And Safety Boundary

This runbook migrates both historical webhook sources into the canonical
registration tables:

- top-level `webhooks` and `webhook_deliveries` in `system_ops.json`;
- rows in the legacy `admin_webhooks` database table.

The importer is deterministic, dry-run first, crash-resumable, and imports every
registration inactive. It does not authorize concurrent legacy writes. Source
fingerprints detect drift; they do not make a partially drained deployment safe.

Use two people for report approval and any structural rollback. One is the
operator; the other independently reviews source mapping, rejection decisions,
artifact paths, and the literal report digest.

## Rollback Boundaries

1. **Before import:** revert application code normally; canonical schema changes
   are additive.
2. **After import and before canonical activity:** structural recovery may be
   possible while the rollback key is retained, the window is unexpired, and
   status reports `legacy_file_restore_permitted=true`.
3. **After any canonical mutation or delivery:** legacy restore is permanently
   closed. Set canonical mode `off`, disable delivery when available, preserve
   canonical tables, and forward-fix.

The durable first-canonical-activity marker, not an operator's recollection,
defines boundary 3. Do not delete or alter canonical tables to manufacture a
rollback.

## 1. Preflight

### 1.1 Capture The Change Record

Record all of the following before changing a node:

- deployment/environment name;
- current application image/tag and Git SHA;
- every application, worker, and administrative process/node;
- database backend and a verified provider/database backup reference;
- active `system_ops.json` path and a hash of its current bytes;
- operator user ID and second-person reviewer;
- intended report, encrypted-backup, rollback-key, and later extraction paths;
- `TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS` and expiry expectation;
- current values for the canonical mode, compatibility, limits, keys, and egress
  policy variables.

Do not print full webhook URLs, secrets, key-ring JSON, rollback-key content, or
legacy source JSON into the change record.

### 1.2 Verify Key And Path Preconditions

Provision the dedicated webhook key ring using
[Admin Webhooks Key Rotation Runbook](Admin_Webhooks_Key_Rotation_Runbook.md).
Do not use BYOK, session, JWT, API-key, or other application credentials as a
canonical fallback.

Use four distinct paths with distinct roles. Example locations:

```text
/var/lib/tldw-webhook-reports/import-2026-08-22.json
/var/lib/tldw-webhook-backups/import-2026-08-22.enc
/etc/tldw-webhook-rollback/import-2026-08-22.key.json
/var/lib/tldw-webhook-recovery/import-2026-08-22.plain.json
```

Requirements:

- each parent directory already exists, is owned by the effective operator, and
  is not group/world writable;
- report, backup, key, extraction, active data, and database paths normalize to
  distinct paths;
- the backup and rollback key are not in the same directory;
- final backup/key/extraction files do not already exist and are not symlinks;
- the encrypted backup, rollback key, and later plaintext extraction are outside
  application data directories;
- artifact storage and its backups follow the deployment's secret-retention
  policy.

The CLI exclusively creates report, backup, key, and extraction artifacts with
mode `0600`, fsync/readback, and no-follow checks. Do not pre-create the files.

### 1.3 Quiesce Every Writer

Choose one of these controlled states:

- drain and stop every process that can write the legacy database or
  `system_ops.json`; or
- roll every process to `TLDW_ADMIN_WEBHOOKS_MODE=migrate`, then keep product
  and operator mutation traffic drained for the import.

A single-node environment change is insufficient in a multi-process deployment.
Old application nodes, CLI sessions, background workers, and admin sessions are
all writers until proved otherwise.

For each node, bypass the load balancer and capture authenticated evidence:

```bash
curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  "${NODE_BASE}/api/v1/admin/webhooks/status"
```

Require `mode=migrate` and `route_selection=canonical` for a migrate rollout.
Canonical routes are always mounted, so do not use 404 responses as a selector
or compatibility probe. Mutations blocked by migrate mode return the canonical
bounded mode error. Also verify there is no direct legacy database/file writer
outside the HTTP processes. Record a signed
operator acknowledgement that **all writers are quiesced**. The CLI's
`--all-writers-quiesced` flag asserts this external fact; it cannot prove it.

## 2. Dry Run And Review

Set the service environment exactly as it will be used for apply, including
database configuration, canonical `migrate` mode, limits, dedicated key ring,
and egress policy. Use the same `--allow-legacy-credential-decryption` choice in
dry-run and apply. Enable that flag only when the reviewed legacy source truly
requires decryption under historical credentials.

```bash
tldw-admin-webhooks import-legacy \
  --dry-run \
  --report /var/lib/tldw-webhook-reports/import-2026-08-22.json \
  --backup /var/lib/tldw-webhook-backups/import-2026-08-22.enc \
  --rollback-key-file /etc/tldw-webhook-rollback/import-2026-08-22.key.json \
  --operator-id 7
```

The command prints a sanitized summary. Review the mode-`0600` report itself and
record:

- `operation_id` and `fingerprint_key_id`;
- whether legacy credential decryption was enabled;
- source fingerprints;
- every accepted source, deterministic canonical ID, redacted target, events,
  timeout, and `secret_rotation_required` state;
- every unresolved source and stable reason code;
- every explicit rejection and its operator/reason;
- projected non-deleted count against the configured limit;
- complete source-to-canonical mapping;
- `requires_system_ops_backup`;
- `report_digest`.

An unresolved source blocks apply. Repair the source while writers remain
quiesced, or make a durable, fingerprint-bound rejection from the approved
reason set:

```bash
tldw-admin-webhooks reject-source \
  --source-kind system_ops \
  --source-identity '<exact-redacted-report-identity>' \
  --source-record-fingerprint '<exact-report-fingerprint>' \
  --reason-code receiver_decommissioned \
  --operator-id 7
```

Allowed reasons are `receiver_decommissioned`, `duplicate_external_config`,
`invalid_legacy_record`, and `operator_excluded`. A changed source fingerprint
invalidates the decision. After any repair or rejection, publish a new dry-run
report and repeat review.

The reviewer must copy the final digest into a separate, immutable change record
and approve that literal value. Inspecting a digest and then deriving the apply
argument from the report in the same command is not approval. In particular,
never use `$(jq -r .report_digest "$REPORT")`, a pipe from the report, or a script
that reads the report to populate `--approved-report-digest` during apply.

## 3. Apply And Resume

Confirm again that all writers remain quiesced. Type or paste the independently
recorded literal digest into the command. The digest below is intentionally
fake and must be replaced with the reviewed 64-lowercase-hex value:

```bash
tldw-admin-webhooks import-legacy \
  --apply \
  --all-writers-quiesced \
  --approved-report-digest 'sha256:0000000000000000000000000000000000000000000000000000000000000000' \
  --report /var/lib/tldw-webhook-reports/import-2026-08-22.json \
  --backup /var/lib/tldw-webhook-backups/import-2026-08-22.enc \
  --rollback-key-file /etc/tldw-webhook-rollback/import-2026-08-22.key.json \
  --operator-id 7
```

When `requires_system_ops_backup=false`, omit `--backup` and
`--rollback-key-file`. If legacy credential decryption was approved in dry-run,
add the exact same `--allow-legacy-credential-decryption` flag to apply.

The importer performs these durable boundaries:

1. reserve the exact operation, approved digest, fingerprints, mapping,
   rejection decisions, and artifact identities;
2. create/read back the separate contextual rollback key;
3. create/read back the encrypted full-file backup and record its ciphertext
   digest;
4. insert canonical registrations and advance migration state in one database
   transaction;
5. decrypt/read back every imported registration and verify counts/mapping;
6. under the file lock, remove only legacy webhook fields and atomically publish
   the updated current object;
7. mark migration complete and retain artifacts for the bounded rollback window.

If the process exits or the host fails at any boundary, do not invent a cleanup
sequence and do not remove a staging/final artifact. Restore the same service
configuration, keep writers quiesced, and rerun the **same apply command** with
the same operation inputs, literal digest, report path, artifact paths, operator
ID, and legacy-decryption flag. The durable state and state-owned file identities
resume idempotently. A source change, path identity mismatch, report mismatch,
or different key configuration fails closed for review.

The report may be unavailable after the operation has durably reserved its
approved plan, but keep the reviewed report whenever possible. Never substitute
a newly generated report or new digest into an in-progress operation.

## 4. Post-Apply Proof

Before admitting any canonical management traffic, record all of the following:

1. The apply command returned phase `complete` and the expected operation ID.
2. Every node's authenticated status reports:
   - migration phase `complete`;
   - the expected imported and rejected counts;
   - `secret_rotation_required_count` matching imported registrations not yet
     rotated;
   - the expected rollback expiry and
     `legacy_file_restore_permitted=true` when file artifacts apply;
   - `key_state=available`;
   - no configured limit is exceeded.
3. Canonical list pagination contains every approved source mapping, all imports
   are inactive, target values are redacted, and imported secrets require
   rotation.
4. The active strict `system_ops.json` snapshot is valid JSON and no longer has
   top-level `webhooks` or `webhook_deliveries`; unrelated top-level fields are
   unchanged.
5. Legacy database rows remain intact; the importer does not drop or sanitize
   the legacy table.
6. When backup artifacts apply, record owner/group/mode/size and the encrypted
   backup's SHA-256 without printing rollback-key content:

```bash
stat -c '%U %G %a %s %n' \
  /var/lib/tldw-webhook-backups/import-2026-08-22.enc \
  /etc/tldw-webhook-rollback/import-2026-08-22.key.json
sha256sum /var/lib/tldw-webhook-backups/import-2026-08-22.enc
```

Successful completion proves the importer performed encrypted artifact
readback and canonical decrypt/readback. Preserve the command output, sanitized
status, source mapping review, artifact metadata, and ciphertext digest as the
migration evidence bundle.

Do not switch mode to `on` merely because import passed. Provision the key ring
and Jobs contract, rotate imported secrets, and pass the read-only
`tldw-admin-webhooks activation-check --phase predeploy` gate first.

## 5. Canonical Incident-Marker Backup And Restore

In canonical mode, `system_ops.json` may contain a top-level
`webhook_pending_events` list. Each entry contains source identity and an
encrypted incident event body. The reconciler commits that event and automatic
fanout to AuthNZ before removing the exact marker. These markers are current
durable work, not legacy migration fields and not disposable cache.

### 5.1 Backup And Readback

For a coordinated backup proof:

1. Preserve every webhook encryption key referenced by current database rows,
   retained backups, and pending markers.
2. Set canonical mode `off`, drain incident/user mutation traffic, and stop the
   webhook runtime on every node. Record that all writers and reconcilers are
   quiesced.
3. Take the approved AuthNZ database backup and copy the active
   `system_ops.json` from the same quiesced window. Keep the file backup private
   and record owner, group, mode, size, timestamp, and SHA-256 without recording
   its content.
4. Read back the copied file through the application's strict
   `_load_store_strict()` reader and parse `webhook_pending_events` through
   `_pending_incident_markers()`. Use an approved maintenance entry point with
   the same application version; do not use `jq`, regex, or permissive JSON
   parsing as integrity proof.
5. Record only the marker count, event IDs, event types, body key IDs, and the
   file digest. Do not print ciphertext or decrypt event bodies into evidence.
6. Prove the database backup is restorable through the database provider's
   normal restore drill and bind its restore reference to the file digest in
   the change record.

Quiescing both stores is the default because a database snapshot taken before a
marker's event commit plus a file snapshot taken after marker removal can lose
the recoverable source coordinate. Do not claim independent uncoordinated
snapshots form one consistent webhook recovery point.

### 5.2 Restore And Reconciliation Proof

Restore only into an isolated recovery environment first:

1. Keep canonical mode `off` and restore both the AuthNZ database and active
   file from the recorded coordinated set. Do not overlay only selected JSON
   fields.
2. Restore the exact key-ring versions required by the backup. Strict-read and
   parse every pending marker. A missing key, invalid ciphertext envelope,
   duplicate source, malformed marker, or unreadable file is a hard stop.
3. Compare the sanitized database migration/key state, file digest, incident
   versions, and pending marker inventory to the backup record.
4. Start one isolated mode-`on` canary with no external traffic. Require fresh
   reconciler/worker status, then allow the reconciler to drain the restored
   markers.
5. Prove each restored marker either inserted its canonical event and expected
   automatic fanout or matched the already committed source event exactly. The
   marker must disappear only after database commit. Capture IDs/counts and
   sanitized status, never plaintext bodies or destination data.
6. Repeat strict file readback and database checks, then destroy the isolated
   restore according to the provider drill procedure.

A file backup containing a marker can safely replay against a database that
already contains its exact event because source identity and body are verified
before exact marker removal. The reverse mismatch is unsafe: restoring an older
database with a newer file from which a marker was already removed can omit an
event. Escalate any asymmetric restore to incident review; never synthesize or
delete a marker manually.

If a required key is unavailable, remain in mode `off`, preserve the restored
bytes unchanged, and recover the approved key version. The reconciler fails
closed and must not skip, discard, or replace an undecryptable marker.

## 6. Structural File Recovery

Use this only after a failed migration decision and only while status reports
`legacy_file_restore_permitted=true`. The extraction command independently
enforces completed migration, retained artifacts, an unexpired window, and no
first canonical activity. A favorable status display does not bypass the CLI.

### 6.1 Stop And Reconfirm

1. Stop/quiesce every canonical and legacy writer on every node.
2. Set `TLDW_ADMIN_WEBHOOKS_MODE=off` everywhere; verify status per node.
3. Capture a fresh database backup and a fresh hash/copy of the current active
   `system_ops.json`.
4. Reconfirm `legacy_file_restore_permitted=true`, retained/unexpired artifacts,
   and no canonical activity.
5. Assign a second person to review the structural merge before publication.

If the status flag is false or extraction returns
`admin_webhook_rollback_window_closed`, stop. Keep canonical mode off and
forward-fix.

### 6.2 Extract To A New Private File

The output must not exist, must be outside application data, and must be in a
pre-existing private parent directory:

```bash
tldw-admin-webhooks extract-rollback-backup \
  --backup /var/lib/tldw-webhook-backups/import-2026-08-22.enc \
  --rollback-key-file /etc/tldw-webhook-rollback/import-2026-08-22.key.json \
  --output /var/lib/tldw-webhook-recovery/import-2026-08-22.plain.json \
  --operator-id 7 \
  --confirm
```

The command writes one new `0600` file and never emits plaintext to stdout.
Never redirect or pipe extraction output, reuse an old plaintext file, or place
the output beside the active store.

### 6.3 Review And Merge Structurally

Use the strict, bounded application reader for both the extracted snapshot and
the current active snapshot. Compare complete top-level structures. The second
person must verify that the proposed result:

- begins with the **current** active object;
- copies only top-level `webhooks` and `webhook_deliveries` from the extracted
  object;
- preserves every unrelated current top-level field and change;
- does not introduce duplicate JSON keys or a non-object root;
- does not alter or delete canonical database tables, rows, or migration state.

Publish only through the existing system-ops lock and atomic writer:
`admin_system_ops_service._STORE_LOCK`, `_store_file_lock()`,
`_load_store_strict()`, and `_atomic_write_store()`. Use a separately reviewed
recovery script or maintenance command that performs one lock-held read,
structural merge, atomic replace, file fsync, and parent-directory fsync. Do not
use shell text processing, direct `Path.write_text`, or whole-file replacement
with the extracted backup.

After publication, strict-read the active file, verify the two restored fields
and every unrelated field, restart only the intended legacy-compatible build,
and capture status and route-selection evidence.

### 6.4 Destroy Plaintext Recovery Material

Delete the extracted file as soon as verification completes, then fsync the
parent directory where operationally supported. Record deletion without
recording content. Filesystem deletion, even with overwrite attempts, may not
physically erase blocks on copy-on-write, journaled, SSD, snapshot, or backed-up
storage. The extraction directory and its backups must therefore be treated as
secret-bearing for their full retention lifecycle.

## 7. Retire Rollback Artifacts

The default rollback window is seven days and may be configured only from 1 to
30 days. Do not retain the active one-time rollback key past the approved
window. After expiry, with the same database and exact artifact paths:

```bash
tldw-admin-webhooks destroy-rollback-key \
  --backup /var/lib/tldw-webhook-backups/import-2026-08-22.enc \
  --rollback-key-file /etc/tldw-webhook-rollback/import-2026-08-22.key.json \
  --operator-id 7 \
  --confirm
```

The operation is durable and resumable: it records retirement in progress,
removes the key before ciphertext, fsyncs parent directories, and records
`retired`. If interrupted, rerun the same command. Never manually delete one
artifact first. A migration with no file backup returns a stable not-applicable
result.

Infrastructure snapshots may retain ciphertext or plaintext recovery blocks
according to separate backup retention. Destroying the one-time rollback key
makes its encrypted backup unusable but is not a claim of physical media erasure.

## Incident Stops

Stop the procedure and preserve evidence when any of these occurs:

- any writer cannot be accounted for or quiesced;
- any node reports a different mode;
- any node reports a route selection other than `canonical`;
- source fingerprints or the approved digest change;
- unresolved records remain;
- artifact ownership, mode, inode identity, or digest differs;
- the canonical source mapping/readback differs from the report;
- status reports canonical activity or closes restore eligibility;
- a command emits an unknown error code.

Do not repair by deleting migration state, canonical tables, backup artifacts,
or current source fields. Keep the service off and forward-fix from the retained
evidence.
