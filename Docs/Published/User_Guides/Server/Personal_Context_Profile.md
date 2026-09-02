# Personal Context Profile on tldw_server

## Purpose and product boundary

tldw_server is the authenticated home peer and canonical server copy of a
Personal Context Profile. [Chatbook is the current full profile editing and
interview interface](https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs%2FUser_Guide%2Fsettings%2Fpersonal-context-profile.md).
The server does not currently provide a complete standalone profile editor.

A standalone Chatbook profile and an existing server profile are independent
peers. They establish one linked profile identity, with a matching eligible
snapshot, only after the user reviews the reconciliation plan and link
completion succeeds. Later edits do not currently preserve that equality. Do
not treat an upload or a matching account as proof that linking is complete.

## Prerequisites and master-key setup

Personal Context routes require successful server authentication with an API
key or JWT. Authentication happens before the server resolves the current
user's storage. Each authenticated user can have one manifest in that user's
`Personalization.db`.

Configure `TLDW_PERSONAL_CONTEXT_MASTER_KEY` before starting the server. Its
value must be strict base64 that decodes to exactly 32 bytes. Generate a fresh
random value with an installed Python 3.11 or newer interpreter:

```bash
python3 -c 'import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode("ascii"))'
```

On Windows PowerShell, use `py -3.11` instead of `python3` (or invoke another
installed, supported Python interpreter explicitly).

Store the output with your protected server secrets and make a secure backup
before creating the first profile. A missing, malformed, or changed key fails
closed: existing profile content becomes locked, and the server does not create
replacement profile keys. Restore the exact original key from backup instead
of attempting to create another profile.

### Trust and transport boundary

Personal Context Sync currently uses `server_trusted_v1`: the authorized home
server can read syncable canonical profile content to validate and materialize
it, then encrypts it at rest. Peer-local at-rest keys protect each peer's stored
copy but are not end-to-end encryption from the authorized home server.

For any non-loopback connection, expose the API only through authenticated
TLS/HTTPS. Otherwise API keys or JWTs and profile content can cross plaintext
transport. Follow the [production hardening checklist](Production_Hardening_Checklist.md)
for reverse-proxy and TLS setup.

That deployment recommendation is not enforced by Chatbook. Chatbook accepts
configured server URLs that use either `http://` or `https://`. Runtime requests
verify TLS by default, but **Settings → Data & Privacy → Network** can select a
custom CA or disable verification. **Test Connection** always uses httpx's
default verification rather than the saved custom-CA or disabled-verification
runtime policy. Operators must therefore secure the server endpoint and ensure
the saved client policy matches the deployment.

## Setup and status workflow

Endpoint request and response details are in the [Personal Context API
reference](../../API-related/Personal_Context_API.md).

1. Configure API-key or JWT authentication and the master key before the server
   starts.
2. Authenticate and call `GET /api/v1/personal-context/status`. Remediate a
   `locked` or `unsupported` state before attempting writes. If the state is
   `purge_pending`, stop: the profile is non-writable and has no current
   completion path.
3. Call `GET /api/v1/sync/capabilities` and confirm that Personal Context is
   available with the required domains, schema version, and quotas. Do not
   bypass capability negotiation.
4. Inspect `GET /api/v1/personal-context/manifest`. If a server-side profile is
   intentionally needed and none exists, `POST /api/v1/personal-context/manifest`
   creates the one allowed manifest and its required global scope.
5. In Chatbook, create or unlock the profile through **Settings → Data &
   Privacy → My Profile**. Then activate and authenticate the server through
   **Settings → Overview → Advanced / Diagnostics → Switch Source / Server**.
   Return to **My Profile → Link to home server**.
6. Before approval, bootstrap exchanges authentication and capability,
   device-registration and public-key, display, schema/quota, and purge-generation
   metadata. It also downloads the server's current Sync-eligible canonical
   snapshot, including record and proposal content, into transient Chatbook
   memory. Durable review state and the visible plan remain content-free: the UI
   shows identifiers, versions, counts, outcomes, and choices rather than profile
   values. No local profile record or proposal content is uploaded before
   approval.
7. Review and approve the content-free reconciliation plan. Treat the peers as
   converged only after link completion publishes the resulting eligible
   snapshot successfully.
8. Do not rely on the link for later changes. Chatbook creates encrypted local
   outbox entries for later syncable mutations, but the shipped app has no
   ongoing Personal Context sync caller. **Overview → Manual Sync** covers Notes
   and Chat only. Ordinary server REST edits likewise remain server-local because
   they are not published into the Personal Context Sync log.

## What is shared

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

| Published during successful reviewed first linking when eligible | Not published by the shipped ongoing application lifecycle |
| --- | --- |
| Canonical manifest in the snapshot resulting from the user-approved content-free reconciliation plan | Later syncable Chatbook mutations: encrypted outbox entries are created but no shipped ongoing Personal Context caller sends them |
| Required global and linked-workspace scopes in that snapshot | Ordinary server REST mutations: the server copy changes but no Personal Context Sync entry publishes them to Chatbook |
| Eligible record heads, tombstones, and proposal review state selected by reconciliation, including approved interview answer content after it becomes a canonical record payload | Device-only or non-syncable records |
| Exact canonical object identities, versions, and bytes for those eligible objects | Runtime agent authority grants, tool availability, local workspace mappings, and enablement |
| — | Peer-local at-rest encryption/recovery keys, local undo data, caches, ciphertext, database row identities, conflict-review metadata, acknowledgement tracking, and other operational state |
| — | Encrypted interview draft and transcript objects are not Sync payloads as such; adaptive interview requests still send prior raw answers to the configured provider, while approved answer content may become a syncable canonical record as described at left |

The home server wraps its Sync integrity key for authenticated registered Chatbook devices; this is not at-rest key sharing.

Ordinary server REST record/proposal mutations are not currently published to linked Chatbook clients.

The versioned [Shared Core
contract](https://github.com/rmusser01/tldw_server/tree/dev/packages/tldw_profile_core)
is the compatibility boundary used by both peers.

## Chatbook interview privacy boundary

Fixed interviews generate questions locally and make no model call. Their
encrypted draft and transcript objects remain peer-local. Adaptive interviews
call the configured default Console provider without tools. Each request sends
the interview audience, allowed topics, attempt number, and eligible
agent-visible records from the exact selected scope. After the first answer,
requests also include all prior answered turns, including raw answer text.

The interview screen reveals the actual provider and model only after the first
provider response finishes, before answer entry is enabled. Use fixed mode when
no model egress is acceptable. In either mode, approved answer text can become
an ordinary canonical record payload; its selected visibility and syncability
then determine whether it is eligible for first-link publication or remains
device-only. A later syncable edit only enters Chatbook's currently undrained
encrypted outbox.

## Export, removal, and purge

Profile exports contain sensitive personal context. Confirm the intended scope,
destination, and access controls before exporting. Plaintext export requires
the exact confirmation `EXPORT PLAINTEXT`. Recovery export requires `EXPORT
RECOVERY` and enforces a minimum of 12 characters. That is only the validation
minimum, not a strength recommendation: use a long, unique passphrase and store
it separately from the exported envelope.

Removing a local Chatbook copy is owned by Chatbook. **Remove local profile**
deletes canonical `PersonalContextRepository` content, including that
repository's encrypted objects and heads, canonical `encrypted_outbox`,
quarantine rows, runtime policy, local mappings, Undo data, and record-link
metadata. It does not delete the server copy or unregister the device. Separate
`SyncStateRepository` state can remain, including link/profile state, staged
encrypted envelopes, remote heads and cursors, and conflict reviews or receipts;
dataset staging keys can remain too.

Chatbook's encrypted recovery export includes canonical local heads, including
device-only records, but the shipped app has no import or restore control.
Canonical profile-key deletion happens after repository rows are removed and
can fail; **Finish secure removal** retries only that canonical key cleanup. It
does not clear separate Sync state, staging keys, the server copy, or device
registration. The server rejects purge requests with `mode: local_copy` as
`server_local_copy_unsupported`.

`POST /api/v1/personal-context/purge` is a global, currently incomplete
operation. It requires `mode: everywhere`, the current purge generation, and
the exact confirmation `DELETE EVERYWHERE`. It advances a server-local purge
fence, removes canonical bodies and server runtime state, blocks further
profile mutations, and leaves the profile in `purge_pending`.

The `personal_context.purge` protocol domain exists.

The server purge endpoint does not publish the protocol purge envelope, and acknowledgement completion is not wired.
Reconnecting devices does not currently clear `purge_pending`, and the current
server has no completion path for that state.

## Troubleshooting

| State | Cause | Safe next action | Current limit |
| --- | --- | --- | --- |
| **Profile locked** | The configured master key is missing, malformed, changed, or cannot decrypt stored key material. | Stop writes, preserve `Personalization.db`, restore the exact original key from secure backup, and restart before checking status again. | There is no server-side bypass or automatic key-recreation path for existing ciphertext. |
| **Adaptive interview privacy or provider failure** | Adaptive mode sends bounded interview context to the default Console provider, and the first request completes before the screen displays the actual provider and model. | Use fixed mode when no model egress is acceptable. If adaptive mode fails, continue ordinary setup and retry later or use fixed mode. | Do not assume that the first adaptive request stayed local. |
| **HTTP or altered TLS verification** | Chatbook accepts HTTP and HTTPS; runtime verification can use default trust, a custom CA, or disabled verification, while **Test Connection** always uses default verification. | Prefer HTTPS with verification enabled and ensure the saved Network policy matches the deployment. | A successful probe does not prove that later runtime requests use the same trust policy. |
| **Post-link change queued** | Chatbook stored the local mutation and encrypted outbox entry, but the shipped app has no ongoing Personal Context sync caller. | Preserve the local profile. Do not claim the server copy changed or direct the user to **Overview → Manual Sync**, which covers Notes and Chat only. | No supported Settings action currently drains this queue. |
| **Capability not negotiated** | The peers do not share the required Personal Context domains, adapter/schema support, or readiness. | Upgrade or correctly configure the incompatible peer, then negotiate again. | There is no supported bypass; upload remains blocked until negotiation succeeds. |
| **First-link publication interrupted** | Reconciliation was approved but link completion did not finish. | Preserve both copies and retry the reviewed link flow. | Do not treat the profiles as converged until completion succeeds. |
| **Version conflict** | Both peers changed the same canonical object from different base versions at the transport boundary. | Preserve the generic Sync conflict metadata. | A shipped ongoing Personal Context cycle and dedicated Settings resolver/status are both absent. |
| **First-link semantic collision** | Different local and server record identities describe the same scope, kind, namespace, and subject during linking. | Use the content-free IDs, versions, outcomes, and local/server choices in the reviewed plan to select which canonical lineage remains active. | The plan does not display profile values, and this resolver is available only during first linking. |
| **Post-link semantic collision** | Different record identities describe the same semantic key after linking. | Preserve both peer copies. | No ongoing Personal Context cycle or dedicated status/resolver is currently shipped. |
| **Local removal incomplete or residual state** | Canonical repository rows were removed, but canonical profile-key cleanup may have failed; separate Sync state and dataset staging keys may remain. | Use **Finish secure removal** only for canonical profile-key cleanup. Preserve residual state until a supported cleanup path exists. | The action does not clear separate Sync state, staging keys, the server copy, or device registration, and the recovery export cannot currently be restored in the app. |
| **Purge pending** | The server purge fence advanced and ordinary profile mutations are blocked. | Preserve operational evidence and treat the profile as non-writable; do not recreate it or assume reconnecting clients will finish deletion. | The current server has no purge acknowledgement-completion path, and reconnecting devices does not clear the state. |

Additional checks:

- **Authentication failure:** verify the deployment's auth mode and provide a
  valid `X-API-KEY` or bearer JWT for the intended user. The server cannot
  expose profile status or recovery details before authentication succeeds.
- **Missing or changed master key:** restore the exact key used when the profile
  was created. Creating a new value is not rotation and cannot decrypt the
  profile; no automatic recovery path exists.
- **Schema or quota incompatibility:** use the content-free bootstrap error
  details to compare required and available values, then upgrade the
  incompatible peer or reduce unsupported requirements before retrying. Link
  completion cannot bypass `personal_context_schema_incompatible` or
  `personal_context_quota_incompatible`.
- **REST edit absent from Chatbook:** ordinary server REST record and proposal
  mutations are not published to linked clients. Preserve the server state,
  avoid duplicating the record blindly, and choose one peer as the manual
  editing authority until publication is implemented. Neither Chatbook Manual
  Sync nor retrying another Sync action can publish the REST mutation because
  no server-origin publication path currently exists.
