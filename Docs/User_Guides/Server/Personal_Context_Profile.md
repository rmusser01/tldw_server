# Personal Context Profile on tldw_server

## Purpose and product boundary

tldw_server is the authenticated home peer and canonical server copy of a
Personal Context Profile. [Chatbook is the current full profile editing and
interview interface](https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/User_Guide/settings/personal-context-profile.md).
The server does not currently provide a complete standalone profile editor.

A standalone Chatbook profile and an existing server profile are independent
peers. They become one canonical linked profile only after the user reviews the
reconciliation plan and link completion succeeds. Do not treat an upload or a
matching account as proof that linking is complete.

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
5. With an active authenticated home server in Chatbook, open:
   **Settings → Data & Privacy → My Profile → Server sync → Link to home server**.
   Review any identity or semantic reconciliation and complete the link before
   profile changes are uploaded.
6. Use Chatbook to edit changes that are expected to travel through the current
   linked Sync flow. The REST API remains useful for server inspection and
   explicit server-local operations, but its ordinary edits are not a
   server-to-Chatbook publication path.

## What is shared

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After a successful reviewed link, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for eligible shared objects.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. The current linked flow publishes eligible Chatbook-originated manifest, scope, record, and proposal changes; purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

The home server wraps its Sync integrity key for authenticated registered Chatbook devices; this is not at-rest key sharing.

Ordinary server REST record/proposal mutations are not currently published to linked Chatbook clients.

The versioned [Shared Core
contract](https://github.com/rmusser01/tldw_server/tree/dev/packages/tldw_profile_core)
is the compatibility boundary used by both peers.

## Export, removal, and purge

Profile exports contain sensitive personal context. Confirm the intended scope,
destination, and access controls before exporting. Plaintext export requires
the exact confirmation `EXPORT PLAINTEXT`. Recovery export requires `EXPORT
RECOVERY` and enforces a minimum of 12 characters. That is only the validation
minimum, not a strength recommendation: use a long, unique passphrase and store
it separately from the exported envelope.

Removing a local Chatbook copy is owned by Chatbook. The server rejects purge
requests with `mode: local_copy` as `server_local_copy_unsupported`.

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
| **Offline or queued** | A Chatbook change remains in its local outbox because the home peer is unreachable or authentication failed. | Keep the local data, restore connectivity and credentials, then retry Sync and inspect Chatbook's outbox/status. | The server cannot inspect a device-local queue until the device delivers it. |
| **Capability not negotiated** | The peers do not share the required Personal Context domains, adapter/schema support, or readiness. | Upgrade or correctly configure the incompatible peer, then negotiate again. | There is no supported bypass; upload remains blocked until negotiation succeeds. |
| **Version conflict** | Both peers changed the same canonical object from different base versions. | Preserve the conflict and inspect the generic Sync conflict status and metadata before making another edit. | No dedicated Personal Context post-link resolver or automatic completion path exists. |
| **First-link semantic collision** | Different local and server record identities describe the same scope, kind, namespace, and subject during linking. | Compare the presented records in Chatbook's reviewed reconciliation and resolve them before completing the link. | This resolver is available only as part of reviewed first-link reconciliation. |
| **Post-link semantic collision** | Different record identities describe the same semantic key after linking. | Preserve both sides and inspect generic Sync conflict status and metadata. | No dedicated Personal Context post-link semantic-collision resolver or completion path exists. |
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
  avoid duplicating the record blindly, and use Chatbook for future edits that
  must use the linked Sync path. Retrying Sync alone cannot publish the REST
  mutation because no server-origin publication path currently exists.
