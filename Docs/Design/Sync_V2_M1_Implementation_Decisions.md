# Sync v2 M1 Implementation Decisions

Date: 2026-05-23
Status: Locked for M1 implementation
Source: `Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md`

## Scope

This document locks the planning-gate decisions for Sync v2 Milestone 1 before
production code changes begin. M1 covers server-connected Chatbook modes only.
Chatbook remains local-only when the user never connects it to a server.

M1 is limited to the authenticated user's personal dataset. The public M1
domains are exactly:

- `notes.note`
- `chat.conversation`
- `chat.message`
- `attachment.ref`

`attachment.ref` is metadata only in M1. Binary/blob transfer is not implemented
in M1, and restore preview must warn when a referenced blob is not available.

## Decision Summary

| Decision | M1 contract |
| --- | --- |
| Sync v2 envelope and state store | Per-user `Databases/user_databases/<user_id>/Sync_v2.db`. |
| Live server projections | Per-user `Databases/user_databases/<user_id>/ChaChaNotes.db`. |
| Profile setup | Explicit `POST /api/v1/sync/profile/bootstrap`. |
| At-rest encryption posture | `server_trusted_v1`, backed by deployment-level coverage for the user database directory. |
| Public domains | Only `notes.note`, `chat.conversation`, `chat.message`, and `attachment.ref`. |

## 1. Per-User Sync DB For Envelopes And State

Sync v2 M1 stores append-only envelopes and sync-owned state in:

```text
Databases/user_databases/<user_id>/Sync_v2.db
```

This database is the source of truth for:

- accepted envelopes
- client device/profile registration state
- default personal dataset metadata
- server cursors and object revision state
- idempotency records by client envelope ID and client sequence
- conflict records
- projection apply status
- replay and repair bookkeeping

Sync v2 personal envelope logs are not stored in the AuthNZ database. Keeping
the envelope log in a per-user database preserves the existing user-content
storage boundary and keeps cross-user isolation straightforward.

## 2. Per-User ChaChaNotes DB For Materialized Projections

Accepted M1 envelopes are materialized into the user's normal Notes and Chat
projection database:

```text
Databases/user_databases/<user_id>/ChaChaNotes.db
```

The append-only Sync v2 envelope log remains authoritative for restore, audit,
replay, and repair. `ChaChaNotes.db` is the live server projection used by the
normal server Notes and Chat APIs.

When Sync v2 is active for a server-connected user, server-origin personal
Notes and Chat mutations must route through Sync v2. A materialized projection
must not be created without a corresponding accepted envelope record. If a
projection write fails after envelope acceptance, the envelope records a failed
or conflict apply status so replay/repair can rebuild the projection from
`Sync_v2.db`.

## 3. Explicit Bootstrap Endpoint

`GET /api/v1/sync/profile` is read-only. It reports the current profile,
capabilities, dataset, cursor, and status, but it must not create durable sync
state.

`POST /api/v1/sync/profile/bootstrap` is the only M1 endpoint that implicitly
creates the default personal dataset. Bootstrap is idempotent and performs these
actions for the authenticated user:

- registers or refreshes the client device
- accepts a stable client-supplied `device_id` when provided
- generates a `device_id` only when the client omits one
- records the optional `client_profile_id`
- creates or returns one active default personal dataset
- marks that dataset with `default_personal: true`
- marks the dataset `client_family` as `chatbook`
- returns the initial server cursor and supported M1 domains

Chatbook clients must persist a returned server-generated `device_id` before
pushing envelopes.

## 4. `server_trusted_v1` At-Rest Encryption Attestation

M1 uses the encryption policy name `server_trusted_v1`.

This is a deployment-level attestation, not a per-field client-key scheme. The
server may advertise M1-ready `server_trusted_v1` only when the configured
deployment attests at-rest encryption coverage for the user database directory
that contains both:

- `Sync_v2.db`
- `ChaChaNotes.db`

Normal authenticated server access unlocks data for trusted or self-hosted
deployments. This is required so server-front-end Chatbook use can read the
materialized Notes and Chat projections through normal server APIs.

Sync profile and capability responses must report the policy, readiness, and
attestation scope explicitly. If the deployment has not attested coverage for
the full per-user database directory, Sync v2 must report the policy as not
ready for M1 rather than implying that only the envelope log is protected.

Later stricter encryption modes can be added without changing the M1
`server_trusted_v1` contract.

## 5. M1 Domains Only

M1 public capabilities must advertise only these domains:

- `notes.note`
- `chat.conversation`
- `chat.message`
- `attachment.ref`

Domain behavior is fixed for M1:

- `notes.note`: whole-object upsert and tombstone with base-state conflict
  checks.
- `chat.conversation`: whole-object conversation metadata upsert and tombstone
  with base-state conflict checks.
- `chat.message`: append and tombstone by stable message ID, with duplicate
  payload hash handling.
- `attachment.ref`: attachment reference metadata only; no blob upload,
  download, or hydration in M1.

Additional domain adapters may exist in the repository as dormant future work,
but they must not be part of the M1 default registry, public capabilities, or
bootstrap dataset domain list.

## Implementation Invariants

- Local-only Chatbook profiles never require server sync setup.
- Server-connected Sync v2 state is scoped to the authenticated user.
- The append-only envelope log is retained for restore and replay.
- Tombstones are first-class envelopes and must not be collapsed away.
- Restore preview must warn about `attachment.ref` entries whose blobs are not
  available through the M1 server contract.
- Accepted Notes and Chat envelopes must be materialized through
  DB_Management-owned projection APIs.
- Projection state can be rebuilt from accepted envelopes.
