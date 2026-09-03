# Personal Context Profile — Server Canonical Storage

Status: Accepted

Date: 2026-08-30

Related task: TASK-13144

Related ADR: `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

Ongoing-sync extension:
`Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md`

## Purpose

The server is one encrypted canonical peer for the same Personal Context
Profile used by Chatbook. A record created by either application has the same
schema, canonical bytes, identifiers, lifecycle, and proposal semantics. The
existing per-user `Personalization.db` remains the server storage owner.

This design covers server key custody and the encrypted repository foundation.
Authenticated APIs, legacy migration, compatibility routes, runtime context,
agent tools, and Sync transport are separate reviewable tasks built on this
boundary.

## Shared contract pin

Both applications use `tldw-profile-core` contract version `0.1.0`. The
accepted Chatbook source snapshot is:

- source commit: `fcb54d736aff7145bf91421fa5f57cf2c5e0ed6d`
- contract digest: `a1e0868dcd873a0c94eb0405934983466ceed68fced4b749489226d9932a5e9b`

The digest covers each relative path plus its bytes for the package metadata,
Python contract modules, JSON Schema, and all v1 conformance fixtures in
deterministic path order. Until the package
has a project release channel, the server carries that exact pure-contract
snapshot under `packages/tldw_profile_core`; a parity test recomputes the digest
and rejects any unversioned drift. Application storage, HTTP, UI, runtime,
authentication, and key-custody code are not part of the shared package.
The server runtime floor is Python 3.11 so every supported installation can
import the pinned contract's `StrEnum`-based models.

## Storage authority

`PersonalizationDB.for_user()` resolves the authenticated user's existing
database path. Its schema gains encrypted key, object-version, object-head,
runtime-policy, and receipt tables. No second user database and no Sync-owned
canonical table is introduced.

One repository owns object-table SQL. It stores cleartext only for opaque
routing identifiers, parent/version linkage, key/schema versions, timestamps,
and byte sizes. Manifest, scope, kind, semantic key, lifecycle, controls,
provenance, proposal content, Undo content, runtime policy, and future outbox
bodies are encrypted.

Each mutation uses `BEGIN IMMEDIATE` on the existing database wrapper and
commits an immutable version plus its compare-and-set head in one transaction.
The repository never claims atomicity with authentication, Sync, logs, exports,
or other database files.

## Key custody and encryption

The only v1 server root-key source is
`TLDW_PERSONAL_CONTEXT_MASTER_KEY`, containing strict base64 for exactly 32
bytes. A missing, malformed, or changed key locks existing profile content. It
never generates a replacement key for an existing profile. Surviving object,
head, runtime, or receipt rows without their profile-key row also lock storage
rather than permitting a new profile to orphan recoverable ciphertext.

Each profile has independent random 32-byte envelope-encryption and integrity
keys. The server master key wraps them with AES-256-GCM and random 96-bit
nonces. Every immutable object version receives a fresh random 32-byte DEK,
which encrypts canonical object bytes and is itself wrapped by the profile
encryption key. Associated data binds profile, object type, object ID, object
version, and envelope schema version while remaining stable across wrapping-key
rotation. The keyed canonical integrity tag is verified before model
parsing.

Key material, plaintext bodies, ciphertext, and raw crypto exception values are
never logged or returned. SQLite secure deletion is enabled. Privacy tests scan
the database, WAL, SHM, logs, diagnostics, temporary paths, and exception text;
WAL remains enabled so repository privacy cleanup cannot break concurrent read
snapshots.

## Operational boundaries

- Authentication and user-database resolution happen before profile lookup.
- Profile and object IDs are always paired with the authenticated profile.
- Cross-user identifiers become the same not-found result as an unknown ID.
- Tombstones and terminal proposal receipts contain no prior canonical body.
- Ordinary encryption-key rotation rewraps DEKs; integrity-key rotation is a
  separately versioned full rebaseline.
- New server routes, migration, legacy compatibility, runtime context, MCP
  tools, and Sync adapters must use the service/repository boundary rather than
  direct object-table SQL.

## Rejected alternatives

- A second Personal Context database: rejected because it fragments per-user
  backup, migration, authentication, and lifecycle ownership.
- Sync V2 as the live store: rejected because transport is not canonical
  application storage.
- Automatic key generation when the configured key is missing: rejected
  because it would silently replace access to existing encrypted data.
- Cleartext routing columns for kind, visibility, semantic keys, or lifecycle:
  rejected because they disclose profile meaning at rest.
- Disabling WAL to erase historical frames: rejected because it breaks the
  existing reader/writer concurrency contract.
