# ADR-002: Personal Context Profile Authority, Encryption, and Sync Boundary

Status: Accepted

Date: 2026-08-30

Amended: 2026-09-02

Related Task: [TASK-13144](../tasks/task-13144%20-%20Add-encrypted-server-Personal-Context-repository.md)

Supersedes: N/A

Superseded by: N/A

## Decision

tldw_server and Chatbook are peer implementations of one versioned Personal
Context Profile contract. The server extends each authenticated user's existing
`Personalization.db` with an encrypted canonical repository; it does not create
a second profile database or use Sync V2 as live canonical storage.

The shared v1 contract is pinned to `tldw-profile-core==0.1.0`, source commit
`fcb54d736aff7145bf91421fa5f57cf2c5e0ed6d`, and contract digest
`a1e0868dcd873a0c94eb0405934983466ceed68fced4b749489226d9932a5e9b`.
Conformance tests enforce the same schema, fixtures, canonical bytes, and
integrity tags in both applications. The server's supported runtime floor is
Python 3.11, matching the pinned contract.

The server requires an explicitly configured 32-byte master key. It wraps
separate random per-profile encryption and integrity keys; every immutable
object version receives a fresh random DEK. Missing, malformed, or changed root
key material locks existing content and never creates replacement profile keys.
Canonical bodies and semantic metadata remain encrypted at rest.

One repository owns canonical object SQL and one service will own mutations.
Authenticated APIs, migration, compatibility routes, runtime context, MCP
tools, and Sync adapters must use those boundaries. Migration is per-user,
forward-only, fenced, idempotent, and has no dual-write interval. Sync transports
canonical whole objects but is neither application authority nor live storage.

After linking, the server is the authoritative Personal Context manifest
sequencer while both runtimes may originate authorized semantic mutations.
Chatbook manifest advances remain speculative until their semantic commit is
accepted and republished by the server. Device-only-only commits create no
Sync outbox rows.

Every direct server semantic mutation commits canonical state, the manifest
advance, and encrypted source-publication rows atomically in
`Personalization.db`. Idempotent relay publishes deterministic envelopes under
a reserved home-authority device identity only after that commit. Pull performs
mandatory recovery relay, and a durable activation baseline plus publication
watermark protects existing links from races during upgrade.

Ongoing synchronization is event-driven and uses bounded persisted retry. It
extends the existing Sync V2 push, pull, and batched conflict-resolution
contracts rather than creating a parallel transport, permanent poll, or push
channel. Conflict candidates remain encrypted and pinned before cursor
advancement; only the affected object freezes.

## Context

The existing server Personalization layer stores user response style and
semantic memories in plaintext. Chatbook now has an encrypted local Personal
Context implementation and governed proposal workflow. The products need to
exchange the same user profile without translating between application-specific
records or allowing one product's runtime settings to become canonical data.

The server already provides per-user database isolation and backup ownership in
`PersonalizationDB`. Extending that database preserves authentication and
operator expectations while allowing a fenced migration from legacy tables.

## Alternatives considered

- Keep independent profile models and translate during Sync: rejected because
  translation loses identifiers, lifecycle semantics, and forward compatibility.
- Store canonical objects only in Sync V2: rejected because transport state is
  not a safe source of truth for application reads and writes.
- Add a separate profile database per user: rejected because it fragments
  backup, purge, migration, and path authority.
- Derive a master key from an API key or generate one on demand: rejected
  because credentials rotate independently and missing configuration must fail
  closed rather than destroy recoverability.
- Store semantic routing fields in clear columns: rejected because kind,
  visibility, state, and semantic keys disclose sensitive profile facts.

## Consequences

- Operators must provision, back up, and rotate an explicit server profile
  master key.
- Existing users require a fenced migration before canonical routes can operate.
- Queries decrypt bounded candidate objects rather than filtering sensitive
  semantic fields in SQL.
- Cross-application compatibility is enforced by a fixed contract version and
  digest, not by best-effort projection.
- Sync domains are added only after both local repositories and migration are
  complete.
- Capability version `1` may be advertised only after atomic source
  publication, recovery relay, reserved authority identity, conflict
  extensions, purge fencing, and activation baselines are ready.
- Links created before ongoing synchronization require an explicit activation
  baseline and canonical home-manifest checkpoint; capability advertisement
  alone does not activate them.
- Cross-database publication is journaled, not atomic. A source row is
  acknowledged only after its deterministic envelope is durable in Sync V2.
- The server-generated OpenAPI/JSON Schema fragment is the wire-contract
  authority for Personal Context capability, conflict, and server-origin
  extensions; Chatbook vendors it with source commit and checksum.

## Links

- [Server design](../../Docs/Design/2026-08-30-personal-context-profile-server-design.md)
- [Ongoing synchronization amendment](../../Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md)
- [Chatbook source specification](https://github.com/rmusser01/tldw_chatbook/blob/main/Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md)
