# 003 — Bundled pixel-migu ownership and first-run seeding

Date: 2026-09-05. Status: Accepted. Task: TASK-13196.

pixel-migu ships as two independent resources: an optional Persona/Buddy starter
and a character with an active Shared Visual Identity expression version.
The starter is copied through the existing draft workflow; it does not change
Buddy selection, enablement, or activation.

For SQLite (the existing Shared Visual Identity backend), prepare the character
before publishing the per-user ChaChaNotes instance. Copy and validate local
assets through the existing storage service. Create the character, ready draft,
active version, binding and completed `builtin_character` idempotency record in
one immediate transaction. The permanent receipt is keyed by owner and seed ID;
replays return without updating any user content. Keep that receipt when a user
renames, customizes, unbinds or deletes the character/pack. A name collision is
recorded as skipped so an existing user character is never adopted or modified.
No schema migration is needed: the existing completed idempotency records have
no expiry and no character/pack foreign key.

The receipt uses a stable seed key, not a content-version key: future bundled-art
changes must not reset customization or deletion. Do not use name lookup as the
replay identity. A failure rolls back all database rows; content-addressed files
may remain reusable after rollback, as in existing Visual Identity imports.
PostgreSQL remains unchanged because Shared Visual Identity currently requires
SQLite. No runtime or storage backend expansion is included.

Alternatives rejected: replacing same-name characters loses user ownership;
rechecking names on each startup resurrects deleted/renamed seeds; globally
shared mutable pack rows violate per-user ownership; a new seed table duplicates
an existing permanent receipt facility.

References: `Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md`,
`Docs/superpowers/specs/2026-05-09-persona-visual-ownership-copy-design.md`.
