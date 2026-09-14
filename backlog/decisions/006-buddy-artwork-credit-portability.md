# ADR-006: Buddy artwork credit portability

Status: Accepted
Date: 2026-09-10
Task: TASK-13242
Extends: [ADR-005](005-independent-buddy-bindings-and-work-ownership.md)

## Context

A live import of the published Trenchcoat pack on dev 50c1f68957 discarded
`metadata/pack.json`'s `source_context.artwork` record. Independent copies and
native exports consequently lost the creator, source URL and license notices.
ADR-005 already requires Buddy-owned artwork attribution.

## Decision

Store the optional version-1 artwork record as `visual_manifest["tldw/artwork"]`
inside the existing manifest JSON. This travels with the artwork and asset
remapping, and needs no database
migration. Native imports accept the existing canonical JSON string in
`source_context.artwork`; exports restore that carrier and remove the internal
manifest field from the exported copy for Chatbook's strict manifest compatibility.
Export fingerprints include the optional artwork carrier explicitly. Credit-free
fingerprints retain their existing shape.
If both carriers are present, they must agree. Independent Buddy creation copies
the record into its owned `attribution.artwork` as well as its manifest.

The record contains only `version`, `creator`, `license`, `source_url`, and
`notices`. Reject unsupported versions, unknown fields, invalid text, malformed
carriers and conflicting claims. Bound the encoded record to 512 KiB, notices
to 64 KiB, creator to 512 bytes, license to 4096 bytes, and URL to 2048 bytes.
Source URLs must be credential-free public HTTPS references. They are never
fetched. Credits and notices are untrusted data, never instructions, execution
policy or authorization. Do not copy unrelated `source_context` properties.

Packs without credits retain their existing behavior. Do not invent attribution
for earlier imports: lost records can only be recovered by re-importing a
credited original and creating a new independent copy. Existing copies remain
independent and are not silently rewritten.

## Alternatives

- A Persona-level reference would break independent ownership after deletion.
- New database columns and migrations add backend work for optional structured
  metadata already supported by the manifest JSON storage contract.
- Preserving arbitrary source context would retain unrelated or sensitive data.
- A credits sidecar outside manifest fingerprints would let credit changes escape
  existing content identity checks.

## Verification

Exercise real preview, commit, copy, source removal, export and re-import; check
exact notices and original asset bytes. Cover malformed and conflicting records,
credit-free compatibility and the existing PostgreSQL storage adapter when
available. Record the published Trenchcoat HTTP round trip separately from unit
test evidence.
