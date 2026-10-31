# Email core correctness follow-up

Tracking: TASK-13251 (identity), TASK-13253 (cursor API). User authorization: address identified issues and continue, 2026-09-13.

## Identity

The legacy Media writer currently merges by upload URL or body hash before normalized email identity is evaluated. Use a deterministic email identity URL scoped to tenant, provider and source, selecting provider message ID, RFC Message-ID, then body hash only when no ID exists. Email writes must not fall through to generic body dedupe. Keep generic media behavior intact. Reuse compatible existing normalized/legacy records on reimport; do not automatically repair already-lost content. Guard normalized media-ID fallback against changing tenant/source/message identity.

Upload sources retain their existing naming contract: file name for EML, container/member reference for archive and nested children. Renaming a source therefore creates a separate source; it must not destroy the original source. Preserve source identity in stored metadata. Preserve repeated imports within a source, including when native persistence is disabled. Test same bodies with different IDs, repeated names, source isolation, missing IDs and consistency of message detail.

Reimports with overwrite disabled must preserve the accepted Media content in the
normalized email store as well. Normalize provider ISO timestamps and RFC dates to
UTC; invalid or overflowing dates must not abort ingestion. Generic document body
dedupe must exclude email rows so documents cannot overwrite their identities.

When an overwrite is accepted, carry email metadata into the new document version.
Update highlight state using the same Media transaction connection so Collections
initialization cannot block it and a rollback also restores highlights. For genuine legacy versions
whose allowlist omitted email fields, preserve richer canonical normalized metadata
when it exists. Live connector labels remain mutable independently of content.

## Cursor API

Add opt-in keyset pagination to the existing search endpoint while preserving current offset clients. Order by normalized internal date descending with nulls last, then internal message ID descending. Bind continuation position to query and tenant, validate malformed/incompatible parameters, and reapply tenant/deletion filters on every page. Freeze relative-date filters for a traversal. Test real SQLite/API pagination with ties, null dates and concurrent inserts.

## Validation scope

Use synthetic mail, temporary DBs and model/network guards. Gmail stays mocked. Refresh core validation and checker output semantics. Attachment binary extraction remains explicitly outside v1 scope; evaluate available PST/Postgres/performance evidence without representing unavailable infrastructure or fixtures as passing live certification.
