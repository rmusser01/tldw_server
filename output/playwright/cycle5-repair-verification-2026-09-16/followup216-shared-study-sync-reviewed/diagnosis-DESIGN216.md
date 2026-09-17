# UAT216 / TASK13260.155 — StudyPack shared sync schema

## Confirmed cause

The owned job 2 remains quarantined and unchanged. Read-only inspection through DatabaseBackendFactory verifies the profile boundary and transaction_read_only=on. Its input has one note source, no workspace_id, and deck_mode=new. No source text, title, credentials or session data was emitted. Retained schema/function metadata proves runtime sync_log has required entity_uuid and no entity_id, while each installed StudyPack, membership and citation trigger has three INSERT references to entity_id.

Official-fixture `test_study_pack_shared_sync_schema.py` reproduces the native CharactersRAGDBError at the actual worker → generation service → create_study_pack boundary: real Media-first PostgreSQL fails, while standalone ChaCha PostgreSQL and both SQLite controls pass (1 failed / 3 passed / zero skips). Existing owner/lifecycle tests mock the Media DB and therefore exercise the standalone entity_id shape.

The private official-PG causal contrast records the payload-free driver exception category UndefinedColumn for the unchanged template. Replacing only the column reference in three trigger definitions allows the real pack, membership and citation writes and all three sync rows (2 controls pass). SQL results/errors are never faked. Only disposable official-fixture trigger definitions are changed by this contrast.

## Approved minimal production change

Inside existing `_ensure_study_pack_schema_postgres`, introspect sync_log in the existing initialization transaction. Follow existing `_apply_schema_v4_postgres`, generic link logging and character-history column-capability patterns: choose from the closed set entity_id/entity_uuid. Apply that selected identifier to exactly nine INSERT column lists in the three proven trigger functions. Values, JSON payload fields, status/owner/version predicates, trigger timing and transactions remain unchanged. No global SQL translation, schema mutation, new migration/version, worker change or raw error disclosure.

This initializer runs on every current-head open and uses CREATE OR REPLACE FUNCTION, so existing runtime trigger bodies are repaired on normal reopen. Verify that against a disposable current-head fixture containing the old installed bodies. Source209 confirms no overlap in this method; shared full-file hashes can include its independently attributed Notes work, so retain method-only snapshots/patch.

## Validation stages

1. RED: actual worker/shared schema; retain safe native schema and original-source contrast. Complete.
2. Controls: actual output owner, pack/deck/cards/membership/citation sync payloads, rollback and reopen replacement; SQLite and both known sync schemas. In progress.
3. GREEN: exact bounded method change; official required PG focused and adjacent lifecycle/owner/storage tests, Ruff/Bandit/diff and hashed review packet. Pending.

No native job/card mutation or provider call is authorized here. Model output is mocked only at the existing remote generation seam. Independent review and parent-owned native replay remain required.

## Separate findings and limits

The required reverse initialization-order probe exposed a distinct earlier failure: ChaCha-first creates sync_log(entity_id), then Media bootstrap tries CREATE INDEX ... ON sync_log(entity_uuid) at media_db/schema/features/core_media.py:371 and aborts before a StudyPack call. The six-case receipt has two failures: this initializer defect and known216. It cannot be counted as a passing216 control or corrected by the three-trigger change. Parent has been notified for separate task attribution; no bypass, xfail or production expansion is applied.

Two neighboring suggestion trigger definitions in the same initializer also hardcode entity_id (suggestion_snapshots_sync_log_fn and suggestion_generation_links_sync_log_fn). They are not exercised by the proven StudyPack create chain. This is a source candidate for a separate association and actual regression, not an established native failure or authorization for a blanket rewrite. Other domain triggers are outside this audit.

The native job's exact driver exception is redacted by design; the causal connection is its matching runtime schema/installed functions and the reproduced exact service failure. The isolated diagnostic identifies UndefinedColumn without restoring raw driver cause/context/query parameters to production errors.
