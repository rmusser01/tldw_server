# UAT239 / UAT255 bounded native acceptance — CLEAR

Tasks TASK13260.181 and TASK13260.197. **21 checks passed; 46 inputs hashed.**

## Accepted evidence

- Earlier empty Character-editor catalogs returned HTTP 200 with zero books in PG-single at `20:09:38.115Z` and PG-multi at `20:17:24.376Z`, under the prior runtime.
- On reviewed revision `2787043410fc918b2c280d90f753d8fd02b6b35b`, Alice's normal UI create returned HTTP 201 at `23:51:02.399Z`, creating WorldBook 1. The immediate list returned HTTP 200 and the exact same record.
- A normal page reload returned the same record at `23:54:20.086Z`. Name, description, enabled state, scan depth 3, token budget 500, recursive scanning false, version 1, and zero entries persisted. Edit UI displayed the name, description, and checked enabled control.
- Character 4's editor requested `world-books?include_disabled=true` and received the populated catalog at `23:55:56.631Z`. Its association request returned HTTP 200 with an empty list. The editor was closed without saving changes.
- Authenticated identity observer events at `23:47:43.435Z`, `23:54:19.936Z`, and `23:54:55.979Z` identify user 2 and match Alice's username in the preserved private fixture. These are identity observations, not deductions from the fictional book name.

Both upgrade copies' source manifests, actual process bindings, original profile/initialization/holder hashes, and API/frontend receipt lifetimes were verified. `world_book_manager.py` matches committed revisions `15c1bd5134fa07b83ba0c4e66f446330ea0b824a` and `2787043410fc918b2c280d90f753d8fd02b6b35b`, the two immutable runtime copies, current source, and the approved implementation review: SHA-256 `fbefcf2e2283b8a8ed26de6c0829e7104c53d187560713b0e833ac5e0bf90d5f`.

## Preserved failures and limits

The earlier create HTTP 500 at `22:37:24.025Z` is retained as a failed attempt. The startup receipt also discloses failed initial launch attempts, an aborted navigation, and a transient HTTP 500 before the successful runtime became ready.

**UAT260 remains open:** the WorldBook reload visibly displays “in 7 hours.” This review accepts catalog/create/readback behavior only and does not accept timestamp chronology.

The populated create/reload/editor sequence covers PG-multi Alice. No entries or associations were added; this is not a broader WorldBook CRUD, duplicate/conflict, owner-isolation, SQLite-native, or full-matrix certification. No browser, runtime, database, model, product, test, Backlog, or Git state was changed. Read-only Git object lookups verified source bytes. Private records stayed in memory and are represented only by hashes and safe comparisons. Unrelated native events, UI, and provider reasoning are not serialized.

Audit: `audit.json`, SHA-256 `3dd602b639d16b2403bab3eacdfeae271ba7a85d729bcf3457e35382463f6949`.
