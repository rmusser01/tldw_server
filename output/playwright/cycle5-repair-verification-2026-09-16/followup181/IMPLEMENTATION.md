# UAT181 / TASK13260.118 implementation

## Result and scope

Standalone Flashcards/Study and their actual Buddy read chain now finish transactions they start on the cached PostgreSQL connection. This releases read locks before a replacement CharactersRAGDB instance runs existing-column bootstrap DDL. Existing caller transactions remain owned by their caller.

Exactly39 `read_only=True` additions in four production files; no SQL, defaults, schema, write, or generic helper changes. `ast-equivalence.json` verifies the frozen source is otherwise AST-identical to baseline. `read-inventory.json` enumerates every callsite; `production-freeze.json` records exact pre-UAT182 hashes and `production.patch` isolates this unit. Baseline product bb5a0c01f1; docs checkpoint b253ca901c.

## Test-first evidence

| Gate | Result | Receipt |
| --- | --- | --- |
| Original read-owner RED | 37 failed /12 passed /0 skipped | uat181-red.redacted.log |
| Populated Buddy extension RED | 8 failed /3 passed /49 deselected /0 skipped | uat181-populated-buddy-red.redacted.log |
| Initial39-site GREEN | 60 passed /0 skipped | uat181-green.redacted.log |
| Final181+182 GREEN | 73 passed /0 skipped | uat181-182-final-green.redacted.log |

The original RED consists of35 standalone INTRANS assertions, one failed SELECT left INERROR, and one actual replacement constructor failing in Flashcards asset-schema initialization. The extension RED proves four actual populated Buddy attachment/activity→cards chains block DDL, plus conversation lookup/list and both workspace lookup branch transactions. Three first-read explicit/nested/backend ownership controls already pass before the repair.

The final tests additionally check the wholly pure inventory in each caller-write/locking/function/CTE control. Methods with pre-existing maintenance or mutation wrappers are excluded from those all-pure controls; only their newly flagged SELECTs are covered by standalone checks. Their existing write ownership is unchanged. Final expanded61-case lifecycle suite passes alongside the separate12-case UAT182 asset suite:73 passed /0 skipped in105.66s. The final test includes the populated binary result in standalone and caller ownership checks.

## Validation

- Ruff production baseline8/current8: identical diagnostic codes/messages, no additions. Existing findings are in conversation/persona stores. New test file passes scoped Ruff.
- Bandit production scope:0 findings,0 errors.
- `git diff --check`: passes.
- Source AST parse/equivalence: passes; exactly39 keyword additions.
- Adjacent required-PG/SQLite regression run:158 passed /4 failed /0 skipped. All four failures reproduce against pre181 source (4 failed /0 skipped); see adjacent-failures.md. They remain separately tracked findings, not skipped tests.

## Commands

All commands begin by activating the existing virtual environment. The private runner uses official function-scoped PostgreSQL fixtures, requiredPG/no skips and the owned55475 cluster; it never connects tests to the native content database.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-182-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py tldw_Server_API/tests/DB_Management/test_flashcard_asset_content_backends.py -q --tb=short
python -m ruff check tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/Buddy_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py
python -m bandit tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/Buddy_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py -f json -o .tmp/uat181-repair-20260916/bandit-current.json
```

## Native evidence and limits

Native failure metadata and diagnosis are retained separately in the parent's committed followup181-native-restart-failure packet. Seven old idle transactions held flashcards AccessShareLocks and blocked replacement bootstrap's AccessExclusiveLock. None held Notes locks. All seven disappeared after the parent stopped its already-draining oldAPI740; actual Notes Retry then succeeded. This is corroborating workaround evidence, not repaired native acceptance or direct host-socket mapping.

The repair intentionally preserves any already-owned transaction, including one opened by another unmodified domain. It does not guarantee the whole application has no idle transactions. No runtime/browser/provider/config/production DB changes or process cleanup were performed by this author.

The original asset-content lifecycle test uses a legitimate missing result only. Populated binary retrieval failed separately on PostgreSQL positional row access, tracked as UAT182/TASK13260.119. UAT181's frozen production still contains that access; its separate repair must not be attributed to these read flags. After the parent applied the separate182 one-line repair, the final61-case lifecycle test added populated bytes to the standalone/caller checks. The combined73-case run passes. This does not alter attribution: the original181 freeze still has positional access; current integrated ChaCha SHA17f1a2db3214b6488fd9cb1e6c137dcdfb08594faca869b71115b9afa799a84d includes both units.

Verification-manifest.json records integrated files for the final run; owned-manifest.json and review-snapshot retain pre182 production attribution plus the final181 test. Independent review/native acceptance remain separate gates.
