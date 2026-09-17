# UAT222 causal ownership investigation

No production file has changed. `causal-method-inventory.json` verifies unchanged ChaCha SHA `0bbf4442fa4f41d53ff61c3261b480387bcbdc866e1f15785244fbdb6065513c` and records exact method line ranges/hashes. Root owns native data, source integration and subsequent same-pack acceptance.

## Native observation

The supplied `study-pack-job5-complete-readback-result.json` verifies authenticated Alice2 reads original job5/pack2 successfully; Bob3 receives404 on job5 but200 on pack2 with owner2 metadata. The exact captured destination is deck10. No new native request or original-data mutation was performed during this diagnosis. Root separately retained the19-request continuation proving other deck/card/assistant negatives and original source/card versions; the isolated tests below do not replace those native receipts.

## Clean causal runs

Both use normal repository pytest configuration, explicit-Jobs official required PostgreSQL fixture runner and a uniquely labelled log. The API tests use established selected actor/database overrides, real PostgreSQL or SQLite content, and real disposable SQLite JobManager. Regenerate only creates a disposable queued record; no worker or model executes.

1. `uat222-corrected-owner-red`:8 PostgreSQL failures,9 passed controls,0 skips,4 warnings,22.86s. Actual HTTP detail leaks200, regenerate accepts202; each of four raw selected store reads returns foreign data; actual assistant for an owned card returns foreign citation or pack metadata. SQLite different-file boundaries and same-file device labels pass.
2. `uat222-corrected-write-red`:10 PostgreSQL failures,10 SQLite controls passed,0 skips,17 prior cases deselected,4 warnings,28.93s. Direct foreign delete, supersede, membership append, citation append/replace, atomic citation+summary replacement and summary update all alter the owner's data. Creating a pack with a foreign destination deck, adding a foreign card to an owned pack and superseding an owned pack with a foreign replacement are wrongly accepted. These are persistence-boundary findings, not claims that an unauthenticated/native HTTP caller exercised every method.

Exact commands:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat222-corrected-owner-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_study_pack_owner_contract.py -q --tb=short
TLDW_UAT_EVIDENCE_LABEL=uat222-corrected-write-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_study_pack_owner_contract.py -k 'mutation or reference_foreign_parent' -q --tb=short
```

The first run occurred before the additional20 mutation/parent cases were added. The current37-case file retains those17 assertions plus the20 new cases; `causal-owner-test.py` freezes the combined current file. Source production hashes stayed constant throughout. Redacted logs and runner command JSON are copied here; raw private logs are not copied.

## Preserved harness corrections

- Initial17-case run:11 failures/6 passed. Eight were the genuine PostgreSQL failures above. Two same-file SQLite controls reused a deck name and hit its existing unique constraint; they now use actor/device-specific deck names. The standalone SQLite device-control seeded a Note without that fixture initializing the optional Notes schema; its source identifier is now a synthetic citation reference, matching existing storage tests. All actual assistant-route fixtures still use real persisted Notes. No production/schema bypass or test skip was added.
- Initial mutation run:17 failures/3 passed/17 deselected. Seven SQLite exception paths exposed a missing exception-class import in the new test. The initial source/log are retained; the import is fixed. Corrected PG exception checks require expected ownership/input exception classes rather than accepting an unrelated backend error. Ten genuine PG mutation/reference failures remain.
- Ruff caught and corrected the test-only import formatting/undefined-local issue before the corrected write run. No product diagnosis rests on either harness error.

## Proposed boundary and qualifications

See DESIGN222.md for the12-method family scope and small existing-helper extension. Do not apply a getter-only repair and claim the assistant/mutation cases fixed. The source's job-status actor authorization, authorized administrator owner-database resolution, normal owned flashcard HTTP gate, SQLite device semantics, row versions, deleted filters, ordering and transaction ownership must remain.

The service's privileged/BYPASSRLS native role is explicit. These official fixture failures prove absent application predicates; this packet does not claim a cold restricted-role acceptance or general RLS audit. No endpoint, cache, source resolver, schema, migration, global SQL translator or UAT223 route edit is proposed.
