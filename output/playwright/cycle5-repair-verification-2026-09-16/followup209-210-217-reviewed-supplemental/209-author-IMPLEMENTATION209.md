# UAT209 / TASK13260.147 — selected-owner Notes persistence

## Current result and review boundary

Source and tests are frozen for independent review. The exact-byte repeat passed **202 tests, zero skips,2 warnings,284.51s** (`uat209-210-217-frozen-green`); all six manifest hashes remained unchanged before/after. The preceding202-case pass overlapped the separately owned UAT216 edit and is retained as an earlier qualified receipt. Native acceptance remains with the parent.

The repair adds selected PostgreSQL owner predicates at the existing Notes, KeywordStore and folder persistence boundary. Foreign private rows now appear absent and cannot be renamed, edited, deleted, restored, merged, attached or transferred through these methods. Both parents of linked resources are checked. Existing generic helper defaults remain unscoped unless a relevant caller explicitly supplies `owner_client_id`; SQLite remains per-file and retains historical device labels. There is no schema, cache, RLS, SQL-translator or operation-lifetime rewrite.

`changed-method-inventory.json` lists the exact 84 changed methods:23 ChaCha helpers/folder methods,26 NoteStore methods and35 KeywordStore methods. This includes all duplicated facade/store link paths and existing generic helpers; it is not84 new abstractions. `attribution-check.json` confirms module/class nonmethod AST content is unchanged. The larger declared review inventory was not treated as permission to change unrelated stores.

## Authority and compatibility

- The actual cold HTTP dependency and cache select canonical owners2 and3 on one shared PostgreSQL backend; SQLite uses separate files. The permanent test performs empty cold list, create, list/detail, own PATCH and reload, then foreign detail/PATCH rejection for both actors.
- Mutations include owner predicates in SQL and in status/version readbacks. Parent checks run inside the existing transaction. The stale-owner test changes ownership after version lookup and proves the guarded mutation rolls back instead of transferring the row.
- Message `client_id` can be a sync/device label. Its live conversation is the authoritative owner. The new parent helper follows that existing contract, with message→conversation lock order, and rejects a message whose conversation is foreign/deleted even if the message label matches the caller. It does not change MessageStore.
- Notes Sync projection requires the selected PostgreSQL canonical client ID; guarded conflict updates/tombstones cannot overwrite another owner's object UUID. SQLite's historical device-label projection behavior is preserved. Existing Sync envelope, projection versions and materializer authority are unchanged.
- Raw `BEGIN`, supplied connection and nested transaction controls show the caller retains rollback decisions. Owned lifecycle, same-name organization reuse, restore, and malformed cross-owner relationship controls remain positive.
- Existing shared-workspace authorization/owner-loader behavior is unchanged; private Notes are not made shareable by passing a workspace ID. Existing Studio, task, attachment and modern graph stores were inspected as scoped predecessors and were not rewritten.
- PostgreSQL Notes/keywords/folders already have RLS. The fixture's real NOSUPERUSER/NOBYPASSRLS role passes actual owned read/write and foreign exclusion after ordinary fixture bootstrap. This is **not** a cold restricted-login bootstrap claim or isolation promise for arbitrary SQL executed by a bypass role. Application predicates protect the accepted privileged runtime path that reproduced the native leak.

## Separately attributed work in shared source

- UAT210/TASK13260.148: only `get_note_tag_edges`' mapped keyword table variable and JOIN replacement. The original five graph methods were already committed47e23bd5f3. Residual compatibility patch/test/report are under `.tmp/uat210-repair-20260917/residual-tag-edges/`.
- UAT217/TASK13260.151: only the PostgreSQL literal keyword search operand changes `ILIKE ?` to `ILIKE (?)`, preserving original escaping. Separate patch/tests and failed-candidate history are under `.tmp/uat217-repair-20260917/`.
- UAT216 is authored by Retry031: `_ensure_study_pack_schema_postgres` only. The six-path review snapshot contains that integrated method. `external216-method.patch` is attribution only; `owned209.patch` excludes it. Retry's frozen method SHA is `ed682318fa4cf77681446231e3fecdd470c108ab1f46a8fd6e7bc301d2173e81`.

`owned209.patch` starts from the baseline with the separate210/217 patches already applied and contains this unit's three production files plus its new156-case test. Do not apply it as a whole-file replacement over another author's shared-file changes. All snapshots/hashes are in `owned-manifest.json`.

## Causal evidence and corrected harness stages

Official DB_Management fixtures provision all temporary databases through the required-PG runner. No native data or model was used. Labels below resolve to `.tmp/fresh-uat-recovery-20260916/<label>.redacted.log` and the matching command JSON.

| Label | Result | Attribution |
| --- | --- | --- |
| `uat209-owned-reads-red` |6 PG failures /6 SQLite pass |Original six native-relevant private read boundaries |
| `uat209-owner-edges-corrected-red` |11 PG failures /11 SQLite pass |10 owner failures plus separately tracked graph row-shape failure |
| `uat209-organization-role-red` |6 PG failures /7 pass |Organization owner failure; restricted-role baseline works |
| `uat209-permanent-initial-red` |22 PG failures /23 pass |Initial permanent regression |
| `uat209-sync-lifetime-red` |2 PG failures /5 pass |Foreign Sync object writes; preservation controls |
| `uat209-expanded-organization-red` |25 fail /25 pass |24 PG owner failures plus one incorrect SQLite parent-create expectation, corrected to preserve existing SQLite behavior |
| `uat209-linked-parent-red` |13 PG failures /22 pass |Both-parent relationships, conversations, folders |
| `uat209-cold-http-baseline-red` |1 PG failure /1 SQLite pass |Private nonmutating replay of original NoteStore against real cold dependency |
| `uat209-graph-sync-read-red` |2 PG failures /2 pass |Separate210 tag JOIN compatibility and209 Sync merge-snapshot owner guard |
| `uat209-message-parent-candidate-red` |2 PG failures /4 pass |Rejected candidate used message device label as owner; corrected before freeze |
| `uat209-deleted-message-parent-red` |1 PG failure /1 SQLite pass |Live message under deleted ancestor must be rejected |
| `uat209-message-parent-green` |6 pass /0 skip |Owned device label, foreign ancestor and deleted ancestor controls |
| `uat209-restricted-role-persistence` |1 pass /0 skip |Actual restricted-role service create/read with existing RLS |

Original private integer-ID/FK harness mistakes, the raw-BEGIN wrapper context-manager mistake and the failed UAT217 escape candidate are retained. They are not counted as product evidence. The folder source-key control preserves global source IDs: a colliding foreign key is rejected rather than reassigned; a distinct owned source works. No uniqueness migration was added.

## Validation

- Final frozen combined suite: **202 passed,0 skipped,2 warnings,284.51s** (`uat209-210-217-frozen-green`). All six snapshot/current hashes match.

- First combined owner/graph/literal suite: **202 passed,0 skipped,2 warnings,291.60s** (`uat209-210-217-final-green`), qualified source overlap described above.
- Adjacent Notes/stores/folders/graph/restore and171/181 read lifetimes: **176 passed,0 skipped,5 warnings,167.34s** (`uat209-adjacent-notes-green`).
- Adjacent Notes organization Sync API: **69 passed,0 skipped,4 warnings,61.16s** (`uat209-adjacent-sync-green`).
- Adjacent runs preceded the final message-parent helper correction; its actual PostgreSQL/device/deleted controls passed separately and are included in the frozen combined repeat.
- Ruff:0 baseline/0 current using baseline bytes with their real logical paths. Six-path current lint0; three test files format clean; compile and diff checks clean.
- Bandit:production baseline0/current0 findings and0 errors; tests0 findings/0 errors with only assertion ruleB101 excluded. Existing closed-table/validated-alias SQL fragments retain parameter-bound values. New expression-local B608 annotations document those fixed fragments, not arbitrary query interpolation. The transient three findings from216's method were handled by its author and are separately attributed.

### Independent command

From repository root:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat209-210-217-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_note_shared_owner_contract.py tldw_Server_API/tests/DB_Management/test_note_graph_query_backends.py tldw_Server_API/tests/DB_Management/test_keyword_literal_search_backends.py -q --tb=short
```

This enforces PostgreSQL-required fixtures and includes SQLite controls:156 owner +24 graph +22 literal cases. Network escalation is needed for the already owned local cluster. No final native acceptance, whole-database security claim, runtime change or commit is made here.
