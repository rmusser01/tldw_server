# Independent UAT218 review — shared Media/ChaCha sync-log contract

## Verdict

**CLEAR after the bounded review correction. No remaining actionable defect found in the approved scope.** Ready for parent integration and separately controlled native acceptance.

Final independent mandatory PostgreSQL/SQLite run: **176 passed, 0 skipped, 4 warnings, 43.20s**, exit0. The two independent custom-literal counterexamples additionally pass: **2 passed, 0 skipped, 1 warning, 1.53s**. All eight corrected source/test hashes and author snapshots matched before and after. Ruff0, production/test Bandit0 findings and0 parse errors, and scoped compilation pass. Test Bandit excludes only assertion ruleB101; existing scanner stderr warnings are not represented as findings or declared absent.

Final author manifest: `424ea3879fea2ad3bdcd01ed266b3439e20c5058b87b460707b5b68f5e10f029`. The initial reviewed manifest was `0503e25583006e78db2ef199129b31754c7cc12e54c567f611c9f2d38e30f43b`. Only the schema helper's exact-comparison correction and its two permanent test cases changed between these releases; the other six paths stayed unchanged.

## Resolved finding

The first release stripped all whitespace from PostgreSQL's CHECK definition before comparing it with the two known application definitions. That also erased whitespace **inside SQL string literals**. A custom check allowing `cre ate` or `create ` was mistaken for the old `create` check and silently replaced with the standard five-operation check during a normal Media reopen.

This was a P2 violation of the approved fail-closed schema guard, not a hypothetical formatting concern. The independent actual-PG probe produced2 expected assertion failures with0 skips in0.96s on eight unchanged initial hashes. The metadata receipts record the original custom definitions and the rewritten definitions. Initial174 regression tests had passed; the additional counterexamples exposed the gap.

The approved correction compares the full canonical `pg_get_constraintdef` output directly with the two known definitions. It does not add a parser or broaden schema acceptance. Equivalent permanent tests first failed2/2; the final176 suite and original private two-case probe now pass. Both custom constraints remain exact after rejection. Original RED metadata copied into the author's packet before the GREEN run is preserved here with explicit `-RED.json` names; corrected metadata uses `-GREEN.json`. Initial source snapshots and RED logs remain available in this packet.

## Production review

Five production paths were reviewed against the approved design and isolated patch:

1. **Schema boundary:** the helper accepts exactly one of the existing `entity_id`/`entity_uuid` columns, verifies known text/nullability/scope shapes, and checks that precisely one CHECK depends only on `operation`. Exact old/current definitions are the only accepted operation constraints. Unknown or ambiguous shapes fail before compatibility DDL. Catalog constraint names use backend identifier quoting; scope names are fixed constants.
2. **Minimal DDL:** only missing nullable `org_id`/`team_id` columns, the known three→five operation CHECK replacement, and payload nullability relaxation are performed. It does not rename/copy/backfill IDs, replace the relation, disable RLS, change schema heads, add dual writes, or weaken required identity/operation nullability. The index substitution targets the existing exact identifier-index statement.
3. **Transaction ownership:** the helper uses only the supplied schema connection and issues no commit, rollback or checkout. It is called before base indexes and by the existing post-core/current-head path. Fresh initialization can validate twice; compatible subsequent calls perform no ALTER. Constructor failure rolls back the existing bootstrap transaction. Direct helper tests preserve caller-pending work and its rollback decision. This review does not assert a wider Media transaction-lifetime redesign.
4. **Runtime contract:** the Media writer selects between two fixed physical column names while retaining bound values. The reader exposes the same public `entity_uuid` key by aliasing legacy storage. Runtime calls do not introspect. Normal constructors always establish the capability; the SQLite path remains unchanged.
5. **ChaCha:** its generic sync reader now preserves SQL NULL as None; non-null JSON and malformed-JSON behavior remain unchanged. Independent AST comparison confirms every other ChaCha method/class/module element is identical to the pre218 baseline. The separate216 trigger and209 owner repairs are not modified.

## Regression quality and limits

The26 new permanent cases use ordinary Media/ChaCha constructors on official disposable PG fixtures and normal separate SQLite files. They cover both initializer orders; real create-with-keyword/cascading delete and link/unlink events; pagination; old note/pack/membership/citation sync fields; subsequent trigger writes; optional and malformed payloads; exact relation/row/constraint/scope/policy preservation on reopen; constructor rollback after a controlled late failure; and supported versus rejected schema shapes.

The actual StudyPack worker/service/source path is exercised with controlled model output and supplied real fixture databases. That is storage/worker integration evidence, not a real-model or native job retry claim. Restricted NOSUPERUSER/NOBYPASSRLS tests verify personal, foreign, organization and team sync visibility after normal fixture bootstrap/reopen. They do not establish arbitrary bypass-role SQL isolation, cold restricted-login provisioning, or a whole-application security guarantee.

The two existing ordering tests retain their assertions and add only a mocked schema collaborator at the actual required position. They do not replace the real constructor controls. The150 adjacent tests cover Media schema/bootstrap, sync read/write/cleanup, runtime PostgreSQL validation,216's shared sync graph and character sync identity behavior.

## Evidence

- `corrected-required-pg176.redacted.log` and `uat218-independent-corrected-command.json`: final independent full command/result.
- `literal-guard-red.redacted.log`, `literal-guard-green.redacted.log`, original probe, and explicit RED/GREEN metadata: resolved finding.
- `corrected-before-source-hashes.json`, `corrected-after-source-hashes.json`, `source-attribution.json`: stable frozen bytes and ChaCha scope proof.
- `corrected-ruff-current.json`, `corrected-bandit-production.json`, `corrected-bandit-tests.json`, `corrected-compile.txt`: fresh scoped static checks.
- `initial-required-pg174.redacted.log`, `before-source-hashes.json`, `after-initial-review-hashes.json`, `initial-reviewed-source/`: initial reviewed release, retained separately.
- The author report's earlier RED/harness corrections were inspected as author evidence, not relabelled as independent reruns. Its historical baseline helper exclusion is explicitly documented because that helper does not exist in the original source.

Only private review artifacts and isolated official fixture state were written. No production/test/task/tracker/git/browser/native runtime edits were made. Native reverse-order startup, actual queued job execution and final UAT acceptance remain parent-owned.
