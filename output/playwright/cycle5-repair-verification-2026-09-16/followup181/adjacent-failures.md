# Adjacent regressions retained for separate tracking

Current run:158 passed,4 failed,0 skipped. Exact command: uat181-adjacent-command.json. None of these tests were skipped, weakened, or edited.

Baseline replay restores the exact36 changed method bodies (covering39 SELECT callsites) from retained pre181 files into the disposable pytest process. Generic helpers/constants are AST-identical outside those methods; the separate182 getter change is also removed by restoring its full baseline method. Source files and application runtime are untouched. baseline_replay.py records this mechanism. Replay:4 failed,0 skipped,4.71s, same four causes. Exact command and receipt: uat181-adjacent-baseline-replay-command.json and .redacted.log.

| Test | Observed failure / attribution |
| --- | --- |
| tldw_Server_API/tests/Persona/test_independent_buddies.py::test_schema_head_v66_has_one_sqlite_and_postgres_step line120 | Hard-coded head66 vs actual67. Existing65-to66 migration step assertions occur after the obsolete head assertion. |
| tldw_Server_API/tests/Persona/test_independent_buddies.py::test_v65_database_upgrade_preserves_conversation_and_adds_independent_storage line85 | SQLite upgrade reaches67, but test asserts66 before checking preserved content. |
| tldw_Server_API/tests/Persona/test_independent_buddies.py::test_postgres_v65_upgrade_installs_buddy_storage_and_forced_tenant_policies line191 | Official realPG upgrade reaches67, but test asserts66 before later policy checks. |
| tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py::test_migration_v39_to_latest_creates_persona_buddies_table line47 | Fixture creates current schema67, then changes only version to39 and drops persona_buddies. It leaves newer note_attachments. At v58-to59 the intentional collision guard refuses an already-existing registry (ChaChaNotes_DB.py11991). This fixture is not a genuine historical schema39 database. |

Recommended separate bounded work: distinguish exact65-to66 step coverage from current-head upgrade coverage in the three assertions; construct an authentic v39 fixture or invoke the specific persona migration boundary according to the actual test contract. Preserve the v59 registry collision guard. This evidence alone does not prove a real historical39 upgrade is broken or establish that it passes. Root owns associated Backlog tasks before edits.
