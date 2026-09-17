# UAT183 / TASK13260.120 independent review

## Verdict

Clear. The four stale migration tests now express their original migration contracts without assuming schema66 is the permanent repository head. No production change or migration guard relaxation is required or included.

Fresh independent run through the official required-PostgreSQL fixture runner: **4 passed, 0 skipped, 4 existing warnings, 4.07 seconds**. The same two frozen files were verified again after the run.

## Scope and exact hashes

| Reviewed file | SHA256 |
| --- | --- |
| tldw_Server_API/tests/Persona/test_independent_buddies.py | `83cf43648014105ecd92652d9a730437f21eafcb920fe8739e03b444ac1f687a` |
| tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py | `db045fb63c887f20cf05cc6eb778137588bafc0642ecfef2bd313a391abce32e` |
| Author owned-manifest.json | `faba454cfa44743905d8988c1e62a7e4486c1baaa0b67dce8486e276480d59ea` |

The tested production ChaChaNotes_DB.py remains `17f1a2db3214b6488fd9cb1e6c137dcdfb08594faca869b71115b9afa799a84d`, the already reviewed integrated181+182 source. The reviewer made no production/test changes.

## Contract review

- **Historical39:** the fixture invokes the real V4 base and registered migration methods4 through38. Every step must advance the recorded version by exactly one. Before seeding, it asserts version39 and absence of both persona_buddies and the later note_attachments registry. It inserts a conversation using historical columns. The temporary seed-only initializer and version cap are restored before normal CharactersRAGDB construction performs the upgrade. All prior buddy columns, index and foreign-key assertions remain; new checks verify the conversation title and owner survive.
- **Fixture honesty:** merely capping the current initializer at39 failed because recent Persona repair still ran. The retained failed attempt explains the seed-only initializer. No historical version is assigned retroactively, no newer table is dropped to disguise the fixture, and no migration method or collision guard is mocked during the upgrade under test.
- **SQLite65→66:** the test still requires exact version66 and all six Buddy tables, with preserved conversation content. It then reopens normally, requires the current head, and checks content preservation again.
- **PostgreSQL65→66:** the test still requires exact version66, boolean deleted-column type, all six relations with enabled and forced row security, and the expected tenant policy expressions. It then reopens at the current PostgreSQL head, checks preserved conversation content, and exercises the existing actual Buddy CRUD, image bytes, attachment, result, acknowledgement, conflict, and delete controls unchanged.
- **Registry contract:** the renamed test retains the exact SQLite65→66 method registration and PostgreSQL method existence checks. Minimum supported version66 replaces only the obsolete assertion that the latest repository version must equal66.

An independent AST comparison found changes in exactly these four test functions. Every original assertion remains except the two exact-head66 comparisons, which become minimum-version comparisons. The only unrelated text removal is the now-unused sqlite3 import in the historical fixture file. The patch, source hashes, and assertion comparison are retained alongside this review.

## Independent verification

`independent-four-green-command.json` contains the full four-node pytest command. The run was launched after activating the project virtual environment, using the existing run-pg-tests.mjs wrapper, official isolated PostgreSQL fixture, requiredPG/no skips, and temporary SQLite files. PostgreSQL was not manually provisioned and the native application database was not used.

`independent-four-green.redacted.log` records exit0 and four passes. `evidence-manifest.json` hashes the command, receipt, reviewed patch, source/assertion comparison, and author manifest. Original four RED failures are retained in the author packet and the prior181 adjacent baseline replay. The reviewer also read the author's final `uat183-adjacent-green.redacted.log`: the same adjacent suite now reports **162 passed, 0 skipped, 139.59 seconds**. This broader run and static checks remain author verification, not independently rerun here.

## Limits

This is a focused migration-test repair review. It does not certify every historical migration, live deployment, or native browser flow. No browser, application runtime, production database, tracker/task, or git mutation was performed. The synthetic historical39 fixture uses the repository's real historical migration methods; it is not a recovered user database.
