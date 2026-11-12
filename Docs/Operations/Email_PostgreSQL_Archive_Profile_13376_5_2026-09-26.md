# PostgreSQL archive persistence profile

Associated task: TASK-13376.5. This diagnostic supports the core email closeout plan; the parent sustained HTTP run supplies the release throughput certificate.

The initial authenticated three-archive diagnostic imported 300 synthetic messages at 47.2045 messages/sec. Each archive contained 100 messages. It preserved exact retry IDs and cross-user isolation but fell below the 50 messages/sec target. The diagnostic report is `/tmp/email_postgres_http_diagnostic_13376.json`.

Two guarded 100-child persistence profiles used the same generated diagnostic PostgreSQL database. The wrapper installed the archive probe's model, DNS, and non-loopback socket guards before importing the application, then persisted synthetic children through the normal archive persistence function with user scope 1. Neither run attempted model processing or outbound connections. Both used `cProfile` and method/statement timers, so their absolute timings are diagnostic measurements rather than sustained HTTP results.

| Measurement | Before | After |
| --- | ---: | ---: |
| Total elapsed seconds | 2.2262 | 1.7234 |
| Messages/sec | 44.9203 | 58.0248 |
| Media persistence seconds | 1.2072 | 1.0388 |
| Accepted-payload read seconds | 0.1230 | 0.1084 |
| Native graph persistence seconds | 0.5998 | 0.3310 |
| SELECT calls including bootstrap | 1,931 | 1,531 |
| DELETE calls | 300 | 0 |

PostgreSQL connection waits dominated the initial profile. Native persistence separately queried IDs after each source and participant upsert and message insert, then deleted three empty relation sets for every newly inserted message. The change uses PostgreSQL `RETURNING id` for source, message, participant, and label writes. SQLite retains its existing ID lookup strategy. A new message has no relations to replace, so only existing messages delete their old participant, label, and attachment relations.

The live PostgreSQL regression graph includes two participants, one label, one attachment, and a provider identity. Its old path issued 20 statements; the updated path remains within the tested 13-statement bound. Replacement retains source/message IDs, preserves participant display names when incoming names are absent, replaces recipients and labels, and removes old attachments. Identity lookup precedence, accepted-payload reads, transaction boundaries, and RLS scope handling remain intact. Native failure still rolls back the graph while retaining the separately committed legacy Media row.

Verification:

- Two expected red failures established the new graph round-trip and empty-relation-delete regressions.
- 18 focused SQLite/PostgreSQL graph and archive tests passed, including late Python and SQL native failures.
- Two strengthened final graph tests passed.
- 96 native identity and sensitive-logging regressions passed.
- Ruff passed. Bandit reported zero findings and errors on the changed production file; the fixed backend-selected SQL suffix is documented and message values remain bound parameters.

Credential-free evidence is retained in [the before profile](evidence/email_core_closeout_13376/postgres_archive_profile_before.json) and [the after profile](evidence/email_core_closeout_13376/postgres_archive_profile_after.json). The two profile-generated temporary roots were removed and recorded in the closeout cleanup receipt. The generated diagnostic database contains the original 300 HTTP messages plus 200 profile messages and must not be reused as an empty final certificate fixture.
