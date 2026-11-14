# C09 explicit PostgreSQL database recovery

Task: TASK-13260.278.18.83.45. Frozen523ef3 plus unpublished WIP; no Git/publication. ADR required: yes, proposed ADR-079; ADR-020 retains backend/path ownership and ADR-059 task editing.

## Approved scope and exclusions

The human selected the whole PostgreSQL database, including all users and Jobs, and excluded unconfigured Litestream from this deployment. C09 still requires live separate-target restore and maintenance recovery; no test or source review accepts those cells.

## Smallest complete interface

- Add an explicit `postgresql` bundle dataset. It is global, selected explicitly, exclusive of the legacy dataset list, and forbids user_id. Legacy omitted-dataset defaults remain unchanged.
- Resolve actual AuthNZ/content/Jobs configurations through their current owners. Require all three to be PostgreSQL with the same explicit host/port/database. Compare effective libpq identities, including configured connection-string precedence. Refuse split/ambiguous configurations before artifacts, pruning, staging or safety snapshots. No database creation or SQL outside DB abstractions.
- Reuse existing snapshot/restore helpers through the internal data-ops service. Do not advertise this new dataset in the legacy individual-backup or schedule UI; those public allowlists remain narrow. Bundle create/download/metadata/delete and import are platform-admin scoped; non-platform listings omit global database archives before pagination.
- Export one PGDMP artifact. Version2 manifest declares `recovery_scope=postgresql_database`, sole dataset, no user target, size/hash and explicit coverage/exclusions. Legacy SQLite/dedicated-auth bundles retain version1 and existing semantics. Old importers refuse version2.
- Dry run validates manifest, actual target backend/scope, complete dump header and hash before reporting compatible. HTTP wet import of PostgreSQL scope refuses with offline-handoff guidance; ordinary bundles retain their existing interface.
- UI names all users/content/Jobs and the current target server, forbids mixed/owner selection, and directs PostgreSQL recovery through the explicit offline helper handoff. Snapshot, file and authority remain captured. Exact terminal datasets are required; no automatic retries or false success on partial/unknown completion.
- Reuse Helper_Scripts/pg_backup_restore.py for offline bundle restore. Require the exact target database name and explicit overwrite/all-clients-stopped acknowledgement. A short-lived DB_Management connection counts idle and active target sessions, excluding only exact CLI-owned backend PIDs; query failure or another session refuses. This is an observational check: the operator must keep API writers and all Jobs/Scheduler consumers stopped and prevent reconnection throughout the operation. Do not create an incidental application pool.
- Take a target safety snapshot, then use the existing single-transaction clean/if-exists restore under the restricted target application owner with no source ownership/ACL replay. Reject --no-drop for global bundles. One bounded operation deadline covers tool and inspection work. Unknown completion keeps the target offline; no automatic retry, restart or worker resume. Read restored data, review queues/leases and explicitly decide resume/cancellation before cold restart and reauthentication. UAT host must enforce its own owned process handoff, not claim a bundle lock blocks ordinary clients.
- Optional TLDW_POSTGRES_BACKUP_DSN names an already-authorized backup principal in the same effective host/port/database. It is used only for backup, never runtime/restore. Without it, the ordinary role may legitimately refuse FORCE-RLS complete-row dumps. Official fixture admin credentials can supply the pre-provisioned test backup principal; application NOSUPERUSER/NOBYPASSRLS roles and policies remain unchanged. Real restore ordering under FORCE RLS still requires proof.
- Separate retention by recovery_scope as well as user_id. Global PostgreSQL archives use a distinct unique filename namespace; legacy global operations cannot prune or overwrite them. No worker retry-budget reset, new installer or model work.

## Coverage and operational limits

A dump includes all database SQL objects/rows, including users and Jobs, that the configured PostgreSQL database actually stores. It does not claim local SQLite families, uploaded binaries, model assets, vector services or cluster-global roles. Keep pg_dump's default complete-row requirement: insufficient privileges fail instead of exporting a filtered tenant view. Target objects absent from the archive are not a claimed deletion operation. Fresh target validation uses official isolated restricted fixtures and normal principals.

References: [PostgreSQL18 pg_dump](https://www.postgresql.org/docs/18/app-pgdump.html) and [pg_restore](https://www.postgresql.org/docs/18/app-pgrestore.html). Their complete-row and single-transaction behavior informs these boundaries; actual installed tool/runtime results are still required.

## Validation sequence

1. Causal configuration/scope tests: explicit co-location succeeds; missing/split/unknown identities, user targets and mixed scopes fail before artifacts or mutation. Retain current wrong-store and narrow AuthNZ refusals.
2. Bundle/API regressions: versioned scope/coverage, one artifact, explicit confirmation, platform admission/list/download boundaries and malicious/missing scope tags. Preserve original SQLite/legacy manifest assertions.
3. Real source-to-distinct-target PG fixture: two ordinary users, representative owned content and Jobs retain identities, data and legitimate login; source remains unchanged. Failed restore is atomic; no partial-row or permission workaround.
4. Both configured clients, captured authority/one pending mutation, scoped compile/lint/Bandit, canonical API schema/types and independent actual-source review. Build only the reviewed current sources.
5. Prepare/review preserving one-tab native operators, then actual separate-target SQLite/PostgreSQL workflows under existing600s host/build,120s startup,15s request/shutdown caps. Screenshot UX, exact original-session handback and prompt removal of completed bulk outputs remain required. C10 stays unopened.
