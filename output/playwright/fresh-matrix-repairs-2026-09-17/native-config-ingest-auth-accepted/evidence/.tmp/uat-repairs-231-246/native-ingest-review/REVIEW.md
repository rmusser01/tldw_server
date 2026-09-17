# Independent native ingest acceptance — UAT233 / 238 / 244 / 245 / 247

**The submitted evidence supports the repaired terminal-warning, PostgreSQL persistence/admission, automatic catalogue update, and reciprocal owner-isolation boundaries. No new product defect was found. Some task acceptance criteria remain narrower than a full native pass; see the explicit limits below.**

Read-only retained-evidence audit: **34/34 checks passed**, **37 input files hashed** in `audit.json`. No browser/API/process/database actions, product tests, Git operations, or source/task/tracker changes were performed. Only this review packet was written.

## Scope and provenance

Original targeted run: `repairs231-250-targeted-20260917`, PG single API/WebUI `18702/18782`, PG multi `18703/18783`. Single ingestion happened on the original targeted source; its same saved item was read successfully after upgrade. The later multi uploads happened after upgrade to `a7d3155a567afb25982eb360ea24b973cc3249c9` via `repairs251-254-upgrade2-20260917`.

Source/profile preservation is reused from `.tmp/uat-repairs-231-246/upgrade254-native-review/REVIEW.md` and its audit: original initialization/holder files and identities were preserved, with real startup/source binding independently checked there. This reviewer did not re-inspect live runtimes. These are targeted PostgreSQL cases, not a fresh full-matrix rerun or a new clean-dependency installation.

## Actual queued jobs and saved content

All timestamps below are 2026-09-17 UTC. Inputs are under `.tmp/uat-repairs-231-246/native-targeted/`.

| Case | Native enqueue → completion | Owner / saved identity | Canonical source |
| --- | --- | --- | --- |
| PG single | HTTP200 enqueue 20:31:17.129; job1 completed 20:32:22; two terminal200 readbacks | owner1, Media1, UUID `d4a0d5e2-6e9a-4d7a-bc9e-f011af4149d0` | Later original-item GET200, 1,914 chars; original version 20:32:22.772 |
| Alice | HTTP200 enqueue 22:08:50.108; job4 completed 22:09:44 | owner2, Media1, UUID `be7b0cbc-a4db-4036-8d53-e1c6f481f79d` | GET200 22:11:24.693; 1,914 chars; version 22:09:44.323 |
| Bob | HTTP200 enqueue 22:18:21.859; job5 completed 22:19:12 | owner3, Media2, UUID `bfbd4ff9-2f2e-4e6d-bc7a-a9e0886ae6d4` | GET200 22:25:34.837; 1,906 chars; version 22:19:11.977 |

Every job has `status: completed`, progress100, nested result `Warning`, a saved media ID/UUID, null error, and **exactly one** warning: provider analysis was truncated before completion. Every own detail response has **null analysis** and completed chunking. This is successful source persistence with unsuccessful analysis, not clean analysis success. Each returned source equals its public fixture exactly after removing the fixture's single final newline; no other character differs.

Exact job proofs: `pg-single/ingest-observed.txt`, `pg-multi/alice-ingest-observed.txt`, `pg-multi/bob-ingest-observed.txt`. Exact content proofs: `pg-single/media-upgraded-read-v2.txt`, `pg-multi/alice-media-selected.txt`, `pg-multi/bob-own-media-selected.txt`. Fixtures are `.tmp/uat-next-matrix-20260916/fixtures/{rowan-observatory-public-20260917,birch-workshop-bob-public-20260917}.txt`.

## Automatic catalogue update and ownership

The actual scripts click **Minimize to Background**, then **Media**, before completion: single 20:32:02.100, Alice 22:09:16.657, Bob 22:18:46.914. The subsequent observation commands only sample the page and accumulated network events; they contain no reload, navigation, search, fill or click. Their settled pages show **Results 1 / 1** and the appropriate owned source. Bob supplies the strongest before/after control: `bob-ingest-minimized-media.txt` actually shows **Results 0 / 0 / Get started**, while `bob-ingest-observed.txt` shows only Birch. Single/Alice immediate body snapshots still show the previous Chat during navigation; their action receipts and resulting Page URLs prove the earlier Media navigation, but are not exact empty-catalogue snapshots.

The observer's URL filter omits bare `/media` list URLs. Thus this packet proves the UI transition through the retained action/observation sequence; it does **not** timestamp the precise list request or establish its latency from job completion.

Reciprocal boundaries are corroborated by retained identity metadata:

- Bob user3: GET Alice `/media/1` **404**; catalogue empty before his upload. `pg-multi/bob-foreign-alice-media-captured.txt`, at22:15:40.948, retains the 22:14:51.372 denial.
- Bob's own `/media/2` **200**, full1,906chars, Results1/Birch only. `pg-multi/bob-own-media-selected.txt`; identity user3 is corroborated at22:25:34.857 in the later accumulated identity receipt.
- Actual Bob logout / Alice login are retained in `bob-ownership-logout.txt` and `alice-ownership-login.txt`. The latter records authenticated **Alice user2** at22:29:31.112; it also retains Bob's earlier identity observations.
- Alice `/media/2` **404**, while Results1 shows only Rowan at22:31:07.974: `pg-multi/alice-foreign-bob-media-v2.txt`.

Distinct IDs and UUIDs on the same multi-user database, followed by correct own reads and reciprocal denials, support the actual sequential two-owner UAT247 acceptance. Native concurrency, high-water and rollback cases remain controlled-test evidence, not claims about this sequential browser run.

## Issue / task acceptance assessment

| Issue / task | Supported acceptance | Limit to retain |
| --- | --- | --- |
| UAT233 / 13260.175 | Original duplicate terminal-warning defect fixed; repeated single readback and both multi owners contain one warning, status and media identity retained. Source access succeeds. | AC3's **post-repair Results-panel saved-with-warning label** is not present in these targeted captures. Do not substitute the API `Warning` for a captured UI label. The original UI already deduplicated correctly, but that is prior evidence. |
| UAT238 / 13260.180 | Native original public-source persistence and canonical content succeed under the preserved restricted-role PG profile; AC1 supported. Reviewed official PG/SQLite regressions support AC2. | AC3 is broader: the separate native-media review proves full-source Chat handoff/answer; source QA, reanalysis and sole-item Trash are not certified by this packet. |
| UAT244 / 13260.186 | Native Minimize→Media catalogue update without manual reload, with Bob empty→one and reciprocal account controls, is supported. Reviewed component controls cover filter/selection intent. | The literal **wizard Open in Media** route in AC1 was not successfully captured. `pg-single/ingest-open-settled.txt` and `pg-multi/alice-ingest-open.txt` landed in Chat and are excluded. Later correct source selection proves access, not that button path. Original SQLite Bob/admin and native deliberate-filter branches were not rerun here. |
| UAT245 / 13260.187 | Fresh initialized PG multi ordinary-user uploads now receive200/queued and persist for Alice and Bob; original missing-quota-schema admission failure is cleared. Reviewed real-PG controls preserve quota enforcement/fail-closed behavior. | Native hard-quota rejection was not exercised, and is not inferred from upload success. |
| UAT247 / 13260.189 | Actual queued uploads for owners2/3 produce unique Media1/2 and UUIDs, preserve both records and deny cross-owner reads. This completes the specifically requested native two-owner boundary. | No claim of concurrent browser uploads or arbitrary explicit-ID writers; those relevant sequence boundaries are covered only by reviewed fixture tests. |

A blanket claim that every task AC is natively complete would overstate these captures. The narrow native fault-boundary passes above are defensible; the specific UI/button/dependent-workflow gaps should remain explicit unless separate retained evidence supplies them.

## Reviewed regression evidence and exclusions

- `.tmp/uat-repairs-231-246/review233-238-247/REVIEW.md`: independently executed **165 passed / zero skipped**, including real restricted-role PG sequence/worker controls and SQLite. Trusted persisted-owner/executor scope, warning aggregation, high-water/rollback/cache/interleaving and foreign-table protections reviewed. Production Bandit0; retained Ruff baseline distinctions apply.
- `.tmp/uat-repairs-231-246/review245/REVIEW.md`: independently executed **52 passed / zero skipped**, fresh/repeated normal initialization, restricted roles, actual quota CRUD/admission/limits/unavailable recovery; Bandit0.
- `.tmp/uat-repairs-231-246/review-media/REVIEW.md`: **193 Media/QuickIngest plus37 actual consumer controls**, no skips; compiler90 unchanged, lint0errors/no added signatures. TS Bandit parse limitations remain; no TS security assurance inferred.
- `.tmp/uat-repairs-231-246/native-media-review/REVIEW.md` independently binds original single source after the UAT253 detail repair and successful exact full-source Chat handoff/canonical answer. This review independently rechecks the saved source bytes but does not repeat that downstream audit.
- Initial Settings-navigation and URL/ReferenceError helper failures are not acceptance. The initial `alice-foreign-bob-media.txt` failed; only corrected `-v2` is used. Misnamed/open attempts that show Chat are excluded. No failure artifact was removed.

Audit JSON SHA256: `4dd68f60a13902f62b5c9b87e7d0f8c3549d5f3d7e8c426bc599f1724aa8aab6`. `audit.mjs` is a local, read-only parser; it hashes inputs and writes only the safe allowlisted audit projection. No credentials, raw private content, authentication headers or model reasoning are copied into this packet.
