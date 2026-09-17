# Independent retention review — PG multi-user final checkpoint

## Disposition

**Retention approved after the recorded row 7 reference correction. No remaining retention blocker found.** This approves evidence fidelity and bounded interpretations, not application acceptance. The frozen matrix records all 48 outcomes with failures; UAT231–246 remain 16 unresolved findings. No full-matrix success, clean-console, vision, or whole-product isolation claim is supported.

Parent task: TASK13260. Product revision: `8f8774e6c868b304a96d95ab82e28389c129a78b`. Review performed offline, without browser, runtime, database, inference, test, source, task, or git operations. Only this report and its JSON index were written. No private credential/helper/raw-log files were opened.

## Integrity and coverage

- Final manifest SHA-256: `1d4d60585e9cb0a48544ce212448559f51a4592f7b1bd93ca04f1be6b4f2c593`.
- All **186 payloads / 1,524,044 bytes** match their retained hashes, byte lengths, source hashes, source lengths, and current working originals. Controller bytes also match the declared working document. There are **zero gzip payloads** in this packet; no packet roundtrip is claimed where compression was not used.
- All **89 PG-multi row-reference usages / 83 distinct paths** resolve to manifested payloads. References for other matrix cells belong to the prior cell checkpoints. This packet is intentionally dependent on `../sqlite-single-checkpoint-1331` for the release gate and harness.
- The current and base release-gate bytes agree: SHA-256 `c692f5d8bb0f1c653e9172515477d0c90cb46002e4dea68fd916c1fa3ff629d0`. All **15** current harness files match both their gate hashes and the retained base copies.
- Retainer SHA-256 `9a3e048d5df5b8b659c9a17d0cc4cc1f64b090c027086b3551fa68de501712bd` matches the working `retain-pg-multi.mjs` exactly. Its relative paths describe the original working context; the copied retainer is provenance, not an instruction to rerun it inside this checkpoint.
- The initial manifest `081a0d27de91f98ffcc2017170a3fbbbea53d887e5b09949bb4247d2a8866015` referenced nonexistent `pirate-created-entry.txt`. The parent corrected that reference to retained `pirate-create-entry.txt`. Comparing the preserved initial manifest proves only the `matrix-progress.json` payload changed; all other 185 payload records are identical. The final manifest records the correction and old manifest hash. No outcome changed.

README, manifest, retainer, and these review auxiliaries sit outside the main payload list by design; the parent's final auxiliary checksum index must bind them. The review JSON binds the three preexisting auxiliaries and this Markdown report. No payload was edited by this reviewer.

## Exclusions and secret-scan limits

The retainer selects regular allowlisted native text/JSON/JavaScript and matching audits, excludes private/credential-named files and directories, and does not copy browser profiles or raw private logs. Safe redacted CLI outputs and sanitized error projections are evidence, not raw logs. Inspection found no symlink payload or excluded private/helper/log payload.

The retained scan reports 29 known-value encodings, 186 candidates, zero known-value matches and zero JWT matches. Its code gathers prepared profile secrets, fixture/provisioning and both PostgreSQL runtime passwords, and the provider-secret list, checking literal/encoded forms before copying. This reviewer inspected that strategy and receipt without reopening credential files; therefore **known-value scanning is an author-run attestation, not an independent secret-value rescan**. An independent JWT-shape scan of every retained decoded payload found zero matches. No claim covers arbitrary unknown secrets or future file additions.

## Independent audit bindings

- UAT245 audit Markdown: `489db100fdefc37ce5b1fc4671bf3479b2bd64cc24f29dbc46c9ceddcff78b2b`; JSON: `291294f7d14fefdfc408c7f10e205842fd59ab5aa85436ff84a18a1646c2869e`.
- UAT246 audit Markdown: `392248aa3a22c494de2baeb8233b25697ce30c8784b771876e873a5b1d550175`; JSON: `29c916876941ddb016644ea6ff796969a586ee36054fd6df6e7b992dac2d117a`.
- All **10 native references** across those audits match their recorded hashes. The one SQLite comparison input is retained in `../sqlite-multi-completed-1650` as `testbot-reloaded.txt.gz`; its decompressed bytes match the original audit hash. The other nine references are present here.
- All **34 cited frozen source-file entries** match the archived files: 18 for quota diagnosis, plus eight PG/SQLite pairs for timeout comparison. This verifies the cited source boundaries, not a new full-archive or git verification. Related Backlog record references remain external historical context; they were not used as native outcome proof. Sanitized projections containing private-log provenance were read, while their underlying private logs were not.

## Decisive native outcomes and limits

### Authentication and actual outage

The startup receipt reports official fixture holders and a restricted runtime role with `rolsuper=false`, `rolbypassrls=false`, no memberships and no creation privileges. This is recorded role evidence, not a fresh database query.

The expiry child context received a real 1,800-second access lifetime at 16:58:30.444Z, then parked with zero pages and service workers. Return began at 17:28:59.858Z: independently calculated **1,829.414 seconds**. Sorted event timestamps show identity 401 at 17:29:00.232Z, refresh 200 at .259, then Alice identity 200 at .283 and owned Notes 200. The scripts use real elapsed time and ordinary navigation, without clock/token/storage injection. This proves access-token expiry recovery, not refresh-token expiry or a forced sign-out contract.

The final controlled API outage has a stop receipt, actual port 18603 `ECONNREFUSED`, visible readiness Retry, replacement API startup, then Alice identity 200 and the same owned Biology Note 200/version 5 at 17:49:46.711Z after Retry. The action does not reenter credentials. Expected offline network errors are not a clean-console failure or an authentication leak.

Browser closure is retained. Stop verification at 17:51:04.681Z records both application ports 18603/18683 refusing connections, with official PostgreSQL holders 29823 and 96865 still alive. This review did not perform new process/port checks.

### Ownership and reciprocal browser Back

The final independent-password-login API helper completes **30 recorded checks** with authenticated Alice 2 and Bob 3. Owned Notes read/write succeed, foreign Note GET and valid PUT return 404, foreign Chat detail/messages return 403, foreign cards return 404, and each populated deck catalogue contains its own owner/name. Own Chat/card responses remain readable. This is the named resource boundary, not exhaustive domain authorization coverage.

The initial 11-record attempt stops because the helper wrongly required 404 for a correctly denied Chat 403. Its Alice edit had already been restored to version 3 before that failure. The corrected complete run restores Alice Note `7b4f7117-4139-4f35-9107-a696145f7810` to original content at **version 5**, and Bob Note `d5e4cbfb-96e9-4fca-9b2d-c25ead5ade8c` to original content at **version 3**. Exact restoration content hashes are in the review JSON. These were real owned write/restore probes; calling the native helper wholly read-only would be false.

Both retained history scripts perform three literal `goBack()` calls after ordinary account switches and wait for the relevant pages. Alice→Bob settles under Bob 3 with blank pasted text/default new-deck fields and no Alice source context; Bob→Alice settles under Alice 2 with blank pasted text/no Bob source context and Alice's existing private deck selected. Do not claim default deck names in that second direction. UAT243 remains a separate ordinary-chat/Character-mode presentation mismatch, not an observed cross-owner disclosure. Populated Media/job ownership could not be accepted because ingestion was blocked.

### Image, ingestion and generation

The PNG upload/unsupported-provider guard is bounded: completion count remains 4 before and after Send/Retry, and the reloaded DOM image is complete with natural size 128×128. This proves the uploaded image element and local guard behavior, not vision inference, image-byte identity, a successful image completion, or true hidden-tab operation. Earlier empty canonical reads precede the reload and are not a settled post-reload canonical proof.

Alice ingestion returns real 413 `storage_quota_exceeded` / `Quota check unavailable` at 17:07:59.477Z; admin repeats it at 17:46:13.017Z. UAT245's sanitized failure and source diagnosis identify missing PostgreSQL `storage_quotas` schema. This is correctly fail-closed quota handling, not an oversized input, exhausted allowance, successful source ingestion, or a quota bypass. Source QA, reanalysis, Trash and populated Media/job checks remain blocked in this cell.

The Biology generation browser request starts 17:14:07.464Z and returns 500 at 17:14:37.473Z after 30.009 seconds, while the sanitized backend projection records a later 200 at 17:14:43.511Z. UAT234 remains failed end-to-end; the later backend result does not prove five visible drafts or a completed five-card Study journey. Manually created ownership cards are separate fixtures.

### Study, reuse, Pirate and TestBot

The reused Pirate answer has three canonical rows and a real ARRR response, but no verified weather data. Note/card/backlink evidence concerns that answer, not an ingested-media-grounded answer. The reviewed card `8629d1cd-6ab9-417d-a807-121c6fac8d93` receives real Easy, Good and Hard writes. Practice with scheduling off leaves the observed write count unchanged. After Good, the UI Hard preview is 6 days while the server supplies/saves 14 days: UAT235 remains. After Hard, the card has zero lapses but analytics reports 33.3333% lapse rate and 66.6667% retention: UAT240 remains. The singular completion reads “1 cards reviewed this session”: UAT242 remains. None of these observations proves generated five-card coverage, mixed scheduling, early End, or unobserved transient toasts.

TestBot's actual complete-v2 gets HTTP 200 headers at 17:28:01.096Z; the body-read attempt ends at 17:28:46.133Z, **45.037 seconds** later, without a captured body. That does not constitute successful model output. The UI shows a timeout; a subsequent settled same-chat view contains only the original user message, with no assistant/BEEP BOOP. Earlier canonical one-row reads occur before the timeout, and the first reload observer itself times out; neither is misrepresented as a successful settled reload. The later settled snapshot supplies the visible one-message outcome. UAT246's client-idle-timeout interpretation is supported, while the upstream cause remains unproven. The separate World Book 500/UAT239 also remains; provider dispatch proceeded despite it. No provider reasoning is reproduced or inferred.

## Final boundary

The retention package is coherent after its one documented reference repair. All recorded failures and harness mistakes remain visible. Forty-eight recorded matrix outcomes are not forty-eight passes. Native acceptance and repairs for the open findings remain outstanding; this review authorizes no runtime restart, repair, task closure, or broader isolation claim.
