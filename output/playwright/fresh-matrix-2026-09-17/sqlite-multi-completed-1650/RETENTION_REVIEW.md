# Independent retention and final-native review — SQLite multi-user

**Verdict: retention is verified; no payload correction is requested.** The newly reviewed admin delete/restore and API outage/Retry receipts support their bounded outcomes. This checkpoint records failures and limits; it is **not full UAT success**. PostgreSQL multi-user was still in progress at retention time.

Parent task: TASK13260. Frozen application revision attributed by the checkpoint: `8f8774e6c868b304a96d95ab82e28389c129a78b`. Manifest captured **2026-09-17T16:55:45.354Z**, SHA256 **`bd5462641d65f8c58c7836e553f6073bb12072c9d1a3952b0c6051eae80dcbb4`**. This review performs offline file/receipt verification only, with no browser, runtime, provider, database, product, Backlog or git action. Only this report and `CHECKPOINT_SHA256SUMS` were written; copied files remain unchanged.

## Integrity and working-source fidelity

| Check | Independent result |
|---|---|
| Main manifest | Exact expected SHA256 |
| Payloads | **258/258** stored-byte hashes and declared byte counts match |
| Stored/logical totals | **2,272,384 bytes / 2,571,330 bytes** |
| Compression | **2/2 gzip roundtrips**, decompressed lengths and source hashes match |
| Working-source parity | **258/258** exact logical bytes match original working files at final verification |
| Allowlist parity | All258 intended candidates present; no extra/missing candidate |
| Symlinks/private filenames | None in retained payloads |
| JWT-shaped string scan | Zero matches in all258 decoded payloads |
| Retainer source | Exact bytes match `.tmp/uat-next-matrix-20260916/retain-sqlite-multi.mjs` |

Compressed payloads are `native/sqlite-multi/analysis-failure-preserved.txt.gz` and `native/sqlite-multi/testbot-reloaded.txt.gz`. Their decompressed bytes preserve the original wrapper evidence, including the original provider text; this report does not reproduce provider reasoning. There is no source-manifest gzip in this particular checkpoint.

The main manifest covers258 payloads. README, manifest and `retention-source.mjs` are expected auxiliary files outside its own file list. The new checksum index binds those auxiliaries, this review and every payload (262 entries), excluding only the checksum index itself. This supplies an explicit binding rather than claiming the main manifest covered its own metadata or this later review.

## Exclusion and credential-scan scope

The retainer chooses only top-level `.txt/.json/.md/.js` files under `native/sqlite-multi`, matching `sqlite-multi-*` audits, the two named safe API helpers, matrix progress and the controller snapshot. Names containing `private` or `credentials` are excluded; only ordinary files enter. Directories, browser profiles, private input helpers, raw console/backend logs and configuration/credential payloads are not copied. Safe redacted log excerpts and administrative Backlog receipts may be present; they are not additional native success evidence. References to private files inside public scripts/wrappers are provenance, not copies of those private files, and were not followed.

The retained code scans candidate bytes against known SQLite/PG profile secrets, provider secrets, provisioning/runtime PG passwords and JWT patterns before writing. The manifest reports **26 known-value encodings,258 candidates,0 known-value matches,0 JWT matches**. This reviewer inspected that strategy and independently repeated only the JWT-shape scan; actual private credentials were not read, so the known-value scan is a retained author result, not a falsely claimed independent secret comparison. Filename exclusion is not treated as a complete secret detector.

The copied retention script is provenance. Its original relative paths assume execution from the working packet; this review did not execute it from the checkpoint or elsewhere.

## Frozen harness and earlier preparation chain

The current release gate SHA256 **`c692f5d8bb0f1c653e9172515477d0c90cb46002e4dea68fd916c1fa3ff629d0`** is byte-identical to the copy retained in the declared base checkpoint `../sqlite-single-checkpoint-1331`. Each of its15 harness files matches both the current working packet and the base copy:

`PROTOCOL.md`, `ISOLATION.md`, `LAUNCHER.md`, `matrix-launcher.mjs`, `initialize-cell.py`, `bootstrap-admin-cell.py`, `pg-holder.mjs`, `pg_role_adapter.py`, `test_hold_official_pg.py`, `pytest-holder.ini`, `matrix-browser.mjs`, `prepare-copies.mjs`, `reviewed-pth-inventory.json`, `observe-native.js`, `events-native.js`.

The base also retains the SQLite-multi archive manifest SHA256 **`703f3cd208450440ee4ce31ba51526200257f363fc7fa3fb07c4811d4fa9cb36`**, referenced by the included source diagnoses. This review verified32 source-diagnosis hash entries against the frozen source archive paths, but did not rehash the entire application archive or repeat source tests. The archive manifest and harness dependency on the base are explicit; this is not a self-contained application-source distribution.

## Row and prior-audit coverage

Every SQLite-multi cell in retained `matrix-progress.json` has evidence present here: **88 references,85 distinct logical payloads,0 missing**, including gzip aliases. The12 row reference counts are **8,11,8,14,2,5,4,4,7,4,6,15**. Evidence for other cells belongs to their earlier or future checkpoints; its absence here is not a SQLite-multi omission. The README/controller/progress retain the234 generation failure,235/240 Study issues,241 handoff issue,242/243 presentation issues,244 stale catalogue, Wikipedia denial and image/visibility limits.

All11 SQLite-multi audit files are retained. Nested JSON audits contain111 input references: **109 exact hash matches**, using this checkpoint plus base `PROTOCOL.md`; the other **two are historical controller versions**. The setup/Chat Markdown table adds **20/20 exact input hash matches**. No native input is missing. All referenced report hashes checked through their companion records match.

The historical controller hashes in the mid-native and late-isolation audits are respectively `d898d4a4bb1845d5e94fdf30461a5cdf10f9ffb4cc74f1ed908ccddac8c8aaf5` and `b5736e57031293953805446dffd1226fdde51d653034a088488fd812b91f3e0b`; the late audit additionally records an earlier controller hash in its history. These old exact controller byte snapshots are **not** separately preserved in this checkpoint; only their hashes/audit descriptions remain. The latest controller snapshot is retained and verified. This is a provenance limit, not a missing native receipt, and no claim of universal historical-controller roundtrip fidelity is made.

Earlier setup/mid-run reports intentionally state pending boundaries at their own capture times. Later receipts and the current controller supersede those pending states; they do not rewrite history. Prior reviews' source diagnoses and acceptance results were not all rerun as product tests.

## Newly reviewed administrator delete, Trash and restore

The safe normal-login receipt records **admin1 at16:42:12.202Z/.238Z**. Own ingest job7 (`0c059604-9c61-4df9-ad56-f72b13a379e0`) completes at **16:43:41.470Z**, owner1, Media1, media UUID **`1589145e-98bf-4c35-900d-5b744672afd3`**. This is the administrator's separately ingested fixture, not Alice's or Bob's Media1. Analysis was off.

- Pre-delete detail200 at **16:43:42.035Z** and later reads contain the original **1,914 characters**, SHA256 **`a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`**, with exactly v1 UUID **`8101d879-6269-4bb0-8710-534ae0952de5`**.
- Native confirmation performs **DELETE204 at16:45:14.733Z**. The following actual active catalogue200 at **16:45:14.794Z** returns items0/total0, and the UI has0/0/no selected Media. This establishes the post-delete empty state even though the earlier pre-delete catalogue had the separately tracked stale-count defect.
- Trash200 at **16:45:16.343Z** contains exactly item1/title and deletion time **16:45:14.727Z**; the UI renders **Sep17,2026,9:45AM**. Trash remains reachable. No permanent deletion was attempted.
- Native Restore returns **200 at16:46:02.481Z** with the exact source and original v1 UUID. Subsequent detail200 at **16:46:03.072Z** and **16:46:33.119Z** preserves them; actual catalogue200 at **16:46:03.054Z** contains item1/total1 and the corrected observation at **16:46:41.084Z** shows1/1/source1914.
- The root media UUID is emitted by the ingest result. Trash and post-restore catalogues provide ID/title, not that root UUID. Exact restoration is independently supported by same ID, exact source bytes and unchanged version UUID; do not claim a directly re-emitted post-restore root UUID.

`admin-restore-final.txt` retains the strict-locator failure **after the successful restore**, because two source paragraphs contain Dr.Mira Vale. `admin-restored-observed.txt` only corrects the read-only locator with `.first()` and captures the settled state; it does not repeat Restore. No failed restore or extra mutation is inferred from the observer error.

Ordinary Alice's permission-denied Delete state is retained in earlier analysis evidence. The administrator used its own source with ordinary admin authority; no role alteration or authorization bypass is demonstrated.

## UAT244 remains a failure

The included UAT244 audit reports the active ingest wizard's missing same-owner catalogue refresh with high confidence, not a causal test. Native evidence has Bob source detail200/1,906 characters and admin detail200/1,914 characters alongside stale0/0 catalogue UI. The admin view remains stale32.412 seconds after its first successful detail read. The cumulative capture has no new catalogue request between completion and deletion. Later restore/navigation obtains items1/total1; that recovery does not erase244 or prove that deletion is necessary. This review confirms the referenced receipts/source-hash bindings and retains the diagnosis's stated causal limits; it does not claim to repair or dynamically retest the implementation.

## Newly reviewed outage, Retry and shutdown

- `api-outage-start.json` records controlled graceful stop of owned API45087 at **16:47:08.245Z** (original startup15:10:23.101Z, source unchanged per receipt). Native reload shows **Backend readiness check failed**, with Retry/settings/diagnostic options, at **16:47:42.691Z**. The independent port receipt at **16:47:43.666Z** records **ECONNREFUSED on18601**.
- The later stop record identifies replacement API94090 with startup **16:48:03.842Z**. The bounded recovery script performs only a visible **Retry** and waits for source content; it does not re-enter credentials. Actual identity returns **admin1/200 at16:49:09.697Z**, followed by **Media1/200 at16:49:09.886Z**, exact1,914-character source and original v1 UUID. Another admin1 identity and RAGhealthy200 follow. The snapshot shows1/1/source. Notifications are still connecting in that immediate snapshot, so fully settled notification recovery is not claimed.
- Same-runtime-profile/source continuity comes from the retained lifecycle records and parent frozen-source attribution, not a fresh reviewer inspection of the live process or private configuration. The new startup is recorded retrospectively in the stop-request receipt; this checkpoint has no separate launch-command/full-config assertion from this reviewer.
- `apps-stop-requested.json` records SIGTERM for API94090/frontend43230 at **16:49:49.961Z**. `apps-stopped-verified.json` at **16:51:18.962Z** records browserClosed, ownedAppsExited and **ECONNREFUSED on both18601/18681**, with data retained. These are offline verified receipts; this reviewer did not send signals or inspect processes.

This supports the cell's controlled API-unavailability recovery and cleanup account. Expected network/authorization errors and the retained observer failure prevent a blanket clean-console claim. It is distinct from upstream provider outage, vision inference, true-hidden-tab behavior or complete application recovery across every subsystem.

## Final scope

No payload corrections, new findings or repair changes are requested by this retention review. The recorded cell outcomes are bounded and include failures. Prior single-user checkpoints and this SQLite-multi checkpoint together still do not establish successful full-matrix UAT; PG-multi remained ongoing. The checksum file binds the exact final review and auxiliaries after all offline checks.
