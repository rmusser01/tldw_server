# Targeted upgrade preparation: UAT246 / UAT251 / UAT253

Parent TASK13260. Read-only design, 2026-09-17. **No upgrade helper, archive copy, initializer, process, browser, fixture or database action was executed.** Only this private report, a sanitized path/stat inventory and their hash manifest were written.

## Recommendation

Use a small **separate targeted-upgrade launcher** with the existing profile and live official PostgreSQL holder. Reuse the existing launcher environment and validation functions on the **original profile**, then select the new frozen code roots and Next entrypoint in a distinct upgrade record. Preserve the original profile, completed initialization, holder receipt, credentials, config, database paths, ports and browser session.

Run ordinary API startup against the existing databases. These three fixes change response headers and repository query bindings; they add no schema migration or account bootstrap. A second first-run initializer is unnecessary. This is **upgrade targeted acceptance**, not a fresh matrix cell or proof that the old initializer ran new code. The later full fresh matrix remains separate.

### Existing data identity

Both targeted cells have original source revision `86458ab88ce3fa62e6518c9d813c3860254ddb2c` and run ID `repairs231-250-targeted-20260917`.

| Cell | API / WebUI ports | Preserved state |
| --- | --- | --- |
| pg-single | 18702 / 18782 | Original profile, auth/content fixture databases, Media 1 and canonical chats |
| pg-multi | 18703 / 18783 | Original profile, auth/content fixture databases, accounts and canonical chats |

`path-inventory.json` records only selected paths, source identity and filesystem metadata. No credentials, configuration contents, database contents or session tokens were inspected for this proposal. Database record identity/content still needs the parent's existing authorized before/after native readback; this inspection does not independently certify those records.

## Why the ordinary launcher cannot switch source

- [prepare, lines194–227](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/matrix-launcher.mjs:194) refuses existing profiles/roots, then creates credentials and rewrites config/setup state. It must not be used on this data.
- [launch, lines234–260](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/matrix-launcher.mjs:234) accepts only the original run ID, checks the original completed preparation/initialization and source fingerprints, and refuses a repeated completed initialization.
- [readPgReceipt, lines134–148](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/matrix-launcher.mjs:134) binds the official fixture holder to its original run/cell/source/commit/interpreter. Editing profile source fields would invalidate that proof. The holder's original source identity describes provisioning; it must not be relabelled as the current API source.
- [holder lifetime](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/test_hold_official_pg.py:21) owns both databases until release. Starting another holder provisions different databases; stopping the existing holder permits fixture teardown. Keep the existing holders alive throughout acceptance.

## Smallest proposed implementation, after release

### Two helper files and focused synthetic tests

1. Add **export keywords only** to the existing private `requireInitialized`, `runtimeEnv` and `frontendEnv` functions in `matrix-launcher.mjs`. Their behavior and the original CLI remain unchanged. Existing exported `privateJson`, `recordPath`, `specFor`, `inspectInputs`, `baseEnv` and safe error reporting are already reusable. Do not duplicate the environment builder or change `readPgReceipt` to accept mismatched source identities.
2. Add one private `matrix-upgrade.mjs` next to that launcher. Only `backend` and `frontend` actions are needed. Inputs identify the original run/cell and the parent's released new copy-preparation manifest. It writes new upgrade receipts/logs under an exclusive upgrade directory, with no writes to old preparation/initialization/process receipts.
3. Add a focused synthetic test file using fake origin/role/process boundaries and temporary fake records. No new generic runner, provisioning adapter, dependency installer or product change is needed.

### Preparation can reuse the existing copy-only helper

[prepare-targeted-copies.mjs](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-repairs-231-246/native-preparation/prepare-targeted-copies.mjs:1) already takes a new run ID, exact committed revision and released `targeted-acceptance` gate; it creates only new archives/dependency copies and receipts. It neither creates a profile nor touches a holder. Its existing checks reuse the isolated Python installation and copy three cell-local Bun trees, preserving startup-hook provenance and correcting the copied self-link.

The parent can use a new copy run ID for this upgrade and retain an explicit upgrade purpose/data-policy annotation in that gate and upgrade receipt. Do not reuse the original copy destination. The new copy run does **not** become a replacement profile run. No copy command was run here.

### Validate old identity, then new code

Before any child process:

- Check original profile realpaths, run/cell/mode/engine and completed preparation. Call `requireInitialized(original)`; bind hashes of the unchanged original profile and initialization receipt in the new private upgrade record.
- Inspect the original source and compare its nine recorded fingerprints, ensuring the environment helpers are still the reviewed original bytes. Call `runtimeEnv(original)` so its existing PG-holder and credential checks run against their true original identity.
- Verify the new copy-only gate, completion receipt, exact commit and **complete tracked-source manifest**. `inspectInputs` alone records only nine fingerprints and does not prove the declared commit. Reject escapes, unexpected symlinks, mutable checkout roots or mismatched cell/source manifests. Bind one new archive to one upgrade/cell using an exclusive record; reject another profile/cell binding or a changed failed attempt rather than silently rebinding.
- Run `inspectInputs` on the new archive using the explicit preserved private Python installation. This checks all three Python roots, no editable fallback, third-party origins, three Bun trees, shared UI and Next/React/TypeScript origins. Preserve reused-dependency disclosure.
- Reuse [pg_role_adapter.py:check_files](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/pg_role_adapter.py:128) before launch. It validates the unchanged private config/holder relationship and live runtime/database identity. No provisioning credentials go to the application. Require the existing direct LOGIN role to remain non-superuser/non-BYPASSRLS/non-INHERIT/non-CREATEDB/non-CREATEROLE/non-replication, without memberships, with row security on.
- Parent stops only the exact old API/frontend processes, preserving holders. The new helper checks the original ports are free and never kills an unknown listener or performs automatic holder cleanup.

### Environment and process binding

| Item | Upgrade behavior |
| --- | --- |
| Backend environment | Start with `await runtimeEnv(original)`; change only `PYTHONPATH` to the inspected new archive, new `apps/mcp-unified/src`, and new `packages/tldw_profile_core/src` |
| Python | Execute the inspected absolute private `bin/python`; do not use copied activation scripts or console-script shebangs |
| Backend command/cwd | Normal `-m uvicorn tldw_Server_API.app.main:app`, original API port, **original profile root cwd** |
| Config, credentials, DBs, caches, uploads, workflow artifacts | Preserve the original environment values and files; do not regenerate, reset, rebase or delete them |
| Frontend | Reuse `frontendEnv` with an ephemeral view containing the original profile/root/ports plus inspected new frontend and a unique new Next dist name; do not persist this view as a profile |
| Next executable/cwd/build | Inspected new archive Next CLI and frontend cwd; exclusive build receipt under the upgrade packet, build output only inside the new archive |
| Logs/process receipts | New upgrade directory, mode0600, unique attempt IDs; no overwrite of old process receipts or raw environment logging |
| Browser | Existing session/origin and existing credential redactor; parent explicitly records the new runtime source receipt with every upgraded acceptance phase |

[config.py:319–334](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/config.py:319) loads dotenv with `override=False`; the explicit new process `PYTHONPATH` therefore takes precedence over the original profile's saved value. Keep `TLDW_ENV_FILE_EXCLUSIVE=true` and the original config/env paths. New frontend dotenv variable names must still be masked by the existing builder.

The original helper also retains an optional ACP stub/config and source-relative `ACP_RUNNER_CWD` ([live profile builder](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/scripts/live-tier-uat/profile.mjs:135)). Preserve that existing configuration, disclose the old helper path and keep ACP out of these three targeted claims. It is not justification to rewrite private configuration or claim ACP ran new code.

## Startup versus initialization proof

The real [main lifespan](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/main.py:1249) calls the [startup sequence](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/services/lifespan_startup_sequence.py:64), core initialization and [AuthNZ startup](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/services/startup_auth.py:37). Normal startup opens the configured pool, checks schema readiness, performs PG extras and existing seed/provider initialization. Content stores retain their ordinary constructor behavior. Do not bypass these steps or seed/repair data through ad hoc SQL.

A process-spawn receipt is only `started`; actual health/readiness and authenticated target reads are separate evidence. Record startup failures without forging a successful initialization. If a later schema requirement genuinely requires the standalone initializer, stop and propose an explicit upgrade initialization attempt with its own new normal-return receipt. The existing [initializer](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/AuthNZ/initialize.py:1360) can generate missing secrets/bootstrap accounts, so blindly rerunning it is broader than these three code-only repairs. Do not overwrite `initialized.private.json` or describe it as proof for the new revision.

## Archive-local storage limitation and actual check

`get_project_root()` derives its root from the imported file, not cwd ([Utils.py:149](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Utils/Utils.py:149)). Moving imports consequently moves these default paths:

- [system operations](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/services/admin_system_ops_service.py:40): `Databases/system_ops.json` and lock;
- [document draft handoffs](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Ingestion_Media_Processing/document_upload_drafts.py:72): `Databases/document_upload_drafts.db`;
- [scraper cookies](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py:461): `Databases/webscraper/`;
- [audio fallback downloads](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py:1687): `Databases/downloads/audio/` absent an explicit configured path.

Read-only metadata inspection of **both actual original targeted archives** found all listed data paths absent, including document DB WAL/SHM, and only a zero-byte `system_ops.json.lock`. No contents were read or transferred. Recheck this exact bounded inventory after old processes stop; fail closed if data appears. Leave old archives and their empty lock untouched. This is not an exhaustive inventory of every possible root-relative feature or a proof that all application state survives arbitrary source relocation.

Original Media 1 is a persisted plaintext ingestion; the retained native job result identifies the original profile's `tmp/media_ingest_job_…` source path, not the archive. Preserve the entire original profile including that temp tree—no cleanup or re-ingestion. UAT253's verified failure was the MediaFiles lookup SQL, including absent-original-file success, not proof that source text was lost. The parent must read the **same Media ID/UUID and source content** after upgrade. Preserve canonical chat IDs and acknowledged message IDs through readback; do not regenerate them merely to produce passing data.

## Receipt and test contract

The upgrade receipt should distinguish:

- immutable origin: original profile/init/holder hashes, original source revision, cell/run, data paths and unchanged port/browser identity;
- new execution: new archive/preparation manifest hashes, new commit, code/dependency origins, helper hash, exact non-secret command/cwd, process PID/start/end and private log path;
- readiness/native evidence: explicit observations linked to that process/source record, with the label `upgrade-targeted-acceptance`.

Never copy credential values/DSNs/environment objects into a public receipt. Preserve failure attempts; do not call fixture teardown or erase old evidence when a child fails. The existing [browser wrapper](/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/matrix-browser.mjs:25) validates the original prepared browser profile and redacts known secrets, but it does **not** attest the currently listening API/frontend revision. Keeping it unchanged is appropriate only with the separate explicit current-runtime receipt per phase.

Before actual use, focused fake-boundary tests should prove: unchanged old proof bytes; rejection of incomplete/mismatched origin or released/foreign/privileged holder; complete new-source/cell manifest confinement; exact inherited auth/storage/ports with only intended code roots changed; new Next masking/cwd/build ownership; no writes before validation; no automatic reset/DB provisioning; port conflict/child failure propagation; failed attempt preservation; and archive-local data appearing before launch blocks the action. Rerun the existing launcher/browser guards after export-only changes. These tests and all actual preparation/startup/native actions remain **unexecuted and gated** in this report.

## Parent's execution order after approval

1. Freeze the committed 246/251/253 source and helper review; release a new copy-only targeted gate with explicit upgrade data policy.
2. Prepare new per-cell archives through the existing copy-only helper, retaining full source/dependency provenance. Create no new profiles/holders.
3. Retain original named-record native evidence, stop exact old app processes, recheck storage/proofs and live roles, then start the new backend/frontend serially with the proposed upgrade helper.
4. Verify normal startup/auth and original Media/chat readback, then perform targeted 246 streaming, 251 Save packs and 253 original Media detail acceptance. Keep old and new source windows distinct.
5. Preserve both original and upgrade evidence. Releasing any holder remains a later explicit parent action after all necessary data acceptance is finished. A fresh full matrix uses its separate new-profile procedure.
