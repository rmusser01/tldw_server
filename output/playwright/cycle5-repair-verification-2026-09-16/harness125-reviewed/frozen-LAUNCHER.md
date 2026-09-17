# Stage4 private launcher draft

TASK13260 and guard repair TASK13260.125. **The full-UAT gate remains closed.** These artifacts are ready for independent review; no real launcher action has been invoked. Guard tests use fake processes/helpers and disposable temporary records only. No frozen application-source archive, dependency copy, real application profile, browser context, server, fixture database or inference request was created by this preparation task.

## Scope and source

- [matrix-launcher.mjs](matrix-launcher.mjs) adapts the existing [recovery launcher](../fresh-uat-recovery-20260916/recovery-launcher.mjs): separate `check-inputs`, `prepare`, `initialize`, `backend` and `frontend` actions.
- [pg-holder.mjs](pg-holder.mjs), [test_hold_official_pg.py](test_hold_official_pg.py) and [pytest-holder.ini](pytest-holder.ini) retain the existing [official fixture holder pattern](../fresh-uat-recovery-20260916/run-pg-holder.mjs). Only the explicitly invoked holder provisions PostgreSQL databases; the app never receives its pytest variables.
- [ISOLATION.md](ISOLATION.md) defines the reviewed source/dependency copy scheme. [PROTOCOL.md](PROTOCOL.md) defines the existing journeys and evidence limits. This adapter adds no UAT runner or acceptance requirements.

The parent must first release one exact commit, archive it separately for the four cells, and prepare the dependency copies described in ISOLATION.md. The scripts intentionally contain no archive, dependency installation, dependency copy, database-admin SQL or browser automation action. All source/helper/config-template reads use the supplied archive, never mutable `HEAD` or imports relative to the controller checkout.

## Prepared-input contract

1. Each source root is a new archive beneath this packet, with no `.git`. Record the complete archive manifest before adding dependencies or runtime files. The launcher's nine source fingerprints detect changes to selected entrypoints/config/helper files; **they are not a complete source manifest or proof of the declared commit**.
2. The one shared private Python dependency copy is also beneath this packet. Recheck the inventory, then disable only the copied `__editable__.tldw_server-0.1.32.pth` and `__editable__.backlog_py-0.1.0.pth` entries. The first maps both the app and MCP package; these are not two separate tldw/MCP entries. Keep originals and a before/after manifest. Use the private interpreter's absolute `bin/python`; copied activation/console-script shebangs can still contain original paths and are not rewritten here.
3. Each cell has its own three copied Bun dependency trees at `apps/node_modules`, `apps/tldw-frontend/node_modules`, and `apps/packages/ui/node_modules`. Preserve relative links, replace the copied absolute `apps/node_modules/node_modules` self-link with `.`, and omit the copied `.vite`/`.cache` state. Do not copy `.next*` output. No package may resolve into the mutable checkout or outside the cell.
4. The launcher sets all three Python roots, disables user-site and bytecode writes, checks normal Python startup/package origins, and rejects active editable finders/worktree paths. It checks Next/React/TypeScript resolution, shared UI and all symlinks in the three dependency trees. Actual Next compilation and dynamic Python imports remain later startup checks.
5. The parent assigns unused API/frontend ports and a unique run ID. Each cell receives new config, planned credentials, data/cache/log paths and a browser session name. `prepare` refuses existing profiles and any canonical source root already bound to a run/cell, and does not create accounts or application databases. It reserves the source in a preparing record before helper writes, then completes that same record. Browser sessions must be created separately by the parent with fresh storage; recording their name does not create or seed them.

The backend keeps normal auth/setup behavior and requires visible provider configuration. It strips the helper's mock-provider overrides and mock-specific workflow egress overrides. The optional helper ACP stub remains configured but unexercised; this is not real ACP certification. Host HOME remains unchanged; application storage overrides include `WORKFLOWS_ARTIFACTS_DIR`, while hardcoded project-relative system-ops, document-draft and scraper state stays under that cell's archive. Record reused Python/frontend dependencies, system Python executable, browser and model assets explicitly; this is fresh application state, not a clean-machine dependency install.

## Later commands — only after parent release

Run commands from the controller repository. Set these values to the parent's reviewed artifacts; placeholders below are not current prepared paths. Source the existing venv to follow repository policy, but the launcher calls the explicit private Python interpreter.

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
MATRIX_PACKET=/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916
MATRIX_RUN_ID='<new-approved-run-id>'
MATRIX_REV='<approved-full-40-character-commit>'
MATRIX_CELL=sqlite-single
CELL_ROOT="$MATRIX_PACKET/sources/$MATRIX_CELL"
DEPENDENCY_VENV="$MATRIX_PACKET/dependencies/python-venv"
MATRIX_API_PORT='<assigned-unused-api-port>'
MATRIX_WEB_PORT='<assigned-unused-frontend-port>'

node "$MATRIX_PACKET/matrix-launcher.mjs" check-inputs "$MATRIX_CELL" \
  --run-id "$MATRIX_RUN_ID" --source-root "$CELL_ROOT" \
  --python-venv "$DEPENDENCY_VENV" --source-commit "$MATRIX_REV"

node "$MATRIX_PACKET/matrix-launcher.mjs" prepare "$MATRIX_CELL" \
  --run-id "$MATRIX_RUN_ID" --source-root "$CELL_ROOT" \
  --python-venv "$DEPENDENCY_VENV" --source-commit "$MATRIX_REV" \
  --api-port "$MATRIX_API_PORT" --web-port "$MATRIX_WEB_PORT"

node "$MATRIX_PACKET/matrix-launcher.mjs" initialize "$MATRIX_CELL" --run-id "$MATRIX_RUN_ID"
node "$MATRIX_PACKET/matrix-launcher.mjs" backend "$MATRIX_CELL" --run-id "$MATRIX_RUN_ID"
# In a separate parent-owned terminal/process session:
node "$MATRIX_PACKET/matrix-launcher.mjs" frontend "$MATRIX_CELL" --run-id "$MATRIX_RUN_ID"
```

The two launch actions stay attached and write only private logs/process receipts. Both require matching completed preparation and initialization records. Initialization uses [initialize-cell.py](initialize-cell.py) with the explicit private interpreter to call the existing frozen initializer's `main(non_interactive=True)`. It writes a fresh attempt receipt only after normal return. The launcher additionally requires exit0 and matching attempt/preparation identity before recording success; cancellation, SystemExit(0), failures and stale receipts cannot certify completion. Multi-user admin creation and actual authentication remain separate normal operator/UI steps.

`initialize` refuses a second completed initialization. A failed/interrupted attempt leaves logs and any partial data intact; inspect those before explicitly retrying initialization on the same prepared profile. Each attempt has a distinct proof path and log name. Restart backend/frontend using the same successful profile; never call `prepare` to recover a running cell or restore blank configuration over user changes. Failed preparation preserves its preparing record/source binding and needs inspection plus a new run ID and fresh archive, not an automatic delete/reset or source reuse.

For `pg-single`/`pg-multi`, use the matching archive, ports and cell value. **Before `prepare`**, run this separate holder in its own parent-owned session:

```sh
MATRIX_PG_CONFIG='<absolute-path-to-reviewed-mode0600-local-cluster-config>'
node "$MATRIX_PACKET/pg-holder.mjs" "$MATRIX_CELL" \
  --run-id "$MATRIX_RUN_ID" --source-root "$CELL_ROOT" \
  --python-venv "$DEPENDENCY_VENV" --source-commit "$MATRIX_REV" \
  --pg-config "$MATRIX_PG_CONFIG"
```

Wait for the private receipt at `holders/<run-id>-<cell>/<cell>.pg-receipt.private.json` to exist with `status: held`; do not print it. Pass its absolute path as `--pg-receipt` and the same `--pg-config` to `prepare`, in addition to the common arguments. The launcher validates mode0600, live holder PID, exact source/venv/commit/run/cell ownership, official fixture names and matching private cluster credentials. The fixtures are `pg_temp_db` for auth and `pg_temp_db_session` for content; PostgreSQL is required, with no Docker fallback. Runtime DSNs are derived from that receipt; no AuthNZ test pool or custom database provisioner is used.

Run **one cell at a time**. The parent owns serial scheduling, verifies every PID/command before stopping it, and keeps the fixture holder alive through the PG cell. Preserve needed evidence before teardown. After the app processes stop, interrupt the exact holder or create its matching `<cell>.release-holder` marker; returning from the test lets the official fixtures remove their own temporary databases. The scripts do not kill unrelated listeners or release another cell's fixtures.

Next parent steps: review this draft; release archive/dependency preparation; retain full source/dependency manifests; run path preflight; check normal initialization/startup and private storage ownership; open a new native browser context; only then execute the approved matrix after its separate gate opens.

## Validation performed now

Syntax, AST, static security/lint checks and isolated guard tests were run. [launcher-static-validation.json](launcher-static-validation.json) records exact commands/results; [launcher-manifest.json](launcher-manifest.json) records current artifact hashes. [Original draft and causal RED](repair125/RED125.md) remain retained. Tests replace inspection, runtime helpers, networking, spawning and the application initializer with fakes; only disposable fixture files are created. No real launcher action, official fixture collection, app import, package installation, health request or native flow was run. Copy/relocation success, source identity, actual imports, compilation, port availability and runtime behavior remain unverified until the later preparation/startup phase.
