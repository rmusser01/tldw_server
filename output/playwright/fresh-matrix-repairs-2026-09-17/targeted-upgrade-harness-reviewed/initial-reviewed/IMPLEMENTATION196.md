# TASK13260.196 — targeted source upgrade helper

Author source/test freeze for independent review. The fixed product candidate is `a7d3155a567afb25982eb360ea24b973cc3249c9`; no new archive was prepared and no runtime was started. Parent owns actual preparation, profile/holder lifetimes, native acceptance, tasks and Git.

## Scope

Exactly three existing export keywords were added to `matrix-launcher.mjs`: `requireInitialized`, `runtimeEnv`, `frontendEnv`. Removing those keywords reproduces the retained baseline byte-for-byte. There is no existing CLI, role, initialization or environment behavior change.

New `matrix-upgrade.mjs` supports only backend/frontend for the two PostgreSQL cell kinds. It validates the original completed profile/init/source, builds the environment through **runtimeEnv(original)**, validates the original live restricted PG holder, checks a distinct released copy gate/complete tracked-source manifest, and records a separate immutable upgrade binding. It then changes Python source roots or the Next executable/cwd/build. Existing config, credentials, data paths, API cwd, ports, accounts and browser identity remain tied to the original profile. It never calls prepare/initialize/bootstrap, provisions or releases a database, resets data or stops an existing process.

New receipts/logs use `targeted-upgrades/<copy-run>/<cell>`, mode0600. Process startup is recorded as started, never as successful initialization or native readiness. Child errors, nonzero exits and requested cancellation preserve failure receipts. Signal handlers target only the spawned child and are removed at exit. No credentials/environment object is printed or serialized into these receipts.

The helper rejects known root-relative data on the old archive and, before first binding, the new archive. Only an absent or empty regular system-operations lock is allowed; symlinked storage parents and dangling lock links are rejected. This is a deliberately bounded inventory, not a claim of complete arbitrary application-state migration. Both original archives had only empty locks in the prior read-only inspection; that is rechecked at launch, not assumed indefinitely.

## RED and GREEN

- Initial missing-capability run: **37 failed / 0 passed**. The old launcher lacks the three exports and no upgrade action exists. The test harness supplies a no-op only when the new module is absent, producing assertion/lifecycle failures rather than an import error. This is evidence that the new behavior was absent, not 37 distinct preexisting product bugs.
- First implemented candidate: **37 passed / 0 skipped**.
- Added two causal edge probes: **2 failed / 38 passed**. A requested cancellation followed by child exit0 was recorded as success, and a symlinked archive storage parent appeared empty. The exact first candidate and failed receipt are retained. Minimal fixes record requested signals and reject non-directory/symlinked storage parents.
- A final receipt-identity probe retained **1 failed** assertion before adding the exact controller/helper hashes to the immutable upgrade binding. A replay against the reconstructed exact two-field removal also fails as expected; that reconstructed file is labelled explicitly.
- Final focused suite: **42 passed / 0 skipped**, including a pg-multi JWT/auth/content identity control and exact controller/helper byte identity.
- Existing launcher/browser suites: **82 passed / 0 skipped**.
- Final combined frozen command: **124 passed / 0 skipped**, exit0. Exact command is in commands.json; it runs only the new fake-boundary guards and existing synthetic harness guards.

The new tests exercise real filesystem validation against disposable synthetic records and the actual original environment/receipt checks. Python source inspection, dynamic profile helper imports, PG live-role subprocesses, network sockets and spawned children are all replaced. No test invokes the real CLI, app initializer, API/frontend process, PG database, browser, provider or native profile. The role rejection tests therefore complement the existing reviewed real role machinery; they are not a new live PG acceptance claim.

Controls include source/gate/cell/revision/archive and path mismatches, incomplete preparation/init, changed immutable proofs, released/foreign/elevated holder metadata, live-role probe failure, unchanged origin bytes, retained auth/data paths, new frontend dotenv masking/build ownership, existing occupied ports, failure/cancellation cleanup and refusal to adopt unrelated source/build state.

## Static verification

- Node syntax checks pass for the three owned source/test files and the private static validator.
- Existing frontend ESLint rules parse all three owned files: **0 errors / 0 warnings**, no ignored paths. Original launcher baseline also has0/0. Next root and installed React version are provided as contextual settings; no lint rules are disabled. The initial contextual React auto-detection warning is retained separately and corrected in the repeatable validator.
- Exact existing-launcher attribution check passes: only the three export keywords differ.
- Bandit run via the project venv reports **0 findings and 3 JavaScript parse errors**. Bandit cannot assess these JavaScript files; its exit0 is not security proof. No Python/backend file changed.
- Owned patch has no trailing whitespace. No TypeScript compiler or backend suite is applicable to this private Node-only helper change.

## Preserved limits and handoff

`REPORT.md` and its original read-only manifest remain untouched. `USAGE196.md` documents the future command shape and the new gate annotations: `dataPolicy: existing-profile-upgrade` plus the original run ID. The existing copy-only helper remains unchanged and the original browser wrapper remains unchanged. The latter attests the browser profile/redaction behavior, so native phases must also link the new runtime source receipt.

The copy manifest and gate are trusted operator-produced provenance from the reviewed copy-only helper, not signed attestations. The helper checks their consistency and actual source/archive bytes; the parent's released commit/copy procedure remains the source-of-truth step. Dependency origins use the existing inspector and existing reused-installation disclosure, not a new package installation or exhaustive dependency audit.

No source/profile/holder receipt was falsified; no old archive, database, credentials, account, process or browser was changed during implementation. Actual old-record readback, startup, native 246/251/253 acceptance and any fresh full matrix remain pending parent action after independent review. New launch failure does not trigger automatic cleanup or rollback of real startup migrations.
