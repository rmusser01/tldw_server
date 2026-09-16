# Private recovery audit — 2026-09-16

Scope: parent TASK13260/.75; static review of the private runtime launcher and official PG fixture holder, plus read-only inspection of parent-observed startup errors. Root owns process starts, model configuration, native UAT, Backlog, and product changes. This audit did not start a service, connect to/create a database, call inference, or edit application code. No full-isolation or native-acceptance claim is made.

## Immediate launcher corrections

1. Port bind failures previously all read “occupied.” The private port probe now distinguishes EADDRINUSE from EPERM/EACCES permission denial and retains unknown error codes. Injected independent error controls passed; no real listener was created by this audit.
2. Runtime PYTHONPATH previously included only the repo. The installed environment could not find `tldw_profile_core`; the package is already present at `packages/tldw_profile_core/src`. Runtime now uses the repo plus both supported source roots from pyproject.toml: `apps/mcp-unified/src` and `packages/tldw_profile_core/src`. Before control could not find the former package; corrected path imports it successfully. No dependency installation.
3. A private cache symlink for Next's build root broke dependency lookup. NODE_PATH temporarily proved/fixed the exact missing CJS Next runtime module, but the next parent startup proved generated ESM aliases remained broken. That shim is removed. Frontend mode now uses the real unique frontend `.next-live-tier-recovery-targeted-NAME-20260916` directory, matching the canonical helper. It is a declared generated-build mutable-root exception. Only an owned symlink to the preserved old profile cache may be unlinked. Existing foreign links/directories are rejected; a profile-private ownership receipt permits reuse of the real directory. No existing application state/cache is deleted.

Exact alias proof: the retained failed cache contains relative symlinks for `@noble/hashes-eebe4b670184febd` and `i18next-77c2941f50c46600`. Both resolve to missing destinations from the `.tmp` real path; the identical relative targets exist from the frontend real build directory. See `real-build-resolution-check.json`. The normal unhashed `@noble/hashes` package is not directly exposed in the frontend; the generated alias is the relevant check. Parent must verify the restarted actual frontend response.

Root's sqlite-single vision config was not regenerated. SHA256 before and after these corrections: `8e6bf8b4a8dc8875d98b76f082e8e38da8a4e5dd2c9f232e8c56e67910417f41`. Runtime env is computed by the launcher at each launch. Prepared env JSON/.env snapshots predate the PYTHONPATH correction; the explicit runtime environment takes precedence. No profile re-prepare should overwrite root's vision changes.

## Credentials, flags, and holders

The runtime/holder each build an independent strict base allowlist: PATH, HOME, USER, LOGNAME, SHELL, LANG, LC_ALL, LC_CTYPE, TZ. Operator application credentials, PYTHONPATH, NODE_OPTIONS, database URLs and proxy variables do not flow from the parent process. HOME retains its real value; CODEX_HOME is not passed. Backend dotenv lookup is exclusive to the private profile. Next reads local dotenv key names to mask them with empty values, not to inherit their values. Generated credentials/config/receipts/logs are private0600 beneath0700 profile parents. Runtime snapshots and logs must not be published raw.

Application environment excludes TEST_MODE, PYTEST_*, TLDW_TEST_* and CHAT_FORCE_MOCK. `EVALUATIONS_TEST_DB_PATH` is the canonical live-tier helper's production storage-path override, not a test-mode switch. Provider mock environment overrides are removed; visible Setup configures normal Chat providers. Root separately configured ordinary vision provider settings in sqlite-single, preserved here. The upstream ACP stub configuration remains optional and unexercised; this profile cannot certify real ACP downstream behavior.

The PG holder invokes only `tldw_Server_API.tests._plugins.postgres` with plugin autoload disabled and private root/config; the fixture plugin owns all creation/deletion. `pg_temp_db` provides auth and `pg_temp_db_session` content. The two named holders have independent fixture databases and lifecycle receipts. The holder's test-only variables are never copied to app env. It reads the private connection config without printing it, disables Docker fallback, requires PG, writes0600 receipts, and returns to fixture cleanup only on its release signal/marker. There is no custom SQL database creation in the harness. Root reported both holders held; this audit did not start, stop, or query them. Receipt validation checks private path/mode, profile, live PID, distinct fixture names, and the owned server connection fields. It is not independent proof of live database health.

## Material isolation limits

| Path/feature | Source and actual behavior | Consequence |
|---|---|---|
| Repository `Databases/system_ops.json` and `.lock` | `admin_system_ops_service.py:40,70-130,502`; `auth_deps.py:1558` calls maintenance lookup during ordinary authenticated requests | Shared maintenance state is read and its lock acquired even outside Admin. No supported path override found. Scoped inspection found maintenance disabled and zero enabled feature flags; root accepted this inactive baseline for targeted118 only. Full fresh matrix needs a separately isolated source root or scoped product configuration support. |
| Repository `Databases/document_upload_drafts.db` | `document_upload_drafts.py:75,198`; lazy store in `media/document_upload_processing.py:81,111,136` | Document upload preview/draft operations can create/update a shared DB. File absent at audit. No env override found. Image Chat use of this store is not established. |
| Repository `Databases/webscraper/{cookies.json,content_hashes.json}` | `enhanced_web_scraping.py:466,510,928-929` | Optional enhanced scraper defaults read/write shared cookies/hash state. Cookies absent; an existing content-hash file is present. Avoid inferring fresh web-scraping state. |
| Workflow artifacts | `Workflows/adapters/_common.py:471-481` | Supported WORKFLOWS_ARTIFACTS_DIR override is not set by this launcher, so optional workflow artifacts default to repo Databases/artifacts. Add a private profile path before exercising this feature. |
| ACP runner source cwd | Canonical `live-tier-uat/profile.mjs:148` uses repo tools/tldw-agent; runner config selects Go and a private config-relative HOME | Session workspace roots and ACP databases are private, but the optional runner is not a filesystem sandbox. Its source/build behavior has not been exercised or certified. |

`shared-state-scope-audit.json` records file existence, size, mode, modification time, root/maintenance key names, and the two inactive booleans/counts only; it contains no settings content or credentials. Existing system_ops data and its zero-byte lock predate this recovery. No shared files were changed by this audit. An authenticated request may still touch locking state; inactive baseline does not mean full isolation.

Confirmed overrides cover auth/content/per-user databases; media/evaluation/jobs/ACP/MCP/audit/monitoring/research/moderation/consent/sandbox storage; scheduler DB; workflow file root; codegraph roots; logging; Python/OS temp; HuggingFace/Torch/NeMo/RAG/Flashrank caches; upload/source allowlists. Workflows DB and workflow scheduler DB defaults use DatabasePaths and the configured private USER_DB_BASE_DIR. MCP disk-space health checks read the repository filesystem but do not prove storage escape. Privilege/user-profile catalogs are read-only configuration in the inspected loaders. Optional audio/model download caches were not exhaustively executed. Preserved external model files have not been modified by this audit; actual read-only OS enforcement is not provided by this launcher.

## Historical preflight comparison

Read `followup147-149/cycle5-native-pg-r3-single-preflight.json` and the corrected `cycle5-pg-r3-source-checkpoint.json`. The prior record used the same normal auth/content PostgreSQL modes, registration/BYOK disabled, blank RAG provider/model defaults, zero provider env overrides, Setup-driven provider configuration, and the same config path override set. The checkpoint records both PG initializers exit0, no test-mode keys, and no configured/fallback SQLite users DB. Those historical receipts establish that earlier run only; they do not validate this newly prepared environment or cover the shared paths above.

## Verification and boundaries

- Node syntax checks pass for launcher and holder; Python holder parses.
- Bandit on the private pytest holder reports four LOW B101 findings for its pytest assertions (profile name, absent old receipt/release marker, distinct fixture DBs), and no other findings. Reviewed in context: the holder is launched through pytest assertion rewriting without `-O` or inherited PYTHONOPTIMIZE. These are test assertions, not omitted application authorization checks. Raw result is retained; this is not a zero-findings Bandit claim. Already-running holders were not altered.
- Actual read-only package lookup/import and exact CJS resolution were checked before/after the first correction.
- Exact generated ESM symlink geometry supports the real-build correction; parent runtime verification remains necessary.
- Injected port controls distinguish three independent error codes.
- Root's vision config hash is unchanged by this audit.
- No framework app import, pytest holder execution, database query, model call, server start, profile reset, or tracked file edit by this subagent.
- Earlier preparation artifacts remain historical; the audit manifest hashes the current private harness/report separately.
