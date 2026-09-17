# Independent native review — UAT251 MCP setup

**CLEAR for bounded native Save packs, sample tool and persisted-state acceptance.** Source/runtime checks and verification are recorded in `audit.json`.

Task TASK-13260.193. Fresh run `mcp251-fresh-targeted-20260917`, PostgreSQL single-user ports 18704/18784, source `a7d3155a567afb25982eb360ea24b973cc3249c9`. The existing completed targeted profiles were preserved; a separate fresh profile is appropriate because their completed setup wizard cannot be reopened normally. No setup reset is claimed or required.

## Native result

- At **22:11:55 UTC, 2026-09-17**, normal **Save packs** submits `/api/v1/setup/first-run/mcp-tools/apply`. The observed response is **200, applied**, profile **1**, assignment **1**, with the five default packs (Research, Learning, Writing, Media Library, Personal Knowledge), **23 unique tools**, no addons and no conflict.
- At **22:13:05**, normal **Run sample tool** submits the validation endpoint. It returns **200, validated, built_in_passed** for **mcp.tools.list**, retaining the same profile, assignment, packs and tool count. External status remains **not_enabled**.
- At **22:14:10**, normal reload reads first-run state with **200**. The persisted `mcp_tools` data retains the same IDs, packs, 23-tool count, validation timestamp, validation run ID, confirmation version and acknowledgement. `completed_steps` includes `mcp_tools`, and the UI advances to **First chat**.

The apply/validate captures occur immediately after the network response and still show busy buttons. They are not treated as settled UI success by themselves. The later persisted state and advancement to First chat close that observation gap. The first-chat step remains uncompleted; overall setup is still `in_progress`.

## Normal fresh setup and scope

The native sequence starts at Choose your setup path, selects the local path, acknowledges privacy, configures the actual local llama.cpp endpoint `http://127.0.0.1:9099/v1`, and records provider validation ready and saved. Balanced ingest defaults are selected; optional audio is skipped and advanced options deferred. The five default MCP packs are checked before Save packs. Endpoint correction and immediate still-loading captures remain retained rather than implying every capture is settled.

The MCP sample validates built-in tool inventory, not external addons or a model completion. Optional audio, first-chat generation, whole-wizard completion and the full 48-cell UAT matrix are outside this verdict.

## Provenance and isolation

The gate releases only issue-specific targeted acceptance, prepares both source copies, and permits only the single-user runtime. Copy completion, source manifests, dependency receipts, preflight and fresh profile bindings are checked. The MCP repository file matches independently reviewed SHA256 `a63c6cf5c51ffa1a702894740240a4767e27f677ddf8512ed1131ad1edd02fda` in workspace, retained review snapshot and fresh runtime source.

The new PostgreSQL receipt uses official `pg_temp_db` and `pg_temp_db_session` fixtures with distinct auth/content databases and a direct-login runtime role. It records no superuser, BYPASSRLS, inheritance, database/role creation, replication or memberships. The programmatic audit checks actual application process/working-directory/source/environment bindings and listeners without emitting credentials or raw environment/logs. Recorded holder/API/Next parent PIDs are **73403 / 74305 / 74329**.

Original completed profile, initialization and fixture-holder record hashes are compared with the earlier independent preservation audit; the new profile and database identities are distinct. This checks those immutable bindings, not entire database contents. JavaScript dependencies and the isolated Python environment are reused, so this is a fresh application profile rather than a clean OS install.

## Evidence and verification

The three authored review files are `REVIEW.md`, `audit.mjs`, and `audit.json`. The audit hashes every selected input, including private records and logs that it reads only programmatically. Its output contains safe identity fields, flags, hashes and comparisons. It also streams the archive hash and checks selected copied source files; it does not independently rehash every file in both source trees.

Run:

```sh
node /Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-repairs-231-246/native-mcp251-review/audit.mjs
```

Process inspection requires read-only access to `ps`/`lsof`. The initial sandbox denied it; the approved safe inspection uses no browser, database or runtime mutation. Two initial audit-schema assumptions were corrected: the initialization receipt says `completed`, and the official holder binds its source/run through environment fields rather than command arguments. Neither was an application failure. Runtime-log hashes describe bytes read at audit time; those live files may append afterward.

No source, Git, task or tracker edits were made. Prior implementation test/security review remains retained context; product tests were not rerun for this native review.

Final checks: **38/38 passed**. The audit binds **73 inputs** (71 stable files plus two live log snapshots); separate hashing found no changed inputs. Node syntax validation passed. Scoped Bandit was run from the project environment on the auditor MJS and reported one JavaScript AST parse error, providing no JavaScript security coverage.
