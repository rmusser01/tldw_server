# CATS Fuzzing Setup and Operations Guide

This guide is for contributors who want to validate `tldw_server` API changes
with the local CATS fuzzing harness.

CATS is a REST API fuzzer and negative testing tool for OpenAPI endpoints. The
project harness wraps CATS with local-only safety defaults, OpenAPI export,
isolated runtime configuration, repeatable block definitions, and summary files
that are easier to review in PRs.

Reference docs:

- [CATS introduction](https://endava.github.io/cats/docs/intro/)
- [CATS installation](https://endava.github.io/cats/docs/getting-started/installation/)
- [Running CATS](https://endava.github.io/cats/docs/getting-started/running-cats/)
- [Interpreting CATS results](https://endava.github.io/cats/docs/getting-started/interpreting-results/)
- [CATS exit codes](https://endava.github.io/cats/docs/getting-started/exit-codes/)
- [CATS authentication header masking](https://endava.github.io/cats/docs/getting-started/masking-headers/)

## Scope

Use this harness for:

- Contract validation of the generated OpenAPI document.
- Local blackbox fuzzing of selected read-only API surfaces.
- Smoke-level detection of unexpected 5xx responses.
- Producing artifact bundles for PR review and debugging.

Do not use this first-slice harness as:

- A production scanner.
- A broad authenticated mutation fuzzer.
- A hard CI gate for the full public-read surface.
- A substitute for endpoint-specific integration tests.

## Safety Model

The default harness is local-only by design.

- Starts uvicorn on `127.0.0.1` for runtime blocks unless `--no-start-server`
  is used.
- Uses a deterministic long test API key for single-user auth.
- Writes AuthNZ and user SQLite databases under the artifact runtime directory.
- Generates `runtime/.env` for OpenAPI export and CATS subprocesses.
- Generates `runtime/config.txt` and forces `TLDW_CONFIG_FILE`,
  `TLDW_CONFIG_PATH`, and `TLDW_CONFIG_DIR` to that file/directory so the
  harness does not read a contributor's normal `Config_Files/config.txt`.
- Generates `runtime/cats-server.env` for uvicorn without guarded test-mode
  flags that would make server startup unsafe.
- Scrubs known sensitive values and provider endpoint overrides from the child
  environment.
- Sets `PYTHON_DOTENV_DISABLED=true` in child processes; the generated env file
  remains as an audit artifact, while the subprocess environment carries the
  actual test settings.
- Refuses to run when real provider or webhook credentials are detected unless
  `--allow-external` is passed.
- Rejects non-loopback `--server-url` values for built-in local-only blocks.
- Passes `--maskHeaders X-API-KEY,Authorization` to CATS and masks those header
  values again in `summary.json`.
- Uses CATS blackbox mode for runtime blocks, so the harness gate focuses on 5xx
  responses rather than normal validation mismatches.

Do not point CATS at production or a server with real user data. CATS
intentionally sends malformed, oversized, and adversarial inputs. The built-in
blocks accept existing servers only on `localhost`, `127.0.0.0/8`, or `::1`.

## Prerequisites

From a normal checkout:

```bash
make tooling-install
source .venv/bin/activate
```

From a worktree that does not have its own `.venv`, either create one or
deliberately activate the main checkout environment:

```bash
source /path/to/tldw_server2/.venv/bin/activate
```

Install CATS and verify it is on `PATH`:

```bash
cats --version
```

The local harness has been verified with CATS `13.8.0`. CATS can be installed
with Homebrew on macOS:

```bash
brew tap endava/tap
brew install cats
```

The upstream project also publishes native binaries and an uberjar. The native
binary does not require Java; the uberjar requires Java 17 or newer.

## Quick Start

Run the contract-only block first. It does not start the API server.

```bash
python -m Helper_Scripts.cats_fuzz --block contract
```

Run the default first-slice local validation:

```bash
python -m Helper_Scripts.cats_fuzz --block contract --block public-read
```

Or use the Makefile target:

```bash
make cats-fuzz
```

Artifacts default to `artifacts/cats-fuzz/`. Use a unique output directory when
comparing runs:

```bash
python -m Helper_Scripts.cats_fuzz \
  --block contract \
  --block public-read \
  --output /tmp/tldw-cats-public-read
```

## Recommended PR Validation Workflow

Run these before asking reviewers to trust a fuzzing-related change:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py \
  tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py \
  -q
```

Then run contract validation:

```bash
python -m Helper_Scripts.cats_fuzz --block contract --output /tmp/tldw-cats-contract
```

Check:

```bash
cat /tmp/tldw-cats-contract/contract/summary.json
```

Expected contract result:

- `exit_code` is `0`.
- `failure_class` is `ok`.
- `cats_version` is populated.
- `openapi_sha256` is populated.

For runtime validation, run:

```bash
python -m Helper_Scripts.cats_fuzz \
  --block contract \
  --block public-read \
  --output /tmp/tldw-cats-public-read
```

The current broad `public-read` block is operational but may time out before
finishing all selected paths. A timeout is acceptable for this first slice when
it is captured as a structured tool failure:

- `public-read/summary.json` exists.
- `failure_class` is `tool`.
- `exit_code` is `124`.
- `public-read/stderr.log` says `Command timed out after 300 seconds`.
- CATS reports exist under `public-read/cats-report/`.

Do not hide this result in a PR. Record it as "public-read timed out after
300s; artifacts were written" and include the output directory.

## CLI Reference

Entrypoint:

```bash
python -m Helper_Scripts.cats_fuzz [options]
```

Options:

- `--block {contract,public-read,auth-read}`: Select a block. Repeat the flag to
  run multiple blocks. Defaults to `contract` and `public-read`.
- `--output PATH`: Artifact directory. Defaults to `artifacts/cats-fuzz`.
- `--cats-bin PATH_OR_NAME`: CATS executable. Defaults to `cats`.
- `--server-url URL`: Use an already running loopback server for runtime blocks.
- `--no-start-server`: Do not start uvicorn. Requires `--server-url` for runtime
  blocks.
- `--start-server`: Explicitly start the isolated loopback uvicorn server.
- `--dry-run`: Pass CATS dry-run mode to runtime blocks.
- `--allow-external`: Allow the parent environment to contain real credentials.
  Known sensitive values are still scrubbed from the child env, and harness auth
  is overwritten with the deterministic test key.

Examples:

```bash
# Contract only, no server.
python -m Helper_Scripts.cats_fuzz --block contract

# Isolated local server plus public-read fuzzing.
python -m Helper_Scripts.cats_fuzz --block contract --block public-read

# Public-read against an existing local server.
python -m Helper_Scripts.cats_fuzz \
  --block public-read \
  --no-start-server \
  --server-url http://127.0.0.1:8000

# Use a non-default CATS binary.
python -m Helper_Scripts.cats_fuzz --block contract --cats-bin /usr/local/bin/cats
```

## Built-In Blocks

| Block | Purpose | Server | Scope | Timeout |
| --- | --- | --- | --- | --- |
| `contract` | Export, validate, and summarize `openapi.json`. | No | Full OpenAPI contract. | 60s per CATS validate/stats command |
| `public-read` | Blackbox fuzz public metadata and health endpoints. | Yes | `GET`/`HEAD` on `/`, `/health`, `/ready`, `/health/ready`, `/api/v1/health`, `/api/v1/health/live`, `/api/v1/health/ready`, `/api/v1/config/docs-info`, `/api/v1/config/quickstart`. | 300s |
| `auth-read` | Authenticated read-only smoke fuzzing. | Yes | `GET`/`HEAD` on `/api/v1/llm/providers`, `/api/v1/mcp/status`, `/api/v1/rag/health/simple`. | 180s |

The runtime blocks skip `POST`, `PUT`, `PATCH`, `DELETE`, and `TRACE`. Mutating
blocks are intentionally not included in this first slice.

## Artifact Layout

For `--output /tmp/tldw-cats-run`, expect:

```text
/tmp/tldw-cats-run/
+-- openapi.json
+-- openapi-export.stdout.log
+-- openapi-export.stderr.log
+-- runtime/
|   +-- .env
|   +-- cats-server.env
|   +-- config.txt
|   +-- users.db
|   +-- user_databases/
+-- server/
|   +-- uvicorn.stdout.log
|   +-- uvicorn.stderr.log
+-- contract/
|   +-- summary.json
|   +-- stdout.log
|   +-- stderr.log
+-- public-read/
    +-- summary.json
    +-- stdout.log
    +-- stderr.log
    +-- cats-report/
        +-- junit.xml
        +-- Test*.html
        +-- Test*.json
```

Keep `summary.json`, CATS reports, and relevant logs when filing an issue or PR.
Do not publish `runtime/` artifacts from a run where you used `--allow-external`
or pointed at a manually started server with real data.

## Interpreting Results

Every block writes `summary.json` with:

- `block`: block name.
- `cats_version`: detected CATS version or `unknown`.
- `openapi_sha256`: checksum of the generated OpenAPI file.
- `command` and `masked_command`: the command with auth header values masked.
- `exit_code`: CATS or harness subprocess exit code.
- `failure_class`: normalized result class.
- `stdout_path`, `stderr_path`, `report_dir`: artifact pointers.

Failure classes:

- `ok`: command exited successfully.
- `usage`: invalid CATS options, invalid harness invocation, or command-line
  usage failure.
- `tool`: CATS/tooling/runtime execution issue, including harness timeouts.
- `api`: CATS completed and reported API behavior that failed the block gate.

CATS itself can return invalid-input and unexpected-execution exit codes, or the
number of fuzzing errors found. The harness maps those raw exits into the
classes above so PR reviewers can triage consistently.

Runtime blocks use blackbox mode. In CATS blackbox mode, 2xx and 4xx mismatches
are ignored for the gate; unexpected 5xx responses are the important API signal.

## Triage Playbooks

### Contract Block Fails

1. Open `contract/summary.json`.
2. Check `failure_class`.
3. Read `contract/stderr.log`.
4. If CATS validation reports an OpenAPI problem, inspect `openapi.json` around
   the failing path/schema.
5. Add or adjust focused OpenAPI shape tests before changing endpoint schemas.

### Runtime Block Reports `api`

1. Open the block `summary.json` and confirm `failure_class` is `api`.
2. Open `cats-report/index.html` or the relevant `Test*.html` files.
3. Identify the path, method, fuzzer, request, and response.
4. Check `server/uvicorn.stderr.log` for the matching stack trace or 5xx log.
5. Reproduce with the `curl` command in the CATS test detail, after replacing
   masked header environment variables with test credentials only.
6. Fix the endpoint or contract mismatch, then rerun the same block.

### Runtime Block Reports `tool`

1. Read the block `stderr.log`.
2. If it says `Command timed out after 300 seconds`, inspect `stdout.log` to see
   which path/fuzzer was active when time expired.
3. Check whether CATS produced partial `cats-report/` files.
4. Split the block or narrow paths/fuzzers in a follow-up task if the timeout is
   due to broad coverage rather than a server hang.
5. If CATS reports an internal exception, capture the CATS version, command, and
   the smallest reproducing OpenAPI path.

### Server Fails To Start

1. Read `server/uvicorn.stderr.log`.
2. Confirm the harness wrote `runtime/cats-server.env`.
3. Confirm no real provider credentials are inherited unless
   `--allow-external` was intentional.
4. Confirm local port binding is allowed in the execution environment.

### Sensitive Credential Detection Blocks A Run

By default, the harness refuses to build its child environment when real provider
or webhook credentials are present. Prefer unsetting those values in the shell.
Use `--allow-external` only when you understand the local risk. The harness still
scrubs known sensitive child variables and overwrites its auth keys.

## Existing Server Mode

Use existing-server mode only for local throwaway servers:

```bash
python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000
python -m Helper_Scripts.cats_fuzz \
  --block public-read \
  --no-start-server \
  --server-url http://127.0.0.1:8000 \
  --output /tmp/tldw-cats-existing-server
```

The harness cannot isolate databases, env files, or credentials for a server you
started yourself. Existing-server mode is still restricted to loopback hosts for
built-in local-only blocks. Do not use this mode with production-like data.

## Cleanup

Artifact directories can be large because CATS writes individual HTML and JSON
test files. After preserving anything needed for review, remove old local output
directories:

```bash
rm -rf artifacts/cats-fuzz /tmp/tldw-cats-*
```

Do not remove artifacts referenced by an active issue, PR, or Backlog task.

## Contributor Checklist

Before posting CATS validation results in a PR:

- Record the exact command.
- Record the CATS version.
- Record the output directory.
- Include each block's `exit_code` and `failure_class`.
- Include whether `public-read` completed or timed out.
- Attach or point to relevant `summary.json`, `stderr.log`, and CATS report
  files when a block is nonzero.
- Confirm no real secrets appear in copied command lines, logs, summaries, or
  reports.
- Run `git diff --check` after editing docs or harness files.

## Known Limitations

- `public-read` currently covers enough of the generated OpenAPI surface that it
  may time out after 300s while still writing useful partial reports.
- `auth-read` is scaffolded and selectable, but it is not the default block.
- CATS `--dry-run` has reached command parsing locally but has shown tool-level
  instability against the large generated tldw OpenAPI; treat it as diagnostic,
  not a required preflight.
- This harness is not wired as a hard CI gate yet. Start with manual local runs
  and make later CI jobs narrower than the broad local exploration block.
