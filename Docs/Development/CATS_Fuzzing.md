# CATS Fuzzing

This harness runs local OpenAPI-driven negative fuzzing against `tldw_server` with
CATS. It is intended for contract validation and first-slice runtime coverage, not
for production traffic.

## Safety

Do not run the default harness against production deployments or real provider
credentials. The default runtime starts loopback uvicorn, uses deterministic long
test API keys, isolates SQLite databases under the artifact runtime directory,
generates a dedicated `TLDW_ENV_FILE`, and rejects inherited provider/webhook
credentials unless `--allow-external` is passed. Runtime blocks use CATS blackbox
mode and gate failures on 5xx responses.

## Setup

Activate the shared project environment and confirm CATS is installed:

```bash
source .venv/bin/activate
cats --version
```

Run the contract-only block:

```bash
python -m Helper_Scripts.cats_fuzz --block contract
```

Run the default first-slice blocks explicitly:

```bash
python -m Helper_Scripts.cats_fuzz --block contract --block public-read
```

Or use the Makefile target:

```bash
make cats-fuzz
```

## Blocks

- `contract`: exports and validates `openapi.json` without calling the API.
- `public-read`: starts or uses a local server and fuzzes public read-only
  endpoints.
- `auth-read`: exists in the manifest for authenticated read-only fuzzing, but is
  not part of the default first slice unless explicitly selected.

## Artifacts

Artifacts are written under `artifacts/cats-fuzz/` by default:

- `openapi.json`
- per-block `summary.json`
- per-block stdout/stderr logs
- CATS reports
- local server logs under `server/`

## Failure Interpretation

Block summaries classify results with `failure_class`:

- `usage`: invalid command/options or harness invocation problem
- `tool`: CATS/tooling execution failure
- `api`: API behavior failed the block gate
- `ok`: block exited successfully

A nonzero `public-read` summary means inspect that block's logs and CATS report
before treating the result as an API regression.
