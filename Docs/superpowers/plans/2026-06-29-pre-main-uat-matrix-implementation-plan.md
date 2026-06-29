# Pre-Main UAT Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved pre-main UAT matrix for PR #1982, fix still-valid release-blocking findings, and produce committed evidence for the main-merge decision.

**Architecture:** Treat UAT as a bounded release gate, not a broad route audit. Run the same persona journeys against isolated local and Docker single-user WebUI environments, prove OpenAI and `llama.cpp` live answer paths, then fix only current-code issues verified by evidence.

**Tech Stack:** FastAPI, Next.js WebUI, Playwright/Bun harnesses, Docker Compose, Backlog.md, pytest, Vitest, Bandit, GitHub PR #1982.

---

## Source Documents

- UAT design: `Docs/superpowers/specs/2026-06-29-pre-main-uat-matrix-design.md`
- Plan task: `TASK-12063`
- PR under test: `rmusser01/tldw_server#1982`
- Frontend app guide: `apps/tldw-frontend/AGENTS.md`
- Root project guide: `AGENTS.md`
- Chat API docs: `Docs/API-related/Chat_API_Documentation.md`
- `llama.cpp` integration docs: `Docs/API-related/llamacpp_integration_modes.md`

## Scope

This plan covers UAT execution and the fix loop for findings discovered during the UAT. It does not merge PR #1982, push to `main`, cancel CI jobs, or replace live provider gates with mocks.

Every repository edit made during execution requires an associated Backlog.md task before the edit begins. Create one task for the UAT evidence/report work and one task per independent code or docs fix if findings require patches.

## File Structure

### Files To Create During Execution

- `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/README.md`
  - Curated final UAT report with matrix status, provider status, issue table, verification commands, links to raw artifacts, and cleanup notes.
- `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
  - Local environment setup, provider preflight, persona journey results, and findings.
- `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
  - Docker environment setup, provider preflight, persona journey results, and findings.
- `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/provider-preflight.md`
  - Exact OpenAI model, exact `llama.cpp` model id, direct provider checks, backend `/chat/completions` checks, and any provider errors.
- `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
  - Finding table with severity, reproduction, status, fix commit, verification, skipped reason, or deferral approval requirement.
- `/tmp/tldw-pre-main-uat/<run-id>/`
  - Raw screenshots, traces, console logs, backend logs, HAR/network captures when available, disposable source files, and copied command output. This raw directory is not committed by default.

### Existing Harnesses To Reuse

- `apps/tldw-frontend/scripts/onboarding-uat/run.mjs`
- `apps/tldw-frontend/scripts/chat-uat-driver.mjs`
- `apps/tldw-frontend/scripts/media-uat-driver.mjs`
- `apps/tldw-frontend/scripts/chars-uat-driver.mjs`
- `apps/tldw-frontend/scripts/media-multi-uat-driver.mjs`
- `apps/tldw-frontend/scripts/research-workspace-uat-runner.mjs`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Targeted specs under `apps/tldw-frontend/e2e/workflows/` and `apps/tldw-frontend/e2e/smoke/`

### Files That May Be Modified Only If Findings Require Fixes

No product-code file is preselected for modification. When a finding is verified:

- Create or update a Backlog.md task for that finding.
- Identify the smallest owned source file(s) from current code.
- Write or update focused tests before patching when the behavior is testable.
- Modify only files necessary for the verified issue.
- Record the touched paths, test output, and Bandit result in the task and evidence report.

## Provider And Answer Path Contract

- OpenAI model: use `gpt-4o-mini` for all required OpenAI preflight and answer-path checks. If the configured server rejects this model or OpenAI credentials are missing, stop and record a blocking provider/config failure rather than substituting another model silently.
- `llama.cpp` model: query `http://127.0.0.1:9099/v1/models` before UAT, record the exact returned model id in `provider-preflight.md`, and use that exact id for direct and app-mediated `llama.cpp` checks. If the endpoint is unreachable or no model id is available, stop and record a blocking provider/config failure.
- Local `llama.cpp` backend endpoint: configure the local backend provider plane with `http://127.0.0.1:9099/v1/chat/completions`.
- Docker `llama.cpp` backend endpoint on macOS: configure the container-visible backend provider plane with `http://host.docker.internal:9099/v1/chat/completions`.
- Core direct answer path: call `POST /api/v1/chat/completions` through the tldw backend with `stream: false`, `temperature: 0`, a small token limit, and a prompt that requires a short response containing the UAT run token.
- Basic document answer path: ingest a disposable document containing the UAT run token, ask about it through the normal WebUI knowledge/search path, and verify the answer cites or references the disposable content.
- Character answer path: create or import a disposable roleplay character, send a roleplay message through character chat, and verify each provider produces a usable response at least once per environment.

## Runtime Isolation Contract

- Use one run id for every environment, fixture, title, tag, note, character, chat, screenshot, and report entry.
- Use `SINGLE_USER_API_KEY=$UAT_API_KEY` for both backend and WebUI in this run.
- For local execution, do not edit `tldw_Server_API/Config_Files/.env` or `tldw_Server_API/Config_Files/config.txt`. Copy `config.txt` into `/tmp/tldw-pre-main-uat/<run-id>/local-runtime/Config_Files/config.txt`, write a run-scoped `.env` beside it, and start the backend with `TLDW_CONFIG_FILE` and `TLDW_ENV_FILE` pointing at those temp files.
- For Docker execution, do not use the default compose project or any existing `tldw_*` volumes. Use `--project-name "$UAT_COMPOSE_PROJECT"` and a UAT compose overlay that mounts run-scoped config, env, and fixtures into the containers.
- Pass the API key to WebUI with `NEXT_PUBLIC_X_API_KEY=$UAT_API_KEY` and to Playwright/harnesses with `TLDW_E2E_API_KEY=$UAT_API_KEY`, `TLDW_API_KEY=$UAT_API_KEY`, and `SINGLE_USER_API_KEY=$UAT_API_KEY`.
- Start local API and WebUI as tracked background processes with PID files and log files under `$LOCAL_PROFILE_ROOT`; never rely on an untracked foreground terminal.
- Persist every run variable needed across tasks in `$RAW_ROOT/uat.env`; task-by-task agents must source that file before running commands after Task 1.
- Treat any fallback to the checked-in `.env`, checked-in `config.txt`, or non-UAT Docker volume as a failed isolation preflight.

## Run State Contract

Task 1 creates `/tmp/tldw-pre-main-uat/<run-id>/uat.env`. At the start of every later task, run:

```bash
source /tmp/tldw-pre-main-uat/<run-id>/uat.env
```

Use the exact `<run-id>` created in Task 1. When a later task creates durable values such as `LLAMA_CPP_MODEL`, `LOCAL_CONFIG_FILE`, `LOCAL_ENV_FILE`, `DOCKER_ENV_FILE`, or `DOCKER_COMPOSE_OVERLAY`, append shell-safe `export` lines to the same `uat.env` file before moving to the next task.

## Task 1: Create UAT Execution Tracking

**Files:**
- Create: `backlog/tasks/<new-task> - Execute-pre-main-UAT-matrix-for-PR-1982.md`
- Create: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/README.md`
- Create: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/provider-preflight.md`
- Create: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
- Create: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
- Create: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`

- [ ] **Step 1: Create the Backlog task**

Run: `backlog task create "Execute pre-main UAT matrix for PR 1982" --label uat --label release --label pr-1982`

Expected: A new task id is printed. Export that exact id as `UAT_TASK_ID` before Step 2 and record it in every UAT evidence file.

- [ ] **Step 2: Create the run id**

Run:

```bash
export RUN_ID="pre-main-uat-$(date -u +%Y%m%d%H%M%S)"
export RAW_ROOT="/tmp/tldw-pre-main-uat/${RUN_ID}"
export EVIDENCE_ROOT="Docs/Product/WebUI/evidence/pre_main_uat/${RUN_ID}"
export UAT_API_KEY="THIS-IS-A-SECURE-KEY-123-UAT-${RUN_ID}"
export LOCAL_PROFILE_ROOT="${RAW_ROOT}/local-runtime"
export DOCKER_PROFILE_ROOT="${RAW_ROOT}/docker-runtime"
export UAT_COMPOSE_PROJECT="${RUN_ID}"
mkdir -p "$RAW_ROOT" "$EVIDENCE_ROOT"
cat > "${RAW_ROOT}/uat.env" <<EOF
export RUN_ID="${RUN_ID}"
export RAW_ROOT="${RAW_ROOT}"
export EVIDENCE_ROOT="${EVIDENCE_ROOT}"
export UAT_API_KEY="${UAT_API_KEY}"
export LOCAL_PROFILE_ROOT="${LOCAL_PROFILE_ROOT}"
export DOCKER_PROFILE_ROOT="${DOCKER_PROFILE_ROOT}"
export UAT_COMPOSE_PROJECT="${UAT_COMPOSE_PROJECT}"
export UAT_TASK_ID="${UAT_TASK_ID}"
EOF
```

Expected: `$RAW_ROOT`, `$EVIDENCE_ROOT`, and `$RAW_ROOT/uat.env` exist and are empty except for files created in later steps.

- [ ] **Step 3: Create evidence Markdown templates**

Run:

```bash
source /tmp/tldw-pre-main-uat/<run-id>/uat.env
for name in README provider-preflight local-single-user docker-single-user findings; do
  file="${EVIDENCE_ROOT}/${name}.md"
  title="$(printf '%s' "$name" | tr '-' ' ')"
  printf '# %s\n\nRun id: `%s`\n\nTask: `%s`\n\nStatus: Not Started\n\n## Notes\n\n' "$title" "$RUN_ID" "${UAT_TASK_ID}" > "$file"
done
```

Expected: All five evidence Markdown files exist before the initial evidence-shell commit.

- [ ] **Step 4: Record source-control state**

Run: `git status --short --branch`

Expected: The branch is `codex/pr1982-ci-fanout-fixes`; any unrelated untracked files are listed in the report and excluded from UAT commits.

- [ ] **Step 5: Record commit and PR state**

Run:

```bash
git rev-parse HEAD
gh pr view 1982 --repo rmusser01/tldw_server --json number,headRefName,baseRefName,mergeStateStatus,headRefOid,url
```

Expected: `headRefName` is `dev`, `baseRefName` is `main`, and the commit SHA is recorded in `README.md`.

- [ ] **Step 6: Seed disposable test content**

Create small files under `$RAW_ROOT/fixtures/`:

- `basic-user-source.md`: title, body, and tag include `$RUN_ID`.
- `advanced-source-alpha.md`: contains a unique fact and tag `uat-alpha-$RUN_ID`.
- `advanced-source-beta.md`: contains a different unique fact and tag `uat-beta-$RUN_ID`.
- `roleplay-character.json`: SillyTavern-compatible disposable character using `$RUN_ID` in name and description.

Expected: Fixture text contains only disposable UAT content and no user-local data.

- [ ] **Step 7: Commit the initial evidence shell**

Run:

```bash
git add Docs/Product/WebUI/evidence/pre_main_uat/${RUN_ID}
git commit -m "docs: add pre-main UAT evidence shell"
```

Expected: Commit succeeds and includes only the evidence shell, not raw `/tmp` artifacts.

## Task 2: Preflight Live Providers Outside The App

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/provider-preflight.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/provider/`

- [ ] **Step 1: Check OpenAI credential presence without printing the secret**

Run: `test -n "$OPENAI_API_KEY" && printf 'OPENAI_API_KEY present\n'`

Expected: Prints `OPENAI_API_KEY present`. If it does not, record a blocking OpenAI configuration failure.

- [ ] **Step 2: Check direct OpenAI chat completion**

Run a direct OpenAI API request using `gpt-4o-mini`, `temperature: 0`, and a prompt that asks for the exact token `ok-$RUN_ID`.

```bash
curl -sf https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gpt-4o-mini\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ok-${RUN_ID}\"}],\"temperature\":0,\"max_tokens\":32}"
```

Expected: HTTP success and a short response containing `ok-$RUN_ID`. Record status code, model, and redacted response excerpt.

- [ ] **Step 3: Check direct `llama.cpp` models endpoint**

Run:

```bash
source .venv/bin/activate
source /tmp/tldw-pre-main-uat/<run-id>/uat.env
export LLAMA_MODELS_JSON="$(curl -sf http://127.0.0.1:9099/v1/models)"
export LLAMA_CPP_MODEL="$(printf '%s' "$LLAMA_MODELS_JSON" | python -c 'import json, sys; data=json.load(sys.stdin); models=data.get("data") or []; print(models[0].get("id", "") if models else "")')"
test -n "$LLAMA_CPP_MODEL"
printf 'llama.cpp model: %s\n' "$LLAMA_CPP_MODEL"
printf 'export LLAMA_CPP_MODEL=%q\n' "$LLAMA_CPP_MODEL" >> "${RAW_ROOT}/uat.env"
```

Expected: JSON contains at least one model id, `LLAMA_CPP_MODEL` is non-empty, and the exact selected id is recorded for the rest of UAT.

- [ ] **Step 4: Check direct `llama.cpp` chat completion**

Run a direct OpenAI-compatible request to `http://127.0.0.1:9099/v1/chat/completions` using the selected model id, `temperature: 0`, and a prompt that asks for the exact token `ok-$RUN_ID`.

```bash
curl -sf http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${LLAMA_CPP_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ok-${RUN_ID}\"}],\"temperature\":0,\"max_tokens\":32}"
```

Expected: HTTP success and a short usable response. If the model does not echo the exact token but produces a valid answer, record the deviation and continue only if app-mediated answer checks can still prove live provider functionality.

- [ ] **Step 5: Store provider preflight evidence**

Update `provider-preflight.md` with provider URL, selected model ids, timestamp, result, and any redacted error output.

Expected: The evidence file has enough detail for another engineer to reproduce the provider checks without exposing secrets.

## Task 3: Configure And Start Local Single-User WebUI

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/local/`

- [ ] **Step 1: Prepare isolated local runtime paths**

Create a run-scoped local runtime directory under `/tmp/tldw-pre-main-uat/<run-id>/local-runtime/` and configure database/config paths to avoid existing user data.

Run:

```bash
source /tmp/tldw-pre-main-uat/<run-id>/uat.env
export LOCAL_CONFIG_DIR="${LOCAL_PROFILE_ROOT}/Config_Files"
export LOCAL_DB_DIR="${LOCAL_PROFILE_ROOT}/Databases"
export LOCAL_CONFIG_FILE="${LOCAL_CONFIG_DIR}/config.txt"
export LOCAL_ENV_FILE="${LOCAL_CONFIG_DIR}/.env"
mkdir -p "${LOCAL_CONFIG_DIR}" "${LOCAL_DB_DIR}/user_databases" "${LOCAL_PROFILE_ROOT}/logs"
{
  printf 'export LOCAL_CONFIG_DIR=%q\n' "$LOCAL_CONFIG_DIR"
  printf 'export LOCAL_DB_DIR=%q\n' "$LOCAL_DB_DIR"
  printf 'export LOCAL_CONFIG_FILE=%q\n' "$LOCAL_CONFIG_FILE"
  printf 'export LOCAL_ENV_FILE=%q\n' "$LOCAL_ENV_FILE"
} >> "${RAW_ROOT}/uat.env"
cp tldw_Server_API/Config_Files/config.txt "${LOCAL_CONFIG_FILE}"
cat > "${LOCAL_ENV_FILE}" <<EOF
AUTH_MODE=single_user
SINGLE_USER_API_KEY=${UAT_API_KEY}
DATABASE_URL=sqlite:///${LOCAL_DB_DIR}/users.db
USER_DB_BASE_DIR=${LOCAL_DB_DIR}/user_databases
USER_DB_BASE_DIR_ALLOWED_ROOTS=${LOCAL_DB_DIR}
TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=${LOCAL_DB_DIR}
INGESTION_SOURCE_ALLOWED_ROOTS=${RAW_ROOT}/fixtures
TLDW_INGESTION_SOURCE_ALLOWED_ROOTS=${RAW_ROOT}/fixtures
OPENAI_API_KEY=${OPENAI_API_KEY}
DEFAULT_LLM_PROVIDER=openai
TLDW_SETUP_ALLOW_REMOTE=false
WORKFLOWS_EGRESS_BLOCK_PRIVATE=false
WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443,9099
EOF
```

Expected: Evidence records exact runtime paths and confirms no existing user data path is used.

- [ ] **Step 2: Configure local `llama.cpp` provider endpoint**

Set `Local-API.llama_api_IP` for the isolated local backend config to `http://127.0.0.1:9099/v1/chat/completions` and set the selected `llama.cpp` model id.

Run:

```bash
export LOCAL_LLAMA_CHAT_URL="http://127.0.0.1:9099/v1/chat/completions"
source .venv/bin/activate
python - <<'PY'
import configparser
import os
from pathlib import Path

config_path = Path(os.environ["LOCAL_CONFIG_FILE"])
parser = configparser.ConfigParser()
parser.read(config_path)
updates = {
    "Setup": {
        "enable_first_time_setup": "true",
        "setup_completed": "false",
    },
    "AuthNZ": {
        "auth_mode": "single_user",
        "single_user_api_key": os.environ["UAT_API_KEY"],
    },
    "Local-API": {
        "llama_api_IP": os.environ["LOCAL_LLAMA_CHAT_URL"],
        "llama_model": os.environ["LLAMA_CPP_MODEL"],
    },
}
for section, values in updates.items():
    if not parser.has_section(section):
        parser.add_section(section)
    for key, value in values.items():
        parser.set(section, key, value)
with config_path.open("w", encoding="utf-8") as handle:
    parser.write(handle)
PY
```

Expected: The backend provider plane resolves `llama.cpp` to the host-local endpoint, not the default `8080/completion` endpoint.

- [ ] **Step 3: Start the local API**

Run:

```bash
source .venv/bin/activate
mkdir -p "${LOCAL_PROFILE_ROOT}/logs"
env \
  TLDW_CONFIG_FILE="${LOCAL_CONFIG_FILE}" \
  TLDW_ENV_FILE="${LOCAL_ENV_FILE}" \
  DATABASE_URL="sqlite:///${LOCAL_DB_DIR}/users.db" \
  USER_DB_BASE_DIR="${LOCAL_DB_DIR}/user_databases" \
  AUTH_MODE=single_user \
  SINGLE_USER_API_KEY="${UAT_API_KEY}" \
  OPENAI_API_KEY="${OPENAI_API_KEY}" \
  python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000 \
  > "${LOCAL_PROFILE_ROOT}/logs/api.log" 2>&1 &
echo $! > "${LOCAL_PROFILE_ROOT}/api.pid"
```

Expected: `${LOCAL_PROFILE_ROOT}/api.pid` contains a running process id, API logs are written to `${LOCAL_PROFILE_ROOT}/logs/api.log`, and API is reachable at `http://127.0.0.1:8000`.

- [ ] **Step 4: Start the local WebUI**

Run from `apps/tldw-frontend`:

```bash
mkdir -p "${LOCAL_PROFILE_ROOT}/logs"
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_API_VERSION=v1 \
NEXT_PUBLIC_X_API_KEY="${UAT_API_KEY}" \
TLDW_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_WEB_URL=http://127.0.0.1:8080 \
TLDW_API_KEY="${UAT_API_KEY}" \
TLDW_E2E_API_KEY="${UAT_API_KEY}" \
bun run dev -- -p 8080 \
  > "${LOCAL_PROFILE_ROOT}/logs/webui.log" 2>&1 &
echo $! > "${LOCAL_PROFILE_ROOT}/webui.pid"
```

Expected: `${LOCAL_PROFILE_ROOT}/webui.pid` contains a running process id, WebUI logs are written to `${LOCAL_PROFILE_ROOT}/logs/webui.log`, and WebUI is reachable at `http://127.0.0.1:8080`.

- [ ] **Step 5: Verify local health**

Run health/config checks against the API and load the WebUI in a browser.

Expected: The WebUI reaches the API, auth mode is single-user, and console has no startup-blocking errors.

## Task 4: Run Local Backend Provider Gates

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/provider-preflight.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/local/provider/`

- [ ] **Step 1: Call backend OpenAI chat path**

Call `POST http://127.0.0.1:8000/api/v1/chat/completions` with `X-API-KEY: $UAT_API_KEY`, `api_provider: "openai"`, `model: "gpt-4o-mini"`, `stream: false`, and a prompt requiring `backend-openai-$RUN_ID`.

```bash
curl -sf http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: ${UAT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"api_provider\":\"openai\",\"model\":\"gpt-4o-mini\",\"stream\":false,\"temperature\":0,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly backend-openai-${RUN_ID}\"}]}"
```

Expected: HTTP success and response content containing or clearly answering the requested run token.

- [ ] **Step 2: Call backend `llama.cpp` chat path**

Call `POST http://127.0.0.1:8000/api/v1/chat/completions` with `X-API-KEY: $UAT_API_KEY`, `api_provider: "llama.cpp"`, the selected model id, `stream: false`, and a prompt requiring `backend-llama-$RUN_ID`.

```bash
curl -sf http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: ${UAT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"api_provider\":\"llama.cpp\",\"model\":\"${LLAMA_CPP_MODEL}\",\"stream\":false,\"temperature\":0,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly backend-llama-${RUN_ID}\"}]}"
```

Expected: HTTP success and a usable non-empty response from the local `llama.cpp` service.

- [ ] **Step 3: Record local provider gate result**

Update `provider-preflight.md` and `local-single-user.md` with request shape, model ids, response status, and redacted response excerpts.

Expected: Both providers are marked pass or a blocking provider failure is documented.

## Task 5: Run Local Basic User Journey

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/local/basic/`

- [ ] **Step 1: Run onboarding or first-entry harness**

Run the existing onboarding UAT driver or equivalent Playwright flow against `http://127.0.0.1:8080`.

Expected: A clean disposable user can reach a usable first-value path without stale local data.

- [ ] **Step 2: Ingest the basic disposable document**

Use the WebUI normal document/media ingestion path with `basic-user-source.md`.

Expected: The document appears in media/search surfaces with the `$RUN_ID` title or tag.

- [ ] **Step 3: Ask about the ingested document with OpenAI**

Use the normal WebUI knowledge/search answer path and ask a short question whose answer is present only in `basic-user-source.md`.

Expected: OpenAI answer references the disposable fact or citation/source evidence from the ingested document.

- [ ] **Step 4: Ask about the ingested document with `llama.cpp`**

Switch the WebUI model/provider to `llama.cpp` and ask the same or equivalent question.

Expected: `llama.cpp` produces a usable answer or clearly exposes a provider-specific limitation; provider failure is blocking unless explicitly approved.

- [ ] **Step 5: Create or import the roleplay character**

Use the character UI to create or import `roleplay-character.json`.

Expected: The character appears with name, role/persona metadata, and no non-UAT data leakage.

- [ ] **Step 6: Start roleplay character chat with OpenAI**

Send a roleplay message through character chat using OpenAI.

Expected: The UI sends the message, shows a live provider response, preserves character context, and persists after navigation or reload.

- [ ] **Step 7: Start roleplay character chat with `llama.cpp`**

Switch the character chat provider to `llama.cpp` and send a roleplay message.

Expected: The UI sends the message, shows a live local provider response, and handles any slower local latency without losing state.

- [ ] **Step 8: Check basic mobile critical screens**

Run Playwright/mobile viewport checks for first entry, document answer evidence, character creation/import, character selection, and character conversation.

Expected: Critical controls remain reachable and no text/control overlap blocks completion.

- [ ] **Step 9: Record findings**

Add every observed defect to `findings.md` with severity, reproduction steps, screenshot/log references, and status `Open`.

Expected: The evidence records pass/fail for every basic journey gate.

## Task 6: Run Local Advanced Knowledge Journey

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/local/advanced/`

- [ ] **Step 1: Ingest the advanced disposable dataset**

Use the WebUI or existing media UAT driver to ingest `advanced-source-alpha.md` and `advanced-source-beta.md`.

Expected: Both sources appear with distinct titles, tags, and searchable facts.

- [ ] **Step 2: Exercise media search and advanced filters**

Search for the alpha-only and beta-only facts, then use tags or source filters to narrow results.

Expected: Results respect filters and do not show unrelated existing data.

- [ ] **Step 3: Ask constrained Knowledge QA questions**

Ask targeted questions with constrained sources and verify source/evidence controls.

Expected: Answers cite or otherwise expose the selected source evidence coherently.

- [ ] **Step 4: Exercise downstream output handoff**

Use note handoff, export, review, or the closest available downstream workflow for the generated result.

Expected: The generated result can be saved or reused and is identifiable by `$RUN_ID`.

- [ ] **Step 5: Exercise destructive operations on UAT-created data**

Delete, trash/restore, or remove only objects created for this run.

Expected: Cleanup controls affect only `$RUN_ID` data and any restore path works as documented by the UI.

- [ ] **Step 6: Check keyboard and mobile critical controls**

Verify keyboard flow and mobile layout for search filters, Knowledge QA controls, evidence/source controls, and output handoff.

Expected: Controls are reachable and usable without pointer-only blockers.

- [ ] **Step 7: Record findings**

Update `findings.md` and `local-single-user.md`.

Expected: The local advanced journey has a clear pass/fail status and evidence references.

## Task 7: Stop Local Stack And Start Docker Single-User WebUI

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/docker/`

- [ ] **Step 1: Stop local services**

Stop exactly the local API and WebUI processes started in Task 3.

Run:

```bash
for pid_file in "${LOCAL_PROFILE_ROOT}/webui.pid" "${LOCAL_PROFILE_ROOT}/api.pid"; do
  if test -f "$pid_file"; then
    pid="$(cat "$pid_file")"
    if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      sleep 2
      kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" || true
    fi
  fi
done
lsof -nP -iTCP:8000 -sTCP:LISTEN || true
lsof -nP -iTCP:8080 -sTCP:LISTEN || true
```

Expected: Ports `8000` and `8080` are free before Docker starts.

- [ ] **Step 2: Prepare isolated Docker runtime**

Create run-scoped Docker env/config/volume names for this UAT. Use an overlay file because `docker-compose.single-user.yml` only passes a narrow set of environment variables into the app container.

Run:

```bash
source /tmp/tldw-pre-main-uat/<run-id>/uat.env
export DOCKER_CONFIG_DIR="${DOCKER_PROFILE_ROOT}/Config_Files"
export DOCKER_ENV_FILE="${DOCKER_CONFIG_DIR}/.env"
export DOCKER_CONFIG_FILE="${DOCKER_CONFIG_DIR}/config.txt"
export DOCKER_COMPOSE_OVERLAY="${DOCKER_PROFILE_ROOT}/docker-compose.uat.yml"
mkdir -p "${DOCKER_CONFIG_DIR}" "${DOCKER_PROFILE_ROOT}/logs"
{
  printf 'export DOCKER_CONFIG_DIR=%q\n' "$DOCKER_CONFIG_DIR"
  printf 'export DOCKER_ENV_FILE=%q\n' "$DOCKER_ENV_FILE"
  printf 'export DOCKER_CONFIG_FILE=%q\n' "$DOCKER_CONFIG_FILE"
  printf 'export DOCKER_COMPOSE_OVERLAY=%q\n' "$DOCKER_COMPOSE_OVERLAY"
} >> "${RAW_ROOT}/uat.env"
cp tldw_Server_API/Config_Files/config.txt "${DOCKER_CONFIG_FILE}"
cat > "${DOCKER_ENV_FILE}" <<EOF
AUTH_MODE=single_user
SINGLE_USER_API_KEY=${UAT_API_KEY}
DATABASE_URL=sqlite:///./Databases/users.db
USER_DB_BASE_DIR=/app/Databases/user_databases
USER_DB_BASE_DIR_ALLOWED_ROOTS=/app/Databases
TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=/app/Databases
INGESTION_SOURCE_ALLOWED_ROOTS=/app/uat-fixtures
TLDW_INGESTION_SOURCE_ALLOWED_ROOTS=/app/uat-fixtures
OPENAI_API_KEY=${OPENAI_API_KEY}
DEFAULT_LLM_PROVIDER=openai
NEXT_PUBLIC_X_API_KEY=${UAT_API_KEY}
NEXT_PUBLIC_API_VERSION=v1
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart
TLDW_INTERNAL_API_ORIGIN=http://app:8000
TLDW_SETUP_ALLOW_REMOTE=1
WORKFLOWS_EGRESS_BLOCK_PRIVATE=false
WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443,9099
EOF
cat > "${DOCKER_COMPOSE_OVERLAY}" <<EOF
services:
  app:
    env_file:
      - ${DOCKER_ENV_FILE}
    environment:
      - TLDW_CONFIG_FILE=/app/uat-config/config.txt
      - TLDW_ENV_FILE=/app/uat-config/.env
      - USER_DB_BASE_DIR=/app/Databases/user_databases
      - USER_DB_BASE_DIR_ALLOWED_ROOTS=/app/Databases
      - TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=/app/Databases
      - INGESTION_SOURCE_ALLOWED_ROOTS=/app/uat-fixtures
      - TLDW_INGESTION_SOURCE_ALLOWED_ROOTS=/app/uat-fixtures
      - TLDW_SETUP_ALLOW_REMOTE=1
    volumes:
      - ${DOCKER_CONFIG_DIR}:/app/uat-config:ro
      - ${RAW_ROOT}/fixtures:/app/uat-fixtures:ro
  webui:
    build:
      args:
        NEXT_PUBLIC_X_API_KEY: ${UAT_API_KEY}
        NEXT_PUBLIC_API_VERSION: v1
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: quickstart
        TLDW_INTERNAL_API_ORIGIN: http://app:8000
    environment:
      - NEXT_PUBLIC_X_API_KEY=${UAT_API_KEY}
      - NEXT_PUBLIC_API_VERSION=v1
      - NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart
      - TLDW_INTERNAL_API_ORIGIN=http://app:8000
EOF
```

Expected: Evidence records the compose files, env file, volume names, and confirms existing user volumes are not used.

- [ ] **Step 3: Configure Docker `llama.cpp` provider endpoint**

Set `Local-API.llama_api_IP` for the Docker backend runtime to `http://host.docker.internal:9099/v1/chat/completions` and set the selected `llama.cpp` model id.

Run:

```bash
export DOCKER_LLAMA_CHAT_URL="http://host.docker.internal:9099/v1/chat/completions"
source .venv/bin/activate
python - <<'PY'
import configparser
import os
from pathlib import Path

config_path = Path(os.environ["DOCKER_CONFIG_FILE"])
parser = configparser.ConfigParser()
parser.read(config_path)
updates = {
    "Setup": {
        "enable_first_time_setup": "true",
        "setup_completed": "false",
    },
    "AuthNZ": {
        "auth_mode": "single_user",
        "single_user_api_key": os.environ["UAT_API_KEY"],
    },
    "Local-API": {
        "llama_api_IP": os.environ["DOCKER_LLAMA_CHAT_URL"],
        "llama_model": os.environ["LLAMA_CPP_MODEL"],
    },
}
for section, values in updates.items():
    if not parser.has_section(section):
        parser.add_section(section)
    for key, value in values.items():
        parser.set(section, key, value)
with config_path.open("w", encoding="utf-8") as handle:
    parser.write(handle)
PY
```

Expected: The container reaches the host-local `llama.cpp` service from inside Docker.

- [ ] **Step 4: Start Docker single-user plus WebUI**

Run the existing Docker single-user and WebUI compose flow with the UAT project name, isolated env/config, and overlay.

```bash
NEXT_PUBLIC_API_URL= \
NEXT_PUBLIC_API_BASE_URL= \
docker compose \
  --project-name "${UAT_COMPOSE_PROJECT}" \
  --env-file "${DOCKER_ENV_FILE}" \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  -f "${DOCKER_COMPOSE_OVERLAY}" \
  up -d --build --wait
```

Expected: API and WebUI containers become healthy and expose the expected local ports.

- [ ] **Step 5: Verify Docker health**

Run API health checks and load the WebUI at the Docker WebUI URL.

Expected: The WebUI reaches the API, auth mode is single-user, and console has no startup-blocking errors.

## Task 8: Run Docker Provider Gates

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/provider-preflight.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/docker/provider/`

- [ ] **Step 1: Call Docker backend OpenAI chat path**

Call the Docker-exposed `POST /api/v1/chat/completions` endpoint with `X-API-KEY: $UAT_API_KEY`, OpenAI, `gpt-4o-mini`, and prompt token `docker-openai-$RUN_ID`.

```bash
curl -sf http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: ${UAT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"api_provider\":\"openai\",\"model\":\"gpt-4o-mini\",\"stream\":false,\"temperature\":0,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly docker-openai-${RUN_ID}\"}]}"
```

Expected: HTTP success and a usable response.

- [ ] **Step 2: Call Docker backend `llama.cpp` chat path**

Call the Docker-exposed `POST /api/v1/chat/completions` endpoint with `X-API-KEY: $UAT_API_KEY`, `api_provider: "llama.cpp"`, selected model id, and prompt token `docker-llama-$RUN_ID`.

```bash
curl -sf http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: ${UAT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"api_provider\":\"llama.cpp\",\"model\":\"${LLAMA_CPP_MODEL}\",\"stream\":false,\"temperature\":0,\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly docker-llama-${RUN_ID}\"}]}"
```

Expected: HTTP success and a usable response from the host-local `llama.cpp` service via `host.docker.internal`.

- [ ] **Step 3: Record Docker provider gate result**

Update `provider-preflight.md` and `docker-single-user.md`.

Expected: Both providers are marked pass or a blocking provider failure is documented.

## Task 9: Run Docker Basic User Journey

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/docker/basic/`

- [ ] **Step 1: Run first-entry checks**

Run onboarding or equivalent first-entry WebUI checks against the Docker WebUI.

Expected: A clean disposable user reaches the first-value path with no stale data.

- [ ] **Step 2: Repeat basic document ingest and answer path**

Ingest `basic-user-source.md`, find it in media/search, and ask about it through OpenAI and `llama.cpp`.

Expected: Both providers complete the required document answer path in Docker.

- [ ] **Step 3: Repeat roleplay character chat path**

Create or import the disposable character, start character chat, send one OpenAI message, send one `llama.cpp` message, and reload.

Expected: Character context, response, and persistence pass in Docker.

- [ ] **Step 4: Repeat mobile critical screens**

Run mobile viewport checks for the same basic screens used in local UAT, including character creation/import before character selection and conversation.

Expected: No mobile-only blocker appears.

- [ ] **Step 5: Record findings**

Update `findings.md` and `docker-single-user.md`.

Expected: The Docker basic journey has a clear pass/fail status and evidence references.

## Task 10: Run Docker Advanced Knowledge Journey

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/docker/advanced/`

- [ ] **Step 1: Repeat advanced disposable dataset ingest**

Ingest alpha and beta advanced sources in Docker.

Expected: Search surfaces show only Docker-run disposable data for this test profile.

- [ ] **Step 2: Repeat advanced search, filters, and Knowledge QA**

Run the advanced source filtering, constrained QA, and evidence-control checks.

Expected: Results and answer evidence match the selected source scope.

- [ ] **Step 3: Repeat downstream output handoff and cleanup**

Save or export the generated result, then delete/restore/remove UAT-created objects through product flows.

Expected: The operation affects only `$RUN_ID` data and cleanup result is recorded.

- [ ] **Step 4: Record findings**

Update `findings.md` and `docker-single-user.md`.

Expected: The Docker advanced journey has a clear pass/fail status and evidence references.

## Task 11: Run Targeted Regression Smoke

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/README.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Raw artifacts: `/tmp/tldw-pre-main-uat/<run-id>/smoke/`

- [ ] **Step 1: Run frontend targeted UAT scripts**

From `apps/tldw-frontend`, run the existing UAT scripts that cover onboarding, chat, media, characters, and research workspace where they can target the active WebUI/API.

Expected: Each script either passes or produces a finding linked to the exact UAT gate it affects.

- [ ] **Step 2: Run targeted smoke specs**

From `apps/tldw-frontend`, run relevant smoke/workflow specs for media ingest/search, Knowledge QA, chat, characters, and route startup.

Expected: No unexplained failure remains outside `findings.md`.

- [ ] **Step 3: Record command output**

Copy command summaries, failure snippets, screenshots, and trace paths into the raw artifact directory and summarize them in `README.md`.

Expected: The evidence report includes exact command names and pass/fail status.

## Task 12: Triage And Fix Verified Findings

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/README.md`
- Create: `backlog/tasks/<new-task> - Fix-<finding-slug>.md` for each independent fix
- Modify: Exact current-code files identified per finding
- Test: Exact focused tests identified per finding

- [ ] **Step 1: Classify each finding**

Use the design severity scale: P0 blocker, P1 release blocker, P2 fix before main if practical, or P3 document/defer.

Expected: Every finding has severity, impact, reproduction, affected environment, provider if applicable, and current status.

- [ ] **Step 2: Verify finding against current code**

Inspect current code and rerun the smallest reproduction.

Expected: Still-valid findings proceed to Step 3; invalid or obsolete findings are marked `Skipped` with a brief reason.

- [ ] **Step 3: Create a Backlog task for a valid fix**

Run: `backlog task create "Fix <finding slug> from pre-main UAT" --label bug --label uat --label pr-1982`

Expected: A task id exists before any source edit.

- [ ] **Step 4: Write or identify the focused regression test**

Add a failing test first when the behavior is practical to test. If the issue is browser-only, add a Playwright reproduction or document why manual verification is the only practical check.

Expected: The failure reproduces the current bug or the skip rationale is recorded.

- [ ] **Step 5: Patch minimally**

Modify only the current-code files needed for the verified issue, following existing patterns.

Expected: The patch addresses the reproduction without unrelated refactors.

- [ ] **Step 6: Run focused verification**

Run the relevant pytest, Vitest, Playwright, or harness command.

Expected: The focused verification passes and output is recorded in the Backlog task and UAT evidence.

- [ ] **Step 7: Run Bandit for touched backend Python scope when applicable**

Run:

```bash
source .venv/bin/activate
python -m bandit -r <touched-python-paths> -f json -o /tmp/bandit_<finding-slug>.json
```

Expected: No new security finding in touched code. If no backend Python files were touched, record the non-code/frontend-only Bandit skip.

- [ ] **Step 8: Rerun affected UAT slice**

Rerun the smallest local or Docker UAT slice that proved the issue.

Expected: The finding moves to `Fixed` with fix commit and verification evidence.

- [ ] **Step 9: Commit the fix**

Run:

```bash
git add <touched-files> <finding-task-file> Docs/Product/WebUI/evidence/pre_main_uat/${RUN_ID}
git commit -m "fix: address pre-main UAT <finding slug>"
```

Expected: Commit contains the fix, test/evidence updates, and matching Backlog task update only.

- [ ] **Step 10: Repeat until no open valid P0-P2 findings remain**

Continue the loop for each valid P0-P2. Fix P3 only when the patch is small and low-risk; otherwise document it with the required approval note.

Expected: `findings.md` has no untriaged finding and no unapproved release blocker.

## Task 13: Final Evidence, Cleanup, And Verification

**Files:**
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/README.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/local-single-user.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/docker-single-user.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/provider-preflight.md`
- Modify: `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/findings.md`
- Modify: UAT execution Backlog task

- [ ] **Step 1: Clean up UAT-created data through product flows**

Use app-supported delete/trash/restore/remove flows for objects whose names, tags, notes, or character ids contain `$RUN_ID`.

Expected: Cleanup touches only UAT-created objects and the result is documented.

- [ ] **Step 2: Stop Docker and local services**

Stop the active Docker compose stack and any local server processes started for this UAT.

Run:

```bash
source /tmp/tldw-pre-main-uat/<run-id>/uat.env
if test -n "${DOCKER_COMPOSE_OVERLAY:-}" && test -f "${DOCKER_COMPOSE_OVERLAY}"; then
  NEXT_PUBLIC_API_URL= \
  NEXT_PUBLIC_API_BASE_URL= \
  docker compose \
    --project-name "${UAT_COMPOSE_PROJECT}" \
    --env-file "${DOCKER_ENV_FILE}" \
    -f Dockerfiles/docker-compose.single-user.yml \
    -f Dockerfiles/docker-compose.webui.yml \
    -f "${DOCKER_COMPOSE_OVERLAY}" \
    down -v --remove-orphans
fi
for pid_file in "${LOCAL_PROFILE_ROOT}/webui.pid" "${LOCAL_PROFILE_ROOT}/api.pid"; do
  if test -f "$pid_file"; then
    pid="$(cat "$pid_file")"
    if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      sleep 2
      if kill -0 "$pid" 2>/dev/null; then
        printf 'Process %s from %s is still running after TERM\n' "$pid" "$pid_file" >&2
      fi
    fi
  fi
done
```

Expected: No UAT API/WebUI process remains running unless the user explicitly asks to keep it open.

- [ ] **Step 3: Run final diff and whitespace checks**

Run:

```bash
git status --short --branch
git diff --check
```

Expected: No whitespace errors. Only expected UAT evidence, Backlog tasks, and verified fix files are modified.

- [ ] **Step 4: Run final targeted regression commands**

Run the focused test set that covers all fixed findings plus the targeted smoke/UAT commands that passed the final matrix.

Expected: Commands pass or any remaining failure is listed in `findings.md` with an approved status.

- [ ] **Step 5: Finalize evidence report**

Update `README.md` with:

- Commit SHA and branch.
- Local and Docker matrix results.
- OpenAI and `llama.cpp` provider results.
- Basic and advanced persona results.
- Finding table with statuses.
- Verification commands and timestamps.
- Cleanup status.
- Remaining risks or approved exceptions.

Expected: A reviewer can read the report without opening raw artifacts and know whether PR #1982 is ready to merge to `main`.

- [ ] **Step 6: Finalize Backlog task**

Update the UAT execution Backlog task with final summary, verification results, touched files, blockers, skipped items, and evidence path.

Expected: Task Definition of Done is checked where applicable.

- [ ] **Step 7: Commit final evidence**

Run:

```bash
git add Docs/Product/WebUI/evidence/pre_main_uat/${RUN_ID} backlog/tasks
git commit -m "docs: record pre-main UAT results"
```

Expected: Final commit contains the evidence report and Backlog task updates.

## Completion Criteria

- Both local and Docker single-user WebUI environments complete the bounded UAT matrix or have explicit approved exceptions.
- OpenAI and `llama.cpp` pass direct and backend-mediated gates in each applicable environment or have explicit approved exceptions.
- Basic document ingest/search/answer and roleplay character chat pass for both providers at least once per environment.
- Advanced knowledge search, constrained QA, evidence controls, downstream handoff, destructive UAT-only operations, keyboard checks, and mobile critical checks are recorded.
- All valid P0-P2 findings are fixed or explicitly approved for deferral.
- P3 findings are fixed when small and safe or documented with a reason.
- Final evidence is committed under `Docs/Product/WebUI/evidence/pre_main_uat/<run-id>/`.
- Backlog tasks record verification, Bandit status or skip, and final summary.
