# Public Onboarding Readiness Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved public onboarding readiness review, produce evidence-backed walkthrough artifacts for the three canonical self-hosting profiles, and end with a findings-first readiness verdict for non-super-technical first-time users.

**Architecture:** Execute the review in a disposable worktree so the walkthroughs can use clean state without contaminating active development data. Freeze the onboarding contract first, then run one golden-path walkthrough per public profile with explicit evidence tags, recording findings immediately in profile artifacts. Finish by deduplicating profile findings into one cross-profile findings report and synthesis with separate `core onboarding readiness` and `audio onboarding readiness` summaries.

**Tech Stack:** Markdown artifacts under `Docs/superpowers/reviews`, shell commands (`rg`, `sed`, `curl`, `git`), Makefile targets, Docker Compose, Python 3, local virtualenv, manual browser inspection for UI-only steps

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current checkout state in a disposable worktree, not a mixed developer workspace
- treat the root README plus the approved canonical onboarding docs as the public onboarding contract
- evaluate all three public profiles: Docker single-user + WebUI, Docker multi-user + Postgres, and local single-user
- use the approved golden-path success bar: setup, first chat/API success, ingest + search/retrieval, and audio verification understanding
- separate `core onboarding readiness` and `audio onboarding readiness` in the final synthesis
- record findings only as `Blocker` or `Major confusion trap`; minor friction stays in walkthrough logs
- tag evidence as `Executed`, `Docs-validated`, `Probable risk`, or `Unverified`
- tag findings as `docs`, `runtime`, or `cross-layer`
- do not modify application source code during this review execution
- do not work around missing provider credentials with undocumented mocks or fake backends
- use one real provider env line and one real model name for provider-dependent chat checks
- use repo-local test content for ingest instead of external URLs

## Shared Runtime Inputs

Set or record these before the profile walkthroughs:

- `REVIEW_PROVIDER_ENV_LINE`
  - Example value: `OPENAI_API_KEY=sk-live-or-test-key`
- `REVIEW_CHAT_MODEL`
  - Example value: `gpt-4o-mini`
- multi-user admin credentials to type during `AuthNZ.initialize`
  - username: `tldw_admin`
  - email: `review@example.test`
  - password: `ReviewPass123!`
- shared ingest fixture
  - `tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.md`
- expected ingest query
  - `tldw text processing endpoint`

If `REVIEW_PROVIDER_ENV_LINE` or `REVIEW_CHAT_MODEL` is missing, do not invent a substitute. Execute the setup and documentation portions anyway, and record provider-dependent runtime steps as blocked by missing review credentials.

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/public-onboarding-readiness/README.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-multi-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-local-single-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-24-public-onboarding-readiness-review-design.md`
- `Docs/superpowers/plans/2026-04-24-public-onboarding-readiness-review-execution-plan.md`

**Canonical onboarding docs to inspect first:**
- `README.md`
- `Docs/Getting_Started/README.md`
- `Docs/Getting_Started/Profile_Docker_Single_User.md`
- `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- `Docs/Getting_Started/Profile_Local_Single_User.md`
- `Docs/Deployment/setup-wizard-guide.md`
- `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md`

**Scratch artifacts allowed during execution:**
- `/tmp/public_onboarding_single_docker.log`
- `/tmp/public_onboarding_multi_docker.log`
- `/tmp/public_onboarding_local_server.log`
- `/tmp/public_onboarding_single_tts.mp3`
- `/tmp/public_onboarding_local_tts.mp3`

## Stage Overview

## Stage 1: Baseline and Onboarding Contract
**Goal:** Create review artifacts, freeze the public onboarding surface, and build the comparison matrix before runtime walkthroughs begin.
**Success Criteria:** The review directory exists, all artifact skeletons are in place, the contract matrix is populated from the canonical docs, and the baseline environment is recorded.
**Tests:** Shell inspection only. No application runtime yet.
**Status:** Not Started

## Stage 2: Docker Single-User + WebUI Walkthrough
**Goal:** Follow the recommended public quickstart path exactly, verify the first-value flow, and capture where the default Docker onboarding succeeds, stalls, or misleads.
**Success Criteria:** The single-user walkthrough artifact captures setup, WebUI observation, provider setup, first chat, ingest/search, and `/setup` discoverability with clear evidence tags.
**Tests:** `make quickstart-prereqs`, `make quickstart`, health/docs/WebUI checks, chat curl, media add/search curl, manual browser checks.
**Status:** Not Started

## Stage 3: Docker Multi-User + Postgres Walkthrough
**Goal:** Follow the public multi-user operator path through admin creation, login, first authenticated value, and audio-path discoverability.
**Success Criteria:** The multi-user walkthrough artifact captures auth setup, first admin creation, login, optional WebUI handoff, first chat, ingest/search, and multi-user audio-path clarity.
**Tests:** Docker Compose startup, `AuthNZ.initialize`, login curl, bearer-auth chat curl, media add/search curl, manual browser checks.
**Status:** Not Started

## Stage 4: Local Single-User Walkthrough
**Goal:** Follow the local single-user path as written, validate whether it behaves like a contributor setup or an approachable public path, and use it as the main execution lane for the setup wizard audio flow.
**Success Criteria:** The local walkthrough artifact captures install/run behavior, provider setup, first chat, ingest/search, the WebUI handoff, and a real `/setup`-driven audio verification attempt.
**Tests:** `make quickstart-install`, `make quickstart-local`, health/docs checks, chat curl, media add/search curl, `/setup` and audio verification curl calls.
**Status:** Not Started

## Stage 5: Findings and Synthesis
**Goal:** Turn the walkthrough evidence into one findings-first report and a final synthesis with clear verdicts and priorities.
**Success Criteria:** Findings are deduplicated, severity-ranked, and source-linked; the synthesis answers the four review questions from the spec; the platform evidence table is complete.
**Tests:** Artifact consistency checks and evidence cross-checks only.
**Status:** Not Started

### Task 1: Create Review Artifacts and Freeze the Baseline

**Files:**
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/README.md`
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md`
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-multi-user-walkthrough.md`
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-local-single-user-walkthrough.md`
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Create: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`
- Inspect: `Docs/superpowers/specs/2026-04-24-public-onboarding-readiness-review-design.md`
- Inspect: `README.md`
- Inspect: `Docs/Getting_Started/README.md`

- [ ] **Step 1: Move into a disposable worktree**

Run:
```bash
git worktree add /tmp/tldw_server_public_onboarding_review HEAD
cd /tmp/tldw_server_public_onboarding_review
```

Expected: all remaining review artifacts are created in `/tmp/tldw_server_public_onboarding_review`, not in an active developer workspace.

- [ ] **Step 2: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/public-onboarding-readiness
```

Expected: the review directory exists and no application source files change.

- [ ] **Step 3: Write `Docs/superpowers/reviews/public-onboarding-readiness/README.md`**

Use this exact content:
```markdown
# Public Onboarding Readiness Review

## Stage Order

1. Baseline and contract matrix
2. Docker single-user + WebUI walkthrough
3. Docker multi-user + Postgres walkthrough
4. Local single-user walkthrough
5. Findings and synthesis

## Review Artifacts

- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-multi-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-local-single-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`

## Rules

- Findings are written before remediation ideas.
- Only `Blocker` and `Major confusion trap` belong in the main findings report.
- Minor friction stays in the walkthrough logs unless it clusters into a larger problem.
- Evidence tags must be one of `Executed`, `Docs-validated`, `Probable risk`, or `Unverified`.
- Findings tags must be one of `docs`, `runtime`, or `cross-layer`.
- The final synthesis must report both `core onboarding readiness` and `audio onboarding readiness`.
```

- [ ] **Step 4: Create the contract matrix scaffold**

Use this exact content:
```markdown
# Public Onboarding Contract Matrix

## Canonical Surfaces

- `README.md`
- `Docs/Getting_Started/README.md`
- `Docs/Getting_Started/Profile_Docker_Single_User.md`
- `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- `Docs/Getting_Started/Profile_Local_Single_User.md`
- `Docs/Deployment/setup-wizard-guide.md`
- `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md`

## Matrix

| Surface | Intended user | Prereqs | Auth setup | Provider setup path | First-value path | Ingest path | Audio path | Verify step | Platform notes | Evidence tag | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

- [ ] **Step 5: Create the three walkthrough scaffolds**

Use these exact title lines:
```markdown
# Docker Single-User + WebUI Walkthrough
# Docker Multi-User + Postgres Walkthrough
# Local Single-User Walkthrough
```

Under each title, use this exact section structure:
```markdown
## Goal

## Canonical Docs Followed

## Environment

## Step Log
| Step | Source doc | Expected user belief | Action taken | Result | Evidence tag | Notes |
| --- | --- | --- | --- | --- | --- | --- |

## Manual UI Observations

## Blockers
_None._

## Major Confusion Traps
_None._

## Minor Frictions
_None._

## Exit Note
```

- [ ] **Step 6: Create the findings and synthesis scaffolds**

Use this exact content for `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`:
```markdown
# Public Onboarding Readiness Findings

## Severity Model

- `Blocker`: setup or first-use is likely to fail outright, or the documented path routes users into failure
- `Major confusion trap`: a non-technical user is likely to choose the wrong path, miss a hidden prerequisite, or get stuck even though recovery is possible

## Blockers
_None._

## Major Confusion Traps
_None._
```

Use this exact content for `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`:
```markdown
# Public Onboarding Readiness Synthesis

## Overall Verdict

## Core Onboarding Readiness

## Audio Onboarding Readiness

## Per-Profile Verdicts

## Platform Evidence Table
| Profile | macOS | Windows/WSL | Linux | Notes |
| --- | --- | --- | --- | --- |

## Top 5 Issues

## Safest Profile To Recommend Today

## Profiles Or Sub-Flows To De-Emphasize

## Open Questions
```

- [ ] **Step 7: Capture the baseline and shared review inputs**

Run:
```bash
pwd
git status --short
git rev-parse --short HEAD
date
command -v docker || true
command -v ffmpeg || true
command -v bun || true
python3 --version
test -n "$REVIEW_PROVIDER_ENV_LINE" && echo "REVIEW_PROVIDER_ENV_LINE=present" || echo "REVIEW_PROVIDER_ENV_LINE=missing"
test -n "$REVIEW_CHAT_MODEL" && echo "REVIEW_CHAT_MODEL=present" || echo "REVIEW_CHAT_MODEL=missing"
```

Expected: the baseline artifact can state the exact workspace, commit, host tools, and whether provider-dependent runtime steps are fully executable.

- [ ] **Step 8: Save the scaffolds before deep review**

Run:
```bash
git diff -- Docs/superpowers/reviews/public-onboarding-readiness
```

Expected: the diff shows only markdown review artifacts and no source-code edits.

### Task 2: Build the Onboarding Contract Matrix and Static Audit

**Files:**
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Inspect: `README.md`
- Inspect: `Docs/Getting_Started/README.md`
- Inspect: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Inspect: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Inspect: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Inspect: `Docs/Deployment/setup-wizard-guide.md`
- Inspect: `Docs/Getting_Started/First_Time_Audio_Setup_CPU.md`
- Inspect: `Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md`
- Inspect: `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md`

- [ ] **Step 1: Freeze the section inventory for the onboarding docs**

Run:
```bash
rg -n '^## ' \
  README.md \
  Docs/Getting_Started/README.md \
  Docs/Getting_Started/Profile_Docker_Single_User.md \
  Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md \
  Docs/Getting_Started/Profile_Local_Single_User.md \
  Docs/Deployment/setup-wizard-guide.md \
  Docs/Getting_Started/First_Time_Audio_Setup_CPU.md \
  Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md \
  Docs/User_Guides/Server/Multi-User_Postgres_Setup.md
```

Expected: a complete section map for the canonical onboarding surface.

- [ ] **Step 2: Read the onboarding docs in full before any runtime walkthrough**

Run:
```bash
sed -n '1,260p' README.md
sed -n '1,220p' Docs/Getting_Started/README.md
sed -n '1,220p' Docs/Getting_Started/Profile_Docker_Single_User.md
sed -n '1,220p' Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md
sed -n '1,220p' Docs/Getting_Started/Profile_Local_Single_User.md
sed -n '1,260p' Docs/Deployment/setup-wizard-guide.md
sed -n '1,260p' Docs/Getting_Started/First_Time_Audio_Setup_CPU.md
sed -n '1,260p' Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md
sed -n '1,220p' Docs/User_Guides/Server/Multi-User_Postgres_Setup.md
```

Expected: the contract matrix is built from the actual public docs, not memory or assumptions.

- [ ] **Step 3: Populate the contract matrix from the docs**

For each canonical surface, fill one matrix row with:
```markdown
| `README.md` | new self-hosters choosing a profile | stated prereqs only | exact auth guidance given | exact provider guidance given | first concrete value promised | ingest path named? | audio path named? | exact verify command or URL | macOS/Windows/Linux notes present? | Docs-validated | unresolved gaps |
```

Expected: the matrix makes it obvious where a profile depends on another doc, where auth handoffs are hidden, and where audio/setup guidance forks.

- [ ] **Step 4: Run one cross-doc consistency scan for path selection and handoffs**

Run:
```bash
rg -n "make quickstart|quickstart-install|Docker multi-user|Local single-user|create the first admin user|AuthNZ.initialize|/setup|First-Time Audio|What to Do Next|Windows users|WSL2|Git Bash|same-origin|show-api-key" \
  README.md \
  Docs/Getting_Started/README.md \
  Docs/Getting_Started/Profile_Docker_Single_User.md \
  Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md \
  Docs/Getting_Started/Profile_Local_Single_User.md \
  Docs/Deployment/setup-wizard-guide.md \
  Docs/Getting_Started/First_Time_Audio_Setup_CPU.md \
  Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md \
  Docs/User_Guides/Server/Multi-User_Postgres_Setup.md
```

Expected: the scan highlights contradictions or undocumented handoffs before runtime begins.

- [ ] **Step 5: Seed only doc-backed risks that clearly deserve runtime confirmation**

Add notes to the findings file using this structure, but keep them out of the final severity sections until a walkthrough confirms or strongly supports them:
```markdown
## Working Notes

- Candidate: `cross-layer`
  - Starts at: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
  - User belief: the profile guide is self-contained enough to reach first login
  - Suspected problem: first-admin creation is deferred into another doc and may not feel like part of the main setup path
  - Runtime confirmation needed: yes
```

- [ ] **Step 6: Snapshot the contract matrix before runtime work**

Run:
```bash
git diff -- Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md
```

Expected: the diff shows a populated matrix that can be referenced from every profile walkthrough.

### Task 3: Execute the Docker Single-User + WebUI Walkthrough

**Files:**
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`
- Inspect: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Inspect: `Docs/Deployment/setup-wizard-guide.md`

- [ ] **Step 1: Reset the Docker single-user review state**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
export COMPOSE_PROJECT_NAME=tldw_ftux_single
docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml down -v --remove-orphans || true
rm -f tldw_Server_API/Config_Files/.env
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
```

Expected: the walkthrough starts from a clean `.env` file and isolated Docker project state.

- [ ] **Step 2: Follow the documented install and run path exactly**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
COMPOSE_PROJECT_NAME=tldw_ftux_single make quickstart-prereqs
COMPOSE_PROJECT_NAME=tldw_ftux_single make quickstart
```

Expected: the default Docker + WebUI quickstart starts without requiring undocumented flags or overrides.

- [ ] **Step 3: Verify the documented endpoints and retrieve the API key**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/docs > /dev/null && echo "docs-ok"
curl -sS http://127.0.0.1:8000/api/v1/config/quickstart
curl -sS http://127.0.0.1:8080 > /dev/null && echo "webui-ok"
make show-api-key
```

Expected: health, docs, quickstart config, WebUI, and API key retrieval all work as described in the profile guide.

- [ ] **Step 4: Perform a manual WebUI observation**

Open `http://localhost:8080` in a browser and record answers to these prompts in `Manual UI Observations`:
```markdown
- First screen shown:
- Is the next step obvious without reading code or logs?
- Is the API key required in-browser, or is the proxy behavior clear?
- Is provider setup explained anywhere visible?
- Is there a visible path to ingest content?
```

Expected: the walkthrough records what a non-technical user would actually see, not only what the HTML source contains.

- [ ] **Step 5: Follow the documented provider-setup handoff and attempt the first chat**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
API_KEY=$(make show-api-key)
printf '\n%s\n' "$REVIEW_PROVIDER_ENV_LINE" >> tldw_Server_API/Config_Files/.env
COMPOSE_PROJECT_NAME=tldw_ftux_single docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml restart
curl -sS http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$REVIEW_CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}]}"
```

Expected: either a valid chat response or a clear user-visible failure. If the restart does not apply the new provider env line, record that exact behavior first, then recover with:
```bash
cd /tmp/tldw_server_public_onboarding_review
COMPOSE_PROJECT_NAME=tldw_ftux_single docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
```

- [ ] **Step 6: Ingest the repo-local sample file and search for it**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS -X POST http://127.0.0.1:8000/api/v1/media/add \
  -H "X-API-KEY: $API_KEY" \
  -F "title=Public Onboarding Sample" \
  -F "media_type=document" \
  -F "chunk_method=words" \
  -F "chunk_size=50" \
  -F "chunk_overlap=10" \
  -F "files=@tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.md;type=text/markdown"

curl -sS -X POST http://127.0.0.1:8000/api/v1/media/search \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"tldw text processing endpoint"}'
```

Expected: the file ingests without needing external URLs, and the search response clearly surfaces the uploaded content.

- [ ] **Step 7: Check `/setup` discoverability and Docker-specific setup friction**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/setup
rg -n "Guided Setup Wizard|/setup|config.txt|rebuild" \
  Docs/Getting_Started/Profile_Docker_Single_User.md \
  Docs/Deployment/setup-wizard-guide.md \
  Docs/Getting_Started/First_Time_Audio_Setup_CPU.md \
  Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md
```

Expected: the walkthrough records whether `/setup` is visible in the default Docker path, and whether the docs make Docker-specific config/rebuild requirements obvious before users try to use the wizard.

- [ ] **Step 8: Preserve logs if anything failed, then tear down the isolated single-user stack**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
COMPOSE_PROJECT_NAME=tldw_ftux_single docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml logs --tail=200 > /tmp/public_onboarding_single_docker.log || true
COMPOSE_PROJECT_NAME=tldw_ftux_single docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml down -v --remove-orphans
```

Expected: the review keeps a short Docker log for evidence and leaves the single-user stack clean before the next profile.

### Task 4: Execute the Docker Multi-User + Postgres Walkthrough

**Files:**
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-multi-user-walkthrough.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`
- Inspect: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Inspect: `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md`
- Inspect: `Docs/Deployment/setup-wizard-guide.md`

- [ ] **Step 1: Reset the Docker multi-user review state and write the multi-user env file**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
export COMPOSE_PROJECT_NAME=tldw_ftux_multi
docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml down -v --remove-orphans || true
rm -f tldw_Server_API/Config_Files/.env
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
SESSION_ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
MCP_JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
MCP_API_KEY_SALT=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
printf '\nAUTH_MODE=multi_user\nDATABASE_URL=postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users\nJWT_SECRET_KEY=%s\nSESSION_ENCRYPTION_KEY=%s\nMCP_JWT_SECRET=%s\nMCP_API_KEY_SALT=%s\ntldw_production=true\n' \
  "$JWT_SECRET_KEY" "$SESSION_ENCRYPTION_KEY" "$MCP_JWT_SECRET" "$MCP_API_KEY_SALT" >> tldw_Server_API/Config_Files/.env
```

Expected: the env file matches the documented multi-user requirements and uses deterministic local test values where the docs do not force operator-specific secrets.

- [ ] **Step 2: Start services and run the documented AuthNZ initialization path**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.yml up -d --build
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.yml exec app \
  python -m tldw_Server_API.app.core.AuthNZ.initialize
```

When prompted, type exactly:
```text
Admin username: tldw_admin
Admin email: review@example.test
Admin password: ReviewPass123!
Confirm password: ReviewPass123!
```

Expected: the multi-user guide is executable as written, and the admin-creation prompt quality is captured in the walkthrough notes.

- [ ] **Step 3: Verify the API, log in, and extract a bearer token**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/docs > /dev/null && echo "docs-ok"
JWT_TOKEN=$(curl -sS -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"tldw_admin","password":"ReviewPass123!"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
echo "${JWT_TOKEN:0:20}..."
```

Expected: login succeeds with the first admin user, and token extraction does not require extra undocumented steps.

- [ ] **Step 4: Add the WebUI overlay and perform a manual login-path observation**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
curl -sS http://127.0.0.1:8080 > /dev/null && echo "webui-ok"
```

Then open `http://localhost:8080/login` in a browser and record:
```markdown
- Is the login route discoverable from the default landing experience?
- Is it obvious that the earlier admin creation step is required before login works?
- Is there any visible guidance about provider setup after login?
```

Expected: the multi-user walkthrough includes the actual login-facing UI contract, not only the API contract.

- [ ] **Step 5: Follow the provider-setup handoff and attempt the first authenticated chat**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
printf '\n%s\n' "$REVIEW_PROVIDER_ENV_LINE" >> tldw_Server_API/Config_Files/.env
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.yml \
  -f Dockerfiles/docker-compose.webui.yml \
  up -d --build
curl -sS http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$REVIEW_CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}]}"
```

Expected: either the first authenticated chat works, or the failure clearly exposes where multi-user onboarding still lacks provider or role guidance.

- [ ] **Step 6: Ingest the repo-local sample file and search for it with bearer auth**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS -X POST http://127.0.0.1:8000/api/v1/media/add \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "title=Public Onboarding Sample Multi" \
  -F "media_type=document" \
  -F "chunk_method=words" \
  -F "chunk_size=50" \
  -F "chunk_overlap=10" \
  -F "files=@tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.md;type=text/markdown"

curl -sS -X POST http://127.0.0.1:8000/api/v1/media/search \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"tldw text processing endpoint"}'
```

Expected: the shortest documented authenticated content flow is observable without inventing hidden setup steps.

- [ ] **Step 7: Probe multi-user audio-path clarity instead of brute-forcing undocumented setup**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
rg -n "Authorization: Bearer|X-API-KEY|audio/voices/catalog|audio/speech|audio/transcriptions|/setup" \
  Docs/Deployment/setup-wizard-guide.md \
  Docs/Getting_Started/First_Time_Audio_Setup_CPU.md \
  Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md

curl -sS http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "Authorization: Bearer $JWT_TOKEN"
```

Expected: the walkthrough records whether multi-user audio setup is actually documented for JWT users. If the first authenticated audio check fails, record the first failing step and stop instead of improvising undocumented multi-user audio setup.

- [ ] **Step 8: Preserve logs if anything failed, then tear down the isolated multi-user stack**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml logs --tail=200 > /tmp/public_onboarding_multi_docker.log || true
COMPOSE_PROJECT_NAME=tldw_ftux_multi docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml down -v --remove-orphans
```

Expected: the multi-user review ends with logs preserved and no running containers left behind.

### Task 5: Execute the Local Single-User Walkthrough

**Files:**
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-local-single-user-walkthrough.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`
- Inspect: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Inspect: `Docs/Deployment/setup-wizard-guide.md`
- Inspect: `README.md`

- [ ] **Step 1: Reset the local review state**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
test -f /tmp/public_onboarding_local_server.pid && kill "$(cat /tmp/public_onboarding_local_server.pid)" || true
rm -f /tmp/public_onboarding_local_server.pid
rm -rf .venv
rm -f tldw_Server_API/Config_Files/.env
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
```

Expected: the local walkthrough starts from a fresh venv and fresh env file inside the disposable worktree.

- [ ] **Step 2: Run the documented install command exactly and record what actually happens**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
make quickstart-install
```

Expected: either it behaves like an install-only step, or it does something stronger than the doc promises. Record the exact behavior in the walkthrough before correcting course.

- [ ] **Step 3: Ensure the local server is running and verify the API**

If Step 2 did not leave the server running, start a controlled local server process with:
```bash
cd /tmp/tldw_server_public_onboarding_review
source .venv/bin/activate
nohup python -m uvicorn tldw_Server_API.app.main:app --reload > /tmp/public_onboarding_local_server.log 2>&1 &
echo $! > /tmp/public_onboarding_local_server.pid
sleep 5
```

Then run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/docs > /dev/null && echo "docs-ok"
curl -sS http://127.0.0.1:8000/api/v1/config/quickstart
API_KEY=$(grep '^SINGLE_USER_API_KEY=' tldw_Server_API/Config_Files/.env | cut -d= -f2-)
echo "$API_KEY"
```

Expected: the local path yields a working API and a discoverable single-user API key.

- [ ] **Step 4: Follow the provider-setup handoff and attempt the first chat**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
printf '\n%s\n' "$REVIEW_PROVIDER_ENV_LINE" >> tldw_Server_API/Config_Files/.env
test -f /tmp/public_onboarding_local_server.pid && kill "$(cat /tmp/public_onboarding_local_server.pid)" || true
source .venv/bin/activate
nohup python -m uvicorn tldw_Server_API.app.main:app --reload > /tmp/public_onboarding_local_server.log 2>&1 &
echo $! > /tmp/public_onboarding_local_server.pid
sleep 5
curl -sS http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$REVIEW_CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}]}"
```

Expected: the local profile can reach the first meaningful API success without extra undocumented environment steps.

- [ ] **Step 5: Ingest the repo-local sample file and search for it**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS -X POST http://127.0.0.1:8000/api/v1/media/add \
  -H "X-API-KEY: $API_KEY" \
  -F "title=Public Onboarding Sample Local" \
  -F "media_type=document" \
  -F "chunk_method=words" \
  -F "chunk_size=50" \
  -F "chunk_overlap=10" \
  -F "files=@tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.md;type=text/markdown"

curl -sS -X POST http://127.0.0.1:8000/api/v1/media/search \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"tldw text processing endpoint"}'
```

Expected: the local profile reaches the same content-retrieval milestone as the Docker path.

- [ ] **Step 6: Inspect the WebUI handoff from the local profile**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
rg -n "Local Profile: Add the WebUI|Run the Web UI|bun install|npm install|NEXT_PUBLIC_API_URL" README.md apps/tldw-frontend/README.md
```

Expected: the walkthrough records whether a non-technical user can follow the local profile into the WebUI without getting lost in contributor-oriented frontend setup.

- [ ] **Step 7: Execute the shortest supported `/setup` audio path on the local server**

First choose the guide lane:
```bash
cd /tmp/tldw_server_public_onboarding_review
HOST_OS=$(uname -s)
HOST_ARCH=$(uname -m)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU_GUIDE"
elif [ "$HOST_OS" = "Darwin" ] && [ "$HOST_ARCH" = "arm64" ]; then
  echo "GPU_GUIDE"
else
  echo "CPU_GUIDE"
fi
```

Then edit `tldw_Server_API/Config_Files/config.txt` so the `[Setup]` block contains:
```ini
[Setup]
enable_first_time_setup = true
setup_completed = false
```

Restart the local server, open `http://127.0.0.1:8000/setup`, and follow the wizard through:
```markdown
1. Save any required config changes
2. Accept or change the recommended audio bundle
3. Provision the selected bundle
4. Run verification
5. Review the readiness report
```

After the wizard, run the API-level checks:
```bash
cd /tmp/tldw_server_public_onboarding_review
curl -sS http://127.0.0.1:8000/api/v1/audio/voices/catalog \
  -H "X-API-KEY: $API_KEY"

curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","voice":"af_bella","input":"Setup verification complete","response_format":"mp3"}' \
  --output /tmp/public_onboarding_local_tts.mp3

curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  -H "X-API-KEY: $API_KEY" \
  -F "file=@/tmp/public_onboarding_local_tts.mp3" \
  -F "model=whisper-1"
```

Expected: the local single-user path gives the clearest supported answer to whether `/setup` can carry a first-time user through audio onboarding.

- [ ] **Step 8: Stop the local server and preserve its log**

Run:
```bash
cd /tmp/tldw_server_public_onboarding_review
test -f /tmp/public_onboarding_local_server.pid && kill "$(cat /tmp/public_onboarding_local_server.pid)" || true
rm -f /tmp/public_onboarding_local_server.pid
test -f /tmp/public_onboarding_local_server.log && tail -n 200 /tmp/public_onboarding_local_server.log || true
```

Expected: the local walkthrough ends cleanly with enough log evidence to support any runtime finding.

### Task 6: Write the Findings Report and Final Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`
- Modify: `Docs/superpowers/reviews/public-onboarding-readiness/README.md`
- Inspect: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md`
- Inspect: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`
- Inspect: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-multi-user-walkthrough.md`
- Inspect: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-local-single-user-walkthrough.md`

- [ ] **Step 1: Promote only confirmed or strongly supported findings into the main report**

Use this exact structure for every entry:
```markdown
### Blocker: Provider setup handoff does not actually reach first chat
- Tag: `cross-layer`
- Profiles: `Docker single-user + WebUI`
- Starts at: `Docs/Getting_Started/Profile_Docker_Single_User.md` provider-setup step
- User belief: adding a provider key and restarting the containers is enough to enable chat
- What happens: the first chat still fails because the documented restart path does not reload the new env into the running app container
- Evidence: `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`, first-chat step plus exact curl output
- Recovery discovered: `docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml up -d --build`
```

Expected: every formal finding has a clear user-level consequence and a source trail.

- [ ] **Step 2: Deduplicate repeated symptoms into one root finding when appropriate**

Run:
```bash
rg -n "Blocker:|Major confusion trap:" Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-*.md
```

Expected: the same root problem is not reported three times under slightly different wording when one cross-profile finding is clearer.

- [ ] **Step 3: Write the final synthesis using the approved output contract**

Make sure `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md` answers these exact questions:
```markdown
- Is the project ready to share publicly with non-technical self-hosters?
- Which of the three public profiles is safest to recommend today?
- Which blockers or major confusion traps must be fixed first?
- Which profiles or sub-flows should be de-emphasized until improved?
```

And include these exact verdict sections:
```markdown
## Overall Verdict
## Core Onboarding Readiness
## Audio Onboarding Readiness
## Per-Profile Verdicts
## Platform Evidence Table
## Top 5 Issues
## Safest Profile To Recommend Today
## Profiles Or Sub-Flows To De-Emphasize
```

- [ ] **Step 4: Cross-check the synthesis against the approved spec**

Run:
```bash
rg -n "core onboarding readiness|audio onboarding readiness|Platform Evidence Table|Top 5 Issues|Safest Profile To Recommend Today|Profiles Or Sub-Flows To De-Emphasize" \
  Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md
```

Expected: the synthesis contains every required section from the approved design.

- [ ] **Step 5: Review the artifact set before handing it back**

Run:
```bash
git diff -- Docs/superpowers/reviews/public-onboarding-readiness
```

Expected: the diff contains only the review artifacts and shows a complete evidence trail from contract matrix through synthesis.

- [ ] **Step 6: Commit the completed review artifacts**

Run:
```bash
git add Docs/superpowers/reviews/public-onboarding-readiness
git commit -m "docs: add public onboarding readiness review"
```

Expected: the completed review is preserved as a docs-only commit with no application code changes.
