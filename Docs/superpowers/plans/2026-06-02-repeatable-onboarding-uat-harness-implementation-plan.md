# Repeatable Onboarding UAT Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PR1 of the solo onboarding V2 roadmap: a repeatable manual/dev UAT harness that starts the repo mock OpenAI-compatible API server, real backend, real WebUI, and Playwright against an isolated first-run profile, then writes screenshots, logs, and a JSON summary.

**Architecture:** Add a frontend-owned runner under `apps/tldw-frontend/scripts/onboarding-uat/` that owns temp runtime setup, port selection, process startup, artifact capture, cleanup, and final summary generation. Add a dedicated Playwright config/specs under `apps/tldw-frontend/e2e/onboarding-uat/` that avoid existing auth/first-run bypass fixtures and drive the WebUI through real backend APIs. Extend `mock_openai_server` only enough to support deterministic scenario controls from static config files, never Playwright route mocks.

**Tech Stack:** Node ESM scripts, Playwright, Next.js dev server, FastAPI/Uvicorn, pytest, Vitest, repo `mock_openai_server`, isolated SQLite/config/env runtime profile.

---

## Source Documents

- Roadmap spec: `Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md`
- Backlog planning task: `TASK-505`
- Prior onboarding implementation plan: `Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md`
- Current WebUI setup route: `apps/packages/ui/src/routes/option-setup.tsx`
- Current onboarding component: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- Current onboarding evidence spec: `apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts`
- Current E2E helpers: `apps/tldw-frontend/e2e/utils/helpers.ts`, `apps/tldw-frontend/e2e/utils/journey-helpers.ts`
- Current Playwright config: `apps/tldw-frontend/playwright.config.ts`
- Current mock server: `mock_openai_server/mock_openai/server.py`, `mock_openai_server/mock_openai/config.py`, `mock_openai_server/mock_openai/responses.py`

## Scope

This PR adds the harness and evidence contract. It must not redesign onboarding UI, add guided diagnostics, add first-source starter questions, or improve local model setup beyond what is necessary to test the current merged behavior. Those are PR2 through PR4.

PR1 must:

- Use the repo `mock_openai_server` for provider behavior.
- Run the real backend and real WebUI.
- Use isolated temp config, `.env`, databases, uploads, logs, and storage.
- Use synthetic secrets only.
- Require a real successful chat response for completion assertions.
- Capture screenshots, backend log, frontend log, mock log, browser console/network failures, and JSON summary.
- Preserve `make quickstart` and existing E2E commands.
- Be a manual/dev command first, not a blocking CI gate.
- Avoid Computer Use; browser automation is Playwright/CDP only.

## File Map

### Mock OpenAI Server Deterministic Controls

- Modify: `mock_openai_server/mock_openai/config.py`
  - Parse scenario controls from static JSON/YAML config.
- Modify: `mock_openai_server/mock_openai/server.py`
  - Apply deterministic failure controls before normal chat/model/embedding handlers.
- Test: `mock_openai_server/tests/test_server.py`
  - Add enabled tests for fail-once chat and static model-list behavior; keep the existing auth tests passing.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/hosted-success.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/chat-fail-once.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/model-unavailable.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/chat/default.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/chat/source-summary.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/embeddings/default.json`

### UAT Runner

- Create: `apps/tldw-frontend/scripts/onboarding-uat/run.mjs`
  - CLI entrypoint for `bun run e2e:onboarding:uat`.
- Create: `apps/tldw-frontend/scripts/onboarding-uat/ports.mjs`
  - Finds unused loopback ports.
- Create: `apps/tldw-frontend/scripts/onboarding-uat/processes.mjs`
  - Starts, waits for, logs, and stops child processes.
- Create: `apps/tldw-frontend/scripts/onboarding-uat/profile.mjs`
  - Builds isolated temp runtime profile and patches config/env.
- Create: `apps/tldw-frontend/scripts/onboarding-uat/artifacts.mjs`
  - Owns run id, artifact paths, redaction, summary merge, and cleanup.
- Test: `apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts`
  - Unit tests for profile, redaction, cleanup, and command assembly.

### Playwright UAT Project

- Create: `apps/tldw-frontend/e2e/onboarding-uat/playwright.config.ts`
  - Dedicated config without the normal webServer auto-start.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/fixtures.ts`
  - First-run page fixture, diagnostics capture, artifact helper, no auth seeding.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts`
  - Matrix definitions for Tier A scenarios.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts`
  - Onboarding form, chat, quick ingest, and assertion helpers.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts`
  - Hosted and local setup-to-first-chat paths.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts`
  - Paste/file/web first-source paths.
- Create: `apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts`
  - Provider validation failure, first-chat fail-once retry, ingest failure retry.
- Create: `apps/tldw-frontend/e2e/fixtures/media/onboarding-uat-note.md`
  - Local file source fixture.
- Create: `apps/tldw-frontend/public/e2e/onboarding-uat-research-note.html`
  - Local web URL source fixture.

### Package Scripts And Guards

- Modify: `apps/tldw-frontend/package.json`
  - Add `e2e:onboarding:uat` and focused unit-test script if useful.
- Modify: `apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts`
  - Add source guard for no `waitForTimeout`, no Playwright route mocks for provider/chat behavior, no normal auth seeding in UAT specs.
- Modify: `Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md` only if implementation discoveries require a narrowly scoped clarification.
- Modify: `backlog/tasks/task-505 - Plan-repeatable-solo-onboarding-UAT-harness.md`
  - Track plan completion and verification.

## Scenario Matrix

Tier A manual run for PR1:

| Scenario id | Viewport | Mock config | Required outcome |
|---|---:|---|---|
| `hosted-openai-first-chat` | desktop + mobile | `hosted-success.json` | Setup connects with synthetic OpenAI key, chat sends one message, assistant response appears. |
| `local-openai-compatible-first-chat` | desktop | `local-success.json` | Setup chooses local/OpenAI-compatible path or custom provider config where current UI supports it, chat succeeds against mock `/v1`. |
| `first-source-paste` | desktop | `hosted-success.json` | User reaches first-source milestone, pastes structured note text, ingest completes, Media route shows result. |
| `first-source-file` | desktop | `hosted-success.json` | Uploads `onboarding-uat-note.md`, ingest completes, Media route shows result. |
| `first-source-web-url` | desktop + mobile | `hosted-success.json` | Ingests `http://127.0.0.1:<web-port>/e2e/onboarding-uat-research-note.html`, ingest completes. |
| `provider-validation-failure-recovery` | desktop | `model-unavailable.json` then `hosted-success.json` | Failure is visible, retry or provider edit path recovers to success. |
| `first-chat-fail-once-retry` | desktop | `chat-fail-once.json` | First chat fails from backend/provider response, retry succeeds without route mocking. |
| `ingest-failure-retry` | desktop | `hosted-success.json` | Deterministic bad local fixture or unsupported input fails, retry with valid fixture succeeds. |

If current UI cannot express a peer local provider choice yet, implement the local scenario through existing config/default-provider controls and record that the richer local setup UI remains PR4 scope. Do not fake unsupported UI with route interception.

## Artifact Contract

Default artifact root:

- Ephemeral local runs: `apps/tldw-frontend/test-results/onboarding-uat/<run-id>/`
- Optional reviewed evidence copy: `Docs/Product/WebUI/evidence/onboarding_uat/<run-id>/`

Each run writes:

- `summary.json`
- `logs/backend.log`
- `logs/frontend.log`
- `logs/mock-openai.log`
- `logs/runner.log`
- `browser/console-and-network.json`
- `screenshots/<scenario-id>/<step>.png`
- `runtime-profile/manifest.redacted.json`

`summary.json` shape:

```json
{
  "run_id": "2026-06-02T12-34-56-789Z-abc123",
  "started_at": "2026-06-02T12:34:56.789Z",
  "finished_at": "2026-06-02T12:38:10.123Z",
  "status": "passed",
  "ports": {
    "backend": 18110,
    "web": 18111,
    "mock_openai": 18112
  },
  "artifacts": {
    "root": "apps/tldw-frontend/test-results/onboarding-uat/...",
    "backend_log": "logs/backend.log",
    "frontend_log": "logs/frontend.log",
    "mock_log": "logs/mock-openai.log"
  },
  "scenarios": [
    {
      "id": "hosted-openai-first-chat",
      "viewport": "desktop",
      "status": "passed",
      "duration_ms": 42000,
      "failure_category": null,
      "screenshots": ["screenshots/hosted-openai-first-chat/01-setup.png"],
      "required_api_failures": [],
      "critical_console_errors": []
    }
  ],
  "skips": [],
  "redaction": {
    "checked": true,
    "leaks": []
  }
}
```

The runner fails with a nonzero exit if:

- Any required scenario fails.
- Required logs or screenshots are missing.
- Browser diagnostics include critical console/page errors.
- Required backend/API calls fail unexpectedly.
- Any synthetic secret appears unredacted in captured artifacts.

## Implementation Tasks

### Task 0: Branch, Backlog, And Baseline

**Files:**
- Reference: `Docs/superpowers/plans/2026-06-02-repeatable-onboarding-uat-harness-implementation-plan.md`
- Backlog: `TASK-505`

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected: branch is known. Existing unrelated dirty files are noted and left untouched.

- [ ] **Step 2: Update Backlog task to implementation-ready**

Update `TASK-505` with this plan path and set status to `In Progress`.

Expected: task links the roadmap spec and this plan.

- [ ] **Step 3: Capture current targeted baselines**

Run:

```bash
source .venv/bin/activate
python -m pytest mock_openai_server/tests/test_server.py -q
```

From `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/e2e-harness-readiness.guard.test.ts
bun run e2e:onboarding -- --reporter=line
```

Expected: record pass/fail in Backlog. Do not fix unrelated failures in Task 0.

- [ ] **Step 4: Commit planning metadata only if changed**

Run:

```bash
git add backlog/tasks/task-505\ -\ Plan-repeatable-solo-onboarding-UAT-harness.md
git commit -m "chore: track onboarding uat harness implementation"
```

Expected: commit only contains Backlog metadata if any metadata changed.

### Task 1: Add Deterministic Mock-Server Scenario Controls

**Files:**
- Modify: `mock_openai_server/mock_openai/config.py`
- Modify: `mock_openai_server/mock_openai/server.py`
- Test: `mock_openai_server/tests/test_server.py`

- [ ] **Step 1: Write failing mock-server tests**

Add tests guarded by existing `RUN_MOCK_OPENAI=1` behavior:

```python
def test_chat_fail_once_then_success(monkeypatch, auth_headers):
    from ..mock_openai.config import MockConfig, ServerConfig
    from ..mock_openai.server import app, get_config_instance

    cfg = MockConfig(
        server=ServerConfig(log_requests=False),
        scenario_failures={
            "chat_completions": [
                {
                    "match": {"model": "gpt-4.1-mini"},
                    "status_code": 503,
                    "message": "UAT transient chat failure",
                    "type": "server_error",
                    "code": "uat_fail_once",
                    "times": 1,
                }
            ]
        },
    )
    app.dependency_overrides[get_config_instance] = lambda: cfg
    try:
        client = TestClient(app)
        payload = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "hello"}]}
        first = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        second = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert first.status_code == 503
        assert second.status_code == 200
    finally:
        app.dependency_overrides.clear()
```

Expected: FAIL because `MockConfig` does not parse `scenario_failures` and server handlers do not apply fail-once controls.

- [ ] **Step 2: Add config dataclasses**

In `mock_openai_server/mock_openai/config.py`, add a typed control:

```python
@dataclass
class ScenarioFailure:
    match: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 500
    message: str = "Mock scenario failure"
    error_type: str = "server_error"
    code: str = "mock_scenario_failure"
    times: int = 1
```

Add `scenario_failures: Dict[str, List[ScenarioFailure]] = field(default_factory=dict)` to `MockConfig`, parse it in `from_dict`, and preserve default behavior when absent.

When parsing JSON/YAML, accept both `type` and `error_type` as aliases for `ScenarioFailure.error_type` so fixture files can resemble OpenAI error payloads without leaking transport details into the server implementation.

Expected: static config files can declare endpoint-specific deterministic failures.

- [ ] **Step 3: Add request matching and counters**

In `mock_openai_server/mock_openai/server.py`, add module-level counters and a helper:

```python
_scenario_failure_counts: dict[tuple[str, int], int] = {}

def maybe_raise_scenario_failure(endpoint: str, request_data: dict[str, object], config: MockConfig) -> None:
    for index, failure in enumerate(config.scenario_failures.get(endpoint, [])):
        if not ResponsePattern(match=failure.match, response_file="").matches(request_data):
            continue
        key = (endpoint, index)
        used = _scenario_failure_counts.get(key, 0)
        if used >= max(0, failure.times):
            continue
        _scenario_failure_counts[key] = used + 1
        raise HTTPException(
            status_code=failure.status_code,
            detail={
                "error": {
                    "message": failure.message,
                    "type": failure.error_type,
                    "code": failure.code,
                }
            },
        )
```

Import `ResponsePattern` from `.config`. Call the helper in chat and embeddings handlers after auth validation and before default response generation. Add a reset at startup so process restarts are clean.

Expected: deterministic fail-once behavior comes from mock-server config, not Playwright.

- [ ] **Step 4: Run mock-server tests**

Run:

```bash
source .venv/bin/activate
RUN_MOCK_OPENAI=1 python -m pytest mock_openai_server/tests/test_server.py -q
```

Expected: new tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add mock_openai_server/mock_openai/config.py mock_openai_server/mock_openai/server.py mock_openai_server/tests/test_server.py
git commit -m "test: add deterministic mock openai scenario controls"
```

Expected: commit contains only mock-server control changes and tests.

### Task 2: Add Static Mock Responses And Source Fixtures

**Files:**
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/hosted-success.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/chat-fail-once.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/model-unavailable.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/chat/default.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/chat/source-summary.json`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/responses/embeddings/default.json`
- Create: `apps/tldw-frontend/e2e/fixtures/media/onboarding-uat-note.md`
- Create: `apps/tldw-frontend/public/e2e/onboarding-uat-research-note.html`

- [ ] **Step 1: Write fixture shape tests**

Add fixture validation inside `apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts` or a separate `__tests__/onboarding-uat-fixtures.test.ts`:

```ts
it("keeps onboarding UAT mock configs static and synthetic", () => {
  const config = JSON.parse(readFileSync("e2e/onboarding-uat/mock-openai/configs/hosted-success.json", "utf8"))
  expect(config.models.data ?? config.models).toBeTruthy()
  expect(JSON.stringify(config)).toContain("gpt-4.1-mini")
  expect(JSON.stringify(config)).not.toContain("sk-")
})
```

Expected: FAIL until files exist.

- [ ] **Step 2: Add hosted success config**

Use this shape:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 0,
    "cors_origins": ["*"],
    "log_requests": true,
    "simulate_errors": false,
    "error_rate": 0
  },
  "streaming": {
    "enabled": true,
    "chunk_delay_ms": 5,
    "words_per_chunk": 6
  },
  "models": [
    { "id": "gpt-4.1-mini", "object": "model", "owned_by": "openai" },
    { "id": "text-embedding-3-small", "object": "model", "owned_by": "openai" }
  ],
  "responses": {
    "chat_completions": {
      "patterns": [
        {
          "match": { "content_regex": ".*summarize|source|research note.*" },
          "response_file": "chat/source-summary.json",
          "priority": 10
        }
      ],
      "default": "chat/default.json"
    },
    "embeddings": {
      "default": "embeddings/default.json"
    }
  }
}
```

The runner will override the port at process startup; configs stay static.

- [ ] **Step 3: Add local success config**

Same model/response shape as hosted, but model ids should include local-looking options:

```json
[
  { "id": "llama3.2:3b", "object": "model", "owned_by": "ollama" },
  { "id": "local-uat-chat", "object": "model", "owned_by": "local-openai-compatible" }
]
```

Expected: local scenario can discover or manually select a local-looking model when supported.

- [ ] **Step 4: Add fail-once and model-unavailable configs**

`chat-fail-once.json` should include:

```json
{
  "scenario_failures": {
    "chat_completions": [
      {
        "match": { "model": "gpt-4.1-mini" },
        "status_code": 503,
        "message": "UAT transient chat failure",
        "type": "server_error",
        "code": "uat_fail_once",
        "times": 1
      }
    ]
  }
}
```

`model-unavailable.json` should either omit the selected model from `/v1/models` or fail matching chat requests with `404` and code `model_not_found`.

- [ ] **Step 5: Add structured source fixtures**

Markdown fixture content:

```md
# Onboarding UAT Research Note

Date: 2026-06-02

## Claims

- A short first-run wizard reduces setup abandonment.
- Deterministic evidence makes onboarding regressions easier to diagnose.

## Action Items

- Verify first chat.
- Add one source.
- Ask for a summary.
```

HTML fixture should expose the same content with headings, bullets, dates, claims, and action items.

- [ ] **Step 6: Run fixture tests**

From `apps/tldw-frontend`:

```bash
bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts
```

Expected: fixture validation passes.

- [ ] **Step 7: Commit**

Run:

```bash
git add apps/tldw-frontend/e2e/onboarding-uat/mock-openai apps/tldw-frontend/e2e/fixtures/media/onboarding-uat-note.md apps/tldw-frontend/public/e2e/onboarding-uat-research-note.html apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
git commit -m "test: add onboarding uat fixtures"
```

Expected: commit contains static fixtures and fixture tests.

### Task 3: Build Runner Profile, Process, And Artifact Helpers

**Files:**
- Create: `apps/tldw-frontend/scripts/onboarding-uat/ports.mjs`
- Create: `apps/tldw-frontend/scripts/onboarding-uat/processes.mjs`
- Create: `apps/tldw-frontend/scripts/onboarding-uat/profile.mjs`
- Create: `apps/tldw-frontend/scripts/onboarding-uat/artifacts.mjs`
- Create or extend: `apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts`

- [ ] **Step 1: Write failing helper tests**

Test these behaviors:

- `redactText("x sk-uat-mock-openai y")` returns a masked value.
- `createRunArtifacts({ preserve: false })` creates `summary.json` path under `test-results/onboarding-uat/<run-id>`.
- `createRuntimeProfile(...)` copies `tldw_Server_API/Config_Files/config.txt` into a temp profile, writes a synthetic `.env`, and never references the developer's real `.env`.
- `buildBackendEnv(...)` includes `TLDW_CONFIG_FILE`, `TLDW_ENV_FILE`, `DATABASE_URL`, `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY`, `DEFAULT_LLM_PROVIDER=openai`, `OPENAI_API_KEY=sk-uat-mock-openai`, and `OPENAI_API_BASE_URL=http://127.0.0.1:<mock-port>/v1`.

Expected: FAIL until helper files exist.

- [ ] **Step 2: Implement `ports.mjs`**

Use `node:net` to bind to port `0` on `127.0.0.1`, close, and return the selected port. Provide `reservePorts(["backend", "web", "mock"])`.

Expected: unit test can reserve distinct ports.

- [ ] **Step 3: Implement `artifacts.mjs`**

Core functions:

```js
export const SYNTHETIC_SECRETS = [
  "sk-uat-mock-openai",
  "THIS-IS-A-SECURE-KEY-123-UAT",
]

export function redactText(value) {
  let out = String(value ?? "")
  for (const secret of SYNTHETIC_SECRETS) {
    out = out.split(secret).join("[REDACTED]")
  }
  out = out.replace(/Bearer\s+sk-[A-Za-z0-9._-]+/g, "Bearer [REDACTED]")
  out = out.replace(/x-api-key:\s*[A-Za-z0-9._-]+/gi, "x-api-key: [REDACTED]")
  return out
}
```

Add `assertNoSecretLeaks(root)` that scans `.json`, `.log`, `.txt`, `.md`, and `.html` artifacts after redaction.

Expected: synthetic secret leakage fails the run.

- [ ] **Step 4: Implement `profile.mjs`**

Build this temp layout:

```text
<tmp>/tldw-onboarding-uat-<run-id>/
  Config_Files/
    config.txt
    .env
  Databases/
    users.db
    user_databases/
  uploads/
  logs/
```

Patch copied config using `node:fs` line replacement for known keys only:

- `[Setup] enable_first_time_setup = true`
- `[Setup] setup_completed = false`
- `[AuthNZ] auth_mode = single_user`
- `[AuthNZ] single_user_api_key = THIS-IS-A-SECURE-KEY-123-UAT`
- `[API] openai_model = gpt-4.1-mini`
- `[API] custom_openai_api_ip = http://127.0.0.1:<mock-port>/v1`
- `[API] custom_openai_api_model = local-uat-chat`
- `[Local-API] ollama_api_IP = http://127.0.0.1:<mock-port>/v1`
- `[Local-API] ollama_model = llama3.2:3b`
- `[TTS-Settings] USER_DB_BASE_DIR = <profile>/Databases/user_databases` when the key exists.
- `[Files] ingestion_source_allowed_roots = <repo>/apps/tldw-frontend/e2e/fixtures/media` if key exists or if setup manager allows adding it.

Write `.env` with:

```dotenv
AUTH_MODE=single_user
SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT
DEFAULT_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-uat-mock-openai
OPENAI_API_BASE_URL=http://127.0.0.1:<mock-port>/v1
DATABASE_URL=sqlite:///<profile>/Databases/users.db
USER_DB_BASE_DIR_ALLOWED_ROOTS=<profile>/Databases
TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=<profile>/Databases
TLDW_SETUP_ALLOW_REMOTE=false
```

Expected: backend reads temp config/env only.

- [ ] **Step 5: Implement `processes.mjs`**

Provide:

- `spawnLoggedProcess({ name, command, args, cwd, env, logPath })`
- `waitForHttpOk(url, { headers, timeoutMs })`
- `stopProcessTree(child)` with SIGTERM then SIGKILL timeout.

All stdout/stderr writes go through `redactText`.

Expected: runner can start services and preserve redacted logs.

- [ ] **Step 6: Run helper tests**

From `apps/tldw-frontend`:

```bash
bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts
```

Expected: tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add apps/tldw-frontend/scripts/onboarding-uat apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
git commit -m "test: add onboarding uat runner helpers"
```

Expected: commit contains runner helper modules and tests.

### Task 4: Add Runner Entrypoint And Package Script

**Files:**
- Create: `apps/tldw-frontend/scripts/onboarding-uat/run.mjs`
- Modify: `apps/tldw-frontend/package.json`
- Test: `apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts`

- [ ] **Step 1: Write failing command assembly tests**

Assert `buildCommands({ ports, profile, mockConfig })` returns commands equivalent to:

- Mock server: `python -m mock_openai.server --config <config> --host 127.0.0.1 --port <mock-port>`
- Backend: `python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port <backend-port>`
- WebUI: `bun run dev -- -p <web-port>`
- Playwright: `bunx playwright test -c e2e/onboarding-uat/playwright.config.ts`

Expected: FAIL until `run.mjs` exports command builders.

- [ ] **Step 2: Implement runner CLI**

CLI flags:

- `--scenario <id>` optional, defaults to all Tier A.
- `--viewport desktop|mobile|all` optional, defaults to `all`.
- `--mock-config <name>` optional for focused debugging.
- `--preserve-runtime` optional, defaults false.
- `--preserve-artifacts` optional, defaults true.
- `--reviewed-evidence` optional copy into `Docs/Product/WebUI/evidence/onboarding_uat/<run-id>`.

Runner flow:

1. Reserve ports.
2. Create artifact root and runtime profile.
3. Start mock server.
4. Wait for `http://127.0.0.1:<mock-port>/health`.
5. Run AuthNZ initialization if required with temp env.
6. Start backend.
7. Wait for `http://127.0.0.1:<backend-port>/api/v1/health`.
8. Start WebUI.
9. Wait for `http://localhost:<web-port>`.
10. Run Playwright with env:
    - `TLDW_ONBOARDING_UAT=1`
    - `TLDW_ONBOARDING_UAT_RUN_ID`
    - `TLDW_ONBOARDING_UAT_ARTIFACT_ROOT`
    - `TLDW_WEB_URL=http://localhost:<web-port>`
    - `TLDW_SERVER_URL=http://127.0.0.1:<backend-port>`
    - `TLDW_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT`
    - `TLDW_MOCK_OPENAI_URL=http://127.0.0.1:<mock-port>/v1`
11. Merge Playwright scenario summaries.
12. Redact and scan artifacts.
13. Stop processes.
14. Delete runtime profile unless `--preserve-runtime`.

- [ ] **Step 3: Add package script**

In `apps/tldw-frontend/package.json`:

```json
"e2e:onboarding:uat": "node scripts/onboarding-uat/run.mjs"
```

Expected: `bun run e2e:onboarding:uat -- --help` prints usage without starting services.

- [ ] **Step 4: Run runner tests**

From `apps/tldw-frontend`:

```bash
bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts
bun run e2e:onboarding:uat -- --help
```

Expected: tests pass and help exits `0`.

- [ ] **Step 5: Commit**

Run:

```bash
git add apps/tldw-frontend/scripts/onboarding-uat/run.mjs apps/tldw-frontend/package.json apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
git commit -m "test: add onboarding uat runner command"
```

Expected: commit contains runner entrypoint, tests, and package script.

### Task 5: Add Dedicated Playwright UAT Fixtures And Config

**Files:**
- Create: `apps/tldw-frontend/e2e/onboarding-uat/playwright.config.ts`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/fixtures.ts`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts`
- Create: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts`
- Modify: `apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts`

- [ ] **Step 1: Write failing guard tests**

Extend `e2e-harness-readiness.guard.test.ts`:

```ts
it("keeps onboarding UAT specs on real provider/backend behavior", () => {
  const files = [
    "e2e/onboarding-uat/setup-happy-path.spec.ts",
    "e2e/onboarding-uat/first-source.spec.ts",
    "e2e/onboarding-uat/recovery.spec.ts",
  ]
  for (const file of files) {
    const source = readSource(file)
    expect(source).not.toContain("page.route(")
    expect(source).not.toContain("seedAuth(")
    expect(source).not.toContain("__tldw_first_run_complete")
    expect(source).not.toContain("waitForTimeout(")
  }
})
```

Expected: FAIL until UAT specs exist.

- [ ] **Step 2: Add Playwright config**

`apps/tldw-frontend/e2e/onboarding-uat/playwright.config.ts`:

```ts
import { defineConfig, devices } from "@playwright/test"

const baseURL = process.env.TLDW_WEB_URL || "http://localhost:18111"

export default defineConfig({
  testDir: ".",
  timeout: 180_000,
  expect: { timeout: 30_000 },
  retries: 0,
  workers: 1,
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "uat-desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "uat-mobile", use: { ...devices["Pixel 7"] } },
  ],
})
```

No `webServer` entry: the runner owns all processes.

- [ ] **Step 3: Add first-run fixture**

`fixtures.ts` should extend base Playwright with:

- `diagnostics`: console/page error/request failure capture.
- `artifact`: helper that writes screenshots and step JSON.
- `firstRunPage`: grants clipboard permissions, clears localStorage/sessionStorage, does not seed auth, does not set `assistant_setup_dismissed`, and does not set `__tldw_first_run_complete`.

Expected: specs start from a clean browser storage profile.

- [ ] **Step 4: Add scenario definitions**

`scenarios.ts` exports Tier A ids, viewport applicability, and optional scenario env filters. Keep scenario data plain JSON-compatible.

- [ ] **Step 5: Add UI helpers**

`helpers.ts` should include:

- `openFirstRunSetup(page)`:
  - Visit `/`.
  - If `first-run-gate-overlay` is visible, click `first-run-get-started`.
  - Else navigate directly to `/setup`.
- `connectSingleUser(page, { serverUrl, apiKey })`:
  - Fill `onboarding-server-url`.
  - Fill `onboarding-api-key`.
  - Click `onboarding-connect`.
  - Wait for `onboarding-success-screen`.
- `sendFirstChat(page, prompt)`:
  - Navigate to `/chat`.
  - Fill `chat-input`.
  - Submit.
  - Wait for completed assistant article using existing `waitForStreamComplete`.
- `captureStep(page, artifact, scenarioId, stepName)`.
- `assertNoCriticalDiagnostics(diagnostics)`.

Use existing quick-ingest helpers from `../utils/journey-helpers` where possible.

- [ ] **Step 6: Run fixture/guard tests**

From `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/e2e-harness-readiness.guard.test.ts
```

Expected: guard passes after UAT files exist.

- [ ] **Step 7: Commit**

Run:

```bash
git add apps/tldw-frontend/e2e/onboarding-uat apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts
git commit -m "test: add onboarding uat playwright harness"
```

Expected: commit contains UAT Playwright config, fixtures, helpers, and guard.

### Task 6: Implement Hosted And Local Setup-To-First-Chat Specs

**Files:**
- Create: `apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts`
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts` as needed.

- [ ] **Step 1: Write hosted happy-path spec**

Spec behavior:

1. Open first-run setup.
2. Fill server URL with `process.env.TLDW_SERVER_URL`.
3. Fill API key with `process.env.TLDW_API_KEY`.
4. Connect.
5. Capture success screen.
6. Click `onboarding-success-chat` or navigate `/chat`.
7. Send `Say "onboarding UAT ready" and one short sentence.`
8. Assert assistant message contains mock response text.
9. Capture chat success.

Expected: this fails before runner/backend/mock are fully wired.

- [ ] **Step 2: Write local OpenAI-compatible path spec**

Use the current UI if it exposes provider/default model selection on success. If it does not expose a true local setup path yet, assert the current limitation explicitly in summary:

```ts
test.skip(!process.env.TLDW_ONBOARDING_UAT_LOCAL_SUPPORTED, "Current UI lacks peer local provider setup; PR4 will expand this path.")
```

If supported, use mock local config and select `local-uat-chat` or `llama3.2:3b`, then send first chat.

Expected: no route mocks. Any skip is recorded in `summary.json`.

- [ ] **Step 3: Add desktop/mobile filtering**

Hosted path runs desktop and mobile. Local path can be desktop only in PR1 unless current UI is already mobile-ready.

- [ ] **Step 4: Run focused specs through runner**

From `apps/tldw-frontend`:

```bash
bun run e2e:onboarding:uat -- --scenario hosted-openai-first-chat
```

Expected: real backend and mock server run; hosted scenario reaches a real assistant response.

- [ ] **Step 5: Commit**

Run:

```bash
git add apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts apps/tldw-frontend/e2e/onboarding-uat/helpers.ts
git commit -m "test: cover onboarding setup to first chat"
```

Expected: commit contains setup-to-chat specs and helper updates.

### Task 7: Implement First-Source UAT Specs

**Files:**
- Create: `apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts`
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts` as needed.

- [ ] **Step 1: Write paste-text scenario**

Current Quick Ingest helpers primarily support URL/file. If the UI supports paste text, add helper `queueTextForQuickIngest(dialog, text)`. If paste text is not currently exposed, record a required failure with `failure_category: "unsupported_current_ui"` and keep the scenario non-skipped so PR1 reveals the gap.

Expected: harness reports whether paste path is possible today.

- [ ] **Step 2: Write file-upload scenario**

Use:

```ts
const filePath = path.resolve(process.cwd(), "e2e/fixtures/media/onboarding-uat-note.md")
const mediaId = await ingestAndWaitForReady(page, { file: filePath }, 180_000)
expect(mediaId).toBeTruthy()
```

Capture setup success, quick ingest results, and media route.

- [ ] **Step 3: Write web URL scenario**

Use:

```ts
const webUrl = `${process.env.TLDW_WEB_URL}/e2e/onboarding-uat-research-note.html`
const mediaId = await ingestAndWaitForReady(page, { url: webUrl }, 180_000)
expect(mediaId).toBeTruthy()
```

Run desktop and mobile.

- [ ] **Step 4: Ensure post-ingest onboarding state is captured**

After successful ingest, revisit `/setup` and assert:

- `onboarding-success-screen` visible.
- `data-ingest-status="success"` or equivalent quick-ingest state visible if the store persists in current session.
- `onboarding-ingest-status` contains `Completed` when available.

Expected: evidence shows the transition from first chat to add first source.

- [ ] **Step 5: Run focused scenarios**

From `apps/tldw-frontend`:

```bash
bun run e2e:onboarding:uat -- --scenario first-source-file
bun run e2e:onboarding:uat -- --scenario first-source-web-url
```

Expected: file and web URL scenarios pass, with paste path outcome recorded.

- [ ] **Step 6: Commit**

Run:

```bash
git add apps/tldw-frontend/e2e/onboarding-uat/first-source.spec.ts apps/tldw-frontend/e2e/onboarding-uat/helpers.ts
git commit -m "test: cover onboarding first source uat paths"
```

Expected: commit contains first-source scenarios.

### Task 8: Implement Recovery Specs

**Files:**
- Create: `apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts`
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/helpers.ts` as needed.
- Modify: `apps/tldw-frontend/scripts/onboarding-uat/run.mjs` if scenario-specific mock config selection is needed.

- [ ] **Step 1: Write provider/model failure recovery scenario**

Use `model-unavailable.json` and a scenario env var that selects unavailable model `missing-uat-model`.

Required assertions:

- UI displays a connection/chat/provider failure without raw stack trace.
- User can edit provider/model or retry after runner switches to success config if the scenario requires a process restart.
- Recovery reaches first-chat success.

If current UI cannot recover inline yet, record a failing scenario. Do not mask it with route mocks.

- [ ] **Step 2: Write first-chat fail-once retry scenario**

Use `chat-fail-once.json`. Flow:

1. Complete setup.
2. Send chat prompt.
3. Assert a visible error or failed assistant state.
4. Click retry if the UI exposes it, or resubmit the same prompt if that is current behavior.
5. Assert assistant response succeeds.

Expected: deterministic fail-once comes from mock server counters.

- [ ] **Step 3: Write ingest failure retry scenario**

Use deterministic invalid local input first. Preferred options:

- A disallowed `file://` or path outside allowed roots, if surfaced through UI.
- A local URL that returns unsupported content type from the WebUI public fixture path.
- A malformed URL rejected by Quick Ingest validation.

Then retry with `onboarding-uat-note.md` or the HTML fixture and assert success.

Expected: failure and retry are both visible in artifacts.

- [ ] **Step 4: Run recovery scenarios**

From `apps/tldw-frontend`:

```bash
bun run e2e:onboarding:uat -- --scenario first-chat-fail-once-retry
bun run e2e:onboarding:uat -- --scenario ingest-failure-retry
```

Expected: retry success paths pass or current product blockers are explicit in summary.

- [ ] **Step 5: Commit**

Run:

```bash
git add apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts apps/tldw-frontend/e2e/onboarding-uat/helpers.ts apps/tldw-frontend/scripts/onboarding-uat/run.mjs
git commit -m "test: cover onboarding recovery uat paths"
```

Expected: commit contains recovery specs and runner updates.

### Task 9: Final Harness Verification And Documentation Notes

**Files:**
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts`
- Modify: `Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md` only if needed.
- Modify: `backlog/tasks/task-505 - Plan-repeatable-solo-onboarding-UAT-harness.md`

- [ ] **Step 1: Run static/unit checks**

From repo root:

```bash
source .venv/bin/activate
RUN_MOCK_OPENAI=1 python -m pytest mock_openai_server/tests/test_server.py -q
```

From `apps/tldw-frontend`:

```bash
bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts __tests__/e2e-harness-readiness.guard.test.ts
```

Expected: tests pass.

- [ ] **Step 2: Run manual/dev UAT harness**

From `apps/tldw-frontend`:

```bash
bun run e2e:onboarding:uat
```

Expected: Tier A scenarios run. If any current product blocker remains, summary must identify it with a concrete scenario id, failure category, screenshots, and logs. Do not call the PR green while required Tier A scenarios fail.

- [ ] **Step 3: Run existing onboarding E2E**

From `apps/tldw-frontend`:

```bash
bun run e2e:onboarding -- --reporter=line
```

Expected: existing onboarding evidence flow still passes.

- [ ] **Step 4: Run backend security scan for touched backend/mock Python**

From repo root:

```bash
source .venv/bin/activate
python -m bandit -r mock_openai_server/mock_openai -f json -o /tmp/bandit_onboarding_uat_mock_openai.json
```

Expected: no new high/medium findings in touched mock-server code.

- [ ] **Step 5: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Dirty worktree contains only intended PR files plus unrelated pre-existing files.

- [ ] **Step 6: Update Backlog**

Update `TASK-505` with:

- Implemented files.
- UAT artifact root from the successful run.
- Verification commands and outcomes.
- Known skips or product blockers, if any.

Expected: Backlog task has final implementation notes for PR1.

- [ ] **Step 7: Final commit**

Run:

```bash
git add apps/tldw-frontend/package.json apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md "backlog/tasks/task-505 - Plan-repeatable-solo-onboarding-UAT-harness.md"
git commit -m "test: document onboarding uat harness verification"
```

Expected: final commit contains only docs/script/package/Backlog updates not already committed.

## Review Notes For Implementers

- Do not use `page.route` to fake provider validation, chat, model listing, or ingestion behavior in UAT specs.
- Do not seed `__tldw_first_run_complete`, `assistant_setup_dismissed`, or normal `tldwConfig` in the UAT fixture before the setup flow.
- Do not read or write the developer's real `.env`, `config.txt`, `Databases`, or uploads.
- Do not log raw `Authorization`, `X-API-KEY`, `.env`, or config secret values.
- Do not add blind sleeps. Use explicit Playwright locators, backend health polling, or existing readiness helpers.
- Keep the manual UAT command out of blocking CI until the team promotes it.
- If current product behavior blocks a Tier A scenario, preserve the failing evidence and summarize the blocker. Do not weaken the scenario by mocking the product path.

## PR Review Checklist

- [ ] Runner starts and stops mock server, backend, WebUI, and Playwright from one command.
- [ ] Runtime profile is isolated and cleaned up by default.
- [ ] Artifacts are complete enough for debugging without immediate rerun.
- [ ] `summary.json` is the pass/fail source of truth.
- [ ] Hosted first-chat path requires a real assistant response.
- [ ] First-source file and web URL paths use real Quick Ingest and backend ingestion.
- [ ] Recovery scenarios are deterministic.
- [ ] Synthetic secrets are redacted and leak scan is enforced.
- [ ] Existing onboarding E2E still works.
- [ ] Backlog task records verification and any current product blockers.
