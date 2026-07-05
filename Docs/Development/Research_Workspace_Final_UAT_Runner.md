# Research Workspace Final UAT Runner

Use this runner when re-certifying the Research Workspace UAT matrix from a
clean browser/session state. It is designed for local and Codex macOS sessions
where browser launch failures must be separated from product failures.

## Prerequisites

- Backend API is running and healthy at `http://127.0.0.1:8000`, or set
  `NEXT_PUBLIC_API_URL`/`TLDW_E2E_SERVER_URL` to the active backend.
- Local network access to `127.0.0.1` is allowed for the Codex session.
- The selected Playwright browser channel can launch in the host session.
- If the WebUI is autostarted, bind it to localhost explicitly. The runner
  defaults to `bun run dev -- -H 127.0.0.1 -p 8080` instead of the broader
  bind that previously failed with `EPERM`.
- Strict sandbox diagnostics require a backend started with the sandbox route
  enabled in `config.txt` or a temporary `TLDW_CONFIG_FILE`. Normal runtime
  does not use `ROUTES_ENABLE=sandbox`; that override is test-only.

Add this to a copied backend config used through `TLDW_CONFIG_FILE`:

```ini
[API-Routes]
enable = sandbox

[Sandbox]
enable_execution = true
```

For real Docker evidence, start the backend with fake execution disabled and a
reachable Docker daemon:

```bash
TLDW_CONFIG_FILE=/tmp/research-workspace-uat-sandbox-config.txt \
SANDBOX_ENABLE_EXECUTION=1 \
TLDW_SANDBOX_DOCKER_FAKE_EXEC=0 \
python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000
```

## Command

From `apps/tldw-frontend`:

```bash
bun run e2e:research-workspace:uat
```

The command runs these focused specs by default:

```text
e2e/workflows/research-workspace.spec.ts
e2e/workflows/research-workspace.real-backend.spec.ts
```

Useful overrides:

```bash
TLDW_WEB_AUTOSTART=false \
TLDW_WEB_URL=http://localhost:8080 \
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN=1 \
TLDW_E2E_EXPECT_SANDBOX_RUN_PHASE=completed \
bun run e2e:research-workspace:uat -- --no-autostart
```

```bash
bun run e2e:research-workspace:uat -- \
  --spec e2e/workflows/research-workspace.spec.ts \
  --grep "opens and closes workspace search"
```

## Evidence

The runner writes two files by default:

- `test-results/research-workspace-final-uat-report.json`
- `test-results/research-workspace-final-uat-evidence.json`

The evidence file has a `status` field:

- `passed`: the focused specs executed without skips, flaky tests, or
  unexpected failures.
- `product_failed`: tests executed but assertions, skips, or flakiness mean the
  product is not certified.
- `environment_blocked`: the browser/server/test harness did not reach product
  assertions. This is not a product pass.

Known environment-blocked signatures include macOS browser-launch Mach port
errors such as `bootstrap_check_in ... Permission denied (1100)` and localhost
bind failures such as `EPERM ... 0.0.0.0:8080`.

## Codex In-App Browser Fallback

If standalone Playwright reports `environment_blocked` in a sandboxed Codex
macOS session, use the in-app browser/CDP surface for the final UAT pass and
record:

- Browser URL, backend URL, auth mode, seeded data, and login/key state.
- Screenshots for each persona state.
- Console errors, page errors, failed network requests, and timing notes.
- Which workflows were verified, blocked, or skipped.

Do not mark blocked or skipped standalone Playwright runs as product passes in
the UAT matrix. Record them as environment-blocked and link the in-app CDP
evidence or the follow-up task that removes the blocker.
