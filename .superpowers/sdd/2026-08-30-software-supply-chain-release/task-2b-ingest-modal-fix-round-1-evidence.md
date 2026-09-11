# Task 2B fix round 1 — terminal close evidence

## Scope and root cause

The Task 2B terminal helper path now calls 'dismissQuickIngest(page, { terminal: true })'. Before this round, that call reached the general 'clickQuickIngestCloseControl()' fallback and pressed Escape if terminal Done and Ant's modal-close button were both absent.

The narrow repair adds the 'terminal' option to 'DismissQuickIngestOptions':

- a visible 'Close the ingest wizard' / Done button remains the first supported action;
- a visible '.ant-modal-close' remains the second supported action;
- in terminal mode only, absence of both now throws 'Quick Ingest terminal dialog is missing a visible Done or modal close control; refusing Escape fallback';
- non-terminal callers retain the existing Escape and processing/minimize behavior.

The React owner remains unchanged: 'apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1819' passes the terminal Results step's 'onClose', and ':1852' keeps the Ant modal open only while 'open && !state.isMinimized'. The route owner remains 'apps/packages/ui/src/components/Sidepanel/Chat/form.tsx:4420-4428': 'onClose' calls 'hideQuickIngestSession()', 'setIngestOpen(false)', and resets auto-process state.

## Behavior-level RED and GREEN

The new real-browser boundary regression is 'apps/tldw-frontend/e2e/quick-ingest-terminal-close.spec.ts'.

It uses Playwright's actual browser page and the real exported helper:

1. a terminal dialog with Done must hide without an Escape key event;
2. a terminal dialog with neither supported control must reject clearly and emit no Escape key event.

RED command:

~~~bash
cd apps/tldw-frontend
TLDW_WEB_AUTOSTART=false bunx playwright test e2e/quick-ingest-terminal-close.spec.ts --project=chromium --workers=1 --reporter=line
~~~

RED result before the terminal-only branch: '1 failed, 1 passed (866ms)'. The missing-control case failed for the intended reason: the promise resolved instead of rejecting, proving the generic helper had pressed Escape.

GREEN command: the same command.

GREEN result after the minimal branch: '2 passed (846ms)'; a type-only test correction was then rerun as '2 passed (853ms)'.

## Exact owned live graph invocation

The final live graph used these isolated owned endpoints:

- Redis: '127.0.0.1:62457'
- deterministic mock provider: '127.0.0.1:18091'
- API: '127.0.0.1:62458'
- WebUI: '127.0.0.1:62459'
- Python: '/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python'
- mock fixture: 'apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json'

From '/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design', the shell-owned services were invoked as:

~~~bash
redis-server --port 62457 --save '' --appendonly no &
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/mock_openai_server \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m mock_openai.server \
  --config apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json \
  --host 127.0.0.1 --port 18091 &
AUTH_MODE=single_user SINGLE_USER_API_KEY=test-api-key-for-e2e-testing-12345 \
REDIS_URL=redis://127.0.0.1:62457/0 \
TLDW_CONFIG_FILE=tldw_Server_API/Config_Files/e2e-critical-config.txt \
DEFAULT_LLM_PROVIDER=openai OPENAI_API_KEY=sk-uat-mock-openai \
OPENAI_API_BASE_URL=http://127.0.0.1:18091/v1 \
CUSTOM_OPENAI_API_IP=http://127.0.0.1:18091/v1 \
CUSTOM_OPENAI_API_KEY=sk-uat-mock-openai CUSTOM_OPENAI_API_MODEL=local-uat-chat \
LLM_PROVIDER_READINESS_PROBE_ENDPOINTS=1 WORKFLOWS_EGRESS_ALLOWED_PORTS='*' \
WORKFLOWS_EGRESS_BLOCK_PRIVATE=false WORKFLOWS_EGRESS_ALLOWLIST=127.0.0.1,localhost \
USER_DB_BASE_DIR=<temporary-owned-directory> PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m uvicorn \
  tldw_Server_API.app.main:app --host 127.0.0.1 --port 62458 &
cd apps/tldw-frontend
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:62458 \
  bun run dev:webpack -- -p 62459 &
TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:62459 \
TLDW_SERVER_URL=http://127.0.0.1:62458 TLDW_E2E_SERVER_URL=http://127.0.0.1:62458 \
TLDW_API_KEY=test-api-key-for-e2e-testing-12345 \
TLDW_E2E_API_KEY=test-api-key-for-e2e-testing-12345 \
TLDW_MOCK_OPENAI_URL=http://127.0.0.1:18091/v1 \
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:62458 \
  bunx playwright test e2e/workflows/journeys/ingest-evaluate-review.spec.ts \
  e2e/workflows/journeys/ingest-search-chat.spec.ts \
  --project=journeys --workers=1 --reporter=line --trace=on
~~~

The shell health-gated each listener before invoking Playwright and terminated the owned services after result capture. The captured live transcript is '/tmp/task2b-round1-live-playwright.log' on the task host.

Final live result: '2 passed, 1 failed (56.2s)'.

- The terminal helper-return regression passed.
- Ingest → Evaluate → Review passed.
- Ingest → Search → Chat reached the later unchanged '/playwright/i' assertion, then failed because the committed mock fixture has no Playwright content pattern and returns its generic default response. This is a distinct deterministic-response contract and is not changed in this package.

## Other verification

~~~bash
cd apps/tldw-frontend
bun run typecheck
git diff --check
~~~

- 'git diff --check': exit '0'.
- 'bun run typecheck': nonzero only on the pre-existing unrelated dirty files 'DocumentationPage.tsx', 'scripts/__tests__/skills-certification-profile.test.ts', and 'scripts/__tests__/skills-certification-runner.test.ts'. The Task 2B helper and terminal-close regression are absent from compiler output.
- No Python file changed; Bandit is not applicable.
