## Task 17: Document, Exercise, Audit, And Prepare Rollout

**Primary files:**
- Create: `Docs/User_Guides/WebUI_Extension/Presentation_Studio.md`
- Create: `Docs/Deployment/Standalone_HTML_Presentations.md`
- Modify: `Docs/API/Slides.md`
- Modify: `Docs/Design/Presentations.md`
- Modify: `Docs/Product/Slides_Infographics_Workproducts_PRD.md`
- Modify: `Docs/MCP/Unified/Modules.md`
- Modify: `tldw_Server_API/app/core/Slides/README.md`
- Modify: `CHANGELOG.md`
- Modify: `Docs/RELEASE_NOTES.md`
- Modify: `apps/tldw-frontend/playwright.config.ts`
- Create: `apps/tldw-frontend/e2e/workflows/presentation-studio-standalone-html.spec.ts`
- Create: `apps/tldw-frontend/e2e/workflows/presentation-studio-standalone-html.security.spec.ts`
- Create: `tldw_Server_API/tests/Slides/test_standalone_html_integration.py`
- Create: `.superpowers/sdd/2026-07-15-standalone-html-presentations-implementation-plan/task-17-report.md`

Task BASE is `e0f2bbb3f25713fec3070409c42a1809aa32340d`.

IMPECCABLE_PREFLIGHT: context=pass product=pass command_reference=pass shape=pass image_gate=skipped:no imagery belongs in integration, security, or rollout documentation mutation=open

- [x] **Step 1: Complete read-only preflight and bind the release-test architecture**

Read `PRODUCT.md`, `DESIGN.md`, the approved Task17 plan, the standalone HTML design, repository instructions, and the required execution-plan, TDD, debugging, verification, and Impeccable skill references. Inspect real analogues for the Slides owner database and job worker, HTTP API composition, all five source adapters, Presentation Studio routing and recovery, browser fixture/network instrumentation, extension source-bearing tripwires, Playwright project selection, and the current Slides/API/PRD/MCP/core documentation.

The backend integration uses the real per-owner `SlidesDatabase`, real `JobManager`, real `StandaloneHtmlGenerationService`, real validator pool, real FastAPI router, and real `process_standalone_html_generation_job`. Mock only the source-adapter and provider boundaries. Submit through HTTP for prompt, chat, media, notes, and RAG; inspect the owner-scoped Jobs envelope; acquire and drive the real worker; finalize and poll; then prove replay, provider-call count, legacy filtering, opted-in list/search/detail, version/save/export, reopen, and default-off readability.

The browser contract is split deliberately. Chromium runs the full deterministic workflow specification and the security specification. Firefox and WebKit run only the security specification. A separate localhost WebUI to `127.0.0.1` API protocol server exercises real browser CORS, preflight, response headers, attachment behavior, and direct-source transport; route fulfillment alone is not evidence for those claims.

- [x] **Step 2: Land the complete test-only patch and capture genuine RED**

Create all three planned tests before changing Playwright configuration or rollout documentation. Browser observability is installed on the context before navigation and covers requests, new pages, service workers, workers, and relevant execution/navigation/source sinks. Correlate assertions with one unique validator-accepted source sentinel instead of globally replacing native behavior. The outline-worker hang probe targets only the fixed outline worker URL, and Monaco security assertions first prove a real `.monaco-editor` exists; fallback behavior is tested separately.

The strict attachment assertion permits exactly one application-owned `Blob` with `application/octet-stream`, assigned only to the temporary fixed-name anchor, never opened in a new context, removed, and revoked within one second. Use a corrupt mock-detail URL payload separately from the valid global-mutation sentinel to prove inert response handling.

Before production/config/docs edits, run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Slides/test_standalone_html_integration.py
cd apps/tldw-frontend
bunx playwright test e2e/workflows/presentation-studio-standalone-html.spec.ts --project=chromium --reporter=line
bunx playwright test e2e/workflows/presentation-studio-standalone-html.security.spec.ts --project=chromium --project=standalone-html-firefox --project=standalone-html-webkit --reporter=line
```

Also capture a Playwright `--list` RED that proves the missing Firefox/WebKit projects and, after configuration, proves Firefox/WebKit select only the security spec while Chromium selects both specs. Every generated idempotency key and project identifier is unique per browser project and run entropy. Security projects use `retries: 0`.

- [x] **Step 3: Make the minimal Playwright and documentation changes**

Add only two security-scoped browser projects. Document the exact API/capability/error/strong-ETag/attachment contracts without executable sample payloads. Preserve existing weak-ETag and synchronous-generation statements explicitly for legacy structured routes only. Never describe executable standalone HTML as sanitized or safe.

Document default-off enablement, closed adapter IDs, tuple allowlisting, key source/rotation, egress kill, worker/reconciler health, fixed limits, schema-v2 backup-first migration and old-binary incompatibility, guarded MCP WebSocket behavior, extension metadata handoff, explicit save/recovery, no preview/execution, provider at-least-once retry semantics, rollback/drain/readability, source-free logging, per-user isolation, and the required human-written Change summary. Amend the PRD only with the narrow opaque-text exception; arbitrary execution remains prohibited.

- [x] **Step 4: Run complete backend, frontend, browser, extension, static, and security gates**

Run the complete matrices in the approved Task17 plan using the repository virtual environment. Do not skip a browser-gated security case. If a browser engine or PostgreSQL fixture is unavailable, record the fixture/tool-reported failure or skip exactly and do not call it passing. Run Bandit on all implementation Python paths changed since the approved design base, inspect the JSON, and run diff/static source-sink checks.

- [ ] **Step 5: Self-review, report, explicit staging, controller audit, and commit**

Complete `task-17-report.md` with preflight analogues, complete RED evidence, exact GREEN counts, browser/project selection, PostgreSQL/browser dispositions, OpenAPI/type/lint/Bandit/static evidence, product/accessibility review, scope deviations, and residual limitations. Stage only approved paths and the three Task17 SDD artifacts. Preserve the protected `antd` symlink and two Watchlist templates. Do not call or edit Backlog, install dependencies, push, or commit before the controller accepts the staged audit.

Browser RED exposed lifecycle, routing-import, keyboard-tab, retained-form,
worker-identity, and security-observability defects in existing Task15 code.
The controller authorized the smallest directly associated runtime and unit-test
paths for those proved defects. Those deviations are enumerated in the global
constraints and report; they do not add a preview or execution surface.

Commit subject after approval:

```text
fix(slides): harden standalone HTML rollout (TASK-12115)
```
