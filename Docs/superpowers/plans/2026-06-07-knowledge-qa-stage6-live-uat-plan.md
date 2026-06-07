# Knowledge QA Stage 6 Live UAT Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic live-backend Knowledge QA fixtures, WebUI and extension live UAT tests, console/network assertions, and release-gate documentation.

**Architecture:** Seed a small, known personal-library fixture and run WebUI plus extension workflows against the running backend. Mocking is allowed only for the deliberate degraded/uncited fixture; cited, no-results, and scoped paths must exercise seeded backend data.

**Tech Stack:** Python fixture helpers, FastAPI backend, Playwright, WebUI E2E, WXT extension E2E, Markdown release checklist.

**Backlog Task:** TASK-2279.8

---

## Boundaries

- Depends on core trust and evidence stages for full assertions.
- Do not call external web providers in default UAT.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Create: `tldw_Server_API/tests/RAG/knowledge_qa_uat_fixtures.py`
- Create: `Helper_Scripts/seed_knowledge_qa_uat.py`
- Create: `apps/tldw-frontend/e2e/fixtures/knowledge-qa-live.ts`
- Create: `apps/tldw-frontend/e2e/ux-audit/knowledge-qa-live-backend.spec.ts`
- Create: `apps/extension/tests/e2e/knowledge-qa-live-backend.spec.ts`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/KnowledgeQAPage.ts`
- Modify: `apps/extension/tests/e2e/utils/real-server.ts`
- Modify: `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md`
- Modify: `Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md`

## Task 1: Build Seeded Fixture Data

- [ ] **Step 1: Write fixture contract**

Create `knowledge_qa_uat_fixtures.py` with constants:

```python
KNOWN_CITED_QUERY = "What does the grounded QA checklist require?"
KNOWN_CITED_ANSWER_PHRASE = "Grounded answers cite visible evidence"
SCOPED_EXCLUDED_PHRASE = "Excluded distractor should not appear"
NO_MATCH_QUERY = "What does the library say about nonexistent basalt telemetry?"
```

- [ ] **Step 2: Add seeding helper**

Create `Helper_Scripts/seed_knowledge_qa_uat.py` that:

- activates only when explicitly run
- creates one source with the cited answer phrase
- creates one distractor source with a tempting excluded phrase
- creates one note source for exact-note scoped search
- records created ids to a JSON fixture manifest

- [ ] **Step 3: Add smoke test for fixture constants**

Add a small Pytest in `tldw_Server_API/tests/RAG/test_knowledge_qa_uat_fixtures.py` to ensure fixture strings are distinct and deterministic.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_knowledge_qa_uat_fixtures.py -v
```

Expected: pass after helper exists.

## Task 2: Add WebUI Live Backend Test

- [ ] **Step 1: Create live fixture helper**

Create `apps/tldw-frontend/e2e/fixtures/knowledge-qa-live.ts` to read the fixture manifest path from an environment variable such as `TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST`.

- [ ] **Step 2: Write failing WebUI live test**

Create `knowledge-qa-live-backend.spec.ts` with tests:

- backend unavailable recovery
- known cited answer with inspectable evidence
- known no-results query
- scoped search excludes distractor
- export preserves trust labels

Run:

```bash
cd apps/tldw-frontend
TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:3000 TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST=/tmp/knowledge-qa-uat.json bunx playwright test e2e/ux-audit/knowledge-qa-live-backend.spec.ts --project=chromium --reporter=line
```

Expected before seeding/running backend: fail with clear precondition error, not a false pass.

## Task 3: Add Extension Live Backend Test

- [ ] **Step 1: Add harness health precheck**

In `apps/extension/tests/e2e/knowledge-qa-live-backend.spec.ts`, first assert the built extension launches `options.html#/knowledge`.

- [ ] **Step 2: Add live extension workflows**

Mirror WebUI live workflows where feasible:

- setup required
- ready search
- cited answer
- no results
- scoped source search
- sync failure visible recovery

Run:

```bash
cd apps/extension
TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST=/tmp/knowledge-qa-uat.json bunx playwright test tests/e2e/knowledge-qa-live-backend.spec.ts --project=chromium-extension --reporter=line
```

Expected: browser tests execute, or WXT blocker is recorded as release-blocking.

## Task 4: Update Release Docs

- [ ] **Step 1: Update UAT checklist**

Add commands, fixture preconditions, expected screenshots/traces, and skip/blocker rules to `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md`.

- [ ] **Step 2: Update user guide if visible behavior changed**

Update `Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md` for trust labels, web fallback origin labels, and unsupported export behavior.

## Task 5: Verify

- [ ] **Step 1: Run backend fixture tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_knowledge_qa_uat_fixtures.py -v
python -m bandit -r Helper_Scripts/seed_knowledge_qa_uat.py tldw_Server_API/tests/RAG/knowledge_qa_uat_fixtures.py -f json -o /tmp/bandit_knowledge_qa_uat.json
```

- [ ] **Step 2: Run WebUI live test against a launched backend**

Record exact backend command, frontend URL, fixture manifest, console errors, and network failures.

- [ ] **Step 3: Run extension live test or record blocker**

If WXT build stalls, capture the command, timeout, and owner in `TASK-2279.8`.

- [ ] **Step 4: Commit**

```bash
git add Helper_Scripts/seed_knowledge_qa_uat.py tldw_Server_API/tests/RAG apps/tldw-frontend/e2e apps/extension/tests/e2e Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md "backlog/tasks/task-2279.8 - Add-Knowledge-QA-live-UAT-fixtures-and-release-gates.md"
git commit -m "test: add knowledge qa live uat gates"
```
