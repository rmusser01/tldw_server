# Persona Chat Quality Fixtures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add redaction-safe deterministic fixture coverage for ordinary Persona Chat quality cases from issue #1552 and the merged taxonomy artifact.

**Architecture:** Keep this as a tests-and-fixtures slice. Store the taxonomy-derived fixture records as a static JSON artifact, validate its schema and label coverage with Python tests, then tie high-confidence cases into existing backend and frontend deterministic test surfaces. Do not change runtime behavior unless a test exposes a real bug.

**Tech Stack:** Python 3.11, pytest, Vitest, TypeScript, existing Persona Chat backend/frontend helpers.

---

### Task 1: Fixture Artifact And Schema Guard

**Files:**
- Create: `tldw_Server_API/tests/fixtures/persona_chat_quality_cases.json`
- Create: `tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py`
- Read: `Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md`

- [x] **Step 1: Write the failing fixture schema test**

Add a pytest test that loads `persona_chat_quality_cases.json` and asserts:
- exactly 20 cases with uppercase `PC-CASE-###` ids
- each case has `assistant_kind`, `assistant_id`, `persona_memory_mode`, `input`, `expected_context`, `response_observation`, `labels`, and `expected_evidence`
- every referenced label exists in the taxonomy Failure Labels table
- no fixture text includes pathlike local machine values, API keys, secrets, or raw-private markers
- the first-pass deterministic labels from issue #1552 are represented

- [x] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py -q`

Expected: FAIL because the fixture artifact does not exist yet.

- [x] **Step 3: Add the minimal fixture artifact**

Create 20 redaction-safe synthetic case records derived from the merged taxonomy. Use only synthetic assistant ids, synthetic input text, expected context flags, selected/rejected exemplar ids, labels, and evidence keys.

- [x] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py -q`

Expected: PASS.

### Task 2: Backend Deterministic Contract Guards

**Files:**
- Modify: `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`
- Modify: `tldw_Server_API/tests/Persona/test_exemplar_retrieval.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`
- Test: `tldw_Server_API/tests/Persona/test_exemplar_retrieval.py`

- [x] **Step 1: Write failing backend tests for missing trace links**

Add assertions that selected runtime prompt-preview/parity and exemplar retrieval tests reference the fixture case ids or labels they prove.

- [x] **Step 2: Run focused backend tests to verify failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_exemplar_retrieval.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q`

Expected: FAIL until fixture metadata helpers are added.

- [x] **Step 3: Implement minimal fixture helper usage**

Import/load the fixture artifact in tests only. Add small assertions that prove current deterministic contracts correspond to `PC-ID-001`, `PC-ID-002`, `PC-EX-002`, `PC-EX-003`, `PC-EX-004`, `PC-EX-005`, `PC-PREV-001`, `PC-MEM-001`, `PC-MEM-002`, and `PC-TRACE-001`.

- [x] **Step 4: Run focused backend tests to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_exemplar_retrieval.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q`

Expected: PASS.

### Task 3: Frontend Deterministic Contract Guards

**Files:**
- Modify: `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts`
- Modify: `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`

- [x] **Step 1: Write failing frontend tests for switch/reset and restore fallback contracts**

Add tests for:
- switching from a character-backed server chat to a persona resets stale assistant metadata before creating the persona chat
- persona restore metadata preserves identity and memory mode
- persona profile fallback keeps assistant kind/id while using generic Persona presentation

- [x] **Step 2: Run focused Vitest tests to verify failure**

Run from `apps/packages/ui`: `bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts src/hooks/__tests__/useServerChatLoader.test.ts --maxWorkers=1`

Expected: FAIL until the missing test helpers or assertions are in place.

- [x] **Step 3: Implement minimal test changes**

Prefer adding tests around exported pure helpers (`ensurePersonaServerChat`, `resolveServerChatAssistantIdentity`, `applyAssistantPresentationToMessages`, and `reportDeferredAssistantPresentationError`) rather than mounting the full chat shell.

- [x] **Step 4: Run focused Vitest tests to verify pass**

Run from `apps/packages/ui`: `bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts src/hooks/__tests__/useServerChatLoader.test.ts --maxWorkers=1`

Expected: PASS.

### Task 4: Verification And Packaging

**Files:**
- Modify: `backlog/tasks/task-247 - Add-deterministic-Persona-Chat-quality-fixtures.md`

- [x] **Step 1: Run combined verification**

Run:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py tldw_Server_API/tests/Persona/test_exemplar_retrieval.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q`
- `bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts src/hooks/__tests__/useServerChatLoader.test.ts --maxWorkers=1` from `apps/packages/ui`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py tldw_Server_API/tests/Persona/test_exemplar_retrieval.py -f json -o /tmp/bandit_persona_chat_quality_fixtures.json`
- `git diff --check`

- [x] **Step 2: Update Backlog task**

Record verification results, known skips/blockers, checked acceptance criteria, and final summary in TASK-247.

- [x] **Step 3: Commit**

Run:
```bash
git add Docs/superpowers/plans/2026-05-10-persona-chat-quality-fixtures.md tldw_Server_API/tests/fixtures/persona_chat_quality_cases.json tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py tldw_Server_API/tests/Persona/test_exemplar_retrieval.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts "backlog/tasks/task-247 - Add-deterministic-Persona-Chat-quality-fixtures.md"
git commit -m "Add deterministic Persona Chat quality fixtures"
```
