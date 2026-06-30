# Persona Chat Telemetry Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize Persona Chat telemetry labels so persona-backed chat is distinguishable from character chat and PC-TEL-001 has deterministic regression coverage.

**Architecture:** Keep telemetry inside the existing `/chat/completions` persona exemplar hook. Extend the metric label set with redaction-safe assistant identity fields and keep `character_id` for existing character-chat compatibility. Expose label grouping through the existing evaluation metrics summary instead of adding a new telemetry backend.

**Tech Stack:** FastAPI chat endpoint, in-process MetricsRegistry, pytest unit tests, Backlog.md task TASK-250.

---

### Task 1: Add PC-TEL-001 Regression Coverage

**Files:**
- Modify: `tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py`
- Modify: `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`
- Reference: `tldw_Server_API/tests/fixtures/persona_chat_quality_cases.json`
- Reference: `Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md`

- [x] **Step 1: Write the failing test**

Add a test that loads `PC-CASE-019`, records persona telemetry with `assistant_kind=persona`, `assistant_id=garden-telemetry`, and `character_id=none`, and asserts the summary exposes persona samples by assistant kind and assistant id.

- [x] **Step 2: Run test to verify it fails**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py::test_persona_telemetry_metrics_summary_groups_persona_backed_labels tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_persona_backed_chat_records_telemetry_with_persona_identity_labels -q`

Result: FAIL. The summary test raised `KeyError: 'samples_by_assistant_kind'`; the endpoint regression also failed before implementation.

### Task 2: Normalize Hook Labels

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/core/Evaluations/persona_telemetry_metrics.py`
- Modify: `tldw_Server_API/app/core/Metrics/metrics_manager.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`

- [x] **Step 1: Extend metrics summary grouping**

Add a small helper in `persona_telemetry_metrics.py` that groups `chat_persona_ioo_ratio` samples by `assistant_kind` and `assistant_id`, defaulting missing labels to `unknown` and `none`.

- [x] **Step 2: Extend chat hook label inputs**

Extend `_record_persona_telemetry_hooks` with `assistant_kind` and `assistant_id` parameters. Add those labels to every histogram and counter while preserving `character_id`.

- [x] **Step 3: Avoid persona alert-window collapse**

Include assistant kind and assistant id in the sustained-alert `window_key` so persona-backed chats with `character_id=none` do not share one alert window.

- [x] **Step 4: Pass persona identity from both call sites**

Pass `assistant_kind=assistant_context.get("assistant_kind")` and `assistant_id=persona_assistant_id` from both streaming and non-streaming persona telemetry call sites.

- [x] **Step 5: Run focused tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py::test_persona_telemetry_metrics_summary_groups_persona_backed_labels tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_persona_backed_chat_records_telemetry_with_persona_identity_labels -q`

Result: PASS, 2 passed.

### Task 3: Verify and Finalize

**Files:**
- Modify: `backlog/tasks/task-250 - Normalize-Persona-Chat-telemetry-labels.md`

- [x] **Step 1: Run focused chat/persona tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_telemetry_metrics_summary.py tldw_Server_API/tests/Persona/test_persona_chat_quality_fixtures.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_persona_backed_chat_records_telemetry_with_persona_identity_labels -q`

Result: PASS, 12 passed. Also ran adjacent existing test `test_persona_backed_chat_appends_persona_exemplar_guidance_in_runtime_path` alone: PASS. A broader run of the entire chat integration file timed out in TestClient app-lifecycle setup after earlier tests passed, so this plan records the narrower deterministic regression verification.

- [x] **Step 2: Run Bandit on touched backend files**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Evaluations/persona_telemetry_metrics.py tldw_Server_API/app/core/Metrics/metrics_manager.py -f json -o /tmp/bandit_persona_chat_telemetry_labels.json`

Result: PASS, 0 findings.

- [x] **Step 3: Run diff hygiene**

Run: `git diff --check`

Result: PASS, no output.

- [x] **Step 4: Update Backlog.md task**

Record changed files, verification output, and any residual gaps in TASK-250.

- [ ] **Step 5: Commit**

Commit message: `fix: normalize persona chat telemetry labels`
