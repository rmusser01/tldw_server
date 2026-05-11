# Persona Chat Judge Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the optional calibrated Persona Chat judge contract and add deterministic contract fixtures without executing a live judge or changing runtime Persona Chat behavior.

**Architecture:** Keep this as a contract-first docs/test slice. A review artifact defines the judge input, output, calibration, privacy, and offline-only rules; a small fixture file provides positive and negative calibration examples; pytest validates the fixture and contract against the existing `PC-*` taxonomy. No endpoint, worker, recipe execution, or runtime chat path changes in this PR.

**Tech Stack:** Markdown contract docs, JSON fixture data, Python 3.11, pytest.

---

### Task 1: Contract Fixture Guard

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py`
- Read: `Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md`
- Future create: `tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json`

- [x] **Step 1: Write the failing fixture contract test**

Add a pytest file that expects `persona_chat_judge_contract_cases.json` to exist and asserts:
- top-level `schema_version` is `persona-chat-judge-contract/v1`
- top-level `offline_only` is `true`
- at least one `pass` and one `fail` case exist
- case ids use `PC-JUDGE-###` and source case ids use `PC-CASE-###`
- assistant kind is `persona`, memory mode is `read_only` or `read_write`, and no case contains local paths, API keys, or private-data markers
- candidate output verdicts, scores, flags, rationale, and evidence fit the contract
- every `expected_flags` item exists in the existing taxonomy failure-label table

- [x] **Step 2: Run the test to verify RED**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py -q`

Expected: fail because the fixture artifact does not exist.

### Task 2: Contract Artifact And Documentation

**Files:**
- Create: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md`
- Create: `tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json`
- Modify: `Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md`

- [x] **Step 1: Create the judge contract review artifact**

Document:
- V1 purpose and non-goals
- judge input envelope
- judge output envelope
- calibration fixture rules
- privacy/redaction requirements
- failure-label mapping to `PC-*` taxonomy
- offline-only behavior and no runtime gating
- future executable-harness prerequisites

- [x] **Step 2: Add minimal calibration fixture cases**

Create two synthetic, redaction-safe fixture cases:
- positive case: persona-consistent prompt-reveal refusal, expected `pass`, no flags
- negative case: read-only memory promise, expected `fail`, `PC-MEM-003`

Keep fixture text synthetic and bounded. Link each case to a source `PC-CASE-###` from the deterministic quality corpus.

- [x] **Step 3: Link the contract from the Stage 2 follow-up doc**

Add a short update to `PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md` pointing to the new contract and reiterating that judge execution remains deferred.

### Task 3: Verification And Packaging

**Files:**
- Modify: `backlog/tasks/task-241.1 - Define-Persona-Chat-judge-evaluation-contract.md`

- [x] **Step 1: Run focused verification**

Run:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py -q`
- `rg -n "TO[D]O|TB[D]|FIX[M]E|PLACE[H]OLDER|\\?\\?" Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md`
- `git diff --check`

Bandit is run on the touched pytest contract validator.

- [x] **Step 2: Update Backlog task**

Record verification, mark acceptance criteria complete, document Bandit applicability, and add a final summary.

- [x] **Step 3: Commit**

Run:
```bash
git add Docs/superpowers/plans/2026-05-11-persona-chat-judge-contract.md Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py "backlog/tasks/task-241.1 - Define-Persona-Chat-judge-evaluation-contract.md"
git commit -m "Define Persona Chat judge contract"
```
