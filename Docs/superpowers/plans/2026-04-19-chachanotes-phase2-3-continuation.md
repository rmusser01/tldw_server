# ChaChaNotes Phase 2.3 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the WIP `NoteStore` and `KeywordStore` extractions, finish the planned `PersonaStateStore` extraction, and wire `CharactersRAGDB` facade delegation for the extracted ChaChaNotes stores.

**Architecture:** Treat this branch as a bounded continuation of issue `#1115` under roadmap checkpoint `#1116`. First restore trustworthy store-level contract coverage by replacing weak tests with explicit assertions against the established `CharactersRAGDB` behavior. Then extract the remaining persona-state persistence surface into a focused store class and finally wire facade delegation in `ChaChaNotes_DB.py` so callers keep using the existing public API while the monolith shrinks.

**Tech Stack:** Python, SQLite/PostgreSQL-backed DB helpers, pytest, FastAPI-adjacent persistence layer, Bandit

---

## File Map

- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
  - Replace the swallowed-exception search test with contract assertions for exact/prefix/punctuation behavior.
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
  - Add a few contract assertions that prove the extracted store matches existing note CRUD/search behavior.
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py`
  - Only if test-first verification exposes real drift from `CharactersRAGDB`.
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
  - Only if test-first verification exposes real drift from `CharactersRAGDB`.
- Create: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
  - Extract the remaining persona-profile, buddy, session, memory, and exemplar persistence methods chosen for the Phase 2.3 seam.
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py`
  - Focused CRUD and search/list tests for the extracted persona-state store.
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
  - Export the new store class alongside existing store exports.
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Instantiate extracted stores in `__init__`.
  - Add dynamic delegation wiring for character, message, note, keyword, and persona-state store methods.
  - Keep public API compatibility intact.

## Task 1: Stabilize NoteStore And KeywordStore Contracts

**Files:**
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Test:
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "keyword or note"`
  - `tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "keyword or note"`

- [x] **Step 1: Write failing store tests that assert real keyword and note contracts**

Add focused assertions for:
- duplicate keyword insert raises the same conflict shape as the facade
- keyword search returns real matches instead of silently swallowing failures
- keyword search accepts punctuation cases already covered in the monolith tests
- note search and deleted-note listing behave like the existing facade

- [x] **Step 2: Run the focused store tests to verify they fail for the right reason**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py -v`

Expected: any failure should point to store behavior drift or a weak assertion, not to unrelated fixture setup.

- [x] **Step 3: Fix the minimal underlying drift in the extracted stores if needed**

Only change production store code when the failing tests prove a real mismatch with the original `CharactersRAGDB` contract.

- [x] **Step 4: Re-run facade-level contract slices**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "keyword or note" tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "keyword or note" -v`

Expected: the extracted stores still agree with the existing public behavior.

## Task 2: Extract PersonaStateStore

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Test:
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py`
  - any existing persona-state DB tests that cover the extracted methods

- [x] **Step 1: Choose the bounded persona-state method set to extract**

Start with the core CRUD/list/update methods around persona profiles, persona buddies, persona sessions, persona memories, and persona exemplars that are already clustered in `ChaChaNotes_DB.py`.

- [x] **Step 2: Write failing tests for the extracted persona-state seam**

Add focused tests covering:
- create/get/list/update for one or two representative persona entities
- soft-delete/restore or archive toggles where already supported
- one memory/session flow that proves the store can coordinate its helper logic

- [x] **Step 3: Implement `PersonaStateStore` with minimal behavior-preserving extraction**

Move logic without changing the public contract. Keep helper usage on the `CharactersRAGDB` instance where that avoids widening scope.

- [x] **Step 4: Run the focused persona-state tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -v`

## Task 3: Wire CharactersRAGDB Delegation And Verify

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Test:
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py`
  - targeted existing facade-level slices touched by delegation

- [x] **Step 1: Instantiate the extracted stores from `CharactersRAGDB.__init__`**

Add store instances for character, message, note, keyword, and persona-state stores alongside the existing conversation store.

- [x] **Step 2: Add delegation wiring for the extracted public methods**

Use the existing conversation delegation pattern rather than hand-written wrappers. Keep delegation lists explicit.

- [x] **Step 3: Run the focused extraction verification set**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -v`

- [x] **Step 4: Run Bandit on the touched DB-management scope**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_chacha_phase2_3.json`

- [x] **Step 5: Update this plan with the actual verification results**

Record the exact passing/failing slices before considering the branch ready for review.

## Task 4: Conservative Monolith Shrink Pass

**Goal:** Remove only dead duplicated `CharactersRAGDB` method bodies in `ChaChaNotes_DB.py` that are already replaced by extracted-store delegation and are pinned by tests.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
- Test:
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
  - `tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py`
  - `tldw_Server_API/tests/Notes_Graph/unit/test_graph_db_queries.py`
  - targeted facade-level slices that already cover delegated behavior

- [x] **Step 1: Add targeted coverage for the delegated surfaces I plan to delete**

Pin representative behavior for:
- `manage_character_tags` and `search_character_cards_by_tags`
- message metadata / RAG-context helpers that are delegated through `MessageStore`
- note graph helper methods already delegated through `NoteStore`
- keyword rename / merge / cross-link helpers where existing coverage is thin at the store seam

- [x] **Step 2: Run the focused tests before code deletion**

Run the smallest slices that prove the added assertions are green and that they exercise the delegated methods rather than unrelated setup.

- [x] **Step 3: Delete only the dead duplicated method bodies already replaced by store delegation**

Bound this pass to verified `character`, `message`, `note`, and `keyword` methods. Do not remove `conversation` capture points or persona helper clusters unless their compatibility is pinned explicitly in this branch.

- [x] **Step 4: Re-run focused verification and update this plan with the results**

Record exact passing slices and stop if any compatibility drift appears.

## Verification Results

- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py -v`
  Result: `12 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py -v`
  Result: `9 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "keyword or note" -v`
  Result: `56 passed`
- `python -m pytest tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "keyword or note" -v`
  Result: `15 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -v`
  Result: `5 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -v`
  Result: `41 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py -v`
  Result: `19 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "keyword or note or persona or message or character" -v`
  Result: `56 passed`
- `python -m pytest tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "keyword or note or message or character" -v`
  Result: `95 passed`
- `python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_chacha_phase2_3.json`
  Result: `0 findings`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py -v`
  Result: `23 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py -v`
  Result: `10 passed`
- `python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_graph_db_queries.py -v`
  Result: `24 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py -v`
  Result: `40 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py -v`
  Result: `23 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "keyword or note or message or character" -v`
  Result: `56 passed`
- `python -m pytest tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "keyword or note or message or character" -v`
  Result: `95 passed`
- `python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_chacha_phase2_3_shrink.json`
  Result: `0 findings`
- `python -m pytest tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "whitespace_only_tags_in_json_string_normalizes_empty or test_pbt_add_character_card" -v`
  Result: `2 passed, 94 deselected`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_character_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py tldw_Server_API/tests/Notes_Graph/unit/test_graph_db_queries.py -v`
  Result: `111 passed`
- `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "keyword or note or message or character or persona" tldw_Server_API/tests/Characters/test_character_functionality_db.py -k "keyword or note or message or character" -v`
  Result: `152 passed`
