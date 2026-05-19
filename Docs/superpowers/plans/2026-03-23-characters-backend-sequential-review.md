# Characters Backend Sequential Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conduct a backend-only, risk-first review of the Characters module and produce stage reports plus a final synthesis with ranked findings, open questions, and coverage gaps.

**Architecture:** This is a read-first audit plan. Work proceeds in bounded stages that each inspect a specific slice of the Characters backend, review the matching tests, run only the smallest relevant validation commands, and write findings before moving to the next slice. Review outputs live under `Docs/superpowers/reviews/characters-backend/` so later remediation work can reference stable findings instead of terminal history.

**Tech Stack:** Python 3, FastAPI, SQLite/PostgreSQL-backed DB helpers, pytest, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/characters-backend/README.md`
- `Docs/superpowers/reviews/characters-backend/2026-03-23-stage1-review-artifacts.md`
- `Docs/superpowers/reviews/characters-backend/2026-03-23-stage2-api-crud-versioning.md`
- `Docs/superpowers/reviews/characters-backend/2026-03-23-stage3-import-validation-export.md`
- `Docs/superpowers/reviews/characters-backend/2026-03-23-stage4-exemplars-worldbooks-search.md`
- `Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md`

**Primary source files to inspect:**
- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- `tldw_Server_API/app/core/Character_Chat/Character_Chat_Lib_facade.py`
- `tldw_Server_API/app/core/Character_Chat/character_limits.py`
- `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py`
- `tldw_Server_API/app/core/Character_Chat/ccv3_parser.py`
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_db.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_io.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_validation.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py`
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py`
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_embeddings.py`
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_telemetry.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/Chat/chat_characters.py`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/Characters/test_characters_endpoint.py`
- `tldw_Server_API/tests/Characters/test_character_functionality_db.py`
- `tldw_Server_API/tests/Characters/test_character_chat_lib.py`
- `tldw_Server_API/tests/Characters/test_ccv3_parser.py`
- `tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py`
- `tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py`
- `tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py`
- `tldw_Server_API/tests/Character_Chat/test_file_mime_detection.py`
- `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py`
- `tldw_Server_API/tests/Character_Chat_NEW/property/test_character_properties.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_cards_fts_bootstrap.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_default_provider.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_manager.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_generation_presets.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_prompt_presets.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_templates.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_embeddings.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_selector.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_telemetry.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_png_export.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py`
- `tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_conversation_character_scope_filters.py`
- `tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py`
- `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py`
- `tldw_Server_API/tests/e2e/test_chats_and_characters.py`
- `tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py`
- `tldw_Server_API/tests/unit/test_character_rate_limiter.py`

## Stage Overview

## Stage 1: Review Artifact Setup
**Goal:** Create stable review output files and capture the inventory of source and test surfaces before detailed inspection.
**Success Criteria:** A review workspace exists, each stage report has a fixed template, and the plan names the exact files and commands used in later stages.
**Tests:** None
**Status:** Complete

## Stage 2: API, CRUD, Versioning, and Restore Integrity
**Goal:** Validate the main character lifecycle from endpoint contract through DB persistence and optimistic concurrency.
**Success Criteria:** Create, query, update, delete, restore, revert, search, and version-history flows are traced with concrete findings or explicit no-finding notes, and any contract mismatches are documented.
**Tests:** `test_characters_endpoint.py`, `test_character_functionality_db.py`, `test_character_chat_lib.py`, `test_character_api.py`, `test_character_cards_fts_bootstrap.py`, `test_character_card_tag_search.py`
**Status:** Not Started

## Stage 3: Import, Validation, Image Handling, and Export
**Goal:** Validate inbound and outbound character payload handling, including file-type checks, parsing, image processing, and export correctness.
**Success Criteria:** Import and export paths are reviewed end-to-end, parser and validator assumptions are documented, and any unsafe or inconsistent content handling is captured with evidence.
**Tests:** `test_ccv3_parser.py`, `test_file_mime_detection.py`, `test_png_export.py`, `test_character_properties.py`, targeted `test_characters_endpoint.py` import or export cases
**Status:** Not Started

## Stage 4: Exemplars, World Books, and Search/Retrieval Behavior
**Goal:** Validate the non-trivial retrieval surfaces attached to characters, including exemplar CRUD/search/selection, embedding integration, world-book attachments, and backend search behavior.
**Success Criteria:** Exemplar and world-book flows are traced through endpoint, service, and DB layers; selection heuristics and fallback paths are reviewed; and search or permission risks are recorded.
**Tests:** `test_character_exemplars_api.py`, `test_character_exemplars_db.py`, `test_persona_exemplar_selector.py`, `test_persona_exemplar_embeddings.py`, `test_persona_exemplar_telemetry.py`, `test_world_book_manager.py`, `test_characters_world_book_permissions_unit.py`, `test_world_book_negatives_and_new_endpoint.py`, `test_dual_backend_characters_retriever.py`
**Status:** Not Started

## Stage 5: Chat Coupling, Rate Limits, and Final Synthesis
**Goal:** Validate how Character state affects chat/session behavior and consolidate the whole review into one ranked findings report.
**Success Criteria:** Character-chat coupling, session scope behavior, rate limiting, and streaming persistence are reviewed with evidence; duplicates across earlier stages are removed; and the final synthesis ranks issues by impact and confidence.
**Tests:** `test_character_chat_endpoints.py`, `test_character_chat_auto_routing.py`, `test_character_chat_stream_and_persist.py`, `test_chat_session_character_scope_api.py`, `test_conversation_character_scope_filters.py`, `test_character_chat_default_provider.py`, `test_character_chat_manager.py`, `test_character_memory.py`, `test_character_rate_limiter.py`, `test_character_chat_sse_unified_flag.py`, `test_chats_and_characters.py`, `test_character_chats_workflow.py`
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Inventory

**Files:**
- Create: `Docs/superpowers/reviews/characters-backend/README.md`
- Create: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage1-review-artifacts.md`
- Create: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage2-api-crud-versioning.md`
- Create: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage3-import-validation-export.md`
- Create: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage4-exemplars-worldbooks-search.md`
- Create: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md`
- Modify: `Docs/superpowers/plans/2026-03-23-characters-backend-sequential-review.md`
- Test: none

- [x] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/characters-backend
```

Expected: the directory exists and no source files change.

- [x] **Step 2: Create one markdown file per stage with a fixed review template**

Each stage file should contain:
```markdown
# Stage N Title

## Scope
## Code Paths Reviewed
## Tests Reviewed
## Validation Commands
## Findings
## Coverage Gaps
## Improvements
## Exit Note
```

- [x] **Step 3: Write `Docs/superpowers/reviews/characters-backend/README.md`**

Document:
- the stage order `1 -> 2 -> 3 -> 4 -> 5`
- the path to each stage report
- the rule that findings must be written before remediation ideas
- the rule that uncertain items are labeled as assumptions, not overstated as bugs

- [x] **Step 4: Verify the workspace starts in a safe state**

Run:
```bash
git status --short
```

Expected: only docs changes appear, or the tree is clean before the review artifacts are created.

- [x] **Step 5: Commit the review scaffold**

Run:
```bash
git add Docs/superpowers/reviews/characters-backend Docs/superpowers/plans/2026-03-23-characters-backend-sequential-review.md
git commit -m "docs: scaffold characters backend review artifacts"
```

Expected: one docs-only commit captures the review workspace.

### Task 2: Execute Stage 2 API, CRUD, Versioning, and Restore Review

**Files:**
- Modify: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage2-api-crud-versioning.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/Character_Chat_Lib_facade.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/character_db.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Characters/test_characters_endpoint.py`
- Test: `tldw_Server_API/tests/Characters/test_character_functionality_db.py`
- Test: `tldw_Server_API/tests/Characters/test_character_chat_lib.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_cards_fts_bootstrap.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py`

- [ ] **Step 1: Map the lifecycle endpoints and helpers**

Run:
```bash
rg -n "@router\\.(get|post|put|delete)|def _build_character_|def _get_characters_restore_retention_days|def .*character" \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py \
  tldw_Server_API/app/core/Character_Chat/Character_Chat_Lib_facade.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_db.py
```

Expected: a compact map of create, list, query, get, update, delete, restore, search, export, and version-related entry points.

- [ ] **Step 2: Trace create, update, delete, restore, and revert into the DB layer**

Run:
```bash
rg -n "add_character_card|get_character_card_by_|query_character_cards|update_character_card|soft_delete_character_card|restore_character_card|get_character_version_history" \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_db.py \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py
```

Expected: every lifecycle mutation maps to a DB method that can be reviewed for version checks and invariant preservation.

- [ ] **Step 3: Record the critical invariants**

Capture in the stage report:
- how uniqueness is enforced
- where optimistic concurrency is checked
- which fields are normalized before persistence
- what restore and revert are allowed to do
- whether endpoint responses match the underlying persistence semantics

- [ ] **Step 4: Review the targeted tests and extract what they actually protect**

For each listed test file, record:
- the main invariant asserted
- which lifecycle branch it covers
- whether the test is strong, weak, or misleading

- [ ] **Step 5: Run the targeted lifecycle tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Characters/test_characters_endpoint.py \
  tldw_Server_API/tests/Characters/test_character_functionality_db.py \
  tldw_Server_API/tests/Characters/test_character_chat_lib.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_cards_fts_bootstrap.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_character_card_tag_search.py -v
```

Expected: tests collect and mostly pass; any failure is either environment noise or a finding that must be explained in the stage report.

- [ ] **Step 6: Write the Stage 2 report**

Record:
- ranked findings with file references
- open questions and assumptions
- coverage gaps
- low-risk improvements that do not distract from the findings

- [ ] **Step 7: Commit the Stage 2 report**

Run:
```bash
git add Docs/superpowers/reviews/characters-backend/2026-03-23-stage2-api-crud-versioning.md
git commit -m "docs: record characters lifecycle review findings"
```

Expected: one docs-only commit containing the Stage 2 report.

### Task 3: Execute Stage 3 Import, Validation, Image Handling, and Export Review

**Files:**
- Modify: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage3-import-validation-export.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/ccv3_parser.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/character_io.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/character_validation.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/character_db.py`
- Test: `tldw_Server_API/tests/Characters/test_ccv3_parser.py`
- Test: `tldw_Server_API/tests/Character_Chat/test_file_mime_detection.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_png_export.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/property/test_character_properties.py`
- Test: `tldw_Server_API/tests/Characters/test_characters_endpoint.py`

- [ ] **Step 1: Map the import and export entry points**

Run:
```bash
rg -n "MAX_CHARACTER_FILE_SIZE|ALLOWED_EXTENSIONS|_detect_mime_type|_validate_file_type|/import|/export|import_and_save_character_from_file|load_character_card_from_string_content|parse_character_book|validate_character_book" \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_io.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_validation.py \
  tldw_Server_API/app/core/Character_Chat/ccv3_parser.py
```

Expected: a clear map of upload validation, parser entry points, and export paths.

- [ ] **Step 2: Trace image handling and normalization**

Run:
```bash
rg -n "image_base64|Image\\.open|thumbnail|WEBP|verify\\(|base64\\.b64decode" \
  tldw_Server_API/app/core/Character_Chat/modules/character_db.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_io.py \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py
```

Expected: every image decode, verify, resize, and re-encode step is visible for review.

- [ ] **Step 3: Inspect parser and validator assumptions**

Record:
- what malformed input is rejected
- what malformed input is silently normalized
- where file extension checks and content checks disagree
- whether export output is stable enough to re-import safely

- [ ] **Step 4: Review the targeted tests and label any gaps**

Check whether the tests cover:
- oversize files
- misleading extensions
- invalid or partial image payloads
- malformed JSON or CCv3 payloads
- export/import round-trip assumptions

- [ ] **Step 5: Run the targeted import or export tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Characters/test_ccv3_parser.py \
  tldw_Server_API/tests/Character_Chat/test_file_mime_detection.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_png_export.py \
  tldw_Server_API/tests/Character_Chat_NEW/property/test_character_properties.py \
  tldw_Server_API/tests/Characters/test_characters_endpoint.py -k "import or export" -v
```

Expected: parser, MIME, and export tests pass; any surprising success or failure becomes either a coverage note or a finding.

- [ ] **Step 6: Write the Stage 3 report**

Record:
- risky validation gaps
- import/export contract inconsistencies
- performance or memory concerns in image handling
- missing regression tests for malformed input classes

- [ ] **Step 7: Commit the Stage 3 report**

Run:
```bash
git add Docs/superpowers/reviews/characters-backend/2026-03-23-stage3-import-validation-export.md
git commit -m "docs: record characters import review findings"
```

Expected: one docs-only commit containing the Stage 3 report.

### Task 4: Execute Stage 4 Exemplars, World Books, and Search/Retrieval Review

**Files:**
- Modify: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage4-exemplars-worldbooks-search.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_embeddings.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_telemetry.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_selector.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_embeddings.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_telemetry.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py`
- Test: `tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py`
- Test: `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py`
- Test: `tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py`

- [ ] **Step 1: Map exemplar and world-book route surfaces**

Run:
```bash
rg -n "exemplar|world-book|world_books|process_context|attach_to_character|detach_from_character|get_character_world_books|search_character_exemplars|select_character_exemplars" \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py \
  tldw_Server_API/app/core/Character_Chat/world_book_manager.py \
  tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
```

Expected: a route-to-service-to-DB map for exemplar CRUD, search, selection, world-book attach or detach, and context processing.

- [ ] **Step 2: Inspect scoring, fallback, and best-effort behavior**

Record:
- when lexical search is used
- when embeddings are used
- what happens when embedding sync or scoring fails
- whether best-effort fallbacks can mask integrity or ranking problems

- [ ] **Step 3: Inspect permission and ownership boundaries**

Confirm:
- which user or character identifiers scope exemplar and world-book data
- whether attach, detach, and list paths enforce expected ownership semantics
- whether retrieval helpers can see deleted or unrelated records unexpectedly

- [ ] **Step 4: Review the targeted tests and identify unprotected branches**

Specifically check for missing tests around:
- fallback scoring paths
- partial embedding failure
- deleted exemplar visibility
- world-book permission drift
- retrieval ordering or pagination assumptions

- [ ] **Step 5: Run the targeted exemplar, world-book, and retriever tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_selector.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_embeddings.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_telemetry.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py \
  tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py \
  tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py -v
```

Expected: the search and attachment surfaces are validated; any failure is either a concrete finding or a clearly explained environment issue.

- [ ] **Step 6: Write the Stage 4 report**

Record:
- correctness or permission findings
- fallback and observability concerns
- performance risks in search or scoring
- test gaps around retrieval semantics

- [ ] **Step 7: Commit the Stage 4 report**

Run:
```bash
git add Docs/superpowers/reviews/characters-backend/2026-03-23-stage4-exemplars-worldbooks-search.md
git commit -m "docs: record characters exemplar review findings"
```

Expected: one docs-only commit containing the Stage 4 report.

### Task 5: Execute Stage 5 Chat Coupling, Rate Limit, and Synthesis Review

**Files:**
- Modify: `Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md`
- Modify: `Docs/superpowers/reviews/characters-backend/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_characters.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/character_limits.py`
- Inspect: `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`
- Test: `tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_conversation_character_scope_filters.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_default_provider.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_manager.py`
- Test: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py`
- Test: `tldw_Server_API/tests/unit/test_character_rate_limiter.py`
- Test: `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py`
- Test: `tldw_Server_API/tests/e2e/test_chats_and_characters.py`
- Test: `tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py`

- [ ] **Step 1: Map how Character state flows into chat and session behavior**

Run:
```bash
rg -n "character_id|conversation_character_scope|load_chat_and_character|get_conversations_for_character|count_conversations_for_user_by_character|first_message|default provider|auto_routing" \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/core/Chat/chat_characters.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_chat.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
```

Expected: a map of where Character metadata affects chat creation, loading, routing, persistence, and filtering.

- [ ] **Step 2: Inspect rate-limit and quota behavior for Character operations**

Run:
```bash
rg -n "check_rate_limit|check_character_limit|resource governor|legacy deprecation|character_chat.default|operation" \
  tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py \
  tldw_Server_API/app/core/Character_Chat/character_limits.py \
  tldw_Server_API/Config_Files/resource_governor_policies.yaml
```

Expected: the relationship between legacy limiters and resource-governor cutover is visible for review.

- [ ] **Step 3: Review the targeted tests and note integration blind spots**

Focus on:
- session scope filtering
- chat persistence and stream recovery
- default-provider resolution
- first-message and greeting behavior
- rate-limit enforcement mismatches between legacy and unified paths

- [ ] **Step 4: Run the targeted chat and rate-limit tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py \
  tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_conversation_character_scope_filters.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_default_provider.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_manager.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py \
  tldw_Server_API/tests/unit/test_character_rate_limiter.py \
  tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py \
  tldw_Server_API/tests/e2e/test_chats_and_characters.py \
  tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py -v
```

Expected: chat-coupling and limiter tests pass or reveal concrete gaps that must be written up.

- [ ] **Step 5: Write the final synthesis**

The synthesis must:
- deduplicate findings from earlier stages
- rank findings by severity and confidence
- separate confirmed defects from open questions
- summarize which areas appear well-covered versus weakly covered
- recommend a fix order without drifting into implementation detail

- [ ] **Step 6: Update the review README with final report links**

Add:
- the final synthesis path
- one-sentence summaries for each stage
- the recommended reading order for maintainers

- [ ] **Step 7: Commit the Stage 5 report and README update**

Run:
```bash
git add Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md Docs/superpowers/reviews/characters-backend/README.md
git commit -m "docs: record characters backend review synthesis"
```

Expected: one docs-only commit containing the final synthesis and index update.

## Notes for Execution

- Use the approved spec at `Docs/superpowers/specs/2026-03-23-characters-backend-review-design.md` as the scope guard.
- Do not drift into frontend review.
- Do not propose fixes before writing findings.
- If a test fails because of environment setup rather than a product issue, record that explicitly instead of forcing a conclusion.
- This plan is analysis-only. Bandit is not required unless source code changes are introduced outside the review-doc workflow.
