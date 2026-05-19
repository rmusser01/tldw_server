# Writing Backend Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Writing backend review and deliver one findings-first, evidence-backed report covering correctness, data integrity, contract drift, maintainability, performance, and critical test gaps across the backend Writing API, manuscript API, and their direct persistence and helper boundaries.

**Architecture:** This is a read-first, risk-first review plan. Execution starts by locking the dirty-worktree baseline and final report contract, then inspects `writing.py` contract and stateful behavior, then traces manuscript CRUD and analysis behavior through the direct `ManuscriptDBHelper` and `CharactersRAGDB` boundaries, and only after that runs the smallest targeted pytest slices needed to confirm or weaken candidate findings. No repository source changes are part of execution; the deliverable is the final in-session review output.

**Tech Stack:** Python 3, pytest, git, rg, find, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- label findings that depend on uncommitted local changes
- keep code scope inside `tldw_Server_API/app/api/v1/endpoints/writing.py`, `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`, their schemas, `tldw_Server_API/app/core/Writing/manuscript_analysis.py`, and the direct `ManuscriptDBHelper` and `CharactersRAGDB` call paths those endpoints use
- exclude frontend Writing Playground files under `apps/`
- exclude `tldw_Server_API/app/core/Writing/note_title.py` and Notes routes unless a Writing endpoint directly depends on them, which is not expected for this review
- prioritize code and tests over stale or scaffolded docs
- separate `Confirmed finding`, `Probable risk`, and `Improvement`
- treat missing tests as findings only when they leave a critical branch, invariant, or externally visible contract weakly defended
- do not modify repository source files during the review itself
- do not run broad blanket suites; use the smallest targeted verification needed to answer a concrete question
- keep blind spots explicit instead of implying unreviewed surfaces are safe

## Review File Map

**No repository source files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-writing-backend-review-design.md`
- `Docs/superpowers/plans/2026-04-07-writing-backend-review-execution-plan.md`

**Primary implementation files to inspect first:**
- `tldw_Server_API/app/api/v1/endpoints/writing.py`
- `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- `tldw_Server_API/app/api/v1/schemas/writing_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`

**Non-authoritative context docs to inspect only if code intent is unclear:**
- `tldw_Server_API/app/core/Writing/README.md`
- `Docs/Plans/2026-04-03-writing-suite-design.md`
- `Docs/Design/Writing.md`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py`
- `tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py`
- `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`
- `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py`
- `tldw_Server_API/tests/Writing/test_manuscript_db.py`
- `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`
- `tldw_Server_API/tests/Writing/test_manuscript_characters_db.py`
- `tldw_Server_API/tests/Writing/test_manuscript_schema_contract.py`
- `tldw_Server_API/tests/Writing/test_manuscript_module_import.py`
- `tldw_Server_API/tests/Writing/test_writing_manuscripts_import.py`

**Scratch artifacts allowed during execution:**
- `/tmp/writing_backend_review_notes.md`
- `/tmp/writing_backend_endpoint_pytest.log`
- `/tmp/writing_backend_manuscript_pytest.log`
- `/tmp/writing_backend_analysis_pytest.log`
- `/tmp/writing_backend_tokenizer_pytest.log`

## Stage Overview

## Stage 1: Baseline and Report Contract
**Goal:** Lock the dirty-worktree baseline, confirm the exact Writing backend review surface, and fix the final report contract before deep reading starts.
**Success Criteria:** The scope boundary, file inventory, and final report structure are fixed before any candidate finding is treated as actionable.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: `writing.py` Contract and Stateful Endpoint Pass
**Goal:** Trace the public Writing API for sessions, templates, themes, defaults, snapshot import/export behavior, capabilities, tokenization, and wordcloud behavior.
**Success Criteria:** Endpoint-visible invariants, schema assumptions, and stateful flows are mapped with exact file references and direct DB helper touchpoints.
**Tests:** Read the Writing endpoint integration tests first, then run only the targeted endpoint slice needed to confirm or weaken suspect behavior.
**Status:** Not Started

## Stage 3: Manuscript CRUD and Persistence Boundary Pass
**Goal:** Trace project, part, chapter, scene, character, world, plot, relationship, citation, link, and reorder flows through `writing_manuscripts.py` and the direct `ManuscriptDBHelper` boundary.
**Success Criteria:** CRUD, optimistic-locking, soft-delete, search, and reorder assumptions are traced far enough to support evidence-backed findings or explicit blind spots.
**Tests:** Read the manuscript endpoint, schema, and DB tests first, then run only the targeted slices needed to settle concrete claims.
**Status:** Not Started

## Stage 4: Manuscript Analysis and Helper Pass
**Goal:** Inspect analysis request validation, provider and model override checks, helper prompt or output shaping, and how analysis results are persisted and invalidated.
**Success Criteria:** Analysis-related risks are tied to exact code paths in `writing_manuscripts.py`, `manuscript_analysis.py`, and direct DB persistence methods, with coverage status made explicit.
**Tests:** Read and run only the analysis-focused integration, DB, and service slices needed to confirm or weaken findings.
**Status:** Not Started

## Stage 5: Targeted Verification and Final Synthesis
**Goal:** Reconcile code reading with targeted test evidence and produce the final findings-first review report.
**Success Criteria:** Every major claim in the final review is backed by code inspection, test inspection, executed verification, or an explicit open-question label.
**Tests:** Only the additional narrow pytest slices needed to settle unresolved findings.
**Status:** Not Started

### Task 1: Lock the Baseline and Final Output Contract

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-writing-backend-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-writing-backend-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/writing.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Inspect: `tldw_Server_API/tests/Writing`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of uncommitted files, including whether scoped Writing backend files already differ from committed history.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Enumerate the exact scoped Writing backend code surface**

Run:
```bash
rg --files tldw_Server_API/app | rg 'api/v1/endpoints/writing(\\.py|_manuscripts\\.py)$|api/v1/schemas/writing(_manuscript)?_schemas\\.py$|core/Writing/manuscript_analysis\\.py$|core/DB_Management/(ChaChaNotes_DB|ManuscriptDB)\\.py$'
```

Expected: the concrete backend file inventory that anchors the review and prevents drift into frontend or Notes-only surfaces.

- [ ] **Step 4: Enumerate the Writing-specific test surface**

Run:
```bash
find tldw_Server_API/tests/Writing -maxdepth 1 -type f | sort
```

Expected: the Writing test inventory that makes endpoint, DB, schema, import, tokenizer, and analysis coverage visible before verification begins.

- [ ] **Step 5: Fix the final response contract before deep reading**

Use this exact final structure:
```markdown
## Findings
- severity-ordered findings
- each item states issue class, confidence, exact file references, impact, and fix direction when clear

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Improvements
- lower-priority maintainability, performance, or quality suggestions that are not immediate bugs

## Verification
- tests run, important files inspected, and what remains unverified
```

### Task 2: Execute the `writing.py` Contract and Stateful Endpoint Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/api/v1/endpoints/writing.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/writing_schemas.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Inspect: `tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py`
- Inspect: `tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py`
- Inspect: `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`
- Test: `tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py`

- [ ] **Step 1: Inventory the `writing.py` route surface**

Run:
```bash
rg -n '^@router\.(get|post|patch|delete)' tldw_Server_API/app/api/v1/endpoints/writing.py
```

Expected: the full route list for version, capabilities, defaults, sessions, templates, themes, snapshots, tokenization, and wordclouds.

- [ ] **Step 2: Read the contract-heavy sections before helper internals**

Run:
```bash
sed -n '1,740p' tldw_Server_API/app/api/v1/endpoints/writing.py
sed -n '1,381p' tldw_Server_API/app/api/v1/schemas/writing_schemas.py
```

Expected: helper, schema, and route context needed to evaluate whether request and response contracts match actual behavior.

- [ ] **Step 3: Read the stateful endpoint sections that are most likely to hide data or versioning bugs**

Run:
```bash
sed -n '1001,1739p' tldw_Server_API/app/api/v1/endpoints/writing.py
sed -n '1740,2016p' tldw_Server_API/app/api/v1/endpoints/writing.py
```

Expected: the create, update, delete, clone, snapshot import/export behavior, tokenize, detokenize, token-count, and wordcloud paths in one pass.

- [ ] **Step 4: Trace the direct DB methods those routes rely on**

Run:
```bash
rg -n 'writing_(session|template|theme|wordcloud)|snapshot|soft_delete|restore' tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
```

Expected: the concrete `CharactersRAGDB` methods that matter for optimistic locking, soft-delete semantics, snapshot merge or replace behavior, and cached wordcloud records.

- [ ] **Step 5: Read the highest-value endpoint tests before running anything**

Run:
```bash
rg -n '^def test_' tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py
sed -n '1,520p' tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py
sed -n '520,1160p' tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py
```

Expected: a map of what is already covered for sessions, templates, themes, snapshots, defaults, capabilities, tokenization, and wordclouds.

- [ ] **Step 6: Run the narrow Writing endpoint verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py \
  -k 'sessions or templates or themes or snapshot or defaults or capabilities or tokenize or detokenize or token_count or wordcloud' \
  -v | tee /tmp/writing_backend_endpoint_pytest.log
```

Expected: targeted PASS coverage for the contract-heavy Writing endpoint surface; if a failure appears, classify whether it is a real scoped defect, a local-worktree side effect, or an environment limitation.

### Task 3: Execute the Manuscript CRUD and Persistence Boundary Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_schema_contract.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_db.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_characters_db.py`
- Test: manuscript endpoint and DB slices

- [ ] **Step 1: Inventory the manuscript route surface before deep reading**

Run:
```bash
rg -n '^@router\.(get|post|patch|delete)' tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
```

Expected: the full route list for projects, parts, chapters, scenes, structure, search, characters, relationships, world info, plot lines, plot events, plot holes, citations, links, research, analysis, and listing endpoints.

- [ ] **Step 2: Read the manuscript route and schema contract sections**

Run:
```bash
sed -n '1,1100p' tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
sed -n '1,691p' tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py
```

Expected: enough context to judge optimistic-locking, null handling, aliasing, typed settings, and route-level error mapping before reading the DB helper in detail.

- [ ] **Step 3: Read the direct manuscript persistence helper surface**

Run:
```bash
rg -n 'class ManuscriptDBHelper|def (create_|get_|list_|update_|delete_|search_|reorder_|link_|unlink_|create_analysis|get_analysis|list_analyses|get_all_scene_texts)' tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
sed -n '1,520p' tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
sed -n '520,1200p' tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
```

Expected: the exact helper behavior the routes rely on for CRUD, search, reordering, linking, and analysis persistence.

- [ ] **Step 4: Read the highest-value manuscript contract and DB tests**

Run:
```bash
rg -n '^def test_' \
  tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py \
  tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py \
  tldw_Server_API/tests/Writing/test_manuscript_schema_contract.py \
  tldw_Server_API/tests/Writing/test_manuscript_db.py \
  tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py \
  tldw_Server_API/tests/Writing/test_manuscript_characters_db.py
```

Expected: a compact inventory of what CRUD, reorder, typed settings, and world or character persistence paths are already defended by tests.

- [ ] **Step 5: Run the narrow manuscript CRUD and DB verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py \
  tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py \
  tldw_Server_API/tests/Writing/test_manuscript_schema_contract.py \
  tldw_Server_API/tests/Writing/test_manuscript_db.py \
  tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py \
  tldw_Server_API/tests/Writing/test_manuscript_characters_db.py \
  -v | tee /tmp/writing_backend_manuscript_pytest.log
```

Expected: targeted PASS or a small number of scoped failures that directly inform manuscript CRUD, locking, or persistence findings.

### Task 4: Execute the Manuscript Analysis and Helper Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Inspect: `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`
- Inspect: `tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py`
- Inspect: `tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py`
- Inspect: `tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py`
- Test: analysis and tokenizer slices

- [ ] **Step 1: Read the manuscript analysis endpoint sections in isolation**

Run:
```bash
sed -n '1970,2225p' tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
```

Expected: the scene, chapter, and project analysis flows, including provider or model override validation, helper calls, score extraction, and persistence behavior.

- [ ] **Step 2: Read the core analysis helper implementation**

Run:
```bash
sed -n '1,220p' tldw_Server_API/app/core/Writing/manuscript_analysis.py
```

Expected: the analysis helper behavior for prompt construction, structured output handling, markdown fence stripping, and content extraction.

- [ ] **Step 3: Read the analysis-focused tests before executing them**

Run:
```bash
rg -n '^def test_' \
  tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py \
  tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py \
  tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py \
  tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py \
  tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py
```

Expected: a map of which override, rate-limit, stale-analysis, service, DB, and tokenizer metadata behaviors are already defended by tests.

- [ ] **Step 4: Run the narrow manuscript analysis verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py \
  tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py \
  tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py \
  -v | tee /tmp/writing_backend_analysis_pytest.log
```

Expected: targeted PASS or a small, analysis-specific failure set that directly informs findings about validation, persistence, stale-state handling, or helper output shaping.

- [ ] **Step 5: Run the narrow tokenizer and provider-metadata verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Writing/test_llm_providers_tokenizer_metadata.py \
  tldw_Server_API/tests/Writing/test_tokenizer_resolver_unit.py \
  -v | tee /tmp/writing_backend_tokenizer_pytest.log
```

Expected: targeted PASS or a small failure set that directly informs findings about Writing capability metadata, tokenizer exactness, and provider-native fallback claims.

### Task 5: Synthesize the Final Review Without Drifting

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-writing-backend-review-design.md`
- Inspect: `/tmp/writing_backend_endpoint_pytest.log`
- Inspect: `/tmp/writing_backend_manuscript_pytest.log`
- Inspect: `/tmp/writing_backend_analysis_pytest.log`
- Inspect: `/tmp/writing_backend_tokenizer_pytest.log`
- Test: only additional narrow slices if a claim is still unresolved

- [ ] **Step 1: Reconcile candidate findings against evidence quality**

Use this rule set:
```markdown
- Confirmed finding: directly supported by code path and, when helpful, confirmed by an executed test or a clearly missing guard
- Probable risk: code strongly suggests a defect, but runtime confirmation is absent or blocked
- Improvement: lower-priority maintainability, performance, or ergonomics issue without a clear immediate bug
```

- [ ] **Step 2: Remove weak or duplicative items before writing the report**

Run:
```bash
python - <<'PY'
from pathlib import Path
text = Path('Docs/superpowers/plans/2026-04-07-writing-backend-review-execution-plan.md').read_text()
red_flags = [
    'TO' + 'DO',
    'TB' + 'D',
    'place' + 'holder',
    'implement ' + 'later',
    'fill in ' + 'details',
    'appropriate error ' + 'handling',
    'handle edge ' + 'cases',
    'Similar to ' + 'Task',
]
hits = [flag for flag in red_flags if flag in text]
print('\n'.join(hits))
PY
```

Expected: no output; if a speculative point still lacks evidence, downgrade it to an open question or remove it.

- [ ] **Step 3: Run only the final dispute-settling tests if a major claim remains unresolved**

Run one of these only if needed:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py -k 'snapshot or wordcloud or capabilities or tokenize or detokenize' -v
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py -k 'optimistic_locking or reorder or not_found' -v
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py -k 'stale_after_update or invalid_analysis_types or runtime_rate_limit' -v
```

Expected: only the smallest additional confirmation needed to settle a disputed high-confidence claim.

- [ ] **Step 4: Write the final findings-first review in-session**

Use this exact section order:
```markdown
## Findings
- severity-ordered findings only

## Open Questions / Assumptions
- unresolved items that materially affect confidence

## Improvements
- lower-priority non-blocking improvements

## Verification
- files inspected, tests run, and remaining blind spots
```

- [ ] **Step 5: Verify the final review stays inside the approved boundary**

Final checks:
```markdown
- no frontend-only findings
- no Notes-only findings
- no provider-internals detours unless a Writing endpoint depends on them directly
- no code-fix suggestions presented as if they were already implemented
- no claims of verification without naming the actual test command or explaining why it was not run
```
