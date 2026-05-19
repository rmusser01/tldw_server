# Chatbooks Deep Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conduct a deep backend and contract-surface review of the Chatbooks module and produce stage-based findings documents plus a final synthesis.

**Architecture:** This is a read-first, evidence-driven review plan. Each stage inspects a bounded set of source files, checks the most relevant existing tests, records findings before suggesting fixes, and only then proceeds to the next stage. Review outputs live under `Docs/superpowers/reviews/chatbooks/` so later remediation work can reference stable findings instead of chat history.

**Tech Stack:** Python 3, FastAPI, pytest, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/chatbooks/README.md`
- `Docs/superpowers/reviews/chatbooks/2026-03-23-stage1-core-backend-review.md`
- `Docs/superpowers/reviews/chatbooks/2026-03-23-stage2-endpoints-jobs-review.md`
- `Docs/superpowers/reviews/chatbooks/2026-03-23-stage3-contract-alignment-review.md`
- `Docs/superpowers/reviews/chatbooks/2026-03-23-stage4-tests-maintainability-review.md`
- `Docs/superpowers/reviews/chatbooks/2026-03-23-stage5-chatbooks-synthesis.md`

**Primary source files to inspect:**
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- `tldw_Server_API/app/core/Chatbooks/quota_manager.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py`
- `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- `tldw_Server_API/app/services/chatbooks_cleanup_service.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- `tldw_Server_API/app/core/Chatbooks/README.md`
- `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md`
- `Docs/Product/Chatbooks_PRD.md`
- `Docs/API-related/chatbook_openapi.yaml`
- `Docs/Schemas/chatbooks_manifest_v1.json`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_path_traversal.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_template_mode_and_dict_strict.py`
- `tldw_Server_API/tests/integration/test_chatbook_integration.py`
- `tldw_Server_API/tests/e2e/test_chatbooks_roundtrip.py`
- `tldw_Server_API/tests/e2e/test_chatbooks_multi_user_roundtrip.py`
- `tldw_Server_API/tests/server_e2e_tests/test_chatbooks_api_workflow.py`
- `tldw_Server_API/tests/server_e2e_tests/test_chatbooks_roundtrip_workflow.py`

## Stage Overview

## Stage 1: Core Backend Correctness and Security
**Goal:** Validate the service, validators, quotas, storage paths, archive handling, and download-signing logic before trusting higher-level API behavior.
**Success Criteria:** Export/import/preview safety checks, retention rules, quota gates, temp-file handling, and download-token behavior are traced with concrete findings or explicit clearance notes.
**Tests:** `test_chatbook_security.py`, `test_chatbooks_path_traversal.py`, `test_chatbooks_signed_urls.py`, `test_chatbook_service.py`, `test_chatbooks_template_mode_and_dict_strict.py`
**Status:** Not Started

## Stage 2: Endpoint and Job Lifecycle Review
**Goal:** Validate request parsing, sync versus async behavior, status transitions, cancellation, cleanup, and download semantics exposed by the API.
**Success Criteria:** Endpoint contracts, job progression, terminal states, and cleanup behavior are documented with evidence and any misleading success or failure behavior is captured.
**Tests:** `test_chatbooks_export_sync.py`, `test_chatbooks_api_preview.py`, `test_chatbooks_api_error_and_preview_mapping.py`, `test_chatbooks_api_path_guard.py`, `test_chatbooks_cancellation.py`, `test_chatbooks_jobs_worker_import_defaults.py`
**Status:** Not Started

## Stage 3: Contract-Surface Alignment Review
**Goal:** Compare schemas, docs, OpenAPI, and the PRD against the current implementation so contract drift is explicit.
**Success Criteria:** Mismatches between code, schemas, docs, and PRD claims are cataloged with severity and downstream impact, with planned gaps kept separate from defects.
**Tests:** `test_chatbook_integration.py`, `test_chatbooks_roundtrip.py`, `test_chatbooks_multi_user_roundtrip.py`, `test_chatbooks_api_workflow.py`, `test_chatbooks_roundtrip_workflow.py`
**Status:** Not Started

## Stage 4: Test Adequacy and Maintainability Risk Review
**Goal:** Determine whether the existing tests cover the risky paths and where file concentration or churn raises regression risk.
**Success Criteria:** Coverage blind spots, false-confidence tests, and structural hotspots are documented separately from confirmed bugs.
**Tests:** Review all Chatbooks-targeting tests plus re-run only the smallest set needed to validate disputed findings.
**Status:** Not Started

## Stage 5: Whole-Surface Synthesis
**Goal:** Consolidate the earlier stages into a ranked Chatbooks findings report and remediation order.
**Success Criteria:** Duplicates are removed, cross-cutting risks are grouped, and the final synthesis clearly distinguishes confirmed findings, open questions, and secondary improvements.
**Tests:** Re-run only the narrowest commands needed to confirm any contested claim from earlier stages.
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Inventory

**Files:**
- Create: `Docs/superpowers/reviews/chatbooks/README.md`
- Create: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage1-core-backend-review.md`
- Create: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage2-endpoints-jobs-review.md`
- Create: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage3-contract-alignment-review.md`
- Create: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage4-tests-maintainability-review.md`
- Create: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage5-chatbooks-synthesis.md`
- Modify: `Docs/superpowers/plans/2026-03-23-chatbooks-deep-review.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/chatbooks
```

Expected: the directory exists with no other workspace changes.

- [ ] **Step 2: Create one markdown file per stage with a fixed findings template**

Each file should contain these headings:
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

- [ ] **Step 3: Write `Docs/superpowers/reviews/chatbooks/README.md`**

Document:
- the stage order `1 -> 2 -> 3 -> 4 -> 5`
- the path of each stage report
- the rule that findings must be written before fixes are proposed
- the evidence bar for confirmed findings versus open questions

- [ ] **Step 4: Verify the workspace is in a safe starting state**

Run:
```bash
git status --short
```

Expected: only the new review-doc paths appear, or the tree is still clean if files have not been created yet.

- [ ] **Step 5: Commit the review scaffold**

Run:
```bash
git add Docs/superpowers/reviews/chatbooks Docs/superpowers/plans/2026-03-23-chatbooks-deep-review.md
git commit -m "docs: scaffold chatbooks review artifacts"
```

Expected: one docs-only commit capturing the review workspace.

### Task 2: Execute Stage 1 Core Backend Correctness and Security Review

**Files:**
- Modify: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage1-core-backend-review.md`
- Inspect: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/quota_manager.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_path_traversal.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_template_mode_and_dict_strict.py`

- [ ] **Step 1: Map the Stage 1 backend entry points**

Run:
```bash
rg -n "def (create_chatbook|import_chatbook|preview_chatbook|cleanup_expired_exports|cancel_export_job|cancel_import_job)|CHATBOOKS_|signed|token|expires_at|temp|Path|validate_" \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_validators.py \
  tldw_Server_API/app/core/Chatbooks/quota_manager.py \
  tldw_Server_API/app/core/DB_Management/db_path_utils.py
```

Expected: a concise map of the core flows, limit settings, and security-sensitive helpers to record in the stage report.

- [ ] **Step 2: Trace archive validation and storage-path safety**

Read the relevant functions and record:
- how filenames and ZIP members are validated
- where symlink and traversal protections are enforced
- where per-user storage roots, temp paths, and export paths are checked for containment

- [ ] **Step 3: Trace export/import/preview and retention behavior**

Confirm:
- where sync and async paths diverge in the service
- how retention and expiry timestamps are calculated
- whether failures can leave inconsistent files, temp artifacts, or job rows behind

- [ ] **Step 4: Review quota and signing logic**

Confirm:
- quota checks that happen in the service versus in the API layer
- how signed download URLs are created and verified
- whether enforcement matches the documented download semantics

- [ ] **Step 5: Read the targeted Stage 1 tests and extract the protected invariants**

For each listed test file, record:
- the main behavior it validates
- any risky path it does not appear to cover

- [ ] **Step 6: Run the targeted Stage 1 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbook_security.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_path_traversal.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_template_mode_and_dict_strict.py -v
```

Expected: tests collect and pass; any failure is triaged as either environment noise or a possible finding that must be documented with evidence.

- [ ] **Step 7: Write the Stage 1 report**

Record:
- confirmed findings ordered by severity
- test gaps or ambiguous behaviors
- low-risk improvements
- an exit note stating whether later stages need to account for a backend-level concern

- [ ] **Step 8: Commit the Stage 1 report**

Run:
```bash
git add Docs/superpowers/reviews/chatbooks/2026-03-23-stage1-core-backend-review.md
git commit -m "docs: record chatbooks core backend review findings"
```

Expected: one commit containing only the Stage 1 report.

### Task 3: Execute Stage 2 Endpoint and Job Lifecycle Review

**Files:**
- Modify: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage2-endpoints-jobs-review.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- Inspect: `tldw_Server_API/app/services/chatbooks_cleanup_service.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`

- [ ] **Step 1: Map the API surface and job-status paths**

Run:
```bash
rg -n "@router\\.(post|get|delete)|async def (create_chatbook|preview_chatbook|list_export_jobs|get_export_job|get_import_job|download_chatbook|cleanup_expired_exports|cancel_export_job|cancel_import_job|remove_export_job|remove_import_job)" \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py
```

Expected: a route inventory with the main entrypoints and job-management operations for the stage report.

- [ ] **Step 2: Trace sync versus async behavior at the API boundary**

Confirm:
- where request schemas are coerced into core enums or flags
- what the API returns for sync versus async export and import
- whether endpoint messaging, status codes, and response models align with runtime behavior

- [ ] **Step 3: Trace job-state transitions and cleanup behavior**

Confirm:
- how pending, in-progress, completed, failed, cancelled, expired, and removed states are represented
- where cancellation is best-effort versus guaranteed
- whether cleanup semantics match endpoint descriptions and worker behavior

- [ ] **Step 4: Review the targeted Stage 2 tests**

For each listed test file, record:
- which endpoint invariant it covers
- which job-lifecycle branches remain unproven

- [ ] **Step 5: Run the targeted Stage 2 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py -v
```

Expected: tests collect and pass; any mismatch between code, tests, and docs is written up as a stage finding or open question.

- [ ] **Step 6: Write the Stage 2 report**

Record:
- confirmed endpoint and job-lifecycle findings ordered by severity
- contract ambiguities that affect API callers
- low-risk improvements and an exit note for Stage 3

- [ ] **Step 7: Commit the Stage 2 report**

Run:
```bash
git add Docs/superpowers/reviews/chatbooks/2026-03-23-stage2-endpoints-jobs-review.md
git commit -m "docs: record chatbooks endpoint review findings"
```

Expected: one commit containing only the Stage 2 report.

### Task 4: Execute Stage 3 Contract-Surface Alignment Review

**Files:**
- Modify: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage3-contract-alignment-review.md`
- Inspect: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/README.md`
- Inspect: `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md`
- Inspect: `Docs/Product/Chatbooks_PRD.md`
- Inspect: `Docs/API-related/chatbook_openapi.yaml`
- Inspect: `Docs/Schemas/chatbooks_manifest_v1.json`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py`
- Test: `tldw_Server_API/tests/integration/test_chatbook_integration.py`
- Test: `tldw_Server_API/tests/e2e/test_chatbooks_roundtrip.py`
- Test: `tldw_Server_API/tests/e2e/test_chatbooks_multi_user_roundtrip.py`
- Test: `tldw_Server_API/tests/server_e2e_tests/test_chatbooks_api_workflow.py`
- Test: `tldw_Server_API/tests/server_e2e_tests/test_chatbooks_roundtrip_workflow.py`

- [ ] **Step 1: Build a contract matrix**

Create a table in the stage report with these columns:
- contract source
- claimed behavior
- implementation location
- status (`matches`, `partial`, `drift`, `planned gap`)

- [ ] **Step 2: Compare schemas and OpenAPI to the endpoint behavior**

Confirm:
- request and response fields that are actually emitted
- status or field names that appear in docs or schemas but not in runtime behavior
- whether response examples or schema expectations could mislead API consumers

- [ ] **Step 3: Compare the code guide and PRD to the current implementation**

Separate:
- true contract drift that misstates current behavior
- known planned gaps already acknowledged in the PRD
- ambiguous wording that could lead an implementer or caller to build against the wrong assumption

- [ ] **Step 4: Read the integration and end-to-end Chatbooks tests**

For each listed test file, record:
- which public workflow it validates
- whether it tests the current documented contract or only a subset of it

- [ ] **Step 5: Run the smallest high-signal Stage 3 verification commands**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbook_integration.py \
  tldw_Server_API/tests/integration/test_chatbook_integration.py \
  -v
```

Expected: targeted integration checks pass; larger e2e files are read for coverage assessment and only executed later if a specific contract claim needs runtime confirmation.

- [ ] **Step 6: Write the Stage 3 report**

Record:
- confirmed contract drift findings ordered by severity
- planned gaps explicitly separated from defects
- any doc or schema changes likely needed after the review

- [ ] **Step 7: Commit the Stage 3 report**

Run:
```bash
git add Docs/superpowers/reviews/chatbooks/2026-03-23-stage3-contract-alignment-review.md
git commit -m "docs: record chatbooks contract alignment findings"
```

Expected: one commit containing only the Stage 3 report.

### Task 5: Execute Stage 4 Test Adequacy and Maintainability Risk Review

**Files:**
- Modify: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage4-tests-maintainability-review.md`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_path_traversal.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_path_guard.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_cancellation.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`
- Inspect: `tldw_Server_API/tests/Chatbooks/test_chatbooks_template_mode_and_dict_strict.py`
- Inspect: `tldw_Server_API/tests/e2e/test_chatbooks_roundtrip.py`
- Inspect: `tldw_Server_API/tests/e2e/test_chatbooks_multi_user_roundtrip.py`
- Inspect: `tldw_Server_API/tests/server_e2e_tests/test_chatbooks_api_workflow.py`
- Inspect: `tldw_Server_API/tests/server_e2e_tests/test_chatbooks_roundtrip_workflow.py`
- Inspect: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Test: optional only if a disputed finding needs confirmation

- [ ] **Step 1: Build a coverage map**

Create a table in the stage report with these columns:
- risky behavior
- source file/function
- tests that cover it
- confidence (`high`, `medium`, `low`)
- note

- [ ] **Step 2: Compare risky paths to current tests**

Focus on:
- async job execution and cancellation edges
- cleanup and expiry enforcement
- import conflict strategies
- large-file and retention limits
- optional provider/database paths
- multi-user isolation claims

- [ ] **Step 3: Quantify maintainability hotspots**

Run:
```bash
wc -l \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_validators.py \
  tldw_Server_API/app/core/Chatbooks/quota_manager.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py
```

Expected: a size snapshot recorded in the stage report to support any structural-risk claims.

- [ ] **Step 4: Check TODOs and recent churn that affect review confidence**

Run:
```bash
rg -n "TODO|FIXME|XXX|HACK" \
  tldw_Server_API/app/core/Chatbooks \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md \
  Docs/Product/Chatbooks_PRD.md

git log --oneline -n 20 -- \
  tldw_Server_API/app/core/Chatbooks \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/tests/Chatbooks
```

Expected: a short list of open parity gaps and recent churn points to fold into the confidence assessment.

- [ ] **Step 5: Write the Stage 4 report**

Record:
- meaningful test gaps
- tests that may give false confidence
- structural risks that materially increase defect probability
- improvement ideas kept separate from confirmed bugs

- [ ] **Step 6: Commit the Stage 4 report**

Run:
```bash
git add Docs/superpowers/reviews/chatbooks/2026-03-23-stage4-tests-maintainability-review.md
git commit -m "docs: record chatbooks test coverage and maintainability risks"
```

Expected: one commit containing only the Stage 4 report.

### Task 6: Produce the Final Chatbooks Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage5-chatbooks-synthesis.md`
- Inspect: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage1-core-backend-review.md`
- Inspect: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage2-endpoints-jobs-review.md`
- Inspect: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage3-contract-alignment-review.md`
- Inspect: `Docs/superpowers/reviews/chatbooks/2026-03-23-stage4-tests-maintainability-review.md`
- Test: optional only for disputed claims that still need confirmation

- [ ] **Step 1: Merge stage findings into one ranked issue list**

Organize findings under:
- `High`
- `Medium`
- `Low`

Each issue should include:
- type
- impact
- reasoning
- file references

- [ ] **Step 2: Write a separate open-questions section**

Only include items where intent or runtime behavior remains ambiguous after code and test review.

- [ ] **Step 3: Write a short residual-risk and coverage summary**

Summarize:
- what appears solid
- which areas still have evidence gaps
- what kind of follow-up validation would reduce uncertainty most

- [ ] **Step 4: If needed, run only the narrowest confirmation command for disputed claims**

Run only if necessary:
```bash
source .venv/bin/activate
python -m pytest <smallest_relevant_test_target> -v
```

Expected: either no command is needed, or one narrowly scoped command resolves the disputed point and is recorded in the synthesis document.

- [ ] **Step 5: Write the final synthesis document**

The document should end with:
- a prioritized remediation order
- a note separating immediate bugs from later cleanup

- [ ] **Step 6: Commit the final synthesis**

Run:
```bash
git add Docs/superpowers/reviews/chatbooks/2026-03-23-stage5-chatbooks-synthesis.md
git commit -m "docs: add chatbooks review synthesis"
```

Expected: one commit containing only the final synthesis document.

- [ ] **Step 7: Verify the final review workspace**

Run:
```bash
git status --short
```

Expected: the tree is clean, or only unrelated pre-existing changes remain.
