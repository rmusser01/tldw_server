# Moderation Backend Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved moderation backend review and deliver one findings-first, evidence-backed report covering correctness, security or privacy issues, persistence and concurrency risks, maintainability concerns, and test gaps across the backend moderation service, admin endpoints, schemas, and real chat moderation caller path.

**Architecture:** This is a read-first, risk-led audit plan. Execution starts by locking the dirty-worktree baseline, creating stage review artifacts, and fixing the final findings contract before deep reading starts. It then inspects policy and parsing behavior in `moderation_service.py`, traces endpoint and `chat.py` caller contracts into permission and enforcement behavior, performs targeted verification for persistence and `If-Match` or ETag claims, and finishes with one severity-ordered synthesis that separates confirmed findings, probable risks, improvements, and open questions.

**Tech Stack:** Python 3, FastAPI, pytest, git, ripgrep, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- label any finding that depends on uncommitted local changes
- keep code scope inside backend moderation surfaces only
- exclude moderation playground UI, frontend services, browser extension moderation views, and unrelated chat behavior
- inspect `tldw_Server_API/app/api/v1/endpoints/chat.py` only at moderation call sites needed to confirm the real enforcement contract
- use the moderation-focused backend tests as the primary evidence set
- separate `Confirmed finding`, `Probable risk`, `Improvement`, and `Open question`
- do not modify repository source files during the review itself
- use the smallest targeted pytest slices needed to confirm or weaken a specific claim
- require stronger verification for persistence, atomicity, reload safety, and `If-Match` or `ETag` claims instead of relying on source inspection alone
- keep blind spots explicit instead of implying unreviewed moderation-adjacent files are safe

## Review File Map

**No repository source files should be modified during execution.**

**Create during execution:**
- `Docs/superpowers/reviews/moderation-backend/README.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage1-baseline-and-inventory.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage5-test-gaps-and-final-synthesis.md`

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-moderation-backend-review-design.md`
- `Docs/superpowers/plans/2026-04-07-moderation-backend-review-execution-plan.md`

**Primary documentation and contract references:**
- `Docs/Code_Documentation/Moderation-Guardrails.md`
- `Docs/Published/Code_Documentation/Moderation-Guardrails.md`
- `tldw_Server_API/app/core/Moderation/README.md`
- `tldw_Server_API/Config_Files/moderation_blocklist.txt`

**Primary source files to inspect first:**
- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/chat.py`

**Moderation-adjacent files to inspect only if an active trace requires them:**
- `tldw_Server_API/app/core/Moderation/category_taxonomy.py`
- `tldw_Server_API/app/core/Moderation/conflict_resolution.py`
- `tldw_Server_API/app/core/Moderation/governance_utils.py`
- `tldw_Server_API/app/core/Moderation/supervised_policy.py`
- `tldw_Server_API/app/core/Moderation/semantic_matcher.py`
- `tldw_Server_API/app/core/Moderation/family_wizard_materializer.py`
- `tldw_Server_API/app/core/Moderation/governance_io.py`

**Primary tests to inspect and selectively run:**
- `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py`
- `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
- `tldw_Server_API/tests/unit/test_moderation_env_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_etag_handling.py`
- `tldw_Server_API/tests/unit/test_moderation_redact_categories.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_contract.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`

**Scratch artifacts allowed during execution:**
- `/tmp/moderation_review_inventory.txt`
- `/tmp/moderation_policy_pytest.log`
- `/tmp/moderation_api_pytest.log`
- `/tmp/moderation_persistence_pytest.log`
- `/tmp/moderation_integration_pytest.log`

## Stage Overview

## Stage 1: Baseline and Review Contract
**Goal:** Lock the dirty-worktree baseline, create stable review artifact files, and fix the final report structure before deep reading begins.
**Success Criteria:** Review notes exist under `Docs/superpowers/reviews/moderation-backend/`, the source and test inventory is captured, and the final response contract is fixed before any candidate finding is treated as actionable.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: Policy Construction and Rule Parsing Pass
**Goal:** Inspect moderation policy loading, override merging, blocklist parsing, action handling, category handling, redaction, and sanitized snippet behavior inside `moderation_service.py`.
**Success Criteria:** Candidate findings about merge behavior, parsing, snippets, redaction, or regex safety are tied to exact service paths and cross-checked against the relevant unit tests.
**Tests:** The policy, parsing, snippet, category, environment, and effective-settings unit tests listed in this plan.
**Status:** Not Started

## Stage 3: Endpoints, Caller Path, and Permissions Pass
**Goal:** Trace the moderation admin endpoints, schema contracts, permission gating, tester behavior, and the real moderation call sites in `chat.py`.
**Success Criteria:** API-visible invariants, auth assumptions, and any mismatch between admin/test behavior and the live caller path are recorded with exact file references and test evidence.
**Tests:** The moderation permissions, tester sample, and selected chat moderation integration tests listed in this plan.
**Status:** Not Started

## Stage 4: Persistence, Concurrency, and Targeted Verification Pass
**Goal:** Validate user-override persistence, blocklist writes, reload semantics, `If-Match` and `ETag` handling, atomicity assumptions, and failure modes.
**Success Criteria:** Persistence and concurrency claims are supported by both source inspection and the narrowest useful verification instead of source reading alone.
**Tests:** The ETag, override contract and validation, runtime override, and selected integration tests listed in this plan.
**Status:** Not Started

## Stage 5: Test-Gap Pass and Final Synthesis
**Goal:** Compare reviewed behavior against the moderation test surface, identify missing or weak invariants, and produce the final ranked report.
**Success Criteria:** The final output separates confirmed findings, probable risks, improvements, and open questions, and every major claim is backed by source inspection, tests, targeted verification, or an explicit confidence downgrade.
**Tests:** Only additional narrow slices needed to settle unresolved claims.
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Capture the Baseline

**Files:**
- Create: `Docs/superpowers/reviews/moderation-backend/README.md`
- Create: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage1-baseline-and-inventory.md`
- Create: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md`
- Create: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md`
- Create: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md`
- Create: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage5-test-gaps-and-final-synthesis.md`
- Inspect: `Docs/superpowers/specs/2026-04-07-moderation-backend-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-moderation-backend-review-execution-plan.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/moderation-backend
```

Expected: the `Docs/superpowers/reviews/moderation-backend` directory exists and no application source files change.

- [ ] **Step 2: Create one markdown file per stage with a fixed evidence template**

Each stage file should contain:
```markdown
# Stage N Title

## Scope
## Files Reviewed
## Tests Reviewed
## Validation Commands
## Confirmed Findings
## Probable Risks
## Improvements
## Open Questions
## Exit Note
```

- [ ] **Step 3: Write `Docs/superpowers/reviews/moderation-backend/README.md`**

Document:
- the stage order `1 -> 2 -> 3 -> 4 -> 5`
- the path to each stage report
- the rule that confirmed findings come before probable risks and improvements
- the rule that persistence and concurrency claims need targeted verification or an explicit confidence downgrade
- the rule that `chat.py` is inspected only at moderation call sites, not as a general chat review

- [ ] **Step 4: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
git rev-parse --short HEAD
git log --oneline -n 20 -- tldw_Server_API/app/core/Moderation tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: a current workspace baseline that makes it clear whether any later finding depends on local edits or only on committed history.

- [ ] **Step 5: Capture the exact source and test inventory**

Run:
```bash
{
  rg --files tldw_Server_API/app/core/Moderation tldw_Server_API/app/api/v1 | rg 'Moderation|moderation|chat\.py$'
  rg --files tldw_Server_API/tests | rg 'moderation'
} | sort | tee /tmp/moderation_review_inventory.txt
```

Expected: one stable inventory file showing the in-scope source and moderation-focused test surface without frontend files.

- [ ] **Step 6: Write the Stage 1 baseline note**

`Docs/superpowers/reviews/moderation-backend/2026-04-07-stage1-baseline-and-inventory.md` must record:
- the dirty-worktree baseline from Step 4
- the short `HEAD` hash from Step 4
- the exact source and test inventory from Step 5
- whether any moderation-related local edits are already present
- the fixed final response structure from Step 7

- [ ] **Step 7: Freeze the final review output contract before deep reading**

Use this exact final response structure:
```markdown
## Findings
### Confirmed findings
- severity, confidence, file references, impact, and fix direction when clear

### Probable risks
- material issues not fully proven, with explicit confidence limits

## Open Questions
- only unresolved ambiguities that materially affect confidence

## Improvements
- lower-priority hardening or maintainability suggestions

## Verification
- files inspected, tests run, and what remains unverified
```

- [ ] **Step 8: Verify the workspace starts in a safe state**

Run:
```bash
git diff --name-only | rg 'tldw_Server_API/app/core/Moderation|tldw_Server_API/app/api/v1/endpoints/moderation.py|tldw_Server_API/app/api/v1/schemas/moderation_schemas.py|tldw_Server_API/app/api/v1/endpoints/chat.py|tldw_Server_API/tests/.+moderation'
```

Expected: either no moderation-related working-tree edits, or a short list that must be called out explicitly in Stage 1 before the review continues.

### Task 2: Execute the Policy Construction and Rule Parsing Pass

**Files:**
- Modify: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md`
- Inspect: `Docs/Code_Documentation/Moderation-Guardrails.md`
- Inspect: `tldw_Server_API/app/core/Moderation/README.md`
- Inspect: `tldw_Server_API/Config_Files/moderation_blocklist.txt`
- Inspect: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_env_parse.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_redact_categories.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`

- [ ] **Step 1: Read the moderation docs and example blocklist first**

Run:
```bash
sed -n '1,260p' Docs/Code_Documentation/Moderation-Guardrails.md
sed -n '1,220p' tldw_Server_API/app/core/Moderation/README.md
sed -n '1,220p' tldw_Server_API/Config_Files/moderation_blocklist.txt
```

Expected: a concrete documented contract for grammar, categories, runtime overrides, and tester behavior before source code assumptions are made.

- [ ] **Step 2: Read the unit tests that define merge, parse, snippet, and category expectations**

Run:
```bash
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_effective_settings.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_env_parse.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_redact_categories.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py
```

Expected: a test-defined view of policy and parser behavior before tracing service internals.

- [ ] **Step 3: Map the moderation service hotspots before detailed reading**

Run:
```bash
rg -n '^    def ' tldw_Server_API/app/core/Moderation/moderation_service.py
rg -n '_load_global_policy|_load_runtime_overrides_file|get_effective_policy|_parse_rule_line|_split_action_directive|_load_block_patterns|_validate_override_rules_strict|check_text|build_sanitized_snippet|redact_text|evaluate_action|_find_match_span|_collect_rule_matches' tldw_Server_API/app/core/Moderation/moderation_service.py
```

Expected: a hotspot index that anchors the detailed read to the functions most likely to create moderation defects.

- [ ] **Step 4: Read the policy-loading, override, and parsing paths in the service**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/Moderation/moderation_service.py
sed -n '260,920p' tldw_Server_API/app/core/Moderation/moderation_service.py
sed -n '900,1238p' tldw_Server_API/app/core/Moderation/moderation_service.py
```

Expected: a complete trace of policy construction, rule parsing, category filtering, snippet generation, and action evaluation before any finding is written.

- [ ] **Step 5: Run the narrow unit verification slice for policy and parser semantics**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py \
  tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py \
  tldw_Server_API/tests/unit/test_moderation_effective_settings.py \
  tldw_Server_API/tests/unit/test_moderation_env_parse.py \
  tldw_Server_API/tests/unit/test_moderation_redact_categories.py \
  tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py | tee /tmp/moderation_policy_pytest.log
```

Expected: a short pytest summary showing either green coverage for the inspected semantics or a directly relevant failure that must be reconciled against source before it becomes a finding.

- [ ] **Step 6: Record Stage 2 findings in the stage note**

Capture:
- merge behavior observations
- parse and category semantics
- snippet or redaction surprises
- any discrepancy between docs, tests, and service behavior
- explicit no-finding notes where the code is stronger than expected

### Task 3: Execute the Endpoints, Caller Path, and Permissions Pass

**Files:**
- Modify: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md`
- Inspect: `Docs/Code_Documentation/Moderation-Guardrails.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`

- [ ] **Step 1: Read the endpoint and schema contracts**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/api/v1/schemas/moderation_schemas.py
sed -n '1,460p' tldw_Server_API/app/api/v1/endpoints/moderation.py
```

Expected: the full request, response, and error contract for moderation admin APIs and the tester surface.

- [ ] **Step 2: Read the permission and tester tests before tracing the caller path**

Run:
```bash
sed -n '1,260p' tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py
sed -n '1,320p' tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py
sed -n '1,260p' tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py
```

Expected: a test-defined picture of endpoint permissions, tester sample behavior, and live chat moderation expectations.

- [ ] **Step 3: Trace the real moderation call sites in `chat.py`**

Run:
```bash
rg -n 'get_moderation_service|evaluate_action|check_text|redact_text|moderation' tldw_Server_API/app/api/v1/endpoints/chat.py
sed -n '2488,2556p' tldw_Server_API/app/api/v1/endpoints/chat.py
sed -n '2978,3010p' tldw_Server_API/app/api/v1/endpoints/chat.py
sed -n '3848,3985p' tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: the concrete input and output moderation call flow used by chat, not just the admin tester approximation.

- [ ] **Step 4: Compare endpoint, schema, tester, and caller semantics**

Check explicitly:
- whether the tester reflects the same phase and action semantics as the chat caller
- whether schema validation is stricter or looser than the service actually expects
- whether permission boundaries are enforced at the router level and reflected in tests
- whether documented behavior matches the response and error payloads actually returned

- [ ] **Step 5: Run the endpoint and caller verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py \
  tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py | tee /tmp/moderation_api_pytest.log
```

Expected: one focused pytest summary that either confirms the reviewed API and caller contracts or exposes a concrete mismatch worth deeper inspection.

- [ ] **Step 6: Record Stage 3 findings in the stage note**

Capture:
- endpoint and schema contract findings
- permission or tester-surface issues
- caller-path mismatches between `chat.py` and admin moderation behavior
- any surprising but intentional behavior that should be downgraded to improvement or open question instead of defect

### Task 4: Execute the Persistence, Concurrency, and Targeted Verification Pass

**Files:**
- Modify: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md`
- Inspect: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_etag_handling.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_user_override_contract.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`

- [ ] **Step 1: Read the persistence and optimistic-concurrency code paths directly**

Run:
```bash
rg -n 'set_user_override|delete_user_override|get_blocklist_state|set_blocklist_lines|append_blocklist_line|delete_blocklist_index|_save_runtime_overrides_file|_load_runtime_overrides_file|If-Match|ETag' \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/app/api/v1/endpoints/moderation.py
sed -n '560,760p' tldw_Server_API/app/core/Moderation/moderation_service.py
sed -n '1238,1415p' tldw_Server_API/app/core/Moderation/moderation_service.py
sed -n '258,410p' tldw_Server_API/app/api/v1/endpoints/moderation.py
```

Expected: a direct trace of file writes, atomic replace behavior, override persistence, version computation, and endpoint conflict handling.

- [ ] **Step 2: Read the unit tests that cover persistence and conflict contracts**

Run:
```bash
sed -n '1,240p' tldw_Server_API/tests/unit/test_moderation_etag_handling.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_user_override_contract.py
sed -n '1,260p' tldw_Server_API/tests/unit/test_moderation_user_override_validation.py
sed -n '1,220p' tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py
```

Expected: a test-defined view of what persistence and conflict behavior is already claimed and what is still unproven.

- [ ] **Step 3: Run the persistence and conflict verification slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/unit/test_moderation_etag_handling.py \
  tldw_Server_API/tests/unit/test_moderation_user_override_contract.py \
  tldw_Server_API/tests/unit/test_moderation_user_override_validation.py \
  tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py | tee /tmp/moderation_persistence_pytest.log
```

Expected: a narrow pytest summary that confirms or weakens claims about persistence, validation, and conflict handling without dragging in unrelated suites.

- [ ] **Step 4: Escalate only unresolved persistence questions to a narrower rerun**

Use one of these exact patterns only if a Stage 4 claim remains unsettled after the prior command:
```bash
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/unit/test_moderation_etag_handling.py -k if_match -vv
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/unit/test_moderation_user_override_contract.py -k persists -vv
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/unit/test_moderation_user_override_validation.py -k rejects -vv
```

Expected: one smaller reproduction or confirmation run tied to a single unsettled claim, not a broader retest.

- [ ] **Step 5: Record Stage 4 findings in the stage note**

Capture:
- persistence guarantees that are actually supported
- atomicity and reload assumptions that are only partial
- `If-Match` or `ETag` behavior that is better or weaker than expected
- any remaining uncertainty that must be labeled as `Probable risk` or `Open question`

### Task 5: Execute the Test-Gap Pass and Final Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage5-test-gaps-and-final-synthesis.md`
- Inspect: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage1-baseline-and-inventory.md`
- Inspect: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md`
- Inspect: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md`
- Inspect: `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md`
- Test: all moderation-focused tests already named in this plan, reusing only the smallest additional slice needed

- [ ] **Step 1: Reconcile stage notes and remove duplicate findings**

Read:
```bash
sed -n '1,260p' Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md
sed -n '1,260p' Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md
sed -n '1,260p' Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md
```

Expected: one deduplicated issue set where the same risk is not reported separately as parser, endpoint, and caller defects unless those are truly distinct failures.

- [ ] **Step 2: Identify the highest-value missing or weakly asserted moderation invariants**

Check explicitly for gaps around:
- user-override merge behavior not covered by direct tests
- tester versus caller-path semantic divergence
- partial-write or reload failure handling
- category and phase interactions across multiple rule sources
- permission boundaries on the admin routes

- [ ] **Step 3: Run only any final narrow pytest slice needed to settle a live dispute**

Choose from the already named moderation files only. Examples:
```bash
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py -k streaming -vv | tee /tmp/moderation_integration_pytest.log
source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py -vv | tee /tmp/moderation_integration_pytest.log
```

Expected: one last targeted confirmation step, not a blanket rerun of every moderation test.

- [ ] **Step 4: Write the final synthesis note**

`Docs/superpowers/reviews/moderation-backend/2026-04-07-stage5-test-gaps-and-final-synthesis.md` must contain:
```markdown
# Stage 5 Test Gaps and Final Synthesis

## Confirmed Findings
## Probable Risks
## Improvements
## Open Questions
## Verification Summary
## Final Response Draft
```

- [ ] **Step 5: Prepare the final user-facing review output**

The in-session final answer must:
- start with findings, ordered by severity
- use file references for each finding
- keep confirmed findings separate from probable risks
- state explicitly when no finding exists in a reviewed area
- include a short verification section naming the tests run and any important limits

- [ ] **Step 6: Perform a final confidence check before sending**

Verify:
- every confirmed finding is backed by code inspection, test inspection, executed verification, or a combination of them
- every probable risk has an explicit uncertainty statement
- every improvement is lower priority than the findings above it
- nothing in the final answer drifts into frontend moderation or unrelated chat review
