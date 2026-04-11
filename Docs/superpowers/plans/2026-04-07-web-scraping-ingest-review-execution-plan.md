# Web_Scraping Ingest Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved `Web_Scraping` ingest-path audit and deliver one consolidated, evidence-backed review covering both public ingest routes, service orchestration, fallback behavior, persistence modes, request safety, validation coverage, and remediation priorities.

**Architecture:** This is a read-first, call-graph-driven audit plan. Execution starts from the two public ingest entrypoints and their request models, follows only reachable code paths into services and core fetch or extraction helpers, records stage findings under `Docs/superpowers/reviews/web-scraping-ingest/`, uses targeted tests and a narrow Bandit supplement to confirm high-risk claims, and finishes with one ranked synthesis that separates confirmed defects, probable risks, and improvements.

**Tech Stack:** Python 3, FastAPI, pytest, Bandit, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/web-scraping-ingest/README.md`
- `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage1-inventory-and-callgraph.md`
- `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage2-public-entrypoints-and-schema.md`
- `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage3-services-fallback-and-persistence.md`
- `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage4-reachable-core-and-request-safety.md`
- `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage5-validation-gaps-and-synthesis.md`

**Primary source files to inspect during the review:**
- `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- `tldw_Server_API/app/services/web_scraping_service.py`
- `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- `tldw_Server_API/app/services/ephemeral_store.py`
- `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
- `tldw_Server_API/app/core/Web_Scraping/filters.py`
- `tldw_Server_API/app/core/Web_Scraping/scoring.py`
- `tldw_Server_API/app/core/http_client.py`
- `tldw_Server_API/app/core/Security/egress.py`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`
- `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`
- `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
- `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`
- `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`
- `tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py`

## Stage Overview

## Stage 1: Review Artifact Setup and Call-Graph Inventory
**Goal:** Create stable review artifacts, capture the exact scoped source and test surface, and freeze the audit output structure before deep reading begins.
**Success Criteria:** Review files exist under `Docs/superpowers/reviews/web-scraping-ingest/`, the scoped source and test inventory is recorded, and the initial ingest call graph is documented.
**Tests:** Tooling/version checks only
**Status:** Complete

## Stage 2: Public Entrypoints and Request-Model Contract Review
**Goal:** Review both public ingest routes and their request-shaping logic for validation, coercion, cookie parsing, forwarding, and client-facing error semantics.
**Success Criteria:** The public contract for `/api/v1/media/ingest-web-content` and `/api/v1/media/process-web-scraping` is documented with confirmed findings or explicit no-finding notes, backed by targeted request-path tests.
**Tests:** Friendly ingest, cookie parsing, strategy validation, usage logging, and custom-header tests
**Status:** Complete

## Stage 3: Service Orchestration, Fallback, and Persistence-Mode Review
**Goal:** Review the orchestration layer for normalization, mode-specific behavior, fallback gating, persistence metadata, and error translation.
**Success Criteria:** The enhanced path, fallback path, `ephemeral` mode, and `persist` mode are traced end to end and any contract mismatches or reliability issues are documented with evidence.
**Tests:** Config precedence, legacy fallback behavior, persistence crawl metadata
**Status:** Complete

## Stage 4: Reachable Core Fetch Path and Request-Safety Review
**Goal:** Follow only the reachable fetch or extraction helpers from the ingest path and verify request safety, robots behavior, header and cookie propagation, and outbound policy enforcement.
**Success Criteria:** Every reachable outbound path reviewed in this stage is classified as confirmed-safe, confirmed-unsafe, or probable-risk with evidence, and Bandit output is triaged rather than dumped verbatim.
**Tests:** HTTP client, robots, and optional filter or router validations when required to settle a claim
**Status:** Complete

## Stage 5: Final Validation Pass, Coverage Gaps, and Ranked Synthesis
**Goal:** Compare the reviewed code paths against the existing test surface, run only the narrowest extra validations needed to settle disputed claims, and produce the final ranked review.
**Success Criteria:** Findings are de-duplicated, coverage gaps are prioritized, remediation is grouped by urgency, and the final review matches the approved output model.
**Tests:** Optional e2e smoke when the environment supports it, plus any narrow follow-up commands needed to confirm unresolved claims
**Status:** Complete

### Task 1: Prepare Review Artifacts and Baseline Inventory

**Files:**
- Create: `Docs/superpowers/reviews/web-scraping-ingest/README.md`
- Create: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage1-inventory-and-callgraph.md`
- Create: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage2-public-entrypoints-and-schema.md`
- Create: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage3-services-fallback-and-persistence.md`
- Create: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage4-reachable-core-and-request-safety.md`
- Create: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage5-validation-gaps-and-synthesis.md`
- Modify: `Docs/superpowers/plans/2026-04-07-web-scraping-ingest-review-execution-plan.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/web-scraping-ingest
```

Expected: the `Docs/superpowers/reviews/web-scraping-ingest` directory exists and no source files under `tldw_Server_API/app/` change.

- [ ] **Step 2: Create one markdown file per stage with a fixed review template**

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

- [ ] **Step 3: Write `Docs/superpowers/reviews/web-scraping-ingest/README.md`**

Document:
```markdown
# Web_Scraping Ingest Review Artifacts

## Stage Order
1. Inventory and call graph
2. Public entrypoints and schema
3. Services, fallback, and persistence
4. Reachable core and request safety
5. Validation gaps and synthesis

## Review Rules
- Findings come before remediation ideas.
- Uncertain claims must be labeled as assumptions or probable risks.
- Security-sensitive claims require direct path proof, targeted validation, or an explicit confidence downgrade.
- Only reachable code paths from the approved ingest scope belong in this review.
```

- [ ] **Step 4: Verify the execution environment before deep reading**

Run:
```bash
source .venv/bin/activate
python -m pytest --version
python -m bandit --version
```

Expected: `pytest` and `bandit` print version information. If `bandit` is unavailable, stop and record that blocker in `2026-04-07-stage1-inventory-and-callgraph.md` before continuing with static security review only.

- [ ] **Step 5: Capture the scoped source and test inventory**

Run:
```bash
source .venv/bin/activate
rg --files \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/app/api/v1/schemas/media_request_models.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/ephemeral_store.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  tldw_Server_API/app/core/Web_Scraping/scraper_router.py \
  tldw_Server_API/app/core/Web_Scraping/filters.py \
  tldw_Server_API/app/core/Web_Scraping/scoring.py \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Security/egress.py \
  tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py \
  tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py \
  tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py \
  tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py \
  tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py \
  tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py \
  tldw_Server_API/tests/Web_Scraping/test_router_validation.py \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py \
  tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py | sort
```

Expected: a stable inventory that includes the approved ingest-path files and targeted test modules without expanding into unrelated application areas beyond directly referenced helpers.

- [ ] **Step 6: Capture the recent churn baseline for the ingest path**

Run:
```bash
git log --oneline -n 20 -- \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/core/Web_Scraping
```

Expected: a first-pass change window that highlights recent refactors around fallback, persistence, and crawl behavior without requiring full-history archaeology.

- [ ] **Step 7: Record the initial call-graph and final output structure in the stage 1 report**

Write:
```markdown
## Initial Ingest Call Graph
- `/api/v1/media/ingest-web-content` -> `ingest_web_content_orchestrate()` -> reachable scrape helpers
- `/api/v1/media/process-web-scraping` -> `process_web_scraping_task()` -> enhanced service or legacy fallback

## Final Review Output Shape
## Findings
1. Severity: concise finding with file references and impact

## Open Questions
- only unresolved assumptions
```

- [ ] **Step 8: Commit the review scaffold**

Run:
```bash
git add Docs/superpowers/reviews/web-scraping-ingest Docs/superpowers/plans/2026-04-07-web-scraping-ingest-review-execution-plan.md
git commit -m "docs: scaffold web scraping ingest review artifacts"
```

Expected: one docs-only commit captures the review workspace before substantive findings are added.

### Task 2: Execute Stage 2 Public Entrypoints and Request-Model Review

**Files:**
- Modify: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage2-public-entrypoints-and-schema.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Inspect: `tldw_Server_API/app/services/web_scraping_service.py:517`
- Test: `tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py`
- Test: `tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py`
- Test: `tldw_Server_API/tests/WebScraping/test_custom_headers_support.py`

- [ ] **Step 1: Read the route definitions, request models, and route-to-service glue**

Run:
```bash
sed -n '1,240p' tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py
sed -n '1,220p' tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py
rg -n "class IngestWebContentRequest|class WebScrapingRequest|field_validator|use_cookies|cookies|custom_headers|crawl_strategy|include_external|score_threshold" tldw_Server_API/app/api/v1/schemas/media_request_models.py
rg -n "ingest_web_content_orchestrate|process_web_scraping_task" tldw_Server_API/app/services/web_scraping_service.py
```

Expected: a compact map of request validation, coercion, forwarding, and the exact service entrypoints for each public route.

- [ ] **Step 2: Write the public contract table into the stage 2 report**

Write:
```markdown
## Public Contract Table
| Route | Request model | Key coercions or validators | Downstream service | Error translation notes |
| --- | --- | --- | --- | --- |
| `/api/v1/media/ingest-web-content` | `IngestWebContentRequest` | cookies, scrape-method aliasing, URL presence | `ingest_web_content_orchestrate()` | preserves `HTTPException`, wraps unexpected errors |
| `/api/v1/media/process-web-scraping` | `WebScrapingRequest` | crawl strategy, headers, cookies, mode | `process_web_scraping_task()` | preserves downstream `HTTPException`, wraps unexpected errors |
```

- [ ] **Step 3: Run the request-path and forwarding tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_friendly_ingest_crawl_flags.py \
  tldw_Server_API/tests/Web_Scraping/test_ingest_cookie_parsing.py \
  tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py \
  tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py \
  tldw_Server_API/tests/WebScraping/test_custom_headers_support.py
```

Expected: PASS, or a clearly documented runtime-backed finding if any request-path contract fails. Any failure here is high-value because it directly weakens the public ingest contract or its observability.

- [ ] **Step 4: Record findings with explicit evidence labels**

Use this finding template in the stage 2 report:
```markdown
## Findings
1. **Severity | Confidence | Route or schema boundary**
   - Files: `path:line`
   - Evidence: failing test | runtime proof | direct code-path proof | probable risk
   - Why it matters: one production-facing sentence
   - Recommended fix direction: one sentence
```

- [ ] **Step 5: Record explicit no-finding notes for validated boundaries**

Write:
```markdown
## Exit Note
- Verified cookie parsing behavior for friendly ingest.
- Verified `process-web-scraping` preserves 400s for invalid crawl strategy.
- Verified usage logging coverage.
- Either verified custom-header forwarding coverage or recorded the reproducible failure that blocked that verification.
```

- [ ] **Step 6: Commit the stage 2 review notes**

Run:
```bash
git add Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage2-public-entrypoints-and-schema.md
git commit -m "docs: add stage 2 web scraping ingest review notes"
```

Expected: one docs-only commit captures the public contract review before service-level notes begin.

### Task 3: Execute Stage 3 Service Orchestration, Fallback, and Persistence Review

**Files:**
- Modify: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage3-services-fallback-and-persistence.md`
- Inspect: `tldw_Server_API/app/services/web_scraping_service.py`
- Inspect: `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- Inspect: `tldw_Server_API/app/services/ephemeral_store.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`

- [ ] **Step 1: Read the orchestration, fallback, and storage entrypoints**

Run:
```bash
rg -n "async def process_web_scraping_task|async def ingest_web_content_orchestrate|_store_ephemeral|_store_persistent|_legacy_web_scraping_fallback_enabled|_collect_fallback_unsupported_controls" tldw_Server_API/app/services/web_scraping_service.py tldw_Server_API/app/services/enhanced_web_scraping_service.py
sed -n '133,940p' tldw_Server_API/app/services/web_scraping_service.py
sed -n '1,760p' tldw_Server_API/app/services/enhanced_web_scraping_service.py
```

Expected: a readable map of normalization, enhanced dispatch, fallback gating, task-id handling, and mode-specific storage behavior.

- [ ] **Step 2: Write the service control-flow summary into the stage 3 report**

Write:
```markdown
## Service Control Flow
- `process_web_scraping_task()` validates crawl controls, then dispatches to the enhanced service.
- Enhanced-service failures may trigger the legacy fallback depending on environment gating.
- `mode="ephemeral"` and `mode="persist"` diverge in storage shape, metadata persistence, and response payload details.

## Failure-Seam Checklist
- input normalization
- fallback eligibility
- degraded-control handling
- metadata persistence
- error translation
```

- [ ] **Step 3: Run the service-layer validation tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py
```

Expected: PASS. Any failure here is a high-confidence contract or reliability issue in the ingest orchestration layer.

- [ ] **Step 4: Inspect the exact storage helpers only if a stage 3 claim depends on them**

Run:
```bash
rg -n "class EphemeralStorage|store_data|get_data|managed_media_database|get_media_repository" tldw_Server_API/app/services/ephemeral_store.py tldw_Server_API/app/core/DB_Management/media_db/api.py
sed -n '1,260p' tldw_Server_API/app/services/ephemeral_store.py
sed -n '1,260p' tldw_Server_API/app/core/DB_Management/media_db/api.py
```

Expected: enough context to confirm or downgrade claims about TTL behavior, persistence lifecycle, or repository ownership without expanding into a full DB-management audit.

- [ ] **Step 5: Record confirmed findings, probable risks, and explicit no-finding notes**

Write:
```markdown
## Exit Note
- Confirmed behavior for config-default versus request-override precedence.
- Confirmed or downgraded claims around fallback gating and degraded controls.
- Confirmed or downgraded claims around crawl metadata persistence in `persist` mode.
```

- [ ] **Step 6: Commit the stage 3 review notes**

Run:
```bash
git add Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage3-services-fallback-and-persistence.md
git commit -m "docs: add stage 3 web scraping ingest review notes"
```

Expected: one docs-only commit captures the orchestration and persistence review.

### Task 4: Execute Stage 4 Reachable Core and Request-Safety Review

**Files:**
- Modify: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage4-reachable-core-and-request-safety.md`
- Inspect: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Inspect: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Inspect: `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
- Inspect: `tldw_Server_API/app/core/Web_Scraping/filters.py`
- Inspect: `tldw_Server_API/app/core/Web_Scraping/scoring.py`
- Inspect: `tldw_Server_API/app/core/http_client.py`
- Inspect: `tldw_Server_API/app/core/Security/egress.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_router_validation.py`

- [ ] **Step 1: Trace the reachable outbound fetch path from the services into the core helpers**

Run:
```bash
rg -n "scrape_article|recursive_scrape|scrape_from_sitemap|scrape_by_url_level|http_fetch|fetch\\(|evaluate_url_policy|is_allowed_by_robots" \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Security/egress.py
```

Expected: a concrete list of reachable fetch and policy functions to inspect, with no need to review unrelated web-search code.

- [ ] **Step 2: Read the request-safety and policy enforcement points**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/http_client.py
sed -n '1,260p' tldw_Server_API/app/core/Security/egress.py
rg -n "robots|cookie|custom_headers|user_agent|evaluate_url_policy|http_fetch|fetch\\(" tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py
```

Expected: enough evidence to classify each reachable outbound path as enforced, bypassable, or ambiguous.

- [ ] **Step 3: Run the direct request-safety validation tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py \
  tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py
```

Expected: PASS. Any failure here weakens confidence in outbound safety or robots behavior.

- [ ] **Step 4: Run the narrow follow-up tests only if a stage 4 claim depends on filter or router semantics**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py \
  tldw_Server_API/tests/Web_Scraping/test_router_validation.py
```

Expected: PASS when executed. Skip this step only if stage 4 findings do not rely on filter or router semantics, and record that skip decision in the stage 4 report.

- [ ] **Step 5: Run Bandit on the touched review scope and capture machine-readable output**

Run:
```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/media/ingest_web_content.py \
  tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py \
  tldw_Server_API/app/api/v1/schemas/media_request_models.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Web_Scraping \
  -f json -o /tmp/bandit_web_scraping_ingest_review.json
```

Expected: `/tmp/bandit_web_scraping_ingest_review.json` is written when `bandit` is available. If `bandit` is still unavailable in the shared project venv, record that blocker in the Stage 4 note and continue with static request-safety review only. When the JSON is available, findings are triaged and summarized in prose; the JSON file itself is not the final review.

- [ ] **Step 6: Record security findings and confidence downgrades**

Write:
```markdown
## Findings
1. **Severity | Confidence | Request-safety issue**
   - Files: `path:line`
   - Evidence: direct code-path proof | failing test | Bandit + manual triage
   - Why it matters: one sentence
   - Confidence downgrade note: include this only when enforcement is ambiguous rather than proven weak
```

- [ ] **Step 7: Commit the stage 4 review notes**

Run:
```bash
git add Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage4-reachable-core-and-request-safety.md
git commit -m "docs: add stage 4 web scraping request-safety review notes"
```

Expected: one docs-only commit captures the reachable core and security review.

### Task 5: Execute Stage 5 Final Validation, Coverage-Gap Review, and Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/web-scraping-ingest/README.md`
- Modify: `Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage5-validation-gaps-and-synthesis.md`
- Inspect: all stage reports for de-duplication
- Test: `tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py`

- [ ] **Step 1: Run the optional end-to-end smoke only when the environment supports it**

Run:
```bash
source .venv/bin/activate
python -m pytest -m e2e -v tldw_Server_API/tests/server_e2e_tests/test_web_scraping_workflow.py
```

Expected: the local workflow passes when the e2e server fixtures are available; the external workflow may skip unless `TLDW_E2E_EXTERNAL_WEB_SCRAPE=1` is set. If the e2e environment is unavailable, record the skip condition rather than forcing the test.

- [ ] **Step 2: Write the final synthesis document**

Use this exact structure in `2026-04-07-stage5-validation-gaps-and-synthesis.md`:
```markdown
# Stage 5 Validation Gaps and Synthesis

## Findings
1. Severity-ranked, evidence-backed findings only

## Open Questions
- only unresolved assumptions that affect confidence

## Coverage Gaps
- covered and failing
- covered but weak
- uncovered

## Remediation Roadmap
- fix now
- fix soon
- opportunistic cleanup
```

- [ ] **Step 3: Cross-check the final synthesis against the approved spec**

Run:
```bash
rg -n "Goal|Primary Scope|Review Questions|Targeted Validation|Security And Request-Safety Audit|Evidence Rules" Docs/superpowers/specs/2026-04-07-web-scraping-ingest-review-design.md
```

Expected: every spec section maps to at least one stage report or explicit no-finding note. Add missing synthesis coverage before closing the review.

- [ ] **Step 4: Perform a placeholder and duplication scan across the review artifacts**

Run:
```bash
rg -n "TODO|TBD|fix later|implement later|placeholder|similar to|add appropriate|write tests for the above" Docs/superpowers/reviews/web-scraping-ingest
```

Expected: no placeholder language remains. If duplicates exist across stage files, collapse them into the strongest evidence location and cross-reference rather than repeating them.

- [ ] **Step 5: Update the review README with the final artifact status**

Write:
```markdown
## Final Status
- Stage 1 complete
- Stage 2 complete
- Stage 3 complete
- Stage 4 complete
- Stage 5 complete

## Final Ranked Review
- See `2026-04-07-stage5-validation-gaps-and-synthesis.md`
```

- [ ] **Step 6: Commit the final synthesis and README updates**

Run:
```bash
git add Docs/superpowers/reviews/web-scraping-ingest/README.md Docs/superpowers/reviews/web-scraping-ingest/2026-04-07-stage5-validation-gaps-and-synthesis.md
git commit -m "docs: finalize web scraping ingest review"
```

Expected: one docs-only commit captures the final ranked review and coverage-gap analysis.
