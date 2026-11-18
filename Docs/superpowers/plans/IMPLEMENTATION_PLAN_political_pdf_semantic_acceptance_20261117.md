# Political PDF semantic acceptance implementation plan

> **For agentic workers:** Use subagent-driven-development or executing-plans for each stage. Preserve stage order and review each change before proceeding.

**Goal:** Detect omitted overview chapters, repair verified footnote attribution context, qualify the changes with Luna and prepare honest human/VIP acceptance.

**Architecture:** Extend existing section-output Jobs, attribution context, Claims checks and eval/report paths. Derive structural coverage from the current text and registered section evidence. Keep policy evidence, attribution context, agent findings and human decisions separate.

**Tech Stack:** Python/FastAPI/pytest/Hypothesis; shared React/TypeScript/Vitest; existing CLI, local MiniLM and OpenAI GPT-6 Luna.

**Spec:** `Docs/Design/POLITICAL_PDF_SEMANTIC_ACCEPTANCE_2026_11_17.md`

**Tracking:** TASK-15505.12 (repairs and qualification), TASK-15505.6 (review and human acceptance). ADR required: no new ADR; proposed ADR-096 and ADR-003/007/015/025/065 govern existing boundaries.

## Global constraints

- Preserve original Job 52, all prior runs, first failures and reviewer corrections.
- No automatic paid retries, invented author assignments, new policy citation authority or new dependencies.
- Use the existing provider gateway, credentials, Jobs, artifact versions and owned Media APIs.
- Human labels stay null until a person supplies them. Agent source review cannot calibrate a judge against humans.
- Keep the PR draft and preserve text outside its managed update block.
- Activate `.venv` for Python, tests, Bandit and backlog-py. Use existing Node 20/React-deduplicated test setup where required and disclose its limits.

## Stage 1: Overview chapter representation

**Goal:** A missing cited chapter cannot be hidden by complete processing counts or a reused overview hash.
**Success Criteria:** Job52 deterministically reports Financial missing; complete, partial, retry and reduction paths behave correctly; reader/export/evals expose the same structural limitation independently of Claims verdicts.
**Status:** Complete

**Files:** `core/Research_Workspace/section_summaries.py`, a focused `section_overview_coverage.py` helper and `tests/Research_Workspace/test_section_overview_coverage.py`; existing section-job tests; shared UI section reader/status/export helpers and tests; `Helper_Scripts/Evals/political_pdf_eval_report.py` and its tests. Backend paths are under `tldw_Server_API/app/` and `tldw_Server_API/tests/`; frontend paths under `apps/packages/ui/src/`.

- [x] Write RED tests for two completed chapters with an overview citing only the first; assert failed/partial output, retained candidate, missing chapter title and no repeated extraction on explicit retry.
- [x] Add literal parser cases: prose citations cover their registered section; titles, code, bare citation inventories, ranges, unknown IDs and ambiguous bindings do not. Include duplicate titles, selected scope and failed/unread chapters.
- [x] Add reduction tests: a compression response dropping a chapter is rejected under unchanged citation bounds; final synthesis cannot omit a completed selected chapter; no-citation input fails before a paid call.
- [x] Implement the smallest pure coverage helper and overview-only reduction checks. Derive the mapping from section evidence; recompute before reuse and set successful input hash only after validation.
- [x] Wire truthful reader/status text, Markdown/offline export and eval metrics. Test a retained nonempty candidate plus error, legacy unchecked artifacts, save/restore and changed text.
- [x] Run targeted RED→GREEN suites, connected section/Claims/export tests, scoped lint/types/Bandit and an independent review. Record exact commands/results in Backlog and commit before stage 2.

**Verification:** Commit `10b6162644`. 319 connected backend tests, 121 eval tests, 88 focused frontend tests and 131 StudioPane tests; the final parser changes pass 54 tests. Full frontend TypeScript, scoped Ruff/ESLint and production Bandit pass. Independent backend/eval and frontend reviews are clear after two bounded parser review rounds. Frozen Job 52 reproduces 29/30 overview chapters, missing Financial; each of the 30 saved chapter inputs has cited prose. Live provider/browser qualification remains Stage 3. RED and final logs are recorded in TASK-15505.12.

**Test example:** an overview `Agency A proposes reform [E1.1.1].` over two completed sections with registered IDs `E1.1.1` and `E2.1.1` must expose section 2 as missing even when both chapters and all portions completed. A later explicit retry returning cited prose for both must retain prior completed chapter dictionaries and avoid extraction calls.

## Stage 2: Literal footnote attribution linkage

**Goal:** Claims and generation receive verified note/backreference context without broadening policy evidence.
**Success Criteria:** Exact Financial note53 resolves to its verified Bowes contribution, ambiguity remains unresolved, and all summary/QA callers receive the same bounded context.
**Status:** Complete

**Files:** `core/Research_Workspace/section_attribution.py`, `section_verification.py`, `section_summaries.py`; `core/Claims_Extraction/workspace_citations.py`, `claims_engine.py`; existing attribution, section-Claims, citation API and Claims unit tests. Add a new versioned political regression fixture through the existing loader.

- [x] Add RED source fixtures for a citation starting after its note label, a shared reference list following multiple contributors, and multiple notes in one cited passage. Assert exact linked marker/heading excerpts, not inferred author strings.
- [x] Add controls for duplicate/inline labels, absent boundaries, backlinks crossing contributors, unrelated bibliography names, malformed/stale navigation, source scope, author caveats and bounded overflow.
- [x] Extend the existing context builder/renderer and QA resolver; retain current policy passages and citation allowlists. Update existing verifier guidance for explicit linked attribution and absence claims.
- [x] Verify context reaches extraction, chapter/overview and QA; changed context invalidates cached checks; existing saved findings/export remain correctly bound.
- [x] Run connected tests, parser properties, scoped lint/Bandit and independent review; commit the qualified repair.

**Verification:** Final combined suite: 534 passed, 10 existing warnings, 48.30 seconds. Scoped Ruff, Bandit and independent review pass. Source audit preserves 2,679 body-only contexts and the 15,916-character overview author notes. Financial note 53 links to the exact Bowes contribution; 55 of 56 notes resolve and the final note remains unresolved without a closing boundary. Agriculture/Energy omit only oversized footnote maps explicitly. The v2 fixture retains all eight previous controls and adds four agent-reviewed attribution controls; human decisions remain null. Logs: `/tmp/political-semantic-stage2-combined.log`, `/tmp/political-semantic-stage2-bandit.json`, `/tmp/footnote-core-final-source-audit.json`.

**Test example:** body `Contribution B\nAuthor B\nTransfer is proposed[53].` and uniquely bounded `53. Transfer details.\n54. Another reference.` must link a cited substring after `53. ` to the exact marker and contribution. Repeating `[53]` under a different contributor must not manufacture a single-author result.

## Stage 3: Frozen Luna retest and source-review packet

**Goal:** Evaluate the repairs without replacing earlier failures or calling agent review human calibration.
**Success Criteria:** Versioned controls and targeted cases have retained first results; the same-PDF full run is structurally audited; exact per-finding review items preserve decision provenance and reject stale/unlabeled calibration inputs.
**Status:** In Progress

**Files:** existing `Helper_Scripts/Evals/` runner/reporter/configs, `tests/Evaluations/` regressions, versioned political fixtures, `Docs/Development/POLITICAL_PDF_MODEL_EVALS.md` and a dated retest/review report.

- [x] Extend offline reporting and its tests for exact per-finding samples, independent agent proposals and nullable human labels; preserve decisions only when their evidence fingerprint is unchanged.
- [x] Freeze new config/run directories and API/worker/runner revisions. Plan the run, then execute the new frozen verification controls and targeted QA/summary cases with no automatic retry.
- [x] Repair the Job 55 evidence-ID/source-range handoff exposed by the targeted run. Bind only validated note-overlapping evidence allowed in each chapter reduction/final call, preserve context budgets and retry work, add RED→GREEN controls and independent review, then freeze a new runtime for explicit retesting.
- [x] Preserve Job 56's first results; repair the separately reproduced Claims aggregation gap with explicit citation/context bindings and context-sensitive cache invalidation. Retain old fixtures and add v4 mixed-contributor controls; qualify with connected tests and independent review before restarting the runtime.
- [x] Run one full Luna evaluation on the original PDF. Retain first failures, ledger entries and any explicit recovery; audit chapter representation, source spans, Claims verdicts and attribution separately.
- [x] After the frozen full run is terminal, apply the bounded comma/semicolon page-annotation repair exposed by browser R04. Qualify shared parsing, historical unchecked receipts and rejection of evaluated promotions; recheck the saved failed answer and fresh browser flow with separate receipts.
- [x] Repair the independently confirmed multi-passage citation handoff (TASK-15505.19) using the existing grouped, qualified claim renderer at chapter synthesis/reduction. Preserve the frozen full run; qualify fresh and saved-partial paths, then run a separate White House/EXIM/Financial targeted Luna check and identify its later revision explicitly.
- [x] Correct the observed backward progress updates (TASK-15505.20) within the same attempt, retaining stage messages and true completion/failure status; qualify uneven chapter sizes, retry behavior and the targeted run's ledger.
- [x] Bind all 180 reviewed control/QA/White House decisions to Job 57's final partial packet; verify offline regeneration preserves them and all 664 raw evidence files remain unchanged.
- [x] Recover the reproduced model-list transport timeout under TASK-15505.21, without retries of paid generation or weakened inventory/citation validation; resume saved work explicitly into fresh ledgers after runtime qualification.
- [x] Source-review the agreed sample as agent review, keeping known regressions separate from untouched samples. Do not report human calibration scores while labels are absent.
- [x] Exercise review/save/reload/export of new findings and record NN/g issues and verification limits.
- [ ] Publish tested changes/results to the existing draft PR; follow CI to terminal.

**Latest evidence:** Original Job 57 retains its first partial failure and all 180 fingerprint-bound agent reviews; all 15 v4 controls pass, with 39 retrieval probes and 26 QA answers complete. Qualified inventory recovery at `d8b21d7abf` allows target Job 60 to complete three chapters/eight portions and overview 3/3, with exact source spans and no execution errors. Its 114 chapter/overview units are agent-reviewed; four review notes survive browser save/reload/Markdown export. Full recovery Job 61 completes 30/30 chapters and 117/117 portions, with overview 30/30, 148 valid reports and 3,525 exact quotes. It preserves the original 11 chapter records, 46 portion records and the text/evidence of two Energy portions whose failed checks are revalidated. All 40 final overview units are agent-reviewed and fingerprint-bound: seven supported, 33 needing revision. Strict semantic acceptance still fails; unreviewed rows remain labeled honestly. One separate v5 run passes all 17 fixed expectations on first attempts. Saved/fresh browser answers each check 2/2 units and reject wrong attribution; UX106 passes 454 connected tests and original-context reload. Full overview/White House review notes survive ordinary reload. TASK-15505.26 fixes stale native-job version labels with 244 connected tests, 26 selected ACP/version tests and an independent 107-test review. Actual reload shows v311; reviewed Markdown preserves both notes and all 863 flagged findings. Final complete-pack export/restore passes all 233 manifest hashes and 15 preservation checks, with remapped PDF navigation visually verified. Current-head CI remains pending; human labels remain null. Details: `Docs/Reviews/POLITICAL_PDF_SEMANTIC_ACCEPTANCE_2026_11_17.md`.

## Stage 4: Destination indexing and participant acceptance

**Goal:** Verify restored vector retrieval and hand the existing uncoached script to the actual VIP.
**Success Criteria:** Owned Media2 indexing is terminal and scoped retrieval works without research changes; participant observations are recorded only when supplied by a real person.
**Status:** In Progress

The ordered model repair and control checkpoints are complete. Job 57's first failure remains retained; explicit recovery Job 61 completes all chapters/portions and its structurally complete overview, while semantic review remains negative. Independent destination checks qualify the earlier restored pack and the reviewed partial Job 57 pack (13 outputs, 223 historical reports, 25 reviews and 14 messages). The final reviewed Job 61 pack also passes export/restore into a new workspace, preserving 13 outputs, 324 historical report occurrences, 30 reviews, 16 messages, four QA reports, 52 citations, the Note and original PDF. The actual participant session remains separate.

- [x] Start destination API, Redis/embedding worker and frontend using new run logs, existing private configuration and preserved databases. Verify the original PDF SHA and saved workspace identity first.
- [x] Submit `POST /api/v1/media/2/embeddings` with the existing MiniLM model and `force_regenerate:false`; retain the returned job ID and poll the existing job/status APIs.
- [x] Verify source-scoped vector/hybrid results and unchanged PDF, reports, review overlays and restored citations.
- [x] Export and restore the final reviewed Job 61 complete pack; independently compare all saved research and prior destination workspaces, then open an imported evidence link in the actual PDF viewer.
- [x] Prepare the existing seven-task VIP script and a blank results sheet containing outcome, time, assistance, errors, confidence and NN/g heuristic/severity: `Docs/Reviews/POLITICAL_PDF_VIP_WALKTHROUGH_2026_11_17.md`.
- [ ] Obtain and incorporate actual VIP observations; never simulate human acceptance. Human claim labels and calibration also remain pending.
- [ ] Record completed versus pending acceptance honestly, update Backlog/PR and remove only this plan after all stages are actually complete. Keep human-written Change summary/merge requirements intact.

**Technical verification:** Destination runtime `441bffcf46` completes one MiniLM job, `ee8f18a9-2f02-4d6c-9107-3cc47cc6ee09`, with 2,261 chunks/vectors and no error. The restored source is queryable with vector readiness true; Redis has no pending messages. Six vector/hybrid probes return 72 results, all scoped to Media 2 with exact canonical excerpts and physical-page bindings. Five probes cover all expected pages; EXIM vector retrieval misses physical 753/759 while hybrid retrieves both. This retained retrieval limitation is not an answer-quality pass.

All nine artifacts, 148 artifact reports, the latest analysis's nine review overlays, twelve messages, two QA reports, forty conversation citations, the Note and original PDF remain unchanged. Navigation nodes, text and page map remain exact, while index completion advances navigation identity from `media_2:v7:50cc51b121` to `media_2:v8:0d889a2613`. Queued/stale ingestion Job 14 remains untouched; it is separate from the completed embedding job.

The actual browser walkthrough opens saved assistant message 12 citation `[6]` through **Open PDF page 872**, reaching `document-workspace?open=2&page=872`. The viewer shows 872/922; visual inspection confirms printed page 839, the Financial Regulatory Agencies heading and CFPB proposal. Media 2 reads and the expected reading-progress write return 200. Only two existing AntD deprecation warnings appear in the captured console log. A subsequent API comparison confirms saved research, navigation and original PDF remain exact after browsing. Private runtime proof and hashed browser receipt: `/tmp/tldw-political-pdf-semantic-20261117/destination-runtime-441bffcf46/{destination-verification,browser-citation-receipt}.json`. This is agent technical verification; actual VIP observations remain blank.

Final pack SHA256: `692b79963eb715f21aac5b8b23220ee88e513a328a091e7da11b02fe726c808d`; restored workspace `ac294a01-64bf-4045-a673-e6a832f90a33`. All 15 preservation checks pass, including exact prior-workspace payloads. Browser evidence E27.1.36 opens physical page 872/922, visibly printed 839. Existing Media 2 vectors are reused; newly queued ingestion is untouched. This final restore does not imply a fresh upload, fresh indexing, semantic acceptance or human usability observations.
