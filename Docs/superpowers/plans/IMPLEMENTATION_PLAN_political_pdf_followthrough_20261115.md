# Political PDF follow-through implementation plan

> **For agentic workers:** Use subagent-driven-development for independent implementation units, with separate code review and integrated verification. Keep Backlog records current using backlog-py.

**Goal:** Close the remaining large political PDF research workflow gaps and deliver verified, reviewable changes.

**Architecture:** Retain the existing Jobs, Research Workspace, Media, Notes and RAG owners. Repair the existing interfaces and evidence flow before adding new mechanisms. Preserve original model outputs and exact citations when recording corrections.

**Tech stack:** FastAPI, SQLite, existing embeddings/Chroma, Next.js shared UI, pytest, Vitest and Playwright.

**Spec:** `Docs/Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md`; follow-up scope is TASK-15505.5 through TASK-15505.9 and the recorded failures in `Docs/Reviews/POLITICAL_PDF_VIP_RETEST_2026_11_15.md`.

## Global constraints

- The user authorized remaining fixes and the real PDF/DeepSeek retest. No credential values enter tracked files, reports, exports or logs.
- Use the same 922-page PDF and preserve failed outputs. Separate retrieval coverage, exact provenance, semantic accuracy and human acceptance.
- Use existing dependencies and canonical persistence; keep source ownership and current authorization checks intact.
- Do not report agent labels as human labels or an automated walkthrough as a participant study. Do not report an uncalibrated judge score.
- ADR check: no new ADR for these repairs/validation. ADR-003 governs Jobs, ADR-007 Workspace, ADR-065 Notes, and ADR-017/052 authenticated authority. Reassess if portability needs a new durable ownership rule.

## Stage 1: Evidence and bounded designs
**Goal:** Reproduce and assign the remaining failures without repeating completed work.
**Success criteria:** Existing tasks are active; each implementation unit identifies affected callers, persistence, exports and tests; frozen failures and held-out source cases are distinct.
**Tests:** Read saved final overview/comparison and FBI retrieval traces against canonical source text; inspect UI and existing review/backup APIs.
**Status:** Complete

- [x] Start TASK-15505.5/.6 and create .7 retrieval and .8 integrated qualification tasks.
- [x] Assign independent discovery for mobile/Notes, synthesis/review and retrieval/embeddings.
- [x] Record each unit's bounded design and ADR assessment before code edits.
- [x] Choose the existing backup path after inspecting its source/Notes coverage; create a separate task before portability edits.

## Stage 2: Repair existing flows
**Goal:** Fix retrieval completeness, review/correction access and mobile reading/save feedback.
**Success criteria:** Each root cause has a failing behavioral regression followed by a minimal passing implementation; affected callers and export/reopen paths agree.
**Tests:** pytest RAG/Research Workspace, Vitest Research Workspace/Notes, scoped lint/typecheck and Bandit; independent reviewer checks authorization, concurrency and draft preservation.
**Status:** In Progress

- [ ] Complete TASK-15505.7 text coverage and configure a real cached embedding model.
- [ ] Complete TASK-15505.6 synthesis grounding and canonical review/corrections.
- [ ] Complete TASK-15505.5 mobile, Notes feedback and measured readability checks.
- [ ] Independently review each unit and commit verified increments with its Backlog changes.

## Stage 3: Integrated qualification
**Goal:** Verify the actual user journey with real data and provider calls.
**Success criteria:** Ingestion, selected chapter summaries, cited answers, correction/reload and export have explicit outcomes. Vector/hybrid indexing and retrieval are real. Accessibility/performance evidence is bounded and reproducible.
**Tests:** Same PDF/DeepSeek browser walkthrough; regression and held-out source audits; keyboard/phone/light-dark checks; broader affected-domain suites; bounded serial/concurrent latency measurements and restore checks.
**Status:** In Progress

- [x] Restore isolated local services without exposing credentials or touching unrelated user data.
- [ ] Verify text/vector/hybrid retrieval separately, then generation against retrieved evidence.
- [x] Run one fixed retest per generation case; preserve failures and diagnose them before any further run.
- [ ] Verify review metadata, Notes, original source and exports after reload/restore.
- [ ] Record NN/g heuristic, severity, evidence, implementation and observed result for each remaining/new issue.

## Stage 4: Delivery and honest acceptance
**Goal:** Deliver the finished local workflow and the requested review artifact.
**Success criteria:** Commits, checks and walkthrough evidence are reviewable; participant-review or deployment requirements are explicit and not falsely marked complete.
**Tests:** Clean diff/secret scan, task-format and task-ID checks, integrated typecheck/Bandit, final service and job inventory.
**Status:** Not Started

- [x] Prepare a concise source-backed human review packet and nontechnical task script for checks that require a person.
- [ ] Update task dispositions and the current acceptance summary from actual results.
- [ ] Deliver to the user's selected destination. A PR merge requires the human-authored change summary defined by repository policy.
- [ ] Remove only this plan after all stages are complete; preserve unfinished external requirements in Backlog.
