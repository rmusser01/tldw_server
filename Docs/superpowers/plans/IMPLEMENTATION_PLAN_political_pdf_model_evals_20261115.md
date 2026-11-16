# Political PDF model evaluations implementation plan

**Goal:** Verify GPT-6 Luna and provide repeatable, honest model comparisons using the same PDF and production APIs.
**Architecture:** Local Python runner and offline report module; existing Media/RAG/Chat/Jobs owners. No new service or dependency.
**Spec:** `Docs/Design/POLITICAL_PDF_MODEL_EVALS_2026_11_15.md`.
**ADR assessment:** No new rule; governing ADRs are linked in the design. Tasks TASK-15505.10 and TASK-15505.11.

## Stage 1: Verify provider compatibility
**Goal:** Fix only demonstrated GPT-6 request incompatibilities in the native OpenAI adapter.
**Success Criteria:** Actual public app completion and one-section job succeed; prior model behavior retained.
**Tests:** Red-first sync/async/stream payload controls, extra-body precedence, reasoning effort, connected adapter suites, Bandit and independent review.
**Status:** Complete

## Stage 2: Implement repeatable execution and reporting
**Goal:** Deliver runner/config and independent offline report with safe attempt preservation.
**Success Criteria:** Offline planning, immutable manifest, append-only attempts, safe resume, explicit live execution and meaningful reports work without paid CI calls.
**Tests:** Fake HTTP interrupted/unknown outcomes, drift, no-repeat resume, secret redaction; report source scope, citations, partial coverage, missing usage and archive integrity.
**Status:** In Progress

## Stage 3: Run the same PDF with Luna
**Goal:** Exercise the runner and actual app with distinct Luna outputs.
**Success Criteria:** Frozen questions and selected 30-chapter scope retain raw evidence; cited answers, review/save/reload and export are verified. Record provider/model/reasoning differences from DeepSeek and preserve semantic failures.
**Tests:** Live runner artifact checks plus source review and browser walkthrough; no automatic paid retry.
**Status:** In Progress

## Stage 4: Review and publish
**Goal:** Add tested tooling, model compatibility and honest evidence to PR #3480.
**Success Criteria:** Independent review, local checks and secret scans pass; docs explain exact command/config usage and remaining human/CI limits.
**Tests:** CLI offline smoke, relevant pytest suites, Bandit, task/diff/publication checks and PR head/check inspection.
**Status:** Not Started

Remove only this plan after all implementation stages complete; retain design and reports.
