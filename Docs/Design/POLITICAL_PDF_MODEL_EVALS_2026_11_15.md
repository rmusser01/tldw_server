# Repeatable political PDF model evaluations

Tasks: TASK-15505.10 (Luna compatibility and live run), TASK-15505.11 (repeatable runner). The user selected a command/config runner for local runs and CI. Preserve the prior DeepSeek outputs and pending human labels.

## Approach

Use a small sibling of `Helper_Scripts/Evals/ingest_rag_eval_workflow.py`, calling the current application APIs. Reusing that script unchanged would create new resources on every run and lose interrupted-call evidence. Extending the Evaluations UI adds an unnecessary surface for the requested command workflow. A new evaluation service/database is unnecessary. The chosen runner owns local orchestration and evidence files; Media, RAG, Chat and Jobs retain production ownership.

Two independent modules: `run_political_pdf_eval.py` orchestrates and records; `political_pdf_eval_report.py` analyzes saved records without network access. Configuration chooses exact providers/models, source PDF and expected hash, existing Media/workspace IDs, frozen cases, retrieval modes, QA contexts and selected outline nodes. The sample has no embedded credential. API authentication is an environment-variable name; provider credentials stay in server configuration. Missing source/workspace IDs must not trigger guessed adoption or overwrite existing research.

## Workflow and evidence

1. `plan` validates configuration, source/fixture hashes and selected scope without API calls; lists the paid operations. `run` requires an explicit execution flag. `resume` continues the same immutable run identity. `report` is offline.
2. Verify original PDF download, physical page count, frozen per-page fingerprints and navigation identity against the owned ingested source. Source ingestion is independent of model choice; explicitly label reused ingestion. Do not duplicate uploads for each model. The runner requires an existing Media item; the separate browser walkthrough verifies upload.
3. Freeze generation-disabled retrieval responses for every selected case/mode, source IDs and expected physical-page coverage. Use the set of expected pages, never returned-hit grades, as the denominator. Record overbroad returned ranges separately.
4. Per model, run the frozen reference-page questions and questions grounded in saved retrieval contexts through the public chat API. Reuse the production section extraction prompt and evidence validation contract. Record exact prompts, first responses, finish reason, returned model identity, reported token usage and latency. No hidden retries or expected-answer text in model prompts.
5. Optional selected-section summaries use existing Workspace output Jobs and a new artifact for each model. Poll known job IDs; retain full terminal artifacts and partial/failed coverage. Resume existing paid work only through its recorded job/artifact, not by recreating it. Preserve review status as draft; never manufacture human acceptance.
6. Validate saved artifact readback and a supplied complete research pack's original PDF/manifest hashes. Backend-only results explicitly do not certify browser export, offline-reader behavior or cross-server UI restore. The live Luna walkthrough checks those separately.

Versioned run directory: `manifest.json` contains resolved nonsecret config, fixture/source/revision identity; `attempts.jsonl` is append-only. Immutable per-attempt request/response/artifact files preserve failures. A started attempt is recorded and flushed before dispatch. HTTP timeouts, interrupted calls and unknown POST outcomes block automatic resubmission. Explicit retry creates a new attempt; known job IDs are polled without another submission. Refuse changed config/source/fixture identity during resume. Reports never store authentication headers, provider secrets or raw environment/config objects; redact known credentials in error bodies.

The common evidence record has `schema_version: 1`, `operation_id`, `attempt`, `event` (`started` or `finished`), `stage`, optional `case_id`, `model_key`, `context`, `mode`, UTC time, and paths relative to the run directory. Finished records add `outcome`, `elapsed_seconds`, `http_status`, optional `job_id`/`artifact_id`, `response_path`, and structured mechanical `checks`. Unknown outcomes remain visible. Report code must reject path escape and malformed records, retain first-attempt failures and avoid claiming unavailable measurements.

## Report and limits

Produce JSON and Markdown model comparisons plus a semantic-review packet. Mechanical checks include source integrity, scoped expected-page coverage, citation/span identity, completion/truncation, section coverage, artifact persistence and archive hashes. Summary provider-call usage is unavailable through the current public Jobs API and must be reported as unavailable; end-to-end job latency is measured. Public chat usage remains measured when returned.

Faithfulness, author attribution, disagreement and completeness are pending semantic review, not automatic passes. Human labels remain unfilled; H05 is now a tuned regression. No uncalibrated judge or token-overlap score is presented as accuracy. CI defaults to fake HTTP/recorded fixtures; paid live runs are explicit and use environment secrets. No dependency, recurring schedule or new UI is required.

## Verification and ADR assessment

Offline tests cover unknown paid outcomes, no-extra-POST resume, explicit retry retention, identity drift, malformed/truncated replies, missing usage, source scope/page ranges, partial summaries, archive tampering, path escape and credential absence. Run the implemented runner against the authorized Luna model, then inspect representative source claims and browser persistence/export. Preserve failures and distinguish direct-context from retrieval-grounded results.

ADR required: no. Existing [ADR-003](../ADR/003-jobs-vs-scheduler-default.md), [ADR-007](../ADR/007-research-workspace-canonical-first-slice-shell.md), [ADR-015](../ADR/015-evaluations-existing-evaluator-integration.md), [ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md), [ADR-049](../ADR/049-chat-history-selection-ownership.md) and [ADR-065](../ADR/065-independent-notes-knowledge-provenance.md) govern production ownership and review provenance. This is a local evaluation client, not a new persistence or authority boundary.
