---
id: TASK-12964
title: Complete research discovery Phase 2B HTML Media handoff design
status: To Do
assignee: []
created_date: 2026-07-14 03:11
labels:
- research
- media
- design
- jobs
- security
- web
dependencies: []
references:
- TASK-12954
- https://github.com/rmusser01/tldw_server/pull/2716
- https://www.sourclip.com/resources/research-sources
- TASK-13139
- TASK-13139.6
- TASK-13139.8
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
- Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md
updated_date: 2026-08-28 01:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete and formalize the paused Phase 2B design for handing source-specific HTML full-text candidates from Research Discovery through the existing /api/v1/media/add chokepoint. Reuse the existing Media context-extraction, persistence, and Jobs machinery. Do not add another ingestion endpoint, Research-owned ingestion behavior, a second web pipeline, a new durable queue, or an implementation task until the revised specification and implementation plan pass review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Discovery emits source-specific html_full_text_url candidates only from explicit provider full-text HTML fields or documented deterministic derivation from stable provider identifiers; landing pages and generic URL probes remain ineligible.
- [ ] #2 The only public handoff remains /api/v1/media/add with media_type=web restricted to discovery selections; Research does not ingest, endpoints do not call endpoints, and /media/ingest/jobs rejects discovery references, web media type, and web-only controls.
- [ ] #3 Each accepted candidate creates one existing media_ingest_item Jobs job using the existing domain, queues, WorkerSDK, owner-scoped Media DB, and shared Media web operation; no new queue, worker domain, or ingestion API is introduced.
- [ ] #4 Bounded HTTP-only extraction reuses the existing context extraction pipeline without persisting raw HTML, browser fallback, shared scraper sessions, persistent cookies, or process-global dedupe state.
- [ ] #5 Crawl, cookies, custom headers, and optional server-configured analysis follow the approved security, limit, dedupe, persistence-first, compare-and-set, partial-result, and credential-scrubbing contracts recorded in the task notes.
- [ ] #6 Idempotency, current configuration ceilings, retry classification, sanitized bounded results, and cooperative cancellation semantics are fully resolved in the specification and covered by a concrete verification strategy.
- [ ] #7 The revised specification passes the required review loop and user approval before a separate Phase 2B implementation plan or implementation task is created.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Handoff task only. Resume the superpowers brainstorming workflow, reconcile the decisions in Implementation Notes against current dev, finish the error-handling/cancellation section, revise the existing discovery specification, run the spec-review loop, obtain user approval, and only then write a uniquely named Phase 2B implementation plan and create implementation tasks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Handoff state as of 2026-07-13. No Phase 2B implementation plan or code work is approved. Resume from current dev with the superpowers brainstorming and spec-review workflow.

Approved boundaries
- Support current discovery providers only when they expose explicit full-text HTML. Normalize this as html_full_text_url. Deterministic construction is allowed only from stable provider identifiers and fixed documented provider host/path rules. Never probe generic URLs or promote landing pages.
- media_type=web is valid only with discovery references. PDF and HTML cannot be mixed. /api/v1/media/add remains the sole public handoff. Research stays resolver-only. Do not add another ingestion endpoint, call endpoints from endpoints, create another web pipeline, or add a new Jobs domain/queue/worker.
- /media/ingest/jobs must reject discovery references, media_type=web, and web-only controls despite sharing AddMediaForm. Omitted perform_analysis remains true for existing media types but resolves false for discovery web.

Jobs and options
- Validate the entire request before creating jobs. Create one existing domain=media_ingest, job_type=media_ingest_item job per accepted selected HTML candidate. Authority comes only from job.owner_user_id; the worker opens that owner Media DB and injects it into the shared Media web operation.
- Credentialed requests require exactly one selected candidate. Noncredentialed requests may select multiple candidates. All queued returns 202, mixed queued/rejected returns ordered 207 results, and request-wide validation remains 4xx.
- Expose only recursive web_crawl controls: max pages, depth, and include external. External defaults off. Crawl, cookies, custom headers, and optional server-configured per-page analysis are available. Requests may only lower server ceilings. Execution uses min(stored enqueue-time effective limit, current server limit), so later loosening never expands queued work and tightening wins.

Fetch and credentials
- Use deterministic bounded HTTP crawling and the existing context-extraction pipeline. Raw HTML is transient and never persisted or returned. Enforce Content-Length plus decoded/decompressed byte limits, bounded MIME handling, per-hop egress and redirect checks, DNS-rebinding defenses, unsafe URL sanitization, and HTTPS downgrade blocking. Phase 2B excludes Playwright/browser fallback and does not claim a hard cancellable parser CPU timeout.
- Discovery jobs use job-local sessions, cookies, server-set cookies, and dedupe state. Bypass persistent CookieManager, ContentDeduplicator, shared scraper sessions/caches, and the enhanced scraper in-process queue.
- Validate and cap custom headers; deny routing and hop-by-hop headers. Credentials go only to the exact original origin and are stripped on any scheme, host, or port change and all external pages. Encrypt the credential envelope before create_job and fail closed if encryption is unavailable. Plaintext remains only in worker memory.
- A central Jobs terminal sanitizer must scrub credentials atomically on every completion, failure, cancellation, quarantine, batch terminal operation, and before archival. If terminal decryption fails, replace the payload with a minimal scrubbed tombstone. Results, errors, progress, events, audit attributes, logs, and persisted URLs must be bounded, sanitized, and secret-free.

Idempotency, persistence, and analysis
- Optional Idempotency-Key applies per accepted candidate, not to the request atomically. Derive an owner-scoped opaque key and request fingerprint. Credential changes require a purpose-separated, versioned server-keyed HMAC commitment supporting primary/secondary rotation. Exact replay reuses the candidate job; conflicting fingerprints return 409. A new key is required after terminal credentialed failure because credentials are scrubbed. Without a header, submissions are independent and Media dedupe is the final guard.
- Every crawled page receives an ordered outcome, but only each unique canonical URL/content result creates a web_document. Duplicates return the existing media ID without touching or analyzing it. Refactor the existing Media repository core to return a structured internal result containing created, media_id, and document version while retaining the legacy tuple wrapper for existing callers.
- Persist extracted content and chunks first with opaque created_by_job_id provenance and analysis status disabled, not_run, completed, or failed. Analysis defaults off and may update only a document created by the same job at the exact expected version through compare-and-set. Existing and concurrently created duplicates are never mutated. Exhausted analysis failures preserve content and produce completed partial results.

Errors, results, and unresolved decision
- Page-specific fetch, policy, MIME, extraction, and exhausted analysis failures may be partial after the root is persisted. Systemic database, credential decryption, required-configuration, and worker failures fail or retry even after partial persistence. Retry only a configured transient set such as 408, 425, 429, 500, 502, 503, and 504 with bounded Retry-After. Persist a compact bounded checkpoint after each committed page and use one result size/encryption/sanitization path.
- UNRESOLVED: JobManager.cancel_job currently terminally cancels processing jobs while workers may continue persisting, and finalize_cancelled cannot store a partial result. Recommended design: add a Jobs-level cooperative cancellation primitive. Queued jobs cancel and scrub immediately; processing jobs set cancel_requested_at while retaining the lease; workers stop at bounded checkpoints and call finalize_cancelled(result=checkpoint); lease recovery terminalizes abandoned cancellation requests using the latest checkpoint. Resolve this before writing the spec. Do not patch only the Media endpoint or worker.

Verification expectations
Cover deterministic candidate derivation; route isolation/defaults; owner-scoped idempotency and credential commitments; every terminal scrub path; SSRF, rebinding, redirects, downgrade, cross-origin stripping, MIME and decompression limits; deterministic crawl budgets and job-local state; no-touch duplicates; transactional creation ownership and analysis compare-and-set; crash/cancellation checkpoints; retry/partial classification; secret-free logs/results; property tests for URL normalization and budget invariants; targeted pytest, compile/diff/lint checks, and Bandit. Revise the existing discovery spec only after the cancellation decision is approved, run the spec reviewer loop, obtain user approval, and then create a separate implementation plan and implementation tasks.

Tracker note: the older Phase 2A planning record uses duplicate ID TASK-12108, which also belongs to unrelated active tasks. It remains marked Done in backlog/tasks because ID-only completion is ambiguous. TASK-12954 is the authoritative completed Phase 2A implementation record.
2026-08-27 program reconciliation (TASK-13139): this task remains the sole owner of Phase 2B HTML-to-Media handoff, its existing Media Jobs crawl, credential envelope, and cooperative cancellation decision. The agent-native program will not create a duplicate crawl job or ingestion path. TASK-13139.6 is a post-Phase-2B checkpoint/stop-reason follow-up and must wait for this design, the Jobs cancellation contract, and the concrete implementation task produced here; TASK-13139.8 likewise adds that future implementation task as a dependency before execution.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
