# Political PDF portability implementation plan

> **For agentic workers:** Use subagent-driven-development, test first, and review the backend and frontend units independently before integration.

**Goal:** Export a complete offline research pack and restore an editable workspace on another server with verified source identities.

**Architecture:** Embed the existing allowlisted Chatbook in an optional complete Workspace ZIP; use authenticated restore mappings to remap typed references before Workspace reconciliation.

**Tech stack:** Existing Chatbooks/FastAPI/Media/Notes APIs, TypeScript, JSZip, pytest, Vitest and Playwright.

**Spec:** `Docs/Design/POLITICAL_PDF_PORTABILITY_2026_11_15.md`.

**Global constraints:** No new dependencies or persistence owner. Preserve original drafts and exact citations. Never use coincident numeric IDs as identity evidence. Never export credentials. Explicitly disclose partial restore; do not claim atomic rollback. ADR check: existing ADR-007/065 governs, no new durable ownership rule.

## Stage 1: Restore identity and provenance
**Goal:** Extend existing Chatbook results and Media restore without replacing its engine.
**Success criteria:** Verified media/Note mappings reach sync and async callers; exact validated page provenance survives restore; ambiguous/failed records do not produce usable mappings.
**Tests:** Add failing Chatbook tests for numeric collisions, Note mapping, exact page-map roundtrip and missing/ambiguous import identities; run existing account import/export suites and scoped Bandit.
**Status:** Complete

- [x] Read `_import_media_items`, `_import_notes`, `_build_import_result`, async result storage and response schema callers.
- [x] Reproduce dropped page metadata and missing mappings in tests; implement optional result fields and canonical provenance reuse.
- [x] Review ownership/error cases and update API response schemas/caller contracts. Independent review clear; final 12 guard tests passed, connected 42 passed, prior broad 118 passed. Scoped Bandit has zero findings; lint has no new findings.

## Stage 2: Complete pack and safe UI import
**Goal:** Provide the requested offline and editable outcomes from existing Workspace actions.
**Success criteria:** Complete ZIP has source bytes, readable reports, Notes and metadata; importing remaps every typed link, resets old mutation authority and preserves the existing workspace on failure.
**Tests:** Vitest malformed paths/size limits, incomplete content, unknown mappings, collision remap, Note markers, artifact evidence, old bundles, stale account and failed save; mounted Header tests verify progress/recovery and no incomplete fallback.
**Status:** Complete

- [x] Build codec/remapping helpers with failing behavioral tests before implementation.
- [x] Add scoped orchestration using existing Chatbook/Notes clients and Workspace reconciliation.
- [x] Add complete-export action and safe import dispatch while retaining legacy snapshot support.
- [x] Run connected export/import/storage suites, lint and TypeScript.

## Stage 3: Separate-server acceptance and delivery
**Goal:** Prove the actual PDF can travel with its research and remain usable.
**Success criteria:** A different server with a colliding Media ID restores the original PDF/checksum, saved Notes and reviewed artifacts; cited pages open the correct document. Offline files open without the original server.
**Tests:** Actual browser export/import, source download hash, Notes/artifact field comparison, citation click, fresh follow-up QA and offline archive inspection.
**Status:** Complete

- [x] Create an isolated destination with an unrelated source occupying the old numeric ID.
- [x] Export/import the same real PDF workspace through the UI and verify all requested contents after reload. Final restore424834aa-0803-4cbe-9938-6162825a14de preserves original PDF SHA, three Notes, four artifacts, 30 selected numbered chapter summaries and reviewed correction. Retry reuses its atomic chat; continuation and fresh authenticated browser restore all eight messages and 20 citations. Notes/review edit and reload pass; Education opens physical352/printed319 on restored Media2.
- [x] Independently review the full diff, record NN/g outcomes and limitations, run task format/ID checks, and commit the verified unit. Backend303 connected +42 independent SQLite/PostgreSQL, frontend167 independent +final109 route checks; TypeScript and scoped lint/Bandit pass. Full repository CI, deployment and human acceptance are separate. Latest generated-answer page precision is tracked under TASK-15505.7, not hidden by migration acceptance.
- [ ] Enroll imported chat atomically in the destination workspace using stable receipt IDs, exact parent remapping and existing citation validation; verify scoped readback, retry identity, owner/scope refusal, rollback and legacy global replay compatibility before live continuation.
