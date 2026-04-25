# Risk Reduction And Velocity Wave Program Design

- Date: 2026-04-15
- Project: tldw_server
- Topic: Organize codebase cleanup and hardening into subsystem waves optimized for risk reduction and developer velocity
- Mode: Design for implementation planning

## 1. Objective

Define a cleanup program for a large, active codebase that is broad enough to cover the maintainer's need for improvement work, but structured tightly enough that implementation can proceed in bounded, reviewable subsystem waves.

The program must:

- prioritize risk reduction before cosmetic cleanup
- choose work that also improves developer velocity in the touched subsystem
- avoid turning cleanup into an open-ended rewrite
- produce one implementation plan at a time, starting with the highest-risk wave

## 2. Program Scope

### In Scope

- program-level sequencing for hardening and cleanup work across the repository
- subsystem-wave definitions, ordering, and entry criteria
- intake and prioritization rules for deciding what belongs in a wave
- execution rules and definition of done for each wave
- the planning boundary for how this program transitions into implementation

### Out of Scope

- a single implementation plan covering all six waves at once
- net-new product features
- repo-wide style-only cleanup
- large refactors that are not tied to risk reduction or clear velocity payoff
- detailed per-issue implementation steps for Waves 2 through 6 in this spec

## 3. Inputs And Evidence Base

This design is grounded in the current repository structure, recent activity, hotspot modules, and existing subsystem reviews already present in the repo.

Primary evidence used:

- top-level repository docs and architecture guides
- current repo structure and test surface across backend, WebUI, and Admin UI
- recent commit history showing active review-fix and regression work
- existing review artifacts:
  - `Docs/superpowers/reviews/db-management/`
  - `Docs/superpowers/reviews/characters-backend/`
  - `Docs/superpowers/reviews/characters-fullstack-delta/`
  - `Docs/superpowers/reviews/evals-module/`
  - `Docs/superpowers/reviews/moderation-backend/`
  - `Docs/superpowers/reviews/web-scraping-ingest/`
- representative hotspot modules and clients:
  - `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - `tldw_Server_API/app/main.py`
  - `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
  - `apps/packages/ui/src/services/tldw/TldwApiClient.ts`

## 4. Priorities And Constraints

### Priority Order

1. highest risk reduction
2. strongest developer-velocity payoff
3. visible improvement only after the first two are satisfied

### Delivery Shape

- Work proceeds in subsystem waves of about one week each.
- Each wave should leave a subsystem safer, better tested, and easier to modify.
- Implementation planning happens wave-by-wave, not program-wide.
- Each implementation plan should define a bounded set of findings that can still be owned, reviewed, and verified coherently within the wave.
- There is no fixed numeric cap on findings. Wave size is constrained by dependency coupling, verification burden, and reviewability rather than staffing budget alone.
- If a subsystem backlog is too coupled or too large to verify coherently as one wave, planning must split it into an explicit follow-on wave or follow-on backlog before implementation begins.

### Non-Goals

- No attempt to "clean up everything" in one pass.
- No architecture purity work that does not materially improve safety or delivery speed.
- No giant monorepo-wide decomposition before the known risk paths are hardened.

## 5. Approaches Considered

### Recommended: Infrastructure-First Waves

Start with shared data, bootstrap, lifecycle, and contract layers, then move outward into feature subsystems and finally broader decomposition.

Pros:

- addresses the highest-severity failure modes first
- reduces downstream churn for later subsystem waves
- aligns with the existing review artifacts already pointing at DB, bootstrap, and lifecycle defects

Cons:

- early wins are less flashy than feature-only cleanup
- some large maintainability hotspots remain until later waves

### Alternative: Feature-Stability-First Waves

Start with the most visible feature subsystems and defer core infrastructure cleanup unless it blocks the feature work.

Pros:

- easier to show user-facing progress early

Cons:

- underlying infrastructure and lifecycle problems keep reappearing in later work
- weaker risk reduction than the recommended approach

### Rejected: Architecture-First Rewrite

Start by splitting oversized files and normalizing boundaries across the repo before targeting concrete risk defects.

Reason rejected:

- too weak on immediate risk reduction
- likely to create broad churn before the known fail-open and lifecycle issues are corrected

## 6. Locked Program Decisions

- This program optimizes for risk reduction first and velocity second.
- Work is batched as subsystem waves, not tiny isolated fixes and not one mega-plan.
- Each wave includes only the work needed to improve safety, determinism, and local maintainability in that subsystem.
- Each wave must include regression coverage for the corrected contracts.
- Implementation planning begins with Wave 1 only. Later waves get their own planning pass only after the active wave has a green targeted verification set and no unresolved high-severity regressions still attributed to that subsystem.
- A wave cannot be used as a catch-all bucket. Shared follow-up waves must name a bounded contract surface before planning starts.

## 7. Wave Template

Every wave contains exactly four lanes:

1. `Confirmed-risk fixes`
   - correctness, security, auth, tenancy, lifecycle, migration, availability, or data-integrity issues
2. `Regression coverage`
   - tests that prove the corrected behavior and would have caught the original defect
3. `Boundary cleanup`
   - small, local refactors that reduce contract drift or hidden coupling in the touched subsystem
4. `Operational clarity`
   - clearer failure signaling, logs, docs, or subsystem contracts needed so maintainers can trust the new behavior

Every wave explicitly excludes:

- net-new features
- repo-wide cleanup outside the subsystem boundary
- broad style rewrites
- decomposition work that does not directly support the subsystem's risk-reduction goal

## 8. Intake And Prioritization Rules

An item belongs in the active wave only if it meets at least one of these tests:

- `Risk`: it can cause bad auth, tenancy leaks, data corruption, silent degradation, misleading success, or availability failures
- `Velocity`: it repeatedly slows delivery because a boundary is unclear, a contract is inconsistent, or tests are flaky or non-deterministic
- `Coupling`: it sits on a shared dependency or lifecycle seam that causes failures in multiple higher-level features

Within a wave, work is ranked in this order:

1. fail-open, security, tenancy, migration, bootstrap, and state-corruption issues
2. order-dependent failures, flaky initialization or shutdown behavior, and misleading success responses
3. contract drift between layers
4. local refactors that reduce churn in the touched subsystem

File size alone is not enough to justify inclusion. Large modules should be touched only when their size is clearly contributing to risk or velocity problems inside the active wave.

## 9. Approved Wave Sequence

### Wave 1: Data And Bootstrap Hardening

Primary targets:

- `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- `tldw_Server_API/app/core/DB_Management/db_migration.py`
- `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
- `tldw_Server_API/app/core/DB_Management/content_backend.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- directly affected callers and tests

Required outcome:

- bootstrap and migration paths fail closed
- tenant-isolation setup cannot partially succeed silently
- trusted path enforcement is real rather than lexical-only
- backend reset and replacement paths do not leak hidden state

Planning note:

- This subsystem is too large to treat as one unlimited week-long backlog.
- Wave 1 planning may include as many confirmed DB/bootstrap findings as can still be executed, reviewed, and verified coherently as one wave.
- If the Wave 1 backlog becomes too coupled, too broad, or too verification-heavy, planning must split the overflow into explicit follow-on DB/bootstrap work rather than hiding it as stretch scope.

### Wave 2: ChaChaNotes And Character/Chat Lifecycle

Primary targets:

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- related character import/export, quota, and versioning paths

Required outcome:

- mixed-suite `503` behavior is reproducible and fixed or tightly isolated
- lifecycle and versioning contracts are consistent
- quota, restore, and similar endpoint outcomes are no longer silently misleading

### Wave 3: Evaluations Reliability

Primary targets:

- evaluation auth surface
- `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- runner, persistence, cancellation, and webhook/batch contract seams

Required outcome:

- run status cannot be corrupted by cancellation or auth identity inconsistencies
- rate-limit and subject semantics are stable
- batch and webhook paths match the service contract instead of drifting

### Wave 4: Web Scraping And Outbound Safety

Primary targets:

- process-web-scraping endpoints
- enhanced scrape branches
- outbound egress and request-safety seams

Required outcome:

- endpoint tests are deterministic
- reachable scrape paths share one documented security model
- request-safety behavior is proven across the branches that matter

### Wave 5: Shared API And Client Boundary Cleanup

Primary targets:

- backend endpoint contract inconsistencies that affect multiple frontend consumers or repeated client-call patterns across subsystems
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- directly affected frontend consumers and contract tests
- explicitly exclude single-feature endpoint quirks that still belong to their owning subsystem waves

Required outcome:

- frontend consumers stop depending on ad hoc endpoint knowledge
- the shared client becomes more predictable and easier to evolve
- contract drift is caught by tests rather than discovered during UI work

### Wave 6: Monolith Decomposition For Maintainability

Primary targets:

- `tldw_Server_API/app/main.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- other oversized endpoint or service modules directly affected by earlier waves

Required outcome:

- the largest files have clearer internal boundaries
- edit surfaces are smaller and side effects more local
- maintainability improves without reopening previously stabilized behavior

### Deferred But Tracked Areas

This six-wave sequence is the first cleanup tranche, not the full long-term backlog for the repository.

Reviewed or remediated areas not explicitly sequenced above are deferred unless they become blockers or escalate in severity during an active wave. Current examples include:

- moderation follow-up
- monitoring follow-up
- metrics follow-up
- embeddings follow-up
- workflows follow-up
- writing follow-up
- audit and AuthNZ follow-up outside the Wave 1 bootstrap path
- deeper MCP or RAG follow-up beyond the specific boundaries named in the six waves

Deferral does not mean those areas are unimportant. It means they do not outrank the six-wave sequence for the current risk-reduction program unless new evidence changes their priority.

## 10. Execution Model

Each wave runs as a bounded review-and-hardening cycle:

1. `Freeze scope`
   - convert existing findings, live defects, and hotspot paths into a bounded wave backlog
2. `Reproduce and rank`
   - confirm which issues are live, already fixed, or only probable risks
3. `Fix highest-risk paths first`
   - start with fail-open behavior, corrupted state transitions, bootstrap failures, and flaky lifecycle behavior
4. `Add regression coverage immediately`
   - every material fix gets the narrowest regression that proves it stays fixed
5. `Do local boundary cleanup`
   - extract helpers, narrow interfaces, and remove silent fallbacks inside the same subsystem
6. `Close with subsystem verification`
   - run targeted tests, relevant mixed-suite tests, and Bandit on touched backend scope before calling the wave complete

## 11. Definition Of Done For A Wave

A wave is complete only when:

- the top confirmed risks in that subsystem are fixed or explicitly downgraded with evidence
- the touched subsystem has stronger regression coverage than before
- mixed-suite or lifecycle failures in scope are fixed, or shown with evidence to originate outside the active subsystem and handed off to the owning subsystem backlog with an explicit next action
- any behavior or contract changes are documented
- the subsystem is easier to modify than before, not merely different

## 12. Planning Boundary

This spec intentionally stops at program design.

Implementation planning should proceed as follows:

- create one detailed implementation plan for `Wave 1: Data And Bootstrap Hardening`
- use the wave template and intake rules in this spec as the planning guardrails
- do not write one plan covering all six waves
- after Wave 1 reaches a stable endpoint, repeat the same design-to-plan process for Wave 2
- for planning purposes, `stable endpoint` means:
  - targeted verification for the wave is green
  - any remaining high-severity defects are proven out of scope for that subsystem or explicitly handed off
  - the wave backlog is no longer carrying hidden stretch items

## 13. Success Criteria

This program is successful if it creates a repeatable way to improve the codebase without losing focus.

Success looks like:

- the first wave targets the highest-risk infrastructure paths already evidenced by current review artifacts
- later waves follow a predictable order instead of becoming an unbounded backlog
- each completed wave leaves behind fewer hidden failure modes and lower local development friction
- the codebase gets safer and easier to change in the same motion rather than treating those as separate efforts
