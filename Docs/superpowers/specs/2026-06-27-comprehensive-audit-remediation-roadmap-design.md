# Comprehensive Audit Remediation Roadmap Design

## Purpose

This design turns the 2026-06-27 comprehensive repository audit into an executable remediation roadmap. It addresses all accepted findings from `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json` while preserving reviewable work boundaries and maximizing safe parallelism.

The roadmap is an umbrella planning artifact. It does not implement fixes and does not create child remediation Backlog tasks. Child tasks should be created only after user approval of this spec.

## Source Artifacts

- Final report: `Docs/superpowers/reviews/2026-06-27-repo-audit/final-report.md`
- Findings index: `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- Remediation backlog draft: `Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md`
- Repeatable audit process: `Docs/superpowers/reviews/2026-06-27-repo-audit/repeatable-audit-process.md`
- Design tracking task: `TASK-12053`

## Audit Scope Summary

The audit accepted 31 findings:

- 0 critical
- 4 high
- 22 medium
- 5 low

Evidence tiers:

- 17 confirmed issues
- 10 likely risks
- 4 improvement opportunities

Validation status:

- 23 validated
- 8 need reproduction before closure

The four high findings are:

- `AUDIT-2026-06-27-AUTH-002`
- `AUDIT-2026-06-27-DB-001`
- `AUDIT-2026-06-27-MEDIA-001`
- `AUDIT-2026-06-27-MEDIA-002`

## Goals

- Group all 31 findings into reviewable remediation tracks.
- Maximize parallel implementation without duplicating architectural decisions.
- Preserve one clear owner and file scope per remediation task.
- Require reproduce-or-refute checkpoints for `needs_reproduction` findings.
- Define universal and track-specific verification gates.
- Define wave-level integration gates so individually passing branches do not diverge after parallel work.
- Provide a proposed Backlog task map with acceptance criteria and verification expectations, without creating child tasks yet.

## Non-Goals

- This design does not implement any remediation code.
- This design does not create the child remediation Backlog tasks.
- This design does not decide final implementation details that require code inspection inside each track.
- This design does not merge or push the audit branch.

## Program Shape

The remediation program should use 11 tracks. Tracks are grouped around shared boundaries rather than one task per finding.

| Track | Priority | Findings | Theme |
| --- | --- | --- | --- |
| 1 | High | `AUTH-001`, `AUTH-002`, `AUTH-003` | AuthNZ impersonation boundary |
| 2 | High | `MEDIA-001`, `MEDIA-002`, `MEDIA-003` | Media authorization and tenant storage |
| 3 | High | `DB-001`, `DB-002` | SQLite migration durability |
| 4 | Medium | `WEBUI-001`, `WEBUI-002`, `APIWEB-001` | WebUI/API contract alignment |
| 5 | Medium | `CHAT-001`, `CHAT-002` | Chat/RAG authorization and logging |
| 6 | Medium | `JOBS-001`, `JOBS-002`, `REL-001` | Durable workflow execution |
| 7A | Medium | `OPS-002`, `DEPS-001`, `DEPS-002` | Supply-chain foundations and worker image hardening |
| 7B | Medium | `OPS-001`, `OPS-003`, `OPS-004`, `OPS-006` | Release verification gates |
| 8 | Medium | `INTEGRATIONS-001`, `INTEGRATIONS-002`, `INTEGRATIONS-003` | Outbound HTTP policy |
| 9 | Medium | `MCP-001`, `MCP-002` | MCP WebSocket authorization and lifecycle |
| 10 | Low | `OPS-005`, `DEPS-003`, `MEDIA-004` | Maintenance and test hygiene |

Track 7 from the audit remediation draft is intentionally split into 7A and 7B. Supply-chain tooling and image hardening should land before release gates rely on those tools.

## Decision Gates

### Gate 1: Auth And Audit Identity Contract

Track 1 owns this gate. The contract should define how privileged impersonation represents actor, subject, scopes, expiry, and audit context. The expected direction is:

- Impersonation tokens are explicitly short-lived.
- Actor and subject are distinguishable in token claims.
- AuthContext or equivalent request identity preserves both actor and subject.
- Durable audit events record impersonation issuance and downstream impersonated actions.
- SQLite and PostgreSQL paths use backend-neutral lookup helpers.

### Gate 2: WebSocket Auth Contract

Tracks 4 and 9 share this gate. The contract should define one WebSocket auth pattern before implementation starts in either track. The expected direction is:

- Query-token WebSocket auth remains disabled or restricted by default.
- Browser and protocol clients send an explicit first-frame auth message.
- Backend WebSocket tests cover default query-token rejection.
- Scoped JWT restrictions apply to protocol streams that accept AuthNZ tokens.
- Frontend/audio and MCP/sandbox implementations reuse the same documented semantics where possible.

Track 4 applies the contract to TTS, STT, voice chat, and OSS billing capability behavior. Track 9 applies it to ACP and sandbox WebSocket streams.

### Gate 3: Durable Workflow Ownership Contract

Track 6 owns this gate. The contract should decide whether accepted workflow execution belongs to Jobs or Scheduler. Based on current repository guidance, user-visible work with recovery, admin controls, retries, and quotas should default to Jobs unless implementation inspection proves the core Scheduler is already the correct owner.

The contract must define:

- Which component owns accepted workflow runs.
- Where idempotency keys are generated and enforced.
- How startup repair handles accepted but unowned work.
- How duplicate schedule fires are collapsed.
- How shutdown, cancellation, and process loss are tested.

## Execution Waves

### Wave 0: Setup And Re-Confirmation

- Create an umbrella remediation Backlog task and approved child tasks.
- Start from a clean worktree refreshed from latest `origin/dev`.
- Re-check whether any audit findings were already fixed by later work.
- Confirm the exact child task list and file ownership.
- Record all confirmed pre-existing fixes as closed-with-evidence rather than reimplementing them.

### Wave 1: High-Risk Foundations

Run in parallel when file ownership is clear:

- Track 1: AuthNZ impersonation boundary
- Track 2: Media authorization and tenant storage
- Track 3: SQLite migration durability

These contain all high findings and have distinct primary ownership.

### Wave 2: Cross-Cutting Medium Tracks

Run in parallel after relevant gates:

- Track 4 after Gate 2
- Track 5
- Track 6 after Gate 3
- Track 8
- Track 9 after Gate 2

Tracks 4 and 9 must not independently invent WebSocket auth semantics.

### Wave 3: CI, Release, And Supply Chain

- Track 7A can start as soon as the roadmap is approved.
- Track 7B can inspect release gates in parallel, but final implementation should wait for 7A decisions about pinned setup, lock/constraints strategy, and required tooling.

### Wave 4: Opportunistic Cleanup

- Track 10 can run whenever it does not overlap active production-code tracks.
- It should not block high or medium remediation work.

## Wave Integration Gates

Parallel branches must be integrated after each wave before dependent work starts.

Each wave integration gate should:

- Rebase or merge completed child branches into a coordination branch.
- Resolve shared contract drift before dependent tracks continue.
- Run cross-track focused tests for affected modules.
- Run `git diff --check`.
- Run Bandit over touched backend production paths when Python code changed.
- Record environment-dependent skips explicitly.
- Update the umbrella Backlog task with findings closed, findings refuted, residual risk, and follow-up work.

No branch should claim a finding is closed until its fix is integrated or the integration gate records why isolated verification is sufficient.

## Closure Rules

A finding is closed only when all of the following are true:

- The original audit claim is addressed directly.
- The implementation change has landed in the task branch.
- Regression coverage or an executable check covers the original failure mode.
- Required focused verification passes.
- Bandit has run over touched Python production paths when Python code changed, or the task records why Bandit is not applicable.
- Track-specific verification has run. If a decisive track-specific check cannot run locally, the task may be marked locally complete with a recorded skip, but the finding remains pending external verification and must not be marked closed.
- Residual risk is documented.

For `needs_reproduction` findings, closure requires either:

- reproduced, fixed, and regression-tested; or
- refuted with concrete evidence and user-visible task notes.

## Universal Verification

Every code-changing task should run:

- Focused unit, integration, frontend, or E2E tests for the changed behavior.
- `git diff --check`.
- Bandit over touched backend production paths when Python production code changes.
- A Backlog update listing findings covered, touched files, tests/scans run, known skips, residual risk, and follow-up tasks.

Every documentation-only task should run:

- Placeholder scan for common filler markers and stale scaffold text.
- `git diff --check`.
- Link/path validation where practical.
- Backlog final summary or implementation notes update.

## Track-Specific Verification

Track-specific checks are required only for relevant tracks. If the local environment cannot run a non-decisive check, the task must record the skip and residual risk. Decisive checks, such as Docker image inspection for image-hardening findings, SBOM generation for SBOM coverage findings, live WebSocket checks for deployed client/server contract findings, and networked dependency/CVE audits for dependency findings, must complete before the affected finding is marked closed.

- AuthNZ: SQLite and PostgreSQL impersonation tests, token claim decode tests, audit attribution tests.
- Media: permission-denial HTTP tests, multi-user MediaWiki DB/vector isolation tests, storage cleanup failure tests.
- DB migrations: file-backed legacy Media DB upgrade tests and atomic failed-migration tests.
- WebUI/API contracts: client first-frame auth tests, backend WebSocket auth tests with query-token auth disabled, OSS billing capability guard tests.
- Chat/RAG: scoped virtual-key tests across alternate generation/search routes and log redaction tests.
- Workflows: process-loss, duplicate-fire, idempotency, shutdown, and startup repair tests.
- Supply chain: lock/constraints validation, pinned tool/action validation, worker image non-root and runtime-layer inspection.
- Release gates: PR image build matrix coverage, all-workflow/composite actionlint coverage, Bun-aware SBOM generation, Kubernetes sample validation.
- Outbound HTTP: private/loopback denial tests, central client/proxy behavior tests, tokenizer URL and provider base URL tests.
- MCP: scoped JWT WebSocket rejection tests, reconnect-disconnect lifecycle cleanup tests.
- Maintenance: dependency automation coverage tests or config validation, separate Bandit profile validation, oversized audio download boundary test.

## Proposed Backlog Task Map

The entries below are proposed child tasks. They should be created only after user approval.

### Decision-Gate Backlog Tasks

Gate 1 stays inside Track 1 because only the AuthNZ impersonation task depends on it. Gate 2 and Gate 3 should become concrete no-finding Backlog decision tasks so implementation tasks can depend on task IDs rather than abstract gate names.

#### Decide WebSocket Auth Contract

- Findings covered: none directly
- Priority: Medium
- Primary ownership: backend/API contract with frontend and MCP input
- Dependencies: none
- Acceptance criteria:
  - The repository documents one WebSocket auth contract for browser clients, ACP streams, and sandbox streams.
  - The contract defines default query-token behavior, first-frame auth semantics, scoped JWT enforcement expectations, and test expectations.
  - Tracks 4 and 9 reference the decision task before implementation.
- Verification:
  - Contract note is linked from Tracks 4 and 9.
  - No implementation branch starts with conflicting WebSocket auth semantics.

#### Decide Durable Workflow Ownership Contract

- Findings covered: none directly
- Priority: Medium
- Primary ownership: backend/workflows, Jobs, Scheduler
- Dependencies: none
- Acceptance criteria:
  - The repository records whether accepted workflow execution is owned by Jobs or Scheduler.
  - The decision defines idempotency, startup repair, duplicate-fire collapse, shutdown, cancellation, and process-loss expectations.
  - Track 6 references the decision task before implementation.
- Verification:
  - Contract note is linked from Track 6.
  - Track 6 acceptance criteria align with the chosen owner.

### 1. Harden AuthNZ Impersonation Boundary

- Findings covered: `AUDIT-2026-06-27-AUTH-001`, `AUDIT-2026-06-27-AUTH-002`, `AUDIT-2026-06-27-AUTH-003`
- Priority: High
- Primary ownership: backend/AuthNZ
- Suggested file ownership: `tldw_Server_API/app/core/AuthNZ/`, AuthNZ API dependencies/endpoints, AuthNZ tests
- Dependencies: Gate 1
- Acceptance criteria:
  - Impersonation token lifetime matches the documented short TTL.
  - Actor and subject survive from token issuance into downstream request context.
  - Durable audit events capture impersonation issuance and impersonated actions.
  - PostgreSQL and SQLite lookup paths use backend-neutral query helpers.
- Verification:
  - Token decode tests assert `exp - iat`.
  - SQLite and PostgreSQL impersonation tests cover user/role lookup.
  - Audit attribution tests assert actor and subject fields.
  - Bandit over touched AuthNZ production paths.
- Parallelism notes: Can run with Tracks 2 and 3. Avoid changing shared audit service contracts without notifying related audit tasks.
- Stop conditions: Pause if existing AuthNZ schema cannot represent actor and subject without a migration.

### 2. Enforce Media Authorization And Tenant-Scoped Ingestion Storage

- Findings covered: `AUDIT-2026-06-27-MEDIA-001`, `AUDIT-2026-06-27-MEDIA-002`, `AUDIT-2026-06-27-MEDIA-003`
- Priority: High
- Primary ownership: backend/media and ingestion
- Suggested file ownership: media endpoints, MediaWiki ingestion, media DB/vector storage helpers, media tests
- Dependencies: decision on `media.create` versus a new `media.process` permission
- Acceptance criteria:
  - Processing-only media endpoints enforce the chosen permission gate.
  - MediaWiki ingest writes DB and vector data under the request user in multi-user mode.
  - Original-file persistence cleans up stored files if DB registration fails.
- Verification:
  - HTTP 403 tests for unauthorized media processing routes.
  - Multi-user MediaWiki ingest isolation tests.
  - Fake storage test asserting compensating delete on DB failure.
  - Bandit over touched media/ingestion production paths.
- Parallelism notes: Can run with Tracks 1 and 3. Coordinate with Track 8 if MediaWiki remote fetching uses outbound HTTP policy.
- Stop conditions: Pause if permission taxonomy requires a broader RBAC migration.

### 3. Repair SQLite Migration Durability

- Findings covered: `AUDIT-2026-06-27-DB-001`, `AUDIT-2026-06-27-DB-002`
- Priority: High
- Primary ownership: backend/database migrations
- Suggested file ownership: DB migration helpers, packaged migration files, migration tests
- Dependencies: decide supported minimum Media DB schema version
- Acceptance criteria:
  - Legacy Media DBs below the supported minimum are either upgraded through a tested path or rejected with an explicit recovery message.
  - Multi-statement migration failure does not leave a successful ledger/schema bump.
  - Migration packaging no longer applies incompatible scripts to the wrong database domain.
- Verification:
  - File-backed legacy Media DB upgrade tests for representative old versions.
  - Failed multi-statement migration atomicity test.
  - Bandit over touched DB production paths if Python changes.
- Parallelism notes: Can run with Tracks 1 and 2. Avoid global migration framework changes that conflict with unrelated database work.
- Stop conditions: Pause if preserving every pre-v22 schema requires unavailable fixture data.

### 4. Align Browser WebSocket And OSS Billing API Contracts

- Findings covered: `AUDIT-2026-06-27-WEBUI-001`, `AUDIT-2026-06-27-WEBUI-002`, `AUDIT-2026-06-27-APIWEB-001`
- Priority: Medium
- Primary ownership: frontend and API contract
- Suggested file ownership: frontend audio clients, WebSocket helpers, OSS settings/billing UI, backend WebSocket tests
- Dependencies: Gate 2
- Acceptance criteria:
  - TTS, STT, and voice chat browser flows use the shared first-frame auth contract.
  - Backend tests reject query-token WebSocket auth when disabled by default.
  - OSS billing UI routes are hidden, disabled, or guarded by a backend capability signal.
- Verification:
  - Frontend client tests for first-frame auth.
  - Backend WebSocket auth tests.
  - OpenAPI or route-contract check for unguarded OSS billing calls.
  - Frontend lint/test command relevant to touched files.
- Parallelism notes: Can run with Track 9 after Gate 2. Coordinate on shared WebSocket helper semantics.
- Stop conditions: Pause if backend has no stable capability endpoint for hosted-only features.

### 5. Centralize Chat/RAG Authorization And Redact Query Logging

- Findings covered: `AUDIT-2026-06-27-CHAT-001`, `AUDIT-2026-06-27-CHAT-002`
- Priority: Medium
- Primary ownership: backend/chat and RAG
- Suggested file ownership: chat endpoints, RAG endpoints, resource authorization helpers, logging tests
- Dependencies: inventory externally reachable routes that spend LLM, RAG, or embedding resources
- Acceptance criteria:
  - Alternate RAG, character completion, document generation, and embedding routes enforce the same virtual-key and max-call rules as primary routes.
  - Info logs do not contain raw user query text.
  - Redacted logs keep enough request context for debugging.
- Verification:
  - HTTP tests with scoped virtual keys across alternate routes.
  - `caplog` tests proving raw query text is absent.
  - Bandit over touched Chat/RAG production paths.
- Parallelism notes: Can run in Wave 2 with Tracks 4, 6, 8, and 9.
- Stop conditions: Pause if endpoint identity or virtual-key semantics are inconsistent across providers.

### 6. Make Workflow Execution Durable And Idempotent

- Findings covered: `AUDIT-2026-06-27-JOBS-001`, `AUDIT-2026-06-27-JOBS-002`, `AUDIT-2026-06-27-REL-001`
- Priority: Medium
- Primary ownership: backend/workflows, Jobs, Scheduler
- Suggested file ownership: workflow service, Jobs or Scheduler ownership code, schedule adapters, workflow tests
- Dependencies: Gate 3
- Acceptance criteria:
  - Accepted workflow runs have one durable execution owner.
  - Startup repair handles accepted but unowned work.
  - Recurring and continuation work use deterministic idempotency keys.
  - Duplicate schedule fires collapse to one run/task.
- Verification:
  - Process-loss and post-acceptance-failure tests.
  - Duplicate schedule-fire tests.
  - Shutdown/startup repair tests.
  - Bandit over touched workflow/Jobs/Scheduler paths.
- Parallelism notes: Can run in Wave 2 after Gate 3. Avoid overlapping with unrelated Scheduler refactors.
- Stop conditions: Pause if Jobs versus Scheduler ownership cannot be decided from local code inspection.

### 7A. Establish Supply-Chain Foundations And Worker Image Hardening

- Findings covered: `AUDIT-2026-06-27-OPS-002`, `AUDIT-2026-06-27-DEPS-001`, `AUDIT-2026-06-27-DEPS-002`
- Priority: Medium
- Primary ownership: CI/release and dependency management
- Suggested file ownership: Python dependency files, Dockerfiles for worker images, CI tool setup, static-analysis setup
- Dependencies: choose Python lock/constraints strategy
- Acceptance criteria:
  - Runtime, CI, Docker, and release installs consume a committed lock/constraints profile or documented equivalent.
  - Static-analysis/action/tool installers are pinned to releases, checksums, or immutable SHAs.
  - Worker and audio-worker images run as non-root and minimize build tooling in runtime layers.
- Verification:
  - Dependency resolution or lock validation command.
  - Docker image build/inspection where available.
  - CI/tool pinning validation.
  - Bandit only if Python production code changes.
- Parallelism notes: Track 7B can inspect release gates in parallel but should wait for final tool setup decisions.
- Stop conditions: Pause if lock strategy affects packaging policy beyond this remediation program.

### 7B. Close Release Verification Gates

- Findings covered: `AUDIT-2026-06-27-OPS-001`, `AUDIT-2026-06-27-OPS-003`, `AUDIT-2026-06-27-OPS-004`, `AUDIT-2026-06-27-OPS-006`
- Priority: Medium
- Primary ownership: CI/release
- Suggested file ownership: GitHub workflows, Docker build matrix, SBOM workflows, Kubernetes samples
- Dependencies: Track 7A decisions for pinned setup and tooling
- Acceptance criteria:
  - PR container build matrix includes every published image, including worker and audio-worker images.
  - actionlint covers all workflow and composite action files.
  - SBOM generation includes Python plus Bun-managed frontend/admin dependencies.
  - Kubernetes sample secrets use safe placeholders or generated values and validate correctly.
- Verification:
  - actionlint over all workflows and composite actions.
  - SBOM workflow dry run or local equivalent.
  - Docker build matrix validation where available.
  - Kubernetes sample validation.
- Parallelism notes: Can start after or alongside Track 7A inspection, but implementation should not duplicate tool setup.
- Stop conditions: Pause if local environment cannot validate Docker or SBOM behavior and CI is required.

### 8. Route Integrations Through Central Outbound HTTP Policy

- Findings covered: `AUDIT-2026-06-27-INTEGRATIONS-001`, `AUDIT-2026-06-27-INTEGRATIONS-002`, `AUDIT-2026-06-27-INTEGRATIONS-003`
- Priority: Medium
- Primary ownership: backend/integrations and HTTP client
- Suggested file ownership: workflow research adapters, tokenizer resolver, weather provider, central HTTP client tests
- Dependencies: define allowed local-provider exceptions
- Acceptance criteria:
  - Workflow research adapters use central outbound HTTP policy for fetches and direct `pdf_url` downloads.
  - Tokenizer resolver uses central HTTP policy or a documented local-provider exception.
  - Weather provider uses central HTTP defaults or explicitly safe client configuration.
- Verification:
  - Private/loopback URL denial tests.
  - Proxy and `trust_env` behavior tests.
  - Tokenizer URL/provider base URL tests.
  - Bandit over touched integration production paths.
- Parallelism notes: Can run in Wave 2. Coordinate with Track 2 if MediaWiki ingestion touches outbound fetch policy.
- Stop conditions: Pause if local providers require network exceptions that are not represented in central policy.

### 9. Enforce MCP Scoped WebSocket Auth And Cleanup Lifecycle

- Findings covered: `AUDIT-2026-06-27-MCP-001`, `AUDIT-2026-06-27-MCP-002`
- Priority: Medium
- Primary ownership: backend/MCP, ACP, sandbox
- Suggested file ownership: ACP WebSockets, sandbox streams, scoped-token helpers, reconnect lifecycle tests
- Dependencies: Gate 2
- Acceptance criteria:
  - Scoped JWT endpoint, method, path, scope, and quota restrictions apply to ACP and sandbox WebSocket handshakes.
  - ACP reconnect replay stops broadcasters and removes event-bus subscribers on disconnect.
  - Existing ownership checks remain intact.
- Verification:
  - Scoped JWT rejection tests for ACP stream, ACP SSH, sandbox run stream, and sandbox stdin.
  - Reconnect-disconnect lifecycle test asserts broadcaster task and subscriber counts return to baseline.
  - Bandit over touched MCP/sandbox production paths.
- Parallelism notes: Can run with Track 4 after Gate 2. Coordinate on shared WebSocket auth semantics.
- Stop conditions: Pause if current scoped-token helper cannot evaluate WebSocket route claims without API changes.

### 10. Clean Up Dependency Automation, Bandit Profiles, And Small Test Gaps

- Findings covered: `AUDIT-2026-06-27-OPS-005`, `AUDIT-2026-06-27-DEPS-003`, `AUDIT-2026-06-27-MEDIA-004`
- Priority: Low
- Primary ownership: CI/maintenance and tests
- Suggested file ownership: dependency automation config, Bandit configuration, oversized audio download tests
- Dependencies: none, but avoid conflicting with Tracks 2 and 7A
- Acceptance criteria:
  - Dependency update automation covers intended nested Bun, Python, and Go roots or documents explicit exclusions.
  - Bandit production profile excludes test directories while a test profile remains available for review.
  - Oversized audio download regression test invokes the downloader and asserts the expected size error.
- Verification:
  - Dependency automation config validation.
  - Bandit profile validation.
  - Focused oversized audio download test.
  - `git diff --check`.
- Parallelism notes: Can run opportunistically when it does not overlap active media or supply-chain work.
- Stop conditions: Pause if dependency automation policy requires maintainer preference on included package roots.

## Backlog Creation Strategy

After this spec is approved:

1. Create one umbrella Backlog remediation task that references this spec and the audit final report.
2. Create the two shared decision-gate tasks for Gate 2 and Gate 3.
3. Create child Backlog tasks from the 11 proposed remediation task map entries.
4. Add dependencies using concrete task IDs: Tracks 4 and 9 depend on the WebSocket auth contract task, Track 6 depends on the durable workflow ownership task, and Track 7B depends on Track 7A.
5. Add any user-selected sequencing constraints.
6. Assign one implementation worktree per active child task.
7. Keep each child task current with reproduction results, implementation notes, verification, residual risk, and final summary.

Backlog tasks should be created through the MCP workflow when available. If Backlog tooling cannot preserve task markers, pause for user approval before direct task-file edits.

## Parallel Agent Operating Rules

- Prefer one agent per child task and one worktree per active branch.
- Each agent receives explicit file ownership, findings covered, acceptance criteria, verification commands, and stop conditions.
- Agents may inspect outside their write scope but must not edit outside the approved scope.
- If a task needs a shared contract change owned by another track, the agent pauses instead of implementing a second version.
- Code review checkpoints happen after the reproduction pass and before final commit.
- No task can claim completion without fresh verification evidence.
- AI-authored PRs require the repository's human-written Change summary before merge readiness.

## Spec Acceptance Criteria

This spec is accepted when:

- All 31 accepted audit findings map to one proposed remediation task.
- The three decision gates are explicit.
- Shared gates that affect multiple remediation tasks have concrete Backlog decision-task modeling.
- The four execution waves and wave integration gates are explicit.
- Universal, documentation-only, and track-specific verification rules are explicit.
- Closure rules prevent overclaiming on `needs_reproduction` findings.
- Child remediation tasks are proposed but not created.
