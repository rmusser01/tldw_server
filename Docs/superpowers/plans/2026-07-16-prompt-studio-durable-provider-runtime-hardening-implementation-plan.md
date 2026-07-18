# Prompt Studio Durable Provider Runtime Hardening Implementation Plan

Parent task: TASK-12963
Tracked unit: TASK-12963.2
PR: https://github.com/rmusser01/tldw_server/pull/2727

## Stage 1: Secret-free optimization configuration

**Goal**: Define one bounded, typed optimization model configuration and reject or scrub credential-bearing internals before database and Jobs persistence.

**Success Criteria**:
- Public optimization and comparison requests normalize to canonical `provider`, `model`, and `parameters` only.
- `api_key`, credential fields, `app_config`, `credentials_resolved`, runtime handles, suffix/prefix variants such as `api_key_override`, `azure_client_secret`, and `github_access_token`, and nested variants cannot cross either durable boundary.
- Legacy persisted rows are scrubbed before execution and never regain server-config fallback authority from caller-controlled flags.

**Tests**:
- Schema and endpoint regressions for direct and recursively nested secret fields.
- Database/Jobs payload assertions proving sentinels and credential handles are absent.
- Concurrent request isolation for two provider/model configurations.

**Status**: Complete

## Stage 2: Owner-scoped runtime lifecycle

**Goal**: Create one execution-scoped `ProviderCredentialRuntime` per durable optimization job and per `/test-cases/run` request.

**Success Criteria**:
- The runtime scope is derived from authoritative owner identity and current server-side membership, never from serialized credentials.
- Each provider/model is resolved server-side and passed to bounded TestRunner/PromptExecutor calls as an authoritative captured snapshot.
- Validated successes are marked exactly once, cancellation drains owned work before close, and runtime close always occurs in `finally`.
- Cached job processors never retain a runtime or credential handle between jobs.

**Tests**:
- Worker and endpoint adapter-boundary tests for resolve/mark/close ordering.
- Event-gated cancellation and same-user/different-user concurrency isolation.
- Revoked/store-unavailable/missing credential failures remain typed, sanitized, and fail closed.
- Admin cross-owner create/status/history parity and worker-time authoritative test-case ownership revalidation.

**Status**: Complete

## Stage 3: Canonical provider/model propagation

**Goal**: Use one normalized non-secret provider/model/parameters contract across evaluation, candidate generation, scoring, and refinement.

**Success Criteria**:
- `model_name`, `llm_model_config`, and legacy aliases normalize once at submission/loading; conflicting aliases fail closed.
- MIPRO, bootstrap, and MCTS use the selected provider/model consistently.
- Any deliberately distinct optimizer model is explicit, non-secret, and resolved through the same job runtime.
- No production optimizer path silently hard-codes OpenAI when another provider was selected; model-only requests resolve authoritatively or fail closed when ambiguous.
- Unsupported historical strategy names are rejected instead of silently running a different billable algorithm.

**Tests**:
- Cross-strategy provider/model propagation regressions.
- Legacy alias compatibility controls.
- Concurrent multi-provider isolation with distinct snapshots and marks.

**Status**: Complete

## Stage 4: Strict provider-failure semantics

**Goal**: Prevent provider/config/capacity failures from becoming zero-score or empty-result successes.

**Success Criteria**:
- Optimization use of TestRunner has a strict mode that propagates sanitized provider failures to Jobs.
- Expected-output mismatches remain ordinary scores, while auth/config/store/capacity/timeout failures fail or retry the job.
- At least one validated baseline provider response is required before an optimization can complete.
- `/test-cases/run` no longer converts provider failure into an empty successful response.
- Simple-create and comparison routes share bounded schemas, rate limits, atomic idempotency, bounded unique fan-out, and explicit retry-safe semantics before any side effect.

**Tests**:
- Adapter-boundary matrices for canonical in-band errors, raised failures, malformed results, capacity exhaustion, and timeout.
- Job completion regressions proving zero validated calls cannot produce `completed`.
- Valid zero-score evaluation compatibility controls.

**Status**: Complete

## Stage 5: Integration and production-safety gate

**Goal**: Verify the durable workflow, direct Prompt Studio surfaces, and adjacent provider paths without regressions.

**Success Criteria**:
- Focused regressions pass at deterministic seeds 12963 and 2727.
- Full Prompt Studio, Jobs, Messages, and affected Chat adapter suites pass.
- Touched production code passes pycompile, fatal Ruff checks, Bandit, secret/sentinel scan, and `git diff --check`.
- Independent review reports no unresolved Critical or Important findings.
- Backlog notes record deployment compatibility and the requester-authored PR Change summary remains an explicit merge gate.

**Tests**:
- Focused and widened seeded pytest matrices.
- Durable payload secret scan and cancellation/concurrency stress cases.
- Post-rebase high-risk rerun before push.
- SQLite and real-PostgreSQL lease exhaustion, completion bookkeeping, post-commit observer, paged cancellation reconciliation, archived cancellation, and tenant-database shutdown parity.

**Status**: Complete
