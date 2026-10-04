# PR3084 email tripwire caller repair implementation plan

Associated task: TASK-13260.278.18.83.45. Human explicitly requested fixing the remaining failures after the monitoring hold.

## Constraints
- Start from published e6a2; exclude all local closure/tracking commits. Integrate reviewed dev 502da5bf task-only changes while preserving all nine published commits and the diagnostic patch.
- Preserve all existing offline guards, assertions, production source and CI until causal evidence supports a repair.
- Earlier three diagnostic attempts remain exhausted. At most three new declared, distinct causal experiments; no unchanged full heavy shard replay, model download or old CI log redownload.
- Existing Python 3.12 environment only; no dependency/cache/model/environment workaround. Match actual bound CI inputs only to reproduce a stated difference.
- No proof files or bundles. Native/UAT/installer limits and separate pending API replacement approval remain intact.

## Stage 1: Identify the forbidden caller
**Goal**: Trace the actual failing path with content-free function metadata.
**Success Criteria**: Caller and causal trigger reproduced, or stronger content-free tripwire diagnostic published with unchanged rejection behavior to identify the hosted caller; no causal fix claim before evidence.
**Tests**: First declared experiment selects only the failing email upload while collecting the actual integration directory, using the bound CI Redis inputs and in-memory call tracing. Outer 600 seconds and per-case 300 seconds. No other test executes. Experiment 1 passed (1 selected/239 deselected/46 warnings/24.23s pytest/32.756s outer), no caller. Add a behavioral red/green test for boundary/caller diagnostics without argument retention, validate all existing email attachment and offline cases, then independently review and publish this diagnostic to obtain the hosted caller. Do not replay experiment 1 unchanged. New completed diagnostic-head job 111463085959/run37210348919 on 1e4212 (pull_request/attempt1/ci.yml verified) was read once in memory: 256 passed, 1 skipped, 5443 warnings, 1 error in 517.97s. The original options0 upload assertions passed; teardown identifies socket.getaddrinfo from Redis ping through acquire_migration_lock and AuthNZ ensure_authnz_tables. Static upload accounting creates an AuthNZ daily ledger even for unlimited uploads; the focused fixture excludes Auth/quota/billing but currently stubs only storage quota.
**Status**: Complete

## Stage 2: Repair the demonstrated cause
**Goal**: Minimal source or fixture correction with the offline contract intact.
**Success Criteria**: Causal red on unchanged source, green after correction; original assertions retained; affected tests pass naturally.
**Tests**: Experiment 2: a real synthetic upload with a cold ingestion ledger cache must not invoke either the usage-quota resolver or AuthNZ ledger initialization. Record both attempts, reject ledger bootstrap to avoid external work, and retain every existing upload assertion and outbound/model guard. Obtain red on the unchanged fixture, then exclude only the AuthNZ upload-accounting helper from this explicitly auth/billing-free harness and run the complete affected offline/attachment files. Red on the unchanged fixture: 1 failed, 40 deselected, 51 warnings in 2.15s pytest/4.480s outer/natural exit1; both quota lookup and ledger initialization occurred once. Correction adds only an AsyncMock for the accounting helper to the explicitly auth/billing-free fixture. Complete affected offline and attachment files: 62 passed, 542 warnings in 14.05s pytest/16.595s outer/natural exit0, no skips, using bound Redis inputs and 600/300-second guards. All original functions/assertions remain identical except exactly one added fixture stub; one new cold-accounting regression. Ruff lint/format and compile0; scoped Bandit baseline/current40 findings,0errors,identical signatures/no new. Production quota/default/ledger policy and CI stay unchanged. Two of three new experiments used; no full hosted-shard/native/UAT acceptance. No broad native, frontend, export or model controls.
**Status**: Complete

## Stage 3: Review and publish
**Goal**: Review and publish the diagnostic, then any independently qualified causal fix to existing PR3084.
**Success Criteria**: Scoped baseline/current Bandit, lint/compile, independent actual-source and staged review, exact-head/dev/human-summary guards and normal hooks/exact-lease publication and fresh corrected-head hosted ingestion outcome; no merge acceptance.
**Tests**: Affected checks and supported owned-task format check. Complete incoming dev review verifies 2,271 task-only paths: 2,269 exact supported normalizations preserve frontmatter/content and are idempotent; separate TASK-13441 completion and TASK-13443 follow-up reviewed. Verify nine-commit range-diff and complete owned binary patch after integration. No unchanged application/native/build controls for this prose-only base. ADR assessment: no new architecture decision; ADR-059 task tooling unchanged. Independent actual-source three-path causal fixture review CLEAR with no findings on candidate patch44858ac4; final staged review/publication and fresh hosted ingestion qualification remain pending. Native/UAT/installer gates remain open.
**Status**: In Progress
