# Chat draft key lifecycle

**Task/spec:** TASK-13454, approved Chat UX Stage 1; latest dev `ba553fdc51` integrated. Delayed acknowledgement guard committed in `be8064b63b`.
**Goal:** Preserve source text/collapse metadata when an existing draft key changes, restore an empty destination explicitly, and protect user edits from late restoration.
**Architecture:** Reuse the existing `useDraftPersistence` identity, pending write, local bucket and setter. Flush source in the existing layout cleanup before key-effect cancellation. Use a committed text/metadata revision solely to fence restoration; it must not turn callback-identity changes into automatic writes. Continue existing owner revocation and tab isolation. No new store/API/framework. This repairs the existing keyed primitive; accepted conversation switch/bind producers and reachable fresh/temporary handles remain required next integrations.
**ADR assessment:** No new ADR; client draft bucket ownership/limits unchanged. Best-effort local retention is not a server save receipt.

## Stage 1: Reproduce key-transition loss
**Goal:** Establish real-registry baseline and meaningful red cases.
**Success Criteria:** Rapid source departure loses latest edit, absent destination inherits source, and deferred restoration overwrites a new edit before the fix.
**Tests:** Existing real keyed-registry composer harness with source/destination metadata and held storage read.
**Status:** Complete
**Evidence:** Three real-registry red cases established lost latest source text, inherited text in an absent destination, and a delayed stored value overwriting newer typing. Independent review added three further red cases: departure while hydration is pending, collapse-only changes after text debounce, and metadata-only changes blocking destination restoration.

## Stage 2: Repair shared ordering and restoration
**Goal:** Flush before key teardown and restore only an unchanged destination presentation.
**Success Criteria:** A→B→A recovers latest A and its collapse metadata; empty B is empty; newer input survives delayed hydration; revoked owner is never flushed/restored.
**Tests:** Red cases plus existing composer/tab/account/persistence suites.
**Status:** Complete
**Evidence:** Detached committed metadata and edit revision, key departure flush before teardown, explicit empty destination restoration, genuine-edit persistence during pending hydration, and retained flush callback after debounce. Untouched delayed destinations remain empty and source drafts intact; explicit clear and account revocation remain fenced.

## Stage 3: Verify, review and record
**Goal:** Reviewed incremental change without weakened privacy or queue behavior.
**Success Criteria:** Relevant suites/typecheck pass, no added lint warnings/errors, independent reviewers resolve major findings, task/plan current and local commit.
**Tests:** Affected suites, WebUI typecheck, baseline-relative lint, diff check and task normalization. Bandit N/A to TS/docs-only scope.
**Status:** Complete
**Evidence:** Six affected suites pass (89 tests), WebUI typecheck passes, touched ESLint files both have zero errors/warnings, diff check clean. Both independent follow-up reviews resolve all Important findings and report no Critical/Important issue in the bounded change. Bandit N/A to TypeScript/docs-only scope. Full conversation integrations remain required next work.
