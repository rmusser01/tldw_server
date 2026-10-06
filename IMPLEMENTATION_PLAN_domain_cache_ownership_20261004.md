# Domain Cache Ownership Implementation

Task: TASK-13425
Spec: Docs/superpowers/specs/2026-10-04-domain-cache-ownership-design.md

ADR required: no. Reuse native authority comparison and account-boundary events;
no new authentication or persistence rule.

## Stage 1: Reproduce
**Goal**: Demonstrate current-dev shared-cache ownership failures.
**Success Criteria**: Actual domain/client tests fail for stale cached values,
in-flight joins, or completion effects across native boundaries.
**Tests**: JWT/API-key/server/org/cookie boundaries and delayed transport.
**Status**: Complete

Evidence: initial current-dev suite had 84 ownership failures and 12 passing
same-owner compatibility cases. Subsequent self-review race tests demonstrated
12 post-await failures and 6 stale configuration-failure cleanup failures before
their respective fixes.

## Stage 2: Fence
**Goal**: Bind shared domain caches to current connection authority and epoch.
**Success Criteria**: All new regressions pass; owned/fresh bypass is unchanged;
old completion cannot remove a newer in-flight request.
**Tests**: Regression file, saved-profile scope tests, related client suites.
**Status**: Complete

## Stage 3: Verify and Publish
**Goal**: Review and publish a generic upstream-only fix against dev.
**Success Criteria**: Focused tests, typecheck results, security/ADR assessment,
task notes, commit, push, and PR are recorded; no private artifacts published.
**Tests**: Focused Vitest, TypeScript, diff/privacy self-review.
**Status**: Complete

Verification basis: upstream dev `502da5bf0ccd1bc3aa4323e0d0fc430f36821a78`.
All 417 tests in ten focused suites passed, including 114 ownership regressions.
Focused TypeScript comparison reports the same seven baseline errors and zero
introduced diagnostics. Full UI checking exceeded Node's default heap; the
focused check completed with an 8 GiB limit. ESLint reports zero errors and the
same 830 warnings as baseline; the new regression file has no warnings.
Bandit is unavailable in the project venv and is not applicable to the
TypeScript-only touched scope. No new auth protocol or credential persistence is
introduced. Diff whitespace validation passed. Independent upstream-diff review
completed and identified a lower-level GET coalescing gap. Six real-transport
regressions failed with the new API-key owner receiving the previous owner's
response. Fenced reads now pass the existing native `configSnapshot`, preventing
transport-only joins while retaining domain single-flight. The updated ten-suite
matrix passes 423 tests; the ownership/scope subset passes 168 tests. Focused
TypeScript still reports the same seven baseline diagnostics and zero introduced
errors. Follow-up commit `a3881455da` is pushed. PR
https://github.com/rmusser01/tldw_server/pull/3170 is attached and ready for review
after verification. No dependency symlinks or private fixtures are published.
Human-authored change summary, upstream CI, and review remain merge gates; this
work does not assert private beta readiness. No merge or deployment was performed.

## Stage 4: Qodo Review Regressions
**Goal**: Investigate all five inline findings on PR #3170.
**Success Criteria**: Controlled RED tests for overlapping storage reads and the
real extension message/worker path; no fixed sleeps or helper-call ordinals.
**Tests**: Public/base/domain ownership, deferred storage failures, extension
authority mismatches, same-principal refresh, bounded storage-read budgets.
**Status**: Complete

## Stage 5: Review Fixes and Verification
**Goal**: Guard stale initialization and worker dispatch, reducing duplicate
config checks without retaining stale server/API-key credentials.
**Success Criteria**: All review regressions and related native suites pass;
native ESM and focused typecheck baseline comparisons recorded; all five review
dispositions supplied to the parent. No worker commit/push/merge/GitHub comment.
**Tests**: Focused Vitest matrix, TypeScript comparison, lint/security review.
**Status**: Complete

ADR reassessment: ADR required: no. Existing connection-authority and account
epoch contracts govern the correction. A safe comparison identifier across the
existing extension request channel is not a new authentication or persistence
protocol. ADR-059 governs task updates. Keep previous publication history above.

### Review Evidence

- Initial ownership RED: 9 failed / 117 passed. Expanded JWT/API-key overlap and
  storage-budget subset: 21 failed / 3 passed (114 unrelated cases skipped).
- Real extension RED: 7 failed / 1 passed (7 existing ingest cases skipped).
  Public/base/domain returned the worker's wrong-account character response.
- Identical cookie-config epoch RED: wrong-account response returned. Direct
  initializer guard mutation: 3 failures. Runtime-override guard mutation:
  wrong runtime API key dispatched. Mixed-scope RED: successful response instead
  of rejection. Each guard restored and verified GREEN.
- Follow-up product-message error/timeout RED: both returned an unchecked direct
  fallback response after a successful epoch handshake. Fenced extension reads
  now reject all messaging failures; direct transport cannot preserve the epoch.
- Final native ESM run: 572/572 tests, 17/17 suites, Node executing
  `node_modules/vitest/vitest.mjs` (Vitest 4.0.18), one worker, no file parallelism.
  Ownership suite: 141; real saved-profile transport: 13; worker suite: 25.
- Expanded focused TypeScript comparison against pre-review HEAD
  `a3881455da`: 83 baseline / 83 current diagnostics, zero introduced (normalize
  file/code/message, ignoring line shifts). Expanding the prior seven-error
  scope to include the worker and its imports exposes existing diagnostics;
  this is not a full typecheck pass. Six new test-fixture type diagnostics were
  corrected before the final comparison; no unrelated baseline fixes.
- Touched-scope ESLint comparison: baseline/current both 0 errors and 712
  warnings, zero introduced diagnostics. This follow-up scope differs from the
  earlier publication's 830-warning scope. Existing Next pages-location warning
  is unchanged when invoking its config from the repository root.
- Project Python basis: 3.12.13 from the existing upstream `.venv`. Bandit is
  absent (`No module named bandit`) and not applicable to TypeScript-only edits.
  Security review: no raw credentials/JWT in IPC, no credential persistence,
  no direct fallback on handshake rejection/error/timeout, epoch rechecked at
  fetch/completion, scoped runtime override disabled, mixed scopes fail closed.
- Task normalization and `git diff --check` pass. Nothing staged; retained
  `apps/packages/ui/node_modules` symlink remains untracked and excluded.
- No worker commit, push, merge, GitHub comment, deployment, or scheduler.
  Parent owns independent review/publication/CI/merge. Owner explicitly stated
  'change summary is waived'. Record this as an owner-authorized policy exception
  to the human-written Change Summary merge gate, not a fabricated human-written
  summary. Parent still requires fresh Qodo/CI/independent review before merge.
  Private integration decisions are outside this upstream-only worker scope.

### Verification Commands

From `apps/packages/ui`, with matching existing dependencies:

```sh
node node_modules/vitest/vitest.mjs run \
  src/services/tldw/__tests__/TldwApiClient.domain-cache-ownership.test.ts \
  src/services/__tests__/server-chat-profile.private-scope.test.ts \
  src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts \
  src/services/tldw/__tests__/TldwApiClient.configuration-guidance.test.ts \
  src/services/tldw/__tests__/TldwApiClient.session-credential-race.test.ts \
  src/services/tldw/__tests__/TldwApiClient.auth-commit.test.ts \
  src/services/tldw/__tests__/TldwAuth.refresh.test.ts \
  src/services/tldw/__tests__/single-user-credential.test.ts \
  src/services/tldw/__tests__/request-core.quickstart.test.ts \
  src/services/tldw/__tests__/request-core.refresh-timeout.test.ts \
  src/services/__tests__/tldw-api-client.quickstart-auth.test.ts \
  src/services/__tests__/tldw-auth.refresh-rotation.test.ts \
  src/services/__tests__/background-proxy.test.ts \
  src/services/__tests__/background-proxy.web-refresh.test.ts \
  src/services/__tests__/chat-surface-scope.test.ts \
  src/entries/__tests__/background.quick-ingest-authority.test.ts \
  src/entries/__tests__/background.recipe-persistence-owner.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Historical RED runs used a locally installed Bun executable. Portable invocations
use `bun run test` with the ownership file and
`-t 'old.*storage read|resolves storage'`, or the worker file and
`-t 'domain cache extension'`, `-t 'cookie-config'`, `-t 'runtime API-key'`,
and `-t 'combined snapshot|runtime API-key|cookie-config'`. Direct initialization
mutation used the ownership file with `-t 'older direct initialize'`.
The product-message fallback RED used the worker file with `-t 'product-message'`.

Historical typecheck: `node --max-old-space-size=8192 --expose-gc --input-type=module`
with the TypeScript compiler API and a disposable external focused config, not
a checked-in verification script. The config extended
`apps/packages/ui/tsconfig.json`, set `incremental: false` and `noEmit: true`,
and cleared `include`. Its explicit root files were the frontend Node type
declarations and `vite-env.d.ts`, UI `src/ambient.d.ts`, `TldwApiClient.ts`,
the `characters.ts` and `chat-rag.ts` domain mixins, and the ownership,
request-scope, and saved-profile test files named above. Resolve these paths
against the current checkout and add all changed `.ts` paths as root names to
reconstruct the focused comparison; the original external config is not a
repository artifact. Baseline compiler-host reads substitute
`git show a3881455da:<path>` for each changed source, without changing checkout
files. Compare diagnostic multisets by file/code/message.
Lint: native ESM ESLint API, `apps/tldw-frontend/eslint.config.mjs`, `lintText`
for all changed `.ts` sources and matching `git show a3881455da:<path>` baseline;
compare diagnostics by file/rule/severity/message.
Backlog: `PYTHONPATH=tools/backlog-py/src` with the existing Python 3.12 `.venv`,
`python -m backlog_py --cwd <checkout> task normalize --check <task-path>`.

### Inline Reply Drafts

**4178139988:** Fixed out-of-order initialization before shared config publication.
Each initialization has a generation; an older completion cannot overwrite a
newer published authority (412) or same-owner refreshed credentials. Cache
authority publication also checks that its captured config is still current.
Controlled delayed storage reads cover JWT/API-key public/base/domain paths and
direct initialize; RED reproduced invalidation/extra dispatch, GREEN preserves
the newer cache and makes no old-owner dispatch. The constructor does not
initialize implicitly; the direct-initialize test covers that entry point.

**4178139978:** Fixed the real extension-message path. IPC carries only a
SHA-256 authority identifier and opaque worker account/lifetime epoch, never
raw API keys or JWTs. A bounded check obtains the epoch; the worker verifies
authority during config resolution and epoch immediately at fetch/completion.
Identical cookie-config logout/login rotates the epoch, not the digest. Errors,
timeouts, cancellation, and mismatches do not fall back or publish caches.
Product-message errors/timeouts after the handshake also cannot fall back to a
direct request, since it cannot carry the verified worker epoch.
Runtime overrides cannot replace checked credentials; mixed snapshot/Service
Prompt scopes fail closed. Real worker tests cover wrong-account rejection,
same-principal JWT refresh, cookie epoch roundtrips, and cancellation.

**4178139980:** Removed all three fixed sleeps. Real transport now waits for the
second fetch dispatch before releasing the old response. Both newer-flight join
tests wait for the real in-flight map lookup, then check two dispatches and both
returned Bob values. Deferred storage replaces initializer mocks. No timer
scheduling assumptions remain in these ownership regressions.

**4178139982:** Removed the resource-dependent helper-call ordinal. The test
signals the account change at controlled payload consumption and asserts public
412 rejection plus a clean newer-owner read. It no longer counts internal
`getDomainCacheRevision` calls and remains green after removing a helper call.

**4178139992:** Removed duplicate `getConfig` token/session reads after fresh
hydration and replaced the character pre-dispatch storage recheck with the
synchronous revision guard. Read-budget tests enforce one effective-config
resolution per cache hit and at most two per fetch for JWT/API-key reads. Fresh
hydration remains necessary for server/API-key/removed-credential changes without
events; `getConfig` alone would retain stale authority. Completion still checks
fresh authority, so the optimization does not weaken cache publication fencing.

## Independent Review Follow-Up: Prompt Handshake Cancellation

**Status:** Incremental fix verified; parent owns final full-suite verification
and independent review. TASK-13425 remains In Progress. No commit, push, staging,
or merge. Existing staged 11-file Qodo patch is preserved.

**Finding:** An unresolved `tldw:connection-authority` reply prevented cancellation
from settling until its messaging timeout (GET 3000 ms, default POST 130000 ms).
The handshake only checked `abortSignal.aborted` before and after its reply/time
race, unlike the later product-message wait, which already observes aborts.

**Fix:** Register a handshake abort listener, recheck the signal after registration,
and reject promptly with AbortError. Remove the listener and clear the timer in
`finally` on success, error, timeout, and cancellation. A late worker reply cannot
dispatch the product request or trigger direct fallback. Explicitly type the
handshake response because the browser API's last callback overload returns void.

**Regression Coverage:** Eight added cases cover unresolved GET/POST handshakes,
late replies after cancellation, pre-aborted signals, abort during messaging,
abort before listener registration, and success/error/timeout cleanup. Success
checks listener removal before product dispatch, not merely eventual cleanup.

**RED:** Before the production fix, the following filter produced 6 failed,
2 passed, and 25 skipped tests in one suite (1.42 seconds). Cancellation outcomes
remained undefined; error/timeout paths had no handshake abort listener.

```sh
node node_modules/vitest/vitest.mjs run \
  src/entries/__tests__/background.quick-ingest-authority.test.ts \
  -t 'promptly cancels an unresolved|does not miss handshake cancellation|removes the handshake abort listener' \
  --maxWorkers=1 --no-file-parallelism
```

**GREEN:** The filter passed 8/8 cases (25 skipped). Final focused verification
after fixture/type corrections passed 225/225 tests in 3 suites (9.98 seconds):
worker authority 33, background proxy 130, web refresh 62.

```sh
node node_modules/vitest/vitest.mjs run \
  src/entries/__tests__/background.quick-ingest-authority.test.ts \
  src/services/__tests__/background-proxy.test.ts \
  src/services/__tests__/background-proxy.web-refresh.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Commands run from `apps/packages/ui`. Existing localStorage experimental warnings
and expected negative-path request diagnostics remain visible. Compiler-host
comparison against the staged sources, using the existing focused config plus
both incrementally touched TypeScript paths, reports 83 baseline/83 current
diagnostics, zero introduced; this is not a full typecheck pass. ESLint `lintText`
comparison of those same two paths reports 0 errors/15 warnings on both sides,
zero introduced; the existing Next pages-location warning remains. Whitespace
checks pass. Staged patch SHA-256 remains
`963bc50270ac847b5078ac3754fafbe2c2a35a7ba3d333942e169916b7601c9d`.

Task history is appended with backlog-py CLI, not manually edited. ADR assessment:
no new ADR; restore existing native cancellation semantics without auth or IPC
protocol changes. Owner Change Summary waiver remains in force. No full-suite,
publication, merge, deployment, or private hosted readiness claim is made here.

### Second-Pass Qodo Corrections

- Observe session `MANUAL_SESSION_KEY` replacement, removal and removal/return
  as worker epoch boundaries. Compare trimmed effective keys and exact credential
  metadata; identical keys, whitespace-only writes and unrelated session writes
  preserve pending work.
- Decode raw local/session storage changes once with the existing
  `safeStorageSerde`. Independent review reproduced the serialized-event gap;
  the fixture now imports the actual serde and emits its serialized values.
- Report only an actual handshake timer expiry as an extension timeout with
  the existing no-fallback marker. Scope mismatches still reject as `412`.
- Assert cancellation resource counts and late-abort behavior instead of a
  specific listener identity or pre-dispatch listener-removal order.
- Use portable verification instructions and update the task's current summary
  while retaining historical checkpoints and the owner's Change Summary waiver.

Parent reproduced five failing session-key/timeout cases before correction.
Final native ESM verification of the 17 files listed above passed **589 tests**,
zero failed or pending. Focused TypeScript comparison against `bc2e816161`
reports **83 baseline / 83 current diagnostics, zero introduced**; this is not
a full typecheck pass. Final independent source review has no actionable
findings. Fresh published-head Qodo, CI and merge remain pending; no deployment
or private hosted readiness is claimed.

### Third-Pass Qodo Corrections

The completed review updated at 20:14Z on `6a50c8fe39` added three findings.
All three were verified against the native worker, proxy and client paths:

- Refresh-session invalidation marker changes now rotate the worker connection
  epoch, rejecting already-resolved credentials before fetch and late responses.
  Ordinary same-principal refresh remains compatible.
- The UI account watcher now observes decoded session manual-key changes using
  the same credential/metadata comparison as the worker. Replacement/return and
  removal/return invalidate cached profiles and messages even without a read
  between boundaries; identical, whitespace-only and unrelated writes do not.
- Client and worker share build-time API-key lookup with unchanged precedence.
  Worker effective-config hydration now includes the fallback key before both
  handshake and request authority validation, while preserving stored keys and
  cookie-session isolation. IPC still contains no raw credentials.

RED: the two-file matrix had **17 failures / 201 passes**. Twelve failures
returned stale cache values across session roundtrips, two accepted invalidated
worker credentials/responses, and the Next environment key failed the authority
handshake. The Vite-key cases also exposed the previous nonstandard
`import.meta` access in the native test transform; the shared lookup now uses
the standard `import.meta.env` access. GREEN: **218/218** in those two files.

The documented 17-file matrix plus `background.effective-auth.test.ts` passed
**675/675 tests across 18 files**, zero failed or pending. The quickstart fixture
now provides a real environment input instead of spying on the extracted private
lookup. The focused compiler-host comparison to current `origin/dev` reports
**83 baseline / 83 current diagnostics, zero introduced**, not a full typecheck
pass. ESLint reports **0 errors / 953 baseline warnings / 950 current warnings**.
Existing Next pages-location and Node localStorage diagnostics remain visible.

ADR reassessment: no new ADR, authentication or persistence protocol. Bandit
cannot parse these TypeScript-only paths; manual security review checks opaque
authority/epoch IPC, checked worker dispatch, cancellation, and no fallback.
Owner Change Summary waiver remains recorded. Fresh rebased-head native
verification, published-head Qodo and all seven active required contexts remain
publication/merge gates; no merge or deployment is claimed at this checkpoint.

### Frontend CI Typecheck Follow-Up (2026-10-06)

**Status:** Verified locally; publication and final-head review/CI pending.
TASK-13425 tracks this authorized repair.

Clean rebase onto current dev `1fc353c3f67c93ba05102e7b0136ac4acac8f510`
preserves all seven patches; two range-diff entries change only surrounding
context because dev replaced the cache maps with its bounded TTL cache.

The actual frontend compiler reproduced the only diagnostic from the failed
`frontend-required` job: TS2339 on `processEnv.NEXT_PUBLIC_X_API_KEY`. The
conditional empty-object fallback inferred `{}`. Replace that temporary object
with guarded direct optional access to `process.env`, following existing
deployment helpers. Vite-first precedence and absent-process handling remain.
No new dependency, authentication protocol, or ADR is needed.

RED, from `apps/tldw-frontend`:

```sh
node --max-old-space-size=8192 node_modules/typescript/bin/tsc \
  --noEmit --incremental false
```

Exit 2 with the single TS2339. GREEN: the identical full frontend compiler
command exited 0 with zero diagnostics. The documented 17-file native matrix
plus `background.effective-auth.test.ts` passed **676/676 across 18 files**,
zero failed or pending. Touched-helper ESLint has zero errors/warnings; task
normalization and whitespace checks pass. Bandit was attempted in the project
venv but cannot parse this TypeScript helper. Manual security review confirms
unchanged key precedence, absent-process handling and no credential logging or
IPC changes. Independent incremental review found no actionable issues, and a
read-only in-memory check with installed Next Webpack confirmed synthetic
public-key substitution with zero emitted files. Publication remains pending.
All eleven Qodo inline threads are resolved; final-head review still requires available Qodo
credits. The owner's Change Summary waiver remains in force.
