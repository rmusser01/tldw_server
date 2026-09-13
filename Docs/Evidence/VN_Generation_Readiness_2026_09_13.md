# VN Generation Readiness Verification

Task: TASK-13249. Parent work: [#2021](https://github.com/rmusser01/tldw_server/issues/2021).
Delivery: [PR #2954](https://github.com/rmusser01/tldw_server/pull/2954).
Baseline: `dev` at `c70387f496d82fcee92926bf3715bf5cd240ba88`.
Design: [generation readiness and recovery](../Design/2026-09-13-vn-generation-readiness.md).

## Before And After

| Evidence | Before | After |
| --- | --- | --- |
| Workbench Start request; regression test | Sent an empty request despite the API requiring an idempotency key. | Start sends a key; a retry after lost acknowledgement reuses the same key. |
| Failed-slot workbench tests and Chromium DOM | No slot-level retry in the monitor. | A named Retry action submits only the failed slot; repeated clicks are blocked while pending. |
| Chromium network mocks and named status region | No active generation polling. | A queued retry advances to completed automatically; the failed-slot row disappears. |
| Initial-load failure component test | No dedicated generation refresh recovery. | Refresh reloads status and configuration without starting work. |
| Pack-switch component tests | Stale asynchronous results and review selection could survive changing packs. | Generation responses from the previous pack are ignored; old items and review selection clear. |
| 390px Chromium DOM measurement and screenshot | Implicit grid track expanded panels to 394px, outside the viewport. | Explicit mobile tracks and a constrained archive input fit the viewport; overflow assertion passes. |
| Owner-scoped API tests | No generation configuration preflight. | Slot/pack/server backend precedence and missing configuration are reported without jobs or worker-health claims. |

The responsive correction is limited to the existing workbench layout and archive
input. No route, shared design system, generation job semantics or provider APIs
were changed.

## Checks

- Backend: **90 passed**, across `test_vn_assets_api.py` and
  `test_generation_jobs.py`; run with the project Python virtual environment.
- Frontend: **31 passed**, eight files via
  `bunx vitest run __tests__/vn-assets --reporter=dot`.
- Browser: **3 passed**, via
  `TLDW_WEB_URL=http://localhost:8087 TLDW_WEB_AUTOSTART=false bunx playwright test e2e/smoke/vn-assets.spec.ts --project=chromium --reporter=line`.
- ESLint: passed for all changed TypeScript/TSX files.
- Bandit: zero findings and no skipped files across `preflight.py`,
  `endpoints/vn_assets.py` and `schemas/vn_asset_schemas.py`.
- `git diff --check`: passed.
- `bun run typecheck`: exit 2 on both the branch and unchanged baseline. Complete
  outputs were byte-for-byte identical (`diff -u` exit 0), with 90 existing
  TypeScript diagnostics in unrelated presentation, prompt and certification
  code. No new diagnostic was introduced; the repository is not typecheck-clean.

The tests were first observed failing for absent preflight support, missing
Start keys, absent slot retry and stale review selection before their fixes.
The mobile overflow assertion also failed before the width correction.

Independent review found three additional refresh races. All were reproduced
with failing tests and corrected: details are read after generation status so a
terminal batch cannot strand older item results; old-pack refresh callbacks are
ignored before touching load state; and same-selection refresh requests share
an in-flight promise and settle all detail requests even when one fails.
A follow-up regression verifies that successful review mutations invalidate an
older refresh and wait for it to finish before reading fresh results, rather
than allowing that earlier snapshot to overwrite an approval.

## Browser Evidence

Chromium exercised the existing create/matrix/review/export workflow and a failed
slot retry with a lost first response, idempotent replay and automatic completion.
Recovery ran at **1440 x 1000** and **390 x 844**. Both failure and recovered
screenshots were captured after `waitForVisualSettle`; mobile failure and desktop
recovery screenshots were visually inspected for legibility and control fit.

The test regenerates `generation-failure.png` and `generation-recovered.png` in
each viewport's directory under `apps/tldw-frontend/test-results/`. These are local
test artifacts, not committed assets. The final run used a restarted webpack dev
server because an earlier run served a stale worktree bundle.

## PR Review Follow-Up

The requested rebase found the branch already based on the latest `dev` commit
above. The requester supplied the human-written `Change summary`, which was
copied verbatim into the PR before it was marked ready.

| Qodo finding | Disposition and regression evidence |
| --- | --- |
| 1: stalled request blocks another pack | Pending commands and their busy state are now keyed by pack. A regression starts pack B while A remains pending, then proves A's completion cannot unlock B. |
| 2: preflight lacks finite rate limit | Authentication precedes the named `vn_assets.preflight` RBAC dependency. Its catalog entry uses standard limits; HTTP regressions verify the same user-specific key for JWT/API-key principals, finite limits, 429 and `Retry-After`. |
| 3: missing test docstrings | Added concise behavior descriptions to all three preflight integration tests. |
| 4: missing test argument types | Added concrete fixture and parametrized argument annotations to the same tests. |
| 5: missing helper return type | Declared the TypeScript setup helper's `void` return contract. |
| 6: absent effective default model | Preflight and five adapters share provider-free model resolution, preserving request/environment/config/built-in precedence. Tests cover cloud defaults, overrides and local model-path non-disclosure. |
| 7: matrix does not refresh diagnostics | Not reproduced: successful matrix application replaces `selectedPack`, which already triggers the preflight effect. Added a regression that passed before changing production logic and verifies fresh diagnostics after matrix application. No redundant request was added. |
| 8: no direct core unit tests | Added isolated calls covering backend precedence, all configuration statuses, model precedence, empty slots, all worker-flag combinations and absence of adapter loading. |

The initial `backend-required` failure identified an omitted generated OpenAPI
fingerprint. Regeneration with CI-compatible schema libraries produces the exact
CI hash (`f68c012c4d2ec10fa9d3b76e99767a054313470afe980b7b40df8127c2dfb61e`),
2095 paths and 3204 schemas. A subsequent exporter `--check` passed after all
endpoint changes. Local dependency overrides were confined to `/tmp`; project
dependencies were not changed.

Independent review caught that the shared limiter normally keys by principal,
which separates JWT and API-key budgets for one user. Preflight now opts into a
user-wide bucket through `per_user=True`. Other endpoint behavior is unchanged;
a frozen-clock unit test verifies opt-in aggregation, legacy principal isolation,
and independent budgets for different users.
The follow-up review found no further issue. Buckets remain process-local, as
with the existing shared limiter; this is not a cluster-wide quota guarantee.

Review verification:

- 136 auth-hardening, preflight, model-resolution and VN API tests passed.
- 46 adapter/configuration and generation-job tests passed.
- 9 privilege-catalog tests passed.
- 33 VN frontend tests passed; scoped ESLint and production Python Ruff passed.
- All 3 Chromium scenarios passed again at desktop and mobile sizes. Failure
  and recovery screenshots were recaptured and visually inspected.
- Bandit reported zero findings across the touched production Python scope.
- Full type checking retained exactly the baseline's 90 diagnostics, verified
  with `diff -u` of diagnostic lines; no new VN diagnostic appeared.

## Limits And Remaining Work

- Browser generation/provider responses were mocked. No paid generation, GPU
  backend, external worker deployment, extension build or full-repository suite
  was exercised.
- Preflight inspects API-process configuration only. It does not verify worker
  liveness, credentials, provider reachability, model loading or available GPU
  capacity; differently configured external workers remain supported.
- Pending client keys last only while the page remains mounted. Browser reload
  recovery and immutable recipe snapshots remain follow-up work in #2021.
- Worker crash recovery and duplicate-delivery persistence are not addressed by
  this slice. #2021 remains open; this is not a full workstream completion claim.
- Issues #2021 through #2027 were reconciled against current implementation and
  registered as actual children of #1391. Other VN workstreams remain separate.
- The human-written summary gate is satisfied. Merge still requires the final
  head's CI checks and review disposition; local tests do not replace that gate.
