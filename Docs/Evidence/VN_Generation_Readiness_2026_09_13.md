# VN Generation Readiness Verification

Task: TASK-13249. Parent work: [#2021](https://github.com/rmusser01/tldw_server/issues/2021).
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
- A human-written `Change summary` explaining what changed and why is required
  before merging the AI-authored PR, per repository policy.
