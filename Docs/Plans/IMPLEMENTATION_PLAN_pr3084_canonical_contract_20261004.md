# PR3084 canonical contract repair — TASK13260.278.18.83.45

Human direction on2026-10-04: address the issues. This authorizes a separate supported-stack canonical regeneration and current-dev integration. Published b538 is the source baseline; local8e failure tracking remains on the original branch and is excluded from publication. Historical frontend/stack authorization holds are superseded only for the declared new operations. Existing lifetime/first-import/native and UAT STOP limits remain.

## Stage 1: Integrate current dev
**Goal**: Safely rebase the published repair source onto immediately verified dev.
**Success Criteria**: All three published commits and the complete six-path owned patch preserved; inspect incoming patches and independent review before publication.
**Tests**: Exact range-diff, binary patch comparison, source-tree delta and freshness checks.
**Status**: Complete

## Stage 2: Prepare supported canonical generation
**Goal**: Use existing Python3.12 in a separate task-owned environment with normal project dev dependencies.
**Success Criteria**: Declared floors met; isolated environment/cache and explicit source binding; no existing dependency/cache/credentials changed.
**Tests**: Version/package metadata and supported procedure review. Install cap600s/no retry; failure STOP.
**Status**: Complete

Setup prerequisites: The independent source/procedure review is clear. Use detached baseline devd7997bc2052ac52c77427157fe3c5b2e2ca1843f at /private/tmp/tldw-pr3084-canonical-20261004, existing Python3.12.11, its own .venv/.uv-cache and normal editable .[dev] install (cap600s/no retry). Native worktree tool was unavailable; exact Git worktree fallback is allowed.

The separate frozen frontend install naturally passed0 in84.225s and retained exact clean4c4f source. Actual openapi-typescript7.13.0/bin resolves in its frontend node_modules, not apps root. Before the ONE unchanged generator attempt, verify candidate apps/tldw-frontend/node_modules absent, create only that temporary owned link to /private/tmp/tldw-uat-frontend-locked-20261003-4c4f/apps/tldw-frontend/node_modules, and verify exact target/bin. Set PYTHON to the proper baseline .venv interpreter, BUN_INSTALL_CACHE_DIR to baseline/.bun-cache and PUPPETEER_CACHE_DIR to frontend/.puppeteer-cache. Exporter forces candidate source root. Remove only the owned link after exact-target verification and natural generator exit. No other dependency link, install or shared cache mutation; generator failure STOP/no retry.

Supported setup result: normal declared editable .[dev] installation naturally0 in86.179s, baseline tracked source clean. Actual Python3.12.11/FastAPI0.142.2/Pydantic2.13.5/Starlette1.7.0 meets declared floors. No environment reconstruction retry or shared target/cache changes.

## Stage 3: Regenerate and inspect the contract
**Goal**: Run official canonical exporter and frontend API-type generator; verify the complete schema delta before accepting a new fingerprint.
**Success Criteria**: Reproduced canonical drift, expected public root/readiness descriptions only, normal type generation, final supported exporter --check natural0. Keep descriptions and all assertions/CI intact.
**Tests**: One baseline export and candidate generation, exact recursive schema comparison, final drift gate; each operation cap600s/no retry. No stopped lifetime/GC/first-import controls or source-suite replay.
**Status**: Complete

Canonical result: official baseline exporter/check naturally0/57.062s, matching e384a65e with2105paths/3245schemas. ONE unchanged candidate API-type generator naturally0/33.915s on supported stack; actual locked openapi-typescript7.13.0 generated normal ignored schema/types and official fingerprint d31bcaa5. Temporary frontend dependency link removed after exact-target validation. Complete recursive comparison of all schema values finds exactly three added descriptions: root GET and /health/ready GET/HEAD, matching source docstrings; no other field, response, permission, parameter, operation or component-schema change. Counts and every non-SHA fingerprint field are identical. Final official candidate --check naturally0/19.907s. No manual digest substitution, description removal, CI/assertion weakening or stopped-control replay.

## Stage 4: Review and publish
**Goal**: Independently review actual patch and normally publish to the existing PR with exact lease/current-dev verification.
**Success Criteria**: Clear independent review; original human summary verbatim; no local8e publication; fresh seven-context hosted gates required. Native/full UAT acceptance remains open.
**Tests**: Diff check, touched-scope validation (Bandit N/A for generated JSON/prose without Python changes), exact committed patch, remote/body readback and new-head CI.
**Status**: In Progress

Independent final four-path review CLEAR: no Critical, Important or Minor findings. Reviewer independently compared every baseline/candidate schema value and both hashes, verified ignored generated outputs, absent owned dependency links, exact retained commit/patch lineage and unchanged task criteria/status. Normal commit and exact-lease publication are next; fresh final-head CI/native/full-UAT gates remain open.
