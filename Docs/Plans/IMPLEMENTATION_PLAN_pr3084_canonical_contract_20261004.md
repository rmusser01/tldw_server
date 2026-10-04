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

## New dev advance before publication

The immediate publication guard stopped before push: dev advanced from d799 to3700e2d6e7e7dc73266f9d2df1e47f97cd8dc70e (PR3144,70paths), while PR remains b538. Local reviewed a1d236ffae is clean/unpublished. Incoming per-user quotas add team/org override routes/schemas and update the fingerprint; d799 canonical results remain historical for that exact source. No old hash is accepted on new dev.

Before this new-source integration, official task records: inspect complete incoming production/config and affected contracts, rebase all4 commits preserving executable code/assertions and tracking, resolve only fingerprint overlap to exact new-dev fingerprint as the temporary baseline, then use the unchanged official exporter/type generator on the existing proper environment (pyproject unchanged/no new install). Move the task-owned detached baseline checkout to exact3700 only with tracked-clean verification; one baseline export/check, one new candidate generation and final check, each600s/no retry, then complete recursive schema comparison and independent final review. This is changed-source qualification; original successful operations and frontend ONE-build ceiling remain unchanged, with no lifetime/native controls or unchanged replay. Stop on source/dependency/generation discrepancy.

New-source result: Changed-source qualification after dev3700: rebase37ef1401d6 preserves first3 commits identically and complete owned binary patch except superseded fingerprint; fourth tracking patch otherwise identical. Complete old a1/new37 committed tree delta exactly70 incoming paths;59 changed/owned Python compile. Independent full incoming production/config review CLEAR/no findings, dependency manifests/main/control-plane/lifetime/assertions unchanged. Existing proper environment reused/no install. Exact new-dev baseline official export/check naturally0/27.344s, SHA873776fadd1cc4412af1dad706b0f640c9dfc0d5065918085de38d7d860547be,2107paths3247schemas. ONE unchanged candidate API-type generation naturally0/29.105s with locked openapi-typescript7.13.0 and exact owned frontend link removed. Complete recursive schema comparison is exactly the3 intended public descriptions; new team/org override paths and schemas retained, all non-SHA fingerprint fields identical. Candidate official SHA f74b0373b658b487a199de268e114ec56275e9e544fdea32312290058829b31c; final official --check naturally0/22.54s. Affected13file Usage/UserProfile/Chat402/officialSQLite group64passed/726warnings/29.47s pytest/35.482s outer/natural0/no skips reported. Incoming production Bandit26files0findings/0errors,22existing baselinefiles0/0; static existing Bandit executable after proper dev environment reported module unavailable, no installation. No own Python edits/new findings; prior identical-source checks retained. Final actual four-path review and normal exact-lease publication next, original8e excluded, human summary/status/AC/DoD unchanged, native/first-import/full UAT and final-head seven contexts open. No lifetime control replay, manual hash/CI/assertion workaround or proof artifacts.
