# Branch-Aware WebUI and Extension Artifact Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `main` the only branch that builds production-ready WebUI and extension artifacts by default, while every other branch builds developer-oriented artifacts, with explicit `:prod` and `:dev` overrides for local work and CI.

**Architecture:** Add one shared build-profile resolver under `apps/scripts`, then route package build entrypoints through small package-specific wrappers that shape environment and artifact export behavior. Keep WebUI profile differences purely env-driven, keep extension canonical internal build roots stable for existing tooling, and materialize `-dev` only on exported install directories and archive names.

**Tech Stack:** Bun, Node.js ESM scripts, Next.js, WXT, Vitest, Playwright helper tests, GitHub Actions YAML

---

## Stage Overview

## Stage 1: Prepare Isolated Context And Baseline
**Goal**: Work from an isolated branch/worktree and confirm the current WebUI and extension script/test surface is reproducible before changing build entrypoints.
**Success Criteria**: A dedicated `codex/` branch exists, dependencies are installed, and baseline targeted Vitest suites pass before new assertions are added.
**Tests**:
- `cd apps/tldw-frontend && bunx vitest run __tests__/frontend-dev-config.test.ts __tests__/frontend-ci-networking-workflows.test.ts`
- `cd apps/extension && bunx vitest run tests/e2e/utils/extension-paths.test.ts tests/e2e/setup/build-extension.test.ts tests/e2e/utils/extension.launch.test.ts`
**Status**: Complete

## Stage 2: Lock The Shared Resolver And WebUI Contract With Red Tests
**Goal**: Add failing tests for branch-to-profile resolution, fallback behavior, WebUI env shaping, and package script wiring before implementing the resolver and wrapper.
**Success Criteria**: New tests fail because the shared resolver does not exist yet, `main`/feature-branch logic is not encoded anywhere, and the WebUI package scripts still call `next build` directly.
**Tests**:
- `cd apps/tldw-frontend && bunx vitest run __tests__/build-profile-resolver.test.ts __tests__/frontend-dev-config.test.ts`
**Status**: Complete

## Stage 3: Implement Shared Resolver And WebUI Profile-Aware Build Scripts
**Goal**: Add the shared resolver, add the WebUI profile wrapper, and wire both Turbopack and webpack artifact paths through the same branch-aware contract.
**Success Criteria**: `build`, `build:prod`, `build:dev`, `compile`, `compile:prod`, and `compile:dev` exist; `main` defaults to quickstart/production, non-`main` defaults to advanced/development, and unknown branch state defaults to `development`.
**Tests**:
- Stage 2 command
- `cd apps/tldw-frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build:dev`
- `cd apps/tldw-frontend && bun run build:prod`
**Status**: Complete

## Stage 4: Implement Extension Profile-Aware Browser And Archive Entry Points
**Goal**: Make browser-specific extension build and zip commands profile-aware, keep canonical internal build roots stable, and export developer-facing unpacked/zip artifacts with `-dev` suffixes.
**Success Criteria**: `build:chrome`, `build:firefox`, `build:edge`, `zip`, and `zip:firefox` become branch-aware; explicit `:prod` and `:dev` variants exist; and non-`main` branch builds export suffixed install/archive artifacts without breaking existing helper paths.
**Tests**:
- `cd apps/extension && bunx vitest run tests/unit/build-profile-wrapper.test.ts tests/e2e/utils/extension-paths.test.ts tests/e2e/setup/build-extension.test.ts`
- `cd apps/extension && bun run build:chrome:dev`
- `cd apps/extension && bun run build:chrome:prod`
- `cd apps/extension && bun run zip:dev`
**Status**: Complete

## Stage 5: Force Production In CI, Update Docs, And Verify End-To-End
**Goal**: Update required workflows and contributor docs so production paths stay explicit in automation and human instructions match the new contract.
**Success Criteria**: At least one required WebUI workflow and one required extension workflow force production builds, docs describe `main` vs non-`main` behavior accurately, and targeted verification passes. If the implementation remains JS/docs-only, explicitly record that Bandit is not applicable because no Python files were touched.
**Tests**:
- `cd apps/tldw-frontend && bunx vitest run __tests__/frontend-ci-networking-workflows.test.ts`
- `cd apps/extension && bunx vitest run tests/unit/workflow-build-profile.test.ts`
- Re-run the Stage 3 and Stage 4 verification commands after doc/workflow changes
**Status**: Complete

## File Map

- `apps/scripts/resolve-build-profile.mjs`
  Responsibility: pure branch/override-to-profile resolution helpers, exported so both package wrappers and tests can share one contract.
- `apps/tldw-frontend/scripts/build-with-profile.mjs`
  Responsibility: map `production` to quickstart env, map `development` to advanced env, validate env, and invoke `next build` with either Turbopack or webpack.
- `apps/tldw-frontend/package.json`
  Responsibility: replace direct `build`/`compile` calls with profile-aware wrapper scripts and explicit `:prod` / `:dev` variants.
- `apps/tldw-frontend/__tests__/build-profile-resolver.test.ts`
  Responsibility: resolver and WebUI env-shaping contract tests.
- `apps/tldw-frontend/__tests__/frontend-dev-config.test.ts`
  Responsibility: package-script contract assertions for new WebUI build/compile scripts.
- `apps/tldw-frontend/__tests__/frontend-ci-networking-workflows.test.ts`
  Responsibility: assert required WebUI workflows still pin advanced-mode E2E env where intended and force `build:prod` where production artifact validation is required.
- `apps/tldw-frontend/README.md`
  Responsibility: explain `main` vs non-`main` artifact behavior and explicit override commands.
- `apps/extension/scripts/build-with-profile.mjs`
  Responsibility: resolve artifact profile, run browser-specific packaged builds against canonical internal roots, and export suffixed developer-facing unpacked directories when needed.
- `apps/extension/scripts/zip-with-profile.mjs`
  Responsibility: run archive packaging with explicit profile control and apply `-dev` suffixes to exported archives for development profile outputs.
- `apps/extension/package.json`
  Responsibility: replace direct browser-specific build/zip commands with profile-aware wrappers and add explicit `:prod` / `:dev` variants.
- `apps/extension/tests/unit/build-profile-wrapper.test.ts`
  Responsibility: assert extension wrapper decisions, exported artifact naming, and stable canonical-root behavior.
- `apps/extension/tests/unit/workflow-build-profile.test.ts`
  Responsibility: assert required extension CI workflows call explicit production scripts.
- `apps/extension/tests/e2e/setup/build-extension.test.ts`
  Responsibility: preserve the canonical-root assumptions used by Playwright global setup after wrapper changes.
- `apps/extension/tests/e2e/utils/extension-paths.test.ts`
  Responsibility: preserve current canonical candidate ordering and, if needed, add exported-dev-artifact helper assertions.
- `apps/extension/README.md`
  Responsibility: document branch-aware artifact behavior and exported `-dev` install/archive outputs.
- `apps/extension/Contributing.md`
  Responsibility: document the developer-facing unpacked artifact locations and explicit override commands.
- `apps/extension/docs/Testing-Guide.md`
  Responsibility: keep test instructions accurate about canonical internal roots versus exported dev install directories.
- `.github/workflows/frontend-ux-gates.yml`
  Responsibility: force WebUI `build:prod` in the production bundle smoke gate.
- `.github/workflows/ui-watchlists-extension-e2e.yml`
  Responsibility: force extension `build:chrome:prod` in a required extension workflow.
- `Docs/superpowers/specs/2026-04-10-branch-aware-webui-extension-artifact-profiles-design.md`
  Responsibility: design reference already corrected for fallback behavior, browser-specific extension entrypoints, and stable canonical internal roots.
- `Docs/superpowers/plans/2026-04-10-branch-aware-webui-extension-artifact-profiles-implementation-plan.md`
  Responsibility: execution checklist and status tracking; update only to reflect actual progress/results.

### Task 1: Prepare The Worktree And Baseline

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-10-branch-aware-webui-extension-artifact-profiles-implementation-plan.md`

- [x] **Step 1: Create or switch to an isolated worktree**

```bash
git worktree add ../tldw_server2-branch-aware-artifacts -b codex/branch-aware-artifact-profiles
```

Expected: a new worktree exists on branch `codex/branch-aware-artifact-profiles`.

- [x] **Step 2: Install workspace dependencies once from the monorepo apps root**

Run: `cd apps && bun install --frozen-lockfile`

Expected: both `apps/tldw-frontend` and `apps/extension` can run Vitest and build wrappers without ad hoc dependency drift.

- [x] **Step 3: Run the baseline WebUI contract tests**

Run: `cd apps/tldw-frontend && bunx vitest run __tests__/frontend-dev-config.test.ts __tests__/frontend-ci-networking-workflows.test.ts`

Expected: PASS. This confirms the current WebUI script/workflow contract is reproducible before changes.

- [x] **Step 4: Run the baseline extension helper tests**

Run: `cd apps/extension && bunx vitest run tests/e2e/utils/extension-paths.test.ts tests/e2e/setup/build-extension.test.ts tests/e2e/utils/extension.launch.test.ts`

Expected: PASS. This confirms the extension’s canonical-root assumptions are stable before wrapper changes.

**Execution notes:** `cd apps && bun install --frozen-lockfile` completed successfully. Baseline WebUI Vitest suite passed (`2` files, `6` tests). Baseline extension Vitest suite passed (`3` files, `8` tests).

### Task 2: Add Shared Resolver And WebUI Red Tests

**Files:**
- Create: `apps/scripts/resolve-build-profile.mjs`
- Create: `apps/tldw-frontend/__tests__/build-profile-resolver.test.ts`
- Modify: `apps/tldw-frontend/__tests__/frontend-dev-config.test.ts`

- [x] **Step 1: Write the failing shared resolver and WebUI env-shaping tests**

```ts
import { describe, expect, it } from "vitest"

import { resolveBuildProfile } from "../../scripts/resolve-build-profile.mjs"
import { shapeWebuiBuildEnv } from "../scripts/build-with-profile.mjs"

describe("resolveBuildProfile", () => {
  it("maps main to production", () => {
    expect(resolveBuildProfile({ branch: "main" })).toBe("production")
  })

  it("maps feature branches to development", () => {
    expect(resolveBuildProfile({ branch: "feat/example" })).toBe("development")
  })

  it("defaults unknown branch state to development", () => {
    expect(resolveBuildProfile({ branch: "" })).toBe("development")
  })

  it("prefers explicit overrides", () => {
    expect(
      resolveBuildProfile({ branch: "main", override: "development" })
    ).toBe("development")
  })
})

describe("shapeWebuiBuildEnv", () => {
  it("forces quickstart settings for production", () => {
    const env = shapeWebuiBuildEnv("production", {
      NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000",
    })

    expect(env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE).toBe("quickstart")
    expect(env.NEXT_PUBLIC_API_URL).toBeUndefined()
    expect(env.TLDW_INTERNAL_API_ORIGIN).toBe("http://127.0.0.1:8000")
  })

  it("requires advanced-mode browser api settings for development", () => {
    const env = shapeWebuiBuildEnv("development", {
      NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000",
    })

    expect(env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE).toBe("advanced")
    expect(env.NEXT_PUBLIC_API_URL).toBe("http://127.0.0.1:8000")
  })
})
```

- [x] **Step 2: Extend the existing package-script contract test to assert new WebUI entrypoints**

Add assertions in `apps/tldw-frontend/__tests__/frontend-dev-config.test.ts` that:
- `build` runs the profile wrapper
- `build:prod` and `build:dev` exist
- `compile`, `compile:prod`, and `compile:dev` exist and route through the same wrapper

- [x] **Step 3: Run the new WebUI tests to verify they fail**

Run: `cd apps/tldw-frontend && bunx vitest run __tests__/build-profile-resolver.test.ts __tests__/frontend-dev-config.test.ts`

Expected: FAIL because the shared resolver and wrapper do not exist and `package.json` still calls `next build` directly.

- [x] **Step 4: Commit the failing-test checkpoint**

```bash
git add apps/tldw-frontend/__tests__/build-profile-resolver.test.ts \
        apps/tldw-frontend/__tests__/frontend-dev-config.test.ts
git commit -m "test: lock webui branch-aware build profile contract"
```

**Execution notes:** The red checkpoint intentionally adds only tests. `bunx vitest run __tests__/build-profile-resolver.test.ts __tests__/frontend-dev-config.test.ts` failed as expected because `../../scripts/resolve-build-profile.mjs` does not exist yet and `package.json` still points `build`/`compile` at direct `next build` commands instead of the future profile wrapper entrypoints.

### Task 3: Implement The Shared Resolver And WebUI Wrapper

**Files:**
- Create: `apps/scripts/resolve-build-profile.mjs`
- Create: `apps/tldw-frontend/scripts/build-with-profile.mjs`
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/tldw-frontend/__tests__/build-profile-resolver.test.ts`
- Modify: `apps/tldw-frontend/__tests__/frontend-dev-config.test.ts`

- [x] **Step 1: Implement the shared profile resolver as a pure helper first**

```js
import { execFileSync } from "node:child_process"

export function normalizeBuildProfile(value) {
  return value === "production" ? "production" : "development"
}

export function resolveBuildProfile({ override, branch } = {}) {
  const explicit = String(override || "").trim().toLowerCase()
  if (explicit === "production" || explicit === "development") {
    return explicit
  }

  const currentBranch = String(branch || "").trim()
  if (currentBranch === "main") {
    return "production"
  }
  if (currentBranch.length > 0) {
    return "development"
  }
  return "development"
}

export function getCurrentGitBranch(cwd = process.cwd()) {
  try {
    return execFileSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], {
      cwd,
      encoding: "utf8",
    }).trim()
  } catch {
    return ""
  }
}
```

- [x] **Step 2: Implement the WebUI wrapper with explicit env shaping**

```js
import { spawn } from "node:child_process"
import { validateNetworkingConfig } from "./validate-networking-config.mjs"
import { getCurrentGitBranch, resolveBuildProfile } from "../../scripts/resolve-build-profile.mjs"

export function shapeWebuiBuildEnv(profile, env = process.env) {
  const nextEnv = { ...env }

  if (profile === "production") {
    delete nextEnv.NEXT_PUBLIC_API_URL
    nextEnv.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    nextEnv.TLDW_INTERNAL_API_ORIGIN =
      nextEnv.TLDW_INTERNAL_API_ORIGIN || "http://127.0.0.1:8000"
  } else {
    nextEnv.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
  }

  validateNetworkingConfig(nextEnv)
  return nextEnv
}
```

Use a single bundler flag so both `build` and `compile` call one wrapper.

- [x] **Step 3: Wire `package.json` through the new wrapper**

Update scripts so they follow this shape:

```json
{
  "build": "node scripts/build-with-profile.mjs --bundler=turbopack && node scripts/verify-shared-token-sync.mjs --dir .next",
  "build:prod": "TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --bundler=turbopack && node scripts/verify-shared-token-sync.mjs --dir .next",
  "build:dev": "TLDW_BUILD_PROFILE=development node scripts/build-with-profile.mjs --bundler=turbopack && node scripts/verify-shared-token-sync.mjs --dir .next",
  "compile": "node scripts/build-with-profile.mjs --bundler=webpack && node scripts/verify-shared-token-sync.mjs --dir .next",
  "compile:prod": "TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --bundler=webpack && node scripts/verify-shared-token-sync.mjs --dir .next",
  "compile:dev": "TLDW_BUILD_PROFILE=development node scripts/build-with-profile.mjs --bundler=webpack && node scripts/verify-shared-token-sync.mjs --dir .next"
}
```

Keep the existing `verify-shared-token-sync.mjs --dir .next` behavior unchanged.

- [x] **Step 4: Run the WebUI tests and targeted build commands**

Run:
- `cd apps/tldw-frontend && bunx vitest run __tests__/build-profile-resolver.test.ts __tests__/frontend-dev-config.test.ts`
- `cd apps/tldw-frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build:dev`
- `cd apps/tldw-frontend && bun run build:prod`

Expected:
- Vitest PASS
- `build:dev` produces an advanced-mode artifact
- `build:prod` produces a quickstart-mode artifact without requiring a browser-visible `NEXT_PUBLIC_API_URL`

- [x] **Step 5: Commit the WebUI slice**

```bash
git add apps/scripts/resolve-build-profile.mjs \
        apps/tldw-frontend/scripts/build-with-profile.mjs \
        apps/tldw-frontend/package.json \
        apps/tldw-frontend/__tests__/build-profile-resolver.test.ts \
        apps/tldw-frontend/__tests__/frontend-dev-config.test.ts
git commit -m "feat: add branch-aware webui build profiles"
```

**Execution notes:** Stage 2 WebUI tests now pass (`2` files, `11` tests). `build:dev` and `build:prod` both passed once rerun outside the desktop sandbox; the sandboxed Turbopack process could not create its helper process on macOS and failed with `Operation not permitted (os error 1)`. Both successful builds emitted pre-existing Turbopack warnings from `apps/tldw-frontend/lib/documentation.ts` about broad documentation file patterns, but completed successfully and preserved the `.next` token-sync verification step.

### Task 4: Add Extension Red Tests And Implement Profile-Aware Browser/Archive Commands

**Files:**
- Create: `apps/extension/scripts/build-with-profile.mjs`
- Create: `apps/extension/scripts/zip-with-profile.mjs`
- Create: `apps/extension/tests/unit/build-profile-wrapper.test.ts`
- Modify: `apps/extension/package.json`
- Modify: `apps/extension/tests/e2e/setup/build-extension.test.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-paths.test.ts`

- [ ] **Step 1: Write the failing extension wrapper contract tests**

```ts
import { describe, expect, it } from "vitest"

import {
  getExportedArtifactDir,
  getExportedZipName,
} from "../../scripts/build-with-profile.mjs"

describe("extension build profile wrapper", () => {
  it("keeps production exported install directories unsuffixed", () => {
    expect(getExportedArtifactDir("chrome-mv3", "production")).toBe("build/chrome-mv3")
  })

  it("exports suffixed dev install directories without renaming canonical internal roots", () => {
    expect(getExportedArtifactDir("chrome-mv3", "development")).toBe("build/chrome-mv3-dev")
  })

  it("suffixes dev archives", () => {
    expect(getExportedZipName("chrome", "development")).toContain("-dev")
  })
})
```

Also extend `apps/extension/tests/e2e/setup/build-extension.test.ts` so it still expects Playwright global setup to look at canonical roots such as `.output/chrome-mv3` and `build/chrome-mv3`, not suffixed exported directories.

- [ ] **Step 2: Run the extension tests to verify they fail**

Run: `cd apps/extension && bunx vitest run tests/unit/build-profile-wrapper.test.ts tests/e2e/setup/build-extension.test.ts tests/e2e/utils/extension-paths.test.ts`

Expected: FAIL because the wrapper files and new package scripts do not exist yet.

- [ ] **Step 3: Implement browser-specific build wrappers that preserve canonical roots**

Use wrapper logic with this shape:

```js
const profile = resolveBuildProfile({
  override: process.env.TLDW_BUILD_PROFILE,
  branch: getCurrentGitBranch(),
})

await runWxtBuild({ browser: "chrome" })
await verifySharedTokens({ target: "chrome-mv3" })

if (profile === "development") {
  await exportInstallDir({
    from: canonicalBuildDir,
    to: path.join("build", "chrome-mv3-dev"),
  })
}
```

Important rules:
- do not rename `.output/chrome-mv3` or any canonical root WXT expects
- do not break existing Playwright helpers that load canonical paths
- only exported developer-facing install directories and archive names gain `-dev`

- [ ] **Step 4: Wire `package.json` through explicit browser and zip variants**

Update `apps/extension/package.json` so these commands exist and use the wrappers:

```json
{
  "build": "bun run locales:sync && bun run build:chrome && bun run build:firefox && bun run build:edge",
  "build:prod": "bun run locales:sync && bun run build:chrome:prod && bun run build:firefox:prod && bun run build:edge:prod",
  "build:dev": "bun run locales:sync && bun run build:chrome:dev && bun run build:firefox:dev && bun run build:edge:dev",
  "build:chrome": "node scripts/build-with-profile.mjs --browser=chrome",
  "build:chrome:prod": "TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --browser=chrome",
  "build:chrome:dev": "TLDW_BUILD_PROFILE=development node scripts/build-with-profile.mjs --browser=chrome",
  "zip": "node scripts/zip-with-profile.mjs --browser=chrome",
  "zip:prod": "TLDW_BUILD_PROFILE=production node scripts/zip-with-profile.mjs --browser=chrome",
  "zip:dev": "TLDW_BUILD_PROFILE=development node scripts/zip-with-profile.mjs --browser=chrome"
}
```

Mirror the same pattern for Firefox and Edge where applicable.

- [ ] **Step 5: Run targeted extension verification**

Run:
- `cd apps/extension && bunx vitest run tests/unit/build-profile-wrapper.test.ts tests/e2e/setup/build-extension.test.ts tests/e2e/utils/extension-paths.test.ts`
- `cd apps/extension && bun run build:chrome:dev`
- `cd apps/extension && test -d build/chrome-mv3-dev`
- `cd apps/extension && bun run build:chrome:prod`
- `cd apps/extension && test -d build/chrome-mv3`
- `cd apps/extension && bun run zip:dev`

Expected:
- tests PASS
- development build exports `build/chrome-mv3-dev`
- production build keeps `build/chrome-mv3`
- development zip includes `-dev` in the archive name

- [ ] **Step 6: Commit the extension slice**

```bash
git add apps/extension/scripts/build-with-profile.mjs \
        apps/extension/scripts/zip-with-profile.mjs \
        apps/extension/package.json \
        apps/extension/tests/unit/build-profile-wrapper.test.ts \
        apps/extension/tests/e2e/setup/build-extension.test.ts \
        apps/extension/tests/e2e/utils/extension-paths.test.ts
git commit -m "feat: add branch-aware extension artifact profiles"
```

### Task 5: Force Production In CI, Update Docs, And Verify Final Contract

**Files:**
- Create: `apps/extension/tests/unit/workflow-build-profile.test.ts`
- Modify: `apps/tldw-frontend/__tests__/frontend-ci-networking-workflows.test.ts`
- Modify: `.github/workflows/frontend-ux-gates.yml`
- Modify: `.github/workflows/ui-watchlists-extension-e2e.yml`
- Modify: `apps/tldw-frontend/README.md`
- Modify: `apps/extension/README.md`
- Modify: `apps/extension/Contributing.md`
- Modify: `apps/extension/docs/Testing-Guide.md`
- Modify: `Docs/superpowers/plans/2026-04-10-branch-aware-webui-extension-artifact-profiles-implementation-plan.md`

- [x] **Step 1: Add failing workflow guard tests**

Extend `apps/tldw-frontend/__tests__/frontend-ci-networking-workflows.test.ts` to assert the WebUI smoke gate calls `bun run build:prod`.

Create `apps/extension/tests/unit/workflow-build-profile.test.ts` with assertions like:

```ts
import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("extension workflow build profile contract", () => {
  it("forces production extension builds in the watchlists required workflow", () => {
    const source = readFileSync(
      path.resolve(__dirname, "../../../..", ".github/workflows/ui-watchlists-extension-e2e.yml"),
      "utf8"
    )

    expect(source).toContain("run: bun run build:chrome:prod")
  })
})
```

- [x] **Step 2: Update the required workflows to force production builds**

Modify:
- `.github/workflows/frontend-ux-gates.yml`
  - change the production bundle smoke gate from `bun run build` to `bun run build:prod`
- `.github/workflows/ui-watchlists-extension-e2e.yml`
  - change the required extension build step from `bun run build:chrome` to `bun run build:chrome:prod`

Leave branch-faithful or dev-oriented workflows alone unless they explicitly validate production artifacts.

- [x] **Step 3: Update contributor docs**

Update docs to say:
- `main` builds production artifacts by default
- non-`main` branches build development artifacts by default
- explicit `:prod` commands are the escape hatch for release-like artifacts on feature branches
- extension developer-facing install dirs and zips get `-dev`, while internal canonical roots used by tooling stay stable

- [x] **Step 4: Run the workflow contract tests**

Run:
- `cd apps/tldw-frontend && bunx vitest run __tests__/frontend-ci-networking-workflows.test.ts`
- `cd apps/extension && bunx vitest run tests/unit/workflow-build-profile.test.ts`

Expected: PASS.

- [x] **Step 5: Re-run final targeted verification**

Run:
- `cd apps/tldw-frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build:dev`
- `cd apps/tldw-frontend && bun run build:prod`
- `cd apps/extension && bun run build:chrome:dev`
- `cd apps/extension && bun run build:chrome:prod`
- `cd apps/extension && bun run zip:dev`
- `cd apps/extension && bun run zip:prod`

Expected: all commands succeed and produce the expected profile-specific outputs.

- [x] **Step 6: Handle the security check requirement explicitly**

If the implementation remains JavaScript/YAML/docs-only, record in the execution notes that Bandit is not applicable because no Python files were touched.

If any Python files are added during implementation, run:

```bash
source .venv/bin/activate && python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_branch_aware_artifacts.json
```

Expected:
- non-applicable case is explicitly documented
- applicable case produces no new findings in touched Python files

- [x] **Step 7: Commit the CI/docs slice**

```bash
git add apps/tldw-frontend/__tests__/frontend-ci-networking-workflows.test.ts \
        apps/extension/tests/unit/workflow-build-profile.test.ts \
        .github/workflows/frontend-ux-gates.yml \
        .github/workflows/ui-watchlists-extension-e2e.yml \
        apps/tldw-frontend/README.md \
        apps/extension/README.md \
        apps/extension/Contributing.md \
        apps/extension/docs/Testing-Guide.md
git commit -m "docs: document branch-aware artifact profile workflow"
```

**Execution notes:** Stage 5 workflow guard tests passed for both apps. Final targeted verification passed for:
- `cd apps/tldw-frontend && bunx vitest run __tests__/build-profile-resolver.test.ts __tests__/frontend-dev-config.test.ts __tests__/frontend-ci-networking-workflows.test.ts`
- `cd apps/tldw-frontend && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run build:dev`
- `cd apps/tldw-frontend && bun run build:prod`
- `cd apps/extension && bunx vitest run tests/unit/build-profile-wrapper.test.ts tests/unit/workflow-build-profile.test.ts tests/e2e/utils/extension-paths.test.ts tests/e2e/setup/build-extension.test.ts`
- `cd apps/extension && bun run build:chrome:dev`
- `cd apps/extension && bun run build:chrome:prod`
- `cd apps/extension && bun run zip:dev`
- `cd apps/extension && bun run zip:prod`

The WebUI Turbopack builds had to be rerun outside the sandbox because the in-sandbox execution failed with `Operation not permitted (os error 1)` while Turbopack spawned a helper process during CSS processing. That was an environment restriction rather than a contract failure. Follow-up hardening also closed two review gaps: development zips now rename the generated archive so only the `-dev` filename remains, and the Playwright global-setup fallback now calls the explicit production wrapper instead of raw `wxt build`. Bandit is not applicable here because only JavaScript/TypeScript, YAML, and Markdown files were touched.
