# Production Extension and Chatbooks Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the normal production Chrome MV3 artifact directly loadable and measurable, then certify a complete archive-only full-account Chatbooks restore in that exact artifact with zero skipped or omitted tests.

**Architecture:** `build:chrome:prod` produces the canonical exported artifact at `apps/extension/build/chrome-mv3`. One direct-output launcher fingerprints and loads that directory without test staging or manifest mutation. Package readiness (`TASK-12098.5`) finishes before final UAT (`TASK-12098.3`), which uses independent source/destination roots, an external-retrieval trap, phase-qualified reports, and the notification work from `TASK-12098.4`.

**Tech Stack:** WXT, Vite/Rollup, TypeScript, Node.js, Playwright, Chromium MV3, React, Python, FastAPI, pytest, Vitest, Bandit.

**Spec:** `Docs/superpowers/specs/2026-07-10-chatbooks-residual-uat-remediation-design.md`

**Backlog:** `TASK-12098.5` for package readiness; `TASK-12098.3` AC 17-20 for final certification.

---

## Stage Overview

| Stage | Goal | Success Criteria | Status |
| --- | --- | --- | --- |
| 1 | Immutable package identity | Fingerprint, symlink, manifest, sanitizer, and direct-launch tests pass | Not Started |
| 2 | Loadable production package | Asset isolation identifies the cause and the source-level fix makes package health pass | Not Started |
| 3 | Build integrity | Duplicate/circular/font warnings are removed and startup graph budgets pass | Not Started |
| 4 | Archive-only UAT | Source quarantine, independent destination, external trap, and phase aggregation are enforced | Not Started |
| 5 | Final certification | Fingerprinted extension and WebUI regression UAT pass with exact IDs and zero skips | Not Started |

## Ownership Boundary

- `TASK-12098.5` owns Tasks 1-5 and closes when package health proves the normal artifact is loadable and unchanged.
- `TASK-12098.3` owns Tasks 6-8 and closes only after `TASK-12098.4` is complete and the final browser evidence passes.
- Diagnostic subset directories may identify the package fault but can never satisfy package health or final UAT.

## File Responsibilities

### Package Identity and Launcher

- Create `apps/extension/tests/e2e/utils/package-output.ts`: manifest validation, regular-file fingerprint, symlink rejection, and package-health result schema.
- Create `apps/extension/tests/e2e/utils/diagnostic-sanitizer.ts`: Node retained-output sanitizer.
- `apps/extension/tests/e2e/utils/extension.ts`: one direct-output launcher with canonical realpath flags and target activation.
- `apps/extension/tests/e2e/utils/extension-id.ts`: observed-target ID resolution only.
- `apps/extension/tests/e2e/utils/extension-build.ts`: delegate to the shared launcher; no staging.
- `apps/extension/tests/e2e/utils/extension-paths.ts`: remove certification staging/key helpers.
- `apps/extension/tests/e2e/setup/build-extension.ts`: build only; never patch output manifests.
- Create `apps/extension/tests/e2e/package-health.spec.ts`: storage and app-ready package-health probe.

### Build Diagnosis and Integrity

- Create `apps/extension/scripts/isolate-package-assets.mjs`: diagnostic-only dependency-closed asset isolation.
- `apps/extension/scripts/report-bundle-size.mjs`: startup graph analyzer.
- Create `apps/extension/startup-budgets.json`: reviewed raw/gzip graph limits.
- `apps/extension/scripts/post-build-tasks.mjs`, `apps/extension/package.json`, `apps/extension/wxt.config.ts`: package-health/budget commands and warning gates.
- `apps/packages/ui/src/entries/shared/AppShell.tsx`: stable app-ready marker.
- `apps/packages/ui/src/hooks/useMediaNavigation.ts`, `apps/packages/ui/src/utils/storage-guard.ts`: canonical exports only.
- Affected `apps/packages/ui/src/**` broad `@/services/tldw` consumers: narrow direct imports.
- `apps/packages/ui/src/assets/tailwind-shared.css`: bundled relative fonts and system monospace fallback.

### Final UAT

- Create `Helper_Scripts/Testing-related/chatbooks_uat_sanitizer.py`: Python retained-output sanitizer using shared vectors.
- Create `Helper_Scripts/Testing-related/chatbooks_sanitizer_vectors.json`: shared Node/Python negative cases.
- `Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py`: explicit artifact/result inputs, quarantine, destination root, trap, aggregation.
- `Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py`: trap metadata and source/archive/destination hash evidence.
- Create `Helper_Scripts/Testing-related/chatbooks_certification_required_test_ids.json`: phase-qualified expected IDs.
- `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`: strict direct-artifact source-export phase.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts`: WebUI regression phase output.

## Stage 0: Preflight

- [ ] Confirm current `origin/dev` is an ancestor and record SHAs in TASK-12098.5:

```bash
cd "$(git rev-parse --show-toplevel)"
git fetch origin
git merge-base --is-ancestor origin/dev HEAD
git rev-parse origin/dev HEAD
git status --short --branch
```

Expected: ancestor check exits 0; the unrelated untracked Watchlists templates remain untouched.

- [ ] Confirm the normal production wrapper still exports `build/chrome-mv3` and no newer implementation already satisfies package health:

```bash
cd "$(git rev-parse --show-toplevel)"
sed -n '1,220p' apps/extension/scripts/build-with-profile.mjs
git log -1 --oneline origin/dev
```

## Task 1: Implement Deterministic Package Identity and Sanitized Evidence

**Files:**
- Create: `apps/extension/tests/e2e/utils/package-output.ts`
- Create: `apps/extension/tests/e2e/utils/package-output.test.ts`
- Create: `apps/extension/tests/e2e/utils/diagnostic-sanitizer.ts`
- Create: `apps/extension/tests/e2e/utils/diagnostic-sanitizer.test.ts`
- Create: `Helper_Scripts/Testing-related/chatbooks_sanitizer_vectors.json`

- [ ] **Step 1: Write failing fingerprint and symlink tests**

The test contract is:

```ts
type PackageFingerprint = {
  algorithm: "sha256"
  digest: string
  fileCount: number
  totalBytes: number
  files: Array<{ path: string; bytes: number; sha256: string }>
}
```

Assert sorted POSIX relative paths, regular files only, deterministic repeated digest, manifest-reference existence, and rejection of every symlink instead of following it.

- [ ] **Step 2: Verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run tests/e2e/utils/package-output.test.ts
```

- [ ] **Step 3: Implement package fingerprinting and result validation**

The result writer must refuse a destination inside the fingerprinted root. It records only `chrome-mv3`, fingerprint, counts, relative file names, timings, and sanitized diagnostics.

- [ ] **Step 4: Write sanitizer vector tests RED**

Shared vectors include API keys, bearer/cookie headers, absolute repository/home paths, browser command lines, archive names, query tokens, and raw child-process output. Expected output must retain useful phase/error labels while replacing sensitive fields.

- [ ] **Step 5: Run sanitizer tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run tests/e2e/utils/diagnostic-sanitizer.test.ts
```

Expected: FAIL because the shared vectors and sanitizer are not implemented.

- [ ] **Step 6: Implement the Node sanitizer and run GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run \
  tests/e2e/utils/package-output.test.ts \
  tests/e2e/utils/diagnostic-sanitizer.test.ts
```

- [ ] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/extension/tests/e2e/utils/package-output.ts apps/extension/tests/e2e/utils/package-output.test.ts apps/extension/tests/e2e/utils/diagnostic-sanitizer.ts apps/extension/tests/e2e/utils/diagnostic-sanitizer.test.ts Helper_Scripts/Testing-related/chatbooks_sanitizer_vectors.json
git commit -m "test(extension): fingerprint production package output"
```

## Task 2: Replace Certification Staging with One Direct Launcher

**Files:**
- Modify: `apps/extension/tests/e2e/utils/extension.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-id.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-build.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-paths.ts`
- Modify: `apps/extension/tests/e2e/setup/build-extension.ts`
- Modify: `apps/extension/tests/e2e/utils/extension.launch.test.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-build.test.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-paths.test.ts`
- Modify: `apps/extension/tests/e2e/setup/build-extension.test.ts`

- [ ] **Step 1: Write failing direct-launch argument tests**

Given `apps/extension/build/chrome-mv3`, assert the launcher:

```ts
expect(args).toContain(`--disable-extensions-except=${canonicalRealpath}`)
expect(args).toContain(`--load-extension=${canonicalRealpath}`)
expect(args.filter((arg) => arg.includes("--load-extension"))).toHaveLength(1)
expect(prepareExtensionLaunchPath).not.toHaveBeenCalled()
```

Also assert no manifest write, key injection, host-permission patch, implicit build, or copied launch path.

- [ ] **Step 2: Verify RED with current staging behavior**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run \
  tests/e2e/utils/extension.launch.test.ts \
  tests/e2e/utils/extension-build.test.ts \
  tests/e2e/utils/extension-paths.test.ts \
  tests/e2e/setup/build-extension.test.ts
```

- [ ] **Step 3: Implement the shared direct-output launcher**

Require `TLDW_EXTENSION_OUTPUT_DIR` for package health/certification. Resolve its realpath once, fingerprint it, pass it to both Chrome flags, and discover extension ID only from an observed service worker or extension page. Reuse runtime `grantHostPermission`; do not patch the manifest.

- [ ] **Step 4: Handle initially absent MV3 workers**

Inspect current targets, derive an observed extension ID if possible, open `options.html`, then wait for either that page or a service worker. Add separate launch, storage-sentinel, and app-ready timeouts. Absence before activation is not a skip.

- [ ] **Step 5: Remove certification use of staging helpers**

Delete or deprecate `prepareExtensionLaunchPath`, locale staging, `E2E_EXTENSION_MANIFEST_KEY`, ID fallbacks, and `applyTestHostPermissions` from the strict path. Non-certifying legacy tests may retain explicitly named compatibility helpers only if no strict command imports them.

- [ ] **Step 6: Run launcher tests GREEN and commit**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run \
  tests/e2e/utils/extension.launch.test.ts \
  tests/e2e/utils/extension-build.test.ts \
  tests/e2e/utils/extension-paths.test.ts \
  tests/e2e/setup/build-extension.test.ts
```

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/extension/tests/e2e/utils/extension.ts apps/extension/tests/e2e/utils/extension-id.ts apps/extension/tests/e2e/utils/extension-build.ts apps/extension/tests/e2e/utils/extension-paths.ts apps/extension/tests/e2e/setup/build-extension.ts apps/extension/tests/e2e/utils/extension.launch.test.ts apps/extension/tests/e2e/utils/extension-build.test.ts apps/extension/tests/e2e/utils/extension-paths.test.ts apps/extension/tests/e2e/setup/build-extension.test.ts
git commit -m "fix(extension): launch canonical production output directly"
```

## Task 3: Add the Package-Health Probe

**Files:**
- Create: `apps/extension/tests/e2e/package-health.spec.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-launch-health.spec.ts`
- Modify: `apps/packages/ui/src/entries/shared/AppShell.tsx`
- Modify: `apps/extension/package.json`
- Modify: `apps/extension/scripts/post-build-tasks.mjs`

- [ ] **Step 1: Write failing app-ready and storage sentinel tests**

The probe must open a known extension page, wait for `data-tldw-app-ready="true"`, write a random value to extension local storage, read the same value, and delete it. It must recompute the package fingerprint after context close.

- [ ] **Step 2: Run package-health tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx playwright test \
  tests/e2e/package-health.spec.ts \
  tests/e2e/utils/extension-launch-health.spec.ts \
  --project=chromium-extension --reporter=line
```

Expected: FAIL because the strict package-health probe, storage sentinel, and app-ready marker do not exist.

- [ ] **Step 3: Add a stable app-ready marker**

Set the marker only after the application shell mounts. Do not couple it to backend authentication or Chatbooks readiness.

- [ ] **Step 4: Add a strict package-health command**

Add `package:health` that requires:

```text
TLDW_E2E_SKIP_EXTENSION_BUILD=1
TLDW_EXTENSION_OUTPUT_DIR=<canonical build/chrome-mv3>
TLDW_PACKAGE_HEALTH_RESULT=<path outside output>
```

The command uses Playwright JSON output and the sanitizer. Missing inputs fail preflight, never skip.

- [ ] **Step 5: Build and run package health outside the sandbox**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bun run compile
bun run build:chrome:prod
```

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
TLDW_E2E_SKIP_EXTENSION_BUILD=1 \
TLDW_EXTENSION_OUTPUT_DIR="$PWD/build/chrome-mv3" \
TLDW_PACKAGE_HEALTH_RESULT=/private/tmp/chatbooks-package-health.json \
bun run package:health
```

Expected: direct launch, storage sentinel, app-ready marker, and pre/post fingerprint equality pass. If launch still stalls, proceed to Task 4; do not weaken this gate.

- [ ] **Step 6: Commit package-health infrastructure**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/extension/tests/e2e/package-health.spec.ts apps/extension/tests/e2e/utils/extension-launch-health.spec.ts apps/packages/ui/src/entries/shared/AppShell.tsx apps/extension/package.json apps/extension/scripts/post-build-tasks.mjs
git commit -m "test(extension): add immutable package health gate"
```

## Task 4: Isolate and Fix the Production Asset-Tree Stall

**Files:**
- Create: `apps/extension/scripts/isolate-package-assets.mjs`
- Create: `apps/extension/scripts/isolate-package-assets.test.mjs`
- Modify after evidence: the WXT/Vite/import/locale/asset source proven responsible
- Update: `backlog/tasks/task-12098.5 - Make-packaged-extension-loadable-and-enforce-startup-budgets.md`

- [ ] **Step 1: Write diagnostic isolator tests**

Test dependency closure, top-level-class grouping, batch bisection, manifest validity, distinct `diagnostic` result type, and refusal to emit a certifying result.

- [ ] **Step 2: Run RED, implement, then run GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/isolate-package-assets.test.mjs
```

- [ ] **Step 3: Run the isolator against the failed canonical build outside the sandbox**

Record each probe's asset classes, file count, total bytes, fingerprint, elapsed launch phase, and sanitized stderr. Start by top-level classes, then bisect dependency-closed batches.

- [ ] **Step 4: Identify a minimal failing set or threshold**

Do not guess. After three unsuccessful isolation strategies, stop and record attempts, errors, and two alternative hypotheses in TASK-12098.5 before requesting guidance.

- [ ] **Step 5: Fix the upstream source and prove the canonical artifact passes**

Modify only the source responsible for emitting or referencing the failing set. Rebuild normally and rerun `package:health` against `build/chrome-mv3`. Diagnostic copies are not acceptance evidence.

- [ ] **Step 6: Commit evidence and source fix**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/extension/scripts/isolate-package-assets.mjs apps/extension/scripts/isolate-package-assets.test.mjs apps/extension backlog/tasks/task-12098.5\ -\ Make-packaged-extension-loadable-and-enforce-startup-budgets.md
git commit -m "fix(extension): remove package launch blocker"
```

## Task 5: Remove Build Warnings and Enforce Startup Graph Budgets

**Files:**
- Modify: `apps/packages/ui/src/hooks/useMediaNavigation.ts`
- Modify: `apps/packages/ui/src/utils/storage-guard.ts`
- Modify: affected broad `@/services/tldw` consumers under `apps/packages/ui/src`
- Modify: `apps/packages/ui/src/assets/tailwind-shared.css`
- Modify: `apps/extension/scripts/report-bundle-size.mjs`
- Create: `apps/extension/startup-budgets.json`
- Modify: `apps/extension/wxt.config.ts`
- Modify: `apps/extension/package.json`
- Create/modify focused build-integrity tests under `apps/extension/scripts/__tests__`

- [ ] **Step 1: Add failing duplicate-export and narrow-import checks**

Assert `MediaNavigationFormat` is canonical in `media-navigation-scope.ts`, `estimateStorageCost` in `storage-budget.ts`, and startup-path files do not import the broad `@/services/tldw` barrel.

- [ ] **Step 2: Add failing font-resolution tests**

Build CSS fixtures and assert every emitted local URL exists, no root-absolute `/fonts/` remains, JetBrains faces are absent, and the system monospace fallback remains.

- [ ] **Step 3: Run build-integrity tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/__tests__/build-integrity.test.mjs scripts/__tests__/font-resolution.test.mjs
```

Expected: FAIL on duplicate exports, broad startup imports, or unresolved font URLs.

- [ ] **Step 4: Implement source fixes and run focused tests**

Replace duplicate exports and broad imports with narrow modules. Use build-resolved relative URLs for existing Inter, Space Grotesk, and Arimo files.

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/__tests__/build-integrity.test.mjs scripts/__tests__/font-resolution.test.mjs
```

Expected: PASS with one canonical export per symbol, no broad startup-path barrel imports, and every emitted local font URL resolving to a production asset.

- [ ] **Step 5: Write startup graph tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/__tests__/startup-graph-budget.test.mjs
```

Expected: FAIL because startup graph traversal and checked-in budgets do not exist.

- [ ] **Step 6: Implement startup graph analysis GREEN**

Parse manifest and HTML module roots, follow static imports recursively, and report raw/gzip bytes without double-counting shared files. Roots: background, each content script, sidepanel, options.

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/__tests__/startup-graph-budget.test.mjs
```

Expected: PASS with dependency-closed graph accounting for every startup-critical root and deterministic budget diagnostics.

- [ ] **Step 7: Record reviewed budgets**

After package health passes, record the measured baseline rounded up with no more than 10% headroom. Optional lazy chunks are reported separately. A budget increase requires changing the checked-in file and naming the dependency/reason.

- [ ] **Step 8: Run production gates**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bun run compile
bun run build:chrome:prod
bun run perf:bundle
```

Expected: zero duplicate-export, circular service-import, unresolved-font, or startup-budget failures. Generic lazy warnings remain visible through the report rather than being globally suppressed.

- [ ] **Step 9: Commit and complete TASK-12098.5 evidence**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/packages/ui/src apps/extension/scripts/report-bundle-size.mjs apps/extension/startup-budgets.json apps/extension/wxt.config.ts apps/extension/package.json apps/extension/scripts/__tests__ backlog/tasks/task-12098.5\ -\ Make-packaged-extension-loadable-and-enforce-startup-budgets.md
git commit -m "perf(extension): enforce startup graph budgets"
```

## Task 6: Harden the Browser UAT for Archive-Only Restore

**Files:**
- Create: `Helper_Scripts/Testing-related/chatbooks_uat_sanitizer.py`
- Modify: `Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py`
- Modify: `Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py`
- Modify: `tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py`

- [ ] **Step 1: Write Python sanitizer tests using shared vectors**

Assert byte-for-byte parity with the Node sanitizer for every vector. Every retained write path, including child stdout/stderr, must call the sanitizer first.

- [ ] **Step 2: Write failing isolation/trap tests**

Assert the harness:

1. hashes source media before export;
2. extracts and hashes archive media;
3. stops the source server and renames/quarantines the entire source root;
4. starts destination with a fresh independent root;
5. points fixture external metadata at a local trap endpoint;
6. fails on any trap hit; and
7. verifies destination bytes and vectors.

- [ ] **Step 3: Run sanitizer and isolation tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py \
  tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py -v
```

Expected: FAIL on missing quarantine, trap, result sanitization, or explicit artifact inputs.

- [ ] **Step 4: Implement explicit artifact/result inputs**

Add required extension arguments: `--extension-output`, `--package-health-result`, and `--result`. Verify package fingerprint before browser launch and after all phases. No implicit build or extension path discovery in certification mode.

- [ ] **Step 5: Implement source quarantine, destination root, and trap**

The destination process must not receive source-root paths. Preserve only sanitized labels in retained JSON. Always restore quarantined test data during harness cleanup, even after failure.

Retain and reassert the existing full-account checks for characters, transcripts, chunks, profile/settings, stored media, vectors, and every fixture record. Hardening the harness must not replace these with aggregate counts.

- [ ] **Step 6: Run focused pytest GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py \
  tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py -v
```

- [ ] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add Helper_Scripts/Testing-related/chatbooks_uat_sanitizer.py Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py
git commit -m "test(chatbooks): isolate full-account restore source"
```

## Task 7: Enforce Phase-Qualified Zero-Skip Certification Reports

**Files:**
- Create: `Helper_Scripts/Testing-related/chatbooks_certification_required_test_ids.json`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_final_certification_report.py`
- Create: `apps/extension/scripts/assert-playwright-no-skips.test.mjs`
- Modify: `Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py`
- Modify: `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts`
- Reuse/modify: `apps/extension/scripts/assert-playwright-no-skips.mjs`

- [ ] **Step 1: Define the checked-in required ID manifest**

Use stable phase-qualified IDs, not display titles. Include source-export, destination-import, archive verification, extension activation/storage, notification standard/restricted/reauth/grant, and WebUI regression phases.

- [ ] **Step 2: Write failing report-set tests**

Test exact equality and failures for missing, renamed, duplicate, extra, skipped, interrupted, or non-passing IDs.

- [ ] **Step 3: Run report tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_final_certification_report.py -v
```

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/assert-playwright-no-skips.test.mjs
```

Expected: FAIL because exact phase-qualified manifest comparison is not implemented.

- [ ] **Step 4: Emit machine-readable phase reports**

Certification mode fails preflight when inputs are absent. Remove `test.skip(!enabled)` from the strict path. Aggregate all phase reports into one sanitized result and compare it to the required manifest.

- [ ] **Step 5: Run report tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_final_certification_report.py -v
```

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bunx vitest run scripts/assert-playwright-no-skips.test.mjs
```

- [ ] **Step 6: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add Helper_Scripts/Testing-related/chatbooks_certification_required_test_ids.json Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py tldw_Server_API/tests/Chatbooks/test_chatbooks_final_certification_report.py apps/extension/tests/e2e/chatbooks-export-download.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts apps/extension/scripts/assert-playwright-no-skips.mjs apps/extension/scripts/assert-playwright-no-skips.test.mjs
git commit -m "test(chatbooks): require complete certification report"
```

## Task 8: Run Final Exact-Artifact Certification

**Prerequisite:** `TASK-12098.4` and `TASK-12098.5` complete with recorded verification.

**Files:**
- Update: `backlog/tasks/task-12098.3 - P2-Chatbooks-backup-import-acceptance-coverage.md`
- Update: `Docs/Reviews/CHATBOOKS_POST_MERGE_UAT_UX_REVIEW_2026_07_09.md`

- [ ] **Step 1: Build and fingerprint the canonical production artifact outside the sandbox**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/extension
bun run compile
bun run build:chrome:prod
TLDW_E2E_SKIP_EXTENSION_BUILD=1 \
TLDW_EXTENSION_OUTPUT_DIR="$PWD/build/chrome-mv3" \
TLDW_PACKAGE_HEALTH_RESULT=/private/tmp/chatbooks-package-health.json \
bun run package:health
```

- [ ] **Step 2: Run extension certification outside the sandbox**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py run \
  --surface extension \
  --root /private/tmp/chatbooks-full-account-browser-uat/extension \
  --api-port 18011 \
  --extension-output "$PWD/apps/extension/build/chrome-mv3" \
  --package-health-result /private/tmp/chatbooks-package-health.json \
  --result /private/tmp/chatbooks-full-account-browser-uat/extension/certification.json
```

Expected: required-ID set equality, zero skips, unchanged package fingerprint, three-way media SHA equality, vectors restored, profile/settings restored, no external trap hits, absent-worker activation, and notification standard/restricted/reauth/grant scenarios passed.

- [ ] **Step 3: Rerun WebUI regression UAT outside the sandbox**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py run \
  --surface webui \
  --root /private/tmp/chatbooks-full-account-browser-uat/webui \
  --api-port 18001 \
  --web-port 18269 \
  --result /private/tmp/chatbooks-full-account-browser-uat/webui/certification.json
```

- [ ] **Step 4: Run final quality gates**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bun run typecheck
bun run build:prod
```

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m bandit -r Helper_Scripts/Testing-related -f json -o /tmp/bandit_chatbooks_uat.json
```

```bash
cd "$(git rev-parse --show-toplevel)"
git diff --check
```

- [ ] **Step 5: Record exact evidence and close tasks**

Record commit SHA, package fingerprint, sanitized report paths, test counts, media SHA, vector IDs, trap count, and Bandit result. Complete TASK-12098.5 before TASK-12098.3. Complete TASK-12098.3 only when AC 17-20 and Definition of Done are checked.

- [ ] **Step 6: Commit final evidence**

```bash
cd "$(git rev-parse --show-toplevel)"
git add backlog/tasks/task-12098.3\ -\ P2-Chatbooks-backup-import-acceptance-coverage.md backlog/tasks/task-12098.5\ -\ Make-packaged-extension-loadable-and-enforce-startup-budgets.md Docs/Reviews/CHATBOOKS_POST_MERGE_UAT_UX_REVIEW_2026_07_09.md
git commit -m "test(chatbooks): certify production extension round trip"
```

## Final Verification Checklist

- [ ] `build/chrome-mv3` is the explicit fingerprinted and loaded directory.
- [ ] No strict path copies, stages, patches, or rewrites the artifact.
- [ ] Pre/post package-health and pre/post final-UAT fingerprints match.
- [ ] Initial worker absence activates through a real extension page.
- [ ] Storage sentinel and app-ready marker pass.
- [ ] Asset-tree root cause is documented and fixed at its source.
- [ ] Duplicate exports, circular service imports, and font resolution warnings are removed.
- [ ] Startup graph budgets pass and lazy chunks remain visible in reports.
- [ ] Source root is quarantined and destination uses a fresh independent root.
- [ ] External trap receives zero requests.
- [ ] Source, archive, and restored media SHA-256 values match; vectors and account state restore.
- [ ] Aggregated phase report exactly matches required IDs with zero skips or omissions.
- [ ] Notification UAT requirements from TASK-12098.4 pass in the extension.
- [ ] WebUI regression UAT, typecheck/build, pytest/Vitest/Playwright, Bandit, and diff checks pass.
