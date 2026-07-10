# Chatbooks Post-Merge UAT Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full-account Chatbooks backup and restore complete successfully through the WebUI and browser extension, report truthful job state, and pass the acceptance paths that failed after PR #2699 merged.

**Architecture:** Preserve the existing Chatbooks endpoint, archive service, Jobs queue, WebUI component, and extension route. Remove the stale legacy-worker restriction that contradicts the archive restore service, normalize terminal job metadata at the backend with a defensive UI fallback for historical jobs, then certify the same workflow through focused backend, WebUI, and packaged-extension tests.

**Tech Stack:** FastAPI, Python async workers, SQLite job records, React/TypeScript, Ant Design, Vitest, Playwright, pytest, Bandit.

---

## Baseline And Scope

Source under test: `origin/dev` at `440478b6cb`.

Confirmed post-merge UAT findings:

1. P0: a full-account archive exported by the WebUI fails when imported through the WebUI because `core_jobs_worker.py` rejects the archive-default `import_media=true` and `import_embeddings=true` flags before the working restore service runs.
2. P1: completed export jobs can persist and render `0%` with zero processed/total items.
3. P2: the packaged browser-extension flow is not certified because the E2E harness can fail to discover an MV3 service worker.
4. P2: three stale integration/docs expectations and one incomplete minimal fixture make the broader Chatbooks verification suite non-green.
5. P1: Backup all still requires users to invent a name and description instead of supporting a one-action safety backup.
6. P1: the archive preview understates full-account restore impact, while an enabled Include all control simultaneously reports `Selected: 0`.
7. P1: essential dark-theme upload, progress, and empty-state text uses raw Ant Design black foregrounds with measured contrast near 1.1:1; multiple switches and select controls have no accessible name.
8. P1: failed imports expose internal multipart flags without a recovery action, and import jobs are identified by UUID instead of archive name.
9. P1: the Jobs tab duplicates the side Job tracker, compresses a ten-column table, uses inconsistent cleanup terminology, and performs destructive cleanup without confirmation.
10. P0: full-account manifests count account profile and settings records, but the importer currently classifies those categories as having no serialized restore payload and skips them with a warning. A full-account archive must carry and restore the actual account-owned profile/settings state permitted by the sensitive-data policy, not inventory placeholders.

This plan does not redesign the Chatbook format, add new account-data categories, or reimplement the Jobs subsystem. Full export continues to include bundled stored media artifacts and all other account data defined by the approved PRD.

## File Map

- `tldw_Server_API/app/services/core_jobs_worker.py`: legacy/core Jobs execution path used by live async Chatbook export and import.
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`: archive import/export service and persisted Chatbook job metadata.
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`: source-format defaults and API job response contract.
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py`: new regression coverage for the live core Jobs adapter.
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`: archive service restore coverage.
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`: source-specific import-default parity coverage.
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py`: export metadata and full-account archive contract coverage.
- `tldw_Server_API/tests/integration/test_chatbook_integration.py`: legacy integration expectations and minimal database fixture behavior.
- `tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py`: documentation/runtime contract checks.
- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`: dropzone styling and defensive terminal-progress rendering.
- `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`: WebUI backup/import/status regression coverage.
- `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.accessibility.test.tsx`: new labels, contrast-token, keyboard, and responsive common-path coverage.
- `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx`: new job layout, naming, recovery, timestamp, and destructive-action coverage.
- `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`: packaged-extension full-account export/import/download acceptance path.
- `apps/extension/tests/e2e/utils/extension.ts`: MV3 extension-id and configuration fallback when no service-worker target appears.
- `apps/extension/tests/e2e/utils/extension.launch.test.ts`: deterministic harness coverage for the no-service-worker fallback.
- `Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py`: new deterministic source-archive, clean-destination, and restore-verification helper for browser UAT.
- `tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py`: new two-user API round trip that verifies media bytes and embedding/vector records in the destination.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts`: new real-server WebUI import path using the media-bearing fixture and clean destination.
- `Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md`: append-only post-merge UAT result and corrected heuristic findings.
- `Docs/Reviews/CHATBOOKS_POST_MERGE_UAT_UX_REVIEW_2026_07_09.md`: post-merge screen/workflow heuristic review and UX acceptance evidence.
- `backlog/tasks/task-12098.1 - P0-Chatbooks-backup-restore-correctness-remediation.md`: reopen contradicted P0 acceptance items and record evidence.
- `backlog/tasks/task-12098.2 - P1-Chatbooks-backup-import-UX-clarity-remediation.md`: record UX remediation scope and evidence.
- `backlog/tasks/task-12098.3 - P2-Chatbooks-backup-import-acceptance-coverage.md`: record acceptance automation and final matrix.

## Task 1: Restore Full Archives Through The Live Async Worker

**Files:**
- Modify: `tldw_Server_API/app/services/core_jobs_worker.py:338`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`

- [x] **Step 1: Write the failing live-worker regression**

Create a focused worker test that enqueues an `action=import`, `source_format=chatbook` payload with `import_media=true` and `import_embeddings=true`. Use a temporary archive and fakes for the Jobs manager and Chatbook service. Assert that `_import_chatbook_sync(...)` is invoked with both flags true and that the public import job finishes `completed`.

- [x] **Step 2: Prove the current failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py -v
```

Expected: FAIL because the legacy worker returns `Media/embedding imports are not supported yet` without calling the archive service.

- [x] **Step 3: Remove the obsolete archive restriction**

In `core_jobs_worker.py`, retain the unsupported conflict-resolution validation, but allow Chatbook archive payloads to reach `_import_chatbook_sync` with the effective media and embedding flags. Do not weaken the endpoint guard that rejects archive restore flags for `openwebui_json` and `openwebui_db` sources.

- [x] **Step 4: Add a bundled-media async restore case**

Extend the regression with an archive containing an inventory-declared bundled media artifact. Assert that the restored bytes land under user-owned storage and that the import job records the imported media/inventory result rather than silently skipping it.

- [x] **Step 5: Run focused restore tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py -v
```

Expected: PASS, including archive defaults true and OpenWebUI archive options rejected.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/services/core_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py
git commit -m "fix: restore full chatbook archives in core worker"
```

## Task 2: Make Terminal Job Status Truthful

**Files:**
- Modify: `tldw_Server_API/app/services/core_jobs_worker.py:251`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py:2205`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx:1018`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py`
- Test: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`

- [x] **Step 1: Write failing backend and UI tests**

Assert that a successfully completed async export persists `progress_percentage=100`, `processed_items=total_items`, archive size, and redacted manifest metadata. Assert separately that the UI renders a historical job with `status=completed` and stale `progress_percentage=0` as complete rather than showing `0%` beside a completed badge.

Also assert that a verified archive returns `post_write_verification=true` and that API timestamps carry an explicit timezone. A UTC timestamp must never be rendered as a future local wall-clock time.

- [x] **Step 2: Prove both failures**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py -k "progress or completed" -v
cd apps/packages/ui
bunx vitest run src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx
```

Expected: backend FAILS because the live worker leaves progress fields at their defaults; UI FAILS because `computeProgress` returns the stale numeric zero before considering terminal status.

- [x] **Step 3: Normalize completion at persistence time**

When export/import succeeds, set progress to 100. Populate total/processed counts from the redacted archive/import result metadata when available. If a completed archive has no countable records, use zero counts with 100% completion; never fabricate content counts.

Persist the archive verification result built by `ChatbookService` in the live core-worker job record. Serialize timestamps with an offset or `Z`; use the browser locale only after timezone semantics are unambiguous.

- [x] **Step 4: Add a historical-job UI fallback**

Update `computeProgress` so terminal `completed` status wins over stale values and renders 100. Failed, cancelled, and expired jobs must retain their last meaningful percentage or render no bar.

- [x] **Step 5: Re-run focused tests and API serialization checks**

Expected: all focused tests pass and `GET /chatbooks/export/jobs` returns terminal status consistent with progress.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/services/core_jobs_worker.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/tests/Chatbooks/test_chatbooks_core_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx
git commit -m "fix: report completed chatbook jobs accurately"
```

## Task 3: Make Backup And Restore Simple, Legible, And Accessible

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx:2360`
- Test: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.accessibility.test.tsx`

- [ ] **Step 1: Add failing accessibility assertions**

Write tests for the complete common path:

- Backup all can submit without manual metadata; generated defaults remain editable.
- Import preview shows the archive's account inventory categories, warnings, sensitive-category summary, and verification state, not only named content items.
- Include all reports `All in archive` or the known archive count, never `Selected: 0` while enabled.
- Conflict resolution has a visible label as well as an accessible name.
- Export mode, Tags, Categories, Prefix imported, Run in background, and every Include all switch have programmatic labels.
- Dropzone title/hint, progress text, and empty-state text use shared semantic foreground tokens in dark theme.
- The 390px viewport has no horizontal overflow, clipped labels, or sub-44px primary touch target.

- [ ] **Step 2: Run the UI tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.accessibility.test.tsx
```

Expected: FAIL on required backup metadata, incomplete archive-impact preview, contradictory Include all count, missing programmatic labels, and dark-theme token violations.

- [ ] **Step 3: Apply shared design tokens**

For full-account mode, generate a useful localized backup name and plain description when fields are empty. Treat those fields as optional customization. Preserve explicit values when the user enters them.

After preview, show a compact `What will be restored` summary from the manifest/account inventory. Keep detailed category warnings behind one `Review N warnings` disclosure so the primary decision remains visible.

Move conflict resolution, prefixing, and background execution under an `Advanced options` disclosure for the normal archive path, using the existing safe defaults. Selective export and OpenWebUI-specific controls remain available without competing with Backup all.

- [ ] **Step 4: Verify keyboard, focus, and contrast behavior**

Apply shared text tokens to the upload, progress, and empty-state descendants that currently inherit Ant Design black. Associate visible labels with all switches and selects using `aria-label`, `aria-labelledby`, or native label structure. Keep the upload area a keyboard-operable button.

Use the live page in both themes at desktop and 390px mobile widths. Require WCAG AA text contrast, visible focus, no horizontal overflow, and stable layout at 200% zoom.

- [ ] **Step 5: Re-run the complete Task 3 UI suite**

Run the same three-file Vitest command from Step 2.

Expected: PASS with generated Backup all metadata, complete restore-impact preview, truthful Include all labels, named controls, and semantic dark-theme foreground tokens.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.accessibility.test.tsx
git commit -m "fix: streamline accessible chatbook backup restore"
```

## Task 4: Make Jobs Scannable, Trustworthy, And Recoverable

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx:3110`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py:112`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py`
- Create: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx`

- [ ] **Step 1: Write failing job-center tests**

Assert that:

- the Jobs tab uses the full content width and does not repeat the side Job tracker;
- export rows show name, status, truthful progress, size, verification, local/explicit-zone timestamp, warnings, and actions without hiding actions behind the rail;
- import rows use archive filename or Chatbook name as the primary label and keep UUID as secondary copyable metadata;
- a failed import shows plain-language cause plus `Review import` or `Choose archive again`, not internal form-field instructions;
- `Remove`, `Remove finished`, and archive cleanup confirm their scope before deleting files or job history;
- cleanup terminology distinguishes expired archive files from finished job-history removal.

- [ ] **Step 2: Prove the current failures**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py \
  -k "job or metadata" -v

cd apps/packages/ui
bunx vitest run src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx
```

Expected: backend FAILS because import-job responses do not expose the redacted human-readable archive identity; UI FAILS because the side rail is rendered on Jobs, rows lead with UUID, errors expose multipart flags, and destructive actions execute immediately.

- [ ] **Step 3: Give the Jobs tab full width**

Render the compact side tracker only on Export and Import. On Jobs, use the entire page width for the tables. Preserve responsive horizontal overflow as a fallback, but keep the primary status and action columns visible at common desktop widths.

- [ ] **Step 4: Add human-readable identity and recovery**

Expose redacted source filename/Chatbook name in import job metadata and use it as the row label. Translate known backend failures into user language while preserving copyable technical detail under disclosure. Route recovery back to Import and preserve the in-memory preview when still available.

- [ ] **Step 5: Guard destructive actions**

Use the existing confirmation pattern to name exactly what is removed: server archive files, one job record, or all finished job records. Keep Download as the primary action for completed exports.

- [ ] **Step 6: Verify desktop, mobile, and keyboard behavior**

At 1280px, all primary columns and actions must be discoverable without the duplicate rail. At 390px and 200% zoom, tables may scroll horizontally but status, row identity, and an action menu must remain reachable. Confirm focus moves to recovery feedback after a failed action.

Re-run both Step 2 commands.

Expected: backend and UI tests PASS before commit.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py
git commit -m "fix: improve chatbook job trust and recovery"
```

## Task 5: Build A Media-Bearing Clean-Destination UAT Fixture

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Create: `Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py`
- Create: `tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`

- [ ] **Step 1: Write failing fixture and two-user round-trip tests**

The fixture test must require `prepare`, `reset-destination`, and `verify` behavior. The E2E test must create a source user with non-secret account profile/settings values, a character, media record, transcript/chunks, a stored media artifact with known SHA-256, and embedding/vector records; export full account; import into a distinct clean destination user; then compare destination profile/settings state, content records, artifact hash, and vector identifiers.

- [ ] **Step 2: Prove the fixture does not exist**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py \
  tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py -v
```

Expected: FAIL because the fixture helper and media-bearing two-user round trip are not implemented.

- [ ] **Step 3: Implement deterministic prepare/reset/verify behavior**

`prepare` writes the full-account archive plus `expected.json`; `reset-destination` initializes an empty target without copying source state; `verify` reads only the destination and fails unless all expected categories, account profile/settings values, media bytes, and embedding/vector identifiers are present. Never satisfy verification from manifest counts or import-job metadata alone.

If the red round trip confirms that account profile/settings are inventory-only placeholders, add a versioned archive payload and restore handler using the existing user/profile/settings abstractions. Apply the approved sensitive-data policy: include required account-owned state, redact preview/log output, and represent intentionally excluded secrets with explicit policy metadata rather than fabricated counts.

- [ ] **Step 4: Run fixture and backend round-trip tests**

Run the Step 2 command again.

Expected: PASS with two distinct users, no same-account conflict skips, restored account profile/settings values, matching media SHA-256, and restored vector identifiers.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py
git commit -m "test: add full account chatbook restore fixture"
```

## Task 6: Certify WebUI And Packaged Extension Workflows

**Files:**
- Modify: `apps/extension/tests/e2e/utils/extension.ts:278`
- Modify: `apps/extension/tests/e2e/chatbooks-export-download.spec.ts`
- Test: `apps/extension/tests/e2e/utils/extension.launch.test.ts`
- Create: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts`
- Create: `Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py`
- Use fixture: `Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py`

- [ ] **Step 1: Write failing extension-launch and browser-orchestrator tests**

Model a persistent extension context with no immediately visible service-worker target. Assert that extension ID resolution and seeded configuration still succeed through the existing extension-page fallback.

Test the browser UAT orchestrator in dry-run/fake-process mode. It must enforce this order independently for WebUI and extension: seed media-bearing source root; start source services; export and capture the browser-downloaded archive path; stop source services; initialize a distinct clean destination root; start destination services; import that exact downloaded archive; stop services; verify destination artifact hash and vector identifiers. The runner must reject a fixture archive substituted for the browser download.

- [ ] **Step 2: Prove the current harness gap**

Run:

```bash
cd apps/extension
bunx vitest run tests/e2e/utils/extension.launch.test.ts

cd ../..
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py -v
```

Expected: extension test FAILS if the launcher requires a service worker before it can seed configuration or resolve the options URL; backend test FAILS because the two-phase browser UAT orchestrator does not exist.

- [ ] **Step 3: Make service-worker discovery optional for page-based flows**

Keep service-worker seeding when available. When absent, resolve the extension ID, open the options page, seed `chrome.storage` from that extension origin, reload once, and verify the sentinel before returning. Do not skip the Chatbooks test solely because an idle MV3 worker target is absent.

- [ ] **Step 4: Add two-phase WebUI and packaged-extension acceptance**

Add an export phase and import phase to the WebUI real-server spec and packaged-extension spec. The export phase connects to the seeded source root, clicks **Backup all**, waits for completion, downloads the archive, and writes the actual download path to the runner result. The import phase starts only after the runner has stopped source services and started a clean destination; it uploads the exact download path, verifies the full account-impact preview, starts import, and waits for a completed job whose metadata reports imported media and embedding categories.

Implement `chatbooks_full_account_browser_uat.py` to own process lifecycle, ports, temporary roots, archive handoff, and cleanup. After each surface's import phase, call the fixture helper's `verify` logic against that surface's destination root. Compare restored media artifact SHA-256 and embedding/vector identifiers to `expected.json`. A completed job without matching destination bytes/vectors is a failure.

- [ ] **Step 5: Run exact browser-export-to-clean-destination-import checks**

Run WebUI end to end:

```bash
source .venv/bin/activate
python Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py run \
  --surface webui \
  --root /tmp/chatbooks-full-account-browser-uat/webui \
  --api-port 18001 \
  --web-port 18269
```

Run packaged extension end to end with a separate clean destination:

```bash
python Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py run \
  --surface extension \
  --root /tmp/chatbooks-full-account-browser-uat/extension \
  --api-port 18011
```

Expected: both commands PASS. Each command must report its browser-downloaded archive path, a distinct source/destination root pair, identical restored media bytes, and restored embedding/vector records. A service-worker skip, fixture-archive substitution, or same-account import fails the runner.

Re-run the Step 2 unit commands before commit.

- [ ] **Step 6: Commit**

```bash
git add apps/extension/tests/e2e/utils/extension.ts apps/extension/tests/e2e/utils/extension.launch.test.ts apps/extension/tests/e2e/chatbooks-export-download.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks-full-account-roundtrip.spec.ts Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py
git commit -m "test: certify browser chatbook backup restore"
```

## Task 7: Repair Stale Integration And Documentation Contracts

**Files:**
- Modify: `tldw_Server_API/tests/integration/test_chatbook_integration.py`
- Modify: `tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py`
- Modify: `Docs/API-related/Chatbook_API_Documentation.md`
- Modify: `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- Modify: published mirrors under `Docs/Published/`

- [ ] **Step 1: Make legacy export tests explicit about selection mode**

Change the old empty-export test to either assert the new full-account default or pass an explicit zero-item allowlist and assert validation. Do not retain the obsolete assumption that omitted/empty content types mean an empty backup.

- [ ] **Step 2: Repair the minimal multi-user fixture**

Ensure the integration fixture exposes all account-inventory tables required by full-account export, including `generated_documents`, or uses the normal migration fixture. Keep the test focused on user isolation rather than accidental schema incompleteness.

- [ ] **Step 3: Align docs tests with archive restore semantics**

Document that Chatbook archives restore bundled media artifacts and embeddings by default, while OpenWebUI JSON/DB sources do not use those archive options. Remove assertions that globally label media/embedding import unsupported.

- [ ] **Step 4: Run the previously failing tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/integration/test_chatbook_integration.py::TestErrorScenarios::test_export_with_database_error \
  tldw_Server_API/tests/integration/test_chatbook_integration.py::TestMultiUserScenarios::test_user_isolation_during_export \
  tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py::test_chatbook_import_docs_match_multipart_contract -v
```

Expected: PASS with runtime and docs describing the same source-specific behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/integration/test_chatbook_integration.py tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py Docs/API-related/Chatbook_API_Documentation.md Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md Docs/Published
git commit -m "test: align chatbook integration contracts"
```

## Task 8: Final Verification And UAT Closeout

**Files:**
- Modify: `Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md`
- Modify: `backlog/tasks/task-12098.1 - P0-Chatbooks-backup-restore-correctness-remediation.md`
- Modify: `backlog/tasks/task-12098.2 - P1-Chatbooks-backup-import-UX-clarity-remediation.md`
- Modify: `backlog/tasks/task-12098.3 - P2-Chatbooks-backup-import-acceptance-coverage.md`

- [ ] **Step 1: Prepare the isolated full-account UAT fixture**

Use the Task 5 helper. `prepare` creates a source account containing a character, account setting, media record, transcript/chunks, one stored media artifact with known SHA-256, and embedding/vector records, then writes a full-account archive and `expected.json`. `reset-destination` creates an empty destination root without copying source databases.

Run:

```bash
source .venv/bin/activate
python Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py prepare \
  --root /tmp/chatbooks-full-account-uat
python Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py reset-destination \
  --root /tmp/chatbooks-full-account-uat
```

Expected: `/tmp/chatbooks-full-account-uat/source/full-account.chatbook` and `expected.json` exist; destination contains initialized empty stores only.

- [ ] **Step 2: Run exact backend suites and two-user restore verification**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Chatbooks \
  tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py \
  tldw_Server_API/tests/integration/test_chatbook_integration.py \
  tldw_Server_API/tests/e2e/test_chatbooks_roundtrip.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  tldw_Server_API/tests/e2e/test_chatbooks_multi_user_roundtrip.py \
  tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py \
  tldw_Server_API/tests/server_e2e_tests/test_chatbooks_roundtrip_workflow.py \
  tldw_Server_API/tests/server_e2e_tests/test_chatbooks_api_workflow.py \
  tldw_Server_API/tests/frontend_e2e/test_chatbooks_workflow.py
```

Expected: PASS. Environment-dependent tests may skip only with their documented prerequisite missing. `test_chatbooks_full_account_media_roundtrip.py` must not skip and must verify a clean destination user's artifact hash and embedding/vector records.

- [ ] **Step 3: Run exact WebUI unit suites**

Run unit/component tests:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.backup-all.test.tsx \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.accessibility.test.tsx \
  src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx \
  src/components/Option/Settings/__tests__/chatbooks.test.tsx \
  src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

Expected: all listed component and client tests PASS.

- [ ] **Step 4: Run the exact WebUI-exported archive round trip**

```bash
source .venv/bin/activate
python Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py run \
  --surface webui \
  --root /tmp/chatbooks-full-account-browser-uat/webui \
  --api-port 18001 \
  --web-port 18269
```

Expected: runner reports the actual WebUI-downloaded archive path, distinct source/destination roots, completed import, matching destination media SHA-256, and restored embedding/vector identifiers.

- [ ] **Step 5: Run the exact extension-exported archive round trip**

```bash
source .venv/bin/activate
python Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py run \
  --surface extension \
  --root /tmp/chatbooks-full-account-browser-uat/extension \
  --api-port 18011
```

Expected: runner reports the actual extension-downloaded archive path, distinct source/destination roots, completed import, matching destination media SHA-256, and restored embedding/vector identifiers. A build-only result or service-worker skip fails.

- [ ] **Step 6: Inspect the produced archive**

Inspect the WebUI and extension archive paths reported by Steps 4 and 5. Verify `manifest.json`, `file_inventory`, account inventory summary, bundled media artifact bytes, pointer-only warnings, and no raw server storage paths or secrets.

- [ ] **Step 7: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/services/core_jobs_worker.py \
  tldw_Server_API/app/core/Chatbooks \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  -f json -o /tmp/bandit_chatbooks_post_merge_uat.json
```

Expected: zero new findings in changed code.

- [ ] **Step 8: Update tracking honestly**

Reopen the stale checked P0 items before implementation. Mark them complete only after the live WebUI and extension round trips pass. Append the post-merge evidence to the review report instead of overwriting the original pre-remediation findings.

- [ ] **Step 9: Commit final evidence**

```bash
git add Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md backlog/tasks/task-12098.1\ -\ P0-Chatbooks-backup-restore-correctness-remediation.md backlog/tasks/task-12098.2\ -\ P1-Chatbooks-backup-import-UX-clarity-remediation.md backlog/tasks/task-12098.3\ -\ P2-Chatbooks-backup-import-acceptance-coverage.md
git commit -m "docs: close chatbooks post-merge UAT"
```

## Release Gate

- [ ] Full-account export from the WebUI completes and downloads a valid archive.
- [ ] A media-bearing full-account archive imported through the WebUI into a clean destination completes with media and embedding restore enabled by archive defaults.
- [ ] A fixture with bundled media bytes proves those bytes are restored under user-owned storage.
- [ ] Destination verification proves restored media bytes match the source SHA-256 and embedding/vector records exist; job completion alone is insufficient.
- [ ] Destination verification proves account profile/settings values present in the source archive are restored according to the approved sensitive-data policy; inventory counts without payloads are insufficient.
- [ ] Completed jobs never display `0%`.
- [ ] Backup all can start without requiring invented metadata, while generated name/description remain editable.
- [ ] Import preview shows the full account-impact summary and never combines Include all with `Selected: 0`.
- [ ] Dropzone, progress, and empty-state text meet WCAG AA in light and dark themes; all switches and selects have accessible names.
- [ ] Jobs uses the full content width, identifies imports by archive name, offers plain-language recovery, and confirms destructive cleanup.
- [ ] Completed archives show verification state and timestamps with correct timezone semantics.
- [ ] The packaged extension completes the same backup/download/import round trip without a service-worker-discovery skip.
- [ ] Focused backend, frontend, integration, docs, and extension suites are green or have explicit non-product environment skips.
- [ ] Bandit reports no new findings.

## UX Review Addendum

The senior UX/HCI audit is recorded at `Docs/Reviews/CHATBOOKS_POST_MERGE_UAT_UX_REVIEW_2026_07_09.md`.

Independent plan review status: **Approved** after three full reviews. The review loop added complete red-green commands, backend coverage for human-readable job identity, deterministic media-bearing source/clean-destination fixtures, and an orchestrator that imports the exact WebUI/extension-downloaded archive before verifying destination bytes and vectors.

It added the following evidence-backed requirements to Tasks 2 through 4:

- timezone-aware terminal state and persisted post-write verification;
- one-action Backup all metadata defaults;
- full account-impact restore preview and non-contradictory Include all counts;
- WCAG AA dark-theme foregrounds and programmatic names for all controls;
- progressive disclosure for conflict, prefix, and execution controls;
- a full-width Jobs view without the duplicate tracker;
- human-readable import identity, plain-language recovery, and explicit destructive confirmation.

Functional full-archive restore remains the first release gate. UX improvements must not degrade a full-account import by disabling media or embeddings.
