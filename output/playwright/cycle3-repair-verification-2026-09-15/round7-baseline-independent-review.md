# Round 7 / dev baseline independent evidence review

## Result

Clear for the stated evidence scope after the qualification follow-up. Round-5 normalization and Git ancestry claims are verified. Round-7 captures support completed-result retention and empty-wizard behavior; the destination admin identity is now explicitly qualified as runner-reported. No new browser/API request was made during this review.

## Round 5 normalization

All 14 SHA256 entries match, with no unindexed bundle files except SHA256SUMS. The copied `uat092-existing-green.log` exactly equals the original `/private/tmp/uat092-existing-green.log` after removing trailing blank lines; final newline remains. README discloses that normalization. The 24-test/3-file result is unchanged.

## Round 7

All 9 SHA256 entries match, with no unindexed bundle files except SHA256SUMS. Captures establish:

- Empty wizard then queued `onboarding-uat-note.md`.
- Quick preset / Extract / Server storage review, followed by one succeeded and zero failed with a source action.
- Actual close/reopen command retaining that completed filename/result at 20:03:49 UTC.
- Signed-out Settings Login Required form, then a Login click at 20:05:51 UTC.
- Empty Quick Ingest Add step at 20:08:41 UTC, Configure 0, `bobFilenamePresent: false` and zero previous source actions.

The README accurately excludes popup destruction, new full matrices and stale-result action tests, and records the runner's two mistaken control assumptions as automation failures.

### Evidence qualification needed

The retained `uat082-admin-login-action.txt` only shows clicking Login; it does not retain the helper's selected credential identity or a successful postlogin principal/role. None of the nine files independently identifies the new authenticated principal as admin. The empty-wizard result is directly supported, but the asserted Bob-to-admin boundary currently relies on the runner's account-selection report. Retain an existing sanitized successful identity capture, or state that identity limitation explicitly. Parent was notified; this is not evidence of a product failure.

## Dev ancestry

Independently checked `dev-baseline-ancestry.json` against local Git:

- Repair branch reflog creation: 2026-09-14T19:24:11-07:00, at `cb8335cf8dc14c778d569a003e6950334a027ecc`.
- That commit's parent is the initial `54ecc7e7735fc082b711d922a27e97ee9999e5ae` checkout.
- `origin/dev` reflog already records `2e1a5e58d3344a1efd578efb4dbfb1c9465e8767` at 19:16:06-07:00, before branch creation.
- Merge base is `c70387f496d82fcee92926bf3715bf5cd240ba88`; exactly 32 remote-dev commits were absent at branch creation.
- At the recorded HEAD `c1b2cef1c24a371f3a223b1059bec6125bcd6d54`, left/right counts are exactly 32/65.
- FETCH_HEAD is dated 2026-09-15T20:07:10.237Z and records the same dev tip. No fresh fetch was performed by this reviewer.
- `origin/dev` is not an ancestor of current HEAD: actual integration remains pending. A legacy read-only `git merge-tree --trivial-merge` preview of these committed trees contains no conflict markers; this is corroborating preview evidence, not an actual merge, test run or validation of uncommitted Chat work.

The tracker opening correction, ENV-008 and TASK-13260.32 plan accurately answer the user's question: this work did not start on the latest already-fetched dev. They distinguish earlier exact-checkout test evidence from latest-dev acceptance and require integration/revalidation before another fresh matrix. The conflict-free preview is not represented as completed integration.

Only this private report was written. No repository, browser, runtime, credential-store or Git-ref changes; active Chat product changes were not reviewed.

## Final round-7 disposition

Re-read the corrected README and tracker evidence link. Both now explicitly say the admin identity is runner-reported from the selected isolated credential helper, with no independent post-login principal/role capture. Recomputed all nine hashes after that edit: all match. The requested correction is sufficient; no remaining evidence finding within the qualified scope. The body still records the runner's Bob-to-admin sequence, while the limitation prevents treating those captures as independent principal verification.

## Bounded dev integration assessment

Compared `c70387f496..2e1a5e58d3` with the committed UAT branch and current unstaged file inventory. Incoming dev spans 77 files across Explainer navigation, VN generation readiness/image-model resolution, and VZ guest diagnostics/drills. Only two incoming files also have committed branch changes; no incoming file overlaps the currently unstaged files. This is an integration assessment, not a full review of the already-merged upstream features or the active Chat correction.

### Concrete intersections and checks

1. **Auth dependency composition:** `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py` adds optional `per_user=False` to RBAC rate limiting, with `per_user=True` used only by the new VN preflight endpoint. Our login/refresh statement-autocommit and acquisition-error handling changes are around line 390, separate from the incoming limiter changes around 2196. Preserve both. Run `tests/AuthNZ_Unit/test_auth_deps_hardening.py`, `test_auth_login_sqlite_connection_scope.py`, and `test_auth_refresh_sqlite_connection_scope.py` together after merging, plus `tests/PrivilegeCatalog/test_privilege_catalog_loader.py` and `tests/VN_Assets/test_vn_assets_api.py`. Check legacy limiter defaults, user/API-key bucket sharing for opted-in VN requests, foreign-pack denial, and login/refresh busy versus terminal failure semantics. No textual conflict or demonstrated semantic contradiction found.

2. **Navigation/locale composition:** Incoming `en/option.json` adds `header.explainer`/`explainerDesc`; our changes add Study next-due copy in a separate object. Preserve both. Incoming route registry/metadata, shortcut defaults and legacy shortcut migration interact with the repaired shell and ICU handling even though those source files do not directly overlap. Run shared `HeaderShortcuts.test.tsx`, `ui-settings.header-shortcuts.test.ts`, `route-metadata.coverage.test.ts`, `route-governance.metadata-coverage.test.ts`, the incoming `option-explainer.route.test.tsx`, and `i18n/__tests__/icu-format.test.ts`; include the existing WebUI App/title checks in the combined acceptance set. The page inventories gain Explainer on WebUI and extension; inventory discovery should include it, without treating discovery as a live route pass.

3. **Incoming VN/image behavior:** `Image_Generation/config.py` centralizes adapter model precedence; the five adapters now use it. Listing changes sd.cpp readiness to the effective preferred model path rather than accepting a valid fallback while the chosen diffusion path is invalid. VN preflight imports the branch's `core.config.route_enabled`, so the combined configuration must retain repaired defaults and route behavior. Run incoming `tests/VN_Assets/test_preflight.py`, existing `test_vn_assets_api.py`, `tests/Image_Generation/test_model_resolution.py`, and `test_image_models_listing.py`. Frontend checks are `__tests__/vn-assets/VNAssetsWorkbench.test.tsx` and incoming `vnAssetIdempotency.test.ts`, covering pending commands, retries after ambiguous failure, stale pack responses, terminal refresh, and non-generating configuration inspection. No source overlap with Chat/RAG generation defaults was found; this is affected dependency coverage, not a newly discovered failure.

4. **Schema/static validation:** Dev's `lib/api/openapi.fingerprint.json` describes its own VN endpoint/schema additions. The UAT branch separately changes chat raw-message listing, Study request/session schemas and caller capabilities. A clean textual merge does not certify that fingerprint against the combined API. Run the existing `make openapi-drift-check` on the merged code, then resolve any genuine contract drift through the existing fingerprint/type workflow. Re-run full frontend TypeScript and scoped lint/Bandit for incoming/merged production changes. The former 90-diagnostic baseline is evidence for the old checkout; compare concrete diagnostic signatures after integration rather than assuming the number must stay 90 if upstream legitimately changes a diagnostic.

5. **VZ scope:** Incoming guest exec output draining and helper protocol-error classification do not intersect the UAT Chat/Notes/Study implementations. Portable coverage is `tools/macos-vz-helper/Tests/test_failure_drill.py`, `test_failure_orchestration.py`, and `test_failure_workflow_paths.py`, plus ordinary helper Swift and guest Go unit tests in their projects. Linux-only `exec_linux_test.go` is not exercised by macOS Go tests. Host-gated VZ launchd/fault-injection tests require explicit prepared fixtures and opt-in; do not start those workflows merely to validate this web merge or count skips as runtime acceptance.

### Concrete bookkeeping risk

Incoming dev adds two active files with logical ID **13249** (`Add-VN-asset-generation-preflight-and-targeted-recovery` and `Expose-Explainer-in-WebUI-and-extension-navigation`). It also adds an archived logical **13259** VZ task while this branch retains the restored active **13259** second-brain roadmap. Git filenames differ, so a clean merge will not diagnose these logical ID collisions. Preserve every record and avoid ambiguous CLI edits by those IDs until task identity is reconciled through the official workflow; do not overwrite the restored roadmap. This is a tracking integration risk, not a demonstrated application defect.

No merge conflict has been demonstrated. The read-only preview concerns committed trees only; final integration, checks, isolated-runtime restart and fresh matrices remain pending under TASK-13260.32. No tests or mutation commands were run for this assessment.
