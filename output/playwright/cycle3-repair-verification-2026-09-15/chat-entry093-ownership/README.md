# UAT093: saved Chat identity ownership correction

TASK13260.33. **Implementation and independent re-review are clear; remaining native acceptance is parent-owned.** Source/test freeze: **2026-09-15T23:08:52.251Z**, seven production and five test paths in `uat093-ownership-final-manifest.json`. No full fresh-matrix completion is claimed.

The real Next WebUI storage shim exposed the failure: independent hook renders lag behind shared storage writes. Playground incorrectly treated stale rendered preference as explicit user intent and cleared a canonical saved conversation. The correction removes that destructive inference and the failed premetadata write/wait workaround. Canonical saved identity wins; explicit picker, inline role-play and template actions own detachment. Existing owner, late profile, local-load and mirror guards remain.

## Results and limits

- Prior complete affected run: **194 tests /11 suites**, independently repeated. The final narrow scene correction was then verified with **50 tests /2 suites** and the **unchanged original draft probe1/1**, independently repeated. These overlap; do not add their counts.
- Original actual-WebUI immediate, held-profile and cross-tab behavior replay: **3/3** through the explicit compatibility runner. Real picker and saved sidebar interactions are also permanent controls.
- ESLint: **1 existing error,132 warnings,0added** versus HEAD; one old warning removed. This is not clean lint.
- Parent's final combined compiler: **90 baseline/90 current signatures,0added/0removed** at23:13:28.866UTC. The root20-file source manifest includes adjacent frozen scopes. This is not a clean typecheck.
- The prior candidate passed native multi canonical Robot at23:04:57: Robot5, original two messages/BEEP BOOP, title, loaded/idle. That observation precedes the final scene-only correction. No native combined Drawer→Form or complete new matrix is inferred here.

The final report and independent review contain commands, exact behavior changes and historical findings. `chat-entry093-native-reassessment` retains the causal native failure, sidebar clear stack and actual-shim RED evidence. Older `chat-entry093` and `chat-entry093-followup` retain unsuccessful earlier timing repairs; their test passes did not establish native success.

## Scene action and final draft correction

Identity/settings acceptance now occurs before scene save. A changed identity detaches synchronously, scene persistence uses its accepted destination, and obsolete completions cannot publish into or close a replacement conversation. Partial failure truthfully reports that identity/settings were applied while scene save failed. It retains the edited scene for retry, with no atomicity or rollback claim.

Independent review then reproduced an additional own-detachment race: rerendering the Drawer against its accepted null destination reloaded defaults and erased the edited scene. A pending accepted-destination/current-action ref now suppresses only that owned reinitialization. External replacement and close→reopen still load their new scenes. `uat093-draft-scope-red.log` records the three permanent failures; final and independent `draft-*-green.log` files establish their corrections. The unchanged original config and fixture remain byte-for-byte preserved.

The pre-draft-fix Drawer was **reconstructed**, not saved contemporaneously. Removing only the four final pending-destination additions yielded `uat093-pre-draft-fix-drawer.reconstructed.tsx`, SHA256 `024398fcf20eafb57d5b7e9f3e4639eeeca414a43d873b5f297e35b4daa57e9c`,28,319bytes. Parent verified the full hash against its contemporaneous23:02:23 candidate manifest. It is the exact candidate that failed, not the older HEAD scene-first source.

## Original probes and replay

All retained original configs and fixture bodies preserve their bytes. The actual-WebUI config import chain is included because `uat093-ownership-web-compat.config.ts` imports it. That runner restores only the matching prior coordinator fixture and retains the actual-shim/isolation/cross-tab behavior transforms; historical hypothetical production transforms are excluded. All production tested by the GREEN runner is current. Restore temp filenames to their original `/private/tmp` paths before using the report commands after temp cleanup.

The retained older raw-setter expectation that preference hydration should detach is intentionally superseded: global/cross-tab preference is not explicit current-tab intent. Original scene-first probes are also historical because the accepted-first action order deliberately changed. Their RED artifacts remain as history, not silently rewritten GREEN. Intermediate RED logs without a dedicated source snapshot are historical observations, not claims that every intermediate state can be replayed from final source. Baseline Form/template copies are included only because their retained RED configs require those exact files.

Form tests render actual Form/template code with a mocked Drawer shell; Drawer tests render actual Drawer and stores with controlled callback/persistence boundaries. The Form-owned guard is tested at the prop seam. These are not a substitute for native combined scene action acceptance.

## Retention verification

`manifest.json` lists55 original artifacts, source/retained sizes and hashes, modification times and transformations. Configs, fixtures, reports and manifests retain original bytes. Log trailing whitespace is normalized. ESLint JSON omits redundant embedded source/output only. `SHA256SUMS` covers every bundle file except itself.

All copied text was scanned against16 unique secret values from the two private cycle3 runtime manifests, JWT patterns and private-key patterns: **0 matches**. JSON is parse-validated. No private runtime manifest, credential, broad log directory or browser profile is included. This bundle has no PNG; native screenshots remain in the previously inspected native bundles. Originals remain unchanged. No product/test/runtime/browser/Git mutations were performed during retention.

## Root final complete affected run

After the final narrow Drawer correction, root reran all11 affected suites at23:27UTC: **197/197 tests pass**. `uat093-root-final-full-green.log` retains that final full run, including the three added scene-draft controls. It supplements the55 original indexed artifacts; prior194 and final197 counts overlap and are not additive. The12 frozen source/test hashes still match. This does not certify the separate native combined scene action or full fresh matrices.
