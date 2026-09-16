# UAT166 / TASK-13260.103 — generated-card count feedback

## Approved bounded design

Native generation succeeds with one returned draft but says `Generated 1 cards.`. GeneratePanel's generated, fully saved and partially saved success strings use fallback-only English text; those exact keys do not exist in bundled English option.json or its public extension source. Use the established ICU plural convention at these three existing fallback messages. Keep count inputs and warning/error branches unchanged; zero generation remains its existing no-cards warning. No new helper, locale sweep, backend, auth, or source-review behavior.

Production scope: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`.
Test scope: existing `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`, using the actual ImportExportTab→GeneratePanel→transfer-summary boundary and the real ICU plugin/English resource for plural regressions. Existing fallback translation fixture will use the actual ICU translator so it understands the same syntax as the application.

Related patterns inspected: UAT137 ReviewProgress plurals with real ICU/English resource; ReviewTab queue-state real-resource controls; i18n/icu-format.ts preserving legacy interpolation; actual WebUI i18n static English bundle and shared plugin.

## Stage 1 — RED (Complete)

Add one/multiple normalized-draft feedback cases (provider count deliberately differs), zero-result warning, fully saved one/multiple, partial saves with one/multiple created and genuine retained failed drafts. Require real toast and rendered transfer-summary copy. Capture current source failures before production edit.

## Stage 2 — GREEN (Complete)

Change only three ICU message defaults; preserve actual draft and successful-save counters. Run focused existing generation/save controls and plural regressions.

## Stage 3 — release (Complete)

Run scoped ESLint, JSON/source checks as applicable and diff whitespace checks. Bandit is Python-only; attempt touched TSX paths and retain parse limitations if required. Freeze exact files, hashes, snapshots, commands and results for independent source review. Parent owns native acceptance, task/tracker records and integration. No browser/runtime/git mutations.

RED receipt: red.log has3 expected singular failures/4 controls passed. Earlier missing router in zero-case fixture was corrected before accepted RED; see red-initial-fixture-error.log.

## Final author verification

Production is exactly three message-default changes; count sources and control flow are unchanged. Final61 tests/4 files pass. Scoped ESLint0errors/86existingwarnings, identical diagnostic multiset to baseline. The initial test helper was renamed from a hook-like name to configureTestTranslations; no lint errors remain. Bandit reports2TSX parse errors and0findings, so provides no TSX assurance. Diff whitespace check exits0. Full-project typecheck belongs to parent integration; no compiler pass is claimed.

Native expected text: **Generated 1 card.** Related feedback: **Saved 1 generated card.**, **Saved 1 card; 1 failed.** Zero retains its existing no-cards warning. Native root acceptance and independent source review remain pending.

### Commands

From apps/tldw-frontend:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx -t 'localized generated-card feedback'
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/ReviewProgress.plurals.test.tsx ../packages/ui/src/i18n/__tests__/icu-format.test.ts
# Reviewer baseline replay; loads baseline source without editing shared files:
bunx vitest run --config ../../.tmp/uat166-repair-20260916/baseline.config.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx -t 'localized generated-card feedback'
```

From repository root:

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx -f json
source .venv/bin/activate && python -m bandit apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx -f json -o .tmp/uat166-repair-20260916/bandit.json
git diff --check -- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
```

The mounted regression uses real ICU plus the actual bundled English resource, exercises normalized drafts despite a mismatched provider count99/invalid extra draft, and verifies real transfer-summary text and toast payloads. Saved/partial cases exercise actual success counters and remaining drafts. Existing41 ImportExport regression cases also pass under actual ICU fallback formatting. No resource was added because these three keys are absent from the existing English resource and its public source; no unrelated locale copy was changed.
