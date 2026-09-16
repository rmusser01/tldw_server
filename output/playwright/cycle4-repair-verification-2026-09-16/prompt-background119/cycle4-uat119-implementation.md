# UAT119: opaque full Prompt editor

TASK13260.59; IMPLEMENTATION_PLAN_uat_cycle_4.md.

Native RED is the transparent full editor captured on4bd4e2dfda in cycle4-targeted-remaining-prompt-sync-failure.png and prompt-layout-observation.json. The observed fixed container background is rgba(0,0,0,0).

Correction replaces exactly four unsupported bg-background classes with the existing bg-bg theme token: standard, recipe, quarantined and mobile-preview shells. No global theme, z-index or layout behavior change.

Validation: existing PromptFullPageEditor structured-prompts suite10 tests passes before and after. Actual PostCSS/Tailwind generation confirms all four references use emitted opaque bg-bg CSS. Scoped ESLint0 errors/5 existing warnings, same signatures as HEAD. git diff --check clean. Python security scan inapplicable to this TSX class-only change. No redundant class-name test or whole compiler rerun was added.

Independent source reviewer review_ingest065 confirmed the diff is exactly those four substitutions and the shared config/light-dark variables support the emitted token. Their earlier actual Tailwind compile independently agrees. Native dark/light screenshots and computed backgrounds still pending after the shared source freeze; no live acceptance or full UAT pass is claimed.

Commands: from apps/packages/ui, ./node_modules/.bin/vitest run src/components/Option/Prompt/__tests__/PromptFullPageEditor.structured-prompts.test.tsx --maxWorkers=1 --no-file-parallelism. ESLint from root uses apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs and stdin baseline.
