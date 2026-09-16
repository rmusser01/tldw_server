# UAT111 retained error correction — ready for review

Task13260.51, baseline a8e5430d7692bf44669971a0187056b1fe7fb442. Frozen 2026-09-16T05:50:38.052Z. One production file, one test file, official task note. No other files owned; no staging, commit, or runtime action.

Notes backlink guard treated a retained local-only error bubble as unsaved work after a successful Retry. Both the initial guard and delayed-read subscription now use the same narrow predicate: recognized decoded assistant display errors are excluded, while real user/assistant drafts, malformed error text, active streams, account/source changes remain protected. No broad assistant exemption or data cleanup.

RED: two expected failures /three passing controls /15 unselected, permanent actual Notes menu cases. Log /private/tmp/cycle4-uat111-display-error-red.log. One initial fixture assertion counted unrelated title lookup; corrected to canonical transcript reads before the final RED. GREEN:20 passed in the complete backlink-labels suite, /private/tmp/cycle4-uat111-display-error-green.log. Includes Character identity, actual Playground restoration, true drafts and stream/account/note changes. Scoped ESLint:0 errors/64 unchanged warnings across2 files, no added/removed signatures (/private/tmp/cycle4-uat111-display-error-lint-comparison.json). Scoped diff check clean. Bandit is not applicable to this TypeScript-only correction. Parent owns combined compiler and native recheck.

Exact UI command from apps/packages/ui:

    ./node_modules/.bin/vitest run src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx --maxWorkers=1 --no-file-parallelism

Exact lint command from repository root:

    node /private/tmp/cycle4-uat111-display-error-static-check.mjs

Hash manifest: /private/tmp/cycle4-uat111-display-error-owned-manifest.json. Native evidence remains /private/tmp/cycle4-uat108-correlated-native-multi-final-report.md; no new native pass claimed after this fix.

## Independent review correction

Refrozen 2026-09-16T05:56:46.632Z. Substantive unACKed images now count as unsaved work before the text-only display-error exemption. Permanent user image-only, assistant image-only, error-plus-image and delayed-image controls reproduced four expected RED failures with 20 passing controls; final full suite 24 passed. Logs: /private/tmp/cycle4-uat111-image-red.log and /private/tmp/cycle4-uat111-image-green.log. Repeated scoped lint remains zero errors /64 unchanged warnings. Manifest above refreshed for all three owned paths.
