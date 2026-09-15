# Chat sidebar and composer layout

TASK13260.38 / UAT097. This bundle records reviewed automated evidence. Native1280x720 geometry, mouse activation, keyboard/resize/mobile checks, final combined compiler and full fresh UAT remain parent-owned and pending. jsdom is not native acceptance.

## Change and evidence

Expanded shortcuts consumed the fixed sidebar height, collapsing the nested recent-list scroller; saved rows overflowed behind the footer. One bounded middle scroller now contains shortcuts and all recent controls/results, with a nonshrinking header/footer outside. Casual composer groups wrap according to their actual pane width instead of forcing viewport-based desktop nowrap. Navigation, selection, action ordering and identity handlers are unchanged.

Original-source replay of unchanged final tests:2 expected failures/43 passes, for absent shared scroller and desktop nowrap. Corrected focused/adjacent run56 passes/6 suites; independent rerun also56/6. These overlapping counts are not additive. The initial45-test runs and original/independent replay logs are retained. Lint covers all4 code/test paths with0 errors,4 unchanged warnings,0 added/removed signatures. Bandit is not applicable to JSX/TypeScript-only scope; full compiler remains parent-owned.

The original exact source replay config and both original source files are byte-preserved. It retains absolute /private/tmp and repository references; restore those named source/config inputs to their original paths before running the frontend Vitest command from repair-report.md. Final source/test snapshots and current-source-freeze.json bind the current reviewed production/test files. Original frozen task hash is historical: task notes legitimately change as review/retention proceeds.

## Native boundary

The tests assert actual rendered containers/classes and keep existing control, collapse, lazy-history, navigation and mobile behavior coverage. They do not calculate flex geometry or prove mouse hit testing. Parent must verify expanded13 shortcuts and Recent with many saved rows at1280x720, footer clearance, actual row mouse selection, composer controls with Runtime rail open, smaller viewport/mobile fallback and keyboard access. Original native geometry/screenshot are referenced by the repair report and remain parent-owned; this automated bundle does not add or claim native acceptance.

## Retention

Only log/Markdown/text trailing whitespace is normalized. Source, probe/config and JSON bytes remain exact; retention-manifest.json records origins and normalization hashes. SHA256SUMS binds every retained artifact except itself. Private known-isolated-credential and JWT/private-key scans passed before writing. No production/tests/runtime/browser/commit/global-document changes were performed for packaging. Final native/compiler additions remain with parent.
