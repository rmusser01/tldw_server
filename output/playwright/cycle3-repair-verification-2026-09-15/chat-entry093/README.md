# UAT093: saved Chat entry and delayed local restoration

Reviewed correction for TASK13260.33, on the integrated dev repair branch. The five source/test files match the frozen manifest. This bundle retains the original failing interacting probes, permanent failures, passing corrections, independent review, scoped lint and the unchanged 90-diagnostic TypeScript baseline.

The settings-return effect no longer repeatedly cancels restoration. Local history reads stop publishing after teardown, replacement, existing restore cancellation or explicit principal invalidation; callers honor an unaccepted result. The existing offline local-history contract is preserved. No new verified-ownership guarantee is made for legacy local records or direct configuration changes.

Parent/implementer and independent relevant runs each pass 81 tests in six suites; these overlap. Both original probes pass unchanged in both runs. The preserved delayed probe now reports a duplicate test-fixture key after the permanent fixture gained that export; repository code has no duplicate. ESLint covers five files with no errors or added normalized warnings. TypeScript retains exactly 90 existing diagnostics and exits 2; this is not a clean typecheck. Bandit is inapplicable to this TypeScript-only scope.

Native failure and successful normal-reload recovery remain in ../chat-mirror-native-round8. Exact repaired entry/backlink/reload acceptance is pending while the adjacent title correction settles. The full fresh single/multi matrices remain pending.

The reports describe their original freeze-time status; the independent review is now clear. Original probe configs and JSON retain their bytes; copied logs/reports normalize trailing whitespace and final newlines. Configs reference this checkout's absolute paths. All files are scanned for known isolated runtime secrets and token/private-key patterns and indexed in SHA256SUMS.
