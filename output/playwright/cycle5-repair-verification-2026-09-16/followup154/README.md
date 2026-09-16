# Settings auth lifecycle fixture repair

Three existing test files now use the same storage shim as the WebUI. Removing those three mock blocks restores the original files exactly, preserving all 77 assertions and account-transition, invalidation, billing and watcher scenarios. Author run: 61 tests across seven suites; independent run: 36 tests across the three affected suites. Runs overlap and are not summed. No skips; scoped ESLint has zero errors or warnings.

This is a test-only correction, with no application change. Baseline 28 failures and exact-heading comparison remain in the neighboring 152–153 bundle. Final combined compiler verification remains pending; Bandit does not apply to TypeScript-only tests.

Final integration verification is retained in [the combined bundle](../followup151-154-combined/README.md):3326/168 shared-UI and332/17 WebUI pass separately; TypeScript remains at90existing diagnostics with0added/removed. Native152/153 acceptance remains pending.
