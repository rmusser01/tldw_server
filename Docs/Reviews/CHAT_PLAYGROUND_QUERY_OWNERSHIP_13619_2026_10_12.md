# Playground harness query ownership — TASK-13619

Search, sticky composer and coordinator harnesses now give each test one isolated real QueryClient. Rerenders retain their provider; caches are cleared after each case. Search/sticky declare only model-catalog I/O while retaining real service exports. Coordinator keeps its existing catalog boundary, preserves real settings exports and removes four redundant inner clients/providers.

All original search, layout, held restore, scope, route-retirement and negative assertions remain. The provider-less baseline was100failed10passed. Final3suites110passed; both normal Node26 client types pass. Bounded independent source review found no Critical, Important or Minor issue. The one test-only coordinator render-alias error from the initial repair was corrected before final verification.

ADR assessment:no new ADR; tests own the existing query/catalog lifecycle. Bandit not applicable:tests/Markdown-only changes. No production readiness/ownership/storage implementation or global native alias changed. Installed-extension/physical AT, remaining CI/UX/H2 acceptance, final requesting-code-review, rebase and merge remain open.

Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_playground_query_ownership_13619_completed.md. PR:https://github.com/rmusser01/tldw_server/pull/3229.
