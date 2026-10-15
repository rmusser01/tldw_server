# Provider-owned notification bridge — TASK-13615

AppProviders mounted a second notification toast bridge outside NotificationLifecycleProvider. The bridge had no context and performed no notification work, but its static import pulled lifecycle dependencies into every page. WebLayout already owns the working provider and bridge, with authentication/header and demo gating.

Remove the inert AppProviders import/render and its obsolete private flag throughout callers and fixtures. Keep ToastProvider global and keep the actual WebLayout notification owner unchanged. Existing auth/nav/draft assertions remain; the one flag-only case now asserts authenticated layout visibility. The dependency guard verifies that the real bridge stays inside its provider.

## Verification

- Owner/import regression controls before repair: 2 failed / 4 passed.
- Actual notification lifecycle/toast, AppProviders children/splash, dependency ownership and app auth/layout: 5 suites / 132 passed.
- Normal Node26 WebUI tsconfig.json and extension tsconfig.compile.json: both passed.
- Fresh isolated Bun1.3.2 frozen-lockfile graph, locked Next16.1.4, production/quickstart Turbopack: build and token sync passed.
- Shared shell: 600.5 KB before, 589.7 KB after, below unchanged600 KB cap; heaviest route868.4 KB below unchanged900 KB cap. No budget or dynamic-loader workaround.
- Independent bounded reviewer: no Critical, Important or Minor issue. Reviewer inspected source; root ran verification.

## Scope and limits

ADR assessment: no new ADR; this removes a duplicate and retains existing React context ownership. Bandit not applicable: TypeScript/TSX and Markdown-only change. Local Docker container execution remains unavailable; exact locked build is local qualification, not a claim that remote CI passed. No installed-extension, physical AT or whole-PR acceptance claim. Remaining CI/UX/H2 work, final requesting-code-review, rebase and merge stay open.

Earlier copied-dependency Turbopack attempts were stopped and reassessed after escaping/incomplete workspace graphs; the repository's exact frozen Bun workflow resolved that environment problem. Existing worktree/live-preview dependencies were not changed.

Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_shell_notification_owner_13615_completed.md. PR: https://github.com/rmusser01/tldw_server/pull/3229.
