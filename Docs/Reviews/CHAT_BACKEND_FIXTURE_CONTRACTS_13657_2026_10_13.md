# Backend fixture contracts — TASK-13657

PR: https://github.com/rmusser01/tldw_server/pull/3229
Base: dc371c0318. ADR assessment: no new ADR; fixture-only qualification preserves existing database admission, chat ownership, and native storage accounting.

Original full-suite reproduction: 2 failed, 22 passed. The quota fake returned a users row for the native ledger aggregate, producing invalid_native_quota_usage. The legacy knowledge fixture attempted a newly forbidden cross-conversation insert, producing parent_message_unavailable before reaching the API.

The quota fixture now supplies distinct user and empty native aggregate records while retaining the real StorageQuotaService and sparse 5GB+1MB upload. The knowledge fixture explicitly proves current DB refusal and zero inserted messages, then overrides only the owned answer read to present the historical malformed link. The foreign parent and owner reads remain real; API403, zero notes and zero flashcards remain required. No production guard is weakened.

Final full suites: 24 passed, 29 warnings, 0 skips, exit0. Ruff passes. Scoped Bandit: 0 findings, 0 errors, B101 excluded for required test assertions. Black formats only touched test functions; complete Python AST is identical. Independent backend source review: no Critical or Important findings. Normal hooks must pass at commit.

Evidence: /private/tmp/chat-pr3229-backend-remaining-red.log, chat-pr3229-backend-13657-green.log and sibling scoped static reports. Broader CI, whole-PR review, final rebase and merge remain open.
