# WebUI initial setup test ownership — TASK-13627

Initial connection setup and cookie/session authority fixtures now live in the frontend package, where actual WebUI storage aliases are configured. Their former shared UI location ran native Plasmo, whose instance get/set fields differ from the WebUI prototype controls used by these WebUI-specific flows. Native storage/history qualification remains in the shared UI package.

Only three relative imports became existing shared source aliases and each file gained a runtime-owner comment. A normalized full-body comparison proves all behavioral assertions and cases are unchanged. Both original suites passed39/39 under WebUI before relocation, and both relocated suites pass39/39. The required frontend CI discovery/sharding includes the new paths. Both normal client type checks pass; those checks qualify client source, not a dedicated test-file type check. Bounded review is clear. Normal exact-file hooks are recorded in the task.

ADR required: no; apply existing WebUI/native test ownership without changing production configuration, storage, authority or credential locks. Bandit N/A TypeScript test-only move. Installed-extension qualification, remaining CI/UX/H2 work, final whole-PR review, rebase and merge remain open.

PR: https://github.com/rmusser01/tldw_server/pull/3229
