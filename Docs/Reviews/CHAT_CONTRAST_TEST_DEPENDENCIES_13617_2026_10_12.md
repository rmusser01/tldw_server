# Shared contrast test dependency ownership — TASK-13617

Shared UI contrast tests imported PostCSS and Tailwind without declaring them in the UI workspace, so isolated frozen CI could not collect them. Add the versions already used by both clients as UI dev dependencies; no runtime dependencies, tests, color rules or contrast thresholds changed.

Clean frozen baseline:both suites failed collection on missing PostCSS. Updated Bun1.3.2 frozen install passes, and both emitted Button color/primary-foreground suites pass18tests for WebUI and extension. Normal Node26 WebUI and extension types pass.

Bun relocated hoisted dependency keys in the lock. Comparing every package version/integrity identity before and after found0added/0removed; no dependency upgrade occurred. The owned isolated test graph includes the real extension Tailwind config rather than relying on primary checkout links.

ADR assessment:no new ADR; existing test-tool dependency ownership correction. Bandit not applicable:package/lock/Markdown-only changes. Bounded independent review is recorded in the task. Whole-PR CI/UX/H2 acceptance, final requesting-code-review, rebase and merge remain open.

Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_shared_contrast_dependencies_13617_completed.md. PR:https://github.com/rmusser01/tldw_server/pull/3229.
