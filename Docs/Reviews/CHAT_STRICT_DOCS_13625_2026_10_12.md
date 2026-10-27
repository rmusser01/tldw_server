# Strict published documentation — TASK-13625

The public mirror now includes canonical ADR068, ADR070, ADR071 and the current ADR index. The supported refresh script produces the same 481-file tree on repeat runs. Its clean/stale replacement and backup-cleanup tests required the current tracked mirror; no refresh transaction change was needed.

Strict compilation failed on optional creation-date metadata for renamed ADR069. The date plugin separately falls back to build timestamps for missing Git history, which can put creation after update. Creation footers are disabled; documented per-page last-updated timestamps and strict builds remain enabled. The canonical site guide explains this choice. The reviewed blank line before ADR070 was removed so the index remains one table; accepted ADR rationale is unchanged.

Both full Docs suites pass52 tests after the final formatting change, including the strict local build and unchanged canonical Site sources. The standalone strict build also passed before that formatting-only correction. The public/private boundary passes. Independent bounded review has no Critical/Important findings; its minor table correction is addressed. Normal hooks are recorded in the task. Twelve pytest warnings are reported by the existing environment; they did not weaken the strict docs compiler.

ADR required: no; publication manifest, source preservation and strict build rules remain intact. Bandit is not applicable to this docs/config/generated-Markdown scope; no Python source or test was edited. Whole-PR CI/UX/H2 qualification, final review, rebase and merge remain open.

PR: https://github.com/rmusser01/tldw_server/pull/3229
