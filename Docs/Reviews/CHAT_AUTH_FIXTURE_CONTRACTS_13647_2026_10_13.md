# Quota and identity fixture contracts — TASK-13647

Five failures were reproduced in four existing suites (5 failed, 22 passed). Production quota reads and guard placement were inspected before changes. Tests now supply the current quota service method and split legacy/native response components; the exact expected-user route set includes the native roleplay append while retaining its first-dependency assertion.

Synthetic migration fixtures still start at99 and explicitly prove migration100 skip behavior before upgrading to actual current head. Both verify the native reservation ledger exists; the missing quota-column fixture still proves no legacy override table was created. Fallback quota values differ from stale user context, proving authoritative service use while preserving exact sanitized logger assertions.

Final connected suites:27 passed,0 skipped,167 warnings,20.97s,process0. Ruff and Black pass on four touched files; whitespace check passes. Scoped Bandit has0 new findings/errors versus parent, with10 unchanged synthetic password/token literal findings (test asserts excluded with B101). Independent bounded review requested two stronger assertions; both are implemented and re-reviewed clear. Exact-file commit hooks required.

ADR assessment:no new ADR; test contracts only, existing quota and identity enforcement unchanged. PR:https://github.com/rmusser01/tldw_server/pull/3229. Whole-PR review, remaining fixes, final rebase and merge remain pending.
