# Protected edit harness admission — TASK-13623

The fake Playground API now advertises the existing strict protected-edit schema and enforces its target, branch, captured prefix and digest before admitting provider work or appending a branch input. It reuses the existing schema fixture. Canonical helpers remain lazy so the harness binds its memory database first.

Conversation identity is captured before helper loading and checked again before writes. A deleted or replaced conversation cannot receive held work, including a replacement with the same ID and rows. Reply settlement also checks the admitted input revision. A lawful change to the excluded original target after atomic admission does not invalidate the admitted branch. Canonical originals remain unchanged by edits.

The saved-chat UI suite preserves its six original journeys and adds a missing-capability refusal control. Twenty-four direct API controls cover valid second-user edits, malformed and stale targets, prefix/context mismatch, provider admission, helper-loading replacement, and held-response retirement. Together with harness and interrupted-reply regressions, all 50 tests pass. Both normal client type checks pass. Independent bounded review is clear after the retirement finding was fixed; normal hooks are recorded in the task.

ADR assessment: no new ADR; this harness follows the existing deployed protected edit contract. Bandit is not applicable to this TypeScript test-only scope. Production API behavior is unchanged. Remaining whole-PR failures, UX/H2 qualification, final independent review, rebase and merge remain open.

PR: https://github.com/rmusser01/tldw_server/pull/3229
