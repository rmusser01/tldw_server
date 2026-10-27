# Catalog and recovery harness contracts — TASK-13614

Character first-use fixtures now supply the current playground:chatModels query and truthful success/pending state. Research-context tests own a real isolated QueryClient and control catalog I/O. Notes keeps real chat-settings exports and verifies H1 identity/message publication with metadata admission still pending at navigation. Composer signals require captured focus admission rather than an obsolete callback spelling.

Popup renderer adapters consistently declare scene, draft-retention, creation-ledger and native-send I/O. Their controller declares idle/ready state. These are renderer interaction tests; separate native admission/durability suites retain responsibility for those boundaries. A new failed-retention control forbids native send/navigation and keeps the exact input.

## Evidence

- Native shared runner before repair: 4 suites / 34 failed / 126 passed.
- First repaired pass: 161 passed / 3 failed, identifying the incomplete popup I/O boundary and staged Notes metadata expectation.
- Focused final popup, failed-retention and Notes controls: 4 passed, 129 excluded by filtering.
- Final six-suite run: five complete suites passed (93 cases); Manager had100 passing cases and one60-second prompt-editor timeout. Its isolated unchanged control passed, then its complete101-case suite passed with one worker. All194 affected cases are therefore qualified across the final complete-suite runs; no timeout or assertion was weakened.
- Actual popup ownership and catalog stale-account/error/refetch controls were included.
- Both normal Node26 client types passed after the final fixture/source state.
- Bounded independent source review found no Critical, Important or Minor issue, including renderer I/O scope and metadata admission.

## Scope and limits

ADR assessment: no new ADR; fixture ownership follows existing catalog, acknowledged recovery and H1 staged-publication decisions. Bandit not applicable: tests/Markdown-only changes. Production readiness, ownership validation and recovery implementations are unchanged. The renderer fixture does not certify installed-extension storage or a physical native server; opt-in owned-server/native coverage remains separate. Native registry-unavailable diagnostic warnings in the research renderer do not establish durable storage acceptance.

Task13615 separately resolves the locked production bundle gate with589.7KB shared shell,132notification/auth/splash checks and unchanged limits. Remote CI, whole-PR/H2 acceptance, final requesting-code-review, rebase and merge remain open.

Completed unique plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_frontend_harness_contracts_13614_completed.md. PR: https://github.com/rmusser01/tldw_server/pull/3229.
