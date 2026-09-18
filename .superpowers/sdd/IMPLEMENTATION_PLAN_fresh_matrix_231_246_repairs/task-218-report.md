# TASK13260.218 / UAT277 — truthful deletion wording

The single-book dialog and bulk confirmation now describe removal from the library. The count remains visible and the bulk sentence handles one or multiple books. The correction changes only two display strings in useWorldBookBulkActions.tsx; callbacks, endpoint arguments, timer, undo and cancellation remain unchanged. It neither promises permanent erasure nor introduces a restore action.

Baseline native13/14 in native-entry274275-review retains permanent-removal wording and the actual soft-deleted response. Root verified route and service default hard_delete=False. Existing bulk selection/keyboard controls pass3/3. No new test was added for this reversible copy-only correction. Scoped ESLint from repository root parses the actual file with0errors/9warnings; initial frontend-CWD attempt ignored the outside-base file and is retained as non-qualifying. Bandit0findings/1TypeScript AST parse error is not TypeScript security assurance. No whole-UI compiler success claimed.

Independent review and committed native single/bulk confirmation inspection with Cancel/no deletion are pending. No source data or runtime changed by this edit.
