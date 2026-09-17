# TASK13260.172 — explicit first-admin bootstrap harness action

This harness-only action bridges normal completed AuthNZ initialization to the existing supported multi-user admin creation function. It uses the unchanged launch preflight, frozen source, exact private interpreter/cwd/runtime environment and PostgreSQL role validation. Single-user cells and repeated completed bootstrap are refused. A fresh per-attempt token and preparation digest bind the success proof; normal true function return plus process exit0 without interruption are both required. Planned credentials remain private; neither argv nor receipts contain their values. This does not reset an existing admin password or create Alice/Bob: those remain normal administrator UI steps.

## Stage1 — causal guards
**Status**: Complete
Add fake-process launcher and fake-import helper controls. No actual profile/database/app import is allowed. Preserve all existing launcher/initializer guards.

## Stage2 — bounded action/helper
**Status**: Complete
Add bootstrap-admin branch within existing launch; new Python helper invokes only supported create_admin_user_non_interactive, rejects unsafe inputs/wrong imports, and writes proof after true normal return.

## Stage3 — verification/freeze
**Status**: Author verification complete; independent review pending
Run all existing launcher/initializer controls plus new tests; syntax/compile/Ruff/Bandit; snapshot exact changed harness/test/docs bytes. Root independently reviews before any real execution. No matrix gate change.
