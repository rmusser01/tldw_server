# Moderation Compiler-Side Delegate Cleanup

Task: TASK-15205
Status: Awaiting human approval
Base: origin/dev at 54b321dcec8cdbb6358581f42c1548696260570b

## Intent

Finish the compiler-side compatibility cleanup following merged PR #3176.
Reduce redundant private service surface without changing supported moderation
behavior. Compilation remains owned by PolicyCompiler; ModerationService keeps
configuration, I/O, logging, persistence, and public dispatch.

## Approaches

1. Remove only repository-unused private compiler delegates. This is the narrow
   structural approach selected by the requester; it keeps active service paths
   and compiler behavior unchanged.
2. Remove all compiler delegates and rewrite active service callers. This would
   also change internal extension dispatch and broaden this review unnecessarily.
3. Retain unused delegates with deprecation warnings. This adds runtime noise to
   undocumented private methods without reducing maintenance surface.

The exact five-method scope below is the implementation proposal resulting from
the current-dev usage audit, not a previously approved exact-name list.

## Scope

Remove only these private ModerationService methods:

- `_split_action_directive()`
- `_resolve_categories_override()`
- `_parse_categories_override()`
- `_compile_user_rule()`
- `_coalesce_bool()`

Repository source searches find no production callers for these names. Three
tests call `_compile_user_rule()`; the other names have no code references beyond
their definitions. Repeat the usage audit before implementation to catch dynamic
lookups and changes since this base. Do not delete their PolicyCompiler owners.

Retain `_parse_rule_line()`, `_parse_regex_expr()`, `_is_regex_dangerous()`,
`_parse_bool_value()`, `_load_block_patterns()`, and `_build_block_patterns()`:
they participate in linting, override validation/loading, or blocklist writes.
Keep `_ALLOWED_ACTIONS` and other constants unchanged in this slice, even where
additional cleanup might be possible.

## Compatibility

Public APIs, model re-exports, policy compilation, validation, report logging,
public evaluation/redaction dispatch, persistence, and locking stay unchanged.
Removing these undocumented private names is an intentional callable-surface
break: repository searches cannot establish that third-party callers never use
them. No public deprecation contract is introduced.

Do not change regex safety heuristics, scan limits, matching/redaction semantics,
Guardian behavior, or LLM surface coverage. Those remain separate work items.

## Tests

Add a focused red/green test asserting that the five obsolete names are absent
from `ModerationService.__dict__`, following the evaluator cleanup precedent.
Preserve regression coverage while replacing test-only private service calls:

- Move dangerous-regex rejection and rule phase/action assertions to direct
  PolicyCompiler tests, preserving literal expected results and issue checks.
- Replace the private compilation warning-log test with public
  `get_effective_policy()` compilation from controlled raw overrides. Ensure the
  test reaches all three compiler warning cases rather than pre-sanitizing them.
  Keep assertions for dropped rules, expected warnings, and absence of sensitive
  rule IDs and paths in logs. Use a separate compiler report test if needed.
- Retain all active delegation, import-compatibility, and public dispatch tests.

## Verification

Before implementation, compilation and the focused compiler/blocklist/user
override suites pass: 67 tests on Python 3.14.3 (146 existing warnings).

After implementation, compile touched Python files first, then run all
moderation unit tests, Guardian supervised-policy tests, Chat moderation
integration tests, selected Workflow moderation adapters, and the targeted real
Audio transcription redaction regression. Run scoped Ruff, formatting checks,
Bandit, diff checks, and a final source audit. Review the complete change before
PR preparation; no unrelated formatting or metadata cleanup is included.

Implementation begins only after approval of this design and a focused
implementation plan. PR creation and merge remain separate actions with the
repository's human-written Change summary and CI requirements.
