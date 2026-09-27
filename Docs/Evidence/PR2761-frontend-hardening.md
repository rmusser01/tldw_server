# PR #2761 frontend gate and dependency follow-up

Tracking: TASK-12116 and TASK-13013.3. These changes extend the candidate based
on `decdf9db77`; the release manifest records the subsequent source commit.

## Fixed CI assertion

Shard 6 failed because the Flashcard cloze-template test asserted on help text
immediately after the form value changed. Ant Design field updates and the
`Form.useWatch`-driven render are separate observations. The test now awaits the
help text itself, preserving the model and draft assertions. The complete file
passes **8 tests** with CI's 15-second per-test timeout. An earlier local run
hit that timeout in a different save-template case; a complete repeat passed.
No application behavior or timeout was changed.

## Shared dependency versions

The extension dependency ranges and shared UI peer ranges now match the existing
WebUI versions. Frozen Bun installation verifies that all three workspaces
resolve the same packages:

| Package | Installed version |
| --- | --- |
| zustand | 5.0.10 |
| dexie-react-hooks | 4.2.0 |
| marked | 17.0.1 |
| d3-dsv | 3.0.1 |
| property-information | 7.1.0 |

The lockfile drops duplicate resolutions. Shared stores already use
`createWithEqualityFn` or stable selectors; the
[Zustand migration guide](https://zustand.docs.pmnd.rs/reference/migrations/migrating-to-v5)
explains the equality-function requirement. Existing Markdown, persisted-store
and migration tests pass **28 tests**; Dexie transaction and TTS clip suites pass
**10 tests**. Both the full nonincremental WebUI typecheck and the extension's
existing strict compile project pass. This does not claim complete extension
browser-build or whole-product migration coverage.

## Incremental strictness

`apps/tldw-frontend/tsconfig.strict.json` enables all strict checks for the shared
absolute-URL guard, safe external URLs, timeout classification and API-key
placeholder handling. A required CI step runs it alongside the unchanged full
WebUI typecheck. A deliberate external probe fails with TS7006 for implicit
`any` and TS2322 for assigning null to a string; the actual four modules pass.
The CI contract test failed before adding the step and passed afterward.

This is a first enforced boundary, not full WebUI strictness. The tracked next
expansion is request transport and credential handling, then stores and hooks,
then component callers. The prior full-project measurements remain **948
noImplicitAny diagnostics in 252 files** and **664 strictNullChecks diagnostics
in 177 files**; those separate baseline counts are not a current combined strict
count and are not waived by the smaller project.

## Hooks enforcement

A 5,154-file scan with `purity`, `static-components` and `use-memo` enabled found
21 additional findings: 8 purity, 11 static-components and 2 use-memo. The two
use-memo findings were Timeline dependency arrays containing `JSON.stringify`.
They now depend directly on the immutable message/history arrays, and
`react-hooks/use-memo` is enabled in the shared lint configuration. Scoped lint
passes with existing test `any` warnings. Classic rules-of-hooks remains enabled
outside the existing E2E override.

The subsequent [hooks follow-up](PR2761-hooks-enforcement.md) repairs purity
and static-component findings and extends enforcement to shared UI sources.
Four disabled compiler-era rules still need repairs or individual justifications:
a full scan found 243 refs, 89 set-state-in-effect, 22 immutability and 49
preserve-manual-memoization findings. These 403 findings remain tracked work.
The full scan had unrelated existing lint diagnostics; it was not a passing
full lint run. TASK-12116 remains open until the remaining strictness/hooks work
and final-head CI evidence are complete.

## Strict timeout calculation and runner harness follow-up

Endpoint timeout selection now lives in the already strict-checked
`request-timeout.ts`, with the public request transport wrapper retaining path
normalization. Missing settings, numeric-string settings, endpoint defaults,
generation overrides and metadata/slides floors retain their existing behavior.
The transport suite passed **19 tests** before extraction; the transport and
utility suites pass **22 tests** afterward. The strict project also passes.
Eleven nullable configuration diagnostics were removed from request
transport. The full separate strict-null scan still reports **653 diagnostics
in 176 files**; that is an inventory, not a passing gate.

The Skills certification runner's test harness now preserves the generic type
of optional overrides and explicitly types asynchronous teardown/captured
callbacks. Its **47 tests** and scoped strict-null compile pass. A separate
combined `noImplicitAny` + `strictNullChecks` baseline before these follow-ups
reported **841 diagnostics in 222 files**. Different compiler flag combinations
produce different inference and diagnostics; these counts must not be added or
presented as final full-WebUI strictness.

Follow-up logs: `/tmp/pr2761-timeout-extraction-after.log`,
`/tmp/pr2761-timeout-strict-gate.log`,
`/tmp/pr2761-strict-followup-shared-lint.log`,
`/tmp/pr2761-skills-strict-tests.log`,
`/tmp/pr2761-skills-strict-compile.log` and
`/tmp/pr2761-skills-strict-lint.log`.

Local logs: `/tmp/pr2761-flashcard-final.log`,
`/tmp/pr2761-aligned-dependency-tests.log`,
`/tmp/pr2761-aligned-dexie-tests.log`,
`/tmp/pr2761-aligned-webui-typecheck.log`,
`/tmp/pr2761-aligned-extension-typecheck.log`,
`/tmp/pr2761-strict-boundaries.log`, `/tmp/pr2761-strict-rejection.log`, and
`/tmp/pr2761-frontend-touched-lint.log`.
