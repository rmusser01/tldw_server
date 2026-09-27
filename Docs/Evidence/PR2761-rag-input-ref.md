# PR2761: RAG input ref ownership

Task: TASK-12116. Source before this batch: `0929b44a5af3bed60142550719f610d048bec3e2`.

`useRagSearchState` created and returned an Ant Design input ref even though it
never used it. `RagSearchBar` attached that ref to its input and focused it in
an effect. Including the ref in the general search-state object caused the
compiler's refs rule to propagate ref restrictions to ordinary render reads
such as `search.draftSettings`.

The fix moves the existing ref declaration into `RagSearchBar`, next to its
only consumers, and removes the hook's unused ref export and type import.
State, search callbacks, effect dependencies, animation-frame scheduling, and
focus cancellation behavior are preserved. This repairs one ownership boundary;
it does not represent 128 independent runtime bugs.

## Verification

- The installed ESLint binary with the existing WebUI configuration and an
  explicit `react-hooks/refs:error` override reproduced **128 errors** in
  `RagSearchBar` before the change.
- Final scoped lint over the component, hook, and new test enforces `refs`,
  `immutability`, `set-state-in-effect`, and `preserve-manual-memoization`:
  **0 errors**, 25 existing warnings (22 component, 3 hook, 0 test).
- Four component characterizations cover opening/reopening autofocus,
  autofocus opt-out, pending-focus cancellation, and edited query/filter
  submission through the real search hook. The initial characterization suite
  passed before the production change. The cancellation assertion was then
  strengthened to disable autofocus while the input remains mounted.
- Final installed Vitest 4.0.18 run: **11 tests passed** across the component
  suite and the existing RAG request/source-metadata suites.
- Full WebUI `tsc --noEmit --incremental false`: **passed**.
- Separate no-emit typecheck including the new test and project test setup:
  **passed**, using a temporary config with the installed Node type roots.
- `git diff --check`: **passed**.
- Bandit was invoked from the project virtual environment on the three touched
  TypeScript files. It reported three AST parser errors because it cannot parse
  TypeScript. This is **not** a successful security scan; the reviewed production
  change only relocates an existing DOM ref and changes no data handling.

Reproduction and verification logs are in `/tmp/pr2761-rag-*`, including
`refs-before.json`, `four-rules-final.json`, `tests-final.log`, and
`typecheck.log` (with the common `pr2761-rag-` prefix).

No global rule changes, new suppressions, dependency changes, builds,
installations, commits, pushes, or release metadata edits were performed.
TASK-12116 remains open for the broader frontend strictness and hook-rule work.

## Full follow-up inventory

A complete scan after this relocation covers **5,161 source files** across
shared UI and WebUI pages, with the same four rules explicitly enabled. It
reports **265 findings**: refs 105, set-state-in-effect 90, immutability 21,
and preserve-manual-memoization 49. The prior recorded baseline was 403.
This is a measured inventory, not subtraction of the scoped fixes. In
particular, the ACP permission clock change from an earlier batch adds one
set-state-in-effect finding relative to that baseline; it remains open.

[Remaining findings](PR2761-remaining-hook-findings.json) lists each file, line,
rule and short diagnostic, plus source hashes for the affected files and the
config. The largest remaining clusters are CodeBlock (14), incoming persona
payloads (12), PromptSelect (11), and Notes graph suggestions (10). These require
individual investigation; diagnostic count alone does not establish runtime
defects or justify suppression. The scan leaves global rule configuration
unchanged. Logs: `/tmp/pr2761-fifth-hook-inventory.log` and `.json`.
