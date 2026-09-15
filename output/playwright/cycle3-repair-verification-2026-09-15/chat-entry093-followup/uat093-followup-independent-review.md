# TASK13260.33 / UAT093 native-race follow-up — independent review clear

No remaining material issue found in the six source/test paths frozen at **2026-09-15T22:11:17.189Z**. All six source/test files, five retained probe configs and five report/compatibility artifacts match `/private/tmp/uat093-followup-frozen-manifest.json` (16 hashes; no mismatch). No repository files, browser/runtime, commits or global documentation changed during this review.

## Assessment

- Validated canonical character/persona identity is selected before metadata readiness, so the prior character cannot be mistaken for a deliberate new picker choice while profile enrichment is delayed.
- The existing synchronous selection-operation revision guards both initial restoration and deferred profile writes. A newer picker action wins before React rerenders, including a picker roundtrip and profile completion in the same turn.
- The initial follow-up still had a shared-loader race: a second metadata response could publish readiness while the first loader's selection was queued/persisting. The final correction waits for stable settlement of the existing commit chain and then rechecks the original controller/owner. Its loop also follows a newer queued picker rather than treating one captured promise as complete. No new queue or state layer was added.
- Messages become ready independently of optional profile enrichment, while the original scope/controller remains alive until enrichment settles. Existing requestScope/signal propagation and replacement, principal-invalidation and unmount checks remain intact.
- The per-mounted-route consumed flag prevents accepted settings-return targets from being reapplied after deliberate selection. It matches the existing once-captured settings context; failed/cancelled local returns remain unaccepted. No new auth request, offline prerequisite or legacy local-history ownership claim was introduced.

I independently checked the installed Plasmo storage hook: its setter awaits storage.set before publishing its own render state. The permanent held-storage test therefore exercises a real timing boundary, using the actual Playground, option store, loader, effective-assistant resolver and useSelectedAssistant with controlled storage/HTTP boundaries.

## Fresh independent verification

All commands ran from `apps/packages/ui` using existing Vitest, `--maxWorkers=1`; the nine-suite run also used `--no-file-parallelism`.

- **128 passed /9 suites**, `/private/tmp/uat093-followup-independent-focused.log`: Playground coordinator; local conversation loader; Playground session persistence; server-loader pure, scope and mirror integration suites; selected-assistant persistence; assistant overlay; Notes backlink labels.
- Original retained metadata selection, transient-clear and held-messages probes: **1 passed each**. Logs `/private/tmp/uat093-followup-independent-{selection,clear,messages}.log`.
- Original concurrent metadata probe: **2 passed**, `/private/tmp/uat093-followup-independent-concurrent.log`.
- Additional independent StrictMode controls: **6 passed**, `/private/tmp/uat093-followup-independent-strict.log`, via `/private/tmp/uat093-followup-strict.config.ts`. These run the current real coordinator cases with delayed profile/messages, consumed-route/new-picker timing, and shared held storage with/without a later queued picker under React.StrictMode.
- Scoped `git diff --check`: exit0.

The original timing configs remain unchanged. Their explicit `*-compat.config.mjs` runners are not represented as unchanged harnesses: the first three supply the retained prior coordinator fixture plus the additive revision/wait mock contract; the concurrent runner restores only the Form stub expected by the original transform, which still inserts two real loaders and releases held metadata together. The current permanent tests use the actual selection hook and verify the stronger held-storage boundary. The original held-messages trace was observational and eventually recovered; it is not described as an originally failing assertion.

Hash evidence: `/private/tmp/uat093-followup-independent-hashes.json`.

## Other verification and limits

I inspected the implementer's ESLint JSON: all six files are covered (none ignored), zero errors, 23 preexisting warnings. Its comparison reports no added/removed warning signatures. Its compiler result reports the exact existing 90-diagnostic merged baseline, not a clean typecheck; I did not repeat the full compiler. Parent owns the final combined compiler checkpoint.

Native saved Robot entry, actual Cedar backlink and settled reload acceptance remain parent-owned and are not inferred from these tests. No live browser, native IndexedDB, service worker or backend was exercised in this independent review. The earlier native failure and initial shared-loader RED evidence remain relevant history; this source review clears the corrected freeze for native verification.
