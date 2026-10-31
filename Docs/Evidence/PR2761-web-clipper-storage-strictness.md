# PR2761 web clipper storage strictness

Tracking: TASK-12116. This is one incremental strict boundary, not completion of
whole-WebUI strictness.

The handoff adapter checked optional extension-storage methods before entering a
Promise executor. TypeScript does not preserve that mutable property narrowing
inside the callback. The original handoff produced TS2722 and TS18048 for each
of `storage.get`, `storage.set`, and `storage.remove`: six diagnostics.

The existing extension read/write/remove implementation now lives in
`apps/packages/ui/src/services/web-clipper/extension-storage.ts`. Each method is
captured in a constant and invoked with its storage receiver through `.call`.
The module describes only the optional Chrome storage/runtime properties it uses;
it does not depend on or modify the shared ambient `any` declaration. Public
handoff APIs, session-before-local preference, browser fallbacks, runtime-error
logging, tombstones, expiry, and callback/Promise settlement behavior are retained.

The required `tsconfig.strict.json` includes this complete runtime adapter. The
same six errors were reproduced after extraction and before the narrowing fix;
the required project passes after the fix. No compiler flag was relaxed.

Verification:

- Existing handoff tests plus six new characterization cases passed **12 tests**
  before extraction and after the fix. An additional conflicting callback/Promise
  case brings final coverage to **13 passing tests**. Cases cover receiver-sensitive
  callback, Promise and dual APIs; missing methods; synchronous throws; Promise
  rejection; runtime errors; local storage fallback; tombstones; and stale handoffs.
- Required strict project: exit 0, zero diagnostics.
- Scoped ESLint on both production modules and the handoff test: exit 0, no file
  diagnostics. The shared configuration emits its existing Next pages-directory
  discovery warning when run from `apps`.
- `git diff --check`: exit 0.
- Bandit was invoked from the repository virtual environment. It cannot parse
  either TypeScript production file (two AST parser errors); this is not a
  successful TypeScript security scan.

The larger handoff import graph is intentionally not claimed as strict-clean:
`PendingClipDraft` transitively loads screenshot capture and HTML extraction.
A focused compile still reports two implicit-any screenshot callback parameters
and a missing Turndown declaration. The six handoff adapter errors are gone.

Logs: `/tmp/pr2761-handoff-strict-red.log`,
`/tmp/pr2761-extension-storage-strict-red.log`,
`/tmp/pr2761-extension-storage-strict-final.log`,
`/tmp/pr2761-handoff-strict-after.log`,
`/tmp/pr2761-handoff-tests-before.log`,
`/tmp/pr2761-handoff-tests-final.log`, `/tmp/pr2761-handoff-lint.log`, and
`/tmp/pr2761-handoff-bandit.json`.
