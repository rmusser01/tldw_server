# E4 worker ownership qualification —2026-10-11

Task implementation remains In Progress for independent controller review. Full report,
exact failed/successful scripts, commands, outputs, target timelines and SHA256 index:
`/Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/.superpowers/sdd/2026-09-27-structured-document-editing/task-10-report.md`
and sibling `task-10-evidence/`.

The service/lease primitives own one executing snapshot and latest pending successor,
share check/format, debounce400ms, permanently retire old authority and reject unsafe
source before dispatch. Results are typed/bounded and source-safe. They do not save,
apply history or wire an editor. I2/I3 must invoke retirement on every authority,
editability, conflict and activation transition and apply through I1's captured-source
seam. Shared fixture SHA256 is unchanged:
`fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7`.

ADR070 records accepted Rulings39–42: fixed isolated Python counterpart; trusted browser
`() => Worker` factory; supported Next webpack qualification; and native asynchronous
Worker.terminate semantics. WXT0.20.13 requires its native `?worker` constructor because
its import.meta.url rewrite otherwise points at missing extension-root worker.ts.
Default Next16.1.4 Turbopack emits a Blob/importScripts bootstrap rejected by self-only
worker CSP. That failure remains an I3/I5 production/default-build gate.

Temporary Next module-loading qualification used the existing dev:webpack script,
self-only worker CSP, synthetic unavailable backend and an aborted unrelated runtime-
config request. The actual readiness guard stayed refusing. This is not authenticated
Notes or complete application startup. Temporary entrypoints were removed; their exact
copies remain under document-editing/__tests__/consumer-fixtures. The actual installed
WXT extension used native local worker assets and normal token guards.

Chromium145.0.7632.6 created25 and eventually destroyed25 dedicated workers in each
consumer. Positive, stalled/crashed/malformed/init/cancel/close/replacement controls,
resource bounds,30pending navigation retirements and6actual startup-retirement churn
cycles passed.80ms UI ticks stayed80.9–81.9ms.1MiB admission/encoding/worker elapsed:
Next10.5/1.9/124.8ms, extension10.1/2.3/71.7ms. Native force-termination may retain
physical threads for about2s after logical retirement; exact target timelines prove
eventual destruction. No JS thread join, zero physical overlap, fixed product sleep,
CDP runtime privilege or heap cap is claimed. Persistent accumulation is not waived.

Targeted command from apps/tldw-frontend: `bun x vitest run` with the four shared
services/document-editing/__tests__ selections worker-client.test.ts, scheduler.test.ts,
json-document.test.ts and yaml-document.test.ts:148passed (7boundary/scheduler,
141engine). Unit mocks/fake timers cover boundaries/debounce only; actual browser
target lifecycle proof is separate. Scoped ESLint/Prettier and whole-web `tsc --noEmit`
pass, including the final churn fixture. No S Python changed, so server Bandit is N/A.
No full suite or production Next build was run. WXT qualification build passed its
normal token checks. I3 service/factory wiring and I5 production packaging remain gates.

## Coordinated review fix1

I2 fixed once in shared createCheckResult: zero-range refusals never enumerate source;
nonzero range validation counts only through the furthest required Unicode scalar
endpoint, without allocating a scalar array. Oversize/excerpt/unsupported Check and
Format refuse without Worker construction or source iteration/encoding. Emoji,
combining marks, CRLF, EOF and out-of-bounds controls remain valid.186targeted tests
cover all affected constructor callers; scoped ESLint/Prettier and whole-web types
exit0 with exact invocation/output/status records under task10 evidence/fix1. Original
silent exits remain Cannot Verify; the old148-record index is unchanged. Seven Node
localStorage notices and the ESLint pages notice remain visible, not proven inherited.
No browser build/lifecycle/benchmark replay or engine rewrite. ADR070/R39–R42 and
all seven integration/platform/evidence dispositions remain; task In Progress.
