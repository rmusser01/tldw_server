# E4 temporary consumer qualification

These fixtures load `browser-qualification.ts` in actual consumer bundles without
adopting the service in Notes. Copy the Next fixture to
`apps/tldw-frontend/pages/e4-worker-qualification.tsx`; copy the two extension
fixtures to `apps/extension/entrypoints/e4-worker-qualification/{main.ts,index.html}`.
Remove those temporary entries afterward. Never ship a qualification route.

Next uses the existing `dev:webpack -- --port 3182` script, a dedicated
`TLDW_NEXT_DIST_DIR`, and a synthetic unavailable backend. The browser harness
narrows worker-src to self, aborts the unrelated runtime-config request and synthetic
backend health request, and leaves ServerReadinessGate refusing. This qualifies
module/worker loading only. Default Turbopack remains unqualified with self-only
worker CSP (its loader generates a Blob URL).

Build WXT with `bun x wxt build -b chrome -e e4-worker-qualification -e background
-e options -e sidepanel`; the existing token verification runs normally. Load the
resulting local unpacked extension in a fresh Chromium profile. The fixture uses
WXT's native `?worker` imports through the trusted factory seam.

Call `globalThis.e4Qualify()` and observe dedicated browser targets and their
actual destruction. Native terminate is asynchronous: do not assert immediate
zero physical threads, and do not use a product sleep as cleanup. The qualified
run observed all25 created targets eventually destroyed, including6 startup-
retirement churn workers, while80ms UI ticks remained below83ms. The two-second
request deadline initiates termination and retires logical authority. See ADR070
and the task10 report for exact raw scripts, target timelines, hashes and limits.
