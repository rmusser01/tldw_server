# WebUI memory fixes (TASK-13450)

The approved scope is all seven findings from the Chrome memory review, with a PR against dev. The reported 30 GB peak has not been reproduced; regression checks exercise the resource lifetimes and allocation paths that can cause growth.

- Replace the duplicated quadratic LCS matrix with the installed diff library, bounded edit search and input sampling. Share the implementation with the worker. Main-thread comparisons allow 1,024 edits / 100 ms; workers allow 2,048 edits / 1,000 ms. If the budget runs out, preserve equal leading/trailing lines around the coarse changed middle. Retired comparisons abort and terminate their worker; closed modals do not compute.
- Give the TTS playground ownership of current and in-progress Blob URLs. Replacement, clearing and unmount cancel synthesis and release URLs. Document TTS uses the same request retirement rule so Stop, replacement and unmount prevent late playback.
- Reuse installed React Virtual for continuous PDF pages, preserving per-page dimensions, navigation, text selection and notes. Thumbnail canvases mount only near the viewport. Avoid canvas creation proportional to document length.
- Preserve internal PDF links and low-zoom final-page navigation. Reapply active search highlights when a remounted text layer finishes rendering, without pulling the reader back during scrolling.
- Retire speech WebSockets and their queued callbacks on Stop, replacement and unmount; prevent the shared streaming player from accepting audio after teardown.
- Keep existing authenticated media transport. Limit embedded previews to 64 MiB before/during body consumption, abort retired downloads, and release playback resources. This deliberately limits large-file previews rather than exposing credentials in a media URL; authenticated native streaming can replace the cap when available.
- Bound shared chat and character caches by bytes and entries, sweep expired entries during access/insertion, and schedule expiry so idle payloads are released. Preserve request-scoped cache bypass and in-flight deduplication.

No new dependencies, auth changes or edits to the existing UAT checkout. Tests cover small valid results, cancellation despite transports ignoring abort, replacement ordering, URL cleanup, bounded PDF page counts, missing/false Content-Length and cache expiry/capacity. The PR remains subject to the human-written Change summary merge gate.

## Verification

- Final affected Vitest run: 117 files, 1,087 passing tests and seven failures. All seven failures are in the unchanged characters-list-all suite and were reproduced independently on clean dev. New regressions pass. A repository-wide UI run was stopped after about 50 minutes without a complete report.
- Production frontend: `bun run typecheck --incremental false` passes. Touched-scope ESLint: zero errors and no new warnings compared with dev; existing warnings remain. `git diff --check` passes.
- Native Chrome, actual React-PDF/React Virtual components, synthetic 1,000-page mixed-size PDF: 24 immediate jumps between pages 500 and 1,000 retained the requested page with at most five page canvases mounted. Internal links, low zoom and lazy thumbnails were also exercised. Peak observed JavaScript heap in the final jump loop was approximately 44 MiB; this excludes native/GPU memory and is not a UAT heap profile.
- Bandit was run from the project virtual environment on the touched TypeScript scope: zero findings, zero Python files scanned. It provides no TypeScript security coverage.
- Read-only code review found no remaining important issues. Initial publication was rebased onto dev's backend Claims update.

## PR review follow-up

Qodo's scroll and comparison findings were reproduced with failing tests before correction. PDF navigation now uses native scrolling for fixed page offsets, avoiding virtualizer retries that fight input. Scroll-reported pages do not trigger navigation, pre-metadata reader movement is preserved, and explicit destinations retain priority. Ordinary revisions with more than 128 changed lines retain useful exact diffs; bounded fallbacks keep common edges. New helper signatures have explicit return types.

After rebasing onto dev `5775d3fbbe`, all 23 PDF/diff memory regressions pass. The affected run has 246 passing tests across 52 files and one Markdown presentation failure reproduced on clean latest dev. Final production frontend typecheck and touched-scope lint pass; lint retains one unchanged warning. Chrome retains manual scroll position and repeated distant-page navigation with five or six nearby canvases. Independent review found no remaining important issues.

ADR required: no new ADR for the review follow-up. It repairs behavior within existing WebUI dependencies, APIs, authentication and resource lifetimes. ADR-059 governs task editing; ADR-006 governs security validation.
