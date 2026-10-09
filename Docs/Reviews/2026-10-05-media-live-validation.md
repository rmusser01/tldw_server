# Media live validation against latest dev — 2026-10-05

Real ingestion, speech transcription, local analysis, vector storage/retrieval, failed-only execution and native Notes recovery passed. The UX is **not fully cleared**: three presentation gaps and one minor copy issue remain tracked below. VoiceOver speech and novice participant comprehension remain unverified.

Follow-up: all four tracked findings are fixed and verified in [the 2026-10-06 report](2026-10-06-media-live-ux-fixes.md). The observations below remain the historical validation of their stated source.

## Source and scope

- Validated dev: `7226596cca5b875ad6e121a15133f69600152258`, verified against remote dev before execution and again during final validation. Media PR [3194](https://github.com/rmusser01/tldw_server/pull/3194) was already merged at `587cd8e9fe3b42eba451b83c1ba690607035eace`; subsequent dev changes include Knowledge actions and shared ingestion/review integration.
- Tracking: TASK-13503. Worktree branch: `codex/media-live-validation-20261005`.
- No application source, dependency, user credential or existing library was changed. This branch contains validation evidence and follow-up tracking only.
- ADR required: **no**. This executes existing processing, persistence and provider behavior. [ADR-059](../ADR/059-backlog-py-task-editor-cutover.md) governs task editing; existing egress and embedding pipeline rules remain in force.
- This was a live WebUI/API validation plus a current Chrome extension production build. Packaged-extension capture gestures were not repeated on this revision; earlier packaged evidence retains its source boundary in [the prior verification report](2026-10-04-media-ux-complete-verification.md).

## Actual outcomes

| Workflow | Observed result | Evidence boundary |
|---|---|---|
| First-use authentication | Unconfigured Media directs to Settings; entering the throwaway API key through Settings enables the actual library. | No authentication fixture or request interception. |
| Single Markdown | Durable ingest job 1 completed; media 1 retained marker `cobalt-heron-742`, title and content. | The separate process-only document endpoint correctly returned no saved ID; persistence was proved through `/media/ingest/jobs`. |
| Mixed import | Two Markdown files, a real `https://example.com` URL, its duplicate and a corrupt PDF produced **5 added / 1 excluded / 3 succeeded and saved / 1 failed / 0 cancelled**. Saved IDs were exactly 3, 4 and 5. | Real extraction and persistence; the invalid PDF was an intentional failure. |
| Saved-set review | Review these saved items selected exactly 3/4/5, excluding earlier imports. Content loaded, the reading footer showed 1 of 3, and 390×844 reading offered Content/Results and Back to results. | Real nested API payload exposed the title/metadata issues below. |
| Resumed import | After navigating/reloading, Recent imports restored all original outcomes. A different-sized replacement file was rejected; reattaching the same original name/size enabled Correct settings. | Original-file identity protection passed. A corrected file is a new source, not an interchangeable original handle. |
| Failed-only execution | Review misleadingly claimed 2 eligible items, but the actual retry accepted **one** job, ID 10, for `recoverable.pdf`; the three prior successes stayed saved. | Execution scope passed; confirmation wording/list state failed. |
| Corrected PDF | A valid PDF in a fresh import completed in 2 seconds as media 7. Actual saved text contained `teal-ibis-286`. | Missing-provider guard first blocked the default analysis run and focused provider choice. Choosing Quick then completed real extraction. |
| Speech transcription | A synthesized 11-second WAV ran through real `whisper-tiny`; media 2 retained the spoken blue-heron sentence and validation number 742 with timestamps. | No canned transcript. A local-availability warning remained even though the managed cache was populated; this run does not establish all model-cache detection behavior. |
| Local analysis | Explicitly loaded `mlx-community/Qwen3-0.6B-4bit` generated and persisted analysis for distinct media 6, mentioning marker `red-kite-573`. | Genuine local inference. The small model produced repetitive output and did not follow the two-sentence prompt; summary quality is not cleared. No commercial provider was configured or exercised. |
| Vector indexing and retrieval | The actual Redis Streams worker generated and stored one 384-dimensional MiniLM embedding for media 1. Status became `has_embeddings=true`; vector and FTS searches both returned the saved item and matching marker. | Actual model weights and Chroma storage, not enqueue-only or stub evidence. This small corpus does not establish ranking quality at scale. |
| Native Notes recovery | Two original Notes moved to Trash through the Notes UI; deletion survived reload; each Restore returned the same ID/title/content. API versions advanced **1 → 2 → 3** and final reload showed both originals. | Actual Notes endpoints and persistence. No recreation masquerading as restore. |
| Keyboard and native accessibility | `j` moved reading 1 → 2; `/` focused the named search field; typing `j` inside search left reading at 2. Native Chrome AX exposed named selections, review action and navigation. | These are keyboard/AX observations, not proof of spoken announcements. |

## Remaining findings and solutions

Severity describes workflow impact; these observations do not establish which historical commit introduced them. Heuristic mapping uses [NN/g’s usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/).

| Task | Severity and issue | User consequence | Proposed solution |
|---|---|---|---|
| TASK-13504 | P2: reading cards and Open items fall back to `Media 3/4/5`, while the list/header show actual source titles. | Users must remember IDs to relate reading panes to their selected sources; recognition and consistency suffer. | Normalize real `source.title` / `source.type` at the shared detail boundary, retain flat-payload compatibility, and use that identity in reading/comparison/export even across list pages. |
| TASK-13505 | P2: web reading begins with raw `[METADATA]` JSON containing hashes and pipeline fields. | Technical scaffolding occupies the mobile reading viewport before the article; content hierarchy and minimalist presentation suffer. | Present extracted article text and readable provenance by default, while preserving raw stored content for deliberate inspection/export. Safely recognize the existing envelope. |
| TASK-13506 | P2: correction Configure/Review says 2 items and lists an already successful URL as ready; saved files appear Invalid because their File handles are absent. Actual retry submits only the failed PDF. | Confirmation does not describe what the action will do, weakening system-status visibility and confidence in recovery. | Derive displayed eligibility/status from the same failed-only target set used by processing; retain prior saved outcomes and exclusions as accurate context. |
| TASK-13507 | P3: one saved item offers `Review these 1 saved items`. | The common single-source action has inconsistent visible and accessible wording. | Apply the existing count-aware translation convention for singular/plural review actions. |

The title defect traces to flat-only enrichment in `Review/hooks/useMediaReviewActions.tsx`; the real detail DTO nests identity under `source`. The metadata extractor returns `content.text` unchanged. The correction execution already filters `retryIds`; Configure and Review independently calculate general queue eligibility, so this finding is a presentation mismatch, **not** a successful-source resubmission defect.

## Reproduction and verification

Temporary state: `/private/tmp/media-live-validation-20261005`. Dedicated WebUI/API/Redis ports: **18881 / 18882 / 18883**. Auth/media/Notes/jobs/vector paths were isolated. Existing installed dependencies and the original project venv were reused read-only; actual Whisper, MiniLM and MLX weights were prepared in task-owned locations. Redis persistence was disabled.

API configuration used `TLDW_CONFIG_FILE`, `AUTH_MODE=single_user`, a throwaway key, isolated `DATABASE_URL` / `USER_DB_BASE_DIR` / `JOBS_DB_PATH`, enabled ingest workers, and `ALLOWED_ORIGINS` for the test WebUI. `MINIMAL_TEST_APP=1` selects a reduced router set; **`TEST_MODE=false`, `TESTING=false`**. Processing remained real. Notes task/activity 404s came from omitted ancillary routers; they did not affect restore. This is not a full production-server validation.

Vector execution used the current pipeline, `REDIS_URL=redis://127.0.0.1:18883/0`, `EMBEDDINGS_REDIS_ALLOW_STUB=false`, and:

```sh
python -m tldw_Server_API.app.core.Embeddings.services.redis_worker --stage all
```

The initial embedding setup mistakenly ran the legacy Jobs worker against the pipeline root and that root failed as unsupported. The final independent job used the correct real Redis worker and completed. The initial local-analysis attempt lacked a loaded MLX model and warned; loading through `/llm/providers/mlx/load` and using a distinct source then produced the successful persisted result. These setup attempts are recorded separately from product findings.

Existing affected tests: **5 files / 288 tests passed, 34.47 s**, from `apps/packages/ui`:

```sh
node node_modules/vitest/vitest.mjs run \
  src/components/Review/__tests__/MediaReviewPage.reading-context.test.tsx \
  src/components/Review/__tests__/MediaReviewPage.stage6.keyboard-scope.test.tsx \
  src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  src/components/Common/QuickIngest/__tests__/WizardResultsStep.navigation.test.tsx \
  src/utils/__tests__/media-detail-content.test.ts \
  --maxWorkers=1 --no-file-parallelism --testTimeout=15000
```

The first run and an isolated file rerun timed out one 40-item footer test at the local default 5000 ms. The final run used the repository CI’s already established 15000 ms timeout; no source/test timeout was edited or test disabled. Local Node was v26.0.0; CI specifies Node 20, so this is scoped local evidence, not a full CI equivalence claim.

Chrome extension production build: `bun run build:chrome:prod` from `apps/extension`, **passed in 37.7 s**, output 49.25 MB. Directly invoking the underlying Node script first lacked the npm-script WXT PATH; the standard package command passed. No WebUI production rebuild, full backend suite, or all-browser extension matrix was run for this validation-only branch. Bandit is not applicable to documentation/task/evidence-only changes; no Python source changed.

## Accessibility and human-study limits

The user explicitly approved a brief foreground VoiceOver check. Native Chrome AX snapshots and navigation were exercised. VoiceOver was launched and the native toggle/navigation attempted, but reading its last spoken phrase returned `-1728`; no verifiable screen-reader speech was captured. Therefore spoken status, focus announcements and the full VoiceOver journey remain **unconfirmed**. VoiceOver was stopped and the original ChatGPT foreground restored. Automated DOM/AX checks must not be described as a screen-reader pass.

An additional live mobile axe scan was attempted. Its initial readiness locator targeted a desktop-only action; the corrected mobile locator worked, but loading the scanner was blocked by the application CSP. No CSP relaxation was made and no live axe pass is claimed. Existing mounted accessibility checks are included in the passing scoped test run.

No novice or experienced human participant study occurred. A participant must still demonstrate comprehension of Saved versus Knowledge readiness, exclusions, retry scope and selected reading. This agent walkthrough provides functional and heuristic evidence, not human success rates or time-on-task measurements.

## Durable evidence and cleanup

Selected real responses, UI snapshots, keyboard observations, native AX observations and test/build results are consolidated in [receipts.json](../../output/playwright/media-live-validation-20261005/receipts.json). Screenshots:

- [Real mixed results](../../output/playwright/media-live-validation-20261005/mixed-real-results.png)
- [Real mobile reading with title/metadata findings](../../output/playwright/media-live-validation-20261005/saved-batch-mobile.png)
- [Misleading correction confirmation](../../output/playwright/media-live-validation-20261005/correction-review.png)
- [Actual corrected-PDF success and singular copy](../../output/playwright/media-live-validation-20261005/corrected-real-results.png)

Final cleanup passed: both task browsers and all task servers/workers were stopped; ports 18881/18882/18883 had no listeners. Original checkout porcelain bytes matched the before snapshot exactly. VoiceOver was absent and ChatGPT was foreground. Managed model weights were preserved in the temporary validation directory; task dependency symlinks and the completed temporary plan were removed. Receipts and TASK-13503 record these checks. The original checkout remains on its existing branch; this validation branch retains the report and four open follow-up tasks. The earlier Media implementation worktree remains archived.
