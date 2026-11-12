# Complete Media UX repair and enhancement design

Approved scope: TASK-13450 audit plus every additional issue and potential improvement. The requester approved all items equally and delegated ordering on 2026-10-04. Coordinator TASK-13500; baseline dev 75ab224081bf140ef52017c1a9b0a04f6878d488; implementation branch codex/media-ux-complete-20261004.

## Product direction

Preserve the existing research-library identity and shared WebUI/extension components. Repair task transitions using the current staged wizard, owned session/runtime, Media Inspector, multi-review, collections and job APIs. No additional ingestion UI or new dependency. Order by dependency: queue → results/recovery → review → Inspector → recent imports and integrated verification.

## ADR check

ADR required: no new ADR. These changes extend local behavior within existing shared UI and owner-fenced ingestion-session persistence, without changing public APIs, worker ownership, authentication, storage authority or dependency policy. Governing decisions: ADR-008 for browser state, ADR-022 for media processing ownership, ADR-059 for canonical Backlog editing. Recent-import records are bounded metadata in the existing session store, not a new backend/history service.

## Global constraints

- Frontend behavior must be identical in WebUI and packaged extension except browser-only capture controls.
- Preserve server/principal authority fencing, stale-operation rejection, cancellation and session recovery. Never persist credentials, File objects or full extracted content in import history.
- Reuse React, TypeScript, Vitest, existing Playwright and installed UI primitives. No new package.
- Keep the current surface colors, typography, density and unrelated functionality.
- All user-facing text uses i18n keys with accurate English defaults and the shipped English locale updated. Do not preserve a known-wrong translated instruction by allowing its key to override corrected behavior; update affected existing locale entries coherently.
- Replacement is an explicit permission independent of Quick/Standard/Deep processing depth.
- Reading fetch/render is bounded to 30 items at once. This is a reading window, not a limit on bulk metadata selection.
- Knowledge readiness is asserted only from affirmative server/result evidence. Settings that request indexing are not proof of readiness.
- Every audit issue and enhancement must map to behavior, a check, and final documentation. Tests exercise behavior; no source-text assertion tests added for normal UI logic.

## 1. Source entry and a dependable queue — TASK-13500.1

The opening contract already carries `source` (origin) and `url` (content). Empty Inspector and multi-review use `{ source: 'manual', url: value }`. Ordinary HTTP(S) URL openings must seed the active wizard once; playlist preflight remains distinct. Applying a new source to an existing draft preserves prior queue entries and settings. Opening a running session must not mutate submitted inputs.

Parsing accepts one URL per line and clearly separated pasted URLs, including comma+whitespace followed by a new HTTP(S) scheme. A comma inside one URL is retained. Combined ambiguous input receives a corrective validation message rather than becoming one apparently valid source. The visible example is one URL per line.

Queue validation, Configure count, Review list and submitted items use the same eligible selection. Invalid inputs, excluded conference items and default duplicate exclusions have explicit reasons. A duplicate is skipped by default; an explicit per-item process-again action permits deliberate repetition. File name/size duplicates follow the existing dedupe definition. Review and Results account for excluded entries so every source has an outcome.

Deep defaults do not turn on overwrite. Switching processing presets preserves an explicit current replacement choice. Duplicate recovery text refers to the replacement control rather than recommending Deep as permission to overwrite. Configure describes settings for this run and separately identifies saved defaults for future runs.

Add supports a labeled extension-only Capture current tab action using the existing browser tab/URL handoff, HTTP(S) validation and playlist builder. It shows the captured URL, reports inaccessible/internal tabs with a corrective message, and does not query browser tabs on WebUI.

## 2. Recovery and saved-batch continuation — TASK-13500.2

Wire existing Results per-item and Retry All callbacks. Requeue only requested retryable failures. Preserve successful outcomes and original source-specific options. File retries require an attached valid File or an explicit reattach action after reload. Ownership changes prevent old callbacks or retries from acting on a different account/server. Retrying does not silently repeat already successful jobs.

Results presents a compact reconciling summary: added, excluded/skipped, succeeded/saved, failed and cancelled, with reasons linked to rows. A successful saved item remains openable through retries. Review these N saved items opens the existing multi-review route with unique saved media IDs, excluding errors, unsaved analysis and invalid queue entries. Reuse the existing review selection setting and route; publish the owned IDs together in one versioned snapshot, with the raw legacy IDs only a compatibility mirror. Do not add a parallel viewer.

Show Saved, Processing and Ready for Knowledge when supported by authoritative outcomes. Without indexing confirmation, say readiness is unconfirmed or indexing requested; never imply readiness from the chosen preset. Distinguish Review extracted content before saving from Review saved items. Preserve durable collection handoffs and per-item Open in Media.

## 3. Predictable multi-review — TASK-13500.3

Click and Enter preview the same item. Selection uses an accessible, named, tab-reachable checkbox; Space on the checkbox selects, with existing Shift range behavior retained. Copy describes these actual actions. Selecting a review set does not erase preview context, and preview content is visibly distinguished from the selected reading set.

The active content title and navigation position use the active context: result preview or ordered selected IDs. Selected navigation is independent of the current search page and handles IDs present only on earlier pages. No Focus(0/N), mismatched item number or No item selected when content is displayed. Mobile preview or starting selected review moves to Content with Back to results. Checkbox toggles keep Results visible while assembling a batch.

Bulk selection may exceed 30. Only the active 30-item reading window is fetched/rendered; windows can be navigated without dropping selected IDs. Show selected total separately from reading-window count and its limit. Batch tags/trash/export/reprocessing act on the full explicit selection, with existing safeguards and bounded on-demand detail work. Do not prefetch full content for every checkbox selection.

Group viewer controls by Reading, Layout and Selection actions. Keep core reading navigation visible and less frequent controls in the existing Options/menu. Clarify Side-by-side layout versus Compare content; preserve existing modes and keyboard shortcuts. Selection/action counts remain visible on desktop and mobile.

## 4. Inspector and accessible bulk actions — TASK-13500.4

Persist Inspector selected IDs across result pagination and search changes while remaining in the same owner. Keep cached selected metadata or retrieve selected details on action; do not prune against the current visible page. Clear selections on owner change. Label Select this page; show total selected and explicit clear. Open selection uses all selected saved Media IDs, with existing Notes handling preserved.

On narrow screens, Results and Content are explicit views. Opening a result shows Content; Back to results preserves search, pagination and selection. Bulk mode keeps Results available and a persistent selection-action bar or accessible drawer outside the shrinking search/filter scroll region. At 390×844, users can see the selected count and primary action without hunting through nested scrolling. Large text and landscape layouts also remain usable.

Sort/export selectors and selection checkboxes have persistent accessible names. Enlarge small touch hit regions without inflating desktop reading density. Keyboard focus survives opening/closing drawers and returning to results.

Bulk deletion is Move N items to trash, with the existing confirmation/recovery pattern. Retain partial-result reporting and selection of failed items; offer Undo where the API supports safe restoration or Open Trash. No permanent-delete semantics added. Display a labeled Add media action in the Inspector and contextual guidance for adding multiple sources and reviewing a batch, dismissible through the existing hint mechanisms.

## 5. Recent imports and complete workflow verification — TASK-13500.5

The existing jobs listing requires a batch ID. Do not invent an unfiltered server API. Extend the existing owner-fenced session store with at most 10 recent-import metadata records, containing session ID, verified authority, timestamps, recognizable source label/count, lifecycle, known batch/job IDs and successful saved media IDs. No credentials, File data or extracted content. Archive a summary when completing/replacing a session. Filter every display/read/action to the current verified authority and test account/server transitions.

Replace raw batch-ID-first monitoring with Recent imports. The current owned session appears from submission, supports existing resume/minimize, and shows live progress. Previous known batches can refresh job status using their stored IDs. A finished summary opens its saved review set. Raw IDs are optional diagnostics, never necessary for the main task. Poll only relevant active/expanded data, release timers and ignore stale owner results.

Contextual batch guidance, distinct saved/index states, compact result summaries, extension capture and bounded reading selection are enhancements delivered in stages above; this stage verifies their complete connection. Documentation explains source entry, duplicate/replacement choices, retry/reattach, saved versus indexed, pre-storage versus saved review, selection scope, 30-item reading windows and recent imports.

## Coverage and verification

| Accepted item                          | Stage | Required behavior check                                                               |
| -------------------------------------- | ----- | ------------------------------------------------------------------------------------- |
| Empty URL handoff                      | 1     | Fresh and restored draft URL survives click/Enter in both callers                     |
| Comma parsing                          | 1     | Two URLs split; legitimate comma URL remains one                                      |
| Invalid/duplicate accounting           | 1–2   | One eligible count and an explained outcome for every input                           |
| Retry                                  | 2     | Retry failures only; prior successes and options persist                              |
| Saved-batch handoff                    | 2     | Correct unique saved IDs reach multi-review                                           |
| Deep overwrite coupling/configure copy | 1     | Preset cannot silently authorize replacement; settings describe this run              |
| Preview/keyboard/navigation            | 3     | Pointer and keyboard agree; title/position/Prev/Next match content across pages       |
| Larger metadata sets/reading cap       | 3     | 40 selected; at most 30 detail fetch/render; window navigation retains all selections |
| Grouped review controls                | 3     | Reading/layout/selection controls remain findable with correct labels                 |
| Inspector scope                        | 4     | Cross-page four-item set survives and bulk actions receive all four                   |
| Mobile reading/bulk visibility         | 4     | 390×844 shows content or selected count/primary action clearly                        |
| Trash recovery                         | 4     | Cancel deletes nothing; confirmed partial outcomes retain failures and offer recovery |
| Accessibility                          | 3–4   | Named keyboard controls, clear focus and practical mobile hit areas                   |
| Active-tab capture                     | 1     | Packaged extension capture queues HTTP(S) and handles restricted tabs                 |
| Recent imports                         | 5     | Submission/reload/resume/history and owner transitions are safe without manual IDs    |
| Contextual guidance/status/summary     | 2,4,5 | Accurate displayed state and direct workflow continuation                             |

Use red–green unit/integration tests around shared functions and mounted components. Run the relevant existing Media/Quick Ingest family once per completed stage, typecheck/format checks on changed scope, and both platform builds at integration. Perform one batched browser pass covering desktop/mobile, single/mixed imports, failed retry, across-page selection, preview, saved batch and packaged extension; repair findings together, then one confirmation pass. Real backend checks use isolated safe content and must not mutate an existing user's library. Never describe simulated processing as real ML reliability.
