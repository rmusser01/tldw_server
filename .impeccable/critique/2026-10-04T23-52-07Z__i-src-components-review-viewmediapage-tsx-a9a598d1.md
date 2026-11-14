---
target: "Latest dev Media WebUI and extension: novice and power-user single/batch ingestion and content review"
total_score: 22
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 5
target_identity: "file:/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/ViewMediaPage.tsx"
target_fingerprint: "sha256:502f0ac826180700fcd2130fde83bb8f5776b42aca46455128b77443668e91ce"
target_path: /Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/ViewMediaPage.tsx
timestamp: 2026-10-04T23-52-07Z
slug: i-src-components-review-viewmediapage-tsx-a9a598d1
---
Method: dual-agent (A: /root/media_design_review · B: /root/media_browser_evidence)

Reviewed the latest remote `dev` fetched for this audit: **75ab224081bf140ef52017c1a9b0a04f6878d488**, on October 4, 2026. [Reviewed commit](https://github.com/rmusser01/tldw_server/commit/75ab224081bf140ef52017c1a9b0a04f6878d488). Tracking: TASK-13450.

**Assessment:** Media has a strong research-tool foundation. Its biggest weaknesses are the transitions between adding, processing, opening, selecting, and reviewing sources. Repair those transitions before expanding functionality.

The review covered WebUI `/media`, `/media-multi`, the active five-stage Quick Ingest wizard, and built Chrome extension Media/options views. Desktop and 390–400 px layouts were inspected with pointer and keyboard interactions. Real latest-dev frontend code ran against safe fixture APIs. Successful and failed processing responses were simulated; real downloads, transcription, indexing, backend deduplication, and recovery were not established. This is expert inspection, not a study with novice participants.

**Design specificity and strengths**

The provenance, transcripts, analysis, collections, reading progress, versions, scoped chat, and content comparison make this recognizably a personal research library. The visual language is consistent. The opportunity is a more coherent research workflow, rather than a visual rebrand.

- The active **Add → Configure → Review → Processing → Results** wizard manages complexity with stages, presets, progressive disclosure, validation, progress, cancellation, and minimization.
- Single-item reading already supports content/analysis, metadata, tags, search within content, reading controls, and version history. Compact density, shortcuts, collections, batch operations, and multi-review selection exist.
- Connection recovery, failure classification/export, Trash, and single-item Undo provide useful foundations. Multi-review preserves selection across pages and enforces its 30-item reading cap with feedback.

**Walkthrough: what to do today and where it breaks**

Assume the server connection is configured. An empty library and an established library use the same underlying Media/wizard components.

| Task | First-time workflow | Experienced workflow | Friction observed |
|---|---|---|---|
| Add one item | Open Quick Ingest. Browse/drop one file, or paste one URL and choose Add URLs. Verify the queued item. Configure, review, process, then Open in Media. | Use existing defaults/preset and Use defaults & process after checking the queue. Minimize processing when needed. | Empty-library Ingest opens the wizard but loses the URL already typed. |
| Ingest multiple | Drop multiple files, or paste **one URL per line**. Verify source count/types. Configure the run, inspect Review, process, then inspect each outcome. | Use the same queue with saved defaults, type-specific settings, and collections. Review pre-storage drafts when that option is enabled. | Comma example, invalid/duplicate accounting, missing failed-item retry, and weak ordinary-batch continuation. |
| Review one saved item | Search/filter Media, select a result, inspect source metadata, then read Content and Analysis. | Use compact density, favorites, content search, versions, editing, scoped chat, and reprocessing. | On narrow screens the results pane can remain over the newly loaded content. |
| Review several | Enable Bulk, select items, and Open selection; alternatively open Multi-Item Review from an item's actions. Select explicit checkboxes, then use Compare, Focus, or Stack. | Build a collection/review set, filter across pages, inspect selected items, compare content, tag/export, and chat about the selection. | Inspector selection clears on pagination. Multi-review retains it, but pointer/keyboard behavior and displayed-item navigation disagree. |

For now, use newline-separated URLs and verify queue counts. In multi-review, use the explicit selection controls rather than trusting “Click to stack.” These are workarounds, not satisfactory final behavior.

In the extension, the built context-menu labels are **Save to Library** and **Analyze without Saving**. Their distinction is useful: the latter is not a saved Media item. The capture handlers were source-reviewed; real third-party context-menu capture was not exercised. The loaded extension's shared Media and wizard views were inspected. [Extension screenshot](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/extension-media.png).

Also distinguish **reviewing extracted content before storage** from **reviewing several saved items**. The current product supports both, but the word “review” alone does not explain which state the user is in.

**NN/G assessment**

The evaluation uses [Jakob Nielsen's ten usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/). The following **quality scores run from 0, poor, to 4, excellent**; they are diagnostic judgments, not an NN/G certification or measured usability percentage.

| # | Heuristic | Quality / 4 | Main finding |
|---|---|---:|---|
| 1 | Visibility of system status | 2 | Processing feedback is useful; preview/content and batch counts can disagree. |
| 2 | Match with users' language | 2 | Research concepts fit; ingest/chunking and several review models need explanation. |
| 3 | User control and freedom | 3 | Back, cancel, minimize, clear, Trash, and single-item Undo exist. |
| 4 | Consistency and standards | 2 | Pointer/keyboard activation, instructions, and selection scope differ. |
| 5 | Error prevention | 2 | Validation exists; comma examples and duplicate handling still invite errors. |
| 6 | Recognition rather than recall | 2 | Hidden actions and reconstructing ordinary batches create memory work. |
| 7 | Flexibility and efficiency | 3 | Strong presets, shortcuts, density, collections, and bulk functionality. |
| 8 | Aesthetic and minimalist design | 2 | Coherent styling; multi-review controls compete with reading. |
| 9 | Error recognition and recovery | 2 | Categorized outcomes/export exist; failed-item retry is unwired. |
| 10 | Help and documentation | 2 | Guidance exists, but several instructions describe different behavior. |
| | **Total** | **22/40** | **Acceptable foundation; significant interaction improvements needed.** |

Issue severity is separate: **S1 cosmetic, S2 minor, S3 major, S4 catastrophic**, estimated from frequency, impact, and persistence using [NN/G's severity framework](https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/). P1 indicates the first repair pass, P2 the next pass, and P3 polish. No S4 problem was demonstrated.

**Five priorities**

1. **[P1 · S3] Preserve URLs when opening the import wizard.**
   Both empty Inspector and empty multi-review accept a URL, then open an empty wizard. In a fresh Inspector, `https://example.com/fresh-single-source` → Ingest produces Queue 0. Inspector passes the URL as origin metadata; multi-review sends no value. This breaks a novice's first successful action and makes them repeat work.
   **Solution:** use one supported URL-opening contract in both callers and the mounted wizard. Queue/fill the validated URL before clearing the originating field. Avoid creating another import modal.
   **Acceptance:** click and Enter from either empty state preserve exactly one source, including in a fresh session; restored drafts retain their previous items.
   [Inspector caller](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Media/ResultsList.tsx:123) · [multi-review caller](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx:37) · [opening contract](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/utils/quick-ingest-open.ts:28) · [screenshot](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/empty-url-handoff.png). Suggested command: `$impeccable harden`.

2. **[P1 · S3] Make the batch URL instructions agree with parsing.**
   The displayed comma-separated example is unsafe. `https://example.com/research-one, https://example.com/research-two` becomes **one apparently valid web-page source** and enables processing. The parser only splits newlines.
   **Solution:** immediately change the example to one URL per line. Recognize unambiguous pasted URL boundaries or reject a combined multi-URL entry with a corrective message. Do not blindly split every comma; legitimate URLs can contain commas.
   **Acceptance:** the example produces two sources, or an explicit correction before processing; legitimate comma-containing URLs stay intact. Add, Configure, and Review show the same eligible count.
   [Parser](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx:305) · [English example](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/assets/locale/en/option.json:1161) · [screenshot](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/comma-url-queue.png). Suggested command: `$impeccable harden`.

3. **[P1 · S3] Finish failed-item recovery in the active wizard.**
   Results already supports Retry/Retry All, but the mounted wizard omits `onRetryItems`. Simulated failures offered Export failed list, Ingest More, and Done, leaving the user to rebuild the failed queue.
   **Solution:** connect the existing retry controls to failed items, preserving source inputs and settings. Give nonretryable failures a specific corrective action. Preserve successful outcomes.
   **Acceptance:** a mixed batch can retry one/all retryable failures without resubmitting successes; fixable configuration errors lead directly to the relevant setting.
   [Host wiring](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1791) · [retry gate](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx:496) · [failed-results screenshot](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/failed-ingest-results.png). Suggested command: `$impeccable harden`.

4. **[P1 · S3] Give preview, selection, and navigation distinct, consistent meanings.**
   The hint says “Click to stack,” but pointer click previews while Enter/Space selects. With items 1 and 2 selected, clicking item 3 leaves those two displayed while status says “Item 3 of 20,” “Previewing 3,” and Focus(0/2). Across-page sets can also show “No item selected” despite displaying selected content.
   **Solution:** pointer click and Enter should perform the same preview action. Use a named, keyboard-reachable checkbox to select. Keep preview visibly separate from the review set, or visibly switch the content being previewed. Navigate the active review set independently of the current search page. On mobile, preview should open Content with Back to results.
   **Acceptance:** pointer and keyboard activation agree; visible title, active-item count, selection count, and Prev/Next always describe the displayed content, including across pages.
   [Row behavior](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx:138) · [viewer predicate](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx:66) · [contradictory state](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/multi-preview-selection.png). Suggested command: `$impeccable clarify`, followed by `$impeccable harden`.

5. **[P1 · S3] Keep reading and bulk actions usable on narrow screens.**
   At 390×844, selecting a single item loads it behind the results pane; reaching content requires a 24 px icon-only collapse rail. Bulk controls have roughly 223 px of content squeezed into a 56 px scrolling region. Desktop bulk actions can also disappear below expanded filters while selected checkboxes remain visible.
   **Solution:** reuse the existing Results/Content mobile pattern for Inspector, open Content after selection, and provide Back to results. Give bulk selection a persistent count/action strip or labeled drawer outside the shrinking filter area.
   **Acceptance:** at 390×844 and desktop, opening a result makes its content visible; selecting an item exposes the count and primary bulk action without hunting through nested scroll regions.
   [Controls container](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/ViewMediaPage.tsx:1300) · [hidden reading](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/mobile-reading-hidden.png) · [compressed bulk controls](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/mobile-bulk-toolbar.png). Suggested command: `$impeccable adapt`.

**Additional issues and concrete improvements**

| Priority / severity | Issue and evidence | Solution and acceptance check |
|---|---|---|
| P2 / S3 | **Inspector drops page-one selections on pagination.** Two selected items become zero on page two; multi-review preserves them. [Pruning](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts:234); [before](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/bulk-page-one.png), [after](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/bulk-page-two.png). | Preserve IDs across pages; distinguish Select this page from Select all matching. Two page-one plus two page-two selections must produce a visible four-item set. |
| P2 / S2 | **Review counts invalid entries as items to process.** Add shows 3 valid/1 invalid; Review lists 4; simulated Results shows 3 successes and no skipped explanation. [Review rendering](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx:234); [screenshot](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/.impeccable/critique/media-dev-75ab224-evidence/batch-review-invalid.png). | Use the same eligible set throughout. Report submitted, skipped, succeeded, and failed separately; retain reasons for exclusions. Counts must reconcile. |
| P2 / S2 | **“Already queued” is only a warning.** The same URL generated two process-web-scraping requests. Backend deduplication remains unverified. [Validation](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx:180). | Skip duplicates by default; make intentional reprocessing explicit. One repeated source must generate one request unless opted in. |
| P2 / S2 | **Ordinary batches lack a direct saved-set review continuation.** Per-item Open Media exists; durable conference collections have their own handoff. [Results next steps](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx:380). | Add Review these N saved items using the existing multi-review selection contract. Open precisely the successful saved IDs. |
| P2 / S2 | **Deep processing enables overwrite_existing.** This couples processing depth with replacement semantics. It also enables review before storage, which provides a guard. Source-only risk: no overwrite was exercised. [Preset](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/presets.ts:47). | Separate replacement permission from processing depth. Switching presets should preserve the explicit replacement choice; Review must identify affected existing content. |
| P2 / S2 | **Inspector bulk deletion has weaker protection than related flows.** Source inspection shows direct deletion without the single-item Undo or multi-review confirmation path. Trash and partial-failure reporting already exist. [Bulk handler](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts:511). | Standardize Move N items to trash and confirmation/Undo or Open Trash. Verify the complete set and partial outcomes. |
| P2 / S2 | **Import monitoring exposes batch IDs and links the panel on completion.** The separate jobs panel starts collapsed and remembers a raw batch ID. Existing wizard/background progress should be retained. [Jobs panel](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Media/MediaIngestJobsPanel.tsx:47). | Surface Recent imports from submission onward with recognizable source/count/status. Users should reopen active or completed imports without finding an ID. |
| P2 / S2 | **Accessible names and touch affordances need attention.** Sort/export comboboxes and multi-review checkboxes were unnamed in snapshots; checkboxes are removed from tab order. Some mobile controls measure 18–26 px. [Checkbox](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx:167); [export selector](/Users/macbook-dev/.codex/worktrees/media-ux-review-dev/tldw_server2/apps/packages/ui/src/components/Review/MediaBulkToolbar.tsx:139). | Add persistent labels and named native selection controls; enlarge useful hit areas. Verify every action with keyboard and a screen reader, including counts and errors. |
| P3 / S2 | **Configure says defaults affect only future items, but settings alter this run.** Turning analysis off changed both already-queued URL requests. | Label settings for This run separately from defaults for future items. Changing analysis should visibly update Review before submission. |

**Persona and cognitive-load findings**

- **Jordan, first-time researcher:** the first URL disappears, “review” refers to two storage states, and a successful batch lacks an obvious Open all saved items continuation. They need one visible Add media entry, outcome-oriented preset descriptions, accurate counts, and a direct path to the saved content.
- **Alex, power user:** accelerators exist, but pagination clears Inspector selection, failed inputs must be reconstructed, and preview status diverges from the review set. They need stable selection scope and reliable batch continuation more than additional commands.
- **Sam, keyboard or small-screen researcher:** row activation changes meaning, selection controls lack names, and content/actions can remain hidden. Explicit preview/select controls and visible Results/Content transitions are necessary.

The wizard's cognitive burden is moderate and reasonably managed. Multi-review has a higher burden: view mode, orientation, Options, navigation, expansion, content comparison, scoped chat, sections, and search compete before reading begins. Group these into **Reading**, **Layout**, and **Selection actions**, retain the current core functionality, and clarify layout Compare versus comparing extracted content. This follows [NN/G's progressive-disclosure guidance](https://www.nngroup.com/articles/progressive-disclosure/).

The emotional valley occurs when the interface appears to accept work but loses input, changes selection, or gives a success count that omits an entry. Accurate handoffs and a saved-batch review action would make the end of the workflow reassuring.

**Recommended repair sequence and potential improvements**

1. Repair shared URL handoff, delimiter guidance, eligible counts, and duplicate handling.
2. Wire existing failed-item retry; add Review these N saved items.
3. Align preview/selection/navigation and Inspector selection scope.
4. Adapt small-screen reading/bulk controls and complete keyboard/label checks.
5. Polish the terminology and hierarchy after these behavioral repairs.

Use the existing wizard, jobs/session data, collections, selection store, and mobile review pattern. A second import UI would increase inconsistency.

Further opportunities: expose existing active-tab capture beside Add in the extension; show Saved / Processing / Ready for Knowledge as distinct states; offer a compact batch summary with its sources and outcomes; keep the 30-item reading cap explicit while letting users manage larger metadata/action sets; and teach adding/reviewing a batch through contextual hints rather than a long tutorial. Validate these with users before expanding them.

**Validation after fixes**

Run the same tasks with a connected development backend and safe test content:

- Fresh user: add one URL and one file; open their saved content and explain whether each is searchable in Knowledge.
- Mixed batch: include valid, invalid, duplicate, and one deliberately failing source; verify every input has an explained outcome and retry preserves successes.
- Established library: select across pages, create/open a review set, compare content, then switch preview without losing selection or status accuracy.
- Keyboard and 390 px viewport: complete import, reading, and selection without hidden actions or unnamed controls.
- Close/reopen during processing: locate the same import and verify status against backend records.

Measure unassisted completion, input/selection rework, count discrepancies, recovery success, and time to first saved content. These are proposed validation measures; no participant success rates were collected.

**Detector result**

Assessment B ran the narrowed detector once across 30 TSX markup files: five Media components and 25 Quick Ingest files. **0 findings, 0 advisories, exit 0; no rule locations or false positives.** It did not catch the interaction defects. Mutable browser injection was attempted, but exact-dev CSP blocked the overlay script; no user-visible overlay ran. Independent browser screenshots, accessibility snapshots, captured fixture requests, and source inspection supplied the evidence.
