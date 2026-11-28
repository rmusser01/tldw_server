---
target: WebUI Quick Ingest modal, all reachable states
total_score: 21
max_score: 40
na_heuristics:
p0_count: 1
p1_count: 4
target_identity: "file:/Users/macbook-dev/.codex/worktrees/one-screen-quick-ingest/tldw_server2/apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx"
target_fingerprint: "sha256:24f0cf482ac1d1d4503f288983531194e328f982d897a2f5eeb8408242b81526"
target_path: /Users/macbook-dev/.codex/worktrees/one-screen-quick-ingest/tldw_server2/apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
timestamp: 2026-11-26T03-38-07Z
slug: ponents-common-quickingestwizardmodal-tsx-2e351f58
---
Method: dual-agent (A: full_modal_design · B: full_modal_evidence)

# WebUI Quick Ingest

| Priority | Reproduced behavior | Proposed interaction |
|---|---|---|
| P0 · Reading an extraction | A successful extract-only response included transcript text. The receipt reported “1 extracted”. More actions contained Add more sources, and the footer offered Done. | Give a single readable extraction **View extracted text**, and give each readable member of a batch **View text**. Provide Copy text and a return path to the receipt. Describe empty or unavailable output explicitly. Scope the proposal to text available in the active run. |
| P1 · Repairing a source | A well-formed URL rejected with HTTP 422 prompted “Check the file format” and **Correct settings**. Adding a replacement put it outside the retry run. Removing the original left zero runnable sources. After draft reload, both a local file and a pasted passage requested reattachment. Neither draft row offered a picker. | Give the failed entry **Edit link**, **Choose file**, or **Paste again** according to the input. Keep the entry’s batch identity, selection, metadata and retry scope. A changed configuration gets **Edit settings** only when it can resolve that error. |
| P1 · Knowing whether work stopped | After a fixture rejected bulk cancellation with HTTP 503, the receipt said “1 cancelled”. With one failed source and one running at 40%, the minimized widget said “Failed”, then disappeared after ten seconds. An inaccessible tracked job ended with a warning to check Media, no affected name, and Done. | Keep active and failed counts visible together. Mark a submitted stop request as pending. Report cancellation only after acknowledgement or a terminal job status. For an unresolved job keep its source name, last status, **Check status** and **Check Media**. Keep already saved items reachable. |
| P1 · Making a consequential choice on a phone | In the 320-pixel close confirmation, the focused Stay button occupied x = −116 to −56. At 390 pixels only its rightmost 14 pixels were in the viewport. The sampled dark-mode title contrast was approximately 1.38:1. In playlist replacement review, one at-risk URL was listed while the action confirmed two sources without explaining the difference. | Reflow the close choices vertically. **Continue in background**, **Stay here**, and **Stop remaining source…** describe three distinct outcomes. Confirm stopping the remaining work. In overwrite review show the item title and URL, how many matches may change, how many other sources will run and the counted final action. |
| P1 · Retaining work while editing | Clear all immediately emptied a queue containing pasted material. On both desktop and phone, returning from replacement review erased an unqueued passage. | Offer **Undo** after Clear all and retain the in-memory inputs during its window. Preserve the selected input, unsent text, queue, scroll and focus across confirmation and Back. |

Severity applies to each affected task. P0 blocks its stated outcome. P1 represents substantial loss, confusion or difficulty. Most straightforward saves still completed.

## Heuristic scorecard


| Nielsen heuristic | Score / 4 | Deciding observation |
|---|---:|---|
| Visibility of system status | 2 | Counts are useful, but bulk cancellation and background status can assert the wrong final state. |
| Match with the real world | 2 | Files, Links and Text are familiar. After refresh a paste becomes a file the person never had. |
| User control and freedom | 2 | Resume and scoped retry work. Clear all and Back can discard work. |
| Consistency and standards | 2 | A result row can offer file reattachment while a draft row merely instructs the user to do it. |
| Error prevention | 2 | Replacements require review. The review omits the unaffected count, and the phone confirmation clips the safest choice. |
| Recognition rather than recall | 2 | Source identity is visible initially. Bare URLs in replacement review discard useful titles. |
| Flexibility and efficiency | 3 | Batch input and failure-only retry save effort. Source replacement breaks that fast path. |
| Aesthetic and minimalist design | 3 | The draft and receipt have a clear action. Processing repeats related status above and below the source list. |
| Help users recognize, diagnose and recover from errors | 1 | A URL correction, a restored file and an interrupted job can each lead to an unavailable remedy. |
| Help and documentation | 2 | File limits and draft hints help. A format error advises checking a file even for a URL. |
| **Total** | **21 / 40** | First full-modal baseline. |

A first-time researcher can choose a source immediately, then run into a provider prerequisite before saving it. A returning batch importer can retry failures efficiently but can lose sight of remaining work after minimizing. A keyboard user reaches the provider and replacement heading, then lands on the document body after skipping analysis, repairing a link or returning from review. The next Tab reached Save, Clear all or Review replacements respectively. Two different source URLs rendered the same visible prefix on phone. An Open in Media row action measured 109 × 20 pixels beside 44-pixel-high footer actions.

The CLI design detector returned zero findings in both source scans. The browser overlay identified two local findings in the empty draft, three during processing, two on the mixed phone receipt and three on the mixed desktop receipt. Confirmations and the recovered receipt had none. Several nested-card flags identified legitimate source grouping. The 10-pixel error badge, long line and cramped spacing warrant attention, but the detector missed the clipped Stay button and the false cancellation receipt. Shipped UI screenshots in the illustrations are clean captures.


## Design specificity and strengths

The counted source action, locally reviewed drafts and the bounded failure-only retry are specific to this product. Generic empty results and format errors discard the source identity that makes those interactions work. Details are in the full review and visual proposal in Docs/Design/2026-11-25-quick-ingest-modal-state-review.md.

## Design decisions to confirm

Should extract-only output remain available for the active run or also after reload? Should the follow-up ship the recovery and job-state fixes ahead of the batch-editing polish?
