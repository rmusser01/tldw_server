# Political PDF UX completion and CI follow-through

PR [#3480](https://github.com/rmusser01/tldw_server/pull/3480), against `dev`. Implementation tracking: TASK-15505.13–15; late CI repairs: TASK-15505.8. [Design](../Design/POLITICAL_PDF_NNG_COMPLETION_2026_11_16.md). Findings retain the IDs and severity from the [Luna retest](POLITICAL_PDF_LUNA_GROUNDING_RETEST_2026_11_15.md), mapped to [Nielsen Norman Group's usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/).

## Four interface findings addressed

| Finding | Heuristic / severity | Implemented behavior and observed evidence |
|---|---|---|
| UX-PDF-69 | H1 status, H6 recognition / 2 | Section-output cards and readers show the recorded model/provider, creation time and exact known selection count with a bounded title preview. Imported metadata is display-only and marked as imported; absent metadata remains unknown. The actual eight Luna outputs show distinct dates and scopes of 1, 2, 5 or 30 sections while retaining `gpt-6-luna`, independent of the current chat model. |
| UX-PDF-70 | H1 status / 2 | Selection and running views show the observed 922-page/30-chapter example of 52–62 minutes, explicitly qualified as an observation rather than an estimate for the current run. Saved progress, completed-chapter reading and unavailable usage/cost are explained. The real creation view displays all qualifications; no per-file ETA appears for a section Job. |
| UX-PDF-71 | H1 status, H3 control / 2 | A running output with a completed chapter exposes “View completed chapters.” The reader follows the existing saved-checkpoint store, retains selection, describes pending/processing sections and disables review edits until terminal status. In an isolated browser fixture, progress changed from 1/3 to 2/3 while chapter 1 remained selected; mobile read chapter 2 with Save/correction disabled. Terminal status enabled review while keeping chapter 2 selected. Owner/workspace/revision/cancellation guards remain. |
| UX-PDF-72 | H6 recognition, H7 efficiency / 2 | Embedded bookmarks win; otherwise Contents displays existing generated navigation. It resolves the selected node through exact canonical page coverage, rather than interpreting printed labels as physical pages. The actual restored PDF (Media 2) exposes 46 entries. Keyboard selection opens Education at physical 352 (printed 319); FTC opens at 902 (printed 869). The malformed “Director, Project” entry reports unavailable and leaves page 902 unchanged. Mobile Sidebar → Document also opens page 352 without horizontal overflow. |

The same PDF is retained: 922 pages, freshly rechecked SHA-256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. This is a UI walkthrough of stored sources, not a fresh ingestion or model run. Original Job 46 remains completed with 30 chapters, 117 portions and eight separate review overlays. No original drafts or reviews were changed for the running-state fixture; backend writes were blocked in its separate browser context.

Desktop inspection used 1440×1000, mobile 390×844, with keyboard navigation. The bounded visual pass caught an unsupported secondary-text class in the new PDF header and inherited dark-mode modal-title color. Both now use existing text tokens; the confirmed Contents heading contrast is 15.58:1 against its rendered background. The changed-components Impeccable scan returned no deterministic findings. This is not a complete accessibility certification, physical-device study or VIP signoff.

Local evidence screenshots are retained under `.playwright-mcp/`: `nng-output-identity-confirmed.png`, `nng-completed-reader-confirmed.png`, `nng-running-reader-desktop.png`, `nng-running-reader-mobile.png`, `nng-pdf-generated-contents-desktop.png` and `nng-pdf-contents-mobile-confirmed.png`. The first PDF screenshot records the initial visual finding; the final mobile image verifies its repair. The isolated running fixture is UI/checkpoint proof, not additional provider-quality evidence.

## What “analysis quality is partially resolved” means

The context and citation-transport repairs improved the output and made the full workflow usable. They do not establish that each statement follows from its citation. A real evidence ID, a copied quote and a completed Job are structural checks.

The retained source review still finds a Financial chapter statement denying an endnote attribution even though note 53 explicitly links the contribution. The overview also has citation-support gaps and a statement about maintaining 64 interceptors where the source proposes buying at least 64 Next Generation Interceptors. Separate agent-proposed correction overlays address reviewed passages without rewriting the original model output. These examples explain why UX-PDF-66, TASK-15505.6 and TASK-15505.12 remain open. Human labels and VIP acceptance remain unset; there is no calibrated semantic accuracy score.

The subsequent [strict Claims integration and retest](POLITICAL_PDF_STRICT_CLAIMS_RETEST_2026_11_16.md) wires actual citation-support judgments into chapters, overviews and QA, preserving findings through review/export. It records its own immutable Luna responses and later NN/g findings. This improves detection and reviewability; it does not retroactively certify the original Job 46 or replace human calibration.

## Local qualification

Counts overlap and must not be summed as unique coverage.

| Scope | Result |
|---|---|
| Output identity and connected output utilities | 109 tests passed |
| Live reader, Stage1/Stage2, review and persistence | 243 tests passed |
| Integrated final identity/reader/Stage1 | 143 tests passed |
| PDF navigation, exact destinations, real ownership leases and connected guards | 396 distinct cases covered across the 395-case connected run and final 331-case run |
| PDF parser / legacy DB import owners / function-size checks | 114 tests passed |
| Late audio / PDF / docling / SSRF cases under CI's actual egress allowlist | 11 tests passed |
| HF tokenizer fixture and connected endpoints | 234 tests passed, offline |
| RAG contract, request deadline, generation bounds and context expansion | 43 tests passed |
| Selected-source SQLite media/chunk retrieval | 2 tests passed |
| Natural-colon retrieval and real research evidence | 93 tests passed; 1 PostgreSQL availability-fixture skip |
| Additional translation/direct scoped retrieval/ratchets | 85 tests passed; 8 PostgreSQL availability-fixture skips; initial line-only scope-baseline failure repaired |
| Final function-size, scope, test-quality and Backlog ratchets | 82 tests passed |
| Security / static checks | Three changed production Python files: Bandit zero findings/errors. Full frontend types passed with a 12 GiB heap after a 4 GiB attempt exhausted memory. No new lint findings; inherited diagnostics are preserved. Independent reviews clear. |

The final visual token/title edits pass a focused 14-case connected rerun (120 unrelated cases deselected by the name filter). An attempted broader local hybrid retrieval run stalled fetching an uncached Qwen tokenizer with network unavailable; it is not reported as a pass. No model download, provider generation, test disabling or expanded baseline allowance was used to hide that limitation. PostgreSQL skips come from the repository's existing fixture.

## Late CI failures and repairs

The wider matrix on documentation-only head `9e7fe080` reported nine failed shards after the earlier seven required gates had passed on `99a16d63`. The previous report correctly left that matrix pending. The current follow-through repairs the observed causes:

- PyMuPDF4LLM 2.x reports `metadata.page_number`; older versions report `metadata.page`. Read both documented one-based forms, reject conflicts/invalid identities, preserve extracted text and withhold unverified page provenance. Modern metadata is tested with fixtures; actual local extraction still uses installed 0.0.27. No dependency change.
- Navigation import-owner tests now assert the core Media owner and endpoint delegation rather than removed endpoint aliases.
- A mocked audio-download test uses a controlled `example.com` URL permitted by CI's existing egress policy, with exact download/MIME/STT assertions. No URL-policy bypass or allowlist expansion is introduced. The six other failures in that late ingestion shard share the repaired PDF page-key incompatibility.
- Two concurrent HF tests now supply an opt-in in-memory tokenizer at the existing loader seam. Actual token-budget preflight, request barriers and local/remote credential/mode assertions remain; production code is unchanged.
- The vector-location precheck reuses the existing mapping/object metadata accessor, preserving canonical retrieval phase contracts.
- Nested component timeouts now stay in the same task. A reproduced Python 3.11 `wait_for` race could swallow the outer request cancellation after synchronous cold setup; a deterministic deadline regression covers it.
- Natural question colons can reach the existing bounded fallback. Known FTS column scopes, compact unknown scopes, quoted phrases and operators stay strict. Ranking and source ownership are unchanged.
- Stale token-limit/deadline/unscoped-chunk expectations now reflect the already implemented 8192-token bound, request-wide timeout and eight-term chunk fallback. Exact boundary, source and cancellation assertions remain. One scope-baseline line moves 3785→3787; its AST is identical, with no new allowance.

Hosted checks must be read on the new pushed head; local qualification is not a claim that pending hosted jobs passed. The PR remains a draft. The requester's human-written Change summary and review/merge gate remain intact. No merge or deployment.

ADR required: no. Existing Research Workspace shell, owned Media navigation, provider/preflight, cancellation and provenance contracts remain authoritative; these are bounded presentation and compatibility repairs.
