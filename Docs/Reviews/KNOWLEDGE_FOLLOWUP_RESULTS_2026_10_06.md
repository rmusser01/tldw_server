# Knowledge follow-up results — 2026-10-06

PR3196 completed the main [Knowledge remediation](KNOWLEDGE_UX_REMEDIATION_2026_10_04.md). This follow-up investigated the remaining browser, provenance, live-model and optional CI work on `dev` at `1fc353c3f67c93ba05102e7b0136ac4acac8f510`. The production changes fix seven demonstrated workflow defects using the existing Notes, media-ingestion, Research and browser APIs. This report records implementation evidence; participant and assistive-technology validation remains open.

## Workflow issues and implemented solutions

Severity is an expert assessment of the observed task consequence, not a measured frequency among users. High means blocked work, lost source identity or misleading evidence qualification. Medium means avoidable duplication, recovery or missing context. The assessment uses [NN/g's usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/), particularly visibility of status, recognition, user control, error prevention and recovery.

| Workflow / issue | Consequence and severity | Implemented solution | Verification |
| --- | --- | --- | --- |
| Ask an added note using a normal multiword question. Notes search treated the whole question as a literal phrase. | A first-time user sees no evidence even though the relevant note exists. High. | RAG uses bounded literal term matching through the existing canonical Notes adapter. Ordinary library phrase search remains unchanged. Owner/deletion filters and SQLite/PostgreSQL behavior stay in the database abstraction. | Natural-language Notes regressions and live before/after query: zero documents became the correct note and review date. |
| Inspect answer support. Unmatched text fabricated offset-zero hard/quote citation spans; ClaimsEngine spans were accepted without enough validation. | A power user may mistake an available reference for verified support. High. | Exact source offsets are required by the heuristic mapper. Verified ClaimsEngine support must identify a returned document and valid evidence; refuted, malformed, unknown or invalid spans cannot receive credit. | Negative, partial-match, numeric, quote and verified/refuted-claim regressions; five live answers inspected separately from citation coverage. |
| Continue a retrieved note into Research. Only retrieved chunks were copied. | An appendix or omitted section is unavailable for subsequent research. Medium. | Fetch the exact owner-scoped canonical note, validate identity/revision/active content, and ingest one full versioned snapshot. Keep the original retrieved excerpts and answer qualifications separately. Failed reads remain visible and retryable. | Mounted failure/retirement tests and a real note changed after retrieval: the uploaded v2 snapshot includes the new appendix while the original excerpt remains unchanged. |
| Continue into a fresh server workspace. Canonical Notes persistence depended on a legacy migration tombstone. | Attached sources could be paired with only a local draft and no structured provenance marker. High. | Wait for the existing server-workspace confirmation and then reuse canonical Notes save/readback. Completed local imports keep their retirement behavior. | Fresh-workspace confirmation regression, discarded-draft regression and actual canonical note save with a validated provenance marker. |
| Attach streamed or restored media chunks. Chunk IDs obscured the canonical media ID. | Existing sources could be copied again as excerpts. Medium. | Resolve `source_id`/`sourceId` as media only for explicit `media_db` results. Preserve Notes and external-web classification. | Streamed/restored regressions reuse canonical media and perform no upload. |
| Review overlapping sources. A note snapshot can reuse media that retrieval also returned. The later group overwrote the earlier evidence. | The saved structured provenance lost the original Notes identity/version. High. | Keep one attachment and retain all payload references resolved to that media ID; snapshot status reflects any snapshot reference. | Both source orders fail before the fix and pass afterward; canonical marker asserts both identities and the original revision. |
| Capture a page through the native extension menu. Capture awaited before requesting the gesture-restricted sidebar, and sidebar acknowledgment was ignored. | A first-time user chooses capture and sees no sidebar or reliable completion. High. | Start sidebar opening inside the original menu gesture, await acknowledgment, and surface failure. Update sibling Notes/Companion/Narrate callers using the shared helper. | Original failure observed through the actual macOS context menu; actual menu-listener and Chrome/Firefox helper regressions pass. Final native-menu success remains unqualified. |

The versioned-snapshot notice explicitly explains which content is complete, which is an excerpt, and that neither refreshes automatically. This supports [visibility of system status](https://www.nngroup.com/articles/visibility-system-status/). The extension fix follows the [Chrome sidePanel API's user-gesture requirement](https://developer.chrome.com/docs/extensions/reference/api/sidePanel).

## Live retrieval and answer evidence

The disposable corpus contains two related greenhouse documents, an unrelated orchard distractor and a canonical research note. Baseline calibration is three days with 120 litres per plot per week; the refined phase uses five days and 90 litres. Operating cost is absent. The note gives 18 November 2026 as the next review date.

Generation used the available local Gemma4-26B-A4B Q4 model through the existing custom OpenAI-compatible provider. These cases used SQLite FTS, selected source IDs, `top_k=8`, at most 384 generation tokens, and application cache, reranking, query expansion, web fallback and intent routing disabled. Embeddings in the disposable runtime were controlled fixtures; this is not a live-vector retrieval qualification. No commercial provider was invoked.

| Case | Answer observed | Exact hard-span coverage | Observed request time |
| --- | --- | --- | --- |
| Baseline calibration | Three days, correct source. | 0 — faithful paraphrase without an exact matching span. | 1,954 ms |
| Refined water use | 90 litres per plot per week. | 1 | 174 ms |
| Two-source comparison | Three versus five days; 120 versus 90 litres. | 0 — paraphrased comparison. | 663 ms |
| Missing operating cost | Explicitly says the provided context lacks the cost. | 0 | 343 ms |
| Canonical note appendix | 18 November 2026. | 1 | 229 ms |

All five returned no API errors. Scoped cases excluded the orchard. Ordinary unscoped Notes retrieval initially returned no documents; after the adapter fix it returned the correct note and answer in 1,886 ms. One live DuckDuckGo request returned HTTP 200 and three NN/g results in 638 ms. These are small-case observations, not p95 latency estimates, a comparative model benchmark or evidence of general answer quality. Hard-span coverage measures exact or explicitly verified evidence anchoring, not semantic truth; correct paraphrases can score zero.

The WebUI also completed real Ask → evidence → Research → canonical note handoffs. API readback proved complete versioned snapshot content and structured marker persistence. A subsequent Quick Notes edit received HTTP 200, reopened with the edit intact, and canonical readback at revision 4 retained the marker and five source attachments. Saved generated notes can themselves be retrieved as Notes; interpretation must continue to distinguish their retained original evidence from new independent evidence.

Credential-free case results and source expectations are retained in [live-evidence.json](artifacts/knowledge-followups-20261006/live-evidence.json). Raw runtime credentials, browser profiles and authentication headers are excluded.

## Validation and limits

- Final affected frontend run: **152 tests passed in six files**. Includes canonical Notes reads, retries, stale owner/workspace fences, server readiness, full content, both overlap orders, provenance round trips and native-menu handler ordering/failure.
- Touched API/RAG/database run: **91 tests passed**, four warnings. External model services are mocked in these deterministic tests; live cases above use real local generation.
- Final WebUI and extension type checks passed. Final WebUI production compile and Chrome extension build passed. Web shared bundle is 589.9 KB gzip against its existing 600 KB budget. The Web build required the established 8 GB Node heap; the earlier default-heap attempt exhausted memory.
- Production Bandit scan of all three changed Python files found **zero issues**. Production TypeScript lint found zero errors and exactly the same warning counts as unchanged `dev`. Touched Python lint retains 27 existing diagnostics in three test files and adds none. `git diff --check` passed.
- An independent reviewer examined the final changes, including the readiness gate and overlap fix, and reported no actionable findings. That review did not replace the executed checks.
- The broad frontend suite **did not pass**: 22,995 passed, 224 failed, 20 skipped across 2,490 files, with two runner errors. Of 224 named failures, 183 reproduced in bounded runs on unchanged `dev`; 41 did not. One of those 41 was this change's async test-fixture return bug and is fixed in the final affected run. A bounded 16-file candidate comparison reproduced seven baseline failures plus one SkillsManager failure; that Skills case then passed isolated on both candidate and unchanged `dev`. Other comparison cases passed. This supports run-order/resource sensitivity as an inference, not a proven cause. No broad-suite-green claim or unrelated product repair is made.

## Email CI dependency

The optional [PR3196 CI run](https://github.com/rmusser01/tldw_server/actions/runs/37412488174) had 239 passing tests, one skip and one forbidden-call teardown error. Four nested-email cases pass with SQLite. The PostgreSQL offline tripwire exposed authentication quota/ledger socket access through media-byte accounting, rather than Knowledge UI code. Existing [PR3016](https://github.com/rmusser01/tldw_server/pull/3016) and [PR3084](https://github.com/rmusser01/tldw_server/pull/3084) already own the isolation repair. A disposable accounting-boundary isolation reproduced **44 passing Email integration tests** while preserving model, job, DNS and outbound-request rejection. This PR does not duplicate those changes; TASK-13511 remains open until the existing repair is integrated and CI confirms it.

## Remaining work and concrete next steps

| Item | Current evidence | Completion requirement |
| --- | --- | --- |
| Provenance independent of editable Markdown | Full-source snapshots, canonical save/readback and validated portable markers are implemented. The marker can still be removed by editing outside the aware client. | Approve the [Notes/Sync sidecar proposal](../Design/2026-10-06-knowledge-followup-source-context.md), create the required ADR, then implement canonical storage, conflict, replay, deletion/restore and old-client compatibility. TASK-13514 remains open for this criterion. |
| Complete external-web source refresh | Web results remain honestly labeled retrieved excerpts. | Define capture permissions, refresh behavior and a source-version contract before treating snippets as complete live sources. |
| Final native extension capture | Original native failure reproduced; final handler regressions/build pass. The owned browser window became unavailable on the current macOS Space. | Run the actual final extension context menu → captured draft → saved note → scoped Ask path. Do not replace it with an injected callback. |
| VoiceOver, Safari/iOS and mobile keyboard | Available Safari rendering/AX controls inspected; desktop browser workflows exercised. | Run spoken navigation with permission and actual iPhone/iPad keyboard/browser behavior. Viewport or accessibility-tree checks do not complete this item. |
| First-time and power-user usability sessions | A [ten-task moderator protocol](KNOWLEDGE_FOLLOWUP_PARTICIPANT_PROTOCOL_2026_10_06.md) is ready. No participant sessions occurred. | Observe independent/assisted success, recovery, source interpretation and trust understanding with real participants. |
| Broad frontend/optional Email CI | Failures and baseline comparisons are retained; Email repair has existing PR owners. | Integrate the existing CI fix and resolve/qualify remaining suite failures before a green release claim. |

The follow-up PR must remain a draft until its own human-written `Change summary` satisfies the [repository merge policy](../superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md). The human summary provided for PR3196 does not describe this new diff.
