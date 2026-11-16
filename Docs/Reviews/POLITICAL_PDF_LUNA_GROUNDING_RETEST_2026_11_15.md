# Luna overview and attribution repair: retest

TASK-15505.12; CI and UI follow-through TASK-15505.8. This extends the [first Luna evaluation](POLITICAL_PDF_LUNA_EVAL_2026_11_15.md). Original outputs, failures and review overlays are retained. Human labels and VIP acceptance remain pending.

The final live acceptance run used `f6450e69f0`. Fresh Job 46 completes all 30 chapters, 117 portions and the overview, with no generation failure. Actual review, complete-pack export, offline reading and fresh-browser cross-server restore pass. Source review still finds a Financial chapter endnote-attribution error and citation-support gaps; the artifact remains `needs_revision`, with eight separate agent-proposed correction overlays. This is a verified workflow and bounded improvement, not full semantic acceptance.

## Fixed inputs and meaning of the checks

The same 922-page PDF has SHA-256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. It remains Media 1 with navigation `media_1:v4:ec5d2af126`. Runs reuse the verified ingestion and original PDF; they do not claim a new upload. OpenAI `gpt-6-luna` uses provider-default reasoning, an 8,192 output-token budget and the existing 2,048-token reduction cap. The 13-case QA fixture and both QA context modes are unchanged.

Completed Jobs, valid citation IDs and exact copied passages are structural checks. They do not prove that every adjacent claim is supported or that an author or qualification is correctly described. All semantic findings below are agent source-review notes, not calibrated accuracy scores or human labels. Cases used to guide repairs are regression cases, not untouched held-out validation.

## What changed

The [repair design](../Design/POLITICAL_PDF_LUNA_SYNTHESIS_REPAIR_2026_11_15.md) records the source evidence and scope. The final implementation:

- Carries literal source headings across their verified contribution ranges. Proposed author names must occur in the particular cited passage or its applicable heading; no fuzzy name matching or first-author inheritance is added.
- Retains exact labeled author-note excerpts, including both sole-responsibility statements and collective-attribution caveats. Notes stay within contribution/reference boundaries and do not authorize acknowledged people as authors.
- Keeps this source context separate from model compression at chapter and overview synthesis. All 28 author notes for all 30 selected chapters fit in 15,916 of the existing 16,000-character overview-context bound. If the bound or a known omission prevents retaining context, the overview fails explicitly, keeps completed chapters and advises selecting fewer sections.
- Uses only the current call's literal registered evidence IDs, explicitly enumerated as individual citations. Range interiors and globally valid but unsupplied IDs remain forbidden. Heading/note excerpts are attribution context, not new claim evidence.
- Retains bounded rejection diagnostics in the owner-scoped artifact: stage/group, response hash, truncated preview and rejected IDs. Raw provider responses do not enter generic errors or server logs. No automatic paid retry is added.
- Requests native JSON mode for OpenAI extraction as already done for DeepSeek, while retaining shape, source-ID and finish checks. JSON mode is not requested for prose synthesis. JSON mode assists syntax; it does not validate meaning or schema. See the [OpenAI guidance](https://developers.openai.com/api/docs/guides/structured-outputs#json-mode).
- Restores accurate failure messages after canonical reload and labels running-output deletion correctly. Specific failure guidance remains available instead of an unchanged retry recommendation.

No new dependency, store, API authority or human-review authority is introduced. Existing ADR-003, 007, 015, 025 and 065 continue to govern.

## Summary evaluation record

| Run | Code | First observed result | What it establishes |
|---|---|---|---|
| Original Jobs 37/38 | Prior baseline | Overview rejected unknown evidence IDs; Job 38 retained 30 chapters / 117 portions | The rejected responses were not retained; the exact historical cause remains unknown. |
| Exact Job-38 first-group diagnostic | Original prompt | 26 valid cited IDs from 94 literal IDs; `stop`; 12.019 seconds | One successful replay does not explain or erase the original failure. |
| Job 40: fresh EXIM | `282094cfe9` | 1 chapter / 2 portions / overview; 120.584 seconds; scope, spans and readback pass | de Rugy/Hazelton continuity improves; some adjacent claims still lack complete citation support. |
| Job 41: fresh full run | `282094cfe9` | 29 chapters / 116 portions; DHS portion rejected; overview completed with 29/30 disclosure; 3,442.716 seconds | Strict passage validation works; Agriculture/DOJ endnote attribution remains wrong. |
| Job 42: fresh Agriculture + DOJ | `273f13561e` | 2 chapters / 11 portions / overview; 301.572 seconds; structural checks pass | No false endnote contributor/denial in this sample; later runs show this was not a reliability guarantee. |
| Job 43: actual UI checkpoint retry | `273f13561e` | 30 chapters / 117 portions / overview; structural checks pass | Recovery preserves all 116 previously completed batches and 29 chapters, including their old semantic errors. |
| Job 44: fresh full run | `3a1bd2744d` | 29 chapters / 116 portions; Labor invalid JSON; overview has 118 valid unique IDs; 3,153.085 seconds | Artifact correctly remains failed; source review finds attribution errors despite valid overview IDs. |
| Job 45: fresh five-chapter retest | `53392cffbe` | 5 chapters / 23 portions / overview; structural checks pass; 708.574 seconds | All five chapter-attribution reviews are clear within scope; overview still reverses Education's responsibility note. |
| Job 46: final fresh acceptance | `f6450e69f0` | 30 chapters / 117 portions / overview; 3,700.275 seconds; all structural checks pass | 58 valid unique overview IDs; remaining source-review findings stay explicit. Actual review/export/restore passes. |

The runs have separate immutable directories under `.artifacts/political-pdf-evals/`: `luna-grounding-exim-20261115`, `luna-grounding-full-20261115`, `luna-grounding-endnotes-20261116`, `luna-grounding-recovery-20261116`, `luna-grounding-final-20261116`, `luna-grounding-context-20261116` and `luna-grounding-acceptance-20261116`.

### Diagnostic limits and source findings

Old chapter text contained citation ranges with interior IDs absent from the literal current-call vocabulary. This motivated explicit enumeration, but is not a proven explanation of Jobs 37/38. The exact successful replay used 16,850 input characters and 4,198 prompt / 894 completion tokens, including 294 reasoning tokens.

Job 41's DHS failure used a workspace source UUID instead of a supplied P passage ID for a heading-only claim. Its original bounded preview/hash are retained; the original E-only counters remain zero. The follow-up removes opaque prompt IDs, confines headings to attribution, enumerates P IDs and records actual rejected structured values. Its terminal envelope SHA-256 is `4091f4fb026f3d11f8e61cb3228fc856d92bed3ad14ef5e7d37921470b144751`.

Agriculture's reference boundary at character 926,030 is correct. Its substantive endnotes do not create another contributor. In several earlier runs, empty reference-author metadata became an invented author division or an unsupported exclusion of Bakst; DOJ showed the same failure. A checkpoint retry correctly preserves such completed drafts, so retry safety cannot establish corrected semantics. Job 43's overview repeated the preserved Agriculture error despite 98 valid unique IDs.

Job 44's Personnel chapter invented an unnamed contribution separate from Devine/Kirk/Dans even though both initial reduction groups contained their exact joint heading. The 38,696-character input required reduction; final synthesis did not reattach source context. Successful intermediate responses were not retained, so the first point at which names disappeared cannot be proven. Literal context now survives compression independently.

Job 44's all-chapter attribution audit reviewed 29 completed chapters: 17 had no issue found within scope, eight had important findings, four had review cautions, and Labor was unreviewable. The selected six-chapter policy/citation audit checked 679 exact quotes. Findings include invented contributor divisions, contradictions of collective-author caveats, and detached endnote/body continuations. Education's sole-responsibility note must not be generalized into the different DHS/Intelligence caveat. Source-backed corrections and current-run evidence IDs are retained separately.

Job 45's Personnel, DHS, Intelligence, Agriculture and Education chapters address those targeted patterns in this sample. Its overview nevertheless turns Education's author-alone/any-other-person qualification into a collective disclaimer, exposing the remaining overview context handoff. The final code carries exact notes independently at that caller too. The two terminal envelope hashes are:

- Job 44: `91c3429cc0e773172d837c3ccee2afd38412fbc748168d22075095598b41dec8`.
- Job 45: `acbc1e3daa89528ef29cd0b3255b7fdd05fb8d16ef3ec9463cdf13533a16759a`.

A separate OpenAI JSON-mode capability probe returned HTTP 200, `stop` and parseable JSON. Its exact quoted-string expectation differed through HTML entities; this is not recorded as a content-fidelity pass.

Job 46's first terminal envelope has SHA-256 `ed65fabf6e45589a0c6cf080a57f0aac9405db2525ff73c641a32e6224919e39`. The fresh CLI run exits 0: selected scope, all exact evidence spans and artifact readback pass. Its 8,180-character overview contains 58 unique valid IDs. All 30 chapter contents/contexts and 117 completed portion objects match their first completed saved outputs through terminal version 152. No automatic retry or later rewrite converts an earlier failure into success.

The final all-chapter attribution audit finds 26 chapters clear within its bounded scope, three cautions and one important residual. Personnel's false split, DHS/Intelligence caveat contradictions, Agriculture/DOJ endnote divisions and the main Bowes continuation error are absent in these fresh samples. Media and HHS overinterpret collective-preparation acknowledgments; FTC's unnamed Onward byline remains the known capture limitation. Financial still denies assigning an endnote despite an explicit note-53 contribution link. A neutral review correction restores that linkage without forcing an extraction author field.

The independent six-chapter policy/citation audit verifies 683 exact quotes. The final overview's earlier Personnel, Agriculture and Education attribution errors are absent in the checked scope; it does not repeat Financial's endnote denial. Five bounded overview findings remain, including direct citation gaps and wording that says maintaining 64 interceptors where the source proposes buying at least 64 Next Generation Interceptors. Valid IDs do not establish claim support. Eight partial correction overlays cover Media, the six selected policy chapters and the overview; they leave unaudited claims and human acceptance explicitly pending.

Citation undercoverage remains distinct from ID validity. Some paragraphs cite only the first passage of a multi-passage claim. Earlier Job 44 also associated a DOJ endnote's cross-reference with the wrong proposal; fresh Job 46 did not repeat that error. These require source review; widening the validator would not repair them. FTC's malformed later outline leaves the afterword tail unresolved: the source names Edwin J. Feulner, but the system conservatively avoids inherited attribution across that uncertain boundary.

## Frozen QA rerun

The separate `luna-grounding-qa-20261115` run retains all 26 first responses without retry. Every response passes shape, supplied-ID and finish checks. Usage totals 46,291 prompt plus 17,859 completion tokens, including 11,119 reasoning tokens. QA request time totals 188.996 seconds, median 7.116 seconds; the full source/retrieval/QA ledger spans 218.853 seconds. These are observations with a separate summary worker active, not a load benchmark.

The QA run used `282094cfe9`. All 26 saved prompts and passage arrays were rebuilt offline and verified byte-identical at final code `f6450e69f0`; the changes affect contextual Jobs calls and synthesis, not the CLI's three-argument extraction prompt. No repeated paid QA calls were needed for unchanged prompts.

| Retrieval mode | Cases with all expected pages | Expected pages found |
|---|---|---|
| Text | 11/13 | 15/17 |
| Vector | 10/13 | 13/17 |
| Hybrid | 12/13 | 16/17 |

Independent source review classifies 19 responses as supported, two as faithful context-limited abstentions, one as a material claim error, one as an unsupported author field, one as a material omission and two as supported with minor issues:

- R03 retrieved assigns a quotation to a 2019 report although the supplied context introduces the 2020 report.
- R04 reference invents “Diane Katz Hazelton” in one raw author field. Its summary/disagreement claim is supported. CLI QA does not run the chapter job's conservative author normalization.
- H03 retrieved omits the explicitly supplied agency firewall; reference-page QA includes it.
- R02/R05 retrieved honestly abstain when selected chunks lack the requested facts. R05 finds the expected page without finding the answer, showing the limit of page-recall metrics.
- Minor issues remain in R04 retrieved attribution and H08 reference wording.

No result was retried to improve these counts. Human labels remain unfilled; these notes are not an automatic accuracy score.

## UI, review and portability

**Workspaces → Manage server Workspaces → Open** loads CLI-created outputs from canonical server state. A generic workspace reload may retain an older local artifact list, which Complete Pack exports. The CLI guide documents reopening canonically and verifying that the intended output is visible. A brief initial manager-shell observation did not reproduce; no route repair is claimed.

Actual canonical reopen verifies incomplete-portion and overview-only warnings. The original failed artifact retains `needs_revision` and five prior corrections. A new, clearly labeled agent-proposed Agriculture correction on old Job 43 survives reopening with its five valid IDs while the original generated chapter remains unchanged. Human acceptance is still pending. Reconnecting to a running saved job displays progress without a generation POST.

Actual browser inspection confirms “Delete output” for a running card and “Delete failed output” for a failed card. The context-limit guidance is covered by a canonical-restoration regression; no synthetic failure was written into the live workspace to manufacture a UI pass.

The final reader exposes the overview and all 30 chapters. Eight actual review saves advance the artifact from version 152 to 160; canonical API readback and browser reload retain all corrections while the entire original analysis remains unchanged. The new artifact stays `needs_revision`. A source citation opens the original PDF at physical page 872 / printed 839.

The actual UI exports a 36,008,084-byte complete pack with SHA-256 `729df241dca848d65b1a0060900018a7b287d0ee4bd791740e39d0c9e18cc71b`. The existing CLI pack stage exits 0 and verifies all 155 manifest entries and the original PDF. Its `browser_restore_verified: false` correctly records that the CLI itself does not perform a browser restore. A separate known-secret scan finds none of five configured values across 161 outer/nested archive files.

Actual destination UI import creates workspace `95162c2c-8e05-4d3b-b1fd-aba221cd3b02`, remaps Media 1 to 2 and assigns a new workspace source ID. Independent source-to-pack-to-destination comparison preserves all eight artifacts, 14 reviews and two historical generation-failure diagnostics. Raw summaries, exact heading/note/evidence quotes, offsets and page spans remain unchanged after verified reference remapping; old producer metadata is historical under `importedProducer`. Historical navigation/node IDs remain provenance.

The destination retains eight chat messages, 24 citations and their 20 recorded page spans. The saved FEC Note's title and 52,992-character content match; provenance metadata was absent on both sides. An independent HTTP download returns the original 4,721,064 PDF bytes and checksum. A new browser context recovers all eight artifacts and 14 reviews, shows all eight chat messages and the saved Note, and opens the restored PDF at page 872/922 with its new Media ID. The overview correction is present with no stale-target warning. Embeddings and provider credentials are not migrated.

The offline browser reader offers all 30 chapter links, retains the eight new correction notes separately from the draft, and shows all eight conversation messages with page-specific PDF links. Independent checks verify ZIP CRCs, safe unique paths and 21,149 local HTML links. Section pages intentionally retain raw drafts; saved amendments are in each artifact's overview report. Private evidence is under `.artifacts/political-pdf-luna-grounding/acceptance/` and the separate `luna-grounding-acceptance-pack-20261116` evaluator run.

## Code and CI qualification

Final code qualification: 239 connected backend tests, 188 connected section-UI/generation tests, and 53 connected pack/bundle/service tests on Node 20 pass. Full frontend TypeScript checking passes. Ruff is clean; Bandit reports zero findings and errors in touched production Python. Scoped lint has zero errors and six unchanged Stage1 warnings. Applicable pre-commit hooks, task format/IDs and the shard guard pass. Independent reviews are clear. Overlapping development runs are not summed.

Hosted earlier heads confirm the pre-commit and shard-coverage repairs, build validation, lint, smoke, accessibility/composer and workspace parity checks. A demonstrated frontend-unit shard failure on Node 20 was reproduced: injected jsdom ArrayBuffers were rejected by native WebCrypto, while Node 24 passed. A test-only native Buffer bridge retains real SHA-256 and all archive-integrity assertions, including a new exact subarray hash vector. Browser production hashing is unchanged. All eight frontend unit shards now pass on `f6450e69f0`; the [previously failing shard](https://github.com/rmusser01/tldw_server/actions/runs/43844527051/job/133237391285) passes 130 tests in 2m57s.

Current CI also exposed stale website-RAG/vector-identity expectations and backend test-quality/line-location ratchets. The exact source/page-header expectation is corrected with 61 connected Node 20 coverage tests. Claims/HyDE tests now assert the preserved vector chunk IDs and separate parent/content mapping; 39 connected tests pass, retaining score, deduplication, FTS and collection controls. Their 18 inherited Ruff findings are unchanged; no unrelated cleanup is included.

The backend repair replaces weak doubles with a real local tokenizer and real local retrieval/ranking/audit/ledger collaborators, and checks summary-write version progression at the test boundary. All five ratchet suites plus changed tests pass together: 210 tests, three warnings, zero skips. Three tenant-baseline entries move only to their current line numbers; complete conditional ASTs are unchanged, independently verified, and no allowance is added. Ruff is clean on that scope. Test-only Bandit scans on that scope have zero findings/errors with ordinary test assertions excluded. Applicable hooks and independent reviews pass. Final-head hosted CI still requires observation after publication. Advisory MyPy output is not the cause of the failing backend gate and is not claimed clean.


The scoped-transport and extension-worker tests now positively exercise the already-authorized workspace-source POST, preserving principal/auth/body checks and unsupported PUT controls; 383 and 489 connected Node 20 coverage tests pass respectively. Full types pass, with no new lint findings. A separate JSON-clone guard now documents exactly one legacy import transformation: its custom replacer remaps nested source IDs and normalizes Date/undefined values. A real import regression verifies those semantics and original-bundle immutability; 70 connected Node 20 tests pass. Production import behavior is unchanged.

An admin-backup setup job failed before tests when the PostgreSQL signing-key download reset its connection. The action now allows two retries, retaining HTTPS, HTTP failure handling, 20-second connect/60-second transfer limits and `signed-by`. The 120-second retry window bounds when another attempt starts, not total wall time. Real local-curl tests cover recovery, exactly three failed attempts without subsequent installation, and the preinstalled-tools path. All 97 connected tests pass; actionlint and Ruff are clean, with no new Bandit findings.

The function-size gate required behavior-preserving helper extractions: PDF parsing dispatch (995→959 lines), ingestion chunk-row projection (885→854), synchronous Chatbook response projection (315→289), and section-summary orchestration (540→288). No baseline allowance increases; only obsolete Chatbook function/navigation-getter entries are removed. PDF/ingestion has 106 connected passing tests; Chatbooks has 74, with a further 53 singleton/quality checks. Independent reviews are clear, production Bandit is clean, and inherited lint findings remain unchanged.

Job 46 ran the frozen `f6450e69f0` runtime, before these CI refactors. It was not rerun against later code. The summary refactor instead has exact before/after parity across 73 test cases and 170 invocations: 628 provider requests (including all prompt bytes/options), 797 checkpoint payloads, 571 progress events, final artifacts and exception results match. Credential objects pass through unchanged and are excluded from trace capture. The fixed-seed trace SHA-256 is `99eee91ed8e6fb434534f2a9f8883998c06655d4585b9fc3912203de88469a01`; the 239 connected backend cases also pass after refactoring. No prompt, provider option, validator or retry behavior changed, and no additional paid model calls were made. The combined function-size, singleton, tenant-scope, test-quality and task-format/ID gates pass: 94 tests, eight warnings and zero skips; the repository task-ID scanner also passes. A fresh complete-PR Bandit scan at `ae466450d2` covers all 54 changed production Python files (52 application modules and two evaluation scripts) with zero findings and zero scan errors.


The next published head, `ae466450d2`, passes the previously failing backend ratchet groups (97, 1,153, 194 and 496 cases) but exposes a later stale OpenAPI fingerprint. Canonical local export exactly reproduces hosted hash `a7bb84ac68b8f5f0a38002669cbdaea6aed8b1ca8166b246a90ebe947ce85459`: 2,144 paths and 3,324 schemas. The added public models are `ImportIdentityMaps` and `ResearchWorkspaceSectionSelection`, both already implemented and tested in this PR. The contract also records existing branch changes to workspace operations/settings/navigation; an unchanged path count does not mean unchanged operations. The snapshot repair uses the existing canonical exporter and type generator, with no new runtime or model behavior. ADR-069 governs the generated artifact owner; no new ADR is required. Regeneration changes only the fingerprint count/hash; `paths.d.ts` is byte-identical. The exact drift check, full frontend types and hooks pass, with independent review clear. Connected API/contract checks pass 197 cases; 155 PostgreSQL variants are skipped by the existing availability fixture, with six warnings. No fixture or test guard is weakened. On `ae466450d2`, all six other required gates pass; the next head must verify the repaired contract gate.


Hosted recovery is verified on implementation head `99a16d63c542adcfbe58f74cd54b148c04c920b8`: all seven required gates and all eight other previously failing jobs pass. The repaired [backend-required gate](https://github.com/rmusser01/tldw_server/actions/runs/43867381583/job/133308088190) passes in 13m13s; the last targeted [shared-UI 31/32 shard](https://github.com/rmusser01/tldw_server/actions/runs/43867381790/job/133299977751) passes in 4m42s at 13:14:42 UTC on November 16. The remaining full matrix is distinct: the observation records 161 successful, 159 queued, four running and 36 skipped jobs, with no failures. Pending jobs are not claimed as passes. Final closeout changes only documentation and task records; the tested implementation stays identical. CI status after that bookkeeping commit remains visible on PR #3480.

TASK-15505.8 delivery/targeted CI qualification is complete. The completed task-owned execution plan is removed per repository guidance; its historical version remains at the tested implementation commit. The design and this report remain durable records. TASK-15505.12 stays open on its Financial attribution criterion, and TASK-15505.6 retains human/VIP acceptance and the NN/g follow-ups below. No merge or deployment occurs.

## Nielsen Norman Group follow-up

These continue the heuristic IDs and severity scale from the first Luna report.

| Issue | Heuristics | Severity | Current status |
|---|---|---|---|
| UX-PDF-66: invented contributors and incomplete claim support | H2 match with the real world; H5 error prevention; H9 diagnose/recover | 3 | Context transport is repaired; final review covers all 30 chapters and the overview. Financial's note-53 attribution and citation-support findings remain open under TASK-15505.6; eight agent-proposed review overlays survive export/restore. |
| UX-PDF-67: misleading or missing overview-failure guidance | H1 visibility of status; H5 error prevention; H9 diagnose/recover | 3 | Strict ID vocabulary, bounded diagnostics and restored-card warnings implemented. Actual retry preserves completed work; oversized author-note context now gives a specific recovery action. |
| UX-PDF-68: running output labeled failed by its delete control | H1 visibility of status; H4 consistency | 2 | Fixed, tested and observed in the browser; confirmation and reconnect behavior remain unchanged. |
| UX-PDF-69: repeated CLI outputs are difficult to distinguish | H1 visibility of status; H6 recognition rather than recall | 2 | Fixed under TASK-15505.13: saved model/provider, creation time, selected-section count/title preview and import/missing-data labels. Actual eight Luna outputs and imported/malformed metadata regressions verified. |
| UX-PDF-70: full-book duration and usage expectations | H1 visibility of status | 2 | Fixed under TASK-15505.13: observed 52–62 minute example is explicitly qualified, with saved-progress/reading guidance and truthful unavailable usage/cost. Actual creation view and running-view regressions verified. |
| UX-PDF-71: completed chapters unavailable during generation | H1 visibility of status; H3 user control | 2 | Fixed under TASK-15505.14: read completed chapters through existing saved checkpoints, retain selection, keep review writes disabled while generating. Controlled isolated desktop/mobile checkpoint→terminal walkthrough passes; no new paid model run is claimed. |
| UX-PDF-72: document reader lacks a chapter outline | H6 recognition rather than recall; H7 efficiency | 2 | Fixed under TASK-15505.15: generated-navigation fallback preserves embedded-bookmark precedence and resolves exact physical pages lazily. Actual restored PDF exposes 46 entries; Education 319→352, FTC 869→902, invalid entry does not jump. Mobile/keyboard checks pass. |

The [November16 UX completion report](POLITICAL_PDF_NNG_COMPLETION_2026_11_16.md) records implementation, source-preservation, visual/contrast evidence, tests and the subsequently discovered broad CI failures. These four interface fixes do not close UX-PDF-66, human labels or VIP acceptance.
