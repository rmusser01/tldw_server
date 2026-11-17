# Political PDF semantic acceptance follow-through — November 17, 2026

Tracking: TASK-15505.12 and TASK-15505.6. [Draft PR #3480](https://github.com/rmusser01/tldw_server/pull/3480), against `dev`. This report extends, rather than replaces, the [November 16 strict Claims retest](POLITICAL_PDF_STRICT_CLAIMS_RETEST_2026_11_16.md).

## Current qualification

The overview coverage repair and exact footnote linkage are implemented. Twelve fixed Luna controls meet their expected outcomes again. Fresh Job 56 completes EXIM and Financial, cites both chapters in its overview, and preserves note 53's chapter connection. Strict semantic acceptance still fails. Source review identifies a separate Claims aggregation gap: the correct contributor contexts lose their association with individual citations in the judge prompt. A bounded shared fix and three new versioned controls address that handoff. Full-book qualification remains pending.

These are agent source reviews, as requested. Human labels remain null and human calibration is not computed. Actual VIP usability observations remain pending. No completed Job, structural pass, or automated support count constitutes acceptance of all generated statements.

## Frozen source and runs

The original 922-page PDF is reused, with SHA256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. This round verifies the existing ingestion and canonical page bindings; it does not claim a new upload. The source workspace is `81bab5d0-2af0-480c-83d3-f4092760a3d5`, Media 1, navigation `media_1:v4:ec5d2af126`.

| Run | Scope and result |
|---|---|
| `luna-semantic-20261117` | First v2 controls: 11/12 expectations met. Literal `[53]` in the positive claim was parsed as an evidence citation and split the claim. The fixture and failed run are retained. |
| `luna-semantic-v3-20261117` | V3 changes that literal to “body marker numbered 53”; evidence and expectations are unchanged. Source checks and all 12 controls pass. Full retrieval, QA and summary stages have not been submitted in this ledger. |
| `luna-semantic-targeted-v3-20261117` | Job 55: Federal Reserve and Financial Regulatory Agencies, five portions, two completed chapters, overview cites 2/2. Canonical readback, selected scope and exact evidence spans pass. Strict semantic acceptance fails. Four H05/H08 QA answers retain their checks and first results. |
| `luna-semantic-targeted-binding-20261117` | Job 56: EXIM and Financial, five portions, two completed chapters and overview cites 2/2. Structural checks pass; strict semantic acceptance fails. Twelve controls pass; all eight R03/R04/H05/H08 QA answers retain first results. API, worker and runner are frozen at `0bb067f229`. |

Directories are under `.artifacts/political-pdf-evals/`; their ledgers, manifests and request/response files are local evidence, not tracked repository artifacts. API and worker production revision is `bba21d65e1`; the v3 runner revision is `cf0634beddf35b951ecb45ec95f2c306d862b66a`. Runtime sidecars retain production file hashes. Model: OpenAI `gpt-6-luna`, output budget 16,384, with existing settings and no automatic paid retry.

The preceding runtime identities apply to the first three rows. The fourth row records its newer frozen runtime separately. No failed result is replaced by a rerun.

The targeted configuration mistakenly selected Federal Reserve where the preparatory description said EXIM. Its actual scope and results remain unchanged. The full configuration was independently resolved against navigation and includes all 30 chapters, including EXIM and Financial. The targeted summary does not qualify EXIM.

## Source review

All 12 controls were checked against canonical source spans. The broader note-link denial is rejected by the model, but its rationale misses the verified chapter category; agent review records that disagreement rather than treating the expected outcome as proof of sound reasoning. Narrow absence of a name inside note 53 remains distinct from denial of its verified contribution link.

All 25 targeted QA findings were independently source-reviewed. The original model counts are 19 supported and six unsupported; agent proposals are 16 supported and nine needing revision. Those nine include ambiguous wording and citation boundaries, not nine fabricated facts. Findings include two clear paraphrase overstatements, two truncated citations, three conditional-exclusivity ambiguities and two document-wide absence scopes. The verifier gives inconsistent judgments on similar conditional wording and abstention claims.

The Financial chapter and overview contribute another 34 reviewed units, including supported and rejected model findings. All 63 cited-passage occurrences resolve to canonical text. Agent proposals are three supported, 24 needing revision and seven non-factual headings or lead-ins. Many negatives describe material present nearby in the chapter but attach an insufficient excerpt. A model-supported overview claim loses qualifications; a bibliography limitation rejected by the model is source-supported. These are bounded source reviews, not acceptance of the unreviewed Federal Reserve chapter or all extraction batches.

The packet keeps exact claim spans, cited passages, receipt identity and immutable row fingerprints. Agent decisions occupy `agent_review`; the original automated finding and independent `human_review` remain unchanged. Regenerating reports must preserve only decisions whose evidence is unchanged.

Private audit receipts are under `/tmp/tldw-political-pdf-semantic-20261117/`: `control-source-span-audit.json`, `targeted-qa-agent-audit.json`, `targeted-summary-agent-audit.json` and `targeted-footnote-cause-audit.json`.

Job 56 adds 47 independently reviewed QA findings: 28 agent-supported and 19 needing revision. Exact claims, offsets, source passages and original judgments match the review packet; its regenerated report preserves all 47 agent decisions and zero human decisions. The new audit is `binding-targeted-qa-agent-audit.json`. R04 attaches a Trade passage from Kent Lassman's contribution to de Rugy and the verifier accepts it. Reconstruction in `binding-r04-attribution-context-audit.json` matches the runtime hashes for both correct headings; the shared judge assembly then loses their citation-ID association. The repair labels each deduplicated context with its cited IDs and valid source ranges, counts the wrapper within the existing limit, and invalidates affected cached checks through the existing evidence digest. Old reports remain historical evidence.

The QA generation requests contain numbered body passages without the verifier's additional heading context. Some attribution abstentions reflect that input difference. The CLI QA branch uses the extraction-prompt contract; it is not a claim to test the browser's conversational prompt. V4 retains all twelve earlier fixed controls and adds the original wrongly accepted mixed-contributor claim, a corrected-contributor counterpart and a Hazelton positive control. Their expected outcomes are agent-reviewed development expectations.

## Generation handoff finding

Job 55 extraction labels note 53's OTS-to-OCC statement as having an unclear relationship to the section author's position. Chapter prose broadens that caveat to say it cannot safely be attributed to Burton or Bowes, and the overview repeats it. The verified source link is to the CFPB contribution headed Robert Bowes; that link does not establish personal note authorship or endorsement.

Reconstruction confirms the extraction and chapter prompts receive the note links within existing budgets. Runtime evidence digests and the overview input hash match. A post-hoc rendering difference for reference-tail contexts comes from JSON key ordering in saved dictionaries; rebuilding the source contexts matches the recorded digests. There is no evidence here of lost transport or corrupted receipts.

The chapter's 28,826 characters of input trigger reduction. Both reduction and final synthesis retain footnote context, but their saved evidence IDs lacked individual source coordinates. Whole-portion coordinates cannot reliably bind `E2.3.6` to note 53. The repair supplies only validated, unambiguous note-overlapping evidence ranges allowed in each current call, retaining them outside model compression. It adds no policy evidence or personal author assignment. Reconstructing Job 55's inputs now gives the correct `[2499297,2500202)` binding in the relevant reduction and final call within 13,052 of 16,000 context characters; existing Claims contexts and overview author notes are unchanged. This reconstruction uses explicitly synthetic compression outputs, not a claim to have recovered the original intermediate model responses.

Qualification: 291 connected backend tests, 136 independent focused tests, and a separate 167-test overview/evaluation regression run pass; these suites overlap and are not summed. Ruff and scoped Bandit are clean. The two UI fixes pass 132 connected tests and 26 independently rerun focused tests, full frontend TypeScript, and project ESLint with only three unchanged warnings. The CI shard guard reports no newly uncovered test files. These checks qualify code behavior, not new model output.

The chapter verifier also names note 54 when explaining the transfer contradiction, although the transfer is in note 53. Its reason is retained as a model error, not copied into the source review as fact.

In Job 56, extraction `E2.3.5`, Financial claim 15 and overview claim 11 describe the OTS-to-OCC transfer as a chapter-level endnote proposal. They no longer deny the chapter connection or invent personal note authorship. Seven other units are explicitly unchecked because complete quoted evidence plus attribution context exceeds 16,000 characters; overview claim 8 requires 16,409. These are enforced limits, not provider failures. The contributor-binding fix does not raise that ceiling or turn unchecked findings into support.

Independent review covers all 44 Job 56 chapter/overview units: four supported, 34 needing revision and six structural headings or lead-ins. All 82 cited occurrences match canonical text. The overview's transfer clause is supported, while another clause in the same unit has a funding-cap wording ambiguity. All 91 QA and summary agent decisions survive offline regeneration; human labels remain null. The shared Claims binding fix passes 304 connected tests, 107 independently rerun focused tests, Ruff, Bandit and independent review. V4 passes the 167-test connected evaluation suite and an independent canonical-source audit. These overlapping checks qualify implementation and fixture integrity; new model results remain pending.

## Browser review and export

The actual Job 55 reader displays “Chapters cited in overview: 2 of 2 completed chapters” separately from citation support. Expanding findings shows saved judgments and PDF links. An explicitly agent-authored `needs_revision` note was saved on the overview, reloaded through the requested workspace route and downloaded in the reviewed Markdown.

Canonical readback advances the artifact from version 19 to 20 while preserving its original content and complete section analysis, including all eight verification reports. The reviewed download contains the note and support findings; SHA256 `86377378e766ac5efb92ad857ad65d6c1f2aae0013ca594e21cb8440fc0dde85`. Clicking the saved `E2.3.6` citation opens Media 1 at physical page 876 of 922, visibly showing printed page 843 and note 53. Initial clicks overlapped development hot reload and produced a delayed duplicate tab; no lost navigation data was observed. The document tour also emits existing Ant Design notification deprecation warnings. Receipt: `targeted-browser-review-receipt.json` in the private audit directory. Fresh full-book offline export and destination vector checks remain pending.

## NN/g follow-through

Findings use [NN/g's ten usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) and [severity scale](https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/). Severity here is an agent assessment, not a participant rating. Existing IDs and historical evidence remain in the earlier reports.

| Finding | Heuristic / severity | State and evidence |
|---|---|---|
| UX-PDF-95: completed chapter processing can conceal an omitted overview chapter | H1 visibility; H5 error prevention / 3 | Implemented and regression-tested. Frozen Job 52 exposes Financial missing (29/30); new Job 55 has 2/2, visible independently from semantic support in the reader and export. Full-book live qualification pending. |
| UX-PDF-96: verified endnote context loses its binding during chapter synthesis | H2 match; H5 error prevention / 3 | Exact note links and strict checks are implemented; the fresh run exposed the evidence-ID coordinate handoff gap. Per-call bindings now pass reduction, source, scope, budget and completed-work controls. Fresh model qualification pending. |
| UX-PDF-97: a missing local chat reference opens a developer error screen during review | H1 visibility; H9 recovery / 2 | Fixed with a narrowly classified nonblocking warning for the ordinary read. Persistent writes, fork guards and real storage errors remain strict. Deterministic missing-history controls pass; the preserved `e38f1677…` workspace reopens without an overlay. The original stale-selection transition is not claimed as independently replayed. No production crash or data loss is established. |
| UX-PDF-98: “Support safe” can be mistaken for citation support | H2 familiar language; H4 consistency / 2 | Fixed. The actual Job 55 card now says “Not redacted” beside “Needs Revision,” with an explanation separating redaction metadata, citation checks and reviewer decisions. Unknown states, neutral styling and restricted-detail hiding are regression-tested. |
| UX-PDF-99: correct contributor context can be attached to the wrong cited passage during verification | H5 error prevention / 3 | Job 56 R04 reproduces the error. Correct source contexts are present, but shared prompt aggregation drops their citation-ID association. A bounded binding fix and versioned positive/negative controls are in qualification. |
| UX-PDF-66: generated analysis still contains attribution, wording and citation gaps | H5 error prevention / 3 | Open. Strict checks and agent reviews expose both generation problems and verifier limitations; they do not make every statement reliable. Human calibration and VIP acceptance remain pending. |

## Remaining acceptance

Complete the bounded generation fix, freeze a new runtime, retain the first failed runs and run the full Luna evaluation. Review the selected fresh claims, exercise the new full export, finish restored Media 2 vector indexing and verify scoped retrieval without changing saved research. The [seven-task VIP script](POLITICAL_PDF_VIP_WALKTHROUGH_2026_11_17.md) and [blank observation sheet](POLITICAL_PDF_VIP_OBSERVATIONS_2026_11_17.csv) are prepared; actual task success, time, assistance, errors and confidence must come from the VIP.

The PR remains draft. On `cf0634bedd`, the extension XS-07 conversation-deletion test fails to find its confirmation dialog; the saved screenshot shows an open context menu instead. Its test and component are unchanged from the previous baseline. A scoped trace investigation is in progress; the first failure is retained. Other current-head checks are queued or running. Older green checks do not qualify a newer commit. No merge or human-written Change summary is supplied by the agent.

ADR assessment: no new ADR; the existing proposed ADR-096 and ADR-003/007/015/025/065 boundaries govern these repairs. See the [design](../Design/POLITICAL_PDF_SEMANTIC_ACCEPTANCE_2026_11_17.md) and [implementation plan](../superpowers/plans/IMPLEMENTATION_PLAN_political_pdf_semantic_acceptance_20261117.md).
