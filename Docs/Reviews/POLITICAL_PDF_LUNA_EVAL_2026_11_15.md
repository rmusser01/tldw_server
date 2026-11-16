# Political PDF: GPT-6 Luna and repeatable evaluations

Tasks: TASK-15505.10 and TASK-15505.11. Extends the [DeepSeek walkthrough](POLITICAL_PDF_FOLLOWTHROUGH_2026_11_15.md) and [frozen claim review](POLITICAL_PDF_CLAIM_REVIEW_2026_11_15.md). Execution began November 15 Pacific / November 16 UTC. This report is in progress; no human semantic acceptance is implied.

## Scope and controls

The source is the same 922-page, 4,721,064-byte PDF, SHA-256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. OpenAI's exact model ID is `gpt-6-luna`; authenticated model lookup and the application's completion response both confirm it. Reasoning remains at the provider default (medium). The adapter removes unsupported sampling parameters and uses `max_completion_tokens`, preserving the requested 8,192-token budget. This differs from the previous DeepSeek configuration; results are observations, not a controlled causal claim about model superiority.

The CLI freezes 13 existing questions and page fingerprints. H05 is a tuned regression. Each model receives both reference-page and retrieved-context questions using the production extraction prompt, without expected answers in the prompt. Reports keep raw first responses, failures, exact requested settings, returned model, token usage when available, and latency. Semantic review remains separate from mechanical validation. See the [runner guide](../Development/POLITICAL_PDF_MODEL_EVALS.md) and [design/ADR assessment](../Design/POLITICAL_PDF_MODEL_EVALS_2026_11_15.md).

## Observed walkthrough

| Step | Observation | Limit |
|---|---|---|
| Provider compatibility | First app call failed with generic `Invalid request`; direct diagnostic identified unsupported `max_tokens`, followed by unsupported sampling. After the adapter fix, the same app request completed in 2.411 seconds. | First failures retained; no model substitution or forced reasoning disablement. |
| Browser upload | Re-uploaded the same PDF through Add Sources. Background job completed in 77 seconds with 922 pages and retained original. Reopening the dialog recovered the saved upload receipt. | Duplicate detection reused the existing Media item with overwrite disabled. |
| One-section summary | Federal Election Commission summary completed with 28 evidence items. The reader distinguishes author positions, proposed changes and unsettled authority. | Agent inspection is not human source acceptance. |
| Full summary | Separate selection contains all 30 numbered chapters and 117 portions; checkpointed generation is in progress. | Front matter and other unselected introductions remain outside scope. |
| Cited QA | CFPB answer retained the Supreme Court condition; vaccine-safety answer named FDA and associated funding. Four messages have canonical server IDs and 12 attached retrieval sources. | Attached retrieval evidence does not establish claim-level citation precision. |
| Review | A review-save attempt retained the draft but reported a stale-output conflict despite matching version numbers. Generation had omitted canonical metadata, causing the strict review guard to reject the payload. | Producer repair underway; no false accepted label was written. |
| Notes | The FEC draft was explicitly saved to a new workspace Note and appeared in its pinned Notes list. | A saved draft does not imply factual acceptance. |

## Frozen question results

The corrected run at `b6b285147b` retained all 26 first responses, with no QA retries: 13 reference-page and 13 retrieved-context answers. All returned `gpt-6-luna`, finished normally, and passed JSON/evidence-ID checks. Reported usage was 45,095 prompt tokens and 20,721 completion tokens, including 13,036 reasoning tokens. Median completion latency was 7.694 seconds; these are serial local observations with a separate summary job running, not a load benchmark.

| Retrieval mode | Cases with every expected page | Expected pages found | Median seconds |
|---|---:|---:|---:|
| Text | 11/13 | 15/17 | 1.232 |
| Vector | 10/13 | 13/17 | 0.290 |
| Hybrid | 12/13 | 16/17 | 1.348 |

All results stayed within the selected source. R02's expected physical page 909 was absent in all three modes. Page coverage is not semantic recall: R05 retrieved the expected page number but omitted the numerical passage needed to answer the question.

Agent source audit classified 22 answers as broadly supported, one as material misattribution, one as a bounded omission, and two as retrieval/context limitations. Minor wording and attribution concerns remain itemized in the private audit; 22 is not an all-claims-perfect accuracy score. Human labels remain unfilled.

| Case/context | Finding |
|---|---|
| R03/reference | Correct report years and arguments, but invented author “Richard A. Hazelton” twice. The original PDF credits Jennifer Hazelton on physical page 757; the supplied reference passages did not support expanding her name. |
| H03/retrieved | Omitted the proposed CDC firewall even though the retrieved context supplied it. Added source-supported material was less relevant to the question. |
| R02/retrieved | Regional-office passage missing. Luna appropriately abstained from naming the requested agency. |
| R05/retrieved | Requested 2007/2014 authorization figures missing. Luna acknowledged that limitation instead of inventing them. |

The [prior DeepSeek audit](POLITICAL_PDF_CLAIM_REVIEW_2026_11_15.md) used a different runtime configuration and did not contain this same two-context matrix. These results support case-level comparison and future controlled runs, not a model ranking or a claim that either model is safe for unattended dissemination.

## NN/g issue tracking

Agent heuristic evaluation uses [NN/g's ten heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) and [0–4 severity scale](https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/), considering frequency, impact and persistence. These are not VIP participant observations.

| ID | User consequence | Heuristics | Severity | Disposition |
|---|---|---|---|---|
| UX-PDF-63 | Selecting a supported GPT-6 model produced a generic invalid-request error because the adapter sent incompatible token/sampling parameters. | H1, H4, H5, H9 | 3 | Fixed in the shared native OpenAI adapter; actual app completion and chapter job passed. |
| UX-PDF-64 | A cross-page retrieval card displays the first page only; some vector cards lack a direct PDF link. The default answer prompt also does not request inline source markers. Correct answer text is harder to verify. | H2, H4, H6 | 3 | Range rendering/persistence, authoritative Media type enrichment and default citation instruction under repair. First UI response retained. |
| UX-PDF-65 | Saving a review on a freshly completed output reports a conflict with the same server version. The draft survives, but the reviewer cannot finish the task. | H1, H3, H9 | 3 | Producer now retains canonical metadata; connected generation-to-review tests pass and the review guard remains intact. Live full-job review is pending. |

## Verification and limits

Native OpenAI compatibility: 228 connected tests passed, targeted Ruff clean, Bandit zero findings, independent review clear. Runner/report: 62 connected offline tests passed; these exercise interrupted requests, immutable resume identity, exact source scope, complete summary coverage, partial failure, archive integrity and protected evidence paths. Test counts overlap other qualification runs and must not be summed into a unique total.

The first live CLI run at `7029d9c08c` passed source checks, retained 39 retrieval responses and one reference-context answer, then stopped before retrieved QA. The runner had omitted the app's explicit `fts_level: chunk`, allowing a whole-book lexical result that its scope guard rejected. This is an evaluator integration failure, not a Luna answer-quality failure. Its immutable evidence remains in `.artifacts/political-pdf-evals/luna-20261115`; no oversized book context was sent. The corrected run will have a separate code revision and evidence directory.

Complete-summary, corrected CLI, review persistence and export results will be added after execution. Summary provider-call usage is unavailable from the public Jobs API. Human source labels, real VIP usability acceptance, full repository CI, merge and deployment remain outside completed verification.
