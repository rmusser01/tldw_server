# Luna overview and attribution repair: retest

TASK-15505.12; CI follow-through TASK-15505.8. This report extends the [first Luna evaluation](POLITICAL_PDF_LUNA_EVAL_2026_11_15.md), preserving its original failed outputs and five review overlays. Implementation and live rerun are in progress. Human labels remain pending.

## Baseline and diagnostic

The same 922-page PDF has SHA-256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. The model remains OpenAI `gpt-6-luna`, provider-default reasoning, requested 8,192 output-token budget and existing 2,048-token reduction cap. The frozen 13-case fixture and both QA context modes are unchanged. Previously failing cases now guide regression testing; they are not untouched held-out validation.

Jobs 37 and 38 rejected unknown evidence IDs during different overview reduction groups. Neither retained the offending response. One reconstructed Job-38 first-group request used the original production prompt and settings: 16,850 input characters, 94 literal registered IDs. It completed in 12.019 seconds, returned `stop`, and cited 26 valid IDs with no unknown reference. Usage was 4,198 prompt / 894 completion tokens, including 294 reasoning tokens. This single successful diagnostic does not explain or erase the original failures.

Saved chapter text includes citation ranges whose interior IDs are absent from the literal current-call vocabulary. That is a concrete ambiguity, not a proven cause of either original rejection. The repair retains the strict current-call intersection and asks for individually enumerated exact IDs; it does not widen permission to all document evidence. Bounded rejection diagnostics will retain the first failures in the owned artifact, with stage/group, response hash, truncated preview and unknown IDs. Generic errors and server logs remain free of raw response text.

Saved source evidence confirms the attribution failure: a fixed processing boundary can remove a byline from later extraction input, then become a false unnamed-contributor claim during synthesis. The [repair design](../Design/POLITICAL_PDF_LUNA_SYNTHESIS_REPAIR_2026_11_15.md) carries exact source-heading excerpts across applicable ranges, preserves real contributor boundaries, and treats unresolved headings and shared reference material conservatively. It does not classify names or automatically inherit a first author.

## Verification record

Code qualification: 199 connected backend tests and 282 connected frontend tests pass. Full frontend TypeScript checking passes. Ruff is clean; Bandit reports zero findings and errors in the touched production Python. Scoped ESLint has zero errors and 31 warnings identical to the prior head. All applicable pre-commit hooks pass, task format/IDs pass, and the shard guard reports zero uncovered files. Independent reviews are clear. Overlapping development runs are not added to these counts.

The exact retained source confirms the intended contributor ranges for HHS, DOJ, EXIM, Trade and Financial Regulatory Agencies. FTC has a malformed later outline entry: its verified body and reference prefix remain usable, while the tail beginning at the verified afterword heading remains unresolved. No author is inherited across that uncertainty. Live reload of the old failed artifact now shows the overview-only warning, preserves `needs_revision` and retains all five saved correction targets.

The fresh full-chapter and 26-QA configuration is separate from prior evidence and does not adopt or overwrite the failed outputs. Results, independent source audit, current-head CI and remaining limits will be recorded after execution. Human semantic acceptance is not implied by source integrity, citation membership, completed Jobs or automated test results.

## Fresh evaluation at `282094cfe9`

The separate EXIM smoke, Job 40, completed one chapter, both extraction portions and its overview in 120.584 seconds. Selected scope, exact evidence spans and canonical artifact readback pass. The chapter and overview consistently distinguish Veronique de Rugy's abolition argument from Jennifer Hazelton's retention argument across the processing boundary; neither invents a third contributor. This supports the targeted continuity repair.

Independent source review still finds incomplete citation coverage within some otherwise source-supported paragraphs. For example, the exposure figures cite `E1.1.3`, while their support is in `E1.1.4`/`E1.1.5`; the sovereign-guarantee claim cites `E1.1.30`, while its support is in `E1.1.31`. Exact evidence IDs do not establish that every adjacent claim is supported. These remain semantic-review findings, not reasons to widen the citation validator.

The full rerun is separate Job 41, using all 30 selected chapters and the unchanged 117 processing portions. The frozen QA matrix uses a separate run directory so both operations can progress independently. Evidence is retained under `.artifacts/political-pdf-evals/luna-grounding-exim-20261115`, `luna-grounding-full-20261115` and `luna-grounding-qa-20261115`; human labels remain unfilled.

All 26 QA first responses completed without retry and passed shape, citation-ID and finish checks. Provider usage totals 46,291 prompt plus 17,859 completion tokens, including 11,119 reasoning tokens. QA request time totals 188.996 seconds, median 7.116 seconds; the full source/retrieval/QA ledger spans 218.853 seconds. These are observations with a separate summary worker active, not a load benchmark.

| Retrieval mode | Cases with all expected pages | Expected pages found |
|---|---|---|
| Text | 11/13 | 15/17 |
| Vector | 10/13 | 13/17 |
| Hybrid | 12/13 | 16/17 |

The independent agent audit classifies 19 responses as supported, two as faithful context-limited abstentions, one as a material claim error, one as an unsupported author field, one as a material omission and two as supported with minor issues. These are review notes, not calibrated accuracy scores or human labels. In particular:

- R03 retrieved assigns the quorum quotation to a 2019 report even though the exact supplied context introduces the 2020 report. The error occurs in the summary and evidence claim.
- R04 reference invents “Diane Katz Hazelton” in one raw `author` field. Its summary and disagreement claim are supported. This diagnostic sends the extraction prompt directly; it does not run the chapter job's `contextual_author` validation, which rejects ungrounded author values.
- H03 retrieved omits the explicitly supplied firewall between the proposed agencies. The reference-page answer includes it.
- R02 and R05 retrieved honestly abstain where selected chunks lack the requested facts. R05 covers the expected page while still missing the answer, demonstrating the limit of page-recall metrics.
- Minor issues remain in R04 retrieved name/citation attribution and H08 reference wording (“narrow” versus the source's more precise definition).

The original and new responses are both retained. No QA result was retried to improve these counts, and no human-review label was filled by an agent.

Job 41's first pass rejected one Homeland Security extraction portion at source characters 465,182–489,182. Its retained preview shows a heading-only dismantling-DHS claim citing the workspace source UUID instead of a supplied `P` passage ID. The validator correctly reports `unknown_source_span`; this is distinct from the old overview evidence-ID rejection. The bounded diagnostic preserves the response hash and preview, but its initially E-only counters report zero supplied/unknown IDs for this structured extraction failure. The qualified follow-up removes opaque IDs from rendered heading context while retaining saved provenance, restricts heading excerpts to attribution, enumerates current P IDs and records parsed rejected passage IDs accurately. The fresh connected run passes 203 tests with no skips; Ruff and Bandit are clean and independent reviews are clear. A deliberate failed-work retry is pending; the running first pass continues on its original loaded code.

The browser's **Workspaces → Manage server Workspaces → Open** path loaded both new CLI-created outputs alongside the two older outputs. Canonical restoration exposed an absent client `errorMessage`, which initially hid the old overview warning. A two-file follow-up retains specific-error guards while deriving that warning from the failed artifact's authoritative section analysis. All 174 connected tests, full types and pre-commit pass; lint has zero errors and six unchanged warnings. Independent review and actual canonical reopen both confirm the warning. The first manager navigation briefly showed only the app shell; a later same-context navigation loaded the manager and its Open action normally, so no route defect or route repair is claimed.

Rolling source audit confirms repaired contributor continuity in the first four chapters and HHS. HHS's substantive portions remain attributed to Roger Severino; reference-only evidence stays unassigned. DOJ's main-body continuation remains attributed to Gene Hamilton, and its current Portland passage cites the appropriate `E17.5.5` rather than the unrelated evidence used before.

The wider review also finds an endnote-specific attribution error. Agriculture labels substantive commentary in notes 97–98 a “Separate unnamed contribution” and says it should not be attributed to Daren Bakst. Canonical source verifies the saved ENDNOTES boundary at character 926,030 and all three exact quoted spans; there is no second contributor heading. The extraction summary does not invent a separate author; the unsupported division first appears in the persisted chapter synthesis. DOJ similarly turns unassigned endnote evidence into an unsupported authorship limitation. The synthesis-only clarification passes 205 connected backend tests, with Ruff/Bandit and independent review clear; a separate fresh Agriculture retest remains pending. The tests verify instruction delivery and unchanged conservative author normalization, not semantic model success. This does not justify automatically assigning all reference material to a chapter author, nor does it erase the original generated drafts.

## Nielsen Norman Group issue follow-up

These entries continue the heuristic IDs and severity scale in the first Luna report.

| Issue | Heuristics | Severity | Current evidence |
|---|---|---|---|
| UX-PDF-66: invented contributors and incomplete claim support | H2: match with the real world; H5: error prevention; H9: recognize, diagnose and recover | 3 | Verified heading contexts and the EXIM smoke repair the observed contributor split. Citation undercoverage remains visible and requires source review; full chapter audit is pending. |
| UX-PDF-67: overview rejection hidden by a misleading incomplete-sections message | H1: visibility of status; H5: error prevention; H9: recognize, diagnose and recover | 3 | Exact current-call citation vocabulary, strict validation and bounded rejection diagnostics are implemented. Actual reload of the original failed artifact now states that the chapters are complete and the overview needs retry. Full overview reliability is pending the fresh run. |
