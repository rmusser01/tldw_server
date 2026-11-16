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
