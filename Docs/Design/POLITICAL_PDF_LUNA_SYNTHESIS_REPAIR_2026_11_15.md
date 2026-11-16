# Luna synthesis repair

TASK-15505.12; user explicitly authorized repair, repeatable Luna evaluation and PR updates. CI follow-through uses TASK-15505.8. This is a bounded repair of existing Jobs/source-evidence behavior, not a new generation service.

## Evidence and intended behavior

Job 38 retained all 30 chapters and 117 extraction portions, but rejected the first overview reduction for an unknown evidence ID. Job 37 failed at a different reduction group. Neither stored the rejected response. One exact first-group replay passed the current validator. Citation ranges in old chapter text also leave ambiguous implied interior IDs; this is not established as the original rejection cause. Keep the existing intersection of literal input IDs and registered evidence. Explicitly enumerate that exact vocabulary and require individual IDs, without inferred range interiors or globally valid but unsupplied citations. Save bounded failure diagnostics in the owned artifact, never raw provider text in logs or generic errors. No automatic generation retry is introduced.

The attribution failure is reproducible from saved chapter text. Fixed 24,000-character portions lose earlier bylines; missing attribution becomes an invented contributor boundary. Reuse exact navigation resolution and literal heading excerpts from the selected source section and its contribution descendants. Do not classify person names or assign the first author to a whole chapter. Only verified contexts overlapping a processing portion enter extraction. Contextual author validation must use the specific cited passage range, not any author appearing elsewhere in a mixed-author batch. Parent headings must not authorize their first author across resolved sibling contributions. Ambiguous/unresolved headings and shared endnotes must not create inherited attribution.

Keep original batch boundaries, source bytes, exact evidence spans and checkpoint source/navigation guards. Give synthesis explicit ordered processing portions and verified heading context; state that a processing boundary never establishes a new contributor. Original drafts and reviewer overlays remain separate. New live output must not overwrite the failed original.

The first fresh full run exposed a further prompt ambiguity: one DHS portion cited the workspace source UUID from heading metadata for a heading-only claim. Retain complete provenance in saved contexts, but render only fields needed to interpret attribution and ranges. Explicitly restrict summary/claim evidence to the current numbered SOURCE passages, list their exact P IDs, and state that heading excerpts are attribution hints rather than additional citable passages. Structured failure diagnostics must report the supplied P-ID vocabulary and actual rejected values, with the same bounds and privacy protections. Preserve the first failed run; any deliberate checkpoint retry is separate evidence and processes only failed work.

The output card must distinguish an overview-only failure from incomplete sections, including previously cached generic errors. Preserve specific checkpoint, permission, provider and save errors.

## Scope and ownership

- Existing `section_summaries.py`: extraction/synthesis prompts, current-call citation vocabulary, bounded owned-artifact diagnostics and context wiring.
- A focused `Research_Workspace/section_attribution.py` helper, if needed to keep navigation/context rules testable; no dependency or API addition.
- Existing Studio generation/card projections and their tests for overview-only wording.
- Research Workspace regression tests; existing CLI/config runner and immutable new run directories for verification.
- Existing CI assignments/contracts and only demonstrated failing checks; no baseline expansion.

## Verification

Red-first tests cover late single-author continuation, real author changes within a page/batch, selected-child scope, ambiguous headings, endnotes, current-call citation restrictions, bounded diagnostics, unchanged quotes/offsets and old checkpoint recovery. Test actual generation-to-card behavior for overview-only failure. Run connected tests, Ruff/ESLint/types where touched, Bandit, independent review and pre-commit. Use a fresh Luna 30-chapter output plus the frozen 26-response QA matrix; retain first failures and source-audit attribution/citation support separately from structural validity. A copied-evidence diagnostic replay may isolate synthesis but must be labeled as such and cannot claim fresh extraction.

## ADR assessment

ADR required: no. ADR-003 (Jobs), ADR-007 (Workspace), ADR-015 (Evaluations), ADR-025 (provider gateway) and ADR-065 (provenance) retain their owners and authority. Additive diagnostics remain in existing owner-scoped artifact metadata; there is no new store, provider bypass or review authority. Human labels and VIP acceptance remain separate from agent verification.
