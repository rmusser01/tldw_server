# Political PDF semantic acceptance follow-through

TASK-15505.12 and TASK-15505.6; draft PR #3480 against dev. On November 17 the requester approved, in order: overview coverage, footnote attribution, regression/source review and Luna evaluation, then destination indexing and an uncoached VIP walkthrough. Agent source review is authorized; it is not a human calibration label.

## Observed failures

Frozen Job 52/version 334 has 30 completed chapters, 117 portions and a 7,068-character overview. Its 51 unique evidence IDs cite 29 chapters; only Financial Regulatory Agencies is absent. Current processing counts and valid citation IDs cannot detect that omission. The overview has no supported Claims units, so using semantic verdicts to count chapter representation would conflate two different checks.

Financial claim-21 cites E27.3.2, canonical characters [2499297,2500202), which starts four characters after the label `53. `. The unique body marker `[53]` at [2470570,2470574) belongs to the verified Robert Bowes CFPB contribution. Existing context gives the verifier only an unassigned reference range. Generation says the excerpt does not establish a contributor, and verification accepts name absence within that substring without receiving the source-backed contribution link. Distinguish that literal excerpt limitation from a broader denial of the link; do not force a verdict or claim personal footnote authorship. Note 53 is [2499293,2499600); the cited passage also overlaps later notes, so assigning one byline to the whole passage would introduce another error.

The complete original artifact, source, navigation, reports, model responses and reviewed pack remain immutable in the previous evaluation runs and private audit receipts. New outputs use new run directories. Explicit retry retains completed chapters and existing bounded failure diagnostics; canonical artifact versions retain prior drafts.

## 1. Overview coverage

Add a deterministic, recomputed structural check: each completed selected chapter must have a literal registered evidence ID attached to prose in the overview. Derive the mapping from actual section evidence, never an ID prefix, title guess, model-declared coverage list or trusted saved count. Exclude headings, fenced code and citation-only inventories. Unknown IDs, ranges and ambiguous duplicate bindings cannot satisfy coverage. This establishes cited representation, not semantic completeness or accuracy.

Preserve chapter identity through overview reductions. For each group, require the chapters represented by that group's actual allowed IDs and supply a bounded title checklist. Keep current-call citation allowlists, source scope, shrink checks, round limits and output budgets. Chapter synthesis retains its existing behavior. A completed chapter with no usable citation in its saved summary fails before another paid overview call with guidance to regenerate that chapter/output.

At final synthesis, check every completed selected chapter. Retain an incomplete candidate and expose a specific overview error without marking the artifact complete. Recompute coverage before reusing a saved overview; a matching input hash alone cannot preserve a failed candidate indefinitely. Set a successful synthesis hash only after structural validation. No automatic paid retry is added.

Use existing artifact state and versioning. Derive display/export/evaluation coverage from the exact current overview and saved sections; avoid adding another trusted receipt or source-identity mapping. Show processing counts, overview cited-chapter coverage and Claims support separately. Offline HTML must show an overview error even when a draft exists. Current and earlier retained drafts need truthful wording.

## 2. Footnote attribution

Extend the existing attribution builder with bounded, literal note-to-body-marker-to-verified-contribution relationships. Keep exact coordinates and excerpts, contribution heading and author-note caveats. Carry these relationships through extraction, synthesis, section evidence and the QA resolver. They establish attribution context only; they do not add policy evidence, authorize citation IDs, establish endorsement, or assign one author to every note in a multi-note passage.

Use compact relationships on reference ranges and project them by the cited note span. Chapter extraction/synthesis receives verified links; corrected chapter text carries the result into overview generation. The existing merged evidence map supplies exact links to overview Claims checks. Preserve the overview's existing literal author-note renderer and 16,000-character limit: the original 30-chapter notes already occupy 15,916 characters, so duplicating footnote context there would break the full-book path.

Resolve unique boundaries and backlinks conservatively. Preserve unresolved status for duplicate labels, missing boundaries, repeated markers across contributors, stale navigation, malformed hierarchy, or context overflow. Real reference lists contain inline numbered entries; a line-only parser must not swallow later notes into an earlier one. Shared endnotes require verified chapter ancestry in QA rather than inheriting the last contributor. Existing evidence digests invalidate affected cached checks when context changes.

Regression cases include correct Bowes linkage, wrong Burton attribution, the retained denial, passage/label splits, multiple notes, unrelated bibliography names, collective-authorship notes and selected-child isolation. Freeze a new fixture version; do not rewrite earlier fixtures or receipts.

Visual source audit confirms the body marker on physical page 872/printed 839 and note 53 on physical 876/printed 843. Note 53's content after its label is 303 characters and contains no personal name; the longer original cited passage also contains note 54's bibliography naming David R. Burton. Keep the literal excerpt-limitation control separate from a false denial of the chapter's verified link.

Job 55's fresh retest exposes a further generation handoff gap: source extraction receives per-passage coordinates, but chapter synthesis and its reductions receive saved E IDs with only whole-portion coordinates. Its 28,826-character Financial input needs reduction; the known note relationship cannot be bound reliably to E2.3.6. Add compact registered evidence-ID-to-source-range bindings inside the existing bounded chapter attribution context, only for evidence overlapping verified note ranges. Derive bindings from validated source evidence and filter them to each call's actual allowed IDs. Carry them outside model compression through every chapter reduction and final synthesis. Preserve the 16,000-character context ceiling with explicit omission instead of partial misleading mappings; retain author notes and unchanged body-only context. Do not add a full-book overview map or change policy evidence or personal-author fields. Existing completed paid work remains intact on retry; fresh artifacts qualify the repair. This stays within the existing ADR assessment.

## 3. Evaluation and review

Job 56's R04 QA audit identifies a separate verifier handoff gap: correct de Rugy and Kent Lassman headings arrive with their own evidence, but aggregation removes the evidence-ID association. Bind each deduplicated attribution block to its exact cited IDs and validated source ranges in the shared Claims prompt, counting the wrapper against the existing evidence limit. Keep raw context receipts and policy passages unchanged. Invalidate context-bearing cached checks through the existing evidence digest when the binding protocol changes; no automatic paid retry or schema change. Preserve the wrongly accepted claim in a new v4 fixture beside corrected-contributor and Hazelton controls. Existing oversized multi-citation units remain explicitly unchecked; this repair does not raise limits or certify model judgments.

Run deterministic regressions first, then freeze API/worker/runner revisions for targeted Luna verification and a fresh whole-book run on the original PDF SHA256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. Retain first attempts, explicit recovery records, exact evidence spans and original review overlays. Report processing, overview representation, Claims coverage/verdicts and source review separately.

Reuse the current CLI/config runner and fingerprint-bound semantic review packet. Add exact per-finding review items where an operation-level label would hide an attribution error: text/span, cited source passages/pages, text/evidence/receipt identity, agent proposal and separate nullable human decision. Known Financial/H05 failures are development regressions, not untouched held-out data. Source-check a separately identified sample without relabeling agent judgments as human calibration. Human metrics must refuse unlabeled or changed evidence; no uncalibrated accuracy claim.

## 4. Destination and VIP acceptance

Restart source/destination services with new logs and unchanged private credential handling; old launchers overwrite historical logs and must not be run unchanged. Reindex restored Media 2 using the owned Media embeddings API and the existing local MiniLM model. Verify terminal index status and scoped vector/hybrid retrieval, preserving the PDF, saved research, reviews and citations.

Reuse the existing seven-task participant script. Record actual task success, time, assistance, errors and confidence with NN/g heuristic/severity. The requester selected agent source review first and VIP usability afterward. Only a real participant can supply uncoached usability observations; those results and human claim calibration remain pending until received.

## ADR check and safety boundaries

ADR required: no new ADR. Proposed ADR-096 already separates cited policy evidence, attribution context, automated findings and reviewer authority. ADR-003/007/015/025/065 govern Jobs, workspace artifacts, evaluation, provider calls and provenance. These repairs reuse those owners without a new endpoint, store, credential scope or dependency. Reassess if implementation requires broader source authority.

Keep the PR draft. Preserve the requester's PR text and human-written Change summary requirement. Commit tested changes with Backlog updates, run scoped Bandit, update the existing managed PR block, and follow normal CI to terminal without bypassing gates or merging.
