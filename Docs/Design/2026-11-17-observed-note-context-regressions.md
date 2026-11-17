# Observed note-context regression controls

Tracked by TASK-15505.25.

Job 60's completed Luna overview says an OTS-transfer endnote has unclear
attribution and context. Its source-backed note 53 relationship identifies Robert
Bowes's CFPB contribution, while not establishing who personally wrote or endorsed
the note. The verifier accepted the broader uncertainty statement. Agent source
review identifies this as a false-acceptance candidate; it is not a human-calibrated
accuracy measurement.

## Bounded design

Add `political_pdf_verification_regressions.v5.json` using the existing fixture
schema. Copy all 15 v4 cases unchanged and append two cases with identical source
passages: the observed negative prose and a positive that explicitly preserves
contribution context without claiming personal note authorship or endorsement.
Keep the first sentence and policy clauses the same. V4 and its saved results
remain immutable. No loader, judge, generation prompt, live configuration or
runtime changes are needed.

The loader requires P-number passage IDs. Only citation tokens are remapped:
`E3.2.10` → `P1`, `E3.3.2` → `P2`, `E3.3.6` → `P3`. Reversing this mapping must
recover the original text hash. The source excerpts remain exact. Their context
comes from the owned canonical source resolver; fixtures must not inject it.

The new expectations use `agent_source_audit` provenance and pending/null human
review. They are development regressions on a known failure, not held-out data.
Adding them does not claim that the model error is fixed or trigger a paid run.
Replay later using a new config/hash/run while retaining the first outcome.

## Source and observation identity

- PDF SHA256: `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`.
- Canonical content SHA256: `f1beb788a6cb84add31d14ec24b89742f9723bae499adc38a1e8092064b36a1a`.
- Job 60, artifact `summary-509b9b9d081641bd93f5101d940402a1`, version 33,
  overview `claim-18`; source navigation `media_1:v4:ec5d2af126`.
- Original review-row SHA256: `658b69fb27ff2c02984f6bd45b38f3975d53d05faf99f91c56824c1818fc49bb`.
- Original claim-text SHA256: `333f845776003be5f0cc10a2ab2c7479a060f2623fe4f6760d5ae48dae61b0dc`.
- Note 53: `[2499293, 2499600)`; supplied passage starts after its label at
  `2499297` and ends at `2500202`, overlapping later bibliography entries.
- Its unique body marker `[53]` is `[2470570, 2470574)` in the CFPB contribution
  beginning at `2464593`. A Burton bibliography name in note 54 does not reassign
  this relationship.

## Verification

Tests exercise the existing hash-bound loader, preserve all historical cases,
check reversible observed-text identity and exact quote hashes, and parse both
cases as complete cited units. A compact source snapshot using audited excerpts
passes through the production workspace citation resolver: note 53 must retain
its Bowes contribution link despite the adjacent Burton bibliography name.
Synthetic test coordinates are explicitly distinct from the original coordinates
above. These offline checks make no model calls and cannot certify semantic model
performance.

## ADR check

ADR required: no new ADR. Proposed [ADR-096](../ADR/096-research-citation-bound-claims-verification.md)
separates attribution context, cited evidence, automated verdicts and reviewer
decisions. [ADR-015](../ADR/015-evaluations-existing-evaluator-integration.md)
retains the existing evaluation integration. This additive fixture introduces no
schema, persistence, ownership, provider or runtime boundary.
