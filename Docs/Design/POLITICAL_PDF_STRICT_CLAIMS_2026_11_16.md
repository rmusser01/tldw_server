# Strict citation support in political PDF research

TASK-15505.16, TASK-15505.17 and TASK-15505.18; PR #3480. The user authorized implementation, Luna verification on the same PDF, review/export integration and CI repairs on 2026-11-16.

## Problem and scope

Chapter extraction preserves exact source passages and synthesis rejects unknown citation IDs, but neither establishes that those passages support the generated assertions. Workspace QA retrieves first and generates through a separate completion, bypassing the optional RAG Claims stage. The observed Financial endnote attribution and interceptor qualifier/action changes are regression cases. They are not held-out measurements or human labels.

Reuse Claims_Extraction for strict source support. Cover extracted evidence claims, final chapter text, final overview text and final workspace QA answers. Carry findings through existing artifact/message metadata, review, Markdown/offline export and complete-pack restore. Keep original generated text and human correction overlays separate. Do not introduce a second Claims store, silently rewrite answers, enable external web fact-checking, or treat document fidelity as real-world truth.

## Decision and alternatives

The shared verifier binds each generated text unit to its literal citation IDs and only the source passages authorized by those IDs. It covers the submitted text explicitly: missing citations, unknown IDs, source changes, unverified units, limits and provider failures cannot be counted as supported. Semantic verification must be required; quote presence, numerical overlap and retrieval scores alone cannot authorize a supported verdict. Preserve attribution and relevant source context without allowing unrelated context to repair an incorrect citation.

Reusing the existing optional RAG flag alone was rejected because the frontend generates its answer separately and standard verification can search different evidence. Building a parallel judge/store was rejected because Claims already owns verification. Silently blocking or deleting all outputs on uncertain support was rejected because users need to inspect and correct useful drafts.

## Data and execution

The existing section-output Job owns generation and verification checkpoints. Reports attach to each batch and section as `claim_verification`, and to the overview as `synthesis_verification`. Reports include their version, checked-text identity, verifier identity, per-unit status/reason/citation IDs, and coverage. Generation status remains separate from verification status. Old artifacts without reports are unchecked; old completion cannot masquerade as a new strict check.

QA uses the shared backend verifier through an authenticated workspace operation. The server resolves exact cited source ranges from owned workspace sources; client-submitted quotes are never trusted as proof. The frontend checks the finalized answer, persists findings on that message and respects account/workspace/message changes while work is pending. Existing provider credential resolution and gateway accounting remain authoritative. Credentials never enter reports or logs.

Calls are bounded and cancellable. Verification error, invalid judge output, truncation or a bound exceeded produces a visible incomplete/unverified finding, preserving generated material. Retry reuses unchanged generation and must not reuse a verification receipt after its text/source identity changes. Source ID remapping on portable restore retains findings as historical checks, without claiming a fresh check against a changed source.

Live follow-through: references/publication-only portions still require a descriptive evidence claim bound to a current passage ID; their metadata must not become invented policy positions or conclusions from the cited works. Keep the nonempty extraction contract and exact server-materialized quotes. A controlled retry preserves the original failed response. Reuse validated terminal support, negative, uncited and known hard-limit findings only when verifier identity, exact draft/evidence hashes and full substantive text coverage remain current. Such reuse preserves the original verdict and uncovered counts; it does not turn an unchecked unit into support. Missing, malformed, stale, temporary-error or unknown checks still retry. These are repairs within ADR-096, not new evidence authority or an automatic generation-retry loop.

## UX and evaluation

Show generation and checking progress, clear supported/needs-review/unchecked labels, reasons and evidence links. Preserve findings in readable exports and structured packs. This follows NN/g visibility of system status, error prevention, recognition rather than recall, and help recognizing/recovering from errors. A reviewer decision is separate from an automated support assessment.

Regressions cover cited-but-unrelated evidence, missing citations, wrong attribution, changed policy verbs/qualifiers, compound claims, malformed judgments, provider failure, text/source changes, cancellation and export/restore. Run deterministic tests with mocked providers, then freeze the runtime and rerun Luna on the original PDF SHA256 `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`. Record structural coverage, verifier coverage/verdicts and source-audited findings separately. No uncalibrated accuracy score or automatic human acceptance.

## ADR check

ADR required: yes. [ADR-096](../ADR/096-research-citation-bound-claims-verification.md) records the durable support boundary and portable findings. Existing ADR-003 governs Jobs ownership, ADR-018 endpoint governance, ADR-025 provider gateway ownership, and ADR-015 reuse of evaluation modules.
