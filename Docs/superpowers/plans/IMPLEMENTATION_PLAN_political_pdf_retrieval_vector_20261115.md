# Political PDF remaining retrieval and vector acceptance

Task: TASK-15505.7. Approved design: [retrieval/vector follow-up](../../Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md#remaining-multifacet-retrieval-and-local-vector-acceptance-task-155057).

ADR assessment: reuse ADR-022/060 for Embeddings and Jobs, ADR-053 for fusion, and ADR-017/052 for existing authority. No new ADR or storage owner is required.

### Live vector adapter correction

The first real MiniLM index completed with 431 vectors. Public vector search exposed three translation errors: chunk IDs became the parent Media ID, stored `chunk_start`/`chunk_end` were not normalized into common offsets, and L2 distances were treated as cosine distances. Fix these boundaries before accepting vector/hybrid results. Keep raw distances, honor the collection's actual distance metric, and return bounded similarity scores. Preserve vector chunk identity separately from `media_id`. Normalize offset aliases and use the existing scoped canonical context verifier to derive page spans; when expansion is disabled, verify with a zero-character budget. Never trust stale vector page metadata or change access rules/context caps. Test a real local Chroma collection, exact canonical Media pages, disabled/enabled expansion, distinct chunks from one source, stale/hidden controls, and the full public endpoint. Parent owns runtime restarts and model-backed acceptance.

## Stage 1: Preserve separate question facets
**Goal**: Recover short-acronym evidence without increasing context or search limits.
**Success Criteria**: The original FBI question retains both review and receiving-office passages through the public pipeline; unrelated source rows and explicit search syntax remain protected.
**Tests**: RED then GREEN for normalized citation directives, uppercase acronym controls, bounded multifacet persisted retrieval, full endpoint evidence/page/security output and cache-policy rollover.
**Status**: Complete — RED 7; combined 165 GREEN, then 9 final parser/property/public-pipeline checks; independent review and live post-restart replay coordinated by parent.

## Stage 2: Verify existing local embeddings
**Goal**: Use the cached MiniLM model through existing local embedding and Jobs paths.
**Success Criteria**: Offline loader/inference succeeds; coordinated isolated configuration indexes the same PDF and reports vector readiness with actual vector evidence.
**Tests**: Local model smoke without downloads, durable job completion/count/status checks, source-scoped vector and hybrid public queries.
**Status**: Complete — real Jobs reindex completed 2,261 × 384 vectors in 72 seconds. Every refined canonical span is exact and at most 512 tokens; no original chunk text was dropped. See the embedding input-coverage plan and follow-through report.

## Stage 3: Evaluate coverage and record limits
**Goal**: Distinguish fixed regressions, held-out retrieval coverage and generated claim faithfulness.
**Success Criteria**: Frozen source-audited held-out cases report per-facet/all-facet coverage and latency by text/vector/hybrid mode; actual answer checks remain separate.
**Tests**: Public pipeline replay of original failures and held-out questions; parent-coordinated real generation only after evidence acceptance. Focused lint/Bandit and independent review precede handoff.
**Status**: Complete — fixed 39-query public replay: expected-page coverage 12/13 text, 10/13 vector, 13/13 hybrid; zero errors. Fresh DeepSeek hybrid FBI answer passed agent source review. Results and serial latency limits are recorded in the follow-through report; human labels remain TASK-15505.6.

### Reopened acceptance: precise page attribution

The final migrated CFPB answer correctly stated the proposal and court condition but cited the preceding page. Retrieved canonical spans retained both physical and printed pages; both prompt formatters reduced those spans to a page-list header above concatenated text containing the previous page's footer. Add generic inline page markers in the existing frontend and backend formatters, keeping source indices and every content character unchanged. Use them only when bounded, ordered offsets account for the exact passage: first/last boundaries align, spans do not overlap, and any gaps contain only preserved separator whitespace. JavaScript must interpret the backend's character offsets as Unicode code points, including astral characters. Invalid or incomplete ranges retain the existing header and whole escaped passage. No new retrieval, context budget, authority or storage rule; existing page provenance and ADR-053 remain governing.

**Status**: Complete. Tests cover two-page footer ambiguity, exact text preservation, whitespace gaps, malformed/mismatched offsets and astral Unicode. Both formatter suites, relevant generation/prompt tests, scoped lint/Bandit, independent review and one unchanged-question live retest are complete.

Implementation and independent review are complete: frontend 27 tests and backend 22 tests passed, including 30 Unicode property examples; independent reviewer ran backend 16 and frontend 11 successfully. Both real saved-passage offline checks place the proposal after the PDF 872 / printed 839 boundary while preserving all 1,664 source characters. Final TypeScript passes; comprehensive Bandit across 22 changed production files has no findings/errors, frontend lint is clean, and backend lint has no new findings.

One real DeepSeek retest after reloading both APIs correctly states the proposal and court condition, citing printed 838–839 / PDF 871–872. That range contains the correct claim page but is broader than a pinpoint citation; residual precision belongs to TASK-15505.6. The previous wrong-page answer is preserved separately. H05 is now regression evidence because it informed this repair; it is no longer untouched held-out acceptance. Canonical reload retains the same ten messages and 27 citations with every server acknowledgement.
