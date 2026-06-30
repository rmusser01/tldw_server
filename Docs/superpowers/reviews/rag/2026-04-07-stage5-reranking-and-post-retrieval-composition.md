# Stage 5 Reranking and Post-Retrieval Composition

## Scope

Review reranking, generation, citations, guardrails, verification, response writing, and agentic or research side paths that touch the active RAG request surface.

## Code Paths Reviewed

- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
  - Rerank control points: `TwoTierReranker.rerank()` (`1476+`), sentinel injection/calibration/gating metadata (`1566+` through `1678+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/generation.py`
  - Generation entry points: `generate_response()` (`557+`), `AnswerGenerator.generate()` (`606+`), `generate_streaming_response()` (`640+`).
  - Generation ownership remains prompt-centric and document-to-string, not claim-or span-centric.
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
  - Leaf heuristics: `check_numeric_fidelity()` (`237+`), `check_numeric_precision()` (`301+`), `build_hard_citations()` (`456+`), `build_quote_citations()` (`547+`), `gate_docs_by_ocr_confidence()` (`702+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/citations.py`
  - Citation builders: `CitationGenerator.generate_citations()` (`507+`), `generate_citations_with_chains()` (`791+`), pipeline wrapper `generate_citations()` (`849+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/response_writer.py`
  - Writer formatting helpers: `format_context_xml()` (`19+`), `get_writer_depth_policy()` (`124+`), `build_writer_system_prompt()` (`169+`), `build_writer_user_prompt()` (`230+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
  - Verification seam: `PostGenerationVerifier.verify_and_maybe_fix()` (`148+`), including claim-level retrieval reuse (`187+`), adaptive second-chance retrieval (`305+`), regeneration (`371+`), and recheck (`393+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
  - Agentic post-retrieval seam: synthetic document creation (`1033+`), grounded generation (`1125+`), hard-citation/numeric/NLI postchecks (`1146+`), sentence-level chunk citations (`1319+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
  - Research side-path contract: `local_db_search` action normalizes `Document` into truncated dict results (`204+`), `research_loop()` orchestrates tool decisions and result reuse (`1004+`).
- Additional orchestration trace reviewed where these seams are actually composed:
  - `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
    - Rerank-to-gate handoff via `reranking_calibration` (`4535+`).
    - Structured writer activation (`5089+`).
    - Hard citations, quote citations, claims, numeric fidelity, and post-verification sequencing (`5360+` through `5888+`).
    - Adaptive rerun and answer adoption checks (`5990+` through `6072+`).

Post-retrieval boundary map:
- Reranking is the first real post-retrieval handoff: `result.documents` enters reranking, then `TwoTierReranker.last_metadata` is copied into `result.metadata["reranking_calibration"]` and can gate generation before any answer exists.
- Generation begins cleanly only on the happy path. After answer text exists, hard citations, quote citations, claims/NLI, numeric fidelity, and post-verification all operate as policy-bearing stages, not passive metadata decorators.
- `citations.py` and `response_writer.py` are mostly leaf utilities. The hidden orchestrators are their call sites in `unified_pipeline.py` and `agentic_chunker.py`.
- The agentic path does not preserve the Stage 4 authoritative retrieval output as the visible working set. It collapses coarse retrieval into one synthetic `Document`, stores original coarse docs in metadata, then runs postchecks against the synthetic document.
- The research path shares retrieval adapters but not the post-retrieval contract. It converts `Document` into agent-friendly dicts and returns `ResearchOutput`, not a pipeline-compatible `Document` working set.

## Tests Reviewed

- `tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py`
  - Protects sentinel exclusion plus calibrated gating metadata on `TwoTierReranker.last_metadata`.
  - Constrains the reranker itself, not downstream pipeline behavior.
  - Checks behavior and metadata.
- `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_two_tier_gate.py`
  - Protects the handoff from two-tier calibration into `generation_gate`.
  - Constrains `unified_pipeline.py` rerank-to-generation boundary.
  - Checks behavior and metadata.
- `tldw_Server_API/tests/RAG_NEW/unit/test_reranker_metrics.py`
  - Protects timeout metrics and bounded scoring under LLM reranker pressure.
  - Constrains `LLMReranker._score_batch()`.
  - Checks behavior and metrics metadata, not ranking quality.
- `tldw_Server_API/tests/RAG_NEW/unit/test_response_writer.py`
  - Protects XML escaping/reindexing and token-budget degradation policy for quality mode.
  - Constrains `response_writer.py` helpers only.
  - Checks structure plus prompt-policy metadata.
- `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_quotes_and_numeric.py`
  - Protects quote-offset verification and numeric normalization across currency/unit variants.
  - Constrains `build_quote_citations()` and `check_numeric_fidelity()`.
  - Checks behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_injection_numeric.py`
  - Protects injection downweighting, missing-number detection, heuristic hard-citation mapping, and long-answer clipping.
  - Constrains `guardrails.py` leaf heuristics.
  - Checks behavior plus citation-structure shape.
- `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_hard_citations_golden.py`
  - Protects sentence citation offsets against exact text slices.
  - Constrains `build_hard_citations()`.
  - Checks behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py`
  - Protects unsupported-claim ratio calculation and retry metric emission when verification fails.
  - Constrains `PostGenerationVerifier.verify_and_maybe_fix()`.
  - Checks behavior plus metrics, not repair quality.
- `tldw_Server_API/tests/RAG_NEW/unit/test_strict_extractive_and_citations.py`
  - Protects strict-extractive answer assembly, hard-citation gate behaviors, env-driven strict mode, and NLI low-confidence behaviors.
  - Constrains `unified_pipeline.py` post-generation control logic.
  - Checks behavior plus metadata.
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py`
  - Protects planner failure fallback, time-budget early stop, and tool-call budget exhaustion.
  - Constrains `agentic_chunker.py` assembly loop and caching path.
  - Checks behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_cache_invalidation.py`
  - Protects intra-doc vector cache invalidation and cache clearing.
  - Constrains `invalidate_intra_doc_vectors()` and `clear_agentic_caches()`.
  - Checks behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_golden_citations.py`
  - Protects golden hard-citation offsets and multi-hop assembled-chunk retention.
  - Constrains the synthetic-document contract inside `agentic_rag_pipeline()`.
  - Checks behavior.
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`
  - Protects API capability advertising, agentic streaming event order, and agentic search smoke shape.
  - Constrains endpoint wiring more than post-retrieval semantics.
  - Mostly checks metadata and response shape; only the streaming test asserts behavior.
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_strict_extractive_nli_api.py`
  - Protects that strict-extractive plus post-verification can execute through the HTTP surface.
  - Constrains endpoint-to-pipeline integration.
  - Mostly checks response shape.
- `tldw_Server_API/tests/RAG_NEW/integration/test_research_agent_loop.py`
  - Protects action normalization, preamble auto-injection, and duplicate-action reuse in the research side path.
  - Constrains `research_loop()` and result-normalization seams.
  - Checks behavior.
- `tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py`
  - Protects acceptance of generation-grounding request fields and metadata blocks when a live RAG server is available.
  - Constrains the end-to-end API surface, not individual seams.
  - Mostly checks metadata shape.
- `tldw_Server_API/tests/e2e/test_rag_post_verification_smoke.py`
  - Protects presence of `post_verification` metadata when live generation runs.
  - Constrains the end-to-end API surface.
  - Mostly checks metadata shape.

## Validation Commands

- Seam inventory:
  - `rg -n "class |def (rerank|generate|stream|verify|gate|cite|write|agentic_|research_|invalidate_|quote_|check_)" tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py tldw_Server_API/app/core/RAG/rag_service/generation.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/citations.py tldw_Server_API/app/core/RAG/rag_service/response_writer.py tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- Supporting post-retrieval control-point trace:
  - `rg -n "reranking_calibration|enable_structured_response|build_hard_citations|build_quote_citations|check_numeric_fidelity|PostGenerationVerifier|verify_and_maybe_fix|response_writer|build_writer_system_prompt|generate_response|generate_streaming_response" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/response_writer.py tldw_Server_API/app/core/RAG/rag_service/guardrails.py tldw_Server_API/app/core/RAG/rag_service/generation.py tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
  - This supporting trace is the command used to back the `unified_pipeline.py` call-site claims and the specific writer/guardrail helper ownership referenced in the findings.
- Targeted post-retrieval tests:
  - `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_two_tier_gate.py tldw_Server_API/tests/RAG_NEW/unit/test_reranker_metrics.py tldw_Server_API/tests/RAG_NEW/unit/test_response_writer.py tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_quotes_and_numeric.py tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_injection_numeric.py tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_hard_citations_golden.py tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py tldw_Server_API/tests/RAG_NEW/unit/test_strict_extractive_and_citations.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_cache_invalidation.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_golden_citations.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_strict_extractive_nli_api.py tldw_Server_API/tests/RAG_NEW/integration/test_research_agent_loop.py tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py tldw_Server_API/tests/e2e/test_rag_post_verification_smoke.py -v`
  - Result in this worktree: `38 passed, 4 skipped, 456 warnings in 170.08s (0:02:50)`.
  - Concrete skips:
    - `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py::test_rag_agentic_search_verification_flags`
      - Skip detail was confirmed with: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py -k verification_flags -rs -v`
      - `-rs` reported: `agentic verification flags skipped due to server error: 500`.
    - `tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py::test_rag_generation_grounding_smoke`
    - `tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py::test_rag_pre_retrieval_clarification_smoke`
    - `tldw_Server_API/tests/e2e/test_rag_post_verification_smoke.py::test_rag_search_with_post_verification_smoke`
      - These three skips share the same fixture-level live-server prerequisite in `tldw_Server_API/tests/e2e/fixtures.py:1012-1019`.
      - Skip reason observed in output: live API client could not connect to `localhost:8000`; pytest reported `Please ensure the server is running ... Last error: [Errno 1] Operation not permitted`.
- Docs-scope security check:
  - `source ../../.venv/bin/activate && python -m bandit -r Docs/superpowers/reviews/rag -f json -o /tmp/bandit_stage5_rag.json`
  - `jq '{errors:(.errors|length),results:(.results|length)}' /tmp/bandit_stage5_rag.json`
  - Result in this worktree: `{"errors":0,"results":0}`.

## Findings

1. High severity, high confidence: post-retrieval no longer has a single authoritative `Document` set once generation-side controls begin.
   - Stage 4 established that concrete retrievers and `MultiDatabaseRetriever` own the authoritative retrieval output. Stage 5 confirms that later stages can silently replace or synthesize that working set.
   - In the main pipeline, numeric-fidelity retry can issue targeted retrieval, merge fresh documents into `result.documents`, and regenerate the answer in place (`unified_pipeline.py:5741-5803`). Post-verification can then run adaptive repair over those mutated documents (`unified_pipeline.py:5831-5888`) and, in the adaptive rerun branch, invoke `unified_rag_pipeline()` again and compare/adopt a new answer after regression checks (`unified_pipeline.py:5990-6072`).
   - `PostGenerationVerifier.verify_and_maybe_fix()` is itself a retrieval-and-regeneration loop, not a pure evaluator. It reuses claim retrieval (`post_generation_verifier.py:187-255`), performs second-chance retrieval with expansion and HyDE (`305-366`), regenerates (`371-386`), and rechecks (`393-456`).
   - The agentic path forks even harder: it collapses coarse retrieval into one synthetic `Document` and makes that synthetic document the visible `result.documents` surface (`agentic_chunker.py:1033-1069`). That means Stage 4’s authoritative retriever output survives only as `metadata["coarse_docs"]`, while postchecks act on the synthetic surrogate (`1125-1340`).

2. High severity, high confidence: guardrails are leaf heuristics in-module but hidden orchestrators in-system.
   - `guardrails.py` itself is narrow and utility-like: numeric checks, hard-citation mapping, quote-citation mapping, injection scoring, and OCR gating do not own retrieval or generation (`guardrails.py:237-530`, `702-716`).
   - The orchestration leak happens at call sites. In the main pipeline, hard citations can gate and overwrite the answer (`unified_pipeline.py:5360-5394`), quote citations attach parallel evidence metadata (`5396-5403`), numeric fidelity can append notes, decline, or trigger retrieval/regeneration (`5741-5828`), and post-verification can trigger another recovery loop (`5831-5888`).
   - The agentic pipeline reuses the same heuristics as control flow, not decoration: hard citations gate the answer, numeric fidelity can retry local retrieval, and NLI low-confidence can append notes or decline (`agentic_chunker.py:1181-1303`).
   - So the modules are leaves, but the effective post-retrieval architecture is not. The real policy owners are the orchestration blocks that interpret heuristic output as abstention and retry decisions.

3. Medium severity, high confidence: citation responsibilities are split across three parallel, loosely coordinated systems with no single authoritative contract.
   - `CitationGenerator.generate_citations()` derives academic citations, chunk citations, and inline marker maps from documents alone (`citations.py:507-566`). `format_inline_citations()` still just appends markers to the end of text (`673-696`), so it does not own sentence-to-evidence grounding.
   - `response_writer.py` separately instructs the model to emit `[number]` citations and formats XML-tagged context (`response_writer.py:19-248`), but it has no runtime tie to `CitationGenerator` or `guardrails.py`.
   - Grounded enforcement lives in a third path: `build_hard_citations()` and `build_quote_citations()` in `guardrails.py`, which the main pipeline uses after generation (`unified_pipeline.py:5360-5403`) and the agentic path reuses against the synthetic chunk (`agentic_chunker.py:1181-1197`, `1319-1339`).
   - Result: prompt-time inline citations, post-hoc dual citations, and hard span citations are parallel systems with no single authoritative claim/span contract. Tests protect pieces of each system, but not their consistency with each other.

4. Medium severity, high confidence: response writing is only partially centralized and duplicates generation policy rather than owning it.
   - `response_writer.py` is cohesive for one sub-mode: XML context, writer prompt, and token-budget depth policy (`response_writer.py:19-248`). The main pipeline only uses it when `enable_structured_response` is on (`unified_pipeline.py:5089-5128`).
   - Outside that mode, generation policy lives elsewhere: `generation.py` still owns generic prompt templates and document-to-string formatting (`generation.py:88-160`, `264-345`, `587-640`), strict-extractive mode bypasses normal generation entirely inside `unified_pipeline.py`, and the agentic path calls `AnswerGenerator` directly with raw `chunk_text` rather than using `response_writer.py` (`agentic_chunker.py:1125-1141`).
   - The tests match that fragmentation. `test_response_writer.py` proves the helper module works, but the real pipeline behavior is mostly asserted in `test_strict_extractive_and_citations.py`, `test_pipeline_two_tier_gate.py`, and agentic tests, not in one central response-writing contract.

5. Medium severity, high confidence: the agentic and research side paths share some internals but do not share a clean post-retrieval contract with the main pipeline.
   - `agentic_rag_pipeline()` reuses `AnswerGenerator`, `ClaimsEngine`, `build_hard_citations()`, `check_numeric_fidelity()`, and `PostGenerationVerifier`, but it does so over a synthetic document produced from assembled snippets (`agentic_chunker.py:1033-1069`, `1125-1340`). That is a compatible API surface, but not the same evidence model as the standard pipeline.
   - `research_agent.py` diverges further. Its `local_db_search` action converts retrieved `Document` objects into truncated dict payloads for agent planning (`research_agent.py:204-257`), and `research_loop()` accumulates `ResearchOutput.all_results`, action dedup state, and preamble metadata rather than returning pipeline-ready documents (`1004-1325`).
   - The research-agent tests protect normalization, dedup, and preamble behavior, not any shared grounding contract. This validates the Stage 4 routing note: media-side retrieval here belongs to a research/agentic fork, not the authoritative retrieval seam.

6. Medium severity, medium confidence: the current tests strongly protect gates and offset behavior, but they do not pin down document-set ownership after retrieval.
   - The strongest behavioral tests here are reranker gating, strict-extractive hard-citation/NLI behaviors, post-verifier metrics, agentic fallback behavior, and golden citation offsets.
   - The weakest area is exactly the Stage 4 carry-forward concern: no targeted test asserts that post-retrieval stages must preserve one authoritative document set, or that any later synthetic/retried set must be surfaced separately from the base retrieval output.
   - In practice, the suite protects metadata correctness and some span behavior better than it protects evidence-set ownership.

## Suggested Refactor/Actions

- Introduce a small post-retrieval state contract that distinguishes `retrieved_documents` from any `derived_documents` used for repair, numeric retry, or agentic synthesis. Later stages can still experiment, but they should stop mutating the Stage 4 evidence set in place without surfacing that ownership change.
- Keep `guardrails.py` and `citations.py` utility-only, and move all abstention/retry decisions into one explicit post-retrieval coordinator. Right now the policy is distributed across `unified_pipeline.py`, `agentic_chunker.py`, and `PostGenerationVerifier`.
- Collapse the citation stack onto one claim/span evidence model. `response_writer` prompt hints, `CitationGenerator` document citations, and `build_hard_citations()` span mappings should either share one contract or have a documented precedence order.
- Either make `response_writer.py` the single prompt-composition surface for grounded generation or demote it to an optional helper in docs. The current split across `generation.py`, `response_writer.py`, strict-extractive assembly, and the agentic path creates overlapping prompt policy with no single owner.
- Decide whether the agentic path is a compatible pipeline mode or a separate product surface. If it is compatible, its synthetic document should be marked as derived evidence while the coarse retrieved documents remain first-class. If it is separate, that fork should be explicit in docs and tests rather than implied by shared helper reuse.
- Keep `research_agent.py` routed as a side path in the final synthesis. Its action/dedup loop is valid, but it does not produce the same post-retrieval object model as the main pipeline.

## Coverage Gaps

- No reviewed test asserts a one-way ownership boundary from retrieval -> reranking -> generation -> verification. The suite allows later stages to replace `result.documents` as long as metadata and answer behaviors still look correct.
- No test checks consistency between the three citation systems: prompt-level `[number]` instructions from `response_writer.py`, `CitationGenerator` dual citations, and guardrail hard/quote citations.
- `test_response_writer.py` is helper-focused. It does not prove the main pipeline or the agentic path actually use the writer contract consistently.
- `test_post_verifier.py` protects ratios and metrics, but not the quality or provenance of the repaired document set returned by adaptive retrieval.
- `test_agentic_golden_citations.py` is strong on offsets, but it still validates citations against the synthetic chunk, not against the original coarse retrieval list that Stage 4 identified as authoritative.
- `test_research_agent_loop.py` protects research iteration semantics, not handoff into grounded generation or citation enforcement.
- Validation skips matter for synthesis:
  - The live-server e2e coverage was unavailable in this environment because connections to `localhost:8000` were blocked with `Operation not permitted`.
  - One agentic verification integration test skipped because the endpoint returned HTTP `500` under the fixture setup, so that specific post-retrieval API seam remains unverified here.

## Exit Note

Final synthesis should emphasize three points.

- Stage 4’s upstream boundary still matters: concrete retrievers own the first authoritative `Document` set.
- Stage 5 shows that this ownership does not stay intact. Numeric retry, adaptive verification, and the agentic synthetic-document path all create second evidence sets after retrieval, sometimes by mutating `result.documents` directly.
- The cleanest architectural split for the final report is therefore not just retrieval vs generation. It is authoritative retrieval output vs derived post-retrieval evidence, with guardrails/citations/verification currently tangled across that line.
