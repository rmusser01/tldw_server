# RAG Pipeline Evaluation - Design

This document specifies the design to evaluate end-to-end RAG configurations using the existing unified RAG pipeline. The goal is to let users compare ingestion/chunking/retrieval/reranking/generation options and obtain step-level metrics and a leaderboard to pick the best configuration for their media and tasks.

## Objectives

- Use the existing unified RAG pipeline (`app/core/RAG/rag_service/unified_pipeline.py`) as the execution engine; do not introduce a separate pipeline.
- Add an evaluation harness that sweeps configurations for:
  - Chunking (retrieval-time enhanced chunking knobs; optional indexing-time chunking in a later phase)
  - Search/retrieval (fts/vector/hybrid + params)
  - Reranking (flashrank/cross-encoder/hybrid)
  - Generation (LLM model/prompt/temperature)
- Compute step-level metrics (chunking, retrieval, generation), aggregate to an overall score, and present a leaderboard.
- Provide caching/fingerprinting to reuse chunksets, embeddings, and retrieval outcomes where safe.
- Reuse unified evaluations API for creation/runs/history; avoid new top-level services.

## Scope (v1)

- Retrieval-time chunking via `enhanced_chunking_integration` toggles in unified pipeline.
- Config sweep across chunking/retrieval/rerank/generation.
- Step-level metrics: chunk stats, retrieval precision/recall/diversity, generation faithfulness/relevance/similarity, latency/cost.
- Persist results in `evaluation_runs.results` and `evaluations_unified` for history.

Out-of-scope (v1):
- Full ingestion orchestration and media capture; assume text is already in DB.
- Ephemeral re-indexing with alternate chunksets (planned v1.1, see Future Work).
- Advanced search (Bayesian optimization). Use grid/random sampling.

## Architecture

1. API/Schema
   - Extend `evaluation_schemas_unified.EvaluationSpec` with `sub_type` to support `rag_pipeline`.
   - Add `RAGPipelineEvalSpec` nested under `eval_spec` to carry sweep configs.

2. Runner
   - Extend `EvaluationRunner._get_evaluation_function` to handle `model_graded + sub_type=rag_pipeline`.
   - Expand grid/random search over sweepable parameters; evaluate each config on provided dataset/queries.
   - Call `unified_rag_pipeline` with matching flags for each config.

3. Metrics
   - Chunking: token length stats; cohesion vs separation (cosine similarities); coverage proxy (proportion of answer-supporting spans present in retrieved chunks downstream).
   - Retrieval: precision/recall (if ground truth spans or gold answers exist), diversity (inverse pairwise similarity), optional MRR/nDCG.
   - Generation: reuse `RAGEvaluator` (faithfulness, claim_faithfulness, relevance, answer_similarity) and `rag_custom_metrics` (attribution, coherence, response time, costs; its completeness metric was deleted in TASK-13635, unused and unable to reach a provider).
   - Aggregation: confidence-weighted overall score; secondary sort by cost and latency.

4. Caching
   - Fingerprint inputs (query, source selection, chunking flags) to cache:
     - Retrieval-time chunksets (if materialized in memory for a run) - optional; primarily we rely on pipeline’s on-the-fly chunking.
     - Embeddings and retrieval outcomes via semantic cache where possible.
   - Config toggles in spec: `cache_chunksets`, `cache_embeddings`, `cache_retrievals`.

## Schemas (API)

Add to `evaluation_schemas_unified.py`:

- `EvaluationSpec.sub_type`: Literal['summarization','rag','response_quality','rag_pipeline'].
- `RAGPipelineEvalSpec` with the following groups:
  - Dataset: inline samples or dataset_id.
  - Chunking sweep (retrieval-time): `method`, `chunk_size`, `overlap`, `structure_aware`, `parent_expansion`, `include_siblings`, `chunk_type_filters`.
  - Retrieval sweep: `search_mode`, `hybrid_alpha`, `top_k`, `min_score`, `keyword_filter`.
  - Reranker sweep: `strategy`, `top_k`, `model` (optional).
  - Generation sweep: `model`, `prompt_template`, `temperature`, `max_tokens`.
  - Search strategy: `grid | random`, `max_trials`.
  - Metrics selection: lists per step (`chunking_metrics`, `retrieval_metrics`, `generation_metrics`).
  - Caching: `cache_chunksets`, `cache_embeddings`, `cache_retrievals`.
  - Execution: `concurrency`, `timeout_seconds`.

Fields that can be swept accept a single value or list; validators normalize to lists for internal use.

### Example (condensed)

```
{
  "name": "my_rag_pipeline_eval",
  "eval_type": "model_graded",
  "eval_spec": {
    "sub_type": "rag_pipeline",
    "rag_pipeline": {
      "dataset": [{
        "input": {"question": "What is X?"},
        "expected": "X is ..."
      }],
      "chunking": {
        "method": ["structure_aware", "sentences"],
        "chunk_size": [512, 768],
        "overlap": [64]
      },
      "retrievers": [{
        "search_mode": ["hybrid"],
        "hybrid_alpha": [0.3, 0.5, 0.7],
        "top_k": [8, 12]
      }],
      "rerankers": [{"strategy": ["flashrank", "cross_encoder"], "top_k": 5}],
      "rag": {"model": ["gpt-4o-mini"], "temperature": [0.1], "max_tokens": [512]},
      "metrics": {
        "retrieval_metrics": ["context_precision", "context_recall", "diversity"],
        "generation_metrics": ["faithfulness", "relevance", "answer_similarity"]
      },
      "search_strategy": "grid",
      "max_trials": 24,
      "caching": {"cache_embeddings": true, "cache_retrievals": true}
    }
  }
}
```

### Corpus Expectations for Ephemeral Indexing

- To compare indexing-time chunking strategies, provide a corpus per sample:
  - Inline under each sample: `input.corpus` or `input.documents`
  - Acceptable formats:
    - Array of strings: `["doc text 1", "doc text 2", ...]` (auto IDs assigned)
    - Array of objects: `[{"id": "doc_1", "text": "..."}, ...]`
- When `index_namespace` is set, the runner creates ephemeral collections per configuration: `{index_namespace}_{cfg_id}`.
- Set `cleanup_collections: true` to delete these collections after the run.

### Sample Dataset with Corpus

```
{
  "name": "rag_pipeline_demo",
  "eval_type": "model_graded",
  "eval_spec": {
    "sub_type": "rag_pipeline",
    "rag_pipeline": {
      "dataset": [
        {
          "input": {
            "question": "What is X?",
            "corpus": [
              "X is a mechanism used to ...",
              {"id": "doc_2", "text": "Details about X under various conditions..."}
            ]
          },
          "expected": {"answer": "X is ...", "relevant_ids": ["doc_2"]}
        }
      ],
      "index_namespace": "rag_eval_ns",
      "cleanup_collections": true,
      "chunking": {"method": ["sentences"], "chunk_size": [512], "overlap": [64]},
      "retrievers": [{"search_mode": ["hybrid"], "hybrid_alpha": [0.3, 0.7], "top_k": [8]}],
      "rerankers": [{"strategy": ["flashrank", "cross_encoder"], "top_k": [5]}],
      "rag": {"model": ["gpt-4o-mini"], "temperature": [0.1]},
      "aggregation_weights": {"rag_overall": 1.0, "retrieval_diversity": 0.1, "chunk_cohesion": 0.1}
    }
  }
}
```

### Server Presets and Cleanup Endpoints

- Save or update a pipeline preset:
  - POST `/api/v1/evaluations/rag/pipeline/presets`
  - Body: `{ "name": "my-preset", "config": { "chunking": {...}, "retriever": {...}, "reranker": {...}, "rag": {...} } }`
  - Response: `{ name, config, created_at?, updated_at? }`

- Get preset by name:
  - GET `/api/v1/evaluations/rag/pipeline/presets/{name}`

- List presets:
  - GET `/api/v1/evaluations/rag/pipeline/presets?limit=50&offset=0`

- Delete preset by name:
  - DELETE `/api/v1/evaluations/rag/pipeline/presets/{name}`

- Cleanup expired ephemeral collections (TTL-based):
  - POST `/api/v1/evaluations/rag/pipeline/cleanup`
  - Response: `{ expired_count, deleted_count, errors? }`

Notes:
- Presets are stored in `pipeline_presets` table and keyed by `name`.
- Ephemeral collections are tracked in `ephemeral_collections` with `ttl_seconds`.
- Set `rag_pipeline.ephemeral_ttl_seconds` and `rag_pipeline.cleanup_collections` in the run spec to control immediate or deferred cleanup.

## Runner Flow

1. Expand grid/random search to a set of configurations.
2. For each sample in dataset and each config:
   - Call `unified_rag_pipeline` with chunking/retrieval/rerank/generation params.
   - Record timings, usage, and results.
   - Compute metrics via `RAGEvaluator` and `rag_custom_metrics`.
3. Aggregate per-config metrics; produce leaderboard + best-config.
4. Persist results to `evaluation_runs.results` (full), and `evaluations_unified` (summary).

## Metrics Details

- Chunking: mean/std token length; cohesion/separation (cosine); coverage proxy (downstream retrieval support rate).
- Retrieval: precision/recall (if gold spans/answers), diversity (inverse similarity), optional MRR/nDCG.
- Generation: faithfulness, relevance, answer_similarity; attribution markers; latency and token costs.
- Aggregation: confidence-weighted overall score; tie-breakers: cost then latency.

## Testing Strategy

- Unit: schema normalization/validators; grid expansion; metric calculators; caching keys/fingerprints.
- Integration: synthetic dataset (2-3 QAs); 2-3 configs per dimension; mock LLM/embeddings for determinism; assert leaderboard ordering and best-config selection.

## Future Work (v1.1)

- Ephemeral re-indexing: build alternate chunksets with `core/Chunking`, embed and store into a namespaced collection; pass `index_namespace` into retrieval to compare indexes.
- Bayesian/BO search strategies for faster convergence.
- Deeper per-source analytics and ablations.
