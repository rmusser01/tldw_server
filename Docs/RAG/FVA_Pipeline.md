# FVA Pipeline - Falsification-Verification Alignment

## Overview

The FVA (Falsification-Verification Alignment) pipeline extends standard claim verification with **active counter-evidence retrieval**. Rather than only looking for supporting evidence, FVA actively searches for contradicting evidence to provide more robust verification results.

This implementation is inspired by the FVA-RAG paper ([arXiv:2512.07015](https://arxiv.org/abs/2512.07015)).

## Key Concepts

### Verification Statuses

The FVA pipeline introduces a new verification status:

| Status | Description |
|--------|-------------|
| `VERIFIED` | Claim is supported by evidence with high confidence |
| `REFUTED` | Claim is contradicted by strong evidence |
| `CONTESTED` | **New**: Claim has significant evidence both supporting AND contradicting it |
| `UNVERIFIED` | Insufficient evidence to determine claim validity |

### Falsification Trigger

Not every claim needs falsification checking. The pipeline uses intelligent triggering based on:

1. **Confidence Threshold**: Claims with verification confidence below the threshold trigger falsification
2. **Sparse Evidence**: Claims with few supporting documents may benefit from counter-evidence search
3. **Forced Types**: Certain claim types (e.g., statistics, causal claims) can be configured to always trigger falsification
4. **Uncertainty**: Claims with neutral/mixed initial evidence

### Anti-Context Retrieval

When falsification is triggered, the pipeline generates counter-queries to find contradicting evidence:

- **Negation queries**: "NOT [claim]", "[claim] is false"
- **Contrary queries**: "[opposite of claim]"
- **Alternative queries**: "[different perspective on claim]"

Each query runs the hybrid (full-text and vector) search of the user's media DB, at most 5 documents per query. `min_relevance_score` (0.3) applies to the full-text scores, min-max normalised within each query's results, before they are fused with the vector results: it drops the weakest full-text matches, and the best one always passes. Fused scores are not filtered again, and vector results have no floor. So no absolute relevance floor applies: the adjudicator's stance check decides which documents count, and a neutral one does not.

Adjudication assesses every counter-evidence document and, on the supporting side, the first 5 retrieved documents, the window the verifier draws its evidence from. Each assessment is one LLM call.

### Adjudication

When both supporting and contradicting evidence exists, the adjudicator weighs the evidence:

- Uses NLI (Natural Language Inference) or LLM-based stance assessment
- Calculates support and contradiction scores
- Determines if evidence is balanced (CONTESTED) or one-sided

## API Usage

### Endpoint

```
POST /api/v1/claims/verify/fva
```

### Query Parameter

- `user_id` (optional): the user to verify for. It counts only for a platform admin, as for the claims rebuild endpoints. The request then reads that user's media DB and vector store, and its LLM calls are quota-checked and recorded for that user. Anyone else's `user_id` is ignored, and the request runs for the caller.

An LLM call refused by the user's quota answers HTTP 402 with the `limit_exceeded` body (`Docs/Operations/Usage_Quotas.md`).

### Request Body

```json
{
  "claims": [
    {
      "text": "Paris is the capital of France.",
      "claim_type": "existence",
      "span_start": 0,
      "span_end": 32
    },
    {
      "text": "The Earth is flat.",
      "claim_type": "existence"
    }
  ],
  "query": "What are facts about geography?",
  "sources": ["media_db"],
  "top_k": 10,
  "fva_config": {
    "enabled": true,
    "confidence_threshold": 0.7,
    "contested_threshold": 0.4,
    "max_concurrent_falsifications": 5,
    "timeout_seconds": 30.0,
    "force_claim_types": ["statistic", "causal"],
    "max_budget_usd": 0.50
  }
}
```

### Response

```json
{
  "results": [
    {
      "claim_text": "Paris is the capital of France.",
      "claim_type": "existence",
      "original_status": "verified",
      "final_status": "verified",
      "confidence": 0.95,
      "falsification_triggered": false,
      "anti_context_found": 0,
      "supporting_evidence": [],
      "contradicting_evidence": [],
      "adjudication": null,
      "rationale": "High confidence verification, falsification skipped.",
      "processing_time_ms": 150.5
    },
    {
      "claim_text": "The Earth is flat.",
      "claim_type": "existence",
      "original_status": "unverified",
      "final_status": "refuted",
      "confidence": 0.92,
      "falsification_triggered": true,
      "anti_context_found": 5,
      "supporting_evidence": [],
      "contradicting_evidence": [
        {
          "doc_id": "doc_123",
          "snippet": "Scientific evidence confirms Earth is an oblate spheroid...",
          "score": 0.95,
          "stance": "contradicts",
          "confidence": 0.92
        }
      ],
      "adjudication": {
        "support_score": 0.1,
        "contradict_score": 0.92,
        "contestation_score": 0.11,
        "rationale": "Strong contradicting evidence (score=0.92) outweighs support (score=0.10)."
      },
      "rationale": "Strong contradicting evidence found.",
      "processing_time_ms": 2500.3
    }
  ],
  "total_claims": 2,
  "falsification_triggered_count": 1,
  "status_changes": {
    "unverified->refuted": 1
  },
  "total_time_ms": 2650.8,
  "budget_exhausted": false
}
```

### Get FVA Settings

```
GET /api/v1/claims/verify/fva/settings
```

Returns current FVA configuration from settings.

## Configuration

Add these settings to `config.txt` under the `[Claims]` section:

```ini
[Claims]
# ... existing claims settings ...

# FVA (Falsification-Verification Alignment) Settings
FVA_ENABLED = true
FVA_CONFIDENCE_THRESHOLD = 0.7
FVA_CONTESTED_THRESHOLD = 0.4
FVA_MAX_CONCURRENT = 5
FVA_TIMEOUT_SECONDS = 30.0
FVA_MAX_BUDGET_RATIO = 0.3
FVA_FORCE_CLAIM_TYPES =
FVA_MIN_CONFIDENCE_FOR_SKIP = 0.9
```

### Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `FVA_ENABLED` | bool | `true` | Enable/disable FVA pipeline globally |
| `FVA_CONFIDENCE_THRESHOLD` | float | `0.7` | Confidence below this triggers falsification |
| `FVA_CONTESTED_THRESHOLD` | float | `0.4` | Ratio threshold for CONTESTED status (0.4 means 40-60% split) |
| `FVA_MAX_CONCURRENT` | int | `5` | Maximum concurrent falsification operations |
| `FVA_TIMEOUT_SECONDS` | float | `30.0` | Timeout for each falsification operation |
| `FVA_MAX_BUDGET_RATIO` | float | `0.3` | Maximum portion of budget for FVA (30%) |
| `FVA_FORCE_CLAIM_TYPES` | string | `` | Comma-separated claim types that always trigger falsification |
| `FVA_MIN_CONFIDENCE_FOR_SKIP` | float | `0.9` | Skip falsification if confidence exceeds this |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FVA Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │   Claims    │───▶│    Standard      │───▶│ Decision  │  │
│  │   Input     │    │   Verification   │    │  Point    │  │
│  └─────────────┘    └──────────────────┘    └─────┬─────┘  │
│                                                   │        │
│                     ┌─────────────────────────────┼────┐   │
│                     │ Falsification Triggered?    │    │   │
│                     │ - Low confidence            │    │   │
│                     │ - Sparse evidence           │    │   │
│                     │ - Forced claim type         │    │   │
│                     └─────────────────────────────┼────┘   │
│                                                   │        │
│                              ┌────────────────────┴───┐    │
│                              ▼                        ▼    │
│                     ┌────────────────┐      ┌───────────┐  │
│                     │ Anti-Context   │      │   Keep    │  │
│                     │   Retrieval    │      │ Original  │  │
│                     └───────┬────────┘      └───────────┘  │
│                             │                              │
│                             ▼                              │
│                     ┌────────────────┐                     │
│                     │  Adjudicator   │                     │
│                     │ (NLI / LLM)    │                     │
│                     └───────┬────────┘                     │
│                             │                              │
│                             ▼                              │
│                     ┌────────────────┐                     │
│                     │ Final Status   │                     │
│                     │ VERIFIED |     │                     │
│                     │ CONTESTED |    │                     │
│                     │ REFUTED        │                     │
│                     └────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Metrics

The FVA pipeline emits the following metrics for observability:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fva_falsification_triggered_total` | Counter | `reason` | Claims where falsification was triggered |
| `fva_status_changes_total` | Counter | `from_status`, `to_status` | Status transitions after adjudication |
| `fva_anti_context_docs` | Histogram | - | Number of anti-context documents retrieved |
| `fva_processing_duration_seconds` | Histogram | `phase` | Processing time by phase |
| `fva_wasted_falsification_total` | Counter | - | Falsifications with no anti-context found |
| `fva_claims_processed_total` | Counter | `final_status` | Total claims processed |
| `fva_adjudication_scores` | Histogram | `score_type` | Distribution of adjudication scores |
| `fva_timeout_total` | Counter | - | Falsification timeouts |
| `fva_budget_exhausted_total` | Counter | - | Budget exhaustion events |

## Programmatic Usage

### Creating an FVA Pipeline

```python
from tldw_Server_API.app.core.Claims_Extraction.fva_pipeline import (
    create_fva_pipeline_from_settings,
    FVAConfig,
    FVAPipeline,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import ClaimsEngine

# Create from settings (recommended)
pipeline = create_fva_pipeline_from_settings(
    claims_engine=claims_engine,
    retriever=retriever,
)

# Or with custom config
config = FVAConfig(
    enabled=True,
    confidence_threshold=0.7,
    contested_threshold=0.4,
)
pipeline = FVAPipeline(
    claims_engine=claims_engine,
    retriever=retriever,
    config=config,
)
```

### Processing Claims

```python
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import Claim

claims = [
    Claim(id="1", text="The sky is blue."),
    Claim(id="2", text="Water boils at 100°C at sea level."),
]

# Process batch
result = await pipeline.process_batch(
    claims=claims,
    query="Scientific facts",
    documents=retrieved_docs,
    user_id="user123",
)

print(f"Total claims: {result.total_claims}")
print(f"Falsifications triggered: {result.falsification_triggered_count}")
print(f"Status changes: {result.status_changes}")
```

## Best Practices

1. **Budget Management**: Set appropriate `max_budget_usd` to prevent runaway costs on large claim sets.

2. **Claim Types**: Configure `force_claim_types` for high-stakes claim categories (statistics, medical claims).

3. **Threshold Tuning**:
   - Lower `confidence_threshold` = more falsification checks = higher accuracy but slower
   - Higher `contested_threshold` = more claims marked as CONTESTED

4. **Timeout Configuration**: Increase `timeout_seconds` for slow retrieval systems.

5. **Concurrency**: Adjust `max_concurrent` based on your retrieval system's capacity.

## Limitations

- FVA adds latency (typically 1-5 seconds per claim when falsification is triggered)
- Requires sufficient document corpus for meaningful counter-evidence retrieval
- NLI-based adjudication works best with factual, objective claims
- Budget constraints may limit falsification coverage on large claim sets

## Related Components

- [Claims Engine](../Code_Documentation/Claims_Extraction/README.md) - Core claim extraction and verification
- [RAG Pipeline](./RAG_Pipeline.md) - Document retrieval for evidence
- [Verification Reports](../Code_Documentation/Claims_Extraction/verification_report.md) - Aggregated verification results
