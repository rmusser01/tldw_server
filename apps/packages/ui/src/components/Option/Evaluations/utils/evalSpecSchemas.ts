/**
 * Evaluation spec schemas
 * Provide defaults + light metadata for the visual spec builder.
 */

export type EvalSpecBuilderKind =
  | "response_quality"
  | "rag"
  | "exact_match"
  | "json"

export interface EvalSpecThresholdField {
  key: string
  label: string
  path: string
  min?: number
  max?: number
  step?: number
}

export interface EvalSpecBooleanField {
  key: string
  label: string
  path: string
}

export interface EvalSpecSchema {
  type: string
  label: string
  description?: string
  builder: EvalSpecBuilderKind
  defaultSpec: Record<string, any>
  metricOptions?: string[]
  thresholdFields?: EvalSpecThresholdField[]
  booleanFields?: EvalSpecBooleanField[]
}

export const EVAL_SPEC_SCHEMAS: Record<string, EvalSpecSchema> = {
  model_graded: {
    type: "model_graded",
    label: "model_graded",
    builder: "json",
    defaultSpec: {
      sub_type: "response_quality",
      metrics: ["relevance", "completeness", "clarity", "accuracy"],
      threshold: 0.7,
      evaluator_model: "openai"
    }
  },
  response_quality: {
    type: "response_quality",
    label: "response_quality",
    builder: "response_quality",
    metricOptions: ["relevance", "completeness", "clarity", "accuracy"],
    thresholdFields: [
      {
        key: "min_score",
        label: "Min score",
        path: "thresholds.min_score",
        min: 0,
        max: 1,
        step: 0.05
      }
    ],
    defaultSpec: {
      metrics: ["relevance", "completeness", "clarity", "accuracy"],
      model: "gpt-3.5-turbo",
      temperature: 0.3,
      thresholds: { min_score: 0.7 }
    }
  },
  rag: {
    type: "rag",
    label: "rag",
    builder: "rag",
    metricOptions: [
      "relevance",
      "faithfulness",
      "answer_similarity",
      "retrieval_precision",
      "answer_relevancy"
    ],
    thresholdFields: [
      {
        key: "min_relevance",
        label: "Min relevance",
        path: "thresholds.min_relevance",
        min: 0,
        max: 1,
        step: 0.05
      },
      {
        key: "min_faithfulness",
        label: "Min faithfulness",
        path: "thresholds.min_faithfulness",
        min: 0,
        max: 1,
        step: 0.05
      },
      {
        key: "min_answer_similarity",
        label: "Min answer similarity",
        path: "thresholds.min_answer_similarity",
        min: 0,
        max: 1,
        step: 0.05
      }
    ],
    defaultSpec: {
      metrics: ["relevance", "faithfulness", "answer_similarity"],
      model: "gpt-3.5-turbo",
      temperature: 0.3,
      thresholds: {
        min_relevance: 0.7,
        min_faithfulness: 0.7,
        min_answer_similarity: 0.7
      }
    }
  },
  rag_pipeline: {
    type: "rag_pipeline",
    label: "rag_pipeline",
    builder: "json",
    defaultSpec: {
      sub_type: "rag_pipeline",
      metrics: ["retrieval_precision", "faithfulness", "answer_relevancy"],
      evaluator_model: "openai"
    }
  },
  geval: {
    type: "geval",
    label: "geval",
    builder: "json",
    defaultSpec: {
      metrics: ["g_eval_score"],
      model: "gpt-3.5-turbo",
      temperature: 0
    }
  },
  exact_match: {
    type: "exact_match",
    label: "exact_match",
    builder: "exact_match",
    booleanFields: [
      {
        key: "case_sensitive",
        label: "Case sensitive",
        path: "case_sensitive"
      }
    ],
    defaultSpec: {
      metrics: ["exact_match"],
      model: "gpt-3.5-turbo",
      temperature: 0,
      case_sensitive: false
    }
  },
  includes: {
    type: "includes",
    label: "includes",
    builder: "json",
    defaultSpec: {
      metrics: ["includes"],
      case_sensitive: false
    }
  },
  fuzzy_match: {
    type: "fuzzy_match",
    label: "fuzzy_match",
    builder: "json",
    defaultSpec: {
      metrics: ["fuzzy_match"],
      threshold: 0.85
    }
  },
  proposition_extraction: {
    type: "proposition_extraction",
    label: "proposition_extraction",
    builder: "json",
    defaultSpec: {
      metrics: ["proposition_extraction"],
      evaluator_model: "openai",
      proposition_schema: ["claim", "evidence"]
    }
  },
  qa3: {
    type: "qa3",
    label: "qa3",
    builder: "json",
    defaultSpec: {
      metrics: ["qa3"],
      evaluator_model: "openai",
      labels: ["good", "borderline", "bad"]
    }
  },
  label_choice: {
    type: "label_choice",
    label: "label_choice",
    builder: "json",
    defaultSpec: {
      metrics: ["label_choice"],
      allowed_labels: ["A", "B", "C"]
    }
  },
  nli_factcheck: {
    type: "nli_factcheck",
    label: "nli_factcheck",
    builder: "json",
    defaultSpec: {
      metrics: ["nli_factcheck"],
      allowed_labels: ["entailed", "contradicted", "neutral"]
    }
  },
  ocr: {
    type: "ocr",
    label: "ocr",
    builder: "json",
    defaultSpec: {
      metrics: ["cer", "wer", "coverage"],
      language: "eng"
    }
  }
}

export const EVAL_SPEC_TYPES = Object.keys(EVAL_SPEC_SCHEMAS)

export const getEvalSpecSchema = (evalType: string): EvalSpecSchema | null =>
  EVAL_SPEC_SCHEMAS[evalType] || null
