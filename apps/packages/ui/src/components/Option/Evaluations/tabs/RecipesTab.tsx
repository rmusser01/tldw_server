/**
 * RecipesTab component
 * Recipe-first workflow for launching evaluation recipes and reading current reports.
 */

import React from "react"
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Collapse,
  Empty,
  Input,
  InputNumber,
  Select,
  Space,
  Spin,
  Tag,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import { useDatasetsList } from "../hooks/useDatasets"
import {
  getRecipeRunUserErrorMessage,
  useCreateRecipeRun,
  useRecipeLaunchReadiness,
  useRecipeManifests,
  useRecipeRunReport,
  useValidateRecipeDataset
} from "../hooks/useRecipes"
import { JsonEditor } from "../components"
import { EmbeddingsModelSelectionConfig } from "./recipe-configs/EmbeddingsModelSelectionConfig"
import { RagAnswerQualityConfig } from "./recipe-configs/RagAnswerQualityConfig"
import { RagRetrievalTuningConfig } from "./recipe-configs/RagRetrievalTuningConfig"
import type {
  DatasetSample,
  RecipeDatasetValidation,
  RecipeManifest
} from "@/services/evaluations"
import { useEvaluationsStore } from "@/store/evaluations"

const { Paragraph, Text, Title } = Typography
const { TextArea } = Input

const DEFAULT_INLINE_DATASETS: Record<string, string> = {
  summarization_quality: JSON.stringify(
    [
      {
        input: "Summarize this transcript into concise bullet notes.",
        expected: "A concise grounded summary."
      }
    ],
    null,
    2
  ),
  embeddings_model_selection: JSON.stringify(
    [
      {
        query_id: "q-1",
        input: "find alpha",
        expected_ids: ["1"]
      }
    ],
    null,
    2
  ),
  rag_retrieval_tuning: JSON.stringify(
    [
      {
        sample_id: "sample-1",
        query: ""
      }
    ],
    null,
    2
  ),
  rag_answer_quality: JSON.stringify(
    [
      {
        sample_id: "sample-1",
        query: "",
        expected_behavior: "answer"
      }
    ],
    null,
    2
  )
}

const DEFAULT_RUN_CONFIGS: Record<string, string> = {
  summarization_quality: JSON.stringify(
    {
      candidate_model_ids: ["openai:gpt-4.1-mini", "local:mistral-small"],
      judge_config: {
        provider: "openai",
        model: "gpt-4.1-mini"
      },
      prompts: {
        system: "Compare candidate summaries for groundedness, coverage, and usefulness."
      },
      weights: {
        grounding: 0.5,
        coverage: 0.3,
        usefulness: 0.2
      },
      comparison_mode: "leaderboard",
      source_normalization: {
        strip_citations: true
      },
      context_policy: {
        mode: "strict"
      },
      execution_policy: {
        max_parallel_candidates: 2
      }
    },
    null,
    2
  ),
  embeddings_model_selection: JSON.stringify(
    {
      comparison_mode: "embedding_only",
      candidates: [
        {
          provider: "openai",
          model: "text-embedding-3-small"
        },
        {
          provider: "local",
          model: "bge-small",
          is_local: true
        }
      ],
      media_ids: [1, 2]
    },
    null,
    2
  ),
  rag_retrieval_tuning: JSON.stringify(
    {
      candidate_creation_mode: "auto_sweep",
      corpus_scope: {
        sources: ["media_db", "notes"]
      },
      weak_supervision_budget: {
        review_sample_fraction: 0.2,
        max_review_samples: 25,
        min_review_samples: 3,
        synthetic_query_limit: 20
      },
      retrieval_config: {
        search_mode: "hybrid",
        top_k: 10,
        hybrid_alpha: 0.7,
        enable_reranking: true,
        reranking_strategy: "flashrank",
        rerank_top_k: 10
      },
      indexing_config: {
        chunking_preset: "baseline"
      }
    },
    null,
    2
  ),
  rag_answer_quality: JSON.stringify(
    {
      evaluation_mode: "fixed_context",
      supervision_mode: "rubric",
      context_snapshot_ref: "",
      candidates: [
        {
          candidate_id: "candidate-1",
          generation_model: "openai:gpt-4.1-mini",
          prompt_variant: "default",
          formatting_citation_mode: "citations_required"
        }
      ]
    },
    null,
    2
  )
}

const prettifySlotName = (slotName: string): string =>
  slotName
    .split("_")
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ")

const prettifySentenceSlotName = (slotName: string): string => {
  const label = slotName.replace(/_/g, " ")
  return label.charAt(0).toUpperCase() + label.slice(1)
}

const prettifyMetricName = (metricName: string): string =>
  metricName
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())

const formatMetricValue = (value: unknown): string =>
  typeof value === "number"
    ? value.toFixed(value >= 1 ? 2 : 3).replace(/0+$/, "").replace(/\.$/, "")
    : String(value)

const parseJsonObject = (text: string, label: string): Record<string, any> => {
  try {
    const parsed = JSON.parse(text)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`${label} must be a JSON object.`)
    }
    return parsed
  } catch (error: any) {
    throw new Error(error?.message || `Invalid ${label} JSON.`)
  }
}

const parseJsonDataset = (text: string): DatasetSample[] => {
  try {
    const parsed = JSON.parse(text)
    if (!Array.isArray(parsed)) {
      throw new Error("Inline dataset must be a JSON array.")
    }
    return parsed as DatasetSample[]
  } catch (error: any) {
    throw new Error(error?.message || "Invalid inline dataset JSON.")
  }
}

const cloneJson = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T

const parseJsonObjectSafe = (text: string): Record<string, any> | null => {
  try {
    const parsed = JSON.parse(text)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null
    }
    return parsed
  } catch {
    return null
  }
}

const parseJsonDatasetSafe = (text: string): DatasetSample[] | null => {
  try {
    return parseJsonDataset(text)
  } catch {
    return null
  }
}

const defaultInlineDatasetForRecipe = (recipeId: string | null): DatasetSample[] => {
  if (!recipeId) return []
  return parseJsonDataset(DEFAULT_INLINE_DATASETS[recipeId] || "[]")
}

const defaultRunConfigForRecipe = (recipeId: string | null): Record<string, any> => {
  if (!recipeId) return {}
  return parseJsonObject(DEFAULT_RUN_CONFIGS[recipeId] || "{}", "Run config")
}

const normalizeLineList = (value: string): string[] =>
  value
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean)

const joinLineList = (items: unknown): string =>
  Array.isArray(items)
    ? items
        .map((item) => String(item || "").trim())
        .filter(Boolean)
        .join("\n")
    : ""

const normalizeCsvList = (value: string): string[] =>
  value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)

const joinCsvList = (items: unknown): string =>
  Array.isArray(items)
    ? items
        .map((item) => String(item || "").trim())
        .filter(Boolean)
        .join(", ")
    : ""

export const RecipesTab: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const [selectedRecipeId, setSelectedRecipeId] = React.useState<string | null>(null)
  const [datasetSource, setDatasetSource] = React.useState<"inline" | "saved">("inline")
  const [selectedDatasetId, setSelectedDatasetId] = React.useState<string | null>(null)
  const [inlineDatasetText, setInlineDatasetText] = React.useState("")
  const [runConfigText, setRunConfigText] = React.useState("")
  const [forceRerun, setForceRerun] = React.useState(false)
  const [validationResult, setValidationResult] =
    React.useState<RecipeDatasetValidation | null>(null)
  const [localError, setLocalError] = React.useState<string | null>(null)
  const [currentRunId, setCurrentRunId] = React.useState<string | null>(null)
  const setActiveTab = useEvaluationsStore((s) => s.setActiveTab)
  const syntheticReviewRecipeKind = useEvaluationsStore(
    (s) => s.syntheticReviewRecipeKind
  )
  const syntheticReviewBatchId = useEvaluationsStore(
    (s) => s.syntheticReviewBatchId
  )
  const setSyntheticReviewRecipeKind = useEvaluationsStore(
    (s) => s.setSyntheticReviewRecipeKind
  )
  const lastAutoOpenedBatchId = React.useRef<string | null>(null)

  const {
    data: manifestsResp,
    isLoading: manifestsLoading,
    isError: manifestsError,
    error: manifestsErrorValue
  } = useRecipeManifests()
  const { data: datasetsResp } = useDatasetsList({ limit: 100, offset: 0 })
  const validateMutation = useValidateRecipeDataset()
  const createRunMutation = useCreateRecipeRun()
  const { data: readinessResp, isLoading: readinessLoading } =
    useRecipeLaunchReadiness(selectedRecipeId)
  const {
    data: reportResp,
    isLoading: reportLoading,
    isError: reportError,
    error: reportErrorValue
  } = useRecipeRunReport(currentRunId)

  const manifests = Array.isArray(manifestsResp?.data) ? manifestsResp.data : []
  const datasets = datasetsResp?.data?.data || []
  const selectedManifest =
    manifests.find((manifest) => manifest.recipe_id === selectedRecipeId) || null
  const launchReadiness = readinessResp?.data || null
  const report = reportResp?.data || null
  const reportPayload = report?.run?.metadata?.recipe_report as Record<string, any> | undefined
  const reportCandidates = Array.isArray(reportPayload?.candidates)
    ? reportPayload?.candidates
    : []
  const reportFailureExamples = Array.isArray(reportPayload?.failure_examples)
    ? reportPayload.failure_examples
    : []
  const isEmbeddingsRecipeReport =
    report?.run?.recipe_id === "embeddings_model_selection"
  const parsedInlineDataset =
    datasetSource === "inline" ? parseJsonDatasetSafe(inlineDatasetText) : null
  const parsedRunConfig = parseJsonObjectSafe(runConfigText)
  const inlineDatasetEditorInvalid =
    datasetSource === "inline" && inlineDatasetText.trim().length > 0 && !parsedInlineDataset
  const runConfigEditorInvalid =
    runConfigText.trim().length > 0 && !parsedRunConfig
  const isLaunchable = selectedManifest?.launchable !== false
  const syntheticReviewButtonLabel =
    selectedManifest?.recipe_id === "rag_retrieval_tuning"
      ? t("evaluations:recipeSyntheticReviewRetrievalCta", {
          defaultValue: "Review synthetic retrieval drafts"
        })
      : t("evaluations:recipeSyntheticReviewAnswerQualityCta", {
          defaultValue: "Review synthetic answer-quality drafts"
        })

  React.useEffect(() => {
    if (!selectedRecipeId && manifests.length > 0) {
      setSelectedRecipeId(manifests[0].recipe_id)
    }
  }, [manifests, selectedRecipeId])

  React.useEffect(() => {
    if (!selectedRecipeId) return
    setDatasetSource("inline")
    setSelectedDatasetId(null)
    setInlineDatasetText(DEFAULT_INLINE_DATASETS[selectedRecipeId] || "[]")
    setRunConfigText(DEFAULT_RUN_CONFIGS[selectedRecipeId] || "{}")
    setValidationResult(null)
    setLocalError(null)
    setCurrentRunId(null)
    setForceRerun(false)
  }, [selectedRecipeId])

  React.useEffect(() => {
    if (datasets.length === 0) {
      setSelectedDatasetId(null)
      return
    }

    setSelectedDatasetId((current) => {
      if (current && datasets.some((dataset) => dataset.id === current)) {
        return current
      }
      return datasets[0]?.id || null
    })
  }, [datasets])

  React.useEffect(() => {
    const recipeKind = syntheticReviewRecipeKind || null
    const batchId = syntheticReviewBatchId || null
    const supportsSyntheticReview =
      recipeKind === "rag_retrieval_tuning" || recipeKind === "rag_answer_quality"

    if (!supportsSyntheticReview || !batchId) {
      return
    }

    if (lastAutoOpenedBatchId.current === batchId) {
      return
    }

    lastAutoOpenedBatchId.current = batchId
    setActiveTab("synthetic-review")
  }, [setActiveTab, syntheticReviewBatchId, syntheticReviewRecipeKind])

  const buildDatasetPayload = (): {
    datasetId?: string
    dataset?: DatasetSample[]
  } => {
    if (datasetSource === "saved") {
      if (!selectedDatasetId) {
        throw new Error("Select a saved dataset before continuing.")
      }
      return { datasetId: selectedDatasetId }
    }
    return {
      dataset: parseJsonDataset(inlineDatasetText)
    }
  }

  const openSyntheticReview = () => {
    if (!selectedManifest?.recipe_id) return
    setSyntheticReviewRecipeKind(selectedManifest.recipe_id)
    setActiveTab("synthetic-review")
  }

  const replaceInlineDataset = (nextDataset: DatasetSample[]) => {
    setInlineDatasetText(JSON.stringify(nextDataset, null, 2))
  }

  const replaceRunConfig = (nextRunConfig: Record<string, any>) => {
    setRunConfigText(JSON.stringify(nextRunConfig, null, 2))
  }

  const updateInlineDataset = (
    updater: (dataset: DatasetSample[]) => DatasetSample[]
  ) => {
    const baseDataset = cloneJson(
      parsedInlineDataset || defaultInlineDatasetForRecipe(selectedRecipeId)
    )
    replaceInlineDataset(updater(baseDataset))
  }

  const updateRunConfig = (
    updater: (runConfig: Record<string, any>) => Record<string, any>
  ) => {
    const baseRunConfig = cloneJson(
      parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)
    )
    replaceRunConfig(updater(baseRunConfig))
  }

  const handleValidate = async () => {
    if (!selectedManifest) return
    try {
      setLocalError(null)
      const datasetPayload = buildDatasetPayload()
      const runConfig = parseJsonObject(runConfigText, "Run config")
      const resp = await validateMutation.mutateAsync({
        recipeId: selectedManifest.recipe_id,
        datasetId: datasetPayload.datasetId,
        dataset: datasetPayload.dataset,
        runConfig
      })
      setValidationResult(resp.data as RecipeDatasetValidation)
    } catch (error: any) {
      setValidationResult(null)
      setLocalError(error?.message || "Failed to validate dataset.")
    }
  }

  const handleRun = async () => {
    if (!selectedManifest) return
    try {
      setLocalError(null)
      const datasetPayload = buildDatasetPayload()
      const runConfig = parseJsonObject(runConfigText, "Run config")
      const resp = await createRunMutation.mutateAsync({
        recipeId: selectedManifest.recipe_id,
        datasetId: datasetPayload.datasetId,
        dataset: datasetPayload.dataset,
        runConfig,
        forceRerun
      })
      setCurrentRunId((resp as any)?.data?.run_id || (resp as any)?.run_id || null)
    } catch (error: any) {
      setLocalError(getRecipeRunUserErrorMessage(error))
    }
  }

  const canEnqueueRuns = launchReadiness?.can_enqueue_runs !== false
  const canReuseCompletedRuns = launchReadiness?.can_reuse_completed_runs !== false
  const runButtonDisabled = !isLaunchable || (!canEnqueueRuns && forceRerun)
  const runButtonLabel = !canEnqueueRuns && canReuseCompletedRuns && !forceRerun
    ? t("evaluations:recipeReuseRunCta", {
        defaultValue: "Try matching run"
      })
    : t("evaluations:recipeRunCta", {
        defaultValue: "Run recipe"
      })

  const renderSummarizationDatasetEditor = () => {
    const samples =
      (parsedInlineDataset || defaultInlineDatasetForRecipe(selectedRecipeId)).map(
        (sample) => ({
          ...sample,
          input: typeof sample.input === "string" ? sample.input : "",
          expected: typeof sample.expected === "string" ? sample.expected : ""
        })
      ) || []

    return (
      <div className="space-y-3">
        <div className="text-xs text-text-muted">
          {t("evaluations:recipeSummarizationDatasetHint", {
            defaultValue:
              "Add source text and, if you have one, a reference summary for each sample."
          })}
        </div>
        {samples.map((sample, index) => (
          <Card
            key={`summarization-sample-${index}`}
            size="small"
            title={t("evaluations:recipeSampleCardTitle", {
              defaultValue: `Sample ${index + 1}`
            })}
            extra={
              <Button
                size="small"
                onClick={() =>
                  updateInlineDataset((dataset) =>
                    dataset.filter((_, datasetIndex) => datasetIndex !== index)
                  )
                }
                disabled={samples.length <= 1}
              >
                {t("common:remove", { defaultValue: "Remove" })}
              </Button>
            }
          >
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <Text strong>
                  {t("evaluations:recipeSourceTextLabel", {
                    defaultValue: `Source text ${index + 1}`
                  })}
                </Text>
                <TextArea
                  aria-label={`Source text ${index + 1}`}
                  rows={4}
                  className="mt-2"
                  value={sample.input}
                  onChange={(event) =>
                    updateInlineDataset((dataset) =>
                      dataset.map((datasetSample, datasetIndex) =>
                        datasetIndex === index
                          ? { ...datasetSample, input: event.target.value }
                          : datasetSample
                      )
                    )
                  }
                />
              </div>
              <div>
                <Text strong>
                  {t("evaluations:recipeReferenceSummaryLabel", {
                    defaultValue: `Reference summary ${index + 1} (optional)`
                  })}
                </Text>
                <TextArea
                  aria-label={`Reference summary ${index + 1} (optional)`}
                  rows={4}
                  className="mt-2"
                  value={sample.expected}
                  onChange={(event) =>
                    updateInlineDataset((dataset) =>
                      dataset.map((datasetSample, datasetIndex) => {
                        if (datasetIndex !== index) return datasetSample
                        const nextValue = event.target.value
                        if (!nextValue.trim()) {
                          const nextSample = { ...datasetSample }
                          delete nextSample.expected
                          return nextSample
                        }
                        return { ...datasetSample, expected: nextValue }
                      })
                    )
                  }
                />
              </div>
            </div>
          </Card>
        ))}
        <Button
          onClick={() =>
            updateInlineDataset((dataset) => [
              ...dataset,
              {
                input: "",
                expected: ""
              }
            ])
          }
        >
          {t("evaluations:recipeAddSampleCta", {
            defaultValue: "Add sample"
          })}
        </Button>
      </div>
    )
  }

  const renderEmbeddingsDatasetEditor = () => {
    const samples =
      (parsedInlineDataset || defaultInlineDatasetForRecipe(selectedRecipeId)).map(
        (sample, index) => ({
          ...sample,
          query_id:
            typeof (sample as any).query_id === "string"
              ? String((sample as any).query_id)
              : `q-${index + 1}`,
          input: typeof sample.input === "string" ? sample.input : "",
          expected_ids: Array.isArray((sample as any).expected_ids)
            ? (sample as any).expected_ids.map((value: unknown) => String(value))
            : []
        })
      ) || []

    return (
      <div className="space-y-3">
        <div className="text-xs text-text-muted">
          {t("evaluations:recipeEmbeddingsDatasetHint", {
            defaultValue:
              "Add a query id, query text, and optional labeled media ids for each retrieval sample."
          })}
        </div>
        {samples.map((sample, index) => (
          <Card
            key={`embeddings-sample-${index}`}
            size="small"
            title={t("evaluations:recipeQueryCardTitle", {
              defaultValue: `Query ${index + 1}`
            })}
            extra={
              <Button
                size="small"
                onClick={() =>
                  updateInlineDataset((dataset) =>
                    dataset.filter((_, datasetIndex) => datasetIndex !== index)
                  )
                }
                disabled={samples.length <= 1}
              >
                {t("common:remove", { defaultValue: "Remove" })}
              </Button>
            }
          >
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <Text strong>
                  {t("evaluations:recipeQueryIdLabel", {
                    defaultValue: `Query ID ${index + 1}`
                  })}
                </Text>
                <Input
                  aria-label={`Query ID ${index + 1}`}
                  className="mt-2"
                  value={sample.query_id}
                  onChange={(event) =>
                    updateInlineDataset((dataset) =>
                      dataset.map((datasetSample, datasetIndex) =>
                        datasetIndex === index
                          ? { ...datasetSample, query_id: event.target.value }
                          : datasetSample
                      )
                    )
                  }
                />
              </div>
              <div>
                <Text strong>
                  {t("evaluations:recipeRelevantMediaLabel", {
                    defaultValue: `Relevant media IDs ${index + 1} (optional)`
                  })}
                </Text>
                <Input
                  aria-label={`Relevant media IDs ${index + 1} (optional)`}
                  className="mt-2"
                  value={joinCsvList(sample.expected_ids)}
                  onChange={(event) =>
                    updateInlineDataset((dataset) =>
                      dataset.map((datasetSample, datasetIndex) => {
                        if (datasetIndex !== index) return datasetSample
                        const expectedIds = normalizeCsvList(event.target.value)
                        if (expectedIds.length === 0) {
                          const nextSample = { ...datasetSample }
                          delete (nextSample as any).expected_ids
                          return nextSample
                        }
                        return { ...datasetSample, expected_ids: expectedIds }
                      })
                    )
                  }
                />
              </div>
              <div className="md:col-span-2">
                <Text strong>
                  {t("evaluations:recipeQueryTextLabel", {
                    defaultValue: `Query text ${index + 1}`
                  })}
                </Text>
                <TextArea
                  aria-label={`Query text ${index + 1}`}
                  rows={3}
                  className="mt-2"
                  value={sample.input}
                  onChange={(event) =>
                    updateInlineDataset((dataset) =>
                      dataset.map((datasetSample, datasetIndex) =>
                        datasetIndex === index
                          ? { ...datasetSample, input: event.target.value }
                          : datasetSample
                      )
                    )
                  }
                />
              </div>
            </div>
          </Card>
        ))}
        <Button
          onClick={() =>
            updateInlineDataset((dataset) => [
              ...dataset,
              {
                query_id: `q-${dataset.length + 1}`,
                input: ""
              }
            ])
          }
        >
          {t("evaluations:recipeAddQueryCta", {
            defaultValue: "Add query"
          })}
        </Button>
      </div>
    )
  }

  const renderSummarizationRunConfigEditor = () => {
    const runConfig = parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)
    const weights = runConfig.weights || {}

    return (
      <div className="grid gap-3 md:grid-cols-2">
        <div className="md:col-span-2">
          <Text strong>
            {t("evaluations:recipeCandidateModelsLabel", {
              defaultValue: "Candidate models"
            })}
          </Text>
          <TextArea
            aria-label="Candidate models"
            rows={4}
            className="mt-2"
            value={joinLineList(runConfig.candidate_model_ids)}
            onChange={(event) =>
              updateRunConfig((current) => ({
                ...current,
                candidate_model_ids: normalizeLineList(event.target.value)
              }))
            }
          />
          <div className="mt-1 text-xs text-text-muted">
            {t("evaluations:recipeCandidateModelsHint", {
              defaultValue: "Enter one model id per line, for example provider:model."
            })}
          </div>
        </div>
        <div>
          <Text strong>
            {t("evaluations:recipeJudgeProviderLabel", {
              defaultValue: "Judge provider"
            })}
          </Text>
          <Input
            aria-label="Judge provider"
            className="mt-2"
            value={String(runConfig.judge_config?.provider || "")}
            onChange={(event) =>
              updateRunConfig((current) => ({
                ...current,
                judge_config: {
                  ...(current.judge_config || {}),
                  provider: event.target.value
                }
              }))
            }
          />
        </div>
        <div>
          <Text strong>
            {t("evaluations:recipeJudgeModelLabel", {
              defaultValue: "Judge model"
            })}
          </Text>
          <Input
            aria-label="Judge model"
            className="mt-2"
            value={String(runConfig.judge_config?.model || "")}
            onChange={(event) =>
              updateRunConfig((current) => ({
                ...current,
                judge_config: {
                  ...(current.judge_config || {}),
                  model: event.target.value
                }
              }))
            }
          />
        </div>
        <div>
          <Text strong>
            {t("evaluations:recipeComparisonModeLabel", {
              defaultValue: "Comparison mode"
            })}
          </Text>
          <select
            aria-label="Comparison mode"
            className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={String(runConfig.comparison_mode || "leaderboard")}
            onChange={(event) =>
              updateRunConfig((current) => ({
                ...current,
                comparison_mode: event.target.value
              }))
            }>
            <option value="leaderboard">leaderboard</option>
            <option value="pairwise">pairwise</option>
          </select>
        </div>
        <div className="md:col-span-2">
          <Text strong>
            {t("evaluations:recipeRubricWeightsLabel", {
              defaultValue: "Rubric weights"
            })}
          </Text>
          <div className="mt-2 grid gap-3 md:grid-cols-3">
            {[
              { key: "grounding", label: "Grounding" },
              { key: "coverage", label: "Coverage" },
              { key: "usefulness", label: "Usefulness" }
            ].map((item) => (
              <div key={item.key}>
                <Text>{item.label}</Text>
                <InputNumber
                  className="mt-2 w-full"
                  min={0}
                  step={0.1}
                  value={Number(weights[item.key] ?? 0)}
                  onChange={(value) =>
                    updateRunConfig((current) => ({
                      ...current,
                      weights: {
                        ...(current.weights || {}),
                        [item.key]: typeof value === "number" ? value : 0
                      }
                    }))
                  }
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderEmbeddingsRunConfigEditor = () => {
    const runConfig = parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)
    const candidates = Array.isArray(runConfig.candidates) ? runConfig.candidates : []

    return (
      <div className="space-y-3">
        <div className="grid gap-3 md:grid-cols-2">
        <div>
          <Text strong>
            {t("evaluations:recipeComparisonModeLabel", {
              defaultValue: "Comparison mode"
            })}
          </Text>
          <select
            aria-label="Comparison mode"
            className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={String(runConfig.comparison_mode || "embedding_only")}
            onChange={(event) =>
              updateRunConfig((current) => ({
                ...current,
                comparison_mode: event.target.value
              }))
            }>
            <option value="embedding_only">embedding_only</option>
            <option value="retrieval_stack">retrieval_stack</option>
          </select>
        </div>
          <div>
            <Text strong>
              {t("evaluations:recipeMediaIdsLabel", {
                defaultValue: "Media IDs"
              })}
            </Text>
            <Input
              aria-label="Media IDs"
              className="mt-2"
              value={joinCsvList(runConfig.media_ids)}
              onChange={(event) =>
                updateRunConfig((current) => ({
                  ...current,
                  media_ids: normalizeCsvList(event.target.value).map((value) =>
                    Number.parseInt(value, 10)
                  )
                }))
              }
            />
          </div>
          <div>
            <Text strong>
              {t("evaluations:recipeTopKLabel", {
                defaultValue: "Top K"
              })}
            </Text>
            <InputNumber
              className="mt-2 w-full"
              min={1}
              value={typeof runConfig.top_k === "number" ? runConfig.top_k : undefined}
              onChange={(value) =>
                updateRunConfig((current) => {
                  const nextConfig = { ...current }
                  if (typeof value === "number") {
                    nextConfig.top_k = value
                  } else {
                    delete nextConfig.top_k
                  }
                  return nextConfig
                })
              }
            />
          </div>
          <div>
            <Text strong>
              {t("evaluations:recipeHybridAlphaLabel", {
                defaultValue: "Hybrid alpha"
              })}
            </Text>
            <InputNumber
              className="mt-2 w-full"
              min={0}
              max={1}
              step={0.1}
              value={
                typeof runConfig.hybrid_alpha === "number"
                  ? runConfig.hybrid_alpha
                  : undefined
              }
              onChange={(value) =>
                updateRunConfig((current) => {
                  const nextConfig = { ...current }
                  if (typeof value === "number") {
                    nextConfig.hybrid_alpha = value
                  } else {
                    delete nextConfig.hybrid_alpha
                  }
                  return nextConfig
                })
              }
            />
          </div>
        </div>
        <div className="space-y-3">
          <Text strong>
            {t("evaluations:recipeCandidateRowsLabel", {
              defaultValue: "Candidates"
            })}
          </Text>
          {candidates.map((candidate, index) => (
            <Card
              key={`candidate-${index}`}
              size="small"
              title={t("evaluations:recipeCandidateCardTitle", {
                defaultValue: `Candidate ${index + 1}`
              })}
              extra={
                <Button
                  size="small"
                  onClick={() =>
                    updateRunConfig((current) => ({
                      ...current,
                      candidates: (Array.isArray(current.candidates)
                        ? current.candidates
                        : []
                      ).filter((_: unknown, candidateIndex: number) => candidateIndex !== index)
                    }))
                  }
                  disabled={candidates.length <= 1}
                >
                  {t("common:remove", { defaultValue: "Remove" })}
                </Button>
              }
            >
              <div className="grid gap-3 md:grid-cols-3">
                <div>
                  <Text strong>{t("common:provider", { defaultValue: "Provider" })}</Text>
                  <Input
                    aria-label={`Provider ${index + 1}`}
                    className="mt-2"
                    value={String(candidate?.provider || "")}
                    onChange={(event) =>
                      updateRunConfig((current) => ({
                        ...current,
                        candidates: (Array.isArray(current.candidates)
                          ? current.candidates
                          : []
                        ).map((item: Record<string, any>, candidateIndex: number) =>
                          candidateIndex === index
                            ? { ...item, provider: event.target.value }
                            : item
                        )
                      }))
                    }
                  />
                </div>
                <div>
                  <Text strong>{t("common:model", { defaultValue: "Model" })}</Text>
                  <Input
                    aria-label={`Model ${index + 1}`}
                    className="mt-2"
                    value={String(candidate?.model || "")}
                    onChange={(event) =>
                      updateRunConfig((current) => ({
                        ...current,
                        candidates: (Array.isArray(current.candidates)
                          ? current.candidates
                          : []
                        ).map((item: Record<string, any>, candidateIndex: number) =>
                          candidateIndex === index
                            ? { ...item, model: event.target.value }
                            : item
                        )
                      }))
                    }
                  />
                </div>
                <div className="flex items-end">
                  <Checkbox
                    aria-label={`Local candidate ${index + 1}`}
                    checked={Boolean(candidate?.is_local)}
                    onChange={(event) =>
                      updateRunConfig((current) => ({
                        ...current,
                        candidates: (Array.isArray(current.candidates)
                          ? current.candidates
                          : []
                        ).map((item: Record<string, any>, candidateIndex: number) =>
                          candidateIndex === index
                            ? { ...item, is_local: event.target.checked }
                            : item
                        )
                      }))
                    }
                  >
                    {t("evaluations:recipeIsLocalLabel", {
                      defaultValue: "Local candidate"
                    })}
                  </Checkbox>
                </div>
              </div>
            </Card>
          ))}
          <Button
            onClick={() =>
              updateRunConfig((current) => ({
                ...current,
                candidates: [
                  ...(Array.isArray(current.candidates) ? current.candidates : []),
                  {
                    provider: "",
                    model: "",
                    is_local: false
                  }
                ]
              }))
            }
          >
            {t("evaluations:recipeAddCandidateCta", {
              defaultValue: "Add candidate"
            })}
          </Button>
        </div>
      </div>
    )
  }

  const renderGuidedDatasetEditor = () => {
    if (selectedManifest?.recipe_id === "embeddings_model_selection") {
      return renderEmbeddingsDatasetEditor()
    }
    return renderSummarizationDatasetEditor()
  }

  const renderGuidedRunConfigEditor = () => {
    if (selectedManifest?.recipe_id === "embeddings_model_selection") {
      return renderEmbeddingsRunConfigEditor()
    }
    return renderSummarizationRunConfigEditor()
  }

  const renderEmbeddingsRecommendationCards = () => {
    const recommendationSlots = report?.recommendation_slots || {}
    const candidateById = new Map<string, any>()
    reportCandidates.forEach((candidate: any) => {
      const keys = [
        candidate.candidate_run_id,
        candidate.candidate_id,
        candidate.model ? `${candidate.provider || ""}:${candidate.model}` : null
      ]
      keys
        .map((key) => (key == null ? "" : String(key).trim()))
        .filter(Boolean)
        .forEach((key) => candidateById.set(key, candidate))
    })
    const reportWarnings = [
      ...(Array.isArray(reportPayload?.warnings) ? reportPayload.warnings : []),
      ...(Array.isArray(reportPayload?.confidence_warnings)
        ? reportPayload.confidence_warnings
        : [])
    ]

    return (
      <div className="space-y-3">
        {reportWarnings.map((warning: unknown, index: number) => (
          <Alert
            key={`embeddings-warning-${index}`}
            type="warning"
            showIcon
            title={String(warning)}
          />
        ))}
        <div className="grid gap-3 md:grid-cols-3">
          {["best_overall", "best_local", "best_cheap"].map((slotName) => {
            const slot = (recommendationSlots as Record<string, any>)[slotName] || {}
            const slotMetadata =
              slot.metadata && typeof slot.metadata === "object" && !Array.isArray(slot.metadata)
                ? slot.metadata
                : {}
            const candidateId = String(
              slotMetadata.candidate_run_id || slot.candidate_run_id || ""
            ).trim()
            const candidate = candidateById.get(candidateId)
            const metrics =
              candidate && typeof candidate.metrics === "object" && candidate.metrics
                ? candidate.metrics
                : {}
            const applyWarnings = Array.isArray(slotMetadata.apply_warnings)
              ? slotMetadata.apply_warnings.map((warning: unknown) => String(warning)).filter(Boolean)
              : []
            const blockedReason =
              applyWarnings.length > 0
                ? applyWarnings.join(" ")
                : slotMetadata.apply_blocked_reason ||
                  slotMetadata.blocked_reason ||
                  slotMetadata.apply_ineligible_reason ||
                  slotMetadata.ineligible_reason
            const modelName = candidate?.model || slotMetadata.model || candidateId
            const providerName = candidate?.provider || slotMetadata.provider

            return (
              <Card key={slotName} size="small" styles={{ body: { padding: 12 } }}>
                <div className="space-y-2">
                  <Text strong>{prettifySentenceSlotName(slotName)}</Text>
                  <div>
                    <div className="font-medium">
                      {modelName || t("evaluations:recipeNoWinner", {
                        defaultValue: "No winner yet"
                      })}
                    </div>
                    {providerName && (
                      <div className="text-xs text-text-muted">{providerName}</div>
                    )}
                  </div>
                  {slot.explanation && (
                    <Paragraph className="mb-0 text-xs">{slot.explanation}</Paragraph>
                  )}
                  {Object.entries(metrics).length > 0 && (
                    <div className="flex flex-wrap gap-2 text-xs">
                      {Object.entries(metrics).map(([key, value]) => (
                        <Tag key={`${slotName}-${key}`}>
                          {prettifyMetricName(key)}: {formatMetricValue(value)}
                        </Tag>
                      ))}
                    </div>
                  )}
                  {slotMetadata.apply_eligible === true ? (
                    <Button size="small">
                      {t("evaluations:recipePreviewRagConfigChange", {
                        defaultValue: "Preview RAG config change"
                      })}
                    </Button>
                  ) : blockedReason ? (
                    <div className="text-xs text-text-muted">{String(blockedReason)}</div>
                  ) : null}
                </div>
              </Card>
            )
          })}
        </div>
        <Collapse
          ghost
          destroyOnHidden
          items={[
            {
              key: "embeddings-raw-report",
              label: t("evaluations:recipeRawReportDetailsLabel", {
                defaultValue: "Raw report details"
              }),
              children: (
                <pre className="max-h-80 overflow-auto rounded border border-border-subtle bg-surface-subtle p-3 text-xs">
                  {JSON.stringify(report, null, 2)}
                </pre>
              )
            }
          ]}
        />
      </div>
    )
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
      <Card
        title={t("evaluations:recipesTitle", {
          defaultValue: "Recipes"
        })}
      >
        {manifestsLoading ? (
          <div className="flex justify-center py-4">
            <Spin />
          </div>
        ) : manifestsError ? (
          <Alert
            type="warning"
            showIcon
            title={t("evaluations:recipesLoadErrorTitle", {
              defaultValue: "Unable to load recipes"
            })}
            description={(manifestsErrorValue as Error | null)?.message}
          />
        ) : manifests.length === 0 ? (
          <Empty
            description={t("evaluations:recipesEmpty", {
              defaultValue: "No recipes are registered yet."
            })}
          />
        ) : (
          <div className="space-y-3">
            {manifests.map((manifest: RecipeManifest) => (
              <Card
                key={manifest.recipe_id}
                size="small"
                styles={{ body: { padding: 12 } }}
                className={
                  selectedRecipeId === manifest.recipe_id
                    ? "border-primary"
                    : undefined
                }
              >
                <div className="space-y-2">
                  <div>
                    <div className="font-medium">{manifest.name}</div>
                    <Paragraph className="mb-0 text-xs text-text-muted">
                      {manifest.description}
                    </Paragraph>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {manifest.supported_modes.map((mode) => (
                      <Tag key={mode}>{mode}</Tag>
                    ))}
                  </div>
                  <Button
                    block
                    type={
                      selectedRecipeId === manifest.recipe_id ? "primary" : "default"
                    }
                    onClick={() => setSelectedRecipeId(manifest.recipe_id)}
                  >
                    {t("evaluations:recipesUseCta", {
                      defaultValue: `Use ${manifest.name}`
                    })}
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </Card>

      <div className="space-y-4">
        {selectedManifest ? (
          <Card
            title={selectedManifest.name}
            extra={<Tag>v{selectedManifest.recipe_version}</Tag>}
          >
            <Paragraph className="text-sm text-text-muted">
              {selectedManifest.description}
            </Paragraph>

            <div className="space-y-3">
              {launchReadiness?.message && (
                <Alert
                  type="warning"
                  showIcon
                  title={launchReadiness.message}
                  description={
                    canReuseCompletedRuns
                      ? t("evaluations:recipeLaunchReadinessReuseHint", {
                          defaultValue:
                            "You can still try to reuse a matching completed run with the current config."
                        })
                      : undefined
                  }
                />
              )}

              <div>
                <Text strong>
                  {t("evaluations:recipeDatasetSourceLabel", {
                    defaultValue: "Dataset source"
                  })}
                </Text>
                <Space className="mt-2">
                  <Button
                    type={datasetSource === "inline" ? "primary" : "default"}
                    onClick={() => setDatasetSource("inline")}
                  >
                    {t("evaluations:recipeInlineDatasetCta", {
                      defaultValue: "Inline dataset"
                    })}
                  </Button>
                  <Button
                    type={datasetSource === "saved" ? "primary" : "default"}
                    onClick={() => setDatasetSource("saved")}
                    disabled={datasets.length === 0}
                  >
                    {t("evaluations:recipeSavedDatasetCta", {
                      defaultValue: "Saved dataset"
                    })}
                  </Button>
                </Space>
                {datasets.length === 0 && (
                  <div className="mt-2 text-xs text-text-muted">
                    {t("evaluations:recipeSavedDatasetEmptyHint", {
                      defaultValue:
                        "No saved datasets yet. Create one from the Datasets tab or use an inline dataset."
                    })}
                  </div>
                )}
              </div>

              {datasetSource === "saved" ? (
                <div>
                  <Text strong>
                    {t("evaluations:recipeSavedDatasetLabel", {
                      defaultValue: "Dataset"
                    })}
                  </Text>
                  <Select
                    className="mt-2 w-full"
                    value={selectedDatasetId || undefined}
                    onChange={(value) => setSelectedDatasetId(value)}
                    options={datasets.map((dataset) => ({
                      value: dataset.id,
                      label: `${dataset.name} (${dataset.sample_count})`
                    }))}
                  />
                </div>
              ) : (
                <div className="space-y-3">
                  <Alert
                    type="info"
                    showIcon
                    title={t("evaluations:recipeGuidedSetupTitle", {
                      defaultValue: "Guided setup"
                    })}
                    description={t("evaluations:recipeGuidedSetupDescription", {
                      defaultValue:
                        "Use the guided fields below for the common setup path. Advanced JSON is still available if you need to fine-tune the payload."
                    })}
                  />

                  {inlineDatasetEditorInvalid && (
                    <Alert
                      type="warning"
                      showIcon
                      title={t("evaluations:recipeInlineJsonInvalidTitle", {
                        defaultValue: "Inline dataset JSON is invalid."
                      })}
                      description={t("evaluations:recipeInlineJsonInvalidDescription", {
                        defaultValue:
                          "Guided edits will restore a valid inline dataset payload."
                      })}
                    />
                  )}

                  {runConfigEditorInvalid && (
                    <Alert
                      type="warning"
                      showIcon
                      title={t("evaluations:recipeRunConfigInvalidTitle", {
                        defaultValue: "Run config JSON is invalid."
                      })}
                      description={t("evaluations:recipeRunConfigInvalidDescription", {
                        defaultValue:
                          "Guided edits will restore a valid run config payload."
                      })}
                    />
                  )}

                  <div>
                    {selectedManifest.recipe_id === "embeddings_model_selection" ? (
                      <EmbeddingsModelSelectionConfig
                        datasetSource="inline"
                        dataset={parsedInlineDataset || defaultInlineDatasetForRecipe(selectedRecipeId)}
                        runConfig={parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)}
                        manifest={selectedManifest}
                        onDatasetChange={replaceInlineDataset}
                        onRunConfigChange={replaceRunConfig}
                      />
                    ) : selectedManifest.recipe_id === "rag_retrieval_tuning" ? (
                      <RagRetrievalTuningConfig
                        datasetSource="inline"
                        dataset={parsedInlineDataset || defaultInlineDatasetForRecipe(selectedRecipeId)}
                        runConfig={parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)}
                        onDatasetChange={replaceInlineDataset}
                        onRunConfigChange={replaceRunConfig}
                      />
                    ) : selectedManifest.recipe_id === "rag_answer_quality" ? (
                      <RagAnswerQualityConfig
                        datasetSource="inline"
                        dataset={parsedInlineDataset || defaultInlineDatasetForRecipe(selectedRecipeId)}
                        runConfig={parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)}
                        onDatasetChange={replaceInlineDataset}
                        onRunConfigChange={replaceRunConfig}
                      />
                    ) : (
                      <div className="space-y-3">
                        <div>
                          <Text strong>
                            {t("evaluations:recipeGuidedDatasetLabel", {
                              defaultValue: "Dataset samples"
                            })}
                          </Text>
                          <div className="mt-2">{renderGuidedDatasetEditor()}</div>
                        </div>

                        <div>
                          <Text strong>
                            {t("evaluations:recipeGuidedRunConfigLabel", {
                              defaultValue: "Run settings"
                            })}
                          </Text>
                          <div className="mt-2">{renderGuidedRunConfigEditor()}</div>
                        </div>
                      </div>
                    )}
                  </div>

                  <Collapse
                    ghost
                    destroyOnHidden
                    items={[
                      {
                        key: "advanced-json",
                        label: t("evaluations:recipeAdvancedJsonLabel", {
                          defaultValue: "Advanced JSON"
                        }),
                        children: (
                          <div className="space-y-3">
                            <div>
                              <Text strong>
                                {t("evaluations:recipeInlineDatasetLabel", {
                                  defaultValue: "Inline dataset JSON"
                                })}
                              </Text>
                              <JsonEditor
                                rows={8}
                                value={inlineDatasetText}
                                onChange={setInlineDatasetText}
                              />
                            </div>
                            <div>
                              <Text strong>
                                {t("evaluations:recipeRunConfigLabel", {
                                  defaultValue: "Run config JSON"
                                })}
                              </Text>
                              <JsonEditor
                                rows={10}
                                value={runConfigText}
                                onChange={setRunConfigText}
                              />
                            </div>
                          </div>
                        )
                      }
                    ]}
                  />
                </div>
              )}

              {datasetSource === "saved" && (
                <div className="space-y-3">
                  {runConfigEditorInvalid && (
                    <Alert
                      type="warning"
                      showIcon
                      title={t("evaluations:recipeRunConfigInvalidTitle", {
                        defaultValue: "Run config JSON is invalid."
                      })}
                      description={t("evaluations:recipeRunConfigInvalidDescription", {
                        defaultValue:
                          "Guided edits will restore a valid run config payload."
                      })}
                    />
                  )}
                  {selectedManifest.recipe_id === "rag_retrieval_tuning" ? (
                    <RagRetrievalTuningConfig
                      datasetSource="saved"
                      dataset={[]}
                      runConfig={parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)}
                      onDatasetChange={() => {}}
                      onRunConfigChange={replaceRunConfig}
                    />
                  ) : selectedManifest.recipe_id === "embeddings_model_selection" ? (
                    <EmbeddingsModelSelectionConfig
                      datasetSource="saved"
                      dataset={[]}
                      runConfig={parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)}
                      manifest={selectedManifest}
                      onDatasetChange={() => {}}
                      onRunConfigChange={replaceRunConfig}
                    />
                  ) : selectedManifest.recipe_id === "rag_answer_quality" ? (
                    <RagAnswerQualityConfig
                      datasetSource="saved"
                      dataset={[]}
                      runConfig={parsedRunConfig || defaultRunConfigForRecipe(selectedRecipeId)}
                      onDatasetChange={() => {}}
                      onRunConfigChange={replaceRunConfig}
                    />
                  ) : (
                    <div>
                      <Text strong>
                        {t("evaluations:recipeGuidedRunConfigLabel", {
                          defaultValue: "Run settings"
                        })}
                      </Text>
                      <div className="mt-2">{renderGuidedRunConfigEditor()}</div>
                    </div>
                  )}
                  <Collapse
                    ghost
                    destroyOnHidden
                    items={[
                      {
                        key: "advanced-run-config-json",
                        label: t("evaluations:recipeAdvancedJsonLabel", {
                          defaultValue: "Advanced JSON"
                        }),
                        children: (
                          <div>
                            <Text strong>
                              {t("evaluations:recipeRunConfigLabel", {
                                defaultValue: "Run config JSON"
                              })}
                            </Text>
                            <JsonEditor
                              rows={10}
                              value={runConfigText}
                              onChange={setRunConfigText}
                            />
                          </div>
                        )
                      }
                    ]}
                  />
                </div>
              )}

              <Checkbox
                checked={forceRerun}
                onChange={(event) => setForceRerun(event.target.checked)}
              >
                {t("evaluations:recipeForceRerunLabel", {
                  defaultValue: "Force rerun even if a matching completed run exists"
                })}
              </Checkbox>

              {!canEnqueueRuns && forceRerun && (
                <Alert
                  type="info"
                  showIcon
                  title={t("evaluations:recipeForceRerunUnavailable", {
                    defaultValue: "Force rerun requires the recipe worker to be available."
                  })}
                />
              )}

              {localError && (
                <Alert
                  type="error"
                  showIcon
                  title={localError}
                />
              )}

              {validationResult && (
                <Alert
                  type={validationResult.valid ? "success" : "warning"}
                  showIcon
                  title={
                    validationResult.valid
                      ? t("evaluations:recipeValidationValid", {
                          defaultValue: "Dataset format is valid."
                        })
                      : t("evaluations:recipeValidationInvalid", {
                          defaultValue: "Dataset format needs attention."
                        })
                  }
                  description={
                    <div className="space-y-1 text-xs">
                      {validationResult.dataset_mode && (
                        <div>
                          {t("evaluations:recipeDatasetModeLabel", {
                            defaultValue: "Dataset mode"
                          })}
                          : {validationResult.dataset_mode}
                        </div>
                      )}
                      {typeof validationResult.sample_count === "number" && (
                        <div>
                          {t("evaluations:recipeSampleCountLabel", {
                            defaultValue: "Samples"
                          })}
                          : {validationResult.sample_count}
                        </div>
                      )}
                      {!readinessLoading && (
                        <div>
                          {t("evaluations:recipeLaunchReadinessLabel", {
                            defaultValue: "Launch readiness"
                          })}
                          :{" "}
                          {canEnqueueRuns
                            ? t("evaluations:recipeLaunchReadyNow", {
                                defaultValue: "ready to start new runs."
                              })
                            : t("evaluations:recipeLaunchReuseOnly", {
                                defaultValue:
                                  "matching-run reuse only until the worker is enabled."
                              })}
                        </div>
                      )}
                      {(validationResult.errors || []).map((error, index) => (
                        <div key={`${error}-${index}`}>{error}</div>
                      ))}
                    </div>
                  }
                />
              )}

              <Space>
                <Button
                  onClick={handleValidate}
                  loading={validateMutation.isPending}
                  disabled={!isLaunchable}
                >
                  {t("evaluations:recipeValidateCta", {
                    defaultValue: "Validate dataset"
                  })}
                </Button>
                <Button
                  type="primary"
                  onClick={handleRun}
                  loading={createRunMutation.isPending}
                  disabled={runButtonDisabled}
                >
                  {runButtonLabel}
                </Button>
                {(selectedManifest.recipe_id === "rag_retrieval_tuning" ||
                  selectedManifest.recipe_id === "rag_answer_quality") && (
                  <Button onClick={openSyntheticReview}>
                    {syntheticReviewButtonLabel}
                  </Button>
                )}
              </Space>
            </div>
          </Card>
        ) : (
          <Card>
            <Empty
              description={t("evaluations:recipeSelectEmpty", {
                defaultValue: "Choose a recipe to begin."
              })}
            />
          </Card>
        )}

        {(currentRunId || report) && (
          <Card
            title={t("evaluations:recipeCurrentRunTitle", {
              defaultValue: "Current run"
            })}
            extra={
              report?.run?.status ? (
                <Tag>{String(report.run.status)}</Tag>
              ) : currentRunId ? (
                <Tag>{currentRunId}</Tag>
              ) : null
            }
          >
            {reportLoading ? (
              <div className="flex justify-center py-4">
                <Spin />
              </div>
            ) : reportError ? (
              <Alert
                type="error"
                showIcon
                title={t("evaluations:recipeReportLoadErrorTitle", {
                  defaultValue: "Unable to load the recipe report"
                })}
                description={(reportErrorValue as Error | null)?.message || undefined}
              />
            ) : report ? (
              <div className="space-y-4">
                {report.confidence_summary && (
                  <Alert
                    type="info"
                    showIcon
                    title={t("evaluations:recipeConfidenceTitle", {
                      defaultValue: "Confidence summary"
                    })}
                    description={
                      <span className="text-xs">
                        {t("evaluations:recipeConfidenceDescription", {
                          defaultValue: "Confidence {{confidence}} across {{count}} samples.",
                          confidence: report.confidence_summary.confidence,
                          count: report.confidence_summary.sample_count
                        })}
                      </span>
                    }
                  />
                )}

                {isEmbeddingsRecipeReport ? (
                  renderEmbeddingsRecommendationCards()
                ) : (
                  <div className="grid gap-3 md:grid-cols-3">
                    {Object.entries(report.recommendation_slots || {}).map(([slotName, slot]) => (
                      <Card key={slotName} size="small" styles={{ body: { padding: 12 } }}>
                        <div className="space-y-1">
                          <Text strong>{prettifySlotName(slotName)}</Text>
                          <div className="text-xs text-text-muted">
                            {slot.candidate_run_id || t("evaluations:recipeNoWinner", {
                              defaultValue: "No winner yet"
                            })}
                          </div>
                          {slot.explanation && (
                            <Paragraph className="mb-0 text-xs">
                              {slot.explanation}
                            </Paragraph>
                          )}
                        </div>
                      </Card>
                    ))}
                  </div>
                )}

                {reportCandidates.length > 0 && (
                  <div className="space-y-2">
                    <Title level={5} className="mb-0">
                      {t("evaluations:recipeCandidatesTitle", {
                        defaultValue: "Candidates"
                      })}
                    </Title>
                    <div className="space-y-2">
                      {reportCandidates.map((candidate: any) => (
                        <Card
                          key={candidate.candidate_id || candidate.model}
                          size="small"
                          styles={{ body: { padding: 12 } }}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div>
                              <div className="font-medium">
                                {candidate.model || candidate.candidate_id}
                              </div>
                              <div className="text-xs text-text-muted">
                                {candidate.provider || ""}
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-2 text-xs">
                              {Object.entries(candidate.metrics || {}).map(([key, value]) => (
                                <Tag key={key}>
                                  {key}: {String(value)}
                                </Tag>
                              ))}
                              {Object.entries(candidate.failure_label_counts || {}).map(
                                ([label, count]) => (
                                  <Tag key={`failure-${label}`}>
                                    {label}: {String(count)}
                                  </Tag>
                                )
                              )}
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}

                {reportFailureExamples.length > 0 && (
                  <div className="space-y-2">
                    <Title level={5} className="mb-0">
                      {t("evaluations:recipeFailureExamplesTitle", {
                        defaultValue: "Failure examples"
                      })}
                    </Title>
                    <div className="space-y-2">
                      {reportFailureExamples.map((example: any, index: number) => (
                        <Card
                          key={`${example.candidate_id || "candidate"}-${example.sample_id || index}`}
                          size="small"
                          styles={{ body: { padding: 12 } }}
                        >
                          <div className="space-y-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <Text strong>{example.sample_id || `Example ${index + 1}`}</Text>
                              {example.candidate_id ? <Tag>{example.candidate_id}</Tag> : null}
                              {Array.isArray(example.failure_labels)
                                ? example.failure_labels.map((label: string) => (
                                    <Tag key={`${example.sample_id || index}-${label}`}>
                                      {label}
                                    </Tag>
                                  ))
                                : null}
                            </div>
                            {example.query ? (
                              <div>
                                <Text strong>
                                  {t("evaluations:recipeFailureQueryLabel", {
                                    defaultValue: "Query"
                                  })}
                                </Text>
                                <Paragraph className="mb-0 mt-1">{example.query}</Paragraph>
                              </div>
                            ) : null}
                            {example.expected_behavior ? (
                              <div className="text-xs text-text-muted">
                                {t("evaluations:recipeFailureExpectedBehaviorLabel", {
                                  defaultValue: "Expected behavior"
                                })}
                                : {String(example.expected_behavior)}
                              </div>
                            ) : null}
                            {example.answer ? (
                              <div>
                                <Text strong>
                                  {t("evaluations:recipeFailureAnswerLabel", {
                                    defaultValue: "Candidate answer"
                                  })}
                                </Text>
                                <Paragraph className="mb-0 mt-1">{example.answer}</Paragraph>
                              </div>
                            ) : null}
                            {example.reference_answer ? (
                              <div>
                                <Text strong>
                                  {t("evaluations:recipeFailureReferenceLabel", {
                                    defaultValue: "Reference answer"
                                  })}
                                </Text>
                                <Paragraph className="mb-0 mt-1">
                                  {example.reference_answer}
                                </Paragraph>
                              </div>
                            ) : null}
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <Text type="secondary" className="text-xs">
                {t("evaluations:recipeRunPending", {
                  defaultValue: "Waiting for the recipe report."
                })}
              </Text>
            )}
          </Card>
        )}
      </div>
    </div>
  )
}

export default RecipesTab
