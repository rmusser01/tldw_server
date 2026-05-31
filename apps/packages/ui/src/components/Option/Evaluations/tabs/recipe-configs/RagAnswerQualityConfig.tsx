import React from "react"
import { Button, Card, Input, Typography } from "antd"
import type { DatasetSample } from "@/services/evaluations"
import { Alert as DsAlert } from "@/components/ui/primitives"
import { useTranslation } from "react-i18next"
import { useGenerateSyntheticEvalDrafts } from "../../hooks/useSyntheticEval"

const { Text } = Typography
const { TextArea } = Input

type ExpectedBehavior = "answer" | "hedge" | "abstain"
type EvaluationMode = "fixed_context" | "live_end_to_end"
type SupervisionMode = "rubric" | "reference_answer" | "pairwise" | "mixed"

type SyntheticAnswerExampleRow = {
  query: string
  expected_behavior: ExpectedBehavior
  reference_answer: string
  retrieved_contexts: string
}

type SyntheticGenerationSummary = {
  batchId: string
  sampleCount: number
}

type Props = {
  datasetSource: "inline" | "saved"
  dataset: DatasetSample[]
  runConfig: Record<string, any>
  onDatasetChange: (next: DatasetSample[]) => void
  onRunConfigChange: (next: Record<string, any>) => void
}

const DEFAULT_CANDIDATE = {
  candidate_id: "candidate-1",
  generation_model: "openai:gpt-4.1-mini",
  prompt_variant: "default",
  formatting_citation_mode: "citations_required"
}

const DEFAULT_SYNTHETIC_EXAMPLE_ROW: SyntheticAnswerExampleRow = {
  query: "",
  expected_behavior: "answer",
  reference_answer: "",
  retrieved_contexts: ""
}

const DEFAULT_TARGET_SAMPLE_COUNT = "12"

const splitLines = (value: string): string[] =>
  value
    .split(/\r?\n/)
    .map((entry) => entry.trim())
    .filter(Boolean)

const normalizeRetrievedContexts = (
  value: unknown
): Array<{ content: string }> => {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => {
      if (typeof item === "string") return { content: item.trim() }
      if (item && typeof item === "object" && typeof (item as any).content === "string") {
        return { content: String((item as any).content).trim() }
      }
      return null
    })
    .filter((item): item is { content: string } => Boolean(item?.content))
}

const formatRetrievedContexts = (value: unknown): string =>
  normalizeRetrievedContexts(value)
    .map((item) => item.content)
    .join("\n")

const parseRetrievedContexts = (value: string): Array<{ content: string }> =>
  splitLines(value).map((content) => ({ content }))

const parseInteger = (value: unknown, fallback: number): number => {
  const next = Number.parseInt(String(value), 10)
  return Number.isFinite(next) ? next : fallback
}

const normalizeDataset = (
  dataset: DatasetSample[],
  evaluationMode: EvaluationMode
): DatasetSample[] => {
  const baseDataset = Array.isArray(dataset) ? dataset : []
  return baseDataset.map((sample, index) => {
    const record = sample && typeof sample === "object" ? (sample as Record<string, any>) : {}
    const normalized: Record<string, any> = {
      sample_id: String(record.sample_id ?? `sample-${index + 1}`).trim() || `sample-${index + 1}`,
      query: typeof record.query === "string" ? record.query : "",
      expected_behavior:
        record.expected_behavior === "hedge" || record.expected_behavior === "abstain"
          ? record.expected_behavior
          : "answer"
    }

    if (evaluationMode === "fixed_context") {
      const retrievedContexts = normalizeRetrievedContexts(record.retrieved_contexts)
      if (retrievedContexts.length > 0) {
        normalized.retrieved_contexts = retrievedContexts
      }
    }

    if (typeof record.reference_answer === "string" && record.reference_answer.trim()) {
      normalized.reference_answer = record.reference_answer
    }

    return normalized as DatasetSample
  })
}

const normalizeRunConfig = (runConfig: Record<string, any>) => {
  const evaluationMode: EvaluationMode =
    String(runConfig.evaluation_mode) === "live_end_to_end"
      ? "live_end_to_end"
      : "fixed_context"
  const supervisionMode: SupervisionMode =
    String(runConfig.supervision_mode) === "reference_answer" ||
    String(runConfig.supervision_mode) === "pairwise" ||
    String(runConfig.supervision_mode) === "mixed"
      ? (String(runConfig.supervision_mode) as SupervisionMode)
      : "rubric"
  const candidates =
    Array.isArray(runConfig.candidates) && runConfig.candidates.length > 0
      ? runConfig.candidates.map((candidate: Record<string, any>, index: number) => ({
          candidate_id:
            String(candidate?.candidate_id ?? `candidate-${index + 1}`).trim() ||
            `candidate-${index + 1}`,
          generation_model: String(candidate?.generation_model ?? "").trim(),
          prompt_variant: String(candidate?.prompt_variant ?? "").trim(),
          formatting_citation_mode: String(candidate?.formatting_citation_mode ?? "").trim()
        }))
      : [{ ...DEFAULT_CANDIDATE }]

  const normalized = {
    ...runConfig,
    evaluation_mode: evaluationMode,
    supervision_mode: supervisionMode,
    candidates
  } as Record<string, any>

  if (evaluationMode === "fixed_context") {
    normalized.context_snapshot_ref = String(runConfig.context_snapshot_ref ?? "").trim()
    delete normalized.retrieval_baseline_ref
  } else {
    normalized.retrieval_baseline_ref = String(runConfig.retrieval_baseline_ref ?? "").trim()
    delete normalized.context_snapshot_ref
  }

  return normalized
}

const serializeSyntheticAnswerExample = (
  row: SyntheticAnswerExampleRow,
  evaluationMode: EvaluationMode
): Record<string, any> | null => {
  const query = row.query.trim()
  const referenceAnswer = row.reference_answer.trim()
  if (!query || !referenceAnswer) return null

  const samplePayload: Record<string, any> = {
    query,
    expected_behavior: row.expected_behavior,
    reference_answer: referenceAnswer
  }

  if (evaluationMode === "fixed_context") {
    const retrievedContexts = parseRetrievedContexts(row.retrieved_contexts)
    if (retrievedContexts.length > 0) {
      samplePayload.retrieved_contexts = retrievedContexts
    }
  }

  return {
    expected_behavior: row.expected_behavior,
    reference_answer: referenceAnswer,
    sample_payload: samplePayload
  }
}

const normalizeAnswerQualityExamples = (
  dataset: DatasetSample[],
  evaluationMode: EvaluationMode
): Record<string, any>[] =>
  dataset
    .map((sample, index): Record<string, any> | null => {
      const record = sample && typeof sample === "object" ? (sample as Record<string, any>) : {}
      const query = String(record.query ?? "").trim()
      const referenceAnswer = String(record.reference_answer ?? "").trim()
      if (!query || !referenceAnswer) return null

      const samplePayload: Record<string, any> = {
        query,
        expected_behavior:
          record.expected_behavior === "hedge" || record.expected_behavior === "abstain"
            ? record.expected_behavior
            : "answer",
        reference_answer: referenceAnswer
      }

      if (evaluationMode === "fixed_context") {
        const retrievedContexts = normalizeRetrievedContexts(record.retrieved_contexts)
        if (retrievedContexts.length > 0) {
          samplePayload.retrieved_contexts = retrievedContexts
        }
      }

      return {
        sample_id:
          String(record.sample_id ?? `sample-${index + 1}`).trim() || `sample-${index + 1}`,
        expected_behavior: samplePayload.expected_behavior,
        reference_answer: referenceAnswer,
        sample_payload: samplePayload
      }
    })
    .filter((item): item is Record<string, any> => item !== null)

export const RagAnswerQualityConfig: React.FC<Props> = ({
  datasetSource,
  dataset,
  runConfig,
  onDatasetChange,
  onRunConfigChange
}) => {
  const { t } = useTranslation(["evaluations", "common"])
  const normalizedRunConfig = React.useMemo(() => normalizeRunConfig(runConfig), [runConfig])
  const normalizedDataset = React.useMemo(
    () => normalizeDataset(dataset, normalizedRunConfig.evaluation_mode),
    [dataset, normalizedRunConfig.evaluation_mode]
  )
  const editableDataset = React.useMemo<DatasetSample[]>(
    () =>
      normalizedDataset.length > 0
        ? normalizedDataset
        : ([{ sample_id: "sample-1", query: "", expected_behavior: "answer" }] as DatasetSample[]),
    [normalizedDataset]
  )
  const generateSyntheticDraftsMutation = useGenerateSyntheticEvalDrafts()
  const hasValidGenerationAnchor =
    normalizedRunConfig.evaluation_mode === "fixed_context"
      ? String(normalizedRunConfig.context_snapshot_ref || "").trim().length > 0
      : String(normalizedRunConfig.retrieval_baseline_ref || "").trim().length > 0
  const [generationTargetCount, setGenerationTargetCount] =
    React.useState(DEFAULT_TARGET_SAMPLE_COUNT)
  const [realExamples, setRealExamples] = React.useState<SyntheticAnswerExampleRow[]>([
    { ...DEFAULT_SYNTHETIC_EXAMPLE_ROW }
  ])
  const [seedExamples, setSeedExamples] = React.useState<SyntheticAnswerExampleRow[]>([
    { ...DEFAULT_SYNTHETIC_EXAMPLE_ROW }
  ])
  const [generationSummary, setGenerationSummary] =
    React.useState<SyntheticGenerationSummary | null>(null)
  const [generationError, setGenerationError] = React.useState<string | null>(null)

  const applyRunConfig = (updater: (current: Record<string, any>) => Record<string, any>) => {
    onRunConfigChange(normalizeRunConfig(updater(normalizeRunConfig(runConfig))))
  }

  const applyDataset = (updater: (current: DatasetSample[]) => DatasetSample[]) => {
    onDatasetChange(
      updater(datasetSource === "inline" ? editableDataset : normalizedDataset)
    )
  }

  const updateDatasetSample = (
    sampleIndex: number,
    updater: (current: Record<string, any>) => Record<string, any>
  ) => {
    applyDataset((current) =>
      current.map((sample, index) =>
        index === sampleIndex ? (updater({ ...(sample as Record<string, any>) }) as DatasetSample) : sample
      )
    )
  }

  const addRealExample = () => {
    setRealExamples((current) => [...current, { ...DEFAULT_SYNTHETIC_EXAMPLE_ROW }])
  }

  const updateRealExample = (
    index: number,
    updater: (current: SyntheticAnswerExampleRow) => SyntheticAnswerExampleRow
  ) => {
    setRealExamples((current) =>
      current.map((row, rowIndex) => (rowIndex === index ? updater({ ...row }) : row))
    )
  }

  const removeRealExample = (index: number) => {
    setRealExamples((current) =>
      current.length <= 1 ? current : current.filter((_, rowIndex) => rowIndex !== index)
    )
  }

  const addSeedExample = () => {
    setSeedExamples((current) => [...current, { ...DEFAULT_SYNTHETIC_EXAMPLE_ROW }])
  }

  const updateSeedExample = (
    index: number,
    updater: (current: SyntheticAnswerExampleRow) => SyntheticAnswerExampleRow
  ) => {
    setSeedExamples((current) =>
      current.map((row, rowIndex) => (rowIndex === index ? updater({ ...row }) : row))
    )
  }

  const removeSeedExample = (index: number) => {
    setSeedExamples((current) =>
      current.length <= 1 ? current : current.filter((_, rowIndex) => rowIndex !== index)
    )
  }

  const handleGenerateSyntheticDrafts = async () => {
    if (!hasValidGenerationAnchor) return
    setGenerationError(null)

    try {
      const response = await generateSyntheticDraftsMutation.mutateAsync({
        recipe_kind: "rag_answer_quality",
        ...(normalizedRunConfig.evaluation_mode === "fixed_context"
          ? {
              context_snapshot_ref: String(normalizedRunConfig.context_snapshot_ref || "").trim()
            }
          : {
              retrieval_baseline_ref: String(normalizedRunConfig.retrieval_baseline_ref || "").trim()
            }),
        target_sample_count: Math.max(parseInteger(generationTargetCount, 12), 1),
        real_examples:
          datasetSource === "saved"
            ? normalizeAnswerQualityExamples(
                normalizedDataset,
                normalizedRunConfig.evaluation_mode
              )
            : realExamples
                .map((row) =>
                  serializeSyntheticAnswerExample(
                    row,
                    normalizedRunConfig.evaluation_mode
                  )
                )
                .filter((item): item is Record<string, any> => item !== null),
        seed_examples: seedExamples
          .map((row) =>
            serializeSyntheticAnswerExample(row, normalizedRunConfig.evaluation_mode)
          )
          .filter((item): item is Record<string, any> => item !== null)
      })

      setGenerationSummary({
        batchId: String(response?.data?.generation_batch_id || ""),
        sampleCount: Array.isArray(response?.data?.samples) ? response.data.samples.length : 0
      })
    } catch (error: any) {
      setGenerationSummary(null)
      setGenerationError(
        error?.message ||
          t("evaluations:syntheticGenerationInlineErrorDescription", {
            defaultValue: "Unable to generate synthetic drafts right now."
          })
      )
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <Text strong>Evaluation mode</Text>
          <select
            aria-label="Evaluation mode"
            className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={normalizedRunConfig.evaluation_mode}
            onChange={(event) =>
              applyRunConfig((current) => ({
                ...current,
                evaluation_mode:
                  event.target.value === "live_end_to_end"
                    ? "live_end_to_end"
                    : "fixed_context"
              }))
            }
          >
            <option value="fixed_context">fixed_context</option>
            <option value="live_end_to_end">live_end_to_end</option>
          </select>
        </div>
        <div>
          <Text strong>Supervision mode</Text>
          <select
            aria-label="Supervision mode"
            className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={normalizedRunConfig.supervision_mode}
            onChange={(event) =>
              applyRunConfig((current) => ({
                ...current,
                supervision_mode: event.target.value
              }))
            }
          >
            <option value="rubric">rubric</option>
            <option value="reference_answer">reference_answer</option>
            <option value="pairwise">pairwise</option>
            <option value="mixed">mixed</option>
          </select>
        </div>
      </div>

      <div>
        <Text strong>
          {normalizedRunConfig.evaluation_mode === "fixed_context"
            ? "Context snapshot reference"
            : "Retrieval baseline reference"}
        </Text>
        <Input
          aria-label={
            normalizedRunConfig.evaluation_mode === "fixed_context"
              ? "Context snapshot reference"
              : "Retrieval baseline reference"
          }
          className="mt-2"
          value={
            normalizedRunConfig.evaluation_mode === "fixed_context"
              ? String(normalizedRunConfig.context_snapshot_ref || "")
              : String(normalizedRunConfig.retrieval_baseline_ref || "")
          }
          onChange={(event) =>
            applyRunConfig((current) =>
              normalizedRunConfig.evaluation_mode === "fixed_context"
                ? {
                    ...current,
                    context_snapshot_ref: event.target.value
                  }
                : {
                    ...current,
                    retrieval_baseline_ref: event.target.value
                  }
            )
          }
        />
      </div>

      <Card
        size="small"
        title={t("evaluations:syntheticAnswerGenerationTitle", {
          defaultValue: "Synthetic generation"
        })}
      >
        <div className="space-y-4">
          {generationSummary && (
            <DsAlert
              variant="success"
              title={t("evaluations:syntheticGenerationSuccessInlineTitle", {
                defaultValue: "Synthetic drafts created"
              })}
            >
              {t("evaluations:syntheticGenerationSuccessInlineDescription", {
                defaultValue:
                  "Batch {{batchId}} created {{count}} drafts and is ready for review.",
                batchId: generationSummary.batchId,
                count: generationSummary.sampleCount
              })}
            </DsAlert>
          )}
          {generationError && (
            <DsAlert
              variant="error"
              title={t("evaluations:syntheticGenerationErrorTitle", {
                defaultValue: "Failed to generate synthetic drafts"
              })}
            >
              {generationError}
            </DsAlert>
          )}

          <div className="space-y-2">
            <Text strong>
              {t("evaluations:syntheticAnswerTotalCountLabel", {
                defaultValue: "Total draft set size"
              })}
            </Text>
            <Input
              aria-label={t("evaluations:syntheticAnswerTotalCountLabel", {
                defaultValue: "Total draft set size"
              })}
              className="mt-2"
              inputMode="numeric"
              value={generationTargetCount}
              onChange={(event) => setGenerationTargetCount(event.target.value)}
            />
            <div className="text-xs text-text-muted">
              {t("evaluations:syntheticAnswerTotalCountHelp", {
                defaultValue:
                  "This is the total draft set size, including preferred real examples and synthetic fill."
              })}
            </div>
          </div>

          <DsAlert
            variant="info"
            title={t("evaluations:syntheticAnswerAnchorSummaryLabel", {
              defaultValue: "Current answer-quality anchor"
            })}
          >
            {
              normalizedRunConfig.evaluation_mode === "fixed_context"
                ? t("evaluations:syntheticAnswerAnchorSummaryDescriptionFixed", {
                    defaultValue: "Fixed-context mode using {{anchor}}",
                    anchor:
                      String(normalizedRunConfig.context_snapshot_ref || "").trim() ||
                      "no context snapshot selected"
                  })
                : t("evaluations:syntheticAnswerAnchorSummaryDescriptionLive", {
                    defaultValue: "Live end-to-end mode using {{anchor}}",
                    anchor:
                      String(normalizedRunConfig.retrieval_baseline_ref || "").trim() ||
                      "no retrieval baseline selected"
                  })
            }
          </DsAlert>

          <div className="space-y-2">
            <Text strong>
              {t("evaluations:syntheticAnswerRealExamplesLabel", {
                defaultValue: "Current dataset samples used as real examples"
              })}
            </Text>
            {datasetSource === "saved" ? (
              <div className="space-y-2">
                {normalizedDataset.length > 0 ? (
                  normalizedDataset.map((sample, index) => {
                    const record = sample as Record<string, any>
                    return (
                      <Card
                        key={`answer-real-example-${index}`}
                        size="small"
                        title={t("evaluations:syntheticAnswerRealExampleTitle", {
                          defaultValue: "Real example {{index}}",
                          index: index + 1
                        })}
                      >
                        <div className="grid gap-2 md:grid-cols-2">
                          <div className="md:col-span-2">
                            <Text strong>
                              {t("evaluations:syntheticAnswerQueryLabel", {
                                defaultValue: "Query"
                              })}
                            </Text>
                            <div className="text-sm text-text-muted">
                              {String(record.query || "") || "No query text"}
                            </div>
                          </div>
                          <div>
                            <Text strong>
                              {t("evaluations:syntheticAnswerExpectedBehaviorLabel", {
                                defaultValue: "Expected behavior"
                              })}
                            </Text>
                            <div className="text-sm text-text-muted">
                              {String(record.expected_behavior || "answer")}
                            </div>
                          </div>
                          <div>
                            <Text strong>
                              {t("evaluations:syntheticAnswerReferenceAnswerLabel", {
                                defaultValue: "Reference answer"
                              })}
                            </Text>
                            <div className="text-sm text-text-muted">
                              {String(record.reference_answer || "") || "None"}
                            </div>
                          </div>
                          {normalizedRunConfig.evaluation_mode === "fixed_context" && (
                            <div className="md:col-span-2">
                              <Text strong>
                                {t("evaluations:syntheticAnswerRetrievedContextsLabel", {
                                  defaultValue: "Retrieved contexts"
                                })}
                              </Text>
                              <div className="text-sm text-text-muted whitespace-pre-wrap">
                                {formatRetrievedContexts(record.retrieved_contexts) || "None"}
                              </div>
                            </div>
                          )}
                        </div>
                      </Card>
                    )
                  })
                ) : (
                  <div className="text-sm text-text-muted">
                    {t("evaluations:syntheticAnswerNoRealExamples", {
                      defaultValue: "No dataset samples are available yet."
                    })}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <Text strong>
                    {t("evaluations:syntheticAnswerInlineRealExamplesLabel", {
                      defaultValue: "Real examples"
                    })}
                  </Text>
                  <Button size="small" onClick={addRealExample}>
                    {t("evaluations:syntheticAnswerAddRealExampleCta", {
                      defaultValue: "Add real example"
                    })}
                  </Button>
                </div>
                {realExamples.map((row, index) => (
                  <Card
                    key={`answer-inline-real-example-${index}`}
                    size="small"
                    title={t("evaluations:syntheticAnswerRealExampleTitle", {
                      defaultValue: "Real example {{index}}",
                      index: index + 1
                    })}
                    extra={
                      <Button
                        size="small"
                        onClick={() => removeRealExample(index)}
                        disabled={realExamples.length <= 1}
                      >
                        {t("evaluations:syntheticAnswerRemoveExampleCta", {
                          defaultValue: "Remove"
                        })}
                      </Button>
                    }
                  >
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="md:col-span-2">
                        <Text strong>
                          {t("evaluations:syntheticAnswerQueryLabel", {
                            defaultValue: "Query"
                          })}{" "}
                          {index + 1}
                        </Text>
                        <Input
                          aria-label={`Real example ${index + 1} query`}
                          className="mt-2"
                          value={row.query}
                          onChange={(event) =>
                            updateRealExample(index, (current) => ({
                              ...current,
                              query: event.target.value
                            }))
                          }
                        />
                      </div>
                      <div>
                        <Text strong>
                          {t("evaluations:syntheticAnswerExpectedBehaviorLabel", {
                            defaultValue: "Expected behavior"
                          })}{" "}
                          {index + 1}
                        </Text>
                        <select
                          aria-label={`Real example ${index + 1} expected behavior`}
                          className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                          value={row.expected_behavior}
                          onChange={(event) =>
                            updateRealExample(index, (current) => ({
                              ...current,
                              expected_behavior: event.target.value as ExpectedBehavior
                            }))
                          }
                        >
                          <option value="answer">answer</option>
                          <option value="hedge">hedge</option>
                          <option value="abstain">abstain</option>
                        </select>
                      </div>
                      <div>
                        <Text strong>
                          {t("evaluations:syntheticAnswerReferenceAnswerLabel", {
                            defaultValue: "Reference answer"
                          })}{" "}
                          {index + 1}
                        </Text>
                        <Input
                          aria-label={`Real example ${index + 1} reference answer`}
                          className="mt-2"
                          value={row.reference_answer}
                          onChange={(event) =>
                            updateRealExample(index, (current) => ({
                              ...current,
                              reference_answer: event.target.value
                            }))
                          }
                        />
                      </div>
                      {normalizedRunConfig.evaluation_mode === "fixed_context" && (
                        <div className="md:col-span-2">
                          <Text strong>
                            {t("evaluations:syntheticAnswerRetrievedContextsLabel", {
                              defaultValue: "Retrieved contexts"
                            })}{" "}
                            {index + 1}
                          </Text>
                          <TextArea
                            aria-label={`Real example ${index + 1} retrieved contexts`}
                            rows={3}
                            className="mt-2"
                            value={row.retrieved_contexts}
                            onChange={(event) =>
                              updateRealExample(index, (current) => ({
                                ...current,
                                retrieved_contexts: event.target.value
                              }))
                            }
                          />
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <Text strong>
                {t("evaluations:syntheticAnswerSeedExamplesLabel", {
                  defaultValue: "Seed examples"
                })}
              </Text>
              <Button size="small" onClick={addSeedExample}>
                {t("evaluations:syntheticAnswerAddSeedExampleCta", {
                  defaultValue: "Add seed example"
                })}
              </Button>
            </div>
            {seedExamples.map((row, index) => (
              <Card
                key={`answer-seed-example-${index}`}
                size="small"
                title={t("evaluations:syntheticAnswerSeedExampleTitle", {
                  defaultValue: "Seed example {{index}}",
                  index: index + 1
                })}
                extra={
                  <Button
                    size="small"
                    onClick={() => removeSeedExample(index)}
                    disabled={seedExamples.length <= 1}
                  >
                    {t("evaluations:syntheticAnswerRemoveExampleCta", {
                      defaultValue: "Remove"
                    })}
                  </Button>
                }
              >
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="md:col-span-2">
                    <Text strong>
                      {t("evaluations:syntheticAnswerQueryLabel", {
                        defaultValue: "Query"
                      })}{" "}
                      {index + 1}
                    </Text>
                    <Input
                      aria-label={`Seed example ${index + 1} query`}
                      className="mt-2"
                      value={row.query}
                      onChange={(event) =>
                        updateSeedExample(index, (current) => ({
                          ...current,
                          query: event.target.value
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Text strong>
                      {t("evaluations:syntheticAnswerExpectedBehaviorLabel", {
                        defaultValue: "Expected behavior"
                      })}{" "}
                      {index + 1}
                    </Text>
                    <select
                      aria-label={`Seed example ${index + 1} expected behavior`}
                      className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                      value={row.expected_behavior}
                      onChange={(event) =>
                        updateSeedExample(index, (current) => ({
                          ...current,
                          expected_behavior: event.target.value as ExpectedBehavior
                        }))
                      }
                    >
                      <option value="answer">answer</option>
                      <option value="hedge">hedge</option>
                      <option value="abstain">abstain</option>
                    </select>
                  </div>
                  <div>
                    <Text strong>
                      {t("evaluations:syntheticAnswerReferenceAnswerLabel", {
                        defaultValue: "Reference answer"
                      })}{" "}
                      {index + 1}
                    </Text>
                    <Input
                      aria-label={`Seed example ${index + 1} reference answer`}
                      className="mt-2"
                      value={row.reference_answer}
                      onChange={(event) =>
                        updateSeedExample(index, (current) => ({
                          ...current,
                          reference_answer: event.target.value
                        }))
                      }
                    />
                  </div>
                  {normalizedRunConfig.evaluation_mode === "fixed_context" && (
                    <div className="md:col-span-2">
                      <Text strong>
                        {t("evaluations:syntheticAnswerRetrievedContextsLabel", {
                          defaultValue: "Retrieved contexts"
                        })}{" "}
                        {index + 1}
                      </Text>
                      <TextArea
                        aria-label={`Seed example ${index + 1} retrieved contexts`}
                        rows={3}
                        className="mt-2"
                        value={row.retrieved_contexts}
                        onChange={(event) =>
                          updateSeedExample(index, (current) => ({
                            ...current,
                            retrieved_contexts: event.target.value
                          }))
                        }
                      />
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>

          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-xs text-text-muted">
              {hasValidGenerationAnchor
                ? t("evaluations:syntheticAnswerReadyHint", {
                    defaultValue:
                      "Generation is ready once your preferred examples and anchor are set."
                  })
                : t("evaluations:syntheticAnswerReadyMissingAnchor", {
                    defaultValue:
                      "Add a context snapshot or retrieval baseline before generating drafts."
                  })}
            </div>
            <Button
              type="primary"
              onClick={() => void handleGenerateSyntheticDrafts()}
              disabled={!hasValidGenerationAnchor || generateSyntheticDraftsMutation.isPending}
              loading={generateSyntheticDraftsMutation.isPending}
            >
              {t("evaluations:syntheticAnswerGenerateCta", {
                defaultValue: "Generate synthetic drafts"
              })}
            </Button>
          </div>
        </div>
      </Card>

      {datasetSource === "inline" && (
        <div className="space-y-3">
          <Text strong>Dataset samples</Text>
          {editableDataset.map((sample, index) => (
            <Card
              key={`rag-answer-sample-${index}`}
              size="small"
              title={`Sample ${index + 1}`}
              extra={
                <Button
                  size="small"
                  onClick={() =>
                    applyDataset((current) =>
                      current.filter((_, datasetIndex) => datasetIndex !== index)
                    )
                  }
                  disabled={editableDataset.length <= 1}
                >
                  Remove
                </Button>
              }
            >
              <div className="grid gap-3 md:grid-cols-2">
                <div>
                  <Text strong>{`Sample ID ${index + 1}`}</Text>
                  <Input
                    aria-label={`Sample ID ${index + 1}`}
                    className="mt-2"
                    value={String((sample as Record<string, any>).sample_id || "")}
                    onChange={(event) =>
                      updateDatasetSample(index, (current) => ({
                        ...current,
                        sample_id: event.target.value
                      }))
                    }
                  />
                </div>
                <div>
                  <Text strong>{`Expected behavior ${index + 1}`}</Text>
                  <select
                    aria-label={`Expected behavior ${index + 1}`}
                    className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    value={String((sample as Record<string, any>).expected_behavior || "answer")}
                    onChange={(event) =>
                      updateDatasetSample(index, (current) => ({
                        ...current,
                        expected_behavior: event.target.value as ExpectedBehavior
                      }))
                    }
                  >
                    <option value="answer">answer</option>
                    <option value="hedge">hedge</option>
                    <option value="abstain">abstain</option>
                  </select>
                </div>
                <div className="md:col-span-2">
                  <Text strong>{`Query text ${index + 1}`}</Text>
                  <TextArea
                    aria-label={`Query text ${index + 1}`}
                    rows={3}
                    className="mt-2"
                    value={String((sample as Record<string, any>).query || "")}
                    onChange={(event) =>
                      updateDatasetSample(index, (current) => ({
                        ...current,
                        query: event.target.value
                      }))
                    }
                  />
                </div>
                {normalizedRunConfig.evaluation_mode === "fixed_context" && (
                  <div className="md:col-span-2">
                    <Text strong>{`Retrieved contexts ${index + 1}`}</Text>
                    <TextArea
                      aria-label={`Retrieved contexts ${index + 1}`}
                      rows={4}
                      className="mt-2"
                      value={formatRetrievedContexts((sample as Record<string, any>).retrieved_contexts)}
                      onChange={(event) =>
                        updateDatasetSample(index, (current) => {
                          const nextContexts = parseRetrievedContexts(event.target.value)
                          const nextSample = { ...current }
                          if (nextContexts.length > 0) {
                            nextSample.retrieved_contexts = nextContexts
                          } else {
                            delete nextSample.retrieved_contexts
                          }
                          return nextSample
                        })
                      }
                    />
                  </div>
                )}
                <div className="md:col-span-2">
                  <Text strong>{`Reference answer ${index + 1} (optional)`}</Text>
                  <TextArea
                    aria-label={`Reference answer ${index + 1} (optional)`}
                    rows={3}
                    className="mt-2"
                    value={String((sample as Record<string, any>).reference_answer || "")}
                    onChange={(event) =>
                      updateDatasetSample(index, (current) => {
                        const nextSample = { ...current }
                        if (event.target.value.trim()) {
                          nextSample.reference_answer = event.target.value
                        } else {
                          delete nextSample.reference_answer
                        }
                        return nextSample
                      })
                    }
                  />
                </div>
              </div>
            </Card>
          ))}
          <Button
            onClick={() =>
              applyDataset((current) => [
                ...current,
                {
                  sample_id: `sample-${current.length + 1}`,
                  query: "",
                  expected_behavior: "answer"
                } as DatasetSample
              ])
            }
          >
            Add sample
          </Button>
        </div>
      )}

      <div className="space-y-3">
        <Text strong>Candidates</Text>
        {normalizedRunConfig.candidates.map((candidate: Record<string, any>, index: number) => (
          <Card
            key={`rag-answer-candidate-${index}`}
            size="small"
            title={`Candidate ${index + 1}`}
            extra={
              <Button
                size="small"
                onClick={() =>
                  applyRunConfig((current) => ({
                    ...current,
                    candidates: (Array.isArray(current.candidates) ? current.candidates : []).filter(
                      (_: unknown, candidateIndex: number) => candidateIndex !== index
                    )
                  }))
                }
                disabled={normalizedRunConfig.candidates.length <= 1}
              >
                Remove
              </Button>
            }
          >
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <Text strong>{`Candidate ID ${index + 1}`}</Text>
                <Input
                  aria-label={`Candidate ID ${index + 1}`}
                  className="mt-2"
                  value={String(candidate.candidate_id || "")}
                  onChange={(event) =>
                    applyRunConfig((current) => ({
                      ...current,
                      candidates: normalizedRunConfig.candidates.map((item, candidateIndex) =>
                        candidateIndex === index
                          ? { ...item, candidate_id: event.target.value }
                          : item
                      )
                    }))
                  }
                />
              </div>
              <div>
                <Text strong>{`Generation model ${index + 1}`}</Text>
                <Input
                  aria-label={`Generation model ${index + 1}`}
                  className="mt-2"
                  value={String(candidate.generation_model || "")}
                  onChange={(event) =>
                    applyRunConfig((current) => ({
                      ...current,
                      candidates: normalizedRunConfig.candidates.map((item, candidateIndex) =>
                        candidateIndex === index
                          ? { ...item, generation_model: event.target.value }
                          : item
                      )
                    }))
                  }
                />
              </div>
              <div>
                <Text strong>{`Prompt variant ${index + 1}`}</Text>
                <Input
                  aria-label={`Prompt variant ${index + 1}`}
                  className="mt-2"
                  value={String(candidate.prompt_variant || "")}
                  onChange={(event) =>
                    applyRunConfig((current) => ({
                      ...current,
                      candidates: normalizedRunConfig.candidates.map((item, candidateIndex) =>
                        candidateIndex === index
                          ? { ...item, prompt_variant: event.target.value }
                          : item
                      )
                    }))
                  }
                />
              </div>
              <div>
                <Text strong>{`Formatting/citation mode ${index + 1}`}</Text>
                <select
                  aria-label={`Formatting/citation mode ${index + 1}`}
                  className="mt-2 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  value={String(candidate.formatting_citation_mode || "citations_required")}
                  onChange={(event) =>
                    applyRunConfig((current) => ({
                      ...current,
                      candidates: normalizedRunConfig.candidates.map((item, candidateIndex) =>
                        candidateIndex === index
                          ? { ...item, formatting_citation_mode: event.target.value }
                          : item
                      )
                    }))
                  }
                >
                  <option value="citations_required">citations_required</option>
                  <option value="markdown_citations">markdown_citations</option>
                  <option value="plain_text">plain_text</option>
                </select>
              </div>
            </div>
          </Card>
        ))}
        <Button
          onClick={() =>
            applyRunConfig((current) => ({
              ...current,
              candidates: [
                ...(Array.isArray(current.candidates) ? current.candidates : []),
                {
                  ...DEFAULT_CANDIDATE,
                  candidate_id: `candidate-${normalizedRunConfig.candidates.length + 1}`
                }
              ]
            }))
          }
        >
          Add candidate
        </Button>
      </div>
    </div>
  )
}
