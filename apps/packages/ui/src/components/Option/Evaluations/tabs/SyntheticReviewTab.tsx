/**
 * SyntheticReviewTab component
 * Shared review queue for synthetic evaluation drafts.
 */

import React from "react"
import {
  Button,
  Card,
  Checkbox,
  Empty,
  Input,
  Select,
  Space,
  Spin,
  Tag,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import { useEvaluationsStore } from "@/store/evaluations"
import { StatePanel } from "@/components/ui/state"
import { EvaluationRecoveryCallout } from "../components/EvaluationRecoveryCallout"
import type { SyntheticEvalDraftSample } from "@/services/evaluations"
import {
  usePromoteSyntheticEvalSamples,
  useReviewSyntheticEvalSample,
  useSyntheticEvalQueue
} from "../hooks/useSyntheticEval"

const { Paragraph, Text, Title } = Typography
const { TextArea } = Input

const REVIEW_STATE_OPTIONS = [
  { value: "all", label: "All states" },
  { value: "draft", label: "Draft" },
  { value: "in_review", label: "In review" },
  { value: "edited", label: "Edited" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Rejected" }
]

const RECIPE_KIND_OPTIONS = [
  { value: "all", label: "All recipes" },
  { value: "rag_retrieval_tuning", label: "RAG Retrieval Tuning" },
  { value: "rag_answer_quality", label: "RAG Answer Quality" }
]

const EMPTY_SYNTHETIC_QUEUE_ITEMS: SyntheticEvalDraftSample[] = []

const queryPreview = (sample: SyntheticEvalDraftSample): string =>
  String(
    sample.sample_payload?.query ||
      sample.sample_payload?.input ||
      sample.sample_payload?.prompt ||
      ""
  ).trim()

export const SyntheticReviewTab: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const storedRecipeKind = useEvaluationsStore((s) => s.syntheticReviewRecipeKind)
  const storedBatchId = useEvaluationsStore((s) => s.syntheticReviewBatchId)
  const storedSampleIds = useEvaluationsStore((s) => s.syntheticReviewSampleIds)
  const setStoredRecipeKind = useEvaluationsStore((s) => s.setSyntheticReviewRecipeKind)

  const [reviewState, setReviewState] = React.useState<string>("draft")
  const [selectedIds, setSelectedIds] = React.useState<string[]>([])
  const [datasetName, setDatasetName] = React.useState("approved synthetic review set")
  const [notesById, setNotesById] = React.useState<Record<string, string>>({})

  const { data, isLoading, isError, error } = useSyntheticEvalQueue({
    recipeKind: storedRecipeKind && storedRecipeKind !== "all" ? storedRecipeKind : null,
    reviewState: reviewState !== "all" ? reviewState : null,
    generationBatchId: storedBatchId || null,
    limit: 100
  })
  const reviewMutation = useReviewSyntheticEvalSample()
  const promoteMutation = usePromoteSyntheticEvalSamples()

  const rawQueueItems = Array.isArray(data?.data?.data)
    ? data.data.data
    : EMPTY_SYNTHETIC_QUEUE_ITEMS
  const queueItems = React.useMemo(
    () =>
      !storedBatchId && storedSampleIds.length > 0
        ? rawQueueItems.filter((sample) => storedSampleIds.includes(sample.sample_id))
        : rawQueueItems,
    [rawQueueItems, storedBatchId, storedSampleIds]
  )
  const totalQueueItems = typeof data?.data?.total === "number" ? data.data.total : queueItems.length

  React.useEffect(() => {
    setSelectedIds((current) => {
      const nextSelectedIds = current.filter((sampleId) =>
        queueItems.some((sample) => sample.sample_id === sampleId)
      )
      return nextSelectedIds.length === current.length ? current : nextSelectedIds
    })
  }, [queueItems])

  const handleReviewAction = async (
    sample: SyntheticEvalDraftSample,
    action: "approve" | "reject" | "edit_and_approve"
  ) => {
    await reviewMutation.mutateAsync({
      sampleId: sample.sample_id,
      action,
      notes: notesById[sample.sample_id] || undefined
    })
  }

  const toggleSelection = (sampleId: string, checked: boolean) => {
    setSelectedIds((current) =>
      checked
        ? Array.from(new Set([...current, sampleId]))
        : current.filter((id) => id !== sampleId)
    )
  }

  const handlePromote = async () => {
    if (selectedIds.length === 0) return
    await promoteMutation.mutateAsync({
      sample_ids: selectedIds,
      dataset_name: datasetName.trim() || "approved synthetic review set"
    })
    setSelectedIds([])
  }

  return (
    <div className="space-y-4">
      <Card
        title={t("evaluations:syntheticReviewTitle", {
          defaultValue: "Synthetic review queue"
        })}
      >
        <Paragraph className="text-sm text-text-muted">
          {t("evaluations:syntheticReviewDescription", {
            defaultValue:
              "Review generated retrieval and answer-quality samples before they are promoted into active eval datasets."
          })}
        </Paragraph>
        <Paragraph className="mt-2 text-xs text-text-muted">
          {t("evaluations:syntheticReviewQueueSummary", {
            defaultValue:
              "{{shown}} of {{total}} samples shown. {{selected}} selected for promotion.",
            shown: queueItems.length,
            total: totalQueueItems,
            selected: selectedIds.length
          })}
        </Paragraph>
        {storedBatchId ? (
          <Paragraph className="mt-2 text-xs text-text-muted">
            {t("evaluations:syntheticReviewBatchSummary", {
              defaultValue: "Showing generated draft batch {{batchId}}.",
              batchId: storedBatchId
            })}
          </Paragraph>
        ) : null}

        <div className="grid gap-3 md:grid-cols-[minmax(0,220px)_minmax(0,220px)_minmax(0,1fr)_auto]">
          <div>
            <Text strong>
              {t("evaluations:syntheticRecipeFilterLabel", {
                defaultValue: "Recipe"
              })}
            </Text>
            <Select
              className="mt-2 w-full"
              aria-label="Synthetic review recipe filter"
              value={storedRecipeKind || "all"}
              onChange={(value) => setStoredRecipeKind(value === "all" ? null : value)}
              options={RECIPE_KIND_OPTIONS}
            />
          </div>
          <div>
            <Text strong>
              {t("evaluations:syntheticStateFilterLabel", {
                defaultValue: "Review state"
              })}
            </Text>
            <Select
              className="mt-2 w-full"
              aria-label="Synthetic review state filter"
              value={reviewState}
              onChange={(value) => setReviewState(value)}
              options={REVIEW_STATE_OPTIONS}
            />
          </div>
          <div>
            <Text strong>
              {t("evaluations:syntheticDatasetNameLabel", {
                defaultValue: "Promoted dataset name"
              })}
            </Text>
            <Input
              className="mt-2"
              aria-label="Synthetic promoted dataset name"
              value={datasetName}
              onChange={(event) => setDatasetName(event.target.value)}
            />
          </div>
          <div className="flex items-end">
            <Button
              type="primary"
              onClick={handlePromote}
              disabled={selectedIds.length === 0}
              loading={promoteMutation.isPending}
            >
              {t("evaluations:syntheticPromoteCta", {
                defaultValue: "Promote selected"
              })}
            </Button>
          </div>
        </div>

        {promoteMutation.data?.data?.dataset_id && (
          <StatePanel
            state="ready"
            className="mt-4"
            title={t("evaluations:syntheticPromoteSuccessInlineTitle", {
              defaultValue: "Promoted dataset created"
            })}
            message={`${promoteMutation.data.data.dataset_id} (${promoteMutation.data.data.sample_count})`}
          />
        )}
      </Card>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <Spin />
        </div>
      ) : isError ? (
        <EvaluationRecoveryCallout
          title={t("evaluations:syntheticReviewLoadErrorTitle", {
            defaultValue: "Unable to load synthetic review queue"
          })}
          endpoint="/api/v1/evaluations/synthetic/queue"
          error={error}
        />
      ) : queueItems.length === 0 ? (
        <Card>
          <Empty
            description={t("evaluations:syntheticReviewEmpty", {
              defaultValue: "No review items match the current filters."
            })}
          />
        </Card>
      ) : (
        <div className="space-y-3">
          {queueItems.map((sample) => (
            <Card
              key={sample.sample_id}
              size="small"
              title={
                <div className="flex flex-wrap items-center gap-2">
                  <Checkbox
                    checked={selectedIds.includes(sample.sample_id)}
                    onChange={(event) =>
                      toggleSelection(sample.sample_id, event.target.checked)
                    }
                    aria-label={`Select ${sample.sample_id}`}
                  />
                  <span>{sample.sample_id}</span>
                  <Tag>{sample.recipe_kind}</Tag>
                  <Tag>{sample.review_state}</Tag>
                  <Tag>{sample.provenance}</Tag>
                  {sample.source_kind ? <Tag>{sample.source_kind}</Tag> : null}
                </div>
              }
            >
              <div className="space-y-3">
                <div>
                  <Text strong>
                    {t("evaluations:syntheticQueryLabel", {
                      defaultValue: "Query"
                    })}
                  </Text>
                  <Paragraph className="mb-0 mt-1">
                    {queryPreview(sample) || t("evaluations:syntheticNoQuery", {
                      defaultValue: "No query text"
                    })}
                  </Paragraph>
                </div>

                <div>
                  <details className="rounded border border-border bg-bg-muted/40 p-3">
                    <summary className="cursor-pointer text-sm font-medium text-text">
                      {t("evaluations:syntheticPayloadLabel", {
                        defaultValue: "Draft payload"
                      })}
                    </summary>
                    <pre className="mt-3 overflow-x-auto rounded bg-bg-muted p-3 text-xs">
                      {JSON.stringify(sample.sample_payload || {}, null, 2)}
                    </pre>
                  </details>
                </div>

                <div>
                  <Text strong>
                    {t("evaluations:syntheticNotesLabel", {
                      defaultValue: "Reviewer notes"
                    })}
                  </Text>
                  <TextArea
                    aria-label={`Review notes ${sample.sample_id}`}
                    className="mt-2"
                    rows={3}
                    value={notesById[sample.sample_id] || ""}
                    onChange={(event) =>
                      setNotesById((current) => ({
                        ...current,
                        [sample.sample_id]: event.target.value
                      }))
                    }
                  />
                </div>

                <Space wrap>
                  <Button
                    onClick={() => void handleReviewAction(sample, "approve")}
                    loading={reviewMutation.isPending}
                  >
                    {t("evaluations:syntheticApproveCta", {
                      defaultValue: "Approve"
                    })}
                  </Button>
                  <Button
                    danger
                    onClick={() => void handleReviewAction(sample, "reject")}
                    loading={reviewMutation.isPending}
                  >
                    {t("evaluations:syntheticRejectCta", {
                      defaultValue: "Reject"
                    })}
                  </Button>
                  <Button
                    type="primary"
                    onClick={() => void handleReviewAction(sample, "edit_and_approve")}
                    loading={reviewMutation.isPending}
                  >
                    {t("evaluations:syntheticEditApproveCta", {
                      defaultValue: "Edit & approve"
                    })}
                  </Button>
                </Space>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

export default SyntheticReviewTab
