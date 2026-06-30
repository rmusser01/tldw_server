/**
 * EvaluationsTab component
 * Main tab for managing evaluations - list, create, edit, delete
 */

import React, { useEffect } from "react"
import {
  Button,
  Card,
  Empty,
  Form,
  Modal,
  Space,
  Spin,
  Tag,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import {
  useEvaluationsList,
  useEvaluationDetail,
  useCreateEvaluation,
  useUpdateEvaluation,
  useDeleteEvaluation,
  useEvaluationDefaults,
  getDefaultEvalSpecForType
} from "../hooks/useEvaluations"
import { useDatasetsList } from "../hooks/useDatasets"
import { useEvaluationsStore } from "@/store/evaluations"
import { CopyButton, CreateEvaluationWizard } from "../components"
import { EvaluationRecoveryCallout } from "../components/EvaluationRecoveryCallout"
import type { EvaluationSummary } from "@/services/evaluations"

const { Text } = Typography

export const EvaluationsTab: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const queryClient = useQueryClient()
  const [form] = Form.useForm()

  // Store state
  const {
    selectedEvalId,
    setActiveTab,
    setSelectedEvalId,
    setSelectedRunId,
    editingEvalId,
    setEditingEvalId,
    createEvalOpen,
    openCreateEval,
    closeCreateEval,
    evalSpecText,
    setEvalSpecText,
    evalSpecError,
    setEvalSpecError,
    inlineDatasetEnabled,
    setInlineDatasetEnabled,
    inlineDatasetText,
    setInlineDatasetText,
    evalIdempotencyKey,
    regenerateEvalIdempotencyKey
  } = useEvaluationsStore((s) => ({
    selectedEvalId: s.selectedEvalId,
    setActiveTab: s.setActiveTab,
    setSelectedEvalId: s.setSelectedEvalId,
    setSelectedRunId: s.setSelectedRunId,
    editingEvalId: s.editingEvalId,
    setEditingEvalId: s.setEditingEvalId,
    createEvalOpen: s.createEvalOpen,
    openCreateEval: s.openCreateEval,
    closeCreateEval: s.closeCreateEval,
    evalSpecText: s.evalSpecText,
    setEvalSpecText: s.setEvalSpecText,
    evalSpecError: s.evalSpecError,
    setEvalSpecError: s.setEvalSpecError,
    inlineDatasetEnabled: s.inlineDatasetEnabled,
    setInlineDatasetEnabled: s.setInlineDatasetEnabled,
    inlineDatasetText: s.inlineDatasetText,
    setInlineDatasetText: s.setInlineDatasetText,
    evalIdempotencyKey: s.evalIdempotencyKey,
    regenerateEvalIdempotencyKey: s.regenerateEvalIdempotencyKey
  }))

  // Queries
  const {
    data: evalListResp,
    isLoading: evalsLoading,
    isError: evalsError,
    error: evalsErrorValue
  } =
    useEvaluationsList({ limit: 20 })
  const {
    data: evalDetailResp,
    isLoading: evalDetailLoading,
    isError: evalDetailError,
    error: evalDetailErrorValue
  } =
    useEvaluationDetail(selectedEvalId)
  const { data: evalDefaults } = useEvaluationDefaults()
  const { data: datasetListResp, isLoading: datasetsLoading } = useDatasetsList()

  // Mutations
  const createEvalMutation = useCreateEvaluation()
  const updateEvalMutation = useUpdateEvaluation()
  const deleteEvalMutation = useDeleteEvaluation()

  const evaluations = evalListResp?.data?.data || []
  const evalDetail = evalDetailResp?.data
  const datasets = datasetListResp?.data?.data || []

  // Initialize defaults
  useEffect(() => {
    if (evalDefaults && !evalSpecText) {
      const defaultType = evalDefaults.defaultEvalType || "response_quality"
      const spec = getDefaultEvalSpecForType(
        defaultType,
        evalDefaults.defaultSpecByType
      )
      setEvalSpecText(JSON.stringify(spec, null, 2))
    }
  }, [evalDefaults, evalSpecText, setEvalSpecText])

  const handleOpenCreate = () => {
    const defaultType = evalDefaults?.defaultEvalType || "response_quality"
    const defaultModel = evalDefaults?.defaultTargetModel || "gpt-3.5-turbo"
    const spec = getDefaultEvalSpecForType(
      defaultType,
      evalDefaults?.defaultSpecByType
    )
    setEvalSpecError(null)
    setEvalSpecText(JSON.stringify(spec, null, 2))
    setInlineDatasetEnabled(false)
    form.setFieldsValue({
      evalType: defaultType,
      runModel: defaultModel,
      name: "",
      description: "",
      datasetId: evalDefaults?.defaultDatasetId || undefined,
      idempotencyKey: evalIdempotencyKey,
      evalMetadataJson: undefined
    })
    openCreateEval()
  }

  const handleOpenEdit = () => {
    if (!selectedEvalId) return
    const selectedSummary = evaluations.find((e) => e.id === selectedEvalId)
    const detail = evalDetail as any
    const type =
      detail?.eval_type || selectedSummary?.eval_type || "response_quality"
    setEvalSpecError(null)
    setEvalSpecText(
      JSON.stringify(
        detail?.eval_spec || getDefaultEvalSpecForType(type),
        null,
        2
      )
    )
    form.setFieldsValue({
      evalType: type,
      name: detail?.name || selectedSummary?.name || selectedEvalId,
      description: detail?.description || selectedSummary?.description,
      datasetId: detail?.dataset_id || selectedSummary?.dataset_id,
      evalMetadataJson: detail?.metadata
        ? JSON.stringify(detail.metadata, null, 2)
        : undefined
    })
    setInlineDatasetEnabled(false)
    openCreateEval(selectedEvalId)
  }

  const handleDelete = () => {
    if (!selectedEvalId) return
    Modal.confirm({
      title: t("evaluations:deleteConfirmTitle", {
        defaultValue: "Delete this evaluation?"
      }),
      content: t("evaluations:deleteConfirmDescription", {
        defaultValue:
          "This will remove the evaluation definition. Runs already created remain in history."
      }),
      okButtonProps: { danger: true, loading: deleteEvalMutation.isPending },
      onOk: () => deleteEvalMutation.mutateAsync(selectedEvalId)
    })
  }

  const handleSubmit = async () => {
    setEvalSpecError(null)
    try {
      const values = await form.validateFields()
      let spec: any
      try {
        spec = JSON.parse(evalSpecText || "{}")
      } catch (e: any) {
        setEvalSpecError(e?.message || "Invalid JSON")
        return
      }

      const sanitizedName = String(values.name || "")
        .trim()
        .replace(/[^a-zA-Z0-9_-]/g, "-")

      let inlineDataset: any[] | undefined
      if (inlineDatasetEnabled && inlineDatasetText.trim().length > 0) {
        try {
          const parsed = JSON.parse(inlineDatasetText)
          if (Array.isArray(parsed)) {
            inlineDataset = parsed
          } else {
            throw new Error("Inline dataset must be an array.")
          }
        } catch (e: any) {
          setEvalSpecError(e?.message || "Invalid inline dataset JSON")
          return
        }
      }

      let metadata: Record<string, any> | undefined
      if (values.evalMetadataJson) {
        try {
          metadata = JSON.parse(values.evalMetadataJson)
        } catch (e: any) {
          setEvalSpecError(e?.message || "Invalid metadata JSON")
          return
        }
      }

      const payload = {
        name: sanitizedName,
        description: values.description,
        eval_type: values.evalType,
        eval_spec: spec,
        dataset_id: inlineDataset ? undefined : values.datasetId || undefined,
        dataset: inlineDataset,
        metadata
      }

      if (editingEvalId) {
        await updateEvalMutation.mutateAsync({
          evalId: editingEvalId,
          payload
        })
      } else {
        await createEvalMutation.mutateAsync({
          payload,
          idempotencyKey: values.idempotencyKey || evalIdempotencyKey
        })
        regenerateEvalIdempotencyKey()
      }

      closeCreateEval()
      form.resetFields()
    } catch {
      // Form validation errors handled by antd
    }
  }

  return (
    <div className="space-y-4">
      {/* Evaluations List */}
      <Card
        data-testid="evaluations-list-card"
        title={t("evaluations:listTitle", {
          defaultValue: "Recent evaluations"
        })}
        extra={
          <Space>
            <Button
              onClick={handleOpenCreate}
              type="primary"
              data-eval-tour="create-evaluation"
              data-testid="evaluations-create-button"
            >
              {t("evaluations:newEvaluationCta", {
                defaultValue: "New evaluation"
              })}
            </Button>
            <Button disabled={!selectedEvalId} onClick={handleOpenEdit}>
              {t("common:edit", { defaultValue: "Edit" })}
            </Button>
            <Button
              danger
              disabled={!selectedEvalId}
              loading={deleteEvalMutation.isPending}
              onClick={handleDelete}
            >
              {t("common:delete", { defaultValue: "Delete" })}
            </Button>
          </Space>
        }
      >
        {evalsLoading ? (
          <div className="flex justify-center py-6">
            <Spin />
          </div>
        ) : evalsError || evalListResp?.ok === false ? (
          <EvaluationRecoveryCallout
            title={t("evaluations:loadErrorTitle", {
              defaultValue: "Unable to load evaluations"
            })}
            message={t("evaluations:loadErrorDescription", {
              defaultValue:
                "Check your tldw server connection and API credentials, then try again."
            })}
            endpoint="/api/v1/evaluations"
            error={evalsErrorValue}
            response={evalListResp}
          />
        ) : evaluations.length === 0 ? (
          <Empty
            description={t("evaluations:emptyList", {
              defaultValue:
                "No evaluations yet. Start with Recipes for guided setup, or create a custom evaluation."
            })}
          >
            <Button type="link" onClick={() => setActiveTab("recipes")}>
              {t("evaluations:openRecipesCta", {
                defaultValue: "Open Recipes"
              })}
            </Button>
          </Empty>
        ) : (
          <div className="flex flex-col gap-2">
            {evaluations.map((ev: EvaluationSummary) => (
              <Card
                key={ev.id}
                size="small"
                className={`cursor-pointer hover:border-primary/70 ${
                  selectedEvalId === ev.id ? "border-primary" : ""
                }`}
                bodyStyle={{ padding: "8px 12px" }}
                onClick={() => {
                  setSelectedEvalId(ev.id)
                  setSelectedRunId(null)
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">
                        {ev.name || ev.id}
                      </span>
                      <CopyButton text={ev.id} />
                    </div>
                    {ev.description && (
                      <span className="text-xs text-text-subtle">
                        {ev.description}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {selectedEvalId === ev.id && (
                      <Tag color="green" className="text-xs">
                        {t("evaluations:selectedTag", {
                          defaultValue: "Selected"
                        })}
                      </Tag>
                    )}
                    {ev.eval_type && (
                      <Tag color="blue" className="text-xs">
                        {ev.eval_type}
                      </Tag>
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </Card>

      {/* Evaluation Details */}
      <Card
        data-testid="evaluations-detail-card"
        title={t("evaluations:detailTitle", {
          defaultValue: "Evaluation details"
        })}
        extra={
          selectedEvalId && (
            <Space>
              <Button size="small" onClick={handleOpenEdit}>
                {t("common:edit", { defaultValue: "Edit" })}
              </Button>
              <Button
                size="small"
                onClick={() =>
                  void queryClient.invalidateQueries({
                    queryKey: ["evaluations", "detail", selectedEvalId]
                  })
                }
              >
                {t("common:refresh", { defaultValue: "Refresh" })}
              </Button>
            </Space>
          )
        }
      >
        {!selectedEvalId ? (
          <Text type="secondary" className="text-xs">
            {t("evaluations:noEvalSelectedDetails", {
              defaultValue: "Select an evaluation to inspect its spec."
            })}
          </Text>
        ) : evalDetailLoading ? (
          <div className="flex justify-center py-4">
            <Spin />
          </div>
        ) : evalDetailError || evalDetailResp?.ok === false ? (
          <EvaluationRecoveryCallout
            title={t("evaluations:detailErrorTitle", {
              defaultValue: "Unable to load evaluation details"
            })}
            endpoint={`/api/v1/evaluations/${selectedEvalId}`}
            error={evalDetailErrorValue}
            response={evalDetailResp}
          />
        ) : evalDetail ? (
          <div className="space-y-2 text-xs">
            <div className="flex flex-wrap gap-3">
              <Tag>
                {t("common:id", { defaultValue: "ID" })}:{" "}
                <code>{evalDetail.id}</code>
                <CopyButton text={evalDetail.id} />
              </Tag>
              {evalDetail.eval_type && (
                <Tag color="blue">{evalDetail.eval_type}</Tag>
              )}
              {evalDetail.dataset_id && (
                <Tag color="purple">
                  {t("evaluations:datasetLabel", {
                    defaultValue: "Dataset"
                  })}
                  : {evalDetail.dataset_id}
                </Tag>
              )}
            </div>
            {evalDetail.description && (
              <div>
                <Text type="secondary">
                  {t("evaluations:descriptionLabel", {
                    defaultValue: "Description"
                  })}
                  {": "}
                </Text>
                <Text>{evalDetail.description}</Text>
              </div>
            )}
            {evalDetail.metadata && (
              <div>
                <Text type="secondary">
                  {t("common:metadata", { defaultValue: "Metadata" })}
                </Text>
                <pre className="mt-1 max-h-32 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text">
                  {JSON.stringify(evalDetail.metadata, null, 2)}
                </pre>
              </div>
            )}
            <div>
              <Text type="secondary">
                {t("evaluations:evalSpecSnippetLabel", {
                  defaultValue: "Evaluation spec (snippet)"
                })}
              </Text>
              <pre className="mt-1 max-h-48 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text">
                {JSON.stringify(evalDetail.eval_spec || {}, null, 2)}
              </pre>
            </div>
          </div>
        ) : null}
      </Card>

      {/* Create/Edit Modal */}
      <Modal
        title={
          editingEvalId
            ? t("evaluations:editEvalModalTitle", {
                defaultValue: "Edit evaluation"
              })
            : t("evaluations:createEvalModalTitle", {
                defaultValue: "New evaluation"
              })
        }
        open={createEvalOpen}
        onCancel={() => {
          closeCreateEval()
          form.resetFields()
        }}
        footer={null}
        width={640}
      >
        <CreateEvaluationWizard
          key={`${createEvalOpen ? "open" : "closed"}-${editingEvalId || "new"}`}
          form={form}
          datasets={datasets}
          datasetsLoading={datasetsLoading}
          evalDefaults={evalDefaults}
          editingEvalId={editingEvalId}
          evalSpecText={evalSpecText}
          evalSpecError={evalSpecError}
          inlineDatasetEnabled={inlineDatasetEnabled}
          inlineDatasetText={inlineDatasetText}
          evalIdempotencyKey={evalIdempotencyKey}
          onSpecChange={setEvalSpecText}
          onSpecError={setEvalSpecError}
          onInlineDatasetEnabled={setInlineDatasetEnabled}
          onInlineDatasetText={setInlineDatasetText}
          onRegenerateIdempotencyKey={() => {
            regenerateEvalIdempotencyKey()
            const nextKey = useEvaluationsStore.getState().evalIdempotencyKey
            form.setFieldsValue({ idempotencyKey: nextKey })
            return nextKey
          }}
          onCancel={() => {
            closeCreateEval()
            form.resetFields()
          }}
          onSubmit={handleSubmit}
          submitting={createEvalMutation.isPending || updateEvalMutation.isPending}
        />
      </Modal>
    </div>
  )
}

export default EvaluationsTab
