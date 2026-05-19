/**
 * DatasetsTab component
 * Tab for managing evaluation datasets - create, list, view, delete
 */

import React from "react"
import {
  Alert,
  Button,
  Card,
  Empty,
  Form,
  Input,
  Modal,
  Pagination,
  Space,
  Spin,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import {
  useDatasetsList,
  useCreateDataset,
  useDeleteDataset,
  useLoadDatasetSamples,
  useCloseDatasetViewer,
  parseSamplesJson
} from "../hooks/useDatasets"
import { useEvaluationsStore } from "@/store/evaluations"
import { CopyButton, DatasetUpload, JsonEditor } from "../components"
import type { DatasetResponse, DatasetSample } from "@/services/evaluations"

const { Text } = Typography

export const DatasetsTab: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const [form] = Form.useForm()
  const [loadingDatasetId, setLoadingDatasetId] = React.useState<string | null>(null)
  const [deletingDatasetId, setDeletingDatasetId] = React.useState<string | null>(null)

  // Store state
  const {
    createDatasetOpen,
    openCreateDataset,
    closeCreateDataset,
    viewingDataset,
    datasetSamples,
    datasetSamplesPage,
    datasetSamplesPageSize,
    datasetSamplesTotal,
    setDatasetSamplesPage
  } = useEvaluationsStore((s) => ({
    createDatasetOpen: s.createDatasetOpen,
    openCreateDataset: s.openCreateDataset,
    closeCreateDataset: s.closeCreateDataset,
    viewingDataset: s.viewingDataset,
    datasetSamples: s.datasetSamples,
    datasetSamplesPage: s.datasetSamplesPage,
    datasetSamplesPageSize: s.datasetSamplesPageSize,
    datasetSamplesTotal: s.datasetSamplesTotal,
    setDatasetSamplesPage: s.setDatasetSamplesPage
  }))

  // Queries & mutations
  const { data: datasetListResp, isLoading: datasetsLoading, isError: datasetsError } =
    useDatasetsList()
  const createDatasetMutation = useCreateDataset()
  const deleteDatasetMutation = useDeleteDataset()
  const loadDatasetMutation = useLoadDatasetSamples()
  const closeViewer = useCloseDatasetViewer()

  const datasets: DatasetResponse[] = datasetListResp?.data?.data || []
  const samplesJsonValue = Form.useWatch("samplesJson", form) || ""
  const metadataJsonValue = Form.useWatch("metadataJson", form) || ""
  const pagedSamples = datasetSamples

  const handleSubmitCreate = async () => {
    try {
      const values = await form.validateFields()
      let samples: DatasetSample[] = [
        {
          input: values.sampleInput,
          expected: values.sampleExpected || undefined
        }
      ]
      if (values.samplesJson) {
        const { samples: parsed, error } = parseSamplesJson(values.samplesJson)
        if (error) {
          form.setFields([{ name: "samplesJson", errors: [error] }])
          return
        }
        if (parsed) {
          samples = parsed
        }
      }

      let metadata: Record<string, any> | undefined
      if (values.metadataJson) {
        try {
          metadata = JSON.parse(values.metadataJson)
        } catch {
          form.setFields([
            {
              name: "metadataJson",
              errors: [
                t("evaluations:invalidJsonError", {
                  defaultValue: "Invalid JSON"
                }) as string
              ]
            }
          ])
          return
        }
      }

      await createDatasetMutation.mutateAsync({
        name: values.name,
        description: values.description,
        samples,
        metadata
      })
      form.resetFields()
      closeCreateDataset()
    } catch {
      // Form validation errors handled by antd
    }
  }

  const handleDeleteDataset = (datasetId: string) => {
    Modal.confirm({
      title: t("evaluations:deleteDatasetConfirmTitle", {
        defaultValue: "Delete this dataset?"
      }),
      content: t("evaluations:deleteDatasetConfirmDescription", {
        defaultValue:
          "This will permanently remove the dataset. Evaluations using it will need a new dataset."
      }),
      okButtonProps: { danger: true },
      onOk: async () => {
        setDeletingDatasetId(datasetId)
        try {
          await deleteDatasetMutation.mutateAsync(datasetId)
        } finally {
          setDeletingDatasetId((current) =>
            current === datasetId ? null : current
          )
        }
      }
    })
  }

  const handleViewDataset = async (datasetId: string) => {
    setLoadingDatasetId(datasetId)
    try {
      await loadDatasetMutation.mutateAsync({
        datasetId,
        page: 1,
        pageSize: datasetSamplesPageSize
      })
    } finally {
      setLoadingDatasetId((current) => (current === datasetId ? null : current))
    }
  }

  return (
    <div className="space-y-4">
      <Card
        title={t("evaluations:datasetsTitle", {
          defaultValue: "Datasets"
        })}
        extra={
          <Button
            onClick={openCreateDataset}
            disabled={createDatasetMutation.isPending}
            data-eval-tour="new-dataset"
          >
            {t("evaluations:newDatasetCta", {
              defaultValue: "New dataset"
            })}
          </Button>
        }
      >
        {datasetsLoading ? (
          <div className="flex justify-center py-4">
            <Spin />
          </div>
        ) : datasetsError || datasetListResp?.ok === false ? (
          <Alert
            type="warning"
            showIcon
            title={t("evaluations:datasetsErrorTitle", {
              defaultValue: "Unable to load datasets"
            })}
          />
        ) : datasets.length === 0 ? (
          <Empty
            description={t("evaluations:datasetsEmpty", {
              defaultValue:
                "No datasets yet. Create one to attach to evaluations."
            })}
          />
        ) : (
          <div className="flex flex-col gap-2">
            {datasets.map((ds) => (
              <Card
                key={ds.id}
                size="small"
                className="hover:border-primary/70"
                styles={{ body: { padding: "8px 12px" } }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{ds.name}</span>
                      <CopyButton text={ds.id} />
                    </div>
                    {ds.description && (
                      <span className="text-xs text-text-subtle">
                        {ds.description}
                      </span>
                    )}
                    <span className="text-xs text-text-subtle">
                      {t("evaluations:datasetSampleCount", {
                        defaultValue: "{{count}} samples",
                        count: ds.sample_count
                      })}
                    </span>
                  </div>
                  <Space>
                    <Button
                      size="small"
                      loading={loadingDatasetId === ds.id}
                      onClick={() => void handleViewDataset(ds.id)}
                    >
                      {t("common:view", { defaultValue: "View" })}
                    </Button>
                    <Button
                      size="small"
                      danger
                      loading={deletingDatasetId === ds.id}
                      onClick={() => handleDeleteDataset(ds.id)}
                    >
                      {t("common:delete", { defaultValue: "Delete" })}
                    </Button>
                  </Space>
                </div>
              </Card>
            ))}
          </div>
        )}
      </Card>

      {/* Create Dataset Modal */}
      <Modal
        title={t("evaluations:createDatasetModalTitle", {
          defaultValue: "New dataset"
        })}
        open={createDatasetOpen}
        onCancel={() => {
          closeCreateDataset()
          form.resetFields()
        }}
        onOk={handleSubmitCreate}
        confirmLoading={createDatasetMutation.isPending}
        okText={t("common:create", { defaultValue: "Create" }) as string}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            label={t("evaluations:datasetNameLabel", {
              defaultValue: "Name"
            })}
            name="name"
            rules={[{ required: true }]}
          >
            <Input
              placeholder={t("evaluations:datasetNamePlaceholder", {
                defaultValue: "my_dataset"
              })}
            />
          </Form.Item>
          <Form.Item
            label={t("evaluations:datasetDescriptionLabel", {
              defaultValue: "Description"
            })}
            name="description"
          >
            <Input.TextArea rows={2} />
          </Form.Item>
          <DatasetUpload
            onSamplesLoaded={(samples) => {
              form.setFieldsValue({
                samplesJson: JSON.stringify(samples, null, 2)
              })
            }}
          />
          <Form.Item
            label={t("evaluations:sampleInputLabel", {
              defaultValue: "Sample input"
            })}
            name="sampleInput"
            dependencies={["samplesJson"]}
            rules={[
              {
                validator: async (_rule, value) => {
                  const samplesJson = form.getFieldValue("samplesJson")
                  if (value || samplesJson) return Promise.resolve()
                  return Promise.reject(
                    new Error(
                      t("evaluations:sampleInputRequired", {
                        defaultValue:
                          "Provide a sample input or upload samples JSON."
                      }) as string
                    )
                  )
                }
              }
            ]}
          >
            <Input.TextArea rows={3} />
          </Form.Item>
          <Form.Item
            label={t("evaluations:sampleExpectedLabel", {
              defaultValue: "Expected output (optional)"
            })}
            name="sampleExpected"
          >
            <Input.TextArea rows={3} />
          </Form.Item>
          <Form.Item
            label={t("evaluations:samplesJsonLabel", {
              defaultValue: "Samples JSON (optional, overrides fields)"
            })}
            name="samplesJson"
          >
            <JsonEditor
              rows={4}
              value={samplesJsonValue}
              onChange={(value) => form.setFieldsValue({ samplesJson: value })}
              placeholder={t("evaluations:samplesJsonPlaceholder", {
                defaultValue:
                  '[{\"input\": {\"question\": \"Q1\"}, \"expected\": {\"answer\": \"A\"}}]'
              })}
            />
          </Form.Item>
          <Form.Item
            label={t("evaluations:datasetMetadataLabel", {
              defaultValue: "Metadata (JSON, optional)"
            })}
            name="metadataJson"
          >
            <JsonEditor
              rows={3}
              value={metadataJsonValue}
              onChange={(value) => form.setFieldsValue({ metadataJson: value })}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* View Dataset Modal */}
      <Modal
        title={t("evaluations:datasetDetailTitle", {
          defaultValue: "Dataset details"
        })}
        open={!!viewingDataset}
        onCancel={closeViewer}
        footer={
          <Button onClick={closeViewer}>
            {t("common:close", { defaultValue: "Close" })}
          </Button>
        }
        width={700}
      >
        {viewingDataset && (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-3">
              <div>
                <Text type="secondary" className="text-xs">
                  {t("common:name", { defaultValue: "Name" })}:{" "}
                </Text>
                <Text className="text-sm font-medium">{viewingDataset.name}</Text>
              </div>
              <div>
                <Text type="secondary" className="text-xs">
                  {t("common:id", { defaultValue: "ID" })}:{" "}
                </Text>
                <code className="text-xs">{viewingDataset.id}</code>
                <CopyButton text={viewingDataset.id} />
              </div>
              <div>
                <Text type="secondary" className="text-xs">
                  {t("evaluations:datasetSampleCountLabel", {
                    defaultValue: "Samples"
                  })}
                  :{" "}
                </Text>
                <Text className="text-sm">{viewingDataset.sample_count}</Text>
              </div>
            </div>

            {viewingDataset.description && (
              <div>
                <Text type="secondary" className="text-xs">
                  {t("evaluations:descriptionLabel", {
                    defaultValue: "Description"
                  })}
                  :{" "}
                </Text>
                <Text className="text-sm">{viewingDataset.description}</Text>
              </div>
            )}

            <div>
              <Text type="secondary" className="text-xs block mb-1">
                {t("evaluations:samplesPreviewLabel", {
                  defaultValue: "Samples preview"
                })}
              </Text>
              {pagedSamples.length === 0 ? (
                <Empty
                  description={t("evaluations:noSamplesPreview", {
                    defaultValue: "No samples to preview"
                  })}
                />
              ) : (
                <div className="space-y-2">
                  {pagedSamples.map((sample, idx) => (
                    <pre
                      key={`${datasetSamplesPage}-${idx}`}
                      className="max-h-32 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text"
                    >
                      {JSON.stringify(sample, null, 2)}
                    </pre>
                  ))}
                </div>
              )}
            </div>

            {datasetSamplesTotal !== null &&
              datasetSamplesTotal > datasetSamplesPageSize && (
                <Pagination
                  current={datasetSamplesPage}
                  pageSize={datasetSamplesPageSize}
                  total={datasetSamplesTotal}
                  size="small"
                  onChange={(page) => {
                    if (!viewingDataset?.id) {
                      setDatasetSamplesPage(page)
                      return
                    }
                    loadDatasetMutation.mutate({
                      datasetId: viewingDataset.id,
                      page,
                      pageSize: datasetSamplesPageSize
                    })
                  }}
                />
              )}
          </div>
        )}
      </Modal>
    </div>
  )
}

export default DatasetsTab
