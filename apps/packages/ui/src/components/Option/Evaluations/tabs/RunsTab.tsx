/**
 * RunsTab component
 * Tab for managing evaluation runs - create, list, view details
 */

import React, { useEffect, useMemo, useState } from "react"
import {
  Button,
  Card,
  Collapse,
  Divider,
  Empty,
  Form,
  Input,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import {
  useRateLimits,
  useRunsList,
  useRunDetail,
  useCreateRun,
  useCancelRun,
  useAdhocEvaluation,
  useBenchmarksCatalog,
  useRunBenchmark,
  extractMetricsSummary,
  adhocEndpointOptions
} from "../hooks/useRuns"
import { useEvaluationsList } from "../hooks/useEvaluations"
import { useEvaluationsStore } from "@/store/evaluations"
import { EvaluationRecoveryCallout } from "../components/EvaluationRecoveryCallout"
import {
  CopyButton,
  JsonEditor,
  EvaluationsBreadcrumb,
  MetricsChart,
  PollingIndicator,
  RateLimitsWidget,
  RunComparisonView,
  StatusBadge
} from "../components"

const { Text } = Typography

export const RunsTab: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const queryClient = useQueryClient()
  const [runForm] = Form.useForm()
  const [compareRunAId, setCompareRunAId] = useState<string | null>(null)
  const [compareRunBId, setCompareRunBId] = useState<string | null>(null)

  // Store state
  const {
    selectedEvalId,
    setSelectedEvalId,
    selectedRunId,
    setSelectedRunId,
    runConfigText,
    setRunConfigText,
    datasetOverrideText,
    setDatasetOverrideText,
    runIdempotencyKey,
    regenerateRunIdempotencyKey,
    quotaSnapshot,
    setQuotaSnapshot,
    isPolling,
    setIsPolling,
    adhocEndpoint,
    setAdhocEndpoint,
    selectedBenchmark,
    setSelectedBenchmark,
    adhocPayloadText,
    setAdhocPayloadText,
    adhocResult
  } = useEvaluationsStore((s) => ({
    selectedEvalId: s.selectedEvalId,
    setSelectedEvalId: s.setSelectedEvalId,
    selectedRunId: s.selectedRunId,
    setSelectedRunId: s.setSelectedRunId,
    runConfigText: s.runConfigText,
    setRunConfigText: s.setRunConfigText,
    datasetOverrideText: s.datasetOverrideText,
    setDatasetOverrideText: s.setDatasetOverrideText,
    runIdempotencyKey: s.runIdempotencyKey,
    regenerateRunIdempotencyKey: s.regenerateRunIdempotencyKey,
    quotaSnapshot: s.quotaSnapshot,
    setQuotaSnapshot: s.setQuotaSnapshot,
    isPolling: s.isPolling,
    setIsPolling: s.setIsPolling,
    adhocEndpoint: s.adhocEndpoint,
    setAdhocEndpoint: s.setAdhocEndpoint,
    selectedBenchmark: s.selectedBenchmark,
    setSelectedBenchmark: s.setSelectedBenchmark,
    adhocPayloadText: s.adhocPayloadText,
    setAdhocPayloadText: s.setAdhocPayloadText,
    adhocResult: s.adhocResult
  }))

  // Queries
  const { data: evalListResp } = useEvaluationsList({ limit: 20 })
  const { data: rateLimitsResp, isLoading: rateLimitsLoading, isError: rateLimitsError } =
    useRateLimits()
  const {
    data: runsListResp,
    isLoading: runsLoading,
    isError: runsError,
    error: runsErrorValue
  } =
    useRunsList(selectedEvalId)
  const {
    data: runDetailResp,
    isLoading: runDetailLoading,
    isError: runDetailError,
    error: runDetailErrorValue
  } =
    useRunDetail(selectedRunId)
  const { data: compareRunAResp } = useRunDetail(compareRunAId, {
    enablePolling: false,
    captureQuota: false
  })
  const { data: compareRunBResp } = useRunDetail(compareRunBId, {
    enablePolling: false,
    captureQuota: false
  })
  const { data: benchmarksResp } = useBenchmarksCatalog()

  // Mutations
  const createRunMutation = useCreateRun()
  const cancelRunMutation = useCancelRun()
  const adhocMutation = useAdhocEvaluation()
  const runBenchmarkMutation = useRunBenchmark()

  const evaluations = evalListResp?.data?.data || []
  const rateLimits = rateLimitsResp?.data
  const runs = runsListResp?.data?.data || []
  const runDetail = runDetailResp?.data
  const compareRunA = compareRunAResp?.data
  const compareRunB = compareRunBResp?.data
  const benchmarks = benchmarksResp?.data?.data || []

  const runMetrics = useMemo(
    () => extractMetricsSummary(runDetail?.results),
    [runDetail]
  )

  const isRunActive = ["running", "pending"].includes(
    String(runDetail?.status || "").toLowerCase()
  )

  const selectedEval = evaluations.find((ev) => ev.id === selectedEvalId)
  const evalBreadcrumbLabel = selectedEval?.name || selectedEvalId || null

  useEffect(() => {
    if (!selectedRunId) {
      setIsPolling(false)
    }
  }, [selectedRunId, setIsPolling])

  useEffect(() => {
    setCompareRunAId(null)
    setCompareRunBId(null)
  }, [selectedEvalId])

  useEffect(() => {
    if (selectedRunId && !compareRunAId) {
      setCompareRunAId(selectedRunId)
    }
  }, [selectedRunId, compareRunAId])

  useEffect(() => {
    if (!compareRunBId && compareRunAId && runs.length > 1) {
      const candidate = runs.find((run) => run.id !== compareRunAId)
      if (candidate) {
        setCompareRunBId(candidate.id)
      }
    }
  }, [compareRunAId, compareRunBId, runs])

  useEffect(() => {
    if (adhocEndpoint !== "benchmark-run") return
    if (selectedBenchmark && benchmarks.some((b) => b.name === selectedBenchmark)) {
      return
    }
    const preferred =
      benchmarks.find((b) => b.name === "bullshit_benchmark")?.name ||
      benchmarks[0]?.name ||
      null
    if (preferred) {
      setSelectedBenchmark(preferred)
    }
  }, [adhocEndpoint, benchmarks, selectedBenchmark, setSelectedBenchmark])

  const handleStartRun = async () => {
    if (!selectedEvalId) return
    const values = await runForm.validateFields().catch(() => null)
    if (!values) return

    let config: Record<string, any> | undefined
    if (values.configJson) {
      try {
        config = JSON.parse(values.configJson)
      } catch {
        return
      }
    }

    let datasetOverride: { samples: any[] } | undefined
    if (values.datasetOverrideJson) {
      try {
        const parsed = JSON.parse(values.datasetOverrideJson)
        if (Array.isArray(parsed)) {
          datasetOverride = { samples: parsed }
        } else if (parsed?.samples && Array.isArray(parsed.samples)) {
          datasetOverride = { samples: parsed.samples }
        }
      } catch {
        return
      }
    }

    await createRunMutation.mutateAsync({
      evalId: selectedEvalId,
      payload: {
        target_model: values.targetModel || "gpt-3.5-turbo",
        config,
        dataset_override: datasetOverride,
        webhook_url: values.webhookUrl || undefined
      },
      idempotencyKey: values.idempotencyKey || runIdempotencyKey
    })
    regenerateRunIdempotencyKey()
  }

  const handleAdhocRun = async () => {
    try {
      const parsed = adhocPayloadText ? JSON.parse(adhocPayloadText) : {}
      if (adhocEndpoint === "benchmark-run") {
        if (!selectedBenchmark) return
        await runBenchmarkMutation.mutateAsync({
          benchmarkName: selectedBenchmark,
          body: parsed
        })
        return
      }
      await adhocMutation.mutateAsync({ endpoint: adhocEndpoint, body: parsed })
    } catch {
      // Error handled in mutation
    }
  }

  return (
    <div className="space-y-3">
      <EvaluationsBreadcrumb evalName={evalBreadcrumbLabel} runId={selectedRunId} />
      <div className="grid gap-4 md:grid-cols-2">
      {/* Left Column - Run Form & List */}
      <div className="space-y-4">
        {/* Eval Selector */}
        <Card
          size="small"
          title={t("evaluations:selectEvaluation", {
            defaultValue: "Select evaluation"
          })}
        >
          <Select
            className="w-full"
            placeholder={t("evaluations:selectEvaluationPlaceholder", {
              defaultValue: "Choose an evaluation to run"
            })}
            value={selectedEvalId}
            onChange={(value) => {
              setSelectedEvalId(value)
              setSelectedRunId(null)
            }}
            options={evaluations.map((e) => ({
              value: e.id,
              label: e.name || e.id
            }))}
            allowClear
          />
        </Card>

        {/* Run Form */}
        <Card
          title={t("evaluations:runsTitle", { defaultValue: "Runs" })}
          extra={<PollingIndicator isPolling={isPolling} />}
        >
          {!selectedEvalId ? (
            <Text type="secondary" className="text-xs">
              {t("evaluations:noEvalSelectedRuns", {
                defaultValue: "Select an evaluation to see its recent runs."
              })}
            </Text>
          ) : (
            <>
              <div className="mb-2 rounded-md border border-border bg-bg-muted/40 p-3 text-xs text-text-muted">
                <Text strong className="block text-text">
                  {t("evaluations:runPollingTitle", {
                    defaultValue: "Run status updates"
                  })}
                </Text>
                <p className="mb-0 mt-1">
                  {t("evaluations:runPollingHint", {
                    defaultValue:
                      "Runs execute asynchronously. The UI polls every ~3s until status leaves running/pending."
                  })}
                </p>
              </div>
              <Form form={runForm} layout="vertical" size="small">
                <Form.Item
                  label={t("evaluations:runModelLabelShort", {
                    defaultValue: "Target model"
                  })}
                  name="targetModel"
                  initialValue="gpt-3.5-turbo"
                >
                  <Input />
                </Form.Item>
                <Form.Item
                  label={t("evaluations:runConfigLabel", {
                    defaultValue: "Config (JSON)"
                  })}
                  name="configJson"
                  initialValue={runConfigText}
                >
                  <JsonEditor
                    rows={3}
                    value={runConfigText}
                    onChange={(v) => {
                      setRunConfigText(v)
                      runForm.setFieldsValue({ configJson: v })
                    }}
                  />
                </Form.Item>

                <Collapse
                  ghost
                  items={[
                    {
                      key: "advanced",
                      label: t("evaluations:advancedOptions", {
                        defaultValue: "Advanced options"
                      }),
                      children: (
                        <>
                          <Form.Item
                            label={t("evaluations:datasetOverrideLabel", {
                              defaultValue: "Dataset override (JSON array of samples)"
                            })}
                            name="datasetOverrideJson"
                            initialValue={datasetOverrideText}
                          >
                            <JsonEditor
                              rows={3}
                              value={datasetOverrideText}
                              onChange={(v) => {
                                setDatasetOverrideText(v)
                                runForm.setFieldsValue({ datasetOverrideJson: v })
                              }}
                              placeholder={t("evaluations:datasetOverridePlaceholder", {
                                defaultValue:
                                  '[{\"input\": {\"question\": \"Q1\"}, \"expected\": {\"answer\": \"A\"}}]'
                              })}
                            />
                          </Form.Item>
                          <Form.Item
                            label={t("evaluations:webhookUrlLabel", {
                              defaultValue: "Webhook URL (optional)"
                            })}
                            name="webhookUrl"
                          >
                            <Input
                              placeholder={t("evaluations:webhookUrlPlaceholder", {
                                defaultValue: "https://example.com/hook"
                              })}
                            />
                          </Form.Item>
                          <Form.Item
                            label={
                              <Tooltip
                                title={t(
                                  "evaluations:idempotencyKeyTooltip",
                                  {
                                    defaultValue:
                                      "Prevents duplicate runs if the browser retries the request."
                                  }
                                )}
                              >
                                <span className="cursor-help underline decoration-dotted">
                                  {t("evaluations:idempotencyKeyLabel", {
                                    defaultValue: "Idempotency key"
                                  })}
                                </span>
                              </Tooltip>
                            }
                            name="idempotencyKey"
                            initialValue={runIdempotencyKey}
                          >
                            <Space.Compact className="w-full">
                              <Input />
                              <Button
                                size="small"
                                onClick={() => {
                                  regenerateRunIdempotencyKey()
                                  runForm.setFieldsValue({
                                    idempotencyKey:
                                      useEvaluationsStore.getState().runIdempotencyKey
                                  })
                                }}
                              >
                                {t("common:regenerate", {
                                  defaultValue: "Regenerate"
                                })}
                              </Button>
                            </Space.Compact>
                          </Form.Item>
                        </>
                      )
                    }
                  ]}
                />

                <Space className="mt-4">
                  <Button
                    type="primary"
                    loading={createRunMutation.isPending}
                    onClick={handleStartRun}
                    data-eval-tour="start-run"
                  >
                    {t("evaluations:startRunCta", {
                      defaultValue: "Start run"
                    })}
                  </Button>
                  {selectedRunId && isRunActive && (
                    <Button
                      danger
                      loading={cancelRunMutation.isPending}
                      onClick={() => cancelRunMutation.mutateAsync(selectedRunId)}
                    >
                      {t("evaluations:cancelRunCta", {
                        defaultValue: "Cancel run"
                      })}
                    </Button>
                  )}
                </Space>
              </Form>

              <Divider className="my-3" />

              {/* Runs List */}
              {runsLoading ? (
                <div className="flex justify-center py-4">
                  <Spin />
                </div>
              ) : runsError || runsListResp?.ok === false ? (
                <EvaluationRecoveryCallout
                  title={t("evaluations:runsErrorTitle", {
                    defaultValue: "Unable to load runs"
                  })}
                  endpoint={`/api/v1/evaluations/${selectedEvalId}/runs`}
                  error={runsErrorValue}
                  response={runsListResp}
                />
              ) : runs.length === 0 ? (
                <Empty
                  description={t("evaluations:runsEmpty", {
                    defaultValue: "No runs yet for this evaluation."
                  })}
                />
              ) : (
                <div className="flex flex-col gap-2 text-xs">
                  {runs.map((run) => (
                    <div
                      key={run.id}
                      className={`flex cursor-pointer items-center justify-between rounded border px-2 py-1 ${
                        selectedRunId === run.id
                          ? "border-primary bg-primary/10"
                          : "border-border"
                      }`}
                      onClick={() => {
                        setSelectedRunId(run.id)
                        setQuotaSnapshot(null)
                      }}
                    >
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">
                            {t("evaluations:runLabel", {
                              defaultValue: "Run {{id}}",
                              id: run.id
                            })}
                          </span>
                          <CopyButton text={run.id} />
                        </div>
                        {run.status && <StatusBadge status={run.status} />}
                      </div>
                      {selectedRunId === run.id && (
                        <Tag color="green" className="text-[11px]">
                          {t("evaluations:selectedTag", {
                            defaultValue: "Selected"
                          })}
                        </Tag>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </Card>

        {/* Ad-hoc Evaluator */}
        <Card
          title={t("evaluations:adhocTitle", {
            defaultValue: "Ad-hoc evaluator"
          })}
          extra={
            <Select
              size="small"
              style={{ width: 180 }}
              value={adhocEndpoint}
              onChange={(val) => {
                setAdhocEndpoint(val)
                if (val === "benchmark-run") {
                  setAdhocPayloadText(
                    JSON.stringify(
                      {
                        limit: 25,
                        api_name: "openai",
                        parallel: 4,
                        save_results: true
                      },
                      null,
                      2
                    )
                  )
                  return
                }
                setAdhocPayloadText(
                  JSON.stringify(
                    val.includes("ocr")
                      ? { image_b64: "<base64-data>" }
                      : { input: "Sample text", reference: "Expected reply" },
                    null,
                    2
                  )
                )
              }}
              options={adhocEndpointOptions}
            />
          }
        >
          <Form layout="vertical" size="small">
            {adhocEndpoint === "benchmark-run" && (
              <Form.Item
                label={t("evaluations:benchmarkLabel", {
                  defaultValue: "Benchmark"
                })}
              >
                <Select
                  value={selectedBenchmark || undefined}
                  onChange={(value) => setSelectedBenchmark(value)}
                  options={benchmarks.map((benchmark) => ({
                    value: benchmark.name,
                    label: benchmark.name
                  }))}
                  placeholder={t("evaluations:benchmarkPlaceholder", {
                    defaultValue: "Choose a benchmark"
                  })}
                />
              </Form.Item>
            )}
            <Form.Item
              label={t("evaluations:runConfigLabel", {
                defaultValue: "Config (JSON)"
              })}
            >
              <JsonEditor
                rows={4}
                value={adhocPayloadText}
                onChange={setAdhocPayloadText}
              />
            </Form.Item>
            <Button
              type="primary"
              loading={adhocMutation.isPending || runBenchmarkMutation.isPending}
              onClick={handleAdhocRun}
            >
              {t("evaluations:startRunCta", {
                defaultValue: "Start run"
              })}
            </Button>
          </Form>
          {adhocResult && (
            <pre className="mt-2 max-h-40 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text">
              {JSON.stringify(adhocResult, null, 2)}
            </pre>
          )}
        </Card>
      </div>

      {/* Right Column - Run Details & Rate Limits */}
      <div className="space-y-4">
        {/* Rate Limits */}
        <Card
          title={t("evaluations:rateLimitsTitle", {
            defaultValue: "Evaluation limits"
          })}
        >
          <RateLimitsWidget
            rateLimits={rateLimits}
            isLoading={rateLimitsLoading}
            isError={rateLimitsError}
            quotaSnapshot={quotaSnapshot}
          />
        </Card>

        {/* Run Details */}
        <Card
          title={t("evaluations:runDetailTitle", {
            defaultValue: "Run details"
          })}
          extra={
            selectedRunId && (
              <Space>
                <Button
                  size="small"
                  onClick={() =>
                    void queryClient.invalidateQueries({
                      queryKey: ["evaluations", "run", selectedRunId]
                    })
                  }
                >
                  {t("common:refresh", { defaultValue: "Refresh" })}
                </Button>
                {isRunActive && (
                  <Button
                    size="small"
                    danger
                    loading={cancelRunMutation.isPending}
                    onClick={() => cancelRunMutation.mutateAsync(selectedRunId)}
                  >
                    {t("evaluations:cancelRunCta", {
                      defaultValue: "Cancel run"
                    })}
                  </Button>
                )}
              </Space>
            )
          }
        >
          {!selectedRunId ? (
            <Text type="secondary" className="text-xs">
              {t("evaluations:noRunSelected", {
                defaultValue: "Select a run to see details."
              })}
            </Text>
          ) : runDetailLoading ? (
            <div className="flex justify-center py-4">
              <Spin />
            </div>
          ) : runDetailError || runDetailResp?.ok === false ? (
            <EvaluationRecoveryCallout
              title={t("evaluations:runDetailErrorTitle", {
                defaultValue: "Unable to load run details"
              })}
              endpoint={`/api/v1/evaluations/runs/${selectedRunId}`}
              error={runDetailErrorValue}
              response={runDetailResp}
            />
          ) : runDetail ? (
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <Text type="secondary">
                  {t("evaluations:runStatusLabel", {
                    defaultValue: "Status"
                  })}
                  {": "}
                </Text>
                <StatusBadge status={runDetail.status} />
              </div>
              <div>
                <Text type="secondary">
                  {t("evaluations:runModelLabelShort", {
                    defaultValue: "Target model"
                  })}
                  {": "}
                </Text>
                <Text>{runDetail.target_model}</Text>
              </div>
              <div className="flex items-center gap-2">
                <Text type="secondary">
                  {t("evaluations:runEvalIdLabel", {
                    defaultValue: "Evaluation"
                  })}
                  {": "}
                </Text>
                <Text>{runDetail.eval_id}</Text>
                <CopyButton text={runDetail.eval_id} />
              </div>

              {runMetrics.length > 0 && (
                <div className="mt-3">
                  <MetricsChart metrics={runMetrics} />
                </div>
              )}

              {runDetail.error_message && (
                <div>
                  <Text type="secondary">
                    {t("evaluations:runErrorLabel", {
                      defaultValue: "Error"
                    })}
                    {": "}
                  </Text>
                  <Text type="danger">{runDetail.error_message}</Text>
                </div>
              )}

              {runDetail.results && (
                <div className="mt-2">
                  <Text type="secondary">
                    {t("evaluations:runResultsLabel", {
                      defaultValue: "Results (snippet)"
                    })}
                  </Text>
                  <pre className="mt-1 max-h-40 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text">
                    {JSON.stringify(runDetail.results, null, 2)}
                  </pre>
                </div>
              )}

              {runDetail.progress && (
                <div className="mt-2">
                  <Text type="secondary">
                    {t("evaluations:runProgressLabel", {
                      defaultValue: "Progress"
                    })}
                  </Text>
                  <pre className="mt-1 max-h-32 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text">
                    {JSON.stringify(runDetail.progress, null, 2)}
                  </pre>
                </div>
              )}

              {runDetail.usage && (
                <div className="mt-2">
                  <Text type="secondary">
                    {t("evaluations:runUsageLabel", {
                      defaultValue: "Usage"
                    })}
                  </Text>
                  <pre className="mt-1 max-h-32 overflow-auto rounded bg-surface2 p-2 text-[11px] text-text">
                    {JSON.stringify(runDetail.usage, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          ) : null}
        </Card>

        {/* Run Comparison */}
        <Card
          title={t("evaluations:compareTitle", {
            defaultValue: "Compare runs"
          })}
        >
          <Space className="mb-3" size="small" wrap>
            <Select
              placeholder={t("evaluations:compareRunAPlaceholder", {
                defaultValue: "Select Run A"
              })}
              value={compareRunAId}
              onChange={(value) => setCompareRunAId(value)}
              options={runs.map((run) => ({
                value: run.id,
                label: t("evaluations:runLabel", {
                  defaultValue: "Run {{id}}",
                  id: run.id
                }) as string
              }))}
              allowClear
              style={{ minWidth: 160 }}
            />
            <Select
              placeholder={t("evaluations:compareRunBPlaceholder", {
                defaultValue: "Select Run B"
              })}
              value={compareRunBId}
              onChange={(value) => setCompareRunBId(value)}
              options={runs.map((run) => ({
                value: run.id,
                label: t("evaluations:runLabel", {
                  defaultValue: "Run {{id}}",
                  id: run.id
                }) as string
              }))}
              allowClear
              style={{ minWidth: 160 }}
            />
          </Space>
          <RunComparisonView runA={compareRunA} runB={compareRunB} />
        </Card>
      </div>
    </div>
  </div>
  )
}

export default RunsTab
