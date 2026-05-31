import { useMutation, useQuery } from "@tanstack/react-query"
import { Modal, Select, notification, Spin, Table } from "antd"
import { Play, CheckCircle2, XCircle, Clock } from "lucide-react"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import {
  runTestCases,
  listPrompts,
  type Prompt,
  type TestRunResult
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { Alert as DsAlert, Badge } from "@/components/ui/primitives"

type TestCaseRunModalProps = {
  open: boolean
  projectId: number
  testCaseIds: number[]
  onClose: () => void
}

export const TestCaseRunModal: React.FC<TestCaseRunModalProps> = ({
  open,
  projectId,
  testCaseIds,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [selectedPromptId, setSelectedPromptId] = useState<number | null>(null)
  const [results, setResults] = useState<TestRunResult[] | null>(null)

  // Fetch prompts for selection
  const { data: promptsResponse } = useQuery({
    queryKey: ["prompt-studio", "prompts", projectId],
    queryFn: () => listPrompts(projectId, { per_page: 100 }),
    enabled: open && projectId !== null
  })

  const prompts: Prompt[] = (promptsResponse as any)?.data?.data ?? []

  // Run mutation
  const runMutation = useMutation({
    mutationFn: () =>
      runTestCases({
        prompt_id: selectedPromptId!,
        test_case_ids: testCaseIds
      }),
    onSuccess: (response) => {
      const data = (response as any)?.data?.data ?? []
      setResults(data)
      notification.success({
        message: t("managePrompts.studio.testCases.runComplete", {
          defaultValue: "Test run complete"
        })
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleRun = () => {
    if (!selectedPromptId) {
      notification.warning({
        message: t("managePrompts.studio.testCases.selectPromptFirst", {
          defaultValue: "Please select a prompt first"
        })
      })
      return
    }
    setResults(null)
    runMutation.mutate()
  }

  const handleClose = () => {
    setResults(null)
    setSelectedPromptId(null)
    onClose()
  }

  const passCount = results?.filter((r) => r.passed).length ?? 0
  const failCount = results?.filter((r) => r.passed === false).length ?? 0

  return (
    <Modal
      open={open}
      onCancel={handleClose}
      title={
        <span className="flex items-center gap-2">
          <Play className="size-5" />
          {t("managePrompts.studio.testCases.runTitle", {
            defaultValue: "Run Test Cases"
          })}
        </span>
      }
      width={800}
      footer={null}
      destroyOnHidden
    >
      <div className="mt-4 space-y-4">
        {!results && (
          <>
            <DsAlert
              variant="info"
              title={t("managePrompts.studio.testCases.runInfo", {
                defaultValue:
                  "Run {{count}} test cases against a prompt to see the outputs.",
                count: testCaseIds.length
              })}
            />

            <div>
              <label className="block text-sm font-medium mb-2">
                {t("managePrompts.studio.testCases.selectPromptToRun", {
                  defaultValue: "Select Prompt to Run Against"
                })}
              </label>
              <Select
                placeholder={t(
                  "managePrompts.studio.testCases.form.selectPrompt",
                  {
                    defaultValue: "Select a prompt..."
                  }
                )}
                value={selectedPromptId}
                onChange={(v) => setSelectedPromptId(v)}
                style={{ width: "100%" }}
                options={prompts.map((p) => ({
                  label: `${p.name} (v${p.version_number})`,
                  value: p.id
                }))}
              />
            </div>
          </>
        )}

        {runMutation.isPending && (
          <div className="py-8 flex flex-col items-center">
            <Spin size="large" />
            <p className="mt-4 text-text-muted">
              {t("managePrompts.studio.testCases.running", {
                defaultValue: "Running test cases..."
              })}
            </p>
          </div>
        )}

        {results && (
          <div className="space-y-4">
            {/* Summary */}
            <div className="flex items-center gap-4 p-3 bg-surface2 rounded-md">
              <span className="text-sm font-medium">
                {t("managePrompts.studio.testCases.runSummary", {
                  defaultValue: "Results"
                })}
              </span>
              <Badge variant="success" size="sm">
                <CheckCircle2 className="size-3" aria-hidden="true" />
                {passCount} passed
              </Badge>
              {failCount > 0 && (
                <Badge variant="danger" size="sm">
                  <XCircle className="size-3" aria-hidden="true" />
                  {failCount} failed
                </Badge>
              )}
            </div>

            {/* Results table */}
            <Table<TestRunResult>
              dataSource={results}
              rowKey="test_case_id"
              size="small"
              pagination={false}
              scroll={{ y: 400 }}
              columns={[
                {
                  title: t("managePrompts.studio.testCases.columns.testCase", {
                    defaultValue: "Test Case"
                  }),
                  dataIndex: "test_case_id",
                  width: 100,
                  render: (id) => `#${id}`
                },
                {
                  title: t("managePrompts.studio.testCases.columns.status", {
                    defaultValue: "Status"
                  }),
                  key: "status",
                  width: 80,
                  render: (_, record) =>
                    record.error ? (
                      <Badge variant="danger" size="sm">
                        <XCircle className="size-3" aria-hidden="true" />
                        Error
                      </Badge>
                    ) : record.passed ? (
                      <Badge variant="success" size="sm">
                        <CheckCircle2 className="size-3" aria-hidden="true" />
                        Pass
                      </Badge>
                    ) : record.passed === false ? (
                      <Badge variant="warning" size="sm">
                        <XCircle className="size-3" aria-hidden="true" />
                        Fail
                      </Badge>
                    ) : (
                      <Badge variant="secondary" size="sm">
                        Run
                      </Badge>
                    )
                },
                {
                  title: t("managePrompts.studio.testCases.columns.output", {
                    defaultValue: "Output"
                  }),
                  dataIndex: "output",
                  render: (output, record) =>
                    record.error ? (
                      <span className="text-danger text-sm">{record.error}</span>
                    ) : (
                      <pre className="text-xs whitespace-pre-wrap max-h-20 overflow-auto">
                        {output?.substring(0, 200)}
                        {output && output.length > 200 && "..."}
                      </pre>
                    )
                },
                {
                  title: t("managePrompts.studio.testCases.columns.time", {
                    defaultValue: "Time"
                  }),
                  key: "time",
                  width: 80,
                  render: (_, record) =>
                    record.execution_time ? (
                      <span className="flex items-center gap-1 text-xs text-text-muted">
                        <Clock className="size-3" />
                        {record.execution_time.toFixed(2)}s
                      </span>
                    ) : (
                      "-"
                    )
                }
              ]}
            />
          </div>
        )}

        <div className="flex justify-end gap-2 pt-4 border-t border-border">
          <Button type="secondary" onClick={handleClose}>
            {t("common:close", { defaultValue: "Close" })}
          </Button>
          {!results && (
            <Button
              type="primary"
              onClick={handleRun}
              loading={runMutation.isPending}
              disabled={!selectedPromptId}
            >
              <Play className="size-4 mr-1" />
              {t("managePrompts.studio.testCases.runBtn", {
                defaultValue: "Run Tests"
              })}
            </Button>
          )}
          {results && (
            <Button type="primary" onClick={() => setResults(null)}>
              {t("managePrompts.studio.testCases.runAgain", {
                defaultValue: "Run Again"
              })}
            </Button>
          )}
        </div>
      </div>
    </Modal>
  )
}
