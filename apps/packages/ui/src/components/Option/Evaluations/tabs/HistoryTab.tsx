/**
 * HistoryTab component
 * Tab for viewing evaluation history
 */

import React from "react"
import {
  Button,
  Card,
  Divider,
  Form,
  Input,
  Select,
  Spin,
  Tag,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import { useFetchHistory, historyTypePresets } from "../hooks/useHistory"
import { useEvaluationsStore } from "@/store/evaluations"
import { CopyButton } from "../components"
import { EvaluationRecoveryCallout } from "../components/EvaluationRecoveryCallout"

const { Text } = Typography

export const HistoryTab: React.FC = () => {
  const { t } = useTranslation(["evaluations", "common"])
  const [form] = Form.useForm()

  // Store state
  const historyResults = useEvaluationsStore((s) => s.historyResults)
  const historyTotalCount = useEvaluationsStore((s) => s.historyTotalCount)

  // Mutations
  const fetchHistoryMutation = useFetchHistory()

  const handleFetch = () => {
    const values = form.getFieldsValue()
    fetchHistoryMutation.mutate(values)
  }

  return (
    <div className="space-y-4">
      <Card
        title={t("evaluations:historyTitle", {
          defaultValue: "History"
        })}
      >
        <Form form={form} layout="vertical" size="small">
          <div className="grid gap-4 md:grid-cols-2">
            <Form.Item
              label={t("evaluations:historyTypeLabel", {
                defaultValue: "Type"
              })}
              name="evaluation_type"
            >
              <Select
                allowClear
                placeholder={t("evaluations:historyTypePlaceholder", {
                  defaultValue: "Filter by evaluation type"
                })}
                options={historyTypePresets}
              />
            </Form.Item>
            <Form.Item
              label={t("evaluations:historyUserLabel", {
                defaultValue: "User ID"
              })}
              name="user_id"
            >
              <Input
                data-testid="history-user-filter"
                placeholder={t("evaluations:historyUserPlaceholder", {
                  defaultValue: "user_123"
                })}
              />
            </Form.Item>
            <Form.Item
              label={t("evaluations:historyStartLabel", {
                defaultValue: "Start date (ISO)"
              })}
              name="start_date"
            >
              <Input
                placeholder={t("evaluations:historyStartPlaceholder", {
                  defaultValue: "2024-01-01T00:00:00Z"
                })}
                type="datetime-local"
              />
            </Form.Item>
            <Form.Item
              label={t("evaluations:historyEndLabel", {
                defaultValue: "End date (ISO)"
              })}
              name="end_date"
            >
              <Input
                placeholder={t("evaluations:historyEndPlaceholder", {
                  defaultValue: "2024-12-31T23:59:59Z"
                })}
                type="datetime-local"
              />
            </Form.Item>
          </div>
          <Button
            type="primary"
            loading={fetchHistoryMutation.isPending}
            onClick={handleFetch}
            data-eval-tour="fetch-history"
          >
            {t("evaluations:historyFetchCta", {
              defaultValue: "Fetch history"
            })}
          </Button>
        </Form>

        <Divider />

        {fetchHistoryMutation.isError ? (
          <EvaluationRecoveryCallout
            title={t("evaluations:historyFetchErrorTitle", {
              defaultValue: "Unable to fetch history"
            })}
            endpoint="/api/v1/evaluations/history"
            error={fetchHistoryMutation.error}
          />
        ) : fetchHistoryMutation.isPending ? (
          <div className="flex justify-center py-4">
            <Spin size="small" />
          </div>
        ) : historyResults.length === 0 ? (
          <Text type="secondary" className="text-xs">
            {t("evaluations:historyEmpty", {
              defaultValue: "Run a query to see recent activity."
            })}
          </Text>
        ) : (
          <div className="space-y-2">
            <Text type="secondary" className="text-xs">
              {t("evaluations:historyResultsCount", {
                defaultValue: "{{count}} results",
                count: historyTotalCount
              })}
            </Text>
            <div className="flex flex-col gap-2">
              {historyResults.map((item) => {
                const typeLabel =
                  item.eval_type ||
                  item.evaluation_type ||
                  item.type ||
                  "unknown"
                const evalId = item.eval_id || item.evaluation_id || item.id
                const userId = item.user_id || item.created_by

                return (
                  <Card
                    key={item.id}
                    size="small"
                    className="hover:border-primary/70"
                    styles={{ body: { padding: "8px 12px" } }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <Tag color="blue" className="text-xs">
                            {typeLabel}
                          </Tag>
                          <Text type="secondary" className="text-[11px]">
                            {item.created_at || ""}
                          </Text>
                        </div>
                        <div className="flex flex-wrap gap-2 mt-1">
                          {evalId && (
                            <div className="flex items-center gap-1">
                              <Text type="secondary" className="text-[11px]">
                                {t("evaluations:historyEvalLabel", {
                                  defaultValue: "Eval"
                                })}
                                :
                              </Text>
                              <code className="text-[11px]">{evalId}</code>
                              <CopyButton text={evalId} />
                            </div>
                          )}
                          {item.run_id && (
                            <div className="flex items-center gap-1">
                              <Text type="secondary" className="text-[11px]">
                                {t("evaluations:historyRunLabel", {
                                  defaultValue: "Run"
                                })}
                                :
                              </Text>
                              <code className="text-[11px]">{item.run_id}</code>
                              <CopyButton text={item.run_id} />
                            </div>
                          )}
                          {userId && (
                            <div className="flex items-center gap-1">
                              <Text type="secondary" className="text-[11px]">
                                {t("evaluations:historyUserShortLabel", {
                                  defaultValue: "User"
                                })}
                                :
                              </Text>
                              <code className="text-[11px]">{userId}</code>
                            </div>
                          )}
                        </div>
                        {item.detail && Object.keys(item.detail).length > 0 && (
                          <pre className="mt-2 max-h-24 overflow-auto rounded bg-surface2 p-2 text-[10px] text-text">
                            {JSON.stringify(item.detail, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  </Card>
                )
              })}
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

export default HistoryTab
