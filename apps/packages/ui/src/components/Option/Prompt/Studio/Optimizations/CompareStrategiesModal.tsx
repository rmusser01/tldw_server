import { useQuery } from "@tanstack/react-query"
import { Modal, Table, Tag, Spin, Alert, Tooltip } from "antd"
import type { ColumnsType } from "antd/es/table"
import {
  Zap,
  Target,
  TrendingUp,
  GitBranch,
  Shuffle,
  Sparkles,
  CheckCircle,
  XCircle
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import {
  getOptimizationStrategies,
  type OptimizationStrategy,
  type StrategyInfo
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { defaultOptimizationStrategies } from "./strategyMetadata"

type CompareStrategiesModalProps = {
  open: boolean
  onClose: () => void
  onSelectStrategy: (strategy: OptimizationStrategy) => void
  selectedStrategy?: OptimizationStrategy
}

const strategyIcons: Record<string, React.ReactNode> = {
  iterative: <Zap className="size-4" />,
  mipro: <Target className="size-4" />,
  bootstrap: <TrendingUp className="size-4" />,
  genetic: <GitBranch className="size-4" />,
  beam_search: <Shuffle className="size-4" />,
  random_search: <Shuffle className="size-4" />
}

export const CompareStrategiesModal: React.FC<CompareStrategiesModalProps> = ({
  open,
  onClose,
  onSelectStrategy,
  selectedStrategy
}) => {
  const { t } = useTranslation(["settings", "common"])

  const { data: strategiesResponse, isLoading } = useQuery({
    queryKey: ["prompt-studio", "optimization-strategies"],
    queryFn: () => getOptimizationStrategies(),
    enabled: open
  })

  const strategies: StrategyInfo[] =
    (strategiesResponse as any)?.data?.data ?? defaultOptimizationStrategies

  const columns: ColumnsType<StrategyInfo> = [
    {
      title: t("managePrompts.studio.optimizations.compare.strategy", {
        defaultValue: "Strategy"
      }),
      key: "name",
      width: 180,
      render: (_, record) => (
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-surface2 text-text-muted">
            {strategyIcons[record.name] || <Sparkles className="size-4" />}
          </div>
          <div>
            <p className="font-medium text-sm">{record.display_name}</p>
          </div>
        </div>
      )
    },
    {
      title: t("managePrompts.studio.optimizations.compare.description", {
        defaultValue: "Description"
      }),
      dataIndex: "description",
      key: "description",
      ellipsis: true
    },
    {
      title: t("managePrompts.studio.optimizations.compare.parameters", {
        defaultValue: "Parameters"
      }),
      key: "params",
      width: 200,
      render: (_, record) => (
        <div className="flex flex-wrap gap-1">
          {record.supported_params.map((param) => (
            <Tooltip
              key={param}
              title={
                record.default_params[param] !== undefined
                  ? `Default: ${record.default_params[param]}`
                  : undefined
              }
            >
              <Tag className="text-xs">{param}</Tag>
            </Tooltip>
          ))}
        </div>
      )
    },
    {
      title: t("managePrompts.studio.optimizations.compare.testCases", {
        defaultValue: "Test Cases"
      }),
      key: "requires_test_cases",
      width: 100,
      align: "center",
      render: (_, record) =>
        record.requires_test_cases ? (
          <Tooltip
            title={t(
              "managePrompts.studio.optimizations.compare.requiresTestCases",
              { defaultValue: "Requires test cases" }
            )}
          >
            <CheckCircle className="size-4 text-success mx-auto" />
          </Tooltip>
        ) : (
          <Tooltip
            title={t(
              "managePrompts.studio.optimizations.compare.noTestCasesRequired",
              { defaultValue: "No test cases required" }
            )}
          >
            <XCircle className="size-4 text-text-muted mx-auto" />
          </Tooltip>
        )
    },
    {
      title: t("managePrompts.studio.optimizations.compare.earlyStopping", {
        defaultValue: "Early Stop"
      }),
      key: "supports_early_stopping",
      width: 100,
      align: "center",
      render: (_, record) =>
        record.supports_early_stopping ? (
          <Tooltip
            title={t(
              "managePrompts.studio.optimizations.compare.supportsEarlyStopping",
              { defaultValue: "Supports early stopping" }
            )}
          >
            <CheckCircle className="size-4 text-success mx-auto" />
          </Tooltip>
        ) : (
          <Tooltip
            title={t(
              "managePrompts.studio.optimizations.compare.noEarlyStopping",
              { defaultValue: "No early stopping" }
            )}
          >
            <XCircle className="size-4 text-text-muted mx-auto" />
          </Tooltip>
        )
    },
    {
      title: t("managePrompts.studio.optimizations.compare.action", {
        defaultValue: "Action"
      }),
      key: "action",
      width: 100,
      align: "center",
      render: (_, record) => (
        <Button
          type={selectedStrategy === record.name ? "primary" : "secondary"}
          size="small"
          onClick={() => {
            onSelectStrategy(record.name as OptimizationStrategy)
            onClose()
          }}
        >
          {selectedStrategy === record.name
            ? t("common:selected", { defaultValue: "Selected" })
            : t("common:select", { defaultValue: "Select" })}
        </Button>
      )
    }
  ]

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={
        <span className="flex items-center gap-2">
          <Sparkles className="size-5" />
          {t("managePrompts.studio.optimizations.compare.title", {
            defaultValue: "Compare Optimization Strategies"
          })}
        </span>
      }
      width={900}
      footer={
        <Button type="secondary" onClick={onClose}>
          {t("common:close", { defaultValue: "Close" })}
        </Button>
      }
    >
      <div className="space-y-4">
        <Alert
          type="info"
          showIcon
          title={t("managePrompts.studio.optimizations.compare.info", {
            defaultValue:
              "Compare different optimization strategies to find the best fit for your use case. Each strategy has different strengths depending on your prompt complexity and test case availability."
          })}
        />

        {isLoading ? (
          <div className="flex justify-center py-12">
            <Spin size="large" />
          </div>
        ) : (
          <Table<StrategyInfo>
            dataSource={strategies}
            columns={columns}
            rowKey="name"
            size="small"
            pagination={false}
            scroll={{ x: 800 }}
            rowClassName={(record) =>
              selectedStrategy === record.name ? "bg-primary/5" : ""
            }
          />
        )}

        <div className="text-xs text-text-muted">
          <p>
            {t("managePrompts.studio.optimizations.compare.legend", {
              defaultValue:
                "Legend: Test Cases = strategy requires test cases for optimization. Early Stop = strategy can stop early if no improvement is detected."
            })}
          </p>
        </div>
      </div>
    </Modal>
  )
}
