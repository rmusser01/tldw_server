import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Modal,
  Steps,
  Form,
  Input,
  Select,
  InputNumber,
  Table,
  notification,
  Checkbox,
  Alert,
  Card,
  Radio
} from "antd"
import {
  Sparkles,
  Zap,
  GitBranch,
  Shuffle,
  TrendingUp,
  Target,
  Scale
} from "lucide-react"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore, type WizardStep } from "@/store/prompt-studio"
import {
  createOptimization,
  listPrompts,
  listTestCases,
  getOptimizationStrategies,
  type Prompt,
  type TestCase,
  type OptimizationCreatePayload,
  type OptimizationStrategy,
  type StrategyInfo
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { CompareStrategiesModal } from "./CompareStrategiesModal"
import { defaultOptimizationStrategies } from "./strategyMetadata"

type CreateOptimizationWizardProps = {
  open: boolean
  projectId: number
  onClose: () => void
}

type FormValues = {
  name?: string
  description?: string
  prompt_id: number
  strategy: OptimizationStrategy
  max_iterations?: number
  model_name?: string
  temperature?: number
}

const WIZARD_STEPS: WizardStep[] = [
  "selectPrompt",
  "selectTestCases",
  "configureModel",
  "review"
]

const strategyIcons: Record<string, React.ReactNode> = {
  iterative: <Zap className="size-5" />,
  mipro: <Target className="size-5" />,
  bootstrap: <TrendingUp className="size-5" />,
  genetic: <GitBranch className="size-5" />,
  beam_search: <Shuffle className="size-5" />,
  random_search: <Shuffle className="size-5" />
}

export const CreateOptimizationWizard: React.FC<
  CreateOptimizationWizardProps
> = ({ open, projectId, onClose }) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const queryClient = useQueryClient()

  const wizardStep = usePromptStudioStore((s) => s.wizardStep)
  const setWizardStep = usePromptStudioStore((s) => s.setWizardStep)
  const resetWizard = usePromptStudioStore((s) => s.resetWizard)

  const [selectedPromptId, setSelectedPromptId] = useState<number | null>(null)
  const [selectedTestCaseIds, setSelectedTestCaseIds] = useState<number[]>([])
  const [selectAllTestCases, setSelectAllTestCases] = useState(true)
  const [selectedStrategy, setSelectedStrategy] =
    useState<OptimizationStrategy>("iterative")
  const [compareModalOpen, setCompareModalOpen] = useState(false)

  // Fetch prompts
  const { data: promptsResponse } = useQuery({
    queryKey: ["prompt-studio", "prompts", projectId],
    queryFn: () => listPrompts(projectId, { per_page: 100 }),
    enabled: open
  })
  const prompts: Prompt[] = (promptsResponse as any)?.data?.data ?? []

  // Fetch test cases
  const { data: testCasesResponse } = useQuery({
    queryKey: ["prompt-studio", "test-cases", projectId],
    queryFn: () => listTestCases(projectId, { per_page: 100 }),
    enabled: open
  })
  const testCases: TestCase[] = (testCasesResponse as any)?.data?.data ?? []

  // Fetch available strategies (fallback to defaults)
  const { data: strategiesResponse } = useQuery({
    queryKey: ["prompt-studio", "optimization-strategies"],
    queryFn: () => getOptimizationStrategies(),
    enabled: open
  })
  const strategies: StrategyInfo[] =
    (strategiesResponse as any)?.data?.data ?? defaultOptimizationStrategies

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (payload: OptimizationCreatePayload) =>
      createOptimization(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "optimizations", projectId]
      })
      notification.success({
        message: t("managePrompts.studio.optimizations.createSuccess", {
          defaultValue: "Optimization started"
        }),
        description: t(
          "managePrompts.studio.optimizations.createSuccessDesc",
          {
            defaultValue: "The optimization job is now running."
          }
        )
      })
      handleClose()
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const currentStepIndex = WIZARD_STEPS.indexOf(wizardStep)

  const handleNext = () => {
    const nextIndex = currentStepIndex + 1
    if (nextIndex < WIZARD_STEPS.length) {
      setWizardStep(WIZARD_STEPS[nextIndex])
    }
  }

  const handlePrev = () => {
    const prevIndex = currentStepIndex - 1
    if (prevIndex >= 0) {
      setWizardStep(WIZARD_STEPS[prevIndex])
    }
  }

  const handleClose = () => {
    resetWizard()
    setSelectedPromptId(null)
    setSelectedTestCaseIds([])
    setSelectAllTestCases(true)
    setSelectedStrategy("iterative")
    form.resetFields()
    onClose()
  }

  const handleSubmit = () => {
    const values = form.getFieldsValue()

    const testCaseIdsToUse = selectAllTestCases
      ? testCases.map((tc) => tc.id)
      : selectedTestCaseIds

    const strategyInfo = strategies.find((s) => s.name === selectedStrategy)
    const defaultParams = strategyInfo?.default_params ?? {}

    const payload: OptimizationCreatePayload = {
      project_id: projectId,
      prompt_id: selectedPromptId!,
      name: values.name || undefined,
      description: values.description || undefined,
      config: {
        strategy: selectedStrategy,
        max_iterations: values.max_iterations ?? defaultParams.max_iterations ?? 10,
        ...defaultParams
      },
      model_config: values.model_name
        ? {
            model_name: values.model_name,
            temperature: values.temperature
          }
        : undefined,
      test_case_ids: testCaseIdsToUse
    }

    createMutation.mutate(payload)
  }

  const canProceedToTestCases = selectedPromptId !== null
  const canProceedToConfig =
    selectedTestCaseIds.length > 0 || selectAllTestCases
  const canSubmit = canProceedToTestCases && canProceedToConfig

  const selectedPrompt = prompts.find((p) => p.id === selectedPromptId)
  const selectedStrategyInfo = strategies.find((s) => s.name === selectedStrategy)

  const renderStepContent = () => {
    switch (wizardStep) {
      case "selectPrompt":
        return (
          <div className="space-y-4">
            <Alert
              type="info"
              showIcon
              title={t(
                "managePrompts.studio.optimizations.wizard.selectPromptInfo",
                {
                  defaultValue:
                    "Select the prompt you want to optimize. The optimization will create improved versions of this prompt."
                }
              )}
            />
            <Table<Prompt>
              dataSource={prompts}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ y: 300 }}
              rowSelection={{
                type: "radio",
                selectedRowKeys: selectedPromptId ? [selectedPromptId] : [],
                onChange: (keys) =>
                  setSelectedPromptId(keys[0] as number | null)
              }}
              columns={[
                {
                  title: "Name",
                  dataIndex: "name",
                  key: "name"
                },
                {
                  title: "Version",
                  dataIndex: "version_number",
                  key: "version",
                  width: 80,
                  render: (v) => `v${v}`
                }
              ]}
            />
          </div>
        )

      case "selectTestCases":
        return (
          <div className="space-y-4">
            <Alert
              type="info"
              showIcon
              title={t(
                "managePrompts.studio.optimizations.wizard.selectTestCasesInfo",
                {
                  defaultValue:
                    "Test cases are used to evaluate each prompt iteration. More test cases = better optimization but slower."
                }
              )}
            />
            <Checkbox
              checked={selectAllTestCases}
              onChange={(e) => {
                setSelectAllTestCases(e.target.checked)
                if (e.target.checked) {
                  setSelectedTestCaseIds([])
                }
              }}
            >
              {t("managePrompts.studio.optimizations.wizard.selectAll", {
                defaultValue: "Use all test cases ({{count}})",
                count: testCases.length
              })}
            </Checkbox>

            {!selectAllTestCases && (
              <Table<TestCase>
                dataSource={testCases}
                rowKey="id"
                size="small"
                pagination={false}
                scroll={{ y: 250 }}
                rowSelection={{
                  selectedRowKeys: selectedTestCaseIds,
                  onChange: (keys) =>
                    setSelectedTestCaseIds(keys as number[])
                }}
                columns={[
                  {
                    title: "Name",
                    key: "name",
                    render: (_, record) =>
                      record.name || `Test Case #${record.id}`
                  },
                  {
                    title: "Golden",
                    key: "is_golden",
                    width: 80,
                    render: (_, record) =>
                      record.is_golden ? (
                        <span className="text-warn">Yes</span>
                      ) : (
                        "No"
                      )
                  }
                ]}
              />
            )}
          </div>
        )

      case "configureModel":
        return (
          <div className="space-y-6">
            {/* Strategy selection */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium">
                  {t("managePrompts.studio.optimizations.wizard.selectStrategy", {
                    defaultValue: "Select Strategy"
                  })}
                </h4>
                <Button
                  type="secondary"
                  size="small"
                  onClick={() => setCompareModalOpen(true)}
                  icon={<Scale className="size-4" />}
                >
                  {t("managePrompts.studio.optimizations.compareStrategies", {
                    defaultValue: "Compare Strategies"
                  })}
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {strategies.slice(0, 6).map((strategy) => (
                  <Card
                    key={strategy.name}
                    size="small"
                    className={`cursor-pointer transition-all ${
                      selectedStrategy === strategy.name
                        ? "border-primary bg-primary/5"
                        : "hover:border-primary/50"
                    }`}
                    onClick={() =>
                      setSelectedStrategy(strategy.name as OptimizationStrategy)
                    }
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`p-2 rounded-md ${
                          selectedStrategy === strategy.name
                            ? "bg-primary/20 text-primary"
                            : "bg-surface2 text-text-muted"
                        }`}
                      >
                        {strategyIcons[strategy.name] || (
                          <Sparkles className="size-5" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm">
                          {strategy.display_name}
                        </p>
                        <p className="text-xs text-text-muted line-clamp-2">
                          {strategy.description}
                        </p>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            {/* Configuration form */}
            <Form form={form} layout="vertical">
              <Form.Item
                name="name"
                label={t("managePrompts.studio.optimizations.wizard.optName", {
                  defaultValue: "Name (optional)"
                })}
              >
                <Input
                  placeholder={t(
                    "managePrompts.studio.optimizations.wizard.optNamePlaceholder",
                    {
                      defaultValue: "e.g., Iterative optimization run 1"
                    }
                  )}
                />
              </Form.Item>

              <Form.Item
                name="max_iterations"
                label={t(
                  "managePrompts.studio.optimizations.wizard.maxIterations",
                  {
                    defaultValue: "Max Iterations"
                  }
                )}
                initialValue={selectedStrategyInfo?.default_params?.max_iterations ?? 10}
              >
                <InputNumber min={1} max={100} style={{ width: "100%" }} />
              </Form.Item>

              <div className="grid grid-cols-2 gap-4">
                <Form.Item
                  name="model_name"
                  label={t(
                    "managePrompts.studio.optimizations.wizard.model",
                    {
                      defaultValue: "Model (optional)"
                    }
                  )}
                >
                  <Select
                    placeholder="Use default"
                    allowClear
                    options={[
                      { label: "GPT-4o", value: "gpt-4o" },
                      { label: "GPT-4o mini", value: "gpt-4o-mini" },
                      {
                        label: "Claude 3.5 Sonnet",
                        value: "claude-3-5-sonnet-latest"
                      }
                    ]}
                  />
                </Form.Item>

                <Form.Item
                  name="temperature"
                  label={t(
                    "managePrompts.studio.optimizations.wizard.temperature",
                    {
                      defaultValue: "Temperature"
                    }
                  )}
                >
                  <InputNumber
                    min={0}
                    max={2}
                    step={0.1}
                    style={{ width: "100%" }}
                    placeholder="0.7"
                  />
                </Form.Item>
              </div>
            </Form>
          </div>
        )

      case "review":
        return (
          <div className="space-y-4">
            <Alert
              type="success"
              showIcon
              title={t(
                "managePrompts.studio.optimizations.wizard.reviewInfo",
                {
                  defaultValue:
                    "Review your optimization settings before starting."
                }
              )}
            />
            <div className="p-4 bg-surface2 rounded-md space-y-3">
              <div>
                <span className="text-text-muted text-sm">Prompt:</span>
                <p className="font-medium">
                  {selectedPrompt?.name} (v{selectedPrompt?.version_number})
                </p>
              </div>
              <div>
                <span className="text-text-muted text-sm">Test Cases:</span>
                <p className="font-medium">
                  {selectAllTestCases
                    ? `All (${testCases.length})`
                    : selectedTestCaseIds.length}
                </p>
              </div>
              <div>
                <span className="text-text-muted text-sm">Strategy:</span>
                <p className="font-medium">
                  {selectedStrategyInfo?.display_name}
                </p>
              </div>
              <div>
                <span className="text-text-muted text-sm">Max Iterations:</span>
                <p className="font-medium">
                  {form.getFieldValue("max_iterations") ?? 10}
                </p>
              </div>
            </div>
          </div>
        )
    }
  }

  return (
    <Modal
      open={open}
      onCancel={handleClose}
      title={
        <span className="flex items-center gap-2">
          <Sparkles className="size-5" />
          {t("managePrompts.studio.optimizations.wizard.title", {
            defaultValue: "Start Optimization"
          })}
        </span>
      }
      width={750}
      footer={null}
      destroyOnHidden
    >
      <div className="mt-4 space-y-6">
        <Steps
          current={currentStepIndex}
          size="small"
          items={[
            { title: "Select Prompt" },
            { title: "Test Cases" },
            { title: "Configure" },
            { title: "Review" }
          ]}
        />

        <div className="min-h-[350px]">{renderStepContent()}</div>

        <div className="flex justify-between pt-4 border-t border-border">
          <Button
            type="secondary"
            onClick={handlePrev}
            disabled={currentStepIndex === 0}
          >
            {t("common:back", { defaultValue: "Back" })}
          </Button>

          <div className="flex gap-2">
            <Button type="secondary" onClick={handleClose}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>

            {wizardStep === "review" ? (
              <Button
                type="primary"
                onClick={handleSubmit}
                loading={createMutation.isPending}
                disabled={!canSubmit}
              >
                {t("managePrompts.studio.optimizations.wizard.startBtn", {
                  defaultValue: "Start Optimization"
                })}
              </Button>
            ) : (
              <Button
                type="primary"
                onClick={handleNext}
                disabled={
                  (wizardStep === "selectPrompt" && !canProceedToTestCases) ||
                  (wizardStep === "selectTestCases" && !canProceedToConfig)
                }
              >
                {t("common:next", { defaultValue: "Next" })}
              </Button>
            )}
          </div>
        </div>
      </div>

      <CompareStrategiesModal
        open={compareModalOpen}
        onClose={() => setCompareModalOpen(false)}
        onSelectStrategy={setSelectedStrategy}
        selectedStrategy={selectedStrategy}
      />
    </Modal>
  )
}
