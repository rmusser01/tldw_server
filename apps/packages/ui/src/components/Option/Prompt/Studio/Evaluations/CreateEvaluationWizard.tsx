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
  Checkbox
} from "antd"
import { BarChart3 } from "lucide-react"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore, type WizardStep } from "@/store/prompt-studio"
import {
  createEvaluation,
  listPrompts,
  listTestCases,
  type Prompt,
  type TestCase,
  type EvaluationCreatePayload,
  type EvaluationConfig
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { Alert as DsAlert } from "@/components/ui/primitives"

type CreateEvaluationWizardProps = {
  open: boolean
  projectId: number
  onClose: () => void
}

type FormValues = {
  name?: string
  description?: string
  prompt_id: number
  test_case_ids: number[]
  model_name?: string
  temperature?: number
  max_tokens?: number
  run_async?: boolean
}

const WIZARD_STEPS: WizardStep[] = [
  "selectPrompt",
  "selectTestCases",
  "configureModel",
  "review"
]

export const CreateEvaluationWizard: React.FC<CreateEvaluationWizardProps> = ({
  open,
  projectId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const queryClient = useQueryClient()

  const wizardStep = usePromptStudioStore((s) => s.wizardStep)
  const setWizardStep = usePromptStudioStore((s) => s.setWizardStep)
  const resetWizard = usePromptStudioStore((s) => s.resetWizard)

  const [selectedPromptId, setSelectedPromptId] = useState<number | null>(null)
  const [selectedTestCaseIds, setSelectedTestCaseIds] = useState<number[]>([])
  const [selectAllTestCases, setSelectAllTestCases] = useState(false)

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

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (payload: EvaluationCreatePayload) => createEvaluation(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "evaluations", projectId]
      })
      notification.success({
        message: t("managePrompts.studio.evaluations.createSuccess", {
          defaultValue: "Evaluation started"
        }),
        description: t("managePrompts.studio.evaluations.createSuccessDesc", {
          defaultValue: "The evaluation is now running in the background."
        })
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
    setSelectAllTestCases(false)
    form.resetFields()
    onClose()
  }

  const handleSubmit = () => {
    const values = form.getFieldsValue()

    const config: EvaluationConfig = {}
    if (values.model_name) config.model_name = values.model_name
    if (values.temperature !== undefined) config.temperature = values.temperature
    if (values.max_tokens !== undefined) config.max_tokens = values.max_tokens

    const testCaseIdsToUse = selectAllTestCases
      ? testCases.map((tc) => tc.id)
      : selectedTestCaseIds

    const payload: EvaluationCreatePayload = {
      project_id: projectId,
      prompt_id: selectedPromptId!,
      test_case_ids: testCaseIdsToUse,
      name: values.name || undefined,
      description: values.description || undefined,
      config: Object.keys(config).length > 0 ? config : undefined,
      run_async: true
    }

    createMutation.mutate(payload)
  }

  const canProceedToTestCases = selectedPromptId !== null
  const canProceedToConfig =
    selectedTestCaseIds.length > 0 || selectAllTestCases
  const canSubmit = canProceedToTestCases && canProceedToConfig

  const selectedPrompt = prompts.find((p) => p.id === selectedPromptId)

  const renderStepContent = () => {
    switch (wizardStep) {
      case "selectPrompt":
        return (
          <div className="space-y-4">
            <DsAlert
              variant="info"
              title={t(
                "managePrompts.studio.evaluations.wizard.selectPromptInfo",
                {
                  defaultValue:
                    "Select the prompt you want to evaluate. The evaluation will run this prompt against your test cases."
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
                  title: t("managePrompts.studio.evaluations.wizard.promptName", {
                    defaultValue: "Name"
                  }),
                  dataIndex: "name",
                  key: "name"
                },
                {
                  title: t("managePrompts.studio.evaluations.wizard.version", {
                    defaultValue: "Version"
                  }),
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
            <DsAlert
              variant="info"
              title={t(
                "managePrompts.studio.evaluations.wizard.selectTestCasesInfo",
                {
                  defaultValue:
                    "Select which test cases to include in this evaluation."
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
              {t("managePrompts.studio.evaluations.wizard.selectAll", {
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
                    title: t(
                      "managePrompts.studio.evaluations.wizard.testCaseName",
                      {
                        defaultValue: "Name"
                      }
                    ),
                    key: "name",
                    render: (_, record) =>
                      record.name || `Test Case #${record.id}`
                  },
                  {
                    title: t(
                      "managePrompts.studio.evaluations.wizard.golden",
                      {
                        defaultValue: "Golden"
                      }
                    ),
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
          <div className="space-y-4">
            <DsAlert
              variant="info"
              title={t(
                "managePrompts.studio.evaluations.wizard.configureModelInfo",
                {
                  defaultValue:
                    "Configure the model settings for this evaluation. Leave blank to use defaults."
                }
              )}
            />
            <Form form={form} layout="vertical">
              <Form.Item
                name="name"
                label={t("managePrompts.studio.evaluations.wizard.evalName", {
                  defaultValue: "Evaluation Name (optional)"
                })}
              >
                <Input
                  placeholder={t(
                    "managePrompts.studio.evaluations.wizard.evalNamePlaceholder",
                    {
                      defaultValue: "e.g., GPT-4o baseline"
                    }
                  )}
                />
              </Form.Item>

              <Form.Item
                name="model_name"
                label={t("managePrompts.studio.evaluations.wizard.model", {
                  defaultValue: "Model"
                })}
              >
                <Select
                  placeholder={t(
                    "managePrompts.studio.evaluations.wizard.modelPlaceholder",
                    {
                      defaultValue: "Use default"
                    }
                  )}
                  allowClear
                  options={[
                    { label: "GPT-4o", value: "gpt-4o" },
                    { label: "GPT-4o mini", value: "gpt-4o-mini" },
                    { label: "GPT-3.5 Turbo", value: "gpt-3.5-turbo" },
                    {
                      label: "Claude 3.5 Sonnet",
                      value: "claude-3-5-sonnet-latest"
                    },
                    { label: "Claude 3 Haiku", value: "claude-3-haiku-20240307" }
                  ]}
                />
              </Form.Item>

              <div className="grid grid-cols-2 gap-4">
                <Form.Item
                  name="temperature"
                  label={t(
                    "managePrompts.studio.evaluations.wizard.temperature",
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

                <Form.Item
                  name="max_tokens"
                  label={t(
                    "managePrompts.studio.evaluations.wizard.maxTokens",
                    {
                      defaultValue: "Max Tokens"
                    }
                  )}
                >
                  <InputNumber
                    min={1}
                    max={128000}
                    style={{ width: "100%" }}
                    placeholder="2048"
                  />
                </Form.Item>
              </div>
            </Form>
          </div>
        )

      case "review":
        return (
          <div className="space-y-4">
            <DsAlert
              variant="success"
              title={t(
                "managePrompts.studio.evaluations.wizard.reviewInfo",
                {
                  defaultValue: "Review your evaluation settings before running."
                }
              )}
            />
            <div className="p-4 bg-surface2 rounded-md space-y-3">
              <div>
                <span className="text-text-muted text-sm">
                  {t("managePrompts.studio.evaluations.wizard.selectedPrompt", {
                    defaultValue: "Prompt:"
                  })}
                </span>
                <p className="font-medium">
                  {selectedPrompt?.name} (v{selectedPrompt?.version_number})
                </p>
              </div>
              <div>
                <span className="text-text-muted text-sm">
                  {t("managePrompts.studio.evaluations.wizard.testCaseCount", {
                    defaultValue: "Test Cases:"
                  })}
                </span>
                <p className="font-medium">
                  {selectAllTestCases
                    ? `All (${testCases.length})`
                    : selectedTestCaseIds.length}
                </p>
              </div>
              <div>
                <span className="text-text-muted text-sm">
                  {t("managePrompts.studio.evaluations.wizard.modelConfig", {
                    defaultValue: "Model:"
                  })}
                </span>
                <p className="font-medium">
                  {form.getFieldValue("model_name") || "Default"}
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
          <BarChart3 className="size-5" />
          {t("managePrompts.studio.evaluations.wizard.title", {
            defaultValue: "Run Evaluation"
          })}
        </span>
      }
      width={700}
      footer={null}
      destroyOnHidden
    >
      <div className="mt-4 space-y-6">
        <Steps
          current={currentStepIndex}
          size="small"
          items={[
            {
              title: t("managePrompts.studio.evaluations.wizard.step1", {
                defaultValue: "Select Prompt"
              })
            },
            {
              title: t("managePrompts.studio.evaluations.wizard.step2", {
                defaultValue: "Test Cases"
              })
            },
            {
              title: t("managePrompts.studio.evaluations.wizard.step3", {
                defaultValue: "Configure"
              })
            },
            {
              title: t("managePrompts.studio.evaluations.wizard.step4", {
                defaultValue: "Review"
              })
            }
          ]}
        />

        <div className="min-h-[300px]">{renderStepContent()}</div>

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
                {t("managePrompts.studio.evaluations.wizard.runBtn", {
                  defaultValue: "Run Evaluation"
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
    </Modal>
  )
}
