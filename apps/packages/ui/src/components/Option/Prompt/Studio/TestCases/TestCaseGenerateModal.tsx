import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Modal, Form, InputNumber, Select, notification, Spin } from "antd"
import { Sparkles } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import {
  generateTestCases,
  listPrompts,
  type GenerateTestCasesPayload,
  type Prompt
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { Alert as DsAlert } from "@/components/ui/primitives"

type TestCaseGenerateModalProps = {
  open: boolean
  projectId: number
  onClose: () => void
}

type FormValues = {
  prompt_id?: number
  count: number
  provider?: string
  model?: string
}

export const TestCaseGenerateModal: React.FC<TestCaseGenerateModalProps> = ({
  open,
  projectId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const queryClient = useQueryClient()

  // Fetch prompts for selection
  const { data: promptsResponse } = useQuery({
    queryKey: ["prompt-studio", "prompts", projectId],
    queryFn: () => listPrompts(projectId, { per_page: 100 }),
    enabled: open && projectId !== null
  })

  const prompts: Prompt[] = (promptsResponse as any)?.data?.data ?? []

  // Generate mutation
  const generateMutation = useMutation({
    mutationFn: (payload: GenerateTestCasesPayload) =>
      generateTestCases(payload),
    onSuccess: (response) => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "test-cases", projectId]
      })
      const count = (response as any)?.data?.data?.length ?? 0
      notification.success({
        message: t("managePrompts.studio.testCases.generateSuccess", {
          defaultValue: "Generated {{count}} test cases",
          count
        })
      })
      onClose()
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleSubmit = (values: FormValues) => {
    const payload: GenerateTestCasesPayload = {
      project_id: projectId,
      prompt_id: values.prompt_id,
      count: values.count,
      provider: values.provider,
      model: values.model
    }
    generateMutation.mutate(payload)
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={
        <span className="flex items-center gap-2">
          <Sparkles className="size-5" />
          {t("managePrompts.studio.testCases.generateTitle", {
            defaultValue: "Generate Test Cases"
          })}
        </span>
      }
      footer={null}
      destroyOnHidden
    >
      <div className="mt-4 space-y-4">
        <DsAlert
          variant="info"
          title={t("managePrompts.studio.testCases.generateInfo", {
            defaultValue:
              "Use AI to automatically generate test cases based on your prompt structure."
          })}
        />

        {generateMutation.isPending ? (
          <div className="py-8 flex flex-col items-center">
            <Spin size="large" />
            <p className="mt-4 text-text-muted">
              {t("managePrompts.studio.testCases.generating", {
                defaultValue: "Generating test cases..."
              })}
            </p>
          </div>
        ) : (
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSubmit}
            initialValues={{ count: 5 }}
          >
            <Form.Item
              name="prompt_id"
              label={t("managePrompts.studio.testCases.form.promptReference", {
                defaultValue: "Based on Prompt (optional)"
              })}
              tooltip={t(
                "managePrompts.studio.testCases.form.promptReferenceHelp",
                {
                  defaultValue:
                    "Select a prompt to generate test cases that match its input structure."
                }
              )}
            >
              <Select
                placeholder={t(
                  "managePrompts.studio.testCases.form.selectPrompt",
                  {
                    defaultValue: "Select a prompt..."
                  }
                )}
                allowClear
                options={prompts.map((p) => ({
                  label: p.name,
                  value: p.id
                }))}
              />
            </Form.Item>

            <Form.Item
              name="count"
              label={t("managePrompts.studio.testCases.form.count", {
                defaultValue: "Number of Test Cases"
              })}
              rules={[
                {
                  required: true,
                  message: t(
                    "managePrompts.studio.testCases.form.countRequired",
                    {
                      defaultValue: "Please specify how many test cases to generate"
                    }
                  )
                }
              ]}
            >
              <InputNumber min={1} max={50} style={{ width: "100%" }} />
            </Form.Item>

            <div className="grid grid-cols-2 gap-4">
              <Form.Item
                name="provider"
                label={t("managePrompts.studio.testCases.form.provider", {
                  defaultValue: "Provider (optional)"
                })}
              >
                <Select
                  placeholder={t(
                    "managePrompts.studio.testCases.form.providerPlaceholder",
                    {
                      defaultValue: "Use default"
                    }
                  )}
                  allowClear
                  options={[
                    { label: "OpenAI", value: "openai" },
                    { label: "Anthropic", value: "anthropic" },
                    { label: "Ollama", value: "ollama" }
                  ]}
                />
              </Form.Item>

              <Form.Item
                name="model"
                label={t("managePrompts.studio.testCases.form.model", {
                  defaultValue: "Model (optional)"
                })}
              >
                <Select
                  placeholder={t(
                    "managePrompts.studio.testCases.form.modelPlaceholder",
                    {
                      defaultValue: "Use default"
                    }
                  )}
                  allowClear
                  options={[
                    { label: "GPT-4o", value: "gpt-4o" },
                    { label: "GPT-4o mini", value: "gpt-4o-mini" },
                    { label: "Claude 3.5 Sonnet", value: "claude-3-5-sonnet-latest" }
                  ]}
                />
              </Form.Item>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button type="secondary" onClick={onClose}>
                {t("common:cancel", { defaultValue: "Cancel" })}
              </Button>
              <Button type="primary" htmlType="submit">
                <Sparkles className="size-4 mr-1" />
                {t("managePrompts.studio.testCases.generateBtn", {
                  defaultValue: "Generate"
                })}
              </Button>
            </div>
          </Form>
        )}
      </div>
    </Modal>
  )
}
