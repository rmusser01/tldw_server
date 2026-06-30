import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Drawer, Form, Input, notification, Skeleton, Collapse, Radio } from "antd"
import React, { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import {
  createPrompt,
  updatePrompt,
  getPrompt,
  previewPromptDefinition,
  type PromptCreatePayload,
  type PromptUpdatePayload,
  type FewShotExample,
  type PromptModule,
  type PromptFormat,
  type StructuredPromptDefinition,
  type StructuredPromptPreviewResponse,
  type StructuredPromptPreviewRequest
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import { StructuredPromptEditor } from "../../Structured/StructuredPromptEditor"
import {
  convertLegacyPromptToStructuredDefinition,
  createDefaultStructuredPromptDefinition,
  renderStructuredPromptLegacySnapshot
} from "../../structured-prompt-utils"

type PromptEditorDrawerProps = {
  open: boolean
  promptId: number | null
  projectId: number
  onClose: () => void
}

type FormValues = {
  name: string
  system_prompt?: string
  user_prompt?: string
  change_description?: string
  few_shot_examples?: string // JSON string for few-shot examples
  modules_config?: string // JSON string for modules config
}

export const PromptEditorDrawer: React.FC<PromptEditorDrawerProps> = ({
  open,
  promptId,
  projectId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const queryClient = useQueryClient()
  const [promptFormat, setPromptFormat] = useState<PromptFormat>("legacy")
  const [structuredDefinition, setStructuredDefinition] =
    useState<StructuredPromptDefinition>(createDefaultStructuredPromptDefinition())
  const [hasStructuredDraft, setHasStructuredDraft] = useState(false)
  const projectedLegacySnapshotRef = React.useRef<{
    system_prompt: string
    user_prompt: string
  } | null>(null)
  const [previewResult, setPreviewResult] =
    useState<StructuredPromptPreviewResponse | null>(null)

  const isEditing = promptId !== null

  // Fetch existing prompt for editing
  const { data: promptResponse, isLoading: isLoadingPrompt } = useQuery({
    queryKey: ["prompt-studio", "prompt", promptId],
    queryFn: () => getPrompt(promptId!),
    enabled: open && isEditing
  })

  const existingPrompt = (promptResponse as any)?.data?.data

  // Set form values when editing
  useEffect(() => {
    if (open && isEditing && existingPrompt) {
      form.setFieldsValue({
        name: existingPrompt.name,
        system_prompt: existingPrompt.system_prompt || "",
        user_prompt: existingPrompt.user_prompt || "",
        change_description: "",
        few_shot_examples: existingPrompt.few_shot_examples
          ? JSON.stringify(existingPrompt.few_shot_examples, null, 2)
          : "",
        modules_config: existingPrompt.modules_config
          ? JSON.stringify(existingPrompt.modules_config, null, 2)
          : ""
      })
      setPromptFormat(existingPrompt.prompt_format || "legacy")
      setStructuredDefinition(
        existingPrompt.prompt_definition ||
          convertLegacyPromptToStructuredDefinition(
            existingPrompt.system_prompt,
            existingPrompt.user_prompt
          )
      )
      setHasStructuredDraft(existingPrompt.prompt_format === "structured")
      projectedLegacySnapshotRef.current = null
      setPreviewResult(null)
    } else if (open && !isEditing) {
      form.resetFields()
      form.setFieldsValue({
        name: "",
        system_prompt: "",
        user_prompt: "",
        change_description: "",
        few_shot_examples: "",
        modules_config: ""
      })
      setPromptFormat("legacy")
      setStructuredDefinition(createDefaultStructuredPromptDefinition())
      setHasStructuredDraft(false)
      projectedLegacySnapshotRef.current = null
      setPreviewResult(null)
    }
  }, [open, isEditing, existingPrompt, form])

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (values: PromptCreatePayload) =>
      createPrompt(values, crypto.randomUUID()),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompts", projectId]
      })
      notification.success({
        message: t("managePrompts.studio.prompts.createSuccess", {
          defaultValue: "Prompt created"
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

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (values: PromptUpdatePayload) =>
      updatePrompt(promptId!, values),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompts", projectId]
      })
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompt", promptId]
      })
      notification.success({
        message: t("managePrompts.studio.prompts.updateSuccess", {
          defaultValue: "Prompt updated"
        }),
        description: t("managePrompts.studio.prompts.newVersionCreated", {
          defaultValue: "A new version has been created."
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

  const previewMutation = useMutation({
    mutationFn: (payload: StructuredPromptPreviewRequest) =>
      previewPromptDefinition(payload),
    onSuccess: (response) => {
      setPreviewResult((response as any)?.data?.data || null)
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const parseJsonField = <T,>(value: string | undefined, defaultValue: T): T => {
    if (!value?.trim()) return defaultValue
    try {
      return JSON.parse(value)
    } catch {
      return defaultValue
    }
  }

  const handleSubmit = (values: FormValues) => {
    const fewShotExamples = parseJsonField<FewShotExample[] | null>(
      values.few_shot_examples,
      null
    )
    const modulesConfig = parseJsonField<PromptModule[] | null>(
      values.modules_config,
      null
    )

    if (isEditing) {
      const payload: PromptUpdatePayload = {
        name: values.name.trim(),
        system_prompt:
          promptFormat === "legacy" ? values.system_prompt?.trim() || null : null,
        user_prompt:
          promptFormat === "legacy" ? values.user_prompt?.trim() || null : null,
        prompt_format: promptFormat,
        prompt_schema_version:
          promptFormat === "structured"
            ? Number((structuredDefinition as any)?.schema_version || 1)
            : null,
        prompt_definition:
          promptFormat === "structured" ? structuredDefinition : null,
        change_description:
          values.change_description?.trim() || "Updated prompt",
        few_shot_examples: fewShotExamples,
        modules_config: modulesConfig
      }
      updateMutation.mutate(payload)
    } else {
      const payload: PromptCreatePayload = {
        project_id: projectId,
        name: values.name.trim(),
        system_prompt:
          promptFormat === "legacy" ? values.system_prompt?.trim() || null : null,
        user_prompt:
          promptFormat === "legacy" ? values.user_prompt?.trim() || null : null,
        prompt_format: promptFormat,
        prompt_schema_version:
          promptFormat === "structured"
            ? Number((structuredDefinition as any)?.schema_version || 1)
            : null,
        prompt_definition:
          promptFormat === "structured" ? structuredDefinition : null,
        change_description: values.change_description?.trim() || "Initial version",
        few_shot_examples: fewShotExamples,
        modules_config: modulesConfig
      }
      createMutation.mutate(payload)
    }
  }

  const handlePreview = (variables: Record<string, string>) => {
    const values = form.getFieldsValue()
    const fewShotExamples = parseJsonField<FewShotExample[] | null>(
      values.few_shot_examples,
      null
    )
    const modulesConfig = parseJsonField<PromptModule[] | null>(
      values.modules_config,
      null
    )

    previewMutation.mutate({
      project_id: projectId,
      signature_id: existingPrompt?.signature_id ?? null,
      prompt_format: promptFormat,
      system_prompt:
        promptFormat === "legacy" ? values.system_prompt?.trim() || null : null,
      user_prompt:
        promptFormat === "legacy" ? values.user_prompt?.trim() || null : null,
      prompt_schema_version:
        promptFormat === "structured"
          ? Number((structuredDefinition as any)?.schema_version || 1)
          : null,
      prompt_definition:
        promptFormat === "structured" ? structuredDefinition : null,
      few_shot_examples: fewShotExamples,
      modules_config: modulesConfig,
      variables
    })
  }

  const validateJson = (_: any, value: string) => {
    if (!value?.trim()) return Promise.resolve()
    try {
      JSON.parse(value)
      return Promise.resolve()
    } catch (e) {
      return Promise.reject(
        new Error(
          t("managePrompts.studio.prompts.invalidJson", {
            defaultValue: "Invalid JSON format"
          })
        )
      )
    }
  }

  const isPending = createMutation.isPending || updateMutation.isPending

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={
        isEditing
          ? t("managePrompts.studio.prompts.editTitle", {
              defaultValue: "Edit Prompt"
            })
          : t("managePrompts.studio.prompts.createTitle", {
              defaultValue: "Create Prompt"
            })
      }
      styles={{ wrapper: { width: promptFormat === "structured" ? 1120 : 640 } }}
      destroyOnHidden
      footer={
        <div className="flex justify-end gap-2">
          <Button type="secondary" onClick={onClose} disabled={isPending}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            onClick={() => form.submit()}
            loading={isPending}
          >
            {isEditing
              ? t("common:save", { defaultValue: "Save" })
              : t("managePrompts.studio.prompts.createBtn", {
                  defaultValue: "Create Prompt"
                })}
          </Button>
        </div>
      }
    >
      {isEditing && isLoadingPrompt ? (
        <Skeleton paragraph={{ rows: 8 }} />
      ) : (
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            name="name"
            label={t("managePrompts.studio.prompts.form.name", {
              defaultValue: "Prompt Name"
            })}
            rules={[
              {
                required: true,
                message: t("managePrompts.studio.prompts.form.nameRequired", {
                  defaultValue: "Please enter a prompt name"
                })
              }
            ]}
          >
            <Input
              placeholder={t(
                "managePrompts.studio.prompts.form.namePlaceholder",
                {
                  defaultValue: "e.g., Customer Support Response"
                }
              )}
              autoFocus
            />
          </Form.Item>

          <div className="mb-6">
            <div className="mb-2 text-sm font-medium text-text">
              Authoring mode
            </div>
            <Radio.Group
              value={promptFormat}
              onChange={(event) => {
                const nextFormat = event.target.value as PromptFormat
                if (nextFormat === promptFormat) {
                  return
                }

                if (nextFormat === "legacy") {
                  const legacySnapshot =
                    renderStructuredPromptLegacySnapshot(structuredDefinition)
                  projectedLegacySnapshotRef.current = {
                    system_prompt: legacySnapshot.systemPrompt,
                    user_prompt: legacySnapshot.userPrompt
                  }
                  form.setFieldsValue({
                    system_prompt: legacySnapshot.systemPrompt,
                    user_prompt: legacySnapshot.userPrompt
                  })
                } else {
                  const currentValues = form.getFieldsValue()
                  const matchesProjectedLegacy =
                    projectedLegacySnapshotRef.current !== null &&
                    currentValues.system_prompt ===
                      projectedLegacySnapshotRef.current.system_prompt &&
                    currentValues.user_prompt ===
                      projectedLegacySnapshotRef.current.user_prompt

                  if (!hasStructuredDraft || !matchesProjectedLegacy) {
                    setStructuredDefinition(
                      convertLegacyPromptToStructuredDefinition(
                        currentValues.system_prompt,
                        currentValues.user_prompt
                      )
                    )
                  }
                  projectedLegacySnapshotRef.current = null
                  setHasStructuredDraft(true)
                }

                setPromptFormat(nextFormat)
                setPreviewResult(null)
              }}
            >
              <Radio.Button value="legacy">Legacy text</Radio.Button>
              <Radio.Button value="structured">Structured builder</Radio.Button>
            </Radio.Group>
          </div>

          {promptFormat === "legacy" ? (
            <>
              <Form.Item
                name="system_prompt"
                label={t("managePrompts.studio.prompts.form.systemPrompt", {
                  defaultValue: "System Prompt"
                })}
                tooltip={t("managePrompts.studio.prompts.form.systemPromptHelp", {
                  defaultValue:
                    "Instructions that set the behavior and context for the AI"
                })}
              >
                <Input.TextArea
                  placeholder={t(
                    "managePrompts.studio.prompts.form.systemPromptPlaceholder",
                    {
                      defaultValue:
                        "You are a helpful customer support agent..."
                    }
                  )}
                  rows={5}
                  showCount
                  maxLength={10000}
                />
              </Form.Item>

              <Form.Item
                name="user_prompt"
                label={t("managePrompts.studio.prompts.form.userPrompt", {
                  defaultValue: "User Prompt Template"
                })}
                tooltip={t("managePrompts.studio.prompts.form.userPromptHelp", {
                  defaultValue:
                    "Template for user messages. Use {{variable}} for inputs."
                })}
              >
                <Input.TextArea
                  placeholder={t(
                    "managePrompts.studio.prompts.form.userPromptPlaceholder",
                    {
                      defaultValue:
                        "Please help the customer with: {{customer_query}}"
                    }
                  )}
                  rows={5}
                  showCount
                  maxLength={10000}
                />
              </Form.Item>
            </>
          ) : (
            <StructuredPromptEditor
              value={structuredDefinition}
              onChange={(nextValue) => {
                setStructuredDefinition(nextValue)
                setHasStructuredDraft(true)
                setPreviewResult(null)
              }}
              previewResult={previewResult}
              previewLoading={previewMutation.isPending}
              onPreview={handlePreview}
            />
          )}

          {isEditing && (
            <Form.Item
              name="change_description"
              label={t("managePrompts.studio.prompts.form.changeDescription", {
                defaultValue: "Change Description"
              })}
              rules={[
                {
                  required: true,
                  message: t(
                    "managePrompts.studio.prompts.form.changeDescriptionRequired",
                    {
                      defaultValue:
                        "Please describe what you changed (required for version history)"
                    }
                  )
                }
              ]}
            >
              <Input
                placeholder={t(
                  "managePrompts.studio.prompts.form.changeDescriptionPlaceholder",
                  {
                    defaultValue: "e.g., Improved tone for friendlier responses"
                  }
                )}
              />
            </Form.Item>
          )}

          <Collapse
            ghost
            items={[
              {
                key: "advanced",
                label: t("managePrompts.studio.prompts.form.advancedOptions", {
                  defaultValue: "Advanced Options"
                }),
                children: (
                  <div className="space-y-4">
                    <Alert
                      variant="info"
                    >
                      {t(
                        "managePrompts.studio.prompts.form.advancedInfo",
                        {
                          defaultValue:
                            "These fields accept JSON. Few-shot examples help the model learn from examples."
                        }
                      )}
                    </Alert>

                    <Form.Item
                      name="few_shot_examples"
                      label={t(
                        "managePrompts.studio.prompts.form.fewShotExamples",
                        {
                          defaultValue: "Few-Shot Examples (JSON)"
                        }
                      )}
                      rules={[{ validator: validateJson }]}
                      tooltip={t(
                        "managePrompts.studio.prompts.form.fewShotHelp",
                        {
                          defaultValue:
                            'Array of {inputs: {...}, outputs: {...}, explanation?: "..."}'
                        }
                      )}
                    >
                      <Input.TextArea
                        placeholder={`[
  {
    "inputs": {"query": "How do I reset my password?"},
    "outputs": {"response": "To reset your password..."},
    "explanation": "Shows helpful step-by-step response"
  }
]`}
                        rows={6}
                        className="font-mono text-sm"
                      />
                    </Form.Item>

                    <Form.Item
                      name="modules_config"
                      label={t(
                        "managePrompts.studio.prompts.form.modulesConfig",
                        {
                          defaultValue: "Modules Configuration (JSON)"
                        }
                      )}
                      rules={[{ validator: validateJson }]}
                      tooltip={t(
                        "managePrompts.studio.prompts.form.modulesHelp",
                        {
                          defaultValue:
                            "Configure special modules like Chain-of-Thought, ReAct, etc."
                        }
                      )}
                    >
                      <Input.TextArea
                        placeholder={`[
  {
    "type": "chain_of_thought",
    "enabled": true,
    "config": {"style": "step_by_step"}
  }
]`}
                        rows={6}
                        className="font-mono text-sm"
                      />
                    </Form.Item>
                  </div>
                )
              }
            ]}
          />
        </Form>
      )}
    </Drawer>
  )
}
