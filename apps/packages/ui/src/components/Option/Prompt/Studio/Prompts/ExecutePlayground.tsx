import { useMutation, useQuery } from "@tanstack/react-query"
import { Drawer, Form, Input, Select, Skeleton, notification, Spin } from "antd"
import { Play, Clock, Zap } from "lucide-react"
import React, { useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import {
  getPrompt,
  executePrompt,
  getLlmProviders,
  type ExecutePromptPayload
} from "@/services/prompt-studio"
import { Button } from "@/components/Common/Button"
import {
  getExecuteDefaultModel,
  getExecuteDefaultProvider,
  getExecuteModelOptions,
  getExecuteProviderOptions,
  isValidExecuteModel,
  isValidExecuteProvider,
  normalizeExecuteProvidersCatalog
} from "./execute-playground-provider-utils"

type ExecutePlaygroundProps = {
  open: boolean
  promptId: number | null
  onClose: () => void
}

type FormValues = {
  provider?: string
  model?: string
  inputs: string // JSON string
}

export const ExecutePlayground: React.FC<ExecutePlaygroundProps> = ({
  open,
  promptId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm<FormValues>()
  const [output, setOutput] = useState<string | null>(null)
  const [executionStats, setExecutionStats] = useState<{
    tokens?: number
    time?: number
  } | null>(null)

  // Fetch prompt details
  const { data: promptResponse, isLoading: isLoadingPrompt } = useQuery({
    queryKey: ["prompt-studio", "prompt", promptId],
    queryFn: () => getPrompt(promptId!),
    enabled: open && promptId !== null
  })

  const {
    data: providersResponse,
    isLoading: isLoadingProviders
  } = useQuery({
    queryKey: ["prompt-studio", "llm-providers"],
    queryFn: () => getLlmProviders(),
    enabled: open
  })

  const prompt = (promptResponse as any)?.data?.data
  const providersPayload = (providersResponse as any)?.data ?? providersResponse
  const providersCatalog = useMemo(
    () => normalizeExecuteProvidersCatalog(providersPayload),
    [providersPayload]
  )
  const providerOptions = useMemo(
    () => getExecuteProviderOptions(providersCatalog),
    [providersCatalog]
  )
  const selectedProvider = Form.useWatch("provider", form)
  const selectedModel = Form.useWatch("model", form)
  const fallbackProvider = getExecuteDefaultProvider(providersCatalog)
  const effectiveProvider = selectedProvider || fallbackProvider
  const modelOptions = useMemo(
    () => getExecuteModelOptions(providersCatalog, effectiveProvider),
    [providersCatalog, effectiveProvider]
  )
  const fallbackModel = getExecuteDefaultModel(providersCatalog, effectiveProvider)

  // Reset output when prompt changes
  useEffect(() => {
    if (open) {
      setOutput(null)
      setExecutionStats(null)
      form.setFieldsValue({
        inputs: "{}",
        provider: undefined,
        model: undefined
      })
    }
  }, [open, promptId, form])

  useEffect(() => {
    if (!selectedModel) return
    if (!isValidExecuteModel(providersCatalog, effectiveProvider, selectedModel)) {
      form.setFieldValue("model", undefined)
    }
  }, [selectedModel, providersCatalog, effectiveProvider, form])

  // Extract variables from user_prompt template
  const extractVariables = (template?: string | null): string[] => {
    if (!template) return []
    const matches = template.match(/\{\{(\w+)\}\}/g) || []
    return matches.map((m) => m.replace(/\{\{|\}\}/g, ""))
  }

  const variables = extractVariables(prompt?.user_prompt)

  // Execute mutation
  const executeMutation = useMutation({
    mutationFn: (payload: ExecutePromptPayload) => executePrompt(payload),
    onSuccess: (response) => {
      const data = (response as any)?.data
      setOutput(data?.output || "No output")
      setExecutionStats({
        tokens: data?.tokens_used,
        time: data?.execution_time
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
      setOutput(null)
      setExecutionStats(null)
    }
  })

  const handleExecute = (values: FormValues) => {
    if (!promptId) return

    if (!isValidExecuteProvider(providersCatalog, values.provider)) {
      notification.error({
        message: t("managePrompts.studio.prompts.invalidProvider", {
          defaultValue: "Provider is not available on this server"
        })
      })
      return
    }

    if (
      !isValidExecuteModel(
        providersCatalog,
        values.provider || fallbackProvider,
        values.model
      )
    ) {
      notification.error({
        message: t("managePrompts.studio.prompts.invalidModel", {
          defaultValue: "Model is not available for the selected provider"
        })
      })
      return
    }

    let inputs: Record<string, any> = {}
    try {
      inputs = JSON.parse(values.inputs || "{}")
    } catch {
      notification.error({
        message: t("managePrompts.studio.prompts.invalidInputs", {
          defaultValue: "Invalid inputs JSON"
        })
      })
      return
    }

    const payload: ExecutePromptPayload = {
      prompt_id: promptId,
      inputs,
      provider: values.provider || undefined,
      model: values.model || undefined
    }

    executeMutation.mutate(payload)
  }

  const generateInputTemplate = () => {
    const template: Record<string, string> = {}
    variables.forEach((v) => {
      template[v] = ""
    })
    form.setFieldValue("inputs", JSON.stringify(template, null, 2))
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <Play className="size-5" />
          {t("managePrompts.studio.prompts.executeTitle", {
            defaultValue: "Execute Prompt"
          })}
        </span>
      }
      size={600}
      destroyOnHidden
    >
      {isLoadingPrompt && <Skeleton paragraph={{ rows: 8 }} />}

      {!isLoadingPrompt && prompt && (
        <div className="space-y-4">
          {/* Prompt info */}
          <div className="p-3 bg-surface2 rounded-md">
            <p className="font-medium">{prompt.name}</p>
            <p className="text-sm text-text-muted">
              Version {prompt.version_number}
            </p>
          </div>

          {/* Variables hint */}
          {variables.length > 0 && (
            <Alert
              variant="info"
              title={t("managePrompts.studio.prompts.variablesDetected", {
                defaultValue: "Variables detected in prompt"
              })}
            >
              <div className="mt-1">
                <p className="text-sm mb-2">
                  {t("managePrompts.studio.prompts.variablesList", {
                    defaultValue: "Found variables: {{variables}}",
                    variables: variables.join(", ")
                  })}
                </p>
                <Button type="ghost" size="sm" onClick={generateInputTemplate}>
                  {t("managePrompts.studio.prompts.generateTemplate", {
                    defaultValue: "Generate input template"
                  })}
                </Button>
              </div>
            </Alert>
          )}

          <Form form={form} layout="vertical" onFinish={handleExecute}>
            <Form.Item
              name="inputs"
              label={t("managePrompts.studio.prompts.form.inputs", {
                defaultValue: "Inputs (JSON)"
              })}
              initialValue="{}"
              rules={[
                {
                  validator: (_, value) => {
                    try {
                      JSON.parse(value || "{}")
                      return Promise.resolve()
                    } catch {
                      return Promise.reject(
                        new Error(
                          t("managePrompts.studio.prompts.invalidJson", {
                            defaultValue: "Invalid JSON format"
                          })
                        )
                      )
                    }
                  }
                }
              ]}
            >
              <Input.TextArea
                rows={4}
                className="font-mono text-sm"
                placeholder={`{\n  "variable_name": "value"\n}`}
              />
            </Form.Item>

            <div className="grid grid-cols-2 gap-4">
              <Form.Item
                name="provider"
                label={t("managePrompts.studio.prompts.form.provider", {
                  defaultValue: "Provider (optional)"
                })}
                rules={[
                  {
                    validator: (_, value) =>
                      isValidExecuteProvider(providersCatalog, value)
                        ? Promise.resolve()
                        : Promise.reject(
                            new Error(
                              t("managePrompts.studio.prompts.invalidProvider", {
                                defaultValue:
                                  "Provider is not available on this server"
                              })
                            )
                          )
                  }
                ]}
              >
                <Select
                  placeholder={t(
                    "managePrompts.studio.prompts.form.providerPlaceholder",
                    {
                      defaultValue: fallbackProvider
                        ? `Default: ${fallbackProvider}`
                        : "Use server default"
                    }
                  )}
                  allowClear
                  showSearch
                  optionFilterProp="label"
                  loading={isLoadingProviders}
                  options={providerOptions}
                />
              </Form.Item>

              <Form.Item
                name="model"
                label={t("managePrompts.studio.prompts.form.model", {
                  defaultValue: "Model (optional)"
                })}
                dependencies={["provider"]}
                rules={[
                  ({ getFieldValue }) => ({
                    validator: (_, value) =>
                      isValidExecuteModel(
                        providersCatalog,
                        getFieldValue("provider") || fallbackProvider,
                        value
                      )
                        ? Promise.resolve()
                        : Promise.reject(
                            new Error(
                              t("managePrompts.studio.prompts.invalidModel", {
                                defaultValue:
                                  "Model is not available for the selected provider"
                              })
                            )
                          )
                  })
                ]}
              >
                <Select
                  allowClear
                  showSearch
                  optionFilterProp="label"
                  options={modelOptions}
                  loading={isLoadingProviders}
                  placeholder={t(
                    "managePrompts.studio.prompts.form.modelPlaceholder",
                    {
                      defaultValue: fallbackModel
                        ? `Default: ${fallbackModel}`
                        : "Use server default"
                    }
                  )}
                  disabled={modelOptions.length === 0 && !fallbackModel}
                />
              </Form.Item>
            </div>

            <p className="text-xs text-text-muted">
              {t("managePrompts.studio.prompts.defaultsHint", {
                defaultValue:
                  "Leave provider/model empty to run with server defaults."
              })}
            </p>

            <Button
              type="primary"
              htmlType="submit"
              loading={executeMutation.isPending}
              className="w-full"
            >
              <Play className="size-4 mr-1" />
              {t("managePrompts.studio.prompts.executeBtn", {
                defaultValue: "Execute"
              })}
            </Button>
          </Form>

          {/* Output section */}
          {(executeMutation.isPending || output !== null) && (
            <div className="mt-6">
              <h4 className="font-medium mb-2">
                {t("managePrompts.studio.prompts.output", {
                  defaultValue: "Output"
                })}
              </h4>

              {executeMutation.isPending ? (
                <div className="p-6 bg-surface2 rounded-md flex items-center justify-center">
                  <Spin size="default" />
                  <span className="ml-2 text-text-muted">
                    {t("managePrompts.studio.prompts.executing", {
                      defaultValue: "Executing..."
                    })}
                  </span>
                </div>
              ) : (
                <>
                  {/* Stats */}
                  {executionStats && (
                    <div className="flex items-center gap-4 mb-2 text-sm text-text-muted">
                      {executionStats.time !== undefined && (
                        <span className="flex items-center gap-1">
                          <Clock className="size-4" />
                          {executionStats.time.toFixed(2)}s
                        </span>
                      )}
                      {executionStats.tokens !== undefined && (
                        <span className="flex items-center gap-1">
                          <Zap className="size-4" />
                          {executionStats.tokens} tokens
                        </span>
                      )}
                    </div>
                  )}

                  {/* Output content */}
                  <div className="p-4 bg-surface2 rounded-md">
                    <pre className="whitespace-pre-wrap text-sm font-mono">
                      {output}
                    </pre>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </Drawer>
  )
}
