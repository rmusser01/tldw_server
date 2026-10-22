import { useQuery } from "@tanstack/react-query"
import { Form, Select, type FormInstance } from "antd"
import { useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { getLlmProviders } from "@/services/prompt-studio"
import {
  normalizeExecuteProvidersCatalog,
  getExecuteProviderOptions,
  getExecuteModelOptions,
  getExecuteDefaultProvider,
  getExecuteDefaultModel,
  isValidExecuteProvider,
  isValidExecuteModel
} from "./Prompts/execute-playground-provider-utils"

export const useStudioModelTarget = (open: boolean, form: FormInstance) => {
  const { t } = useTranslation("settings")
  const { data: response } = useQuery({
    queryKey: ["prompt-studio", "llm-providers"],
    queryFn: () => getLlmProviders(),
    enabled: open
  })
  const catalog = useMemo(
    () => normalizeExecuteProvidersCatalog(response?.data ?? response),
    [response]
  )
  const provider = Form.useWatch("provider", { form, preserve: true })
  const model = Form.useWatch("model_name", { form, preserve: true })
  const effectiveProvider = provider || getExecuteDefaultProvider(catalog) || undefined
  const effectiveModel = model || getExecuteDefaultModel(catalog, effectiveProvider) || undefined

  useEffect(() => {
    if (model && !isValidExecuteModel(catalog, effectiveProvider, model)) {
      form.setFieldValue("model_name", undefined)
    }
  }, [catalog, effectiveProvider, form, model])

  const resolveTarget = () => {
    const values = form.getFieldsValue(true)
    const selectedProvider = values.provider || getExecuteDefaultProvider(catalog) || undefined
    const selectedModel = values.model_name || getExecuteDefaultModel(catalog, selectedProvider) || undefined
    if (!isValidExecuteProvider(catalog, selectedProvider)) {
      throw new Error(t("managePrompts.studio.prompts.invalidProvider", {
        defaultValue: "Provider is not available on this server"
      }))
    }
    if (!isValidExecuteModel(catalog, selectedProvider, selectedModel)) {
      throw new Error(t("managePrompts.studio.prompts.invalidModel", {
        defaultValue: "Model is not available for the selected provider"
      }))
    }
    return { provider: selectedProvider, model: selectedModel }
  }

  return {
    resolveTarget,
    provider: effectiveProvider,
    model: effectiveModel,
    providerOptions: getExecuteProviderOptions(catalog),
    modelOptions: getExecuteModelOptions(catalog, effectiveProvider)
  }
}

export const StudioModelTargetFields = ({ target }: {
  target: ReturnType<typeof useStudioModelTarget>
}) => {
  const { t } = useTranslation("settings")
  return (
    <div className="grid grid-cols-2 gap-4">
      <Form.Item name="provider" label={t("managePrompts.studio.prompts.form.provider", {
        defaultValue: "Provider (optional)"
      })}>
        <Select allowClear showSearch optionFilterProp="label"
          placeholder={target.provider || "Use default"}
          options={target.providerOptions} />
      </Form.Item>
      <Form.Item name="model_name" label={t("managePrompts.studio.prompts.form.model", {
        defaultValue: "Model (optional)"
      })}>
        <Select allowClear showSearch optionFilterProp="label"
          placeholder={target.model || "Use default"}
          options={target.modelOptions} />
      </Form.Item>
    </div>
  )
}
