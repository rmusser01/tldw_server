import { Form, Input, InputNumber, Select, Switch } from "antd"
import type { FormInstance } from "antd"
import { useTranslation } from "react-i18next"

type GroupedModelOption = {
  label: React.ReactNode
  options: Array<{
    label: React.ReactNode
    value: string
    searchLabel: string
  }>
}

interface ModelBasicsTabProps {
  form: FormInstance
  selectedModel: string | null
  onModelChange: (value: string | null | undefined) => void
  modelOptions: GroupedModelOption[]
  modelsLoading: boolean
  isOCREnabled?: boolean
  ocrLanguage?: string
  ocrLanguages: Array<{ value: string; label: string }>
  onOcrLanguageChange?: (value: string) => void
}

export function ModelBasicsTab({
  selectedModel,
  onModelChange,
  modelOptions,
  modelsLoading,
  isOCREnabled,
  ocrLanguage,
  ocrLanguages,
  onOcrLanguageChange
}: ModelBasicsTabProps) {
  const { t } = useTranslation("common")

  return (
    <div className="space-y-4">
      {isOCREnabled && (
        <div className="flex flex-col space-y-2 mb-3">
          <span className="text-text">OCR Language</span>
          <Select
            showSearch
            style={{ width: "100%" }}
            options={ocrLanguages}
            value={ocrLanguage}
            filterOption={(input, option) =>
              option!.label.toLowerCase().indexOf(input.toLowerCase()) >= 0 ||
              option!.value.toLowerCase().indexOf(input.toLowerCase()) >= 0
            }
            onChange={onOcrLanguageChange}
          />
        </div>
      )}

      <Form.Item
        label={t("modelSettings.form.model.label", { defaultValue: "API / model" })}
        help={t("modelSettings.form.model.help", {
          defaultValue: "Changing the API/model applies immediately. Save applies your parameter edits."
        })}>
        <Select
          showSearch
          value={selectedModel || undefined}
          onChange={(value) => onModelChange(value)}
          placeholder={t("playground:composer.modelPlaceholder", "API / model")}
          options={modelOptions as any}
          loading={modelsLoading}
          allowClear
          optionLabelProp="label"
          popupMatchSelectWidth={false}
          styles={{
            popup: {
              root: {
                maxHeight: "calc(100vh - 220px)",
                overflowY: "auto"
              }
            }
          }}
          listHeight={560}
          filterOption={(input, option) => {
            const normalizedInput = input.toLowerCase()
            const rawSearchLabel =
              (option as any)?.searchLabel ??
              (typeof option?.label === "string" ? option.label : "")
            const normalizedLabel = String(rawSearchLabel).toLowerCase()
            return normalizedLabel.includes(normalizedInput)
          }}
        />
      </Form.Item>

      <Form.Item
        name="temperature"
        label={t("modelSettings.form.temperature.label")}>
        <InputNumber
          style={{ width: "100%" }}
          placeholder={t("modelSettings.form.temperature.placeholder")}
        />
      </Form.Item>

      <Form.Item
        name="numCtx"
        label={t("modelSettings.form.numCtx.label")}>
        <InputNumber
          style={{ width: "100%" }}
          placeholder={t("modelSettings.form.numCtx.placeholder")}
        />
      </Form.Item>

      <Form.Item
        name="numPredict"
        label={t("modelSettings.form.numPredict.label")}>
        <InputNumber
          style={{ width: "100%" }}
          placeholder={t("modelSettings.form.numPredict.placeholder")}
        />
      </Form.Item>

      <Form.Item
        name="keepAlive"
        help={t("modelSettings.form.keepAlive.help")}
        label={t("modelSettings.form.keepAlive.label")}>
        <Input placeholder={t("modelSettings.form.keepAlive.placeholder")} />
      </Form.Item>

      <Form.Item
        name="seed"
        help={t("modelSettings.form.seed.help")}
        label={t("modelSettings.form.seed.label")}>
        <InputNumber
          style={{ width: "100%" }}
          placeholder={t("modelSettings.form.seed.placeholder")}
        />
      </Form.Item>

      <Form.Item
        name="jsonMode"
        label={t("modelSettings.form.jsonMode.label", "JSON Mode")}
        help={t(
          "modelSettings.form.jsonMode.help",
          "Force the model to output valid JSON. Only works with models that support structured output."
        )}
        valuePropName="checked">
        <Switch />
      </Form.Item>
    </div>
  )
}
