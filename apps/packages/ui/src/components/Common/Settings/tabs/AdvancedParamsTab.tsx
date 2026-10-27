import { Collapse, Form, Input, InputNumber, Select, Switch } from "antd"
import type { FormInstance } from "antd"
import { useTranslation } from "react-i18next"
import { useMemo, useCallback } from "react"

import { useSelectedModelSamplingSupport } from "@/hooks/useSelectedModelSamplingSupport"

interface AdvancedParamsTabProps {
  form: FormInstance
  providerOptions: Array<{ value: string; label: string }>
  /** The chat's model, to say when Top K or Min P can't be used (#3283). */
  selectedModel?: string | null
}

export function AdvancedParamsTab({
  providerOptions,
  selectedModel
}: AdvancedParamsTabProps) {
  const { t } = useTranslation("common")
  const { topKUnavailableReason, minPUnavailableReason } =
    useSelectedModelSamplingSupport(selectedModel)

  const historyOrderOptions = useMemo(
    () => [
      {
        value: "desc",
        label: t(
          "modelSettings.form.historyMessageOrder.options.desc",
          "Newest first"
        )
      },
      {
        value: "asc",
        label: t(
          "modelSettings.form.historyMessageOrder.options.asc",
          "Oldest first"
        )
      }
    ],
    [t]
  )

  const slashInjectionOptions = useMemo(
    () => [
      {
        value: "system",
        label: t(
          "modelSettings.form.slashCommandInjectionMode.options.system",
          "System message"
        )
      },
      {
        value: "preface",
        label: t(
          "modelSettings.form.slashCommandInjectionMode.options.preface",
          "Preface user message"
        )
      },
      {
        value: "replace",
        label: t(
          "modelSettings.form.slashCommandInjectionMode.options.replace",
          "Replace user message"
        )
      }
    ],
    [t]
  )

  const validateJsonObject = useCallback(
    (_: unknown, value?: string) => {
      if (!value || value.trim().length === 0) {
        return Promise.resolve()
      }
      try {
        const parsed = JSON.parse(value)
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          return Promise.resolve()
        }
      } catch {
        // fall through
      }
      return Promise.reject(
        new Error(
          t(
            "modelSettings.form.guardrails.invalidJson",
            "Enter a valid JSON object."
          )
        )
      )
    },
    [t]
  )

  return (
    <div className="space-y-4">
      <Collapse
        ghost
        defaultActiveKey={[]}
        className="border-none bg-transparent"
        items={[
          {
            key: "sampling",
            label: t("modelSettings.advanced", "Advanced Sampling"),
            children: (
              <>
                <Form.Item
                  name="topK"
                  label={t("modelSettings.form.topK.label")}
                  extra={topKUnavailableReason}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.topK.placeholder")}
                    disabled={Boolean(topKUnavailableReason)}
                  />
                </Form.Item>

                <Form.Item
                  name="topP"
                  label={t("modelSettings.form.topP.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.topP.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="numGpu"
                  label={t("modelSettings.form.numGpu.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.numGpu.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="minP"
                  label={t("modelSettings.form.minP.label")}
                  extra={minPUnavailableReason}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.minP.placeholder")}
                    disabled={Boolean(minPUnavailableReason)}
                  />
                </Form.Item>

                <Form.Item
                  name="repeatPenalty"
                  label={t("modelSettings.form.repeatPenalty.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.repeatPenalty.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="repeatLastN"
                  label={t("modelSettings.form.repeatLastN.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.repeatLastN.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="tfsZ"
                  label={t("modelSettings.form.tfsZ.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.tfsZ.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="numKeep"
                  label={t("modelSettings.form.numKeep.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.numKeep.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="numThread"
                  label={t("modelSettings.form.numThread.label")}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.numThread.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="useMMap"
                  label={t("modelSettings.form.useMMap.label")}>
                  <Switch />
                </Form.Item>

                <Form.Item
                  name="useMlock"
                  label={t("modelSettings.form.useMlock.label")}>
                  <Switch />
                </Form.Item>

                <Form.Item
                  name="reasoningEffort"
                  label={t("modelSettings.form.reasoningEffort.label")}>
                  <Input
                    style={{ width: "100%" }}
                    placeholder={t("modelSettings.form.reasoningEffort.placeholder")}
                  />
                </Form.Item>
              </>
            )
          },
          {
            key: "request",
            label: t("modelSettings.requestOverrides", "Request Overrides"),
            children: (
              <>
                <Form.Item
                  name="historyMessageLimit"
                  label={t("modelSettings.form.historyMessageLimit.label")}
                  help={t("modelSettings.form.historyMessageLimit.help")}>
                  <InputNumber
                    min={1}
                    style={{ width: "100%" }}
                    placeholder={t(
                      "modelSettings.form.historyMessageLimit.placeholder"
                    )}
                  />
                </Form.Item>

                <Form.Item
                  name="historyMessageOrder"
                  label={t("modelSettings.form.historyMessageOrder.label")}
                  help={t("modelSettings.form.historyMessageOrder.help")}>
                  <Select
                    allowClear
                    placeholder={t(
                      "modelSettings.form.historyMessageOrder.placeholder"
                    )}
                    options={historyOrderOptions}
                  />
                </Form.Item>

                <Form.Item
                  name="slashCommandInjectionMode"
                  label={t("modelSettings.form.slashCommandInjectionMode.label")}
                  help={t("modelSettings.form.slashCommandInjectionMode.help")}>
                  <Select
                    allowClear
                    placeholder={t(
                      "modelSettings.form.slashCommandInjectionMode.placeholder"
                    )}
                    options={slashInjectionOptions}
                  />
                </Form.Item>

                <Form.Item
                  name="apiProvider"
                  label={t("modelSettings.form.apiProvider.label")}
                  help={t("modelSettings.form.apiProvider.help")}>
                  <Select
                    allowClear
                    showSearch
                    placeholder={t("modelSettings.form.apiProvider.placeholder")}
                    options={providerOptions}
                    optionFilterProp="label"
                  />
                </Form.Item>

                <Form.Item
                  name="extraHeaders"
                  label={t("modelSettings.form.extraHeaders.label")}
                  help={t("modelSettings.form.extraHeaders.help")}
                  rules={[{ validator: validateJsonObject }]}>
                  <Input.TextArea
                    rows={4}
                    placeholder={t("modelSettings.form.extraHeaders.placeholder")}
                  />
                </Form.Item>

                <Form.Item
                  name="extraBody"
                  label={t("modelSettings.form.extraBody.label")}
                  help={t("modelSettings.form.extraBody.help")}
                  rules={[{ validator: validateJsonObject }]}>
                  <Input.TextArea
                    rows={4}
                    placeholder={t("modelSettings.form.extraBody.placeholder")}
                  />
                </Form.Item>
              </>
            )
          }
        ]}
      />
    </div>
  )
}
