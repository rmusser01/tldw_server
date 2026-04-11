import { AutoComplete, Button, Form, Input, InputNumber, Select, Slider, Switch, Tooltip } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"
import { AlertCircle, ChevronDown, ChevronUp } from "lucide-react"
import { LabelWithHelp } from "@/components/Common/LabelWithHelp"

type DictionaryEntryCreateFormProps = {
  form: any
  adding: boolean
  advancedMode: boolean
  onToggleAdvancedMode: () => void
  onSubmit: (values: any) => void | Promise<void>
  onPatternChange: (value: string) => void
  onReplacementChange: () => void
  onTypeChange: (value: string) => void
  regexError: string | null
  regexServerError: string | null
  entryGroupOptions: Array<{ value: string; label: React.ReactNode }>
  normalizeProbabilityValue: (value: unknown, fallback?: number) => number
  formatProbabilityFrequencyHint: (value: unknown) => string
}

export const DictionaryEntryCreateForm: React.FC<DictionaryEntryCreateFormProps> = ({
  form,
  adding,
  advancedMode,
  onToggleAdvancedMode,
  onSubmit,
  onPatternChange,
  onReplacementChange,
  onTypeChange,
  regexError,
  regexServerError,
  entryGroupOptions,
  normalizeProbabilityValue,
  formatProbabilityFrequencyHint
}) => {
  const { t } = useTranslation(["common", "option"])
  const advancedOptionsPanelId = React.useId()

  return (
    <div className="border border-border rounded-lg p-4 bg-surface2/30 mt-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-text">
          {t("option:dictionaries.addEntry", "Add New Entry")}
        </h4>
        <button
          type="button"
          className="flex items-center gap-1 text-xs text-text-muted hover:text-text transition-colors"
          onClick={onToggleAdvancedMode}
          aria-expanded={advancedMode}
          aria-controls={advancedOptionsPanelId}
        >
          {advancedMode ? (
            <>
              <ChevronUp className="w-3 h-3" />
              {t("option:dictionaries.simpleMode", "Simple mode")}
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3" />
              {t("option:dictionaries.advancedMode", "Advanced options")}
            </>
          )}
        </button>
      </div>

      <Form layout="vertical" form={form} onFinish={onSubmit}>
        <div className="grid gap-3 sm:grid-cols-2">
          <Form.Item
            name="pattern"
            label={
              <LabelWithHelp
                label={t("option:dictionaries.patternLabel", "Find")}
                help={t(
                  "option:dictionaries.patternHelp",
                  "Text to find. For simple terms like 'KCl', just type it. For patterns, select Regex type and use /pattern/flags format."
                )}
                required
              />
            }
            rules={[{ required: true, message: "Pattern is required" }]}
            validateStatus={regexError || regexServerError ? "error" : undefined}
            help={regexError || regexServerError}
          >
            <Input
              placeholder={t("option:dictionaries.patternPlaceholder", "e.g., gonna or /colour/i")}
              className="font-mono"
              onChange={(event) => onPatternChange(event.target.value)}
              aria-describedby="pattern-help"
            />
          </Form.Item>
          <Form.Item
            name="replacement"
            label={
              <LabelWithHelp
                label={t("option:dictionaries.replacementLabel", "Replace with")}
                help={t(
                  "option:dictionaries.replacementHelp",
                  "The text that will replace matches. For regex, you can use $1, $2 for capture groups."
                )}
                required
              />
            }
            rules={[{ required: true, message: "Replacement is required" }]}
          >
            <Input
              placeholder={t("option:dictionaries.replacementPlaceholder", "e.g., going to or color")}
              onChange={onReplacementChange}
              aria-describedby="replacement-help"
            />
          </Form.Item>
        </div>

        <Form.Item
          name="type"
          label={
            <LabelWithHelp
              label={t("option:dictionaries.typeLabel", "Match type")}
              help={t(
                "option:dictionaries.typeHelp",
                "Literal matches exact text. Regex allows pattern matching with regular expressions."
              )}
            />
          }
          initialValue="literal"
          >
          <Select
            options={[
              { label: t("option:dictionaries.typeLiteral", "Literal (exact match)"), value: "literal" },
              { label: t("option:dictionaries.typeRegex", "Regex (pattern match)"), value: "regex" }
            ]}
            onChange={onTypeChange}
          />
        </Form.Item>
        <Form.Item noStyle shouldUpdate={(prev, current) => prev.type !== current.type}>
          {() =>
            form.getFieldValue("type") === "regex" ? (
              <div className="-mt-2 mb-3 rounded border border-border bg-surface px-3 py-2 text-xs text-text-muted">
                Regex helper: `.*` = any text, `\b` = word boundary, `(group)` can be reused as
                `$1` in replacement.
              </div>
            ) : null
          }
        </Form.Item>

        {(regexError || regexServerError) && (
          <div className="flex items-start gap-2 p-2 mb-3 rounded bg-danger/10 text-danger text-xs">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <div className="font-medium">Regex validation issue</div>
              <div className="text-danger/80">{regexError || regexServerError}</div>
            </div>
          </div>
        )}

        {advancedMode && (
          <div id={advancedOptionsPanelId}>
            <div className="grid gap-3 sm:grid-cols-2 mt-3 pt-3 border-t border-border">
              <Form.Item
                name="probability"
                label={
                  <LabelWithHelp
                    label={t("option:dictionaries.probabilityLabel", "Probability")}
                    help={t(
                      "option:dictionaries.probabilityHelp",
                      "Chance of applying this replacement (0-1). Use 1 for always, 0.5 for 50% of the time."
                    )}
                  />
                }
                initialValue={1}
                rules={[
                  {
                    type: "number",
                    min: 0,
                    max: 1,
                    message: "Probability must be between 0 and 1."
                  }
                ]}
              >
                <InputNumber min={0} max={1} step={0.01} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item noStyle shouldUpdate={(prev, current) => prev.probability !== current.probability}>
                {() => {
                  const probabilityValue = Number(
                    normalizeProbabilityValue(form.getFieldValue("probability"), 1).toFixed(2)
                  )
                  return (
                    <div className="-mt-2 mb-3">
                      <Slider
                        min={0}
                        max={1}
                        step={0.01}
                        value={probabilityValue}
                        onChange={(value) => {
                          const nextValue = Array.isArray(value) ? value[0] : value
                          form.setFieldValue(
                            "probability",
                            Number(normalizeProbabilityValue(nextValue, 1).toFixed(2))
                          )
                        }}
                        aria-label="Probability slider"
                      />
                      <div className="text-xs text-text-muted">
                        {formatProbabilityFrequencyHint(probabilityValue)}
                      </div>
                    </div>
                  )
                }}
              </Form.Item>
              <Form.Item
                name="group"
                label={
                  <LabelWithHelp
                    label={t("option:dictionaries.groupLabel", "Group")}
                    help={t(
                      "option:dictionaries.groupHelp",
                      "Optional category for organizing entries (e.g., 'medications', 'abbreviations')."
                    )}
                  />
                }
              >
                <AutoComplete
                  options={entryGroupOptions}
                  placeholder={t("option:dictionaries.groupPlaceholder", "e.g., medications")}
                  filterOption={(inputValue, option) =>
                    String(option?.value || "")
                      .toLowerCase()
                      .includes(inputValue.toLowerCase())
                  }
                />
              </Form.Item>
              <Form.Item
                name="max_replacements"
                label={
                  <LabelWithHelp
                    label={t("option:dictionaries.maxReplacementsLabel", "Max replacements")}
                    help={t(
                      "option:dictionaries.maxReplacementsHelp",
                      "Probability controls whether this entry fires. Max replacements limits how many replacements happen when it does."
                    )}
                  />
                }
              >
                <InputNumber min={0} style={{ width: "100%" }} placeholder="Unlimited" />
              </Form.Item>
              <Form.Item
                name={["timed_effects", "sticky"]}
                label={
                  <LabelWithHelp
                    label="Sticky (seconds)"
                    help="Keep this replacement active for additional messages after it fires. Use 0 to disable."
                  />
                }
                initialValue={0}
              >
                <InputNumber min={0} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item
                name={["timed_effects", "cooldown"]}
                label={
                  <LabelWithHelp
                    label="Cooldown (seconds)"
                    help="Minimum wait time before this entry can fire again. Use 0 to disable."
                  />
                }
                initialValue={0}
              >
                <InputNumber min={0} style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item
                name={["timed_effects", "delay"]}
                label={
                  <LabelWithHelp
                    label="Delay (seconds)"
                    help="Wait time before this entry becomes eligible to run. Use 0 to disable."
                  />
                }
                initialValue={0}
              >
                <InputNumber min={0} style={{ width: "100%" }} />
              </Form.Item>
              <div className="flex gap-4">
                <Form.Item
                  name="enabled"
                  label={t("option:dictionaries.enabledLabel", "Enabled")}
                  valuePropName="checked"
                  initialValue={true}
                >
                  <Switch checkedChildren="On" unCheckedChildren="Off" />
                </Form.Item>
                <Form.Item
                  name="case_sensitive"
                  label={
                    <LabelWithHelp
                      label={t("option:dictionaries.caseSensitiveLabel", "Case sensitive")}
                      help={t(
                        "option:dictionaries.caseSensitiveHelp",
                        "When off (default), 'KCl' matches 'kcl', 'KCL', etc. Recommended off for medical terms."
                      )}
                    />
                  }
                  valuePropName="checked"
                  initialValue={false}
                >
                  <Switch checkedChildren="On" unCheckedChildren="Off" />
                </Form.Item>
              </div>
            </div>
          </div>
        )}

        <Tooltip title="Ctrl+Enter">
          <Button
            type="primary"
            htmlType="submit"
            loading={adding}
            disabled={!!regexError || !!regexServerError}
            className="w-full mt-3"
          >
            {t("option:dictionaries.addEntryButton", "Add Entry")}
          </Button>
        </Tooltip>
      </Form>
    </div>
  )
}
