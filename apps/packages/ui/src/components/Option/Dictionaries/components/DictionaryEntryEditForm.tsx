import {
  AutoComplete,
  Button,
  Form,
  Input,
  InputNumber,
  Select,
  Slider,
  Switch,
} from "antd"
import React from "react"
import { ChevronDown, ChevronUp } from "lucide-react"
import { LabelWithHelp } from "@/components/Common/LabelWithHelp"

type DictionaryEntryEditFormProps = {
  form: any
  updatingEntry: boolean
  onSubmit: (values: any) => void | Promise<void>
  entryGroupOptions: Array<{ value: string; label: React.ReactNode }>
  normalizeProbabilityValue: (value: unknown, fallback?: number) => number
  formatProbabilityFrequencyHint: (value: unknown) => string
}

export const DictionaryEntryEditForm: React.FC<DictionaryEntryEditFormProps> = ({
  form,
  updatingEntry,
  onSubmit,
  entryGroupOptions,
  normalizeProbabilityValue,
  formatProbabilityFrequencyHint,
}) => {
  const advancedOptionsPanelId = React.useId()
  const watchedProbability = Form.useWatch("probability", form)
  const watchedGroup = Form.useWatch("group", form)
  const watchedMaxReplacements = Form.useWatch("max_replacements", form)
  const watchedTimedEffects = Form.useWatch("timed_effects", form)
  const watchedCaseSensitive = Form.useWatch("case_sensitive", form)

  const hasNonDefaultAdvancedValues = React.useMemo(() => {
    const prob = watchedProbability
    const group = watchedGroup
    const maxReplacements = watchedMaxReplacements
    const sticky = watchedTimedEffects?.sticky
    const cooldown = watchedTimedEffects?.cooldown
    const delay = watchedTimedEffects?.delay
    const caseSensitive = watchedCaseSensitive
    return (
      (typeof prob === "number" && prob !== 1) ||
      (typeof group === "string" && group.trim() !== "") ||
      (typeof maxReplacements === "number" && maxReplacements > 0) ||
      (typeof sticky === "number" && sticky > 0) ||
      (typeof cooldown === "number" && cooldown > 0) ||
      (typeof delay === "number" && delay > 0) ||
      caseSensitive === true
    )
  }, [
    watchedCaseSensitive,
    watchedGroup,
    watchedMaxReplacements,
    watchedProbability,
    watchedTimedEffects,
  ])

  const [advancedMode, setAdvancedMode] = React.useState(hasNonDefaultAdvancedValues)

  React.useEffect(() => {
    if (hasNonDefaultAdvancedValues) {
      setAdvancedMode(true)
    }
  }, [hasNonDefaultAdvancedValues])

  return (
    <Form layout="vertical" form={form} onFinish={onSubmit}>
      <Form.Item
        name="pattern"
        label={
          <LabelWithHelp
            label="Find"
            help="Text to find. For simple terms, just type them. For patterns, select Regex type and use /pattern/flags format."
          />
        }
        rules={[{ required: true }]}>
        <Input placeholder="e.g., gonna or /colour/i" className="font-mono" />
      </Form.Item>
      <Form.Item
        name="replacement"
        label={
          <LabelWithHelp
            label="Replace with"
            help="The text that will replace matches. For regex, you can use $1, $2 for capture groups."
          />
        }
        rules={[{ required: true }]}>
        <Input placeholder="e.g., going to or color" />
      </Form.Item>
      <Form.Item
        name="type"
        label={
          <LabelWithHelp
            label="Match type"
            help="Literal matches exact text. Regex allows pattern matching with regular expressions."
          />
        }
        initialValue="literal">
        <Select
          options={[
            { label: "Literal (exact match)", value: "literal" },
            { label: "Regex (pattern match)", value: "regex" },
          ]}
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
      <Form.Item
        name="enabled"
        label="Enabled"
        valuePropName="checked"
        initialValue={true}>
        <Switch checkedChildren="On" unCheckedChildren="Off" />
      </Form.Item>
      <button
        type="button"
        className="flex items-center gap-1 text-xs text-text-muted hover:text-text mb-2"
        onClick={() => setAdvancedMode((prev) => !prev)}
        aria-expanded={advancedMode}
        aria-controls={advancedOptionsPanelId}
      >
        {advancedMode ? "Simple mode" : "Advanced options"}
        {advancedMode ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {advancedMode && (
        <div id={advancedOptionsPanelId}>
          <Form.Item
            name="probability"
            label={
              <LabelWithHelp
                label="Probability"
                help="Chance of applying this replacement (0-1). Use 1 for always, 0.5 for 50% of the time."
              />
            }
            initialValue={1}
            rules={[
              {
                type: "number",
                min: 0,
                max: 1,
                message: "Probability must be between 0 and 1.",
              },
            ]}>
            <InputNumber min={0} max={1} step={0.01} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item
            noStyle
            shouldUpdate={(prev, current) => prev.probability !== current.probability}>
            {() => {
              const probabilityValue = Number(
                normalizeProbabilityValue(form.getFieldValue("probability"), 1).toFixed(2)
              )
              return (
                <div className="mt-[-8px] mb-3">
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
                label="Group"
                help="Optional category for organizing entries (e.g., 'medications', 'abbreviations')."
              />
            }>
            <AutoComplete
              options={entryGroupOptions}
              placeholder="e.g., medications"
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
                label="Max Replacements"
                help="Probability controls whether this entry fires. Max replacements caps how many times it can apply per message."
              />
            }>
            <InputNumber min={0} style={{ width: "100%" }} />
          </Form.Item>
          <div className="grid gap-3 sm:grid-cols-3">
            <Form.Item
              name={["timed_effects", "sticky"]}
              label={
                <LabelWithHelp
                  label="Sticky (seconds)"
                  help="Keep this replacement active for additional messages after it fires. Use 0 to disable."
                />
              }
              initialValue={0}>
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
              initialValue={0}>
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
              initialValue={0}>
              <InputNumber min={0} style={{ width: "100%" }} />
            </Form.Item>
          </div>
          <Form.Item
            name="case_sensitive"
            label={
              <LabelWithHelp
                label="Case sensitive"
                help="When off (default), matching ignores capitalization."
              />
            }
            valuePropName="checked">
            <Switch checkedChildren="On" unCheckedChildren="Off" />
          </Form.Item>
        </div>
      )}
      <Button type="primary" htmlType="submit" loading={updatingEntry} className="w-full">
        Save Changes
      </Button>
    </Form>
  )
}
