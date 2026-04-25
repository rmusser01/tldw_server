import { Button, Form, Input, InputNumber, Switch, Tooltip, Tag } from "antd"
import React from "react"
import { Select } from "antd"
import { HelpCircle } from "lucide-react"
import { normalizeKeywordList } from "./worldBookEntryUtils"
import { getSettingLabel, getSettingDescription } from "./worldBookLabelUtils"
import {
  WORLD_BOOK_FORM_DEFAULTS,
  WORLD_BOOK_STARTER_TEMPLATES,
  buildWorldBookFormPayload,
  getWorldBookStarterTemplate,
  hasDuplicateWorldBookName,
  normalizeWorldBookName
} from "./worldBookFormUtils"
import {
  ACCESSIBLE_SWITCH_TEXT_PROPS,
  type WorldBookFormMode
} from "./worldBookManagerUtils"

// Helper component for form field labels with tooltips
export const LabelWithHelp: React.FC<{ label: string; help: string }> = ({ label, help }) => (
  <span className="inline-flex items-center gap-1">
    {label}
    <Tooltip title={help}>
      <HelpCircle className="w-4 h-4 text-text-muted cursor-help" />
    </Tooltip>
  </span>
)

// Keyword preview component for real-time feedback
export const KeywordPreview: React.FC<{ value?: unknown }> = ({ value }) => {
  const keywords = normalizeKeywordList(value)
  if (keywords.length === 0) return null
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {keywords.map((k, i) => <Tag key={i}>{k}</Tag>)}
    </div>
  )
}

export type WorldBookFormProps = {
  mode: WorldBookFormMode
  form: any
  worldBooks: Array<{ id?: number; name?: string }>
  submitting: boolean
  currentWorldBookId?: number | null
  maxRecursiveDepth?: number
  showTechnicalLabels?: boolean
  onSubmit: (values: Record<string, any>) => void
}

export const WorldBookForm: React.FC<WorldBookFormProps> = ({
  mode,
  form,
  worldBooks,
  submitting,
  currentWorldBookId,
  maxRecursiveDepth = 10,
  showTechnicalLabels,
  onSubmit
}) => {
  const technical = showTechnicalLabels ?? false
  const submitLabel = mode === "create" ? "Create" : "Save"
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = React.useState(false)
  const advancedSettingsContentId = React.useId()
  const recursiveScanningEnabled = Boolean(Form.useWatch("recursive_scanning", form))
  const recursiveDepthLimit =
    typeof maxRecursiveDepth === "number" && Number.isFinite(maxRecursiveDepth)
      ? Math.max(1, Math.round(maxRecursiveDepth))
      : 10
  const handleTemplateChange = React.useCallback(
    (templateKey?: string) => {
      if (!templateKey) return
      const template = getWorldBookStarterTemplate(templateKey)
      if (!template) return

      const currentValues = form.getFieldsValue()
      const templateDefaults = { ...WORLD_BOOK_FORM_DEFAULTS, ...(template.defaults || {}) }
      const nextValues: Record<string, any> = {
        template_key: template.key,
        scan_depth: templateDefaults.scan_depth,
        token_budget: templateDefaults.token_budget,
        recursive_scanning: templateDefaults.recursive_scanning,
        enabled: templateDefaults.enabled
      }

      if (!normalizeWorldBookName(currentValues?.name)) {
        nextValues.name = template.suggestedName
      }
      if (!normalizeWorldBookName(currentValues?.description)) {
        nextValues.description = template.description
      }
      form.setFieldsValue(nextValues)
    },
    [form]
  )

  return (
    <Form
      layout="vertical"
      form={form}
      initialValues={WORLD_BOOK_FORM_DEFAULTS}
      onFinish={(values) => onSubmit(buildWorldBookFormPayload(values, mode))}
    >
      {mode === "create" && (
        <Form.Item name="template_key" label="Starter Template (optional)">
          <Select
            allowClear
            placeholder="Choose a starter template"
            options={WORLD_BOOK_STARTER_TEMPLATES.map((template) => ({
              label: template.label,
              value: template.key
            }))}
            onChange={(value) => handleTemplateChange(value)}
          />
        </Form.Item>
      )}
      <Form.Item
        name="name"
        label="Name"
        rules={[
          { required: true, whitespace: true, message: "Name is required" },
          {
            validator: (_: any, value: string) => {
              const candidate = normalizeWorldBookName(value)
              if (!candidate) return Promise.resolve()
              if (hasDuplicateWorldBookName(candidate, worldBooks, { excludeId: currentWorldBookId })) {
                return Promise.reject(new Error(`A world book named "${candidate}" already exists.`))
              }
              return Promise.resolve()
            }
          }
        ]}
      >
        <Input />
      </Form.Item>
      <Form.Item name="description" label="Description (optional)">
        <Input />
      </Form.Item>
      <Form.Item name="enabled" label="Enabled" valuePropName="checked">
        <Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} />
      </Form.Item>
      <details
        className="mb-4"
        open={advancedSettingsOpen}
        onToggle={(event) => {
          const nextOpen = (event.currentTarget as HTMLDetailsElement).open
          setAdvancedSettingsOpen(nextOpen)
        }}
      >
        <summary
          className="cursor-pointer text-sm text-text-muted hover:text-text"
          aria-expanded={advancedSettingsOpen}
          aria-controls={advancedSettingsContentId}
        >
          Matching & Budget
        </summary>
        <div id={advancedSettingsContentId} className="mt-3 pl-2 border-l-2 border-border space-y-0">
          <Form.Item
            name="scan_depth"
            label={<LabelWithHelp label={getSettingLabel("scan_depth", technical)} help={getSettingDescription("scan_depth", technical)} />}
          >
            <InputNumber style={{ width: "100%" }} min={1} max={20} />
          </Form.Item>
          <Form.Item
            name="token_budget"
            label={<LabelWithHelp label={getSettingLabel("token_budget", technical)} help={getSettingDescription("token_budget", technical)} />}
          >
            <InputNumber style={{ width: "100%" }} min={50} max={5000} />
          </Form.Item>
          <Form.Item
            name="recursive_scanning"
            label={<LabelWithHelp label={getSettingLabel("recursive_scanning", technical)} help={getSettingDescription("recursive_scanning", technical)} />}
            valuePropName="checked"
          >
            <Switch {...ACCESSIBLE_SWITCH_TEXT_PROPS} />
          </Form.Item>
        </div>
      </details>
      {recursiveScanningEnabled && (
        <div
          data-testid={`recursive-scanning-warning-${mode}`}
          className="mb-4 rounded border border-warn/50 bg-warn/10 px-3 py-2 text-xs text-text"
        >
          Recursive scanning can cause entries to trigger each other. Matching depth is limited
          to {` ${recursiveDepthLimit} `}levels.
        </div>
      )}
      <Button type="primary" htmlType="submit" loading={submitting} className="w-full">
        {submitLabel}
      </Button>
    </Form>
  )
}
