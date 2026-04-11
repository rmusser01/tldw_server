import React from "react"
import { Alert, Button, Drawer, Form, Input, Select, Collapse, Tooltip, Space, Tag, Modal, Grid } from "antd"
import { useTranslation } from "react-i18next"
import { ChevronDown, ChevronUp, Info, Cloud, Plus, Trash2 } from "lucide-react"
import type {
  FewShotExample,
  PromptFormat,
  PromptSyncStatus,
  PromptSourceSystem,
  StructuredPromptDefinition
} from "@/db/dexie/types"
import { useFormDraft, formatDraftAge } from "@/hooks/useFormDraft"
import { previewStructuredPromptServer, type StructuredPromptPreviewResponse } from "@/services/prompts-api"
import {
  estimatePromptTokens,
  getPromptTokenBudgetState
} from "./prompt-length-utils"
import {
  extractTemplateVariables,
  tokenizeTemplateVariableHighlights,
  validateTemplateVariableSyntax
} from "./prompt-template-variable-utils"
import { StructuredPromptEditor } from "./Structured/StructuredPromptEditor"
import {
  convertLegacyPromptToStructuredDefinition,
  createDefaultStructuredPromptDefinition,
  renderStructuredPromptLegacySnapshot,
  stableSerializePromptSnapshot
} from "./structured-prompt-utils"
import { VersionHistoryDrawer } from "./Studio/Prompts/VersionHistoryDrawer"

type DrawerFewShotExample = {
  input: string
  output: string
  explanation?: string | null
}

const readExampleRecordValue = (
  record: Record<string, unknown> | undefined,
  preferredKey: string
): string => {
  if (!record) return ""
  const preferredValue = record[preferredKey]
  if (typeof preferredValue === "string") {
    return preferredValue
  }
  for (const value of Object.values(record)) {
    if (typeof value === "string") {
      return value
    }
  }
  return ""
}

const normalizeFewShotExamplesForForm = (
  examples:
    | Array<
        | {
            input?: string
            output?: string
            explanation?: string | null
            inputs?: Record<string, unknown>
            outputs?: Record<string, unknown>
          }
        | FewShotExample
      >
    | null
    | undefined
): DrawerFewShotExample[] => {
  if (!Array.isArray(examples)) return []
  return examples
    .map((example) => {
      const candidate = example as
        | {
            input?: string
            output?: string
            explanation?: string | null
            inputs?: Record<string, unknown>
            outputs?: Record<string, unknown>
          }
        | undefined
      if (!candidate || typeof candidate !== "object") {
        return null
      }

      const input =
        typeof candidate.input === "string"
          ? candidate.input
          : readExampleRecordValue(candidate.inputs, "input")
      const output =
        typeof candidate.output === "string"
          ? candidate.output
          : readExampleRecordValue(candidate.outputs, "output")

      const normalizedExample: DrawerFewShotExample = {
        input,
        output,
        explanation:
          typeof candidate.explanation === "string"
            ? candidate.explanation
            : null
      }
      return normalizedExample
    })
    .filter((example): example is DrawerFewShotExample => {
      if (!example) return false
      return (
        example.input.trim().length > 0 ||
        example.output.trim().length > 0 ||
        !!example.explanation
      )
    })
}

const mapFewShotExamplesForSubmit = (
  examples: DrawerFewShotExample[] | null | undefined
): FewShotExample[] | null => {
  if (!Array.isArray(examples)) return null

  const normalized = examples
    .map((example) => ({
      input: (example.input || "").trim(),
      output: (example.output || "").trim(),
      explanation: (example.explanation || "").trim()
    }))
    .filter((example) => example.input.length > 0 || example.output.length > 0)
    .map(
      (example): FewShotExample => ({
        inputs: { input: example.input },
        outputs: { output: example.output },
        ...(example.explanation ? { explanation: example.explanation } : {})
      })
    )

  return normalized.length > 0 ? normalized : null
}

const normalizePromptDraftSnapshot = (values: {
  name?: unknown
  author?: unknown
  details?: unknown
  system_prompt?: unknown
  user_prompt?: unknown
  promptFormat?: unknown
  structuredPromptDefinition?: unknown
  keywords?: unknown
  fewShotExamples?: unknown
}) => {
  const normalizeString = (value: unknown) =>
    typeof value === "string" ? value.trim() : ""
  const keywords = Array.isArray(values?.keywords)
    ? values.keywords
        .map((keyword) => (typeof keyword === "string" ? keyword.trim() : ""))
        .filter((keyword) => keyword.length > 0)
        .sort()
    : []
  const fewShotExamples = normalizeFewShotExamplesForForm(
    values?.fewShotExamples as any
  )
    .map((example) => ({
      input: example.input.trim(),
      output: example.output.trim(),
      explanation: (example.explanation || "").trim()
    }))
    .filter(
      (example) =>
        example.input.length > 0 ||
        example.output.length > 0 ||
        example.explanation.length > 0
    )

  return {
    name: normalizeString(values?.name),
    author: normalizeString(values?.author),
    details: normalizeString(values?.details),
    system_prompt: normalizeString(values?.system_prompt),
    user_prompt: normalizeString(values?.user_prompt),
    promptFormat: values?.promptFormat === "structured" ? "structured" : "legacy",
    structuredPromptDefinition:
      values?.promptFormat === "structured" &&
      values?.structuredPromptDefinition &&
      typeof values.structuredPromptDefinition === "object"
        ? values.structuredPromptDefinition
        : null,
    keywords,
    fewShotExamples
  }
}

interface PromptDrawerProps {
  open: boolean
  onClose: () => void
  mode: "create" | "edit"
  initialValues?: {
    id?: string
    name?: string
    author?: string
    details?: string
    system_prompt?: string
    user_prompt?: string
    promptFormat?: PromptFormat
    promptSchemaVersion?: number | null
    structuredPromptDefinition?: StructuredPromptDefinition | null
    keywords?: string[]
    // Sync fields
    serverId?: number | null
    syncStatus?: PromptSyncStatus
    sourceSystem?: PromptSourceSystem
    studioProjectId?: number | null
    lastSyncedAt?: number | null
    // Advanced fields
    fewShotExamples?:
      | Array<
          | {
              input?: string
              output?: string
              explanation?: string | null
              inputs?: Record<string, unknown>
              outputs?: Record<string, unknown>
            }
          | FewShotExample
        >
      | null
    modulesConfig?: Array<{ name: string; enabled: boolean }> | null
    changeDescription?: string | null
    versionNumber?: number | null
  }
  onSubmit: (values: any) => void
  isLoading: boolean
  allTags: string[]
}

export const PromptDrawer: React.FC<PromptDrawerProps> = ({
  open,
  onClose,
  mode,
  initialValues,
  onSubmit,
  isLoading,
  allTags
}) => {
  const { t } = useTranslation(["settings", "common"])
  const screens = Grid.useBreakpoint()
  const [form] = Form.useForm()
  const [showSystemHelp, setShowSystemHelp] = React.useState(false)
  const [showUserHelp, setShowUserHelp] = React.useState(false)
  const [versionHistoryOpen, setVersionHistoryOpen] = React.useState(false)
  const [closeConfirmOpen, setCloseConfirmOpen] = React.useState(false)
  const [promptFormat, setPromptFormat] = React.useState<PromptFormat>("legacy")
  const [structuredPromptDefinition, setStructuredPromptDefinition] =
    React.useState<StructuredPromptDefinition>(
      createDefaultStructuredPromptDefinition()
    )
  const [structuredPreviewResult, setStructuredPreviewResult] =
    React.useState<StructuredPromptPreviewResponse | null>(null)
  const [structuredPreviewLoading, setStructuredPreviewLoading] =
    React.useState(false)
  const systemPromptValue = Form.useWatch("system_prompt", form)
  const userPromptValue = Form.useWatch("user_prompt", form)
  const systemTemplateVariables = React.useMemo(
    () => extractTemplateVariables(systemPromptValue),
    [systemPromptValue]
  )
  const userTemplateVariables = React.useMemo(
    () => extractTemplateVariables(userPromptValue),
    [userPromptValue]
  )
  const draftEditId = React.useMemo(() => {
    if (mode !== "edit") return undefined
    if (initialValues?.id) return String(initialValues.id)
    if (typeof initialValues?.serverId === "number") {
      return `server-${initialValues.serverId}`
    }
    return undefined
  }, [mode, initialValues?.id, initialValues?.serverId])
  const draftStorageKey = React.useMemo(
    () => `tldw-prompt-drawer-draft-${mode}-${draftEditId || "new"}`,
    [mode, draftEditId]
  )
  const initialSnapshot = React.useMemo(
    () => normalizePromptDraftSnapshot(initialValues || {}),
    [initialValues]
  )

  // Draft auto-save
  const { hasDraft, draftData, saveDraft, clearDraft, applyDraft, dismissDraft } = useFormDraft({
    storageKey: draftStorageKey,
    formType: mode,
    editId: draftEditId,
    autoSaveInterval: 30000
  })

  // Check if prompt is synced
  const isSynced = initialValues?.serverId != null
  const syncStatus = initialValues?.syncStatus || "local"
  const versionHistoryPromptId =
    typeof initialValues?.serverId === "number" ? initialValues.serverId : null
  const canViewVersionHistory = mode === "edit" && versionHistoryPromptId !== null

  React.useEffect(() => {
    if (open && initialValues) {
      form.setFieldsValue({
        ...initialValues,
        fewShotExamples: normalizeFewShotExamplesForForm(initialValues.fewShotExamples),
        promptFormat: initialValues.promptFormat || "legacy",
        promptSchemaVersion:
          initialValues.promptSchemaVersion ??
          (initialValues.promptFormat === "structured" ? 1 : null),
        structuredPromptDefinition:
          initialValues.structuredPromptDefinition ||
          createDefaultStructuredPromptDefinition()
      })
      setPromptFormat(initialValues.promptFormat || "legacy")
      setStructuredPromptDefinition(
        initialValues.structuredPromptDefinition ||
          createDefaultStructuredPromptDefinition()
      )
      setStructuredPreviewResult(null)
    }
    if (open && mode === "create") {
      form.resetFields()
      setPromptFormat(
        (initialValues?.promptFormat as PromptFormat | undefined) || "legacy"
      )
      setStructuredPromptDefinition(
        initialValues?.structuredPromptDefinition ||
          createDefaultStructuredPromptDefinition()
      )
      setStructuredPreviewResult(null)
    }
  }, [open, initialValues, mode, form])

  React.useEffect(() => {
    if (!open) return
    form.setFieldValue("promptFormat", promptFormat)
    form.setFieldValue(
      "promptSchemaVersion",
      promptFormat === "structured" ? 1 : null
    )
    form.setFieldValue(
      "structuredPromptDefinition",
      promptFormat === "structured" ? structuredPromptDefinition : null
    )
  }, [form, open, promptFormat, structuredPromptDefinition])

  React.useEffect(() => {
    if (!open) {
      setVersionHistoryOpen(false)
      setCloseConfirmOpen(false)
    }
  }, [open])

  // Auto-save on form value changes
  React.useEffect(() => {
    if (!open) return
    const interval = setInterval(() => {
      const values = form.getFieldsValue()
      const hasContent =
        values.name ||
        values.system_prompt ||
        values.user_prompt ||
        (promptFormat === "structured" &&
          Array.isArray(structuredPromptDefinition?.blocks) &&
          structuredPromptDefinition.blocks.length > 0)
      if (hasContent) {
        saveDraft(values)
      }
    }, 30000)
    return () => clearInterval(interval)
  }, [open, form, promptFormat, saveDraft, structuredPromptDefinition])

  const handleFinish = (values: any) => {
    clearDraft()
    const structuredSnapshot =
      promptFormat === "structured"
        ? renderStructuredPromptLegacySnapshot(structuredPromptDefinition)
        : null
    onSubmit({
      ...values,
      promptFormat,
      promptSchemaVersion: promptFormat === "structured" ? 1 : null,
      structuredPromptDefinition:
        promptFormat === "structured" ? structuredPromptDefinition : null,
      system_prompt:
        structuredSnapshot?.systemPrompt ?? values?.system_prompt,
      user_prompt: structuredSnapshot?.userPrompt ?? values?.user_prompt,
      fewShotExamples: mapFewShotExamplesForSubmit(values?.fewShotExamples)
    })
  }

  const handleConvertToStructured = React.useCallback(() => {
    const currentValues = form.getFieldsValue(true)
    setPromptFormat("structured")
    setStructuredPromptDefinition(
      convertLegacyPromptToStructuredDefinition(
        currentValues?.system_prompt,
        currentValues?.user_prompt
      )
    )
    setStructuredPreviewResult(null)
  }, [form])

  const handleStructuredPreview = React.useCallback(
    async (variables: Record<string, string>) => {
      try {
        setStructuredPreviewLoading(true)
        const result = await previewStructuredPromptServer({
          prompt_format: "structured",
          prompt_schema_version: 1,
          prompt_definition: structuredPromptDefinition,
          variables
        })
        setStructuredPreviewResult(result)
      } catch {
        setStructuredPreviewResult(null)
      } finally {
        setStructuredPreviewLoading(false)
      }
    },
    [structuredPromptDefinition]
  )

  const closeWithoutSaving = React.useCallback(() => {
    clearDraft()
    setCloseConfirmOpen(false)
    onClose()
  }, [clearDraft, onClose])

  const handleRequestClose = React.useCallback(() => {
    const currentSnapshot = normalizePromptDraftSnapshot(
      form.getFieldsValue(true) || {}
    )
    const hasDirtyValues =
      form.isFieldsTouched(true) ||
      stableSerializePromptSnapshot(currentSnapshot) !==
        stableSerializePromptSnapshot(initialSnapshot)

    if (hasDirtyValues) {
      setCloseConfirmOpen(true)
      return
    }
    closeWithoutSaving()
  }, [closeWithoutSaving, form, initialSnapshot])

  const title =
    mode === "create"
      ? t("managePrompts.modal.addTitle")
      : t("managePrompts.modal.editTitle")

  const renderPromptLengthCounter = React.useCallback(
    (value: string | undefined, testId: string) => {
      const text = value || ""
      const charCount = text.length
      const tokenEstimate = estimatePromptTokens(text)
      const budgetState = getPromptTokenBudgetState(tokenEstimate)
      const budgetClass =
        budgetState === "danger"
          ? "text-danger"
          : budgetState === "warning"
            ? "text-warn"
            : "text-text-muted"

      return (
        <div className={`text-xs ${budgetClass}`} data-testid={testId}>
          {t("managePrompts.form.lengthCounter", {
            defaultValue: "{{chars}} chars / ~{{tokens}} tokens",
            chars: charCount.toLocaleString(),
            tokens: tokenEstimate.toLocaleString()
          })}
          {budgetState !== "normal" && (
            <span className="ml-2">
              {budgetState === "danger"
                ? t("managePrompts.form.lengthCounterDanger", {
                    defaultValue: "High token load"
                  })
                : t("managePrompts.form.lengthCounterWarning", {
                    defaultValue: "Approaching high token load"
                  })}
            </span>
          )}
        </div>
      )
    },
    [t]
  )

  const renderPromptFieldInsights = React.useCallback(
    ({
      value,
      variables,
      counterTestId,
      variablesTestId,
      previewTestId
    }: {
      value?: string
      variables: string[]
      counterTestId: string
      variablesTestId: string
      previewTestId: string
    }) => {
      const tokens = tokenizeTemplateVariableHighlights(value)
      const hasHighlightedVariables = tokens.some((token) => token.isVariable)

      return (
        <div className="space-y-2">
          {renderPromptLengthCounter(value, counterTestId)}
          {variables.length > 0 && (
            <div className="flex flex-wrap items-center gap-1" data-testid={variablesTestId}>
              <span className="text-xs text-text-muted">
                {t("managePrompts.form.templateVariables.label", {
                  defaultValue: "Variables:"
                })}
              </span>
              {variables.map((variableName) => (
                <Tag key={variableName} className="text-xs">
                  {`{{${variableName}}}`}
                </Tag>
              ))}
            </div>
          )}
          {hasHighlightedVariables && (
            <div
              className="rounded-md bg-surface2 p-2 text-xs font-mono whitespace-pre-wrap break-words"
              data-testid={previewTestId}
            >
              {tokens.map((token, index) =>
                token.isVariable ? (
                  <mark
                    key={`${token.variableName || "var"}-${index}`}
                    className="rounded bg-primary/20 px-0.5 text-primary"
                  >
                    {token.text}
                  </mark>
                ) : (
                  <span key={`text-${index}`}>{token.text}</span>
                )
              )}
            </div>
          )}
        </div>
      )
    },
    [renderPromptLengthCounter, t]
  )

  const templateFieldValidator = React.useCallback(
    async (_: unknown, value?: string) => {
      const validation = validateTemplateVariableSyntax(value)
      if (validation.isValid) {
        return Promise.resolve()
      }

      if (validation.code === "unmatched_braces") {
        return Promise.reject(
          new Error(
            t("managePrompts.form.templateVariables.unmatched", {
              defaultValue:
                "Template variables must use balanced braces, like {{variable_name}}."
            })
          )
        )
      }

      const invalidToken =
        validation.invalidTokens && validation.invalidTokens.length > 0
          ? validation.invalidTokens[0]
          : "{{invalid}}"
      return Promise.reject(
        new Error(
          t("managePrompts.form.templateVariables.invalid", {
            defaultValue:
              "Invalid template variable '{{token}}'. Use letters, numbers, and underscores only.",
            token: invalidToken
          })
        )
      )
    },
    [t]
  )

  const formatLastSync = (timestamp: number | null | undefined) => {
    if (!timestamp) return null
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

    if (diffMins < 1) return t("common:justNow", "Just now")
    if (diffMins < 60) return t("common:minutesAgo", "{{count}}m ago", { count: diffMins })
    if (diffHours < 24) return t("common:hoursAgo", "{{count}}h ago", { count: diffHours })
    return t("common:daysAgo", "{{count}}d ago", { count: diffDays })
  }

  // Build collapsible items for advanced sections
  const collapseItems = []

  // Advanced section (for synced prompts or if data exists)
  if (mode === "create" || mode === "edit") {
    collapseItems.push({
      key: "advanced",
      label: (
        <span className="font-medium">
          {t("managePrompts.drawer.sectionAdvanced", { defaultValue: "Advanced" })}
        </span>
      ),
      children: (
        <div className="space-y-4">
          <Form.Item
            name="changeDescription"
            label={t("managePrompts.form.changeDescription.label", {
              defaultValue: "Change description"
            })}
            help={t("managePrompts.form.changeDescription.help", {
              defaultValue: "Describe what changed in this version (for version history)."
            })}
          >
            <Input
              placeholder={t("managePrompts.form.changeDescription.placeholder", {
                defaultValue: "e.g., Added clearer instructions for code formatting"
              })}
              data-testid="prompt-drawer-change-desc"
            />
          </Form.Item>

          <Form.List name="fewShotExamples">
            {(fields, { add, remove, move }) => (
              <div className="space-y-3" data-testid="prompt-drawer-few-shot-section">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-medium text-text-muted">
                      {t("managePrompts.drawer.fewShotExamples", {
                        defaultValue: "Few-shot examples"
                      })}
                    </div>
                    <p className="text-xs text-text-muted mt-1">
                      {t("managePrompts.drawer.fewShotExamplesHint", {
                        defaultValue:
                          "Add input/output examples to improve response consistency."
                      })}
                    </p>
                  </div>
                  <Button
                    size="small"
                    type="dashed"
                    icon={<Plus className="size-3" />}
                    onClick={() =>
                      add({
                        input: "",
                        output: "",
                        explanation: null
                      })
                    }
                    data-testid="prompt-drawer-few-shot-add"
                  >
                    {t("common:add", { defaultValue: "Add" })}
                  </Button>
                </div>

                {fields.length === 0 && (
                  <p className="text-xs text-text-muted">
                    {t("managePrompts.drawer.fewShotExamplesEmpty", {
                      defaultValue: "No examples yet."
                    })}
                  </p>
                )}

                {fields.map((field, index) => (
                  <div
                    key={field.key}
                    className="rounded-md border border-border p-3 space-y-2 bg-surface2/50"
                    data-testid={`prompt-drawer-few-shot-item-${index}`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-text-muted">
                        {t("managePrompts.drawer.fewShotExampleLabel", {
                          defaultValue: "Example {{count}}",
                          count: index + 1
                        })}
                      </span>
                      <div className="flex items-center gap-1">
                        <Button
                          type="text"
                          size="small"
                          icon={<ChevronUp className="size-3" />}
                          disabled={index === 0}
                          onClick={() => move(index, index - 1)}
                          data-testid={`prompt-drawer-few-shot-move-up-${index}`}
                          aria-label={t("managePrompts.drawer.fewShotMoveUp", {
                            defaultValue: "Move example up"
                          })}
                        />
                        <Button
                          type="text"
                          size="small"
                          icon={<ChevronDown className="size-3" />}
                          disabled={index === fields.length - 1}
                          onClick={() => move(index, index + 1)}
                          data-testid={`prompt-drawer-few-shot-move-down-${index}`}
                          aria-label={t("managePrompts.drawer.fewShotMoveDown", {
                            defaultValue: "Move example down"
                          })}
                        />
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<Trash2 className="size-3" />}
                          onClick={() => remove(field.name)}
                          data-testid={`prompt-drawer-few-shot-remove-${index}`}
                          aria-label={t("common:delete", { defaultValue: "Delete" })}
                        />
                      </div>
                    </div>

                    <Form.Item
                      name={[field.name, "input"]}
                      label={t("managePrompts.drawer.fewShotInput", {
                        defaultValue: "Input"
                      })}
                      className="mb-0"
                    >
                      <Input.TextArea
                        autoSize={{ minRows: 2, maxRows: 6 }}
                        placeholder={t("managePrompts.drawer.fewShotInputPlaceholder", {
                          defaultValue: "Sample input"
                        })}
                        data-testid={`prompt-drawer-few-shot-input-${index}`}
                      />
                    </Form.Item>

                    <Form.Item
                      name={[field.name, "output"]}
                      label={t("managePrompts.drawer.fewShotOutput", {
                        defaultValue: "Output"
                      })}
                      className="mb-0"
                    >
                      <Input.TextArea
                        autoSize={{ minRows: 2, maxRows: 6 }}
                        placeholder={t("managePrompts.drawer.fewShotOutputPlaceholder", {
                          defaultValue: "Expected output"
                        })}
                        data-testid={`prompt-drawer-few-shot-output-${index}`}
                      />
                    </Form.Item>
                  </div>
                ))}
              </div>
            )}
          </Form.List>

          {initialValues?.versionNumber && (
            <div className="flex items-center justify-between gap-2 text-xs text-text-muted">
              <span>
                {t("managePrompts.drawer.versionNumber", {
                  defaultValue: "Version {{version}}",
                  version: initialValues.versionNumber
                })}
              </span>
              {canViewVersionHistory && (
                <Button
                  type="link"
                  size="small"
                  className="!px-0"
                  onClick={() => setVersionHistoryOpen(true)}
                  data-testid="prompt-drawer-view-history"
                >
                  {t("managePrompts.drawer.viewHistory", {
                    defaultValue: "View history"
                  })}
                </Button>
              )}
            </div>
          )}
        </div>
      )
    })
  }

  // Sync section (only in edit mode when synced)
  if (mode === "edit" && isSynced) {
    collapseItems.push({
      key: "sync",
      label: (
        <span className="font-medium flex items-center gap-2">
          {t("managePrompts.drawer.sectionSync", { defaultValue: "Sync Status" })}
          <Tag color={syncStatus === "synced" ? "green" : syncStatus === "pending" ? "gold" : "default"} className="text-xs">
            {syncStatus === "synced" ? t("settings:managePrompts.sync.synced", "Synced") :
             syncStatus === "pending" ? t("settings:managePrompts.sync.pending", "Pending") :
             t("settings:managePrompts.sync.local", "Local")}
          </Tag>
        </span>
      ),
      children: (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Cloud className="size-4 text-primary" />
            <span className="text-sm">
              {t("managePrompts.drawer.linkedToServer", {
                defaultValue: "Linked to Prompt Studio"
              })}
            </span>
          </div>

          {initialValues?.studioProjectId && (
            <div className="text-xs text-text-muted">
              {t("managePrompts.drawer.projectId", {
                defaultValue: "Project ID: {{id}}",
                id: initialValues.studioProjectId
              })}
            </div>
          )}

          {initialValues?.serverId && (
            <div className="text-xs text-text-muted">
              {t("managePrompts.drawer.serverId", {
                defaultValue: "Server ID: {{id}}",
                id: initialValues.serverId
              })}
            </div>
          )}

          {initialValues?.lastSyncedAt && (
            <div className="text-xs text-text-muted">
              {t("managePrompts.drawer.lastSynced", {
                defaultValue: "Last synced: {{time}}",
                time: formatLastSync(initialValues.lastSyncedAt)
              })}
            </div>
          )}
        </div>
      )
    })
  }

  return (
    <Drawer
      placement="right"
      size={screens.sm ? 480 : "100%"}
      open={open}
      onClose={handleRequestClose}
      title={title}
      footer={
        <div className="flex justify-end gap-2">
          <Button onClick={handleRequestClose}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            loading={isLoading}
            onClick={() => form.submit()}
          >
            {isLoading
              ? t("managePrompts.form.btnSave.saving")
              : t("managePrompts.form.btnSave.save")}
          </Button>
        </div>
      }
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        initialValues={{ keywords: [] }}
      >
        {/* Draft recovery banner */}
        {hasDraft && draftData && (
          <Alert
            type="info"
            showIcon
            className="mb-4"
            title={t("managePrompts.drawer.draftRecovered", {
              defaultValue: "Unsaved draft found ({{age}})",
              age: formatDraftAge(draftData.savedAt)
            })}
            action={
              <Space>
                <Button
                  size="small"
                  type="primary"
                  onClick={() => {
                    const recovered = applyDraft()
                    if (recovered) {
                      form.setFieldsValue(recovered)
                    }
                  }}
                >
                  {t("common:restore", { defaultValue: "Restore" })}
                </Button>
                <Button size="small" onClick={dismissDraft}>
                  {t("common:dismiss", { defaultValue: "Dismiss" })}
                </Button>
              </Space>
            }
          />
        )}

        {/* Section: Identity */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-text-muted mb-3 flex items-center gap-2">
            {t("managePrompts.drawer.sectionIdentity", { defaultValue: "Identity" })}
            {isSynced && (
              <Tooltip title={t("managePrompts.drawer.syncedIndicator", { defaultValue: "Synced with Prompt Studio" })}>
                <Cloud className="size-3 text-primary" />
              </Tooltip>
            )}
          </h3>
          <div className="space-y-4">
            <Form.Item
              name="name"
              label={t("managePrompts.form.title.label")}
              rules={[
                {
                  required: true,
                  message: t("managePrompts.form.title.required")
                }
              ]}
            >
              <Input
                placeholder={t("managePrompts.form.title.placeholder")}
                data-testid="prompt-drawer-name"
              />
            </Form.Item>

            <Form.Item
              name="author"
              label={t("managePrompts.form.author.label", { defaultValue: "Author" })}
            >
              <Input
                placeholder={t("managePrompts.form.author.placeholder", {
                  defaultValue: "Optional author"
                })}
                data-testid="prompt-drawer-author"
              />
            </Form.Item>
          </div>
        </div>

        {/* Section: Prompt Content */}
        <div className="mb-6">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-sm font-medium text-text-muted">
              {t("managePrompts.drawer.sectionContent", { defaultValue: "Prompt Content" })}
            </h3>
            {promptFormat === "legacy" && (
              <Button
                type="default"
                onClick={handleConvertToStructured}
              >
                Convert to structured
              </Button>
            )}
          </div>
          <div className="space-y-4">
            {promptFormat === "structured" && (
              <Alert
                type="info"
                showIcon
                title="Structured prompt"
                description="Raw text fields are now locked for compatibility. Edit the block builder below; save and preview use the structured definition."
              />
            )}
            <Form.Item
              name="system_prompt"
              rules={[
                {
                  validator: templateFieldValidator
                }
              ]}
              label={
                <span className="flex items-center gap-1">
                  {t("managePrompts.form.systemPrompt.labelImproved", {
                    defaultValue: "AI Instructions"
                  })}
                  <Tooltip title={t("managePrompts.form.systemPrompt.tooltip", {
                    defaultValue: "Also known as 'System prompt'. Sets how the AI should behave."
                  })}>
                    <Info className="size-3 text-text-muted cursor-help" />
                  </Tooltip>
                </span>
              }
              help={
                <span>
                  {t("managePrompts.form.systemPrompt.help", {
                    defaultValue: "Sets the AI's behavior and persona. Sent as the system message."
                  })}
                  <button
                    type="button"
                    className="ml-1 text-primary hover:underline text-xs"
                    onClick={() => setShowSystemHelp(!showSystemHelp)}
                  >
                    {showSystemHelp
                      ? t("common:showLess", { defaultValue: "Show less" })
                      : t("common:learnMore", { defaultValue: "Learn more" })}
                  </button>
                </span>
              }
              extra={renderPromptFieldInsights({
                value: systemPromptValue,
                variables: systemTemplateVariables,
                counterTestId: "prompt-drawer-system-counter",
                variablesTestId: "prompt-drawer-system-vars",
                previewTestId: "prompt-drawer-system-preview"
              })}
            >
              <Input.TextArea
                placeholder={t("managePrompts.form.systemPrompt.placeholder", {
                  defaultValue: "Optional system prompt sent as the system message"
                })}
                autoSize={{ minRows: 3, maxRows: 10 }}
                data-testid="prompt-drawer-system"
                disabled={promptFormat === "structured"}
              />
            </Form.Item>

            {/* Expandable help for system prompt */}
            {showSystemHelp && (
              <div className="bg-surface2 p-3 rounded-md text-xs text-text-muted -mt-2 mb-2">
                <p className="font-medium mb-2">
                  {t("managePrompts.form.systemPrompt.helpTitle", {
                    defaultValue: "What are AI Instructions?"
                  })}
                </p>
                <p className="mb-2">
                  {t("managePrompts.form.systemPrompt.helpDesc", {
                    defaultValue: "AI Instructions (system prompts) define how the AI should behave throughout the conversation. They're sent before any user messages and set the context, tone, and capabilities."
                  })}
                </p>
                <p className="font-medium mb-1">
                  {t("managePrompts.form.systemPrompt.helpExampleTitle", {
                    defaultValue: "Example:"
                  })}
                </p>
                <pre className="bg-surface p-2 rounded text-xs overflow-x-auto">
                  {`You are a helpful code review assistant.\nFocus on:\n- Code quality and best practices\n- Performance implications\n- Security concerns`}
                </pre>
              </div>
            )}

            <Form.Item
              name="user_prompt"
              rules={[
                {
                  validator: templateFieldValidator
                }
              ]}
              label={
                <span className="flex items-center gap-1">
                  {t("managePrompts.form.userPrompt.labelImproved", {
                    defaultValue: "Message Template"
                  })}
                  <Tooltip title={t("managePrompts.form.userPrompt.tooltip", {
                    defaultValue: "Also known as 'User prompt'. A template you can quickly insert."
                  })}>
                    <Info className="size-3 text-text-muted cursor-help" />
                  </Tooltip>
                </span>
              }
              help={
                <span>
                  {t("managePrompts.form.userPrompt.help", {
                    defaultValue: "Template inserted as the user message when using this prompt."
                  })}
                  <button
                    type="button"
                    className="ml-1 text-primary hover:underline text-xs"
                    onClick={() => setShowUserHelp(!showUserHelp)}
                  >
                    {showUserHelp
                      ? t("common:showLess", { defaultValue: "Show less" })
                      : t("common:learnMore", { defaultValue: "Learn more" })}
                  </button>
                </span>
              }
              extra={renderPromptFieldInsights({
                value: userPromptValue,
                variables: userTemplateVariables,
                counterTestId: "prompt-drawer-user-counter",
                variablesTestId: "prompt-drawer-user-vars",
                previewTestId: "prompt-drawer-user-preview"
              })}
            >
              <Input.TextArea
                placeholder={t("managePrompts.form.userPrompt.placeholder", {
                  defaultValue: "Optional user prompt template"
                })}
                autoSize={{ minRows: 3, maxRows: 10 }}
                data-testid="prompt-drawer-user"
                disabled={promptFormat === "structured"}
              />
            </Form.Item>

            {/* Expandable help for user prompt */}
            {showUserHelp && (
              <div className="bg-surface2 p-3 rounded-md text-xs text-text-muted -mt-2 mb-2">
                <p className="font-medium mb-2">
                  {t("managePrompts.form.userPrompt.helpTitle", {
                    defaultValue: "What are Message Templates?"
                  })}
                </p>
                <p className="mb-2">
                  {t("managePrompts.form.userPrompt.helpDesc", {
                    defaultValue: "Message templates are pre-written text you can quickly insert into your chat input. They save time on repetitive requests and ensure consistent phrasing."
                  })}
                </p>
                <p className="font-medium mb-1">
                  {t("managePrompts.form.userPrompt.helpExampleTitle", {
                    defaultValue: "Example:"
                  })}
                </p>
                <pre className="bg-surface p-2 rounded text-xs overflow-x-auto">
                  {`Please review the following code and provide:\n1. A brief summary\n2. Potential issues\n3. Suggestions for improvement\n\nCode:\n{paste your code here}`}
                </pre>
              </div>
            )}

            {promptFormat === "structured" && (
              <StructuredPromptEditor
                value={structuredPromptDefinition}
                onChange={setStructuredPromptDefinition}
                previewResult={structuredPreviewResult}
                previewLoading={structuredPreviewLoading}
                onPreview={handleStructuredPreview}
              />
            )}
          </div>
        </div>

        {/* Section: Organization */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-text-muted mb-3">
            {t("managePrompts.drawer.sectionOrganization", { defaultValue: "Organization" })}
          </h3>
          <div className="space-y-4">
            <Form.Item
              name="keywords"
              label={t("managePrompts.tags.label", { defaultValue: "Keywords" })}
            >
              <Select
                mode="tags"
                allowClear
                placeholder={t("managePrompts.tags.placeholder", {
                  defaultValue: "Add keywords"
                })}
                options={allTags.map((tag) => ({ label: tag, value: tag }))}
                data-testid="prompt-drawer-keywords"
              />
            </Form.Item>

            <Form.Item
              name="details"
              label={t("managePrompts.form.details.label", {
                defaultValue: "Notes"
              })}
            >
              <Input.TextArea
                placeholder={t("managePrompts.form.details.placeholder", {
                  defaultValue: "Add context or usage notes"
                })}
                autoSize={{ minRows: 2, maxRows: 6 }}
                data-testid="prompt-drawer-details"
              />
            </Form.Item>
          </div>
        </div>

        {/* Collapsible Advanced/Sync Sections */}
        {collapseItems.length > 0 && (
          <Collapse
            items={collapseItems}
            bordered={false}
            className="bg-transparent"
            expandIconPlacement="end"
          />
        )}
      </Form>
      {canViewVersionHistory && (
        <VersionHistoryDrawer
          open={versionHistoryOpen}
          promptId={versionHistoryPromptId}
          onClose={() => setVersionHistoryOpen(false)}
        />
      )}
      <Modal
        open={closeConfirmOpen}
        title={t("managePrompts.drawer.unsavedTitle", {
          defaultValue: "Unsaved changes"
        })}
        onCancel={() => setCloseConfirmOpen(false)}
        footer={
          <div className="flex justify-end gap-2">
            <Button
              onClick={() => setCloseConfirmOpen(false)}
              data-testid="prompt-drawer-unsaved-cancel"
            >
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button danger onClick={closeWithoutSaving} data-testid="prompt-drawer-unsaved-discard">
              {t("common:discard", { defaultValue: "Discard" })}
            </Button>
            <Button
              type="primary"
              onClick={() => {
                setCloseConfirmOpen(false)
                form.submit()
              }}
              data-testid="prompt-drawer-unsaved-save"
            >
              {t("common:save", { defaultValue: "Save" })}
            </Button>
          </div>
        }
      >
        <p>
          {t("managePrompts.drawer.unsavedDescription", {
            defaultValue: "You have unsaved changes. Close anyway?"
          })}
        </p>
      </Modal>
    </Drawer>
  )
}
