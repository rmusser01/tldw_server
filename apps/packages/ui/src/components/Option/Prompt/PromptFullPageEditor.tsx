import React, { useCallback, useEffect, useState } from "react"
import { Collapse, Form, Input, Select } from "antd"
import { useTranslation } from "react-i18next"
import { ArrowLeft, Save } from "lucide-react"
import { Alert } from "@/components/ui/primitives"
import type { PromptFormat, StructuredPromptDefinition } from "@/db/dexie/types"
import {
  previewStructuredPromptServer,
  type StructuredPromptPreviewResponse
} from "@/services/prompts-api"
import { PromptEditorPreview } from "./PromptEditorPreview"
import { StructuredPromptEditor } from "./Structured/StructuredPromptEditor"
import { useFormDraft, formatDraftAge } from "@/hooks/useFormDraft"
import {
  estimatePromptTokens,
  getPromptTokenBudgetState,
} from "./prompt-length-utils"
import { validateTemplateVariableSyntax } from "./prompt-template-variable-utils"
import {
  convertLegacyPromptToStructuredDefinition,
  createDefaultStructuredPromptDefinition,
  renderStructuredPromptLegacySnapshot,
  stableSerializePromptSnapshot
} from "./structured-prompt-utils"

const { TextArea } = Input

type PromptFullPageEditorProps = {
  open: boolean
  onClose: () => void
  mode: "create" | "edit"
  initialValues?: Record<string, any> | null
  onSubmit: (values: any) => void
  isLoading: boolean
  allTags: string[]
}

const DRAFT_KEY_PREFIX = "tldw-prompt-fullpage-draft-"

const normalizePromptDraftSnapshot = (
  values: Record<string, any> | null | undefined
) => {
  const normalizeString = (value: unknown) =>
    typeof value === "string" ? value.trim() : ""
  const keywords = Array.isArray(values?.keywords)
    ? values.keywords
        .map((keyword: unknown) =>
          typeof keyword === "string" ? keyword.trim() : ""
        )
        .filter((keyword: string) => keyword.length > 0)
        .sort()
    : []

  return {
    name: normalizeString(values?.name),
    author: normalizeString(values?.author),
    details: normalizeString(values?.details),
    system_prompt: normalizeString(values?.system_prompt),
    user_prompt: normalizeString(values?.user_prompt),
    promptFormat:
      values?.promptFormat === "structured" ? "structured" : "legacy",
    structuredPromptDefinition:
      values?.promptFormat === "structured" &&
      values?.structuredPromptDefinition &&
      typeof values.structuredPromptDefinition === "object"
        ? values.structuredPromptDefinition
        : null,
    keywords,
    changeDescription: normalizeString(values?.changeDescription)
  }
}

export const PromptFullPageEditor: React.FC<PromptFullPageEditorProps> = ({
  open,
  onClose,
  mode,
  initialValues,
  onSubmit,
  isLoading,
  allTags,
}) => {
  const { t } = useTranslation(["settings", "common"])
  const [form] = Form.useForm()
  const [dirty, setDirty] = useState(false)
  const [showMobilePreview, setShowMobilePreview] = useState(false)
  const [promptFormat, setPromptFormat] = useState<PromptFormat>("legacy")
  const [structuredPromptDefinition, setStructuredPromptDefinition] =
    useState<StructuredPromptDefinition>(createDefaultStructuredPromptDefinition())
  const [structuredPreviewResult, setStructuredPreviewResult] =
    useState<StructuredPromptPreviewResponse | null>(null)
  const [structuredPreviewLoading, setStructuredPreviewLoading] = useState(false)

  const draftKey = `${DRAFT_KEY_PREFIX}${mode === "edit" ? initialValues?.id || "new" : "new"}`
  const { hasDraft, draftData, clearDraft, saveDraft, applyDraft, lastSaved } = useFormDraft({
    storageKey: draftKey,
    formType: mode,
    editId: mode === "edit" ? initialValues?.id : undefined,
    autoSaveInterval: 5000,
  })

  const systemPromptValue = Form.useWatch("system_prompt", form) || ""
  const userPromptValue = Form.useWatch("user_prompt", form) || ""
  const initialSnapshot = React.useMemo(
    () => normalizePromptDraftSnapshot(initialValues),
    [initialValues]
  )

  useEffect(() => {
    if (!open) return

    if (hasDraft && draftData) {
      const recovered = applyDraft()
      if (recovered) {
        form.setFieldsValue(recovered)
        setPromptFormat(
          recovered.promptFormat === "structured" ? "structured" : "legacy"
        )
        setStructuredPromptDefinition(
          recovered.structuredPromptDefinition ||
            createDefaultStructuredPromptDefinition()
        )
        setStructuredPreviewResult(null)
        setDirty(false)
        return
      }
    }

    if (initialValues) {
      form.setFieldsValue({
        name: initialValues.name || "",
        author: initialValues.author || "",
        details: initialValues.details || "",
        system_prompt: initialValues.system_prompt || "",
        user_prompt: initialValues.user_prompt || "",
        promptFormat:
          initialValues.promptFormat === "structured" ? "structured" : "legacy",
        promptSchemaVersion:
          initialValues.promptSchemaVersion ??
          (initialValues.promptFormat === "structured" ? 1 : null),
        structuredPromptDefinition:
          initialValues.structuredPromptDefinition ||
          createDefaultStructuredPromptDefinition(),
        keywords: initialValues.keywords || [],
        changeDescription: initialValues.changeDescription || "",
      })
      setPromptFormat(
        initialValues.promptFormat === "structured" ? "structured" : "legacy"
      )
      setStructuredPromptDefinition(
        initialValues.structuredPromptDefinition ||
          createDefaultStructuredPromptDefinition()
      )
    } else {
      form.resetFields()
      setPromptFormat("legacy")
      setStructuredPromptDefinition(createDefaultStructuredPromptDefinition())
    }

    setStructuredPreviewResult(null)
    setDirty(false)
  }, [open, initialValues, hasDraft, draftData, applyDraft, form])

  useEffect(() => {
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

  useEffect(() => {
    if (!open || !dirty) return
    saveDraft(form.getFieldsValue(true))
  }, [open, dirty, promptFormat, structuredPromptDefinition, form, saveDraft])

  useEffect(() => {
    if (!open || !dirty) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
    }
    window.addEventListener("beforeunload", handler)
    return () => window.removeEventListener("beforeunload", handler)
  }, [open, dirty])

  const handleRequestClose = useCallback(() => {
    const currentSnapshot = normalizePromptDraftSnapshot(form.getFieldsValue(true))
    const hasDirtyValues =
      dirty ||
      form.isFieldsTouched(true) ||
      stableSerializePromptSnapshot(currentSnapshot) !==
        stableSerializePromptSnapshot(initialSnapshot)

    if (hasDirtyValues) {
      const ok = window.confirm(
        t("managePrompts.drawer.unsavedChanges", {
          defaultValue: "You have unsaved changes. Discard them?",
        })
      )
      if (!ok) return
    }

    clearDraft()
    onClose()
  }, [clearDraft, dirty, form, initialSnapshot, onClose, t])

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault()
        form.submit()
      }
      if (e.key === "Escape") {
        e.preventDefault()
        handleRequestClose()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open, form, handleRequestClose])

  const handleFinish = (values: any) => {
    const keywords = Array.isArray(values.keywords)
      ? values.keywords
          .map((k: string) => (typeof k === "string" ? k.trim() : ""))
          .filter((k: string) => k.length > 0)
          .sort()
      : []
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
      keywords,
      name: (values.name || "").trim(),
      author: (values.author || "").trim(),
      details: (values.details || "").trim(),
      system_prompt: (
        structuredSnapshot?.systemPrompt ?? values.system_prompt ?? ""
      ).trim(),
      user_prompt: (
        structuredSnapshot?.userPrompt ?? values.user_prompt ?? ""
      ).trim(),
    })
    clearDraft()
    setDirty(false)
  }

  const handleConvertToStructured = useCallback(() => {
    const currentValues = form.getFieldsValue(true)
    setPromptFormat("structured")
    setStructuredPromptDefinition(
      convertLegacyPromptToStructuredDefinition(
        currentValues?.system_prompt,
        currentValues?.user_prompt
      )
    )
    setStructuredPreviewResult(null)
    setDirty(true)
  }, [form])

  const handleStructuredPreview = useCallback(
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

  const templateFieldValidator = (_: any, value: string) => {
    if (!value) return Promise.resolve()
    const result = validateTemplateVariableSyntax(value)
    if (!result.isValid) {
      const msg = result.code === "unmatched_braces"
        ? "Unmatched {{ or }} braces"
        : `Invalid template variable: ${result.invalidTokens?.[0] ?? ""}`
      return Promise.reject(msg)
    }
    return Promise.resolve()
  }

  const structuredLegacySnapshot = React.useMemo(
    () =>
      promptFormat === "structured"
        ? renderStructuredPromptLegacySnapshot(structuredPromptDefinition)
        : null,
    [promptFormat, structuredPromptDefinition]
  )
  const previewSystemPrompt =
    structuredLegacySnapshot?.systemPrompt || systemPromptValue
  const previewUserPrompt =
    structuredLegacySnapshot?.userPrompt || userPromptValue
  const sysTokens = estimatePromptTokens(previewSystemPrompt)
  const userTokens = estimatePromptTokens(previewUserPrompt)
  const sysBudget = getPromptTokenBudgetState(sysTokens)
  const userBudget = getPromptTokenBudgetState(userTokens)

  const tokenLabel = (count: number, budget: string) => {
    const color =
      budget === "danger"
        ? "text-danger"
        : budget === "warning"
        ? "text-warning"
        : "text-text-muted"
    return (
      <span className={`text-xs ${color}`}>
        {count} chars / ~{Math.round(count / 4)} tokens
      </span>
    )
  }

  if (!open) return null

  const title =
    mode === "edit"
      ? initialValues?.name
        ? `Editing: ${initialValues.name}`
        : "Edit Prompt"
      : "New Prompt"

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-background"
      data-testid="prompt-full-page-editor"
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleRequestClose}
            className="flex items-center gap-1 rounded px-2 py-1 text-sm text-text-muted hover:bg-surface2 hover:text-text"
            data-testid="full-editor-back"
          >
            <ArrowLeft className="size-4" />
            Back to Prompts
          </button>
          <span className="text-sm font-medium text-text">{title}</span>
          {lastSaved && (
            <span className="text-xs text-text-muted">
              Draft saved {formatDraftAge(lastSaved)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRequestClose}
            className="rounded px-3 py-1.5 text-sm text-text-muted hover:bg-surface2"
            data-testid="full-editor-cancel"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => form.submit()}
            disabled={isLoading}
            className="flex items-center gap-1 rounded bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong disabled:opacity-50"
            data-testid="full-editor-save"
          >
            <Save className="size-3.5" />
            {isLoading ? "Saving..." : "Save"}
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <div className="flex-[55] overflow-y-auto border-r border-border p-6">
          <Form
            form={form}
            layout="vertical"
            onFinish={handleFinish}
            onValuesChange={() => setDirty(true)}
            className="mx-auto max-w-2xl space-y-6"
          >
            <div>
              <h3 className="mb-3 text-sm font-semibold text-text">Identity</h3>
              <div className="grid grid-cols-2 gap-4">
                <Form.Item
                  name="name"
                  label="Title"
                  rules={[{ required: true, message: "Title is required" }]}
                >
                  <Input
                    placeholder="e.g. Code Review Assistant"
                    data-testid="full-editor-name"
                  />
                </Form.Item>
                <Form.Item name="author" label="Author">
                  <Input placeholder="Optional" data-testid="full-editor-author" />
                </Form.Item>
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-sm font-semibold text-text">
                Prompt Content
              </h3>
              {promptFormat === "legacy" && (
                <div className="mb-4">
                  <button
                    type="button"
                    onClick={handleConvertToStructured}
                    className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:border-primary/40 hover:text-primary"
                  >
                    Convert to structured
                  </button>
                </div>
              )}
              {promptFormat === "structured" && (
                <Alert
                  variant="info"
                  className="mb-4"
                  title={t("managePrompts.fullEditor.structuredPromptTitle", {
                    defaultValue: "Structured prompt"
                  })}
                >
                  {t("managePrompts.fullEditor.structuredPromptDesc", {
                    defaultValue:
                      "This prompt now uses ordered blocks. The raw system and user fields are locked for compatibility; save and preview use the structured definition."
                  })}
                </Alert>
              )}
              <Form.Item
                name="system_prompt"
                label="AI Instructions (System Prompt)"
                rules={[{ validator: templateFieldValidator }]}
              >
                <TextArea
                  autoSize={{ minRows: 6, maxRows: 30 }}
                  placeholder="You are a helpful assistant that..."
                  data-testid="full-editor-system-prompt"
                  disabled={promptFormat === "structured"}
                />
              </Form.Item>
              <div className="mb-4 -mt-2">
                {tokenLabel(previewSystemPrompt.length, sysBudget)}
              </div>

              <Form.Item
                name="user_prompt"
                label="Message Template (User Prompt)"
                rules={[{ validator: templateFieldValidator }]}
              >
                <TextArea
                  autoSize={{ minRows: 4, maxRows: 20 }}
                  placeholder="Analyze the following: {{input}}"
                  data-testid="full-editor-user-prompt"
                  disabled={promptFormat === "structured"}
                />
              </Form.Item>
              <div className="-mt-2">
                {tokenLabel(previewUserPrompt.length, userBudget)}
              </div>

              {promptFormat === "structured" && (
                <div className="mt-4">
                  <StructuredPromptEditor
                    value={structuredPromptDefinition}
                    onChange={(nextValue) => {
                      setStructuredPromptDefinition(nextValue)
                      setStructuredPreviewResult(null)
                      setDirty(true)
                    }}
                    previewResult={structuredPreviewResult}
                    previewLoading={structuredPreviewLoading}
                    onPreview={handleStructuredPreview}
                  />
                </div>
              )}
            </div>

            <div>
              <h3 className="mb-3 text-sm font-semibold text-text">
                Organization
              </h3>
              <Form.Item name="keywords" label="Tags">
                <Select
                  mode="tags"
                  placeholder="Add tags..."
                  options={allTags.map((tag) => ({ label: tag, value: tag }))}
                  data-testid="full-editor-keywords"
                />
              </Form.Item>
              <Form.Item name="details" label="Notes">
                <TextArea
                  autoSize={{ minRows: 2, maxRows: 6 }}
                  placeholder="Internal notes about this prompt..."
                  data-testid="full-editor-details"
                />
              </Form.Item>
            </div>

            <Collapse
              ghost
              items={[
                {
                  key: "advanced",
                  label: (
                    <span className="text-sm font-medium text-text-muted">
                      Advanced
                    </span>
                  ),
                  children: (
                    <div className="space-y-4">
                      <Form.Item
                        name="changeDescription"
                        label="Change Description"
                      >
                        <Input placeholder="What changed in this version?" />
                      </Form.Item>
                    </div>
                  ),
                },
              ]}
            />
          </Form>
        </div>

        <div className="hidden flex-[45] md:flex flex-col bg-surface">
          <PromptEditorPreview
            systemPrompt={previewSystemPrompt}
            userPrompt={previewUserPrompt}
          />
        </div>
      </div>

      <div className="md:hidden">
        <button
          type="button"
          onClick={() => setShowMobilePreview(!showMobilePreview)}
          className="fixed bottom-4 right-4 z-50 rounded-full bg-primary px-4 py-2 text-sm font-medium text-white shadow-lg"
        >
          {showMobilePreview ? "Editor" : "Preview"}
        </button>
        {showMobilePreview && (
          <div className="fixed inset-0 z-40 bg-background pt-12">
            <PromptEditorPreview
              systemPrompt={previewSystemPrompt}
              userPrompt={previewUserPrompt}
            />
          </div>
        )}
      </div>
    </div>
  )
}
