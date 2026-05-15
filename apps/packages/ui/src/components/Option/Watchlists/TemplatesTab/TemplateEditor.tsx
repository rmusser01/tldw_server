import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import {
  Alert,
  Button,
  Checkbox,
  Divider,
  Form,
  Input,
  Modal,
  Radio,
  Select,
  Space,
  Tabs,
  message
} from "antd"
import { useTranslation } from "react-i18next"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  composeWatchlistTemplateSection,
  createWatchlistTemplate,
  fetchWatchlistRuns,
  getWatchlistTemplate,
  getWatchlistTemplateVersions,
  validateWatchlistTemplate,
  type TemplateComposerFlowSection
} from "@/services/watchlists"
import type {
  WatchlistTemplate,
  WatchlistTemplateCreate,
  WatchlistTemplateVersionSummary
} from "@/types/watchlists"
import { WatchlistsHelpTooltip } from "../shared"
import { TemplateCodeEditor, type TemplateCodeEditorHandle } from "./TemplateCodeEditor"
import { TemplateVariablesPanel } from "./TemplateVariablesPanel"
import { TemplateSnippetPalette } from "./TemplateSnippetPalette"
import { TemplatePreviewPane } from "./TemplatePreviewPane"
import { VisualComposerPane } from "./VisualComposerPane"
import {
  buildTemplateSavePayload,
  hasTemplateAdvancedContext,
  shouldWarnOnTemplateModeChange,
  type TemplateAuthoringMode
} from "./template-mode"
import {
  buildTemplateFromRecipe,
  createDefaultTemplateRecipeOptions,
  TEMPLATE_RECIPE_DEFINITIONS,
  type TemplateRecipeId,
  type TemplateRecipeOptions
} from "./template-recipes"
import {
  createEmptyComposerAst,
  type ComposerAst
} from "./composer-types"
import {
  compileComposerAstToTemplate,
  computeComposerSyncHash,
  parseTemplateToComposerAst
} from "./composer-roundtrip"
import { trackWatchlistsPreventionTelemetry } from "@/utils/watchlists-prevention-telemetry"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"

interface TemplateEditorProps {
  template: WatchlistTemplate | null
  open: boolean
  onClose: (saved?: boolean) => void
}

export const TemplateEditor: React.FC<TemplateEditorProps> = ({
  template,
  open,
  onClose
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)
  const [form] = Form.useForm()
  const [saving, setSaving] = useState(false)
  const [loadingVersion, setLoadingVersion] = useState(false)
  const [activeTab, setActiveTab] = useState<"visual" | "editor" | "preview" | "docs">("visual")
  const [authoringMode, setAuthoringMode] = useState<TemplateAuthoringMode>("basic")
  const [templateVersions, setTemplateVersions] = useState<WatchlistTemplateVersionSummary[]>([])
  const [selectedVersion, setSelectedVersion] = useState<number | undefined>(undefined)
  const [loadedVersion, setLoadedVersion] = useState<number | null>(null)
  const [loadedContentBaseline, setLoadedContentBaseline] = useState("")
  const [composerAst, setComposerAst] = useState<ComposerAst>(createEmptyComposerAst())
  const [composerNeedsRepair, setComposerNeedsRepair] = useState(false)
  const [selectedComposeRunId, setSelectedComposeRunId] = useState<number | undefined>(undefined)
  const editorRef = useRef<TemplateCodeEditorHandle>(null)
  const [validationErrors, setValidationErrors] = useState<Array<{ line?: number | null; column?: number | null; message: string }>>([])
  const [availableRuns, setAvailableRuns] = useState<Array<{ id: number; label: string }>>([])
  const [selectedRecipeId, setSelectedRecipeId] = useState<TemplateRecipeId>("briefing_md")
  const [recipeOptions, setRecipeOptions] = useState<TemplateRecipeOptions>(
    createDefaultTemplateRecipeOptions()
  )
  const restoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)

  const isEditing = !!template
  const authoringContext = isEditing ? "edit" : "create"
  const formatValue = Form.useWatch("format", form)
  const contentValue = Form.useWatch("content", form)
  const selectedRecipeDefinition = useMemo(
    () =>
      TEMPLATE_RECIPE_DEFINITIONS.find((definition) => definition.id === selectedRecipeId) ||
      TEMPLATE_RECIPE_DEFINITIONS[0],
    [selectedRecipeId]
  )

  const syncComposerFromContent = (content: string) => {
    setComposerAst(parseTemplateToComposerAst(content))
  }

  const applyComposerAst = (nextAst: ComposerAst) => {
    setComposerAst(nextAst)
    setComposerNeedsRepair(false)
    const compiled = compileComposerAstToTemplate(nextAst)
    form.setFieldsValue({ content: compiled })
  }

  useLayoutEffect(() => {
    if (open) {
      if (!wasOpenRef.current) {
        restoreFocusTargetRef.current = getFocusableActiveElement()
      }
      wasOpenRef.current = true
      return
    }

    if (wasOpenRef.current) {
      wasOpenRef.current = false
      restoreFocusToElement(restoreFocusTargetRef.current)
    }
  }, [open])

  const loadTemplate = async (templateName: string, version?: number) => {
    setLoadingVersion(true)
    try {
      const result = await getWatchlistTemplate(templateName, version ? { version } : undefined)
      const resolvedContent = result.content || ""
      const resolvedComposerAst =
        result.composer_ast && typeof result.composer_ast === "object"
          ? (result.composer_ast as unknown as ComposerAst)
          : parseTemplateToComposerAst(resolvedContent)
      form.setFieldsValue({
        name: result.name,
        description: result.description || "",
        content: resolvedContent,
        format: result.format || "html"
      })
      setLoadedVersion(typeof result.version === "number" ? result.version : null)
      setLoadedContentBaseline(resolvedContent)
      setComposerAst(resolvedComposerAst)
      setComposerNeedsRepair(result.composer_sync_status === "needs_repair")
    } finally {
      setLoadingVersion(false)
    }
  }

  useEffect(() => {
    if (open && template) {
      setSelectedComposeRunId(undefined)
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_authoring_started",
        surface: "template_editor",
        mode: "advanced",
        context: "edit"
      })
      Promise.all([
        getWatchlistTemplate(template.name),
        getWatchlistTemplateVersions(template.name).catch(() => ({ items: [] }))
      ])
        .then(([result, versions]) => {
          const resolvedContent = result.content || ""
          const resolvedComposerAst =
            result.composer_ast && typeof result.composer_ast === "object"
              ? (result.composer_ast as unknown as ComposerAst)
              : parseTemplateToComposerAst(resolvedContent)
          form.setFieldsValue({
            name: result.name,
            description: result.description || "",
            content: resolvedContent,
            format: result.format || "html"
          })
          setLoadedVersion(typeof result.version === "number" ? result.version : null)
          setLoadedContentBaseline(resolvedContent)
          setComposerAst(resolvedComposerAst)
          setComposerNeedsRepair(result.composer_sync_status === "needs_repair")
          setTemplateVersions(Array.isArray(versions.items) ? versions.items : [])
          setSelectedVersion(undefined)
          setAuthoringMode("advanced")
          setActiveTab("visual")
          setSelectedRecipeId("briefing_md")
          setRecipeOptions(createDefaultTemplateRecipeOptions())
        })
        .catch((err) => {
          console.error("Failed to load template:", err)
          message.error(t("watchlists:templates.loadError", "Failed to load template"))
          setTemplateVersions([])
          setSelectedVersion(undefined)
          setLoadedVersion(null)
          setLoadedContentBaseline("")
          setComposerAst(createEmptyComposerAst())
          setComposerNeedsRepair(false)
          setAuthoringMode("advanced")
          setActiveTab("visual")
          setSelectedRecipeId("briefing_md")
          setRecipeOptions(createDefaultTemplateRecipeOptions())
        })
    } else if (open) {
      setSelectedComposeRunId(undefined)
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_authoring_started",
        surface: "template_editor",
        mode: "basic",
        context: "create"
      })
      form.resetFields()
      form.setFieldsValue({ format: "html", content: DEFAULT_HTML_TEMPLATE })
      setTemplateVersions([])
      setSelectedVersion(undefined)
      setLoadedVersion(null)
      setLoadedContentBaseline(DEFAULT_HTML_TEMPLATE)
      setComposerAst(parseTemplateToComposerAst(DEFAULT_HTML_TEMPLATE))
      setComposerNeedsRepair(false)
      setAuthoringMode("basic")
      setActiveTab("visual")
      setSelectedRecipeId("briefing_md")
      setRecipeOptions(createDefaultTemplateRecipeOptions())
    }
  }, [open, template, form, t])

  useEffect(() => {
    if (!open || activeTab !== "visual") return
    const compiled = compileComposerAstToTemplate(composerAst).trim()
    const content = String(contentValue || "").trim()
    if (compiled !== content) {
      setComposerNeedsRepair(true)
    }
  }, [activeTab, composerAst, contentValue, open])

  // Fetch recent completed runs for live preview
  useEffect(() => {
    if (!open) {
      setAvailableRuns([])
      return
    }
    fetchWatchlistRuns({
      watchlist_id: selectedWatchlistId ?? undefined,
      q: "completed",
      size: 20
    })
      .then((res) => {
        setAvailableRuns(
          (res.items || []).map((r) => ({
            id: r.id,
            label: `Run #${r.id}${r.started_at ? ` - ${r.started_at}` : ""}`,
          }))
        )
      })
      .catch(() => {
        setAvailableRuns([])
      })
  }, [open, selectedWatchlistId])

  const handleLoadSelectedVersion = async () => {
    if (!template || !selectedVersion) return
    try {
      await loadTemplate(template.name, selectedVersion)
      message.success(
        t("watchlists:templates.versionLoaded", "Loaded template version {{version}}", { version: selectedVersion })
      )
    } catch (err) {
      console.error("Failed to load template version:", err)
      message.error(t("watchlists:templates.versionLoadError", "Failed to load template version"))
    }
  }

  const handleLoadLatest = async () => {
    if (!template) return
    try {
      await loadTemplate(template.name)
      setSelectedVersion(undefined)
      message.success(t("watchlists:templates.latestLoaded", "Loaded latest template version"))
    } catch (err) {
      console.error("Failed to load latest template version:", err)
      message.error(t("watchlists:templates.latestLoadError", "Failed to load latest template"))
    }
  }

  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      // Server-side validation
      try {
        const validation = await validateWatchlistTemplate(values.content, values.format)
        if (!validation.valid) {
          setValidationErrors(validation.errors)
          message.error(
            t(
              "watchlists:templates.syntaxErrorBeforeSave",
              "Could not save template. Fix syntax errors, then try again."
            )
          )
          return
        }
        setValidationErrors([])
      } catch (validationErr) {
        // Validation endpoint unavailable — clear stale markers and proceed
        console.warn("Template validation endpoint unavailable:", validationErr)
        setValidationErrors([])
      }
      setSaving(true)
      const content = String(values.content || "")
      const currentAstMatchesContent =
        compileComposerAstToTemplate(composerAst).trim() === content.trim()
      const astToSave = currentAstMatchesContent
        ? composerAst
        : parseTemplateToComposerAst(content)
      const compiledFromAst = compileComposerAstToTemplate(astToSave)
      const composerSyncStatus =
        compiledFromAst.trim() === content.trim() ? "in_sync" : "needs_repair"
      if (!currentAstMatchesContent) {
        setComposerAst(astToSave)
      }
      const payload: WatchlistTemplateCreate = {
        ...buildTemplateSavePayload(values, isEditing),
        composer_ast: astToSave as unknown as Record<string, unknown>,
        composer_schema_version: astToSave.schema_version || "1.0.0",
        composer_sync_hash: computeComposerSyncHash(content, astToSave),
        composer_sync_status: composerSyncStatus
      }
      await createWatchlistTemplate(payload)
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_authoring_saved",
        surface: "template_editor",
        mode: authoringMode,
        context: authoringContext
      })
      message.success(
        isEditing
          ? t("watchlists:templates.updated", "Template updated")
          : t("watchlists:templates.created", "Template created")
      )
      onClose(true)
    } catch (err: any) {
      if (err.errorFields) return
      console.error("Failed to save template:", err)
      message.error(t("watchlists:templates.saveError", "Failed to save template"))
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    if (!open || isEditing) return
    const touched = form.isFieldTouched("content")
    if (touched) return
    if (formatValue === "md") {
      form.setFieldsValue({ content: DEFAULT_MARKDOWN_TEMPLATE })
    } else if (formatValue === "html") {
      form.setFieldsValue({ content: DEFAULT_HTML_TEMPLATE })
    }
  }, [formatValue, form, isEditing, open])

  useEffect(() => {
    if (!open || activeTab !== "visual") return
    syncComposerFromContent(String(form.getFieldValue("content") || ""))
  }, [activeTab, form, open])

  const hasVersionDrift = useMemo(() => {
    if (!isEditing || typeof loadedVersion !== "number") return false
    return String(contentValue || "") !== loadedContentBaseline
  }, [contentValue, isEditing, loadedContentBaseline, loadedVersion])

  const composerSections = useMemo<TemplateComposerFlowSection[]>(
    () =>
      (composerAst.nodes || [])
        .map((node) => ({
          id: node.id,
          content: String(node.source || "")
        }))
        .filter((section) => section.content.trim().length > 0),
    [composerAst]
  )

  const applyFlowSectionsToComposer = (sections: TemplateComposerFlowSection[]) => {
    if (!Array.isArray(sections) || sections.length === 0) return
    const sectionMap = new Map(sections.map((section) => [section.id, section.content]))
    const nextAst: ComposerAst = {
      ...composerAst,
      nodes: (composerAst.nodes || []).map((node) =>
        sectionMap.has(node.id)
          ? { ...node, source: String(sectionMap.get(node.id) || "") }
          : node
      )
    }
    applyComposerAst(nextAst)
  }

  const handleRepairVisualLayout = () => {
    const currentContent = String(form.getFieldValue("content") || "")
    setComposerAst(parseTemplateToComposerAst(currentContent))
    setComposerNeedsRepair(false)
    message.success(
      t(
        "watchlists:templates.repairVisualLayoutSuccess",
        "Visual layout re-synced from the current template code."
      )
    )
  }

  const hasAdvancedTemplateContext = hasTemplateAdvancedContext({
    isEditing,
    selectedVersion,
    activeTab,
    hasVersionDrift,
    validationErrorCount: validationErrors.length
  })

  const handleAuthoringModeChange = (nextMode: TemplateAuthoringMode) => {
    if (shouldWarnOnTemplateModeChange({
      currentMode: authoringMode,
      nextMode,
      hasAdvancedContext: hasAdvancedTemplateContext
    })) {
      message.info(
        t(
          "watchlists:templates.modeHiddenToolsNotice",
          "Advanced template tools are hidden in Basic mode. Your content and version context are preserved."
        )
      )
    }
    if (authoringMode !== nextMode && nextMode === "basic" && activeTab === "docs") {
      setActiveTab("editor")
    }
    void trackWatchlistsPreventionTelemetry({
      type: "watchlists_authoring_mode_changed",
      surface: "template_editor",
      from_mode: authoringMode,
      to_mode: nextMode,
      context: authoringContext
    })
    setAuthoringMode(nextMode)
  }

  const insertSnippet = (snippet: string) => {
    let newContent = ""
    if (editorRef.current) {
      editorRef.current.insertSnippet(snippet)
      // Sync form value from editor
      const newVal = editorRef.current.getValue()
      newContent = String(newVal || "")
      form.setFieldsValue({ content: newContent })
    } else {
      const current = String(form.getFieldValue("content") || "")
      const needsSpacer = current.length > 0 && !current.endsWith("\n")
      newContent = `${current}${needsSpacer ? "\n\n" : ""}${snippet}`
      form.setFieldsValue({ content: newContent })
    }
    syncComposerFromContent(newContent)
    setActiveTab("editor")
  }

  const setRecipeOption = (
    key: keyof TemplateRecipeOptions,
    checked: boolean
  ) => {
    setRecipeOptions((previous) => ({
      ...previous,
      [key]: checked
    }))
  }

  const handleApplyRecipe = () => {
    const generated = buildTemplateFromRecipe(selectedRecipeId, recipeOptions)
    const valuesToSet: Record<string, string> = {
      format: generated.format,
      content: generated.content
    }
    if (!isEditing) {
      const currentName = String(form.getFieldValue("name") || "").trim()
      const currentDescription = String(form.getFieldValue("description") || "").trim()
      if (!currentName) {
        valuesToSet.name = generated.suggestedName
      }
      if (!currentDescription) {
        valuesToSet.description = generated.suggestedDescription
      }
    }
    form.setFieldsValue(valuesToSet)
    form.setFields([
      { name: "content", touched: true },
      { name: "format", touched: true }
    ])
    syncComposerFromContent(generated.content)
    void trackWatchlistsPreventionTelemetry({
      type: "watchlists_template_recipe_applied",
      surface: "template_editor",
      recipe: selectedRecipeId,
      mode: authoringMode
    })
    setValidationErrors([])
    message.success(t("watchlists:templates.recipe.applied", "Template recipe applied."))
  }

  const handleGenerateComposerSection = async (input: {
    run_id: number
    block_id: string
    prompt: string
    input_scope: "all_items" | "top_items" | "selected_items"
    style?: string
    length_target: "short" | "medium" | "long"
  }) => composeWatchlistTemplateSection(input)

  const QUICK_SNIPPETS = [
    { label: t("watchlists:templates.snippetLoop", "Items loop"), snippet: "{% for item in items %}\n{{ item.title }}\n{% endfor %}" },
    { label: t("watchlists:templates.snippetGroupsLoop", "Groups loop"), snippet: "{% for group in groups %}\n## {{ group.name }}\n{% for item in group.items %}\n- {{ item.title }}\n{% endfor %}\n{% endfor %}" },
    { label: t("watchlists:templates.snippetBriefingSummary", "Briefing summary"), snippet: "{% if has_briefing_summary %}\n{{ briefing_summary }}\n{% endif %}" },
    { label: t("watchlists:templates.snippetSummary", "Summary block"), snippet: "{% if item.summary %}\n{{ item.summary }}\n{% endif %}" },
    { label: t("watchlists:templates.snippetGeneratedAt", "Generated timestamp"), snippet: "{{ generated_at }}" },
  ]

  const tabItems = [
    {
      key: "visual",
      label: t("watchlists:templates.visual.tab", "Visual"),
      children: (
        <VisualComposerPane
          ast={composerAst}
          onChange={applyComposerAst}
          runs={availableRuns}
          selectedRunId={selectedComposeRunId}
          onSelectedRunIdChange={setSelectedComposeRunId}
          onGenerateSection={handleGenerateComposerSection}
        />
      )
    },
    {
      key: "editor",
      label: t("watchlists:templates.editor.tab", "Editor"),
      children: (
        authoringMode === "advanced" ? (
          <div className="space-y-3">
            <div className="rounded-lg border border-border p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="flex items-center gap-1 text-xs font-medium text-text-muted">
                  {t("watchlists:templates.quickInsert", "Quick insert snippets")}
                  <WatchlistsHelpTooltip topic="jinja2" />
                </div>
                <Button size="small" type="link" onClick={() => setActiveTab("docs")}>
                  {t("watchlists:templates.openDocs", "Open variables & sample context")}
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {QUICK_SNIPPETS.map((s, i) => (
                  <Button key={i} size="small" onClick={() => insertSnippet(s.snippet)}>
                    {s.label}
                  </Button>
                ))}
              </div>
            </div>
            <Form.Item
              name="content"
              rules={[{ required: true, message: t("watchlists:templates.contentRequired", "Template content is required") }]}
              className="mb-0"
            >
              <TemplateCodeEditor
                ref={editorRef}
                value={contentValue || ""}
                onChange={(v) => form.setFieldsValue({ content: v })}
                format={formatValue === "md" ? "md" : "html"}
                height={400}
                validationErrors={validationErrors}
              />
            </Form.Item>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="rounded-lg border border-border bg-surface p-3" data-testid="template-recipe-builder">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-medium text-text-muted">
                  {t("watchlists:templates.recipe.title", "Recipe builder")}
                </div>
                <Button
                  size="small"
                  data-testid="template-recipe-apply"
                  onClick={handleApplyRecipe}
                >
                  {t("watchlists:templates.recipe.apply", "Apply recipe")}
                </Button>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <div>
                  <div className="mb-1 text-xs text-text-muted">
                    {t("watchlists:templates.recipe.recipeType", "Template recipe")}
                  </div>
                  <Select
                    size="small"
                    value={selectedRecipeId}
                    onChange={(value) => setSelectedRecipeId(value as TemplateRecipeId)}
                    options={TEMPLATE_RECIPE_DEFINITIONS.map((definition) => ({
                      value: definition.id,
                      label: t(definition.labelKey, definition.fallbackLabel)
                    }))}
                  />
                  <div className="mt-1 text-xs text-text-muted">
                    {t(
                      selectedRecipeDefinition.descriptionKey,
                      selectedRecipeDefinition.fallbackDescription
                    )}
                  </div>
                </div>
                <div className="space-y-1">
                  <Checkbox
                    checked={recipeOptions.includeLinks}
                    onChange={(event) => setRecipeOption("includeLinks", event.target.checked)}
                  >
                    {t("watchlists:templates.recipe.options.includeLinks", "Include source links")}
                  </Checkbox>
                  {selectedRecipeDefinition.supports.executiveSummary ? (
                    <Checkbox
                      checked={recipeOptions.includeExecutiveSummary}
                      onChange={(event) =>
                        setRecipeOption("includeExecutiveSummary", event.target.checked)
                      }
                    >
                      {t(
                        "watchlists:templates.recipe.options.includeExecutiveSummary",
                        "Include executive summary section"
                      )}
                    </Checkbox>
                  ) : null}
                  {selectedRecipeDefinition.supports.publishedAt ? (
                    <Checkbox
                      checked={recipeOptions.includePublishedAt}
                      onChange={(event) =>
                        setRecipeOption("includePublishedAt", event.target.checked)
                      }
                    >
                      {t(
                        "watchlists:templates.recipe.options.includePublishedAt",
                        "Include published timestamp"
                      )}
                    </Checkbox>
                  ) : null}
                  {selectedRecipeDefinition.supports.tags ? (
                    <Checkbox
                      checked={recipeOptions.includeTags}
                      onChange={(event) => setRecipeOption("includeTags", event.target.checked)}
                    >
                      {t("watchlists:templates.recipe.options.includeTags", "Include item tags")}
                    </Checkbox>
                  ) : null}
                </div>
              </div>
            </div>
            <div className="text-xs text-text-muted">
              {t(
                "watchlists:templates.modeBasicEditorHint",
                "Start in Basic mode with plain text or Markdown. Switch to Advanced only if you need variables, loops, or version tools."
              )}
            </div>
            <Form.Item
              name="content"
              rules={[{ required: true, message: t("watchlists:templates.contentRequired", "Template content is required") }]}
              className="mb-0"
            >
              <Input.TextArea
                rows={18}
                value={contentValue || ""}
                onChange={(event) => form.setFieldsValue({ content: event.target.value })}
                placeholder={t(
                  "watchlists:templates.contentPlaceholder",
                  "Start with plain text or Markdown. Advanced users can add Jinja2 tags later."
                )}
              />
            </Form.Item>
          </div>
        )
      )
    },
    {
      key: "preview",
      label: t("watchlists:templates.preview.tab", "Preview"),
      children: (
        <TemplatePreviewPane
          content={contentValue || ""}
          format={formatValue === "md" ? "md" : "html"}
          runs={availableRuns}
          sections={composerSections}
          onApplyFlowSections={applyFlowSectionsToComposer}
        />
      )
    },
    {
      key: "docs",
      label: t("watchlists:templates.docs.tab", "Variables & Snippets"),
      children: (
        <div className="grid grid-cols-2 gap-4" style={{ minHeight: 300 }}>
          <div>
            <div className="text-xs font-semibold text-text-muted mb-2">{t("watchlists:templates.docs.variables", "Variables")}</div>
            <TemplateVariablesPanel onInsert={insertSnippet} />
          </div>
          <div>
            <div className="text-xs font-semibold text-text-muted mb-2">{t("watchlists:templates.docs.snippets", "Snippets")}</div>
            <TemplateSnippetPalette
              format={formatValue === "md" ? "md" : "html"}
              onInsert={insertSnippet}
            />
          </div>
        </div>
      )
    }
  ].filter((item) => authoringMode === "advanced" || item.key !== "docs")

  return (
    <Modal
      title={
        isEditing
          ? t("watchlists:templates.editTitle", "Edit Template")
          : t("watchlists:templates.createTitle", "Create Template")
      }
      open={open}
      onCancel={() => onClose()}
      width={1200}
      footer={
        <Space>
          <Button onClick={() => onClose()}>
            {t("common:cancel", "Cancel")}
          </Button>
          <Button type="primary" onClick={handleSave} loading={saving}>
            {t("common:save", "Save")}
          </Button>
        </Space>
      }
    >
      <Form form={form} layout="vertical" className="mt-4">
        <div className="mb-4 rounded-lg border border-border bg-surface p-3">
          <div className="mb-2 text-sm font-medium">
            {t("watchlists:templates.modeLabel", "Editing mode")}
          </div>
          <Radio.Group
            value={authoringMode}
            onChange={(event) =>
              handleAuthoringModeChange(event.target.value as TemplateAuthoringMode)
            }
            optionType="button"
            buttonStyle="solid"
            size="small"
          >
            <Radio.Button value="basic" data-testid="template-editor-mode-basic">
              {t("watchlists:templates.modeBasic", "Basic")}
            </Radio.Button>
            <Radio.Button value="advanced" data-testid="template-editor-mode-advanced">
              {t("watchlists:templates.modeAdvanced", "Advanced")}
            </Radio.Button>
          </Radio.Group>
          <div className="mt-2 text-xs text-text-muted">
            {authoringMode === "basic"
              ? t(
                "watchlists:templates.modeHelpBasic",
                "Basic mode is no-code: pick a recipe, edit text, and preview your output."
              )
              : t(
                "watchlists:templates.modeHelpAdvanced",
                "Advanced mode adds Jinja2 snippets, variable docs, and version tools."
              )}
          </div>
          {authoringMode === "basic" && hasAdvancedTemplateContext && (
            <div className="mt-2 text-xs text-text-muted">
              {t(
                "watchlists:templates.modeHiddenToolsNotice",
                "Advanced template tools are hidden in Basic mode. Your content and version context are preserved."
              )}
            </div>
          )}
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Form.Item
            name="name"
            label={t("watchlists:templates.fields.name", "Template Name")}
            rules={[{ required: true, message: t("watchlists:templates.nameRequired", "Name is required") }]}
          >
            <Input
              placeholder={t("watchlists:templates.namePlaceholder", "my-briefing-template")}
              disabled={isEditing}
            />
          </Form.Item>

          <Form.Item
            name="format"
            label={t("watchlists:templates.fields.format", "Output Format")}
          >
            <Radio.Group>
              <Radio value="html">HTML</Radio>
              <Radio value="md">Markdown</Radio>
            </Radio.Group>
          </Form.Item>

          <Form.Item
            name="description"
            label={t("watchlists:templates.fields.description", "Description")}
          >
            <Input placeholder={t("watchlists:templates.descriptionPlaceholder", "Optional description...")} />
          </Form.Item>
        </div>

        {isEditing && authoringMode === "advanced" && (
          <>
            <Divider className="my-3" />
            <div className="mb-4 rounded-lg border border-border p-3">
              <div className="mb-2 text-sm font-medium">
                {t("watchlists:templates.versionTools", "Version tools")}
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto_auto]">
                <Select
                  allowClear
                  value={selectedVersion}
                  onChange={(value) => setSelectedVersion(value)}
                  placeholder={t("watchlists:templates.selectVersion", "Select a historical version")}
                  options={templateVersions.map((entry) => ({
                    value: entry.version,
                    label: entry.is_current
                      ? t("watchlists:templates.versionCurrent", "v{{version}} (current)", { version: entry.version })
                      : t("watchlists:templates.versionLabel", "v{{version}}", { version: entry.version })
                  }))}
                />
                <Button onClick={handleLoadSelectedVersion} disabled={!selectedVersion} loading={loadingVersion}>
                  {t("watchlists:templates.loadVersion", "Load version")}
                </Button>
                <Button onClick={handleLoadLatest} loading={loadingVersion}>
                  {t("watchlists:templates.loadLatest", "Load latest")}
                </Button>
              </div>
              {typeof loadedVersion === "number" && (
                <div className="mt-2 text-xs text-text-muted">
                  {t("watchlists:templates.loadedVersionHint", "Currently loaded: v{{version}}. Saving restores this content as a new latest version.", { version: loadedVersion })}
                </div>
              )}
              {hasVersionDrift && (
                <Alert
                  className="mt-3"
                  type="warning"
                  showIcon
                  title={t("watchlists:templates.unsavedDrift", "Current editor content differs from the loaded version.")}
                />
              )}
            </div>
          </>
        )}

        {isEditing && authoringMode === "basic" && (
          <div className="mb-4 text-xs text-text-muted">
            {t(
              "watchlists:templates.modeVersionToolsHidden",
              "Version tools are available in Advanced mode."
            )}
          </div>
        )}

        {activeTab === "visual" && composerNeedsRepair && (
          <Alert
            className="mb-4"
            type="warning"
            showIcon
            title={t(
              "watchlists:templates.visualOutOfSyncTitle",
              "Visual layout may be out of sync"
            )}
            description={t(
              "watchlists:templates.visualOutOfSyncDescription",
              "The template was edited in code mode and the visual block layout may need repair."
            )}
            action={(
              <Button
                size="small"
                onClick={handleRepairVisualLayout}
                data-testid="template-editor-repair-visual-layout"
              >
                {t("watchlists:templates.repairVisualLayout", "Repair layout")}
              </Button>
            )}
          />
        )}

        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as "visual" | "editor" | "preview" | "docs")}
          items={tabItems}
        />
      </Form>
    </Modal>
  )
}

const DEFAULT_HTML_TEMPLATE = `<!DOCTYPE html>
<html>
<head>
  <title>{{ title }} - {{ generated_at }}</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
    .item { margin-bottom: 24px; padding: 16px; background: rgb(249 249 249); border-radius: 8px; }
    .item-title { font-size: 1.1em; font-weight: 600; margin-bottom: 8px; }
    .item-title a { color: rgb(37 99 235); text-decoration: none; }
    .item-meta { font-size: 0.85em; color: #666; margin-bottom: 8px; }
    .item-summary { line-height: 1.6; }
  </style>
</head>
<body>
  <div class="header">
    <h1>{{ title }}</h1>
    <p>Generated: {{ generated_at }} | Items: {{ item_count }}</p>
  </div>

  {% if has_briefing_summary %}
  <div style="background: #f0f7ff; padding: 16px; border-radius: 8px; margin-bottom: 24px;">
    <h2>Executive Summary</h2>
    <p>{{ briefing_summary }}</p>
  </div>
  {% endif %}

  {% for item in items %}
  <div class="item">
    <div class="item-title">
      <a href="{{ item.url }}" target="_blank">{{ item.title }}</a>
    </div>
    <div class="item-meta">
      {{ item.published_at | default('Unknown date') }}
    </div>
    {% if item.llm_summary %}
    <div class="item-summary">{{ item.llm_summary }}</div>
    {% elif item.summary %}
    <div class="item-summary">{{ item.summary }}</div>
    {% endif %}
  </div>
  {% endfor %}
</body>
</html>`

const DEFAULT_MARKDOWN_TEMPLATE = `# {{ title }}

Generated: {{ generated_at }}
Items: {{ item_count }}

{% if has_briefing_summary %}
## Executive Summary

{{ briefing_summary }}

---
{% endif %}

{% for item in items %}
## {{ item.title }}
{{ item.url }}

{% if item.llm_summary %}
{{ item.llm_summary }}
{% elif item.summary %}
{{ item.summary }}
{% endif %}

{% if item.published_at %}- Published: {{ item.published_at }}{% endif %}
{% if item.tags %}- Tags: {{ item.tags | join(", ") }}{% endif %}

---
{% endfor %}
`
