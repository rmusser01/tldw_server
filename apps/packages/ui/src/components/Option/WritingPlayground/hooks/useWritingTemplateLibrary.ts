/**
 * Hook: useWritingTemplateLibrary
 *
 * Manages template and theme browsing, CRUD, import/export,
 * form state, and default restoration.
 */

import React from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type { TFunction } from "i18next"
import {
  createWritingTemplate,
  createWritingTheme,
  deleteWritingTemplate,
  deleteWritingTheme,
  getWritingDefaults,
  listWritingTemplates,
  listWritingThemes,
  updateWritingTemplate,
  updateWritingTheme,
  type WritingTemplateListResponse,
  type WritingTemplateResponse,
  type WritingThemeListResponse,
  type WritingThemeResponse
} from "@/services/writing-playground"
import {
  buildTemplateForm,
  buildTemplatePayload,
  buildThemeForm,
  buildThemePayload,
  EMPTY_TEMPLATE_FORM,
  EMPTY_THEME_FORM,
  isRecord,
  normalizeTemplatePayload,
  normalizeThemeResponse,
  sanitizeThemeCss,
  type TemplateFormState,
  type ThemeFormState
} from "./utils"
import {
  DEFAULT_TEMPLATE_CATALOG,
  DEFAULT_THEME_CATALOG,
  buildDuplicateName
} from "../writing-template-theme-utils"
import { extractImportedTemplateItems } from "../writing-template-import-utils"
import { extractImportedThemeItems } from "../writing-theme-import-utils"

export interface UseWritingTemplateLibraryDeps {
  isOnline: boolean
  hasWriting: boolean
  hasTemplates: boolean
  hasThemes: boolean
  hasServerDefaultsCatalog: boolean
  selectedTemplateName: string | null
  selectedThemeName: string | null
  handleTemplateChange: (name: string | null) => void
  handleThemeChange: (name: string | null) => void
  settingsDisabled: boolean
  t: TFunction
}

export function useWritingTemplateLibrary(deps: UseWritingTemplateLibraryDeps) {
  const {
    isOnline,
    hasWriting,
    hasTemplates,
    hasThemes,
    hasServerDefaultsCatalog,
    selectedTemplateName,
    selectedThemeName,
    handleTemplateChange,
    handleThemeChange,
    settingsDisabled,
    t
  } = deps

  const queryClient = useQueryClient()

  // --- State ---
  const [templatesModalOpen, setTemplatesModalOpen] = React.useState(false)
  const [templateForm, setTemplateForm] =
    React.useState<TemplateFormState>(EMPTY_TEMPLATE_FORM)
  const [editingTemplate, setEditingTemplate] =
    React.useState<WritingTemplateResponse | null>(null)
  const [templateImporting, setTemplateImporting] = React.useState(false)
  const [templateRestoringDefaults, setTemplateRestoringDefaults] =
    React.useState(false)
  const [themesModalOpen, setThemesModalOpen] = React.useState(false)
  const [themeForm, setThemeForm] =
    React.useState<ThemeFormState>(EMPTY_THEME_FORM)
  const [editingTheme, setEditingTheme] =
    React.useState<WritingThemeResponse | null>(null)
  const [themeImporting, setThemeImporting] = React.useState(false)
  const [themeRestoringDefaults, setThemeRestoringDefaults] =
    React.useState(false)

  // --- Refs ---
  const templateFileInputRef = React.useRef<HTMLInputElement | null>(null)
  const themeFileInputRef = React.useRef<HTMLInputElement | null>(null)

  // --- Queries ---
  const { data: writingDefaultsData } = useQuery({
    queryKey: ["writing-defaults"],
    queryFn: () => getWritingDefaults(),
    enabled: isOnline && hasWriting && hasServerDefaultsCatalog,
    staleTime: 5 * 60 * 1000
  })

  const {
    data: templatesData,
    isLoading: templatesLoading,
    error: templatesError
  } = useQuery({
    queryKey: ["writing-templates"],
    queryFn: () => listWritingTemplates({ limit: 200 }),
    enabled: isOnline && hasWriting && hasTemplates,
    staleTime: 60 * 1000
  })

  const {
    data: themesData,
    isLoading: themesLoading,
    error: themesError
  } = useQuery({
    queryKey: ["writing-themes"],
    queryFn: () => listWritingThemes({ limit: 200 }),
    enabled: isOnline && hasWriting && hasThemes,
    staleTime: 60 * 1000
  })

  const templates = templatesData?.templates ?? []
  const themes = themesData?.themes ?? []

  // --- Catalogs ---
  const templateDefaultCatalog = React.useMemo(() => {
    if (
      Array.isArray(writingDefaultsData?.templates) &&
      writingDefaultsData.templates.length > 0
    ) {
      return writingDefaultsData.templates.map((item) => ({
        name: String(item.name || "").trim() || "default",
        payload: isRecord(item.payload) ? item.payload : {},
        schema_version:
          typeof item.schema_version === "number" ? item.schema_version : 1,
        is_default: item.is_default !== false
      }))
    }
    return DEFAULT_TEMPLATE_CATALOG
  }, [writingDefaultsData?.templates])

  const themeDefaultCatalog = React.useMemo(() => {
    if (
      Array.isArray(writingDefaultsData?.themes) &&
      writingDefaultsData.themes.length > 0
    ) {
      return writingDefaultsData.themes.map((item) => ({
        name: String(item.name || "").trim() || "default",
        class_name:
          typeof item.class_name === "string" ? item.class_name : "",
        css: typeof item.css === "string" ? item.css : "",
        schema_version:
          typeof item.schema_version === "number" ? item.schema_version : 1,
        is_default: item.is_default !== false,
        order: typeof item.order === "number" ? item.order : 0
      }))
    }
    return DEFAULT_THEME_CATALOG
  }, [writingDefaultsData?.themes])

  // --- Derived ---
  const defaultTemplate =
    templates.find((template) => template.is_default) ?? templates[0] ?? null
  const selectedTemplate =
    templates.find((template) => template.name === selectedTemplateName) ?? null
  const effectiveTemplate = normalizeTemplatePayload(
    selectedTemplate ?? defaultTemplate
  )
  const templateOptions = templates.map((template) => ({
    value: template.name,
    label: template.name
  }))
  const defaultTheme = themes.find((theme) => theme.is_default) ?? themes[0] ?? null
  const selectedTheme =
    themes.find((theme) => theme.name === selectedThemeName) ?? null
  const effectiveTheme = normalizeThemeResponse(selectedTheme ?? defaultTheme)
  const themeOptions = themes.map((theme) => ({
    value: theme.name,
    label: theme.name
  }))
  const activeThemeClassName = effectiveTheme.className.trim()
  const activeThemeCss = React.useMemo(
    () => sanitizeThemeCss(effectiveTheme.css),
    [effectiveTheme.css]
  )

  // --- Mutations ---
  const createTemplateMutation = useMutation({
    mutationFn: (input: Parameters<typeof createWritingTemplate>[0]) =>
      createWritingTemplate(input),
    onSuccess: (template) => {
      queryClient.setQueryData<WritingTemplateListResponse | undefined>(
        ["writing-templates"],
        (prev) => {
          if (!prev) {
            return { templates: [template], total: 1 }
          }
          const exists = prev.templates.some(
            (item) => item.name === template.name
          )
          if (exists) {
            return {
              ...prev,
              templates: prev.templates.map((item) =>
                item.name === template.name ? template : item
              )
            }
          }
          const total = Number.isFinite(prev.total)
            ? prev.total + 1
            : prev.templates.length + 1
          return {
            ...prev,
            templates: [template, ...prev.templates],
            total
          }
        }
      )
      queryClient.invalidateQueries({ queryKey: ["writing-templates"] })
      setEditingTemplate(template)
      setTemplateForm(buildTemplateForm(template))
      if (!selectedTemplateName) {
        handleTemplateChange(template.name)
      }
    }
  })

  const updateTemplateMutation = useMutation({
    mutationFn: (payload: {
      template: WritingTemplateResponse
      input: Parameters<typeof updateWritingTemplate>[1]
    }) =>
      updateWritingTemplate(
        payload.template.name,
        payload.input,
        payload.template.version
      ),
    onSuccess: (template, payload) => {
      queryClient.invalidateQueries({ queryKey: ["writing-templates"] })
      setEditingTemplate(template)
      setTemplateForm(buildTemplateForm(template))
      if (selectedTemplateName === payload.template.name) {
        handleTemplateChange(template.name)
      }
    }
  })

  const deleteTemplateMutation = useMutation({
    mutationFn: (payload: { template: WritingTemplateResponse }) =>
      deleteWritingTemplate(payload.template.name, payload.template.version),
    onSuccess: (_data, payload) => {
      queryClient.invalidateQueries({ queryKey: ["writing-templates"] })
      if (editingTemplate?.name === payload.template.name) {
        setEditingTemplate(null)
        setTemplateForm({ ...EMPTY_TEMPLATE_FORM })
      }
      if (selectedTemplateName === payload.template.name) {
        handleTemplateChange(null)
      }
    }
  })

  const createThemeMutation = useMutation({
    mutationFn: (input: Parameters<typeof createWritingTheme>[0]) =>
      createWritingTheme(input),
    onSuccess: (theme) => {
      queryClient.setQueryData<WritingThemeListResponse | undefined>(
        ["writing-themes"],
        (prev) => {
          if (!prev) {
            return { themes: [theme], total: 1 }
          }
          const exists = prev.themes.some((item) => item.name === theme.name)
          if (exists) {
            return {
              ...prev,
              themes: prev.themes.map((item) =>
                item.name === theme.name ? theme : item
              )
            }
          }
          const total = Number.isFinite(prev.total)
            ? prev.total + 1
            : prev.themes.length + 1
          return {
            ...prev,
            themes: [theme, ...prev.themes],
            total
          }
        }
      )
      queryClient.invalidateQueries({ queryKey: ["writing-themes"] })
      setEditingTheme(theme)
      setThemeForm(buildThemeForm(theme))
      if (!selectedThemeName) {
        handleThemeChange(theme.name)
      }
    }
  })

  const updateThemeMutation = useMutation({
    mutationFn: (payload: {
      theme: WritingThemeResponse
      input: Parameters<typeof updateWritingTheme>[1]
    }) =>
      updateWritingTheme(
        payload.theme.name,
        payload.input,
        payload.theme.version
      ),
    onSuccess: (theme, payload) => {
      queryClient.invalidateQueries({ queryKey: ["writing-themes"] })
      setEditingTheme(theme)
      setThemeForm(buildThemeForm(theme))
      if (selectedThemeName === payload.theme.name) {
        handleThemeChange(theme.name)
      }
    }
  })

  const deleteThemeMutation = useMutation({
    mutationFn: (payload: { theme: WritingThemeResponse }) =>
      deleteWritingTheme(payload.theme.name, payload.theme.version),
    onSuccess: (_data, payload) => {
      queryClient.invalidateQueries({ queryKey: ["writing-themes"] })
      if (editingTheme?.name === payload.theme.name) {
        setEditingTheme(null)
        setThemeForm({ ...EMPTY_THEME_FORM })
      }
      if (selectedThemeName === payload.theme.name) {
        handleThemeChange(null)
      }
    }
  })

  // --- Handlers ---
  const updateTemplateForm = React.useCallback(
    (patch: Partial<TemplateFormState>) => {
      setTemplateForm((prev) => ({ ...prev, ...patch }))
    },
    []
  )

  const updateThemeForm = React.useCallback(
    (patch: Partial<ThemeFormState>) => {
      setThemeForm((prev) => ({ ...prev, ...patch }))
    },
    []
  )

  const handleTemplateSelect = React.useCallback(
    (template: WritingTemplateResponse) => {
      setEditingTemplate(template)
      setTemplateForm(buildTemplateForm(template))
    },
    []
  )

  const handleTemplateNew = React.useCallback(() => {
    setEditingTemplate(null)
    setTemplateForm({ ...EMPTY_TEMPLATE_FORM })
  }, [])

  const handleTemplateDuplicate = React.useCallback(() => {
    const sourceName = templateForm.name.trim() || editingTemplate?.name || ""
    if (!sourceName) return
    const duplicateName = buildDuplicateName(
      sourceName,
      templates.map((item) => item.name)
    )
    createTemplateMutation.mutate({
      name: duplicateName,
      payload: buildTemplatePayload(templateForm),
      schema_version: editingTemplate?.schema_version ?? 1,
      is_default: false
    })
  }, [createTemplateMutation, editingTemplate, templateForm, templates])

  const handleTemplateRestoreDefaults = React.useCallback(async () => {
    if (templateRestoringDefaults) return
    setTemplateRestoringDefaults(true)
    try {
      for (const item of templateDefaultCatalog) {
        const existing = templates.find((template) => template.name === item.name)
        if (existing) {
          await updateWritingTemplate(
            existing.name,
            {
              name: item.name,
              payload: item.payload,
              schema_version: item.schema_version,
              is_default: item.is_default
            },
            existing.version
          )
        } else {
          await createWritingTemplate(item)
        }
      }
      await queryClient.invalidateQueries({ queryKey: ["writing-templates"] })
      handleTemplateChange(templateDefaultCatalog[0]?.name ?? null)
    } finally {
      setTemplateRestoringDefaults(false)
    }
  }, [
    handleTemplateChange,
    queryClient,
    templateDefaultCatalog,
    templateRestoringDefaults,
    templates
  ])

  const handleOpenTemplatesModal = React.useCallback(() => {
    const baseTemplate =
      selectedTemplate ?? defaultTemplate ?? templates[0] ?? null
    if (baseTemplate) {
      setEditingTemplate(baseTemplate)
      setTemplateForm(buildTemplateForm(baseTemplate))
    } else {
      setEditingTemplate(null)
      setTemplateForm({ ...EMPTY_TEMPLATE_FORM })
    }
    setTemplatesModalOpen(true)
  }, [defaultTemplate, selectedTemplate, templates])

  const handleThemeSelect = React.useCallback((theme: WritingThemeResponse) => {
    setEditingTheme(theme)
    setThemeForm(buildThemeForm(theme))
  }, [])

  const handleThemeNew = React.useCallback(() => {
    setEditingTheme(null)
    setThemeForm({ ...EMPTY_THEME_FORM })
  }, [])

  const handleThemeDuplicate = React.useCallback(() => {
    const sourceName = themeForm.name.trim() || editingTheme?.name || ""
    if (!sourceName) return
    const duplicateName = buildDuplicateName(
      sourceName,
      themes.map((item) => item.name)
    )
    createThemeMutation.mutate({
      name: duplicateName,
      ...buildThemePayload(themeForm),
      schema_version: editingTheme?.schema_version ?? 1,
      is_default: false
    })
  }, [createThemeMutation, editingTheme, themeForm, themes])

  const handleThemeRestoreDefaults = React.useCallback(async () => {
    if (themeRestoringDefaults) return
    setThemeRestoringDefaults(true)
    try {
      for (const item of themeDefaultCatalog) {
        const existing = themes.find((theme) => theme.name === item.name)
        if (existing) {
          await updateWritingTheme(
            existing.name,
            {
              class_name: item.class_name,
              css: item.css,
              schema_version: item.schema_version,
              is_default: item.is_default,
              order: item.order
            },
            existing.version
          )
        } else {
          await createWritingTheme(item)
        }
      }
      await queryClient.invalidateQueries({ queryKey: ["writing-themes"] })
      handleThemeChange(themeDefaultCatalog[0]?.name ?? null)
    } finally {
      setThemeRestoringDefaults(false)
    }
  }, [
    handleThemeChange,
    queryClient,
    themeDefaultCatalog,
    themeRestoringDefaults,
    themes
  ])

  const handleOpenThemesModal = React.useCallback(() => {
    const baseTheme = selectedTheme ?? defaultTheme ?? themes[0] ?? null
    if (baseTheme) {
      setEditingTheme(baseTheme)
      setThemeForm(buildThemeForm(baseTheme))
    } else {
      setEditingTheme(null)
      setThemeForm({ ...EMPTY_THEME_FORM })
    }
    setThemesModalOpen(true)
  }, [defaultTheme, selectedTheme, themes])

  const handleTemplateSave = React.useCallback(() => {
    const name = templateForm.name.trim()
    if (!name) return
    const payload = buildTemplatePayload(templateForm)
    if (editingTemplate) {
      updateTemplateMutation.mutate({
        template: editingTemplate,
        input: {
          name,
          payload,
          schema_version: editingTemplate.schema_version,
          is_default: templateForm.isDefault
        }
      })
      return
    }
    createTemplateMutation.mutate({
      name,
      payload,
      schema_version: 1,
      is_default: templateForm.isDefault
    })
  }, [
    createTemplateMutation,
    editingTemplate,
    templateForm,
    updateTemplateMutation
  ])

  const handleThemeSave = React.useCallback(() => {
    const name = themeForm.name.trim()
    if (!name) return
    const payload = buildThemePayload(themeForm)
    if (editingTheme) {
      updateThemeMutation.mutate({
        theme: editingTheme,
        input: {
          name,
          ...payload,
          schema_version: editingTheme.schema_version,
          is_default: themeForm.isDefault
        }
      })
      return
    }
    createThemeMutation.mutate({
      name,
      ...payload,
      schema_version: 1,
      is_default: themeForm.isDefault
    })
  }, [
    createThemeMutation,
    editingTheme,
    themeForm,
    updateThemeMutation
  ])

  const exportTemplate = React.useCallback(
    (template: WritingTemplateResponse) => {
      const payload = {
        name: template.name,
        payload: template.payload,
        schema_version: template.schema_version,
        is_default: template.is_default
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json"
      })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `${template.name || "template"}.json`
      link.click()
      URL.revokeObjectURL(url)
    },
    []
  )

  const handleTemplateImport = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return
      setTemplateImporting(true)
      try {
        const text = await file.text()
        const parsed = JSON.parse(text)
        const items = extractImportedTemplateItems(parsed)
        if (!items.length) {
          throw new Error("No templates found in file.")
        }
        for (const item of items) {
          const { name, payload, schemaVersion, isDefault } = item
          const existing = templates.find((tmpl) => tmpl.name === name)
          if (existing) {
            await updateWritingTemplate(
              existing.name,
              {
                name,
                payload,
                schema_version: schemaVersion,
                is_default: isDefault
              },
              existing.version
            )
          } else {
            await createWritingTemplate({
              name,
              payload,
              schema_version: schemaVersion,
              is_default: isDefault
            })
          }
        }
        queryClient.invalidateQueries({ queryKey: ["writing-templates"] })
      } finally {
        setTemplateImporting(false)
        event.target.value = ""
      }
    },
    [queryClient, templates]
  )

  const exportTheme = React.useCallback((theme: WritingThemeResponse) => {
    const payload = {
      name: theme.name,
      class_name: theme.class_name,
      css: theme.css,
      schema_version: theme.schema_version,
      is_default: theme.is_default,
      order: theme.order
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json"
    })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `${theme.name || "theme"}.json`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }, [])

  const handleThemeImport = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return
      setThemeImporting(true)
      try {
        const raw = await file.text()
        const parsed = JSON.parse(raw)
        const items = extractImportedThemeItems(parsed)
        if (items.length === 0) return
        for (const item of items) {
          const name = item.name
          const existing = themes.find((theme) => theme.name === name)
          const payload = {
            class_name: item.className,
            css: item.css,
            schema_version: item.schemaVersion,
            is_default: item.isDefault,
            order: item.order
          }
          if (existing) {
            await updateWritingTheme(existing.name, payload, existing.version)
          } else {
            await createWritingTheme({ name, ...payload })
          }
        }
        queryClient.invalidateQueries({ queryKey: ["writing-themes"] })
      } finally {
        setThemeImporting(false)
        event.target.value = ""
      }
    },
    [queryClient, themes]
  )

  // --- Disabled flags ---
  const templateSelectDisabled =
    settingsDisabled || !hasTemplates || templatesLoading || Boolean(templatesError)
  const templateSaveLoading =
    createTemplateMutation.isPending || updateTemplateMutation.isPending
  const templateSaveDisabled =
    templateSaveLoading || !templateForm.name.trim()
  const templateExportDisabled = !editingTemplate
  const templateDuplicateDisabled =
    !editingTemplate || templateSaveLoading || deleteTemplateMutation.isPending
  const templateRestoreDefaultsDisabled =
    templateSaveLoading ||
    deleteTemplateMutation.isPending ||
    templateImporting ||
    templateRestoringDefaults
  const templateDeleteDisabled =
    !editingTemplate || deleteTemplateMutation.isPending
  const templateFormDisabled =
    templateSaveLoading ||
    deleteTemplateMutation.isPending ||
    templateRestoringDefaults

  const themeSelectDisabled =
    settingsDisabled || !hasThemes || themesLoading || Boolean(themesError)
  const themeSaveLoading =
    createThemeMutation.isPending || updateThemeMutation.isPending
  const themeSaveDisabled = themeSaveLoading || !themeForm.name.trim()
  const themeExportDisabled = !editingTheme
  const themeDuplicateDisabled =
    !editingTheme || themeSaveLoading || deleteThemeMutation.isPending
  const themeRestoreDefaultsDisabled =
    themeSaveLoading ||
    deleteThemeMutation.isPending ||
    themeImporting ||
    themeRestoringDefaults
  const themeDeleteDisabled =
    !editingTheme || deleteThemeMutation.isPending
  const themeFormDisabled =
    themeSaveLoading || deleteThemeMutation.isPending || themeRestoringDefaults

  return {
    // queries
    templates, templatesLoading, templatesError,
    themes, themesLoading, themesError,
    // derived
    effectiveTemplate,
    effectiveTheme,
    templateOptions,
    themeOptions,
    activeThemeClassName,
    activeThemeCss,
    defaultTemplate,
    selectedTemplate,
    defaultTheme,
    selectedTheme,
    templateDefaultCatalog,
    themeDefaultCatalog,
    // state
    templatesModalOpen, setTemplatesModalOpen,
    templateForm, setTemplateForm,
    editingTemplate, setEditingTemplate,
    templateImporting,
    templateRestoringDefaults,
    themesModalOpen, setThemesModalOpen,
    themeForm, setThemeForm,
    editingTheme, setEditingTheme,
    themeImporting,
    themeRestoringDefaults,
    // refs
    templateFileInputRef,
    themeFileInputRef,
    // mutations
    createTemplateMutation,
    updateTemplateMutation,
    deleteTemplateMutation,
    createThemeMutation,
    updateThemeMutation,
    deleteThemeMutation,
    // callbacks
    updateTemplateForm,
    updateThemeForm,
    handleTemplateSelect,
    handleTemplateNew,
    handleTemplateDuplicate,
    handleTemplateRestoreDefaults,
    handleOpenTemplatesModal,
    handleThemeSelect,
    handleThemeNew,
    handleThemeDuplicate,
    handleThemeRestoreDefaults,
    handleOpenThemesModal,
    handleTemplateSave,
    handleThemeSave,
    exportTemplate,
    handleTemplateImport,
    exportTheme,
    handleThemeImport,
    // disabled flags
    templateSelectDisabled,
    templateSaveLoading,
    templateSaveDisabled,
    templateExportDisabled,
    templateDuplicateDisabled,
    templateRestoreDefaultsDisabled,
    templateDeleteDisabled,
    templateFormDisabled,
    themeSelectDisabled,
    themeSaveLoading,
    themeSaveDisabled,
    themeExportDisabled,
    themeDuplicateDisabled,
    themeRestoreDefaultsDisabled,
    themeDeleteDisabled,
    themeFormDisabled
  }
}
