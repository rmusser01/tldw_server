import React, { useEffect, useState } from "react"
import { useMutation, type QueryClient } from "@tanstack/react-query"
import { notification } from "antd"
import {
  deletePromptById,
  savePrompt,
  updatePrompt,
  restorePrompt,
  permanentlyDeletePrompt,
  emptyTrash,
  incrementPromptUsage
} from "@/db/dexie/helpers"
import { renderStructuredPromptLegacySnapshot } from "../structured-prompt-utils"
import type { SingleTextRecipeDefinition } from "@/components/Common/PromptAssist/recipes/types"
import {
  buildRecipePromptFields,
  classifyPromptRecipe
} from "../prompt-recipe-library"
import { useSearchParams } from "react-router-dom"

export interface UsePromptEditorDeps {
  queryClient: QueryClient
  isOnline: boolean
  t: (key: string, opts?: Record<string, any>) => string
  guardPrivateMode: () => boolean
  getPromptTexts: (prompt: any) => { systemText: string | undefined; userText: string | undefined }
  getPromptKeywords: (prompt: any) => string[]
  getPromptRecordById: (id: string) => any
  confirmDanger: (options: any) => Promise<boolean>
  syncPromptAfterLocalSave: (localId: string) => Promise<{
    attempted: boolean
    success: boolean
    error?: string
  }>
  recipePersistenceAvailable: boolean
  onEmptyTrashSuccess?: () => void
}

export function usePromptEditor(deps: UsePromptEditorDeps) {
  const {
    queryClient,
    t,
    guardPrivateMode,
    getPromptTexts,
    getPromptKeywords,
    getPromptRecordById,
    confirmDanger,
    syncPromptAfterLocalSave,
    recipePersistenceAvailable,
    onEmptyTrashSuccess
  } = deps

  const [searchParams, setSearchParams] = useSearchParams()
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerMode, setDrawerMode] = useState<"create" | "edit">("create")
  const [editId, setEditId] = useState("")
  const [drawerInitialValues, setDrawerInitialValues] = useState<any>(null)
  const [fullEditorOpen, setFullEditorOpen] = useState(false)
  const [fullEditorMode, setFullEditorMode] = useState<"create" | "edit">("create")
  const [fullEditorInitialValues, setFullEditorInitialValues] = useState<any>(null)

  const normalizePromptPayload = React.useCallback((values: any) => {
    const keywords = values?.keywords ?? values?.tags ?? []
    const promptName = values?.name || values?.title
    const promptFormat = values?.promptFormat === "structured" ? "structured" : "legacy"
    const structuredPromptDefinition =
      promptFormat === "structured" ? values?.structuredPromptDefinition ?? null : null
    const recipeFields =
      promptFormat === "structured" && values?.promptSchemaVersion === 2
        ? buildRecipePromptFields(structuredPromptDefinition)
        : null
    const structuredSnapshot =
      promptFormat === "structured" && !recipeFields
        ? renderStructuredPromptLegacySnapshot(structuredPromptDefinition)
        : null
    const normalizedSystemPrompt =
      recipeFields?.system_prompt ?? structuredSnapshot?.systemPrompt ?? values?.system_prompt
    const normalizedUserPrompt =
      recipeFields?.user_prompt ?? structuredSnapshot?.userPrompt ?? values?.user_prompt
    const hasSystemPrompt = !!(normalizedSystemPrompt?.trim())
    const resolvedContent =
      recipeFields?.content ??
      values?.content ??
      structuredSnapshot?.content ??
      (hasSystemPrompt ? normalizedSystemPrompt : normalizedUserPrompt) ??
      normalizedSystemPrompt ??
      normalizedUserPrompt

    return {
      ...values,
      title: promptName,
      name: promptName,
      tags: keywords,
      keywords,
      content: resolvedContent,
      promptFormat,
      promptSchemaVersion: recipeFields?.promptSchemaVersion ?? (promptFormat === "structured" ? values?.promptSchemaVersion ?? 1 : null),
      structuredPromptDefinition:
        recipeFields?.structuredPromptDefinition ?? structuredPromptDefinition,
      system_prompt: normalizedSystemPrompt,
      user_prompt: normalizedUserPrompt,
      author: values?.author,
      details: values?.details,
      is_system: recipeFields?.is_system ?? hasSystemPrompt
    }
  }, [])

  const markPromptAsUsed = React.useCallback(
    async (promptId: string) => {
      if (!promptId) return
      try {
        await incrementPromptUsage(promptId)
        await queryClient.invalidateQueries({
          queryKey: ["fetchAllPrompts"]
        })
      } catch {
        // Usage tracking should not block prompt insertion into chat.
      }
    },
    [queryClient]
  )

  const buildPromptUpdatePayload = React.useCallback(
    (prompt: any, overrides: Partial<any> = {}) => {
      const { systemText, userText } = getPromptTexts(prompt)
      const promptName = prompt?.name || prompt?.title || "Untitled Prompt"
      const hasSystemPrompt =
        typeof systemText === "string" && systemText.trim().length > 0
      const resolvedContent =
        prompt?.content ??
        (hasSystemPrompt ? systemText : userText) ??
        systemText ??
        userText ??
        ""

      const nextKeywords =
        overrides?.keywords ??
        overrides?.tags ??
        getPromptKeywords(prompt) ??
        []

      return {
        id: prompt.id,
        title: promptName,
        name: promptName,
        content: resolvedContent,
        is_system: hasSystemPrompt,
        keywords: nextKeywords,
        tags: nextKeywords,
        favorite:
          typeof overrides?.favorite === "boolean"
            ? overrides.favorite
            : !!prompt?.favorite,
        author: prompt?.author,
        details: prompt?.details,
        system_prompt: systemText,
        user_prompt: userText,
        ...overrides
      }
    },
    [getPromptKeywords, getPromptTexts]
  )

  const { mutate: savePromptMutation, isPending: savePromptLoading } =
    useMutation({
      mutationFn: async (payload: any) => {
        const savedPrompt = await savePrompt(payload)
        const syncState = await syncPromptAfterLocalSave(savedPrompt.id)
        return {
          id: savedPrompt.id,
          syncState
        }
      },
      onSuccess: ({ syncState }) => {
        queryClient.invalidateQueries({
          queryKey: ["fetchAllPrompts"]
        })
        setDrawerOpen(false)
        setDrawerInitialValues(null)
        notification.success({
          message: t("managePrompts.notification.addSuccess"),
          description: t("managePrompts.notification.addSuccessDesc")
        })
        void syncState
      },
      onError: (error) => {
        notification.error({
          message: t("managePrompts.notification.error"),
          description:
            error?.message || t("managePrompts.notification.someError")
        })
      }
    })

  const { mutate: updatePromptDirect } = useMutation({
    mutationFn: async (payload: any) => {
      const id = await updatePrompt(payload)
      await syncPromptAfterLocalSave(id)
      return id
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.notification.error"),
        description:
          error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: updatePromptMutation, isPending: isUpdatingPrompt } =
    useMutation({
      mutationFn: async (data: any) => {
        const id = await updatePrompt({
          ...data,
          id: editId
        })
        const syncState = await syncPromptAfterLocalSave(id)
        return {
          id,
          syncState
        }
      },
      onSuccess: ({ syncState }) => {
        queryClient.invalidateQueries({
          queryKey: ["fetchAllPrompts"]
        })
        setDrawerOpen(false)
        setDrawerInitialValues(null)
        notification.success({
          message: t("managePrompts.notification.updatedSuccess"),
          description: t("managePrompts.notification.updatedSuccessDesc")
        })
        void syncState
      },
      onError: (error) => {
        notification.error({
          message: t("managePrompts.notification.error"),
          description:
            error?.message || t("managePrompts.notification.someError")
        })
      }
    })

  const { mutate: deletePrompt } = useMutation({
    mutationFn: deletePromptById,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
      queryClient.invalidateQueries({
        queryKey: ["fetchDeletedPrompts"]
      })
      notification.success({
        message: t("managePrompts.notification.deletedSuccess"),
        description: t("managePrompts.notification.movedToTrash", {
          defaultValue: "The prompt has been moved to trash. You can restore it within 30 days."
        })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.notification.error"),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: restorePromptMutation } = useMutation({
    mutationFn: restorePrompt,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
      queryClient.invalidateQueries({
        queryKey: ["fetchDeletedPrompts"]
      })
      notification.success({
        message: t("managePrompts.notification.restoredSuccess", { defaultValue: "Prompt restored" }),
        description: t("managePrompts.notification.restoredSuccessDesc", { defaultValue: "The prompt has been restored from trash." })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.notification.error"),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: permanentDeletePromptMutation } = useMutation({
    mutationFn: permanentlyDeletePrompt,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fetchDeletedPrompts"]
      })
      notification.success({
        message: t("managePrompts.notification.permanentDeleteSuccess", { defaultValue: "Prompt permanently deleted" }),
        description: t("managePrompts.notification.permanentDeleteSuccessDesc", { defaultValue: "The prompt has been permanently removed." })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.notification.error"),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: emptyTrashMutation, isPending: isEmptyingTrash } = useMutation({
    mutationFn: emptyTrash,
    onSuccess: (count) => {
      queryClient.invalidateQueries({
        queryKey: ["fetchDeletedPrompts"]
      })
      onEmptyTrashSuccess?.()
      notification.success({
        message: t("managePrompts.notification.trashEmptied", { defaultValue: "Trash emptied" }),
        description: t("managePrompts.notification.trashEmptiedDesc", {
          defaultValue: "{{count}} prompts permanently deleted.",
          count
        })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.notification.error"),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const openCreateDrawer = React.useCallback(
    (initialValues: Record<string, unknown> | null = null) => {
      if (guardPrivateMode()) return
      setDrawerMode("create")
      setEditId("")
      setDrawerInitialValues(initialValues)
      setDrawerOpen(true)
    },
    [guardPrivateMode]
  )

  const openFullEditor = React.useCallback(
    (promptRecord?: any) => {
      if (promptRecord?.id) {
        const { systemText, userText } = getPromptTexts(promptRecord)
        setFullEditorMode("edit")
        setEditId(promptRecord.id)
        setFullEditorInitialValues({
          id: promptRecord.id,
          name: promptRecord?.name || promptRecord?.title,
          author: promptRecord?.author,
          details: promptRecord?.details,
          system_prompt: systemText,
          user_prompt: userText,
          promptFormat: promptRecord?.promptFormat ?? "legacy",
          promptSchemaVersion: promptRecord?.promptSchemaVersion ?? null,
          structuredPromptDefinition:
            promptRecord?.structuredPromptDefinition != null
              ? structuredClone(promptRecord.structuredPromptDefinition)
              : null,
          keywords: promptRecord?.keywords ?? promptRecord?.tags ?? [],
          changeDescription: promptRecord?.changeDescription,
        })
        setSearchParams({ edit: promptRecord.id }, { replace: true })
      } else {
        setFullEditorMode("create")
        setFullEditorInitialValues(promptRecord || null)
        setSearchParams({ new: "1" }, { replace: true })
      }
      setFullEditorOpen(true)
    },
    [getPromptTexts, setSearchParams]
  )

  const closeFullEditor = React.useCallback(() => {
    setFullEditorOpen(false)
    setFullEditorInitialValues(null)
    const newParams = new URLSearchParams(searchParams)
    newParams.delete("edit")
    newParams.delete("new")
    setSearchParams(newParams, { replace: true })
  }, [searchParams, setSearchParams])

  const openEditDrawer = React.useCallback(
    (record: any) => {
      if (guardPrivateMode()) return
      setEditId(record.id)
      setDrawerMode("edit")
      const { systemText, userText } = getPromptTexts(record)
      setDrawerInitialValues({
        id: record?.id,
        name: record?.name || record?.title,
        author: record?.author,
        details: record?.details,
        system_prompt: systemText,
        user_prompt: userText,
        promptFormat: record?.promptFormat ?? "legacy",
        promptSchemaVersion: record?.promptSchemaVersion ?? null,
        structuredPromptDefinition: record?.structuredPromptDefinition ?? null,
        keywords: getPromptKeywords(record),
        serverId: record?.serverId,
        syncStatus: record?.syncStatus,
        sourceSystem: record?.sourceSystem,
        studioProjectId: record?.studioProjectId,
        lastSyncedAt: record?.lastSyncedAt,
        fewShotExamples: record?.fewShotExamples,
        modulesConfig: record?.modulesConfig,
        changeDescription: record?.changeDescription,
        versionNumber: record?.versionNumber
      })
      setDrawerOpen(true)
    },
    [getPromptKeywords, getPromptTexts, guardPrivateMode]
  )

  const handleDrawerSubmit = React.useCallback(
    (values: any) => {
      const payload = normalizePromptPayload(values)
      if (drawerMode === "create") {
        savePromptMutation(payload)
      } else {
        updatePromptMutation(payload)
      }
    },
    [drawerMode, normalizePromptPayload, savePromptMutation, updatePromptMutation]
  )

  const handleFullEditorSubmit = React.useCallback(
    (values: any) => {
      const payload = normalizePromptPayload(values)
      if (fullEditorMode === "create") {
        savePromptMutation(payload)
      } else {
        updatePromptMutation(payload)
      }
    },
    [fullEditorMode, normalizePromptPayload, savePromptMutation, updatePromptMutation]
  )

  const handleSaveRecipeAsNew = React.useCallback(
    async (definition: SingleTextRecipeDefinition) => {
      if (!recipePersistenceAvailable || guardPrivateMode()) {
        throw new Error("recipe_persistence_unavailable")
      }
      const fields = buildRecipePromptFields(definition)
      const sourceName =
        fullEditorInitialValues?.name ||
        fullEditorInitialValues?.title ||
        "Untitled recipe"
      const title = `${sourceName} (Copy)`
      const saved = await savePrompt({
        ...fields,
        title,
        name: title,
        keywords: fullEditorInitialValues?.keywords ?? [],
        tags: fullEditorInitialValues?.keywords ?? [],
        author: fullEditorInitialValues?.author,
        details: fullEditorInitialValues?.details
      })
      await syncPromptAfterLocalSave(saved.id)
      await queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      setFullEditorOpen(false)
      setFullEditorInitialValues(null)
    },
    [
      fullEditorInitialValues,
      guardPrivateMode,
      queryClient,
      recipePersistenceAvailable,
      syncPromptAfterLocalSave
    ]
  )

  const handleUpdateRecipe = React.useCallback(
    async (
      savedSourceId: string,
      definition: SingleTextRecipeDefinition
    ) => {
      if (!recipePersistenceAvailable || guardPrivateMode()) {
        throw new Error("recipe_persistence_unavailable")
      }
      if (savedSourceId !== editId) {
        throw new Error("recipe_source_identity_mismatch")
      }
      const sourceRecord = getPromptRecordById(savedSourceId)
      const sourceRecipe = classifyPromptRecipe(sourceRecord)
      if (sourceRecipe.kind !== "recipe") {
        throw new Error("invalid_saved_recipe")
      }
      const fields = buildRecipePromptFields(definition)
      if (
        fields.structuredPromptDefinition.assembly_config.target_role !==
        sourceRecipe.target
      ) {
        throw new Error("recipe_target_mismatch")
      }
      const id = await updatePrompt(
        buildPromptUpdatePayload(sourceRecord, {
          ...fields,
          id: savedSourceId,
          fewShotExamples: sourceRecord.fewShotExamples,
          modulesConfig: sourceRecord.modulesConfig,
          syncPayloadVersion: sourceRecord.syncPayloadVersion,
          versionNumber: sourceRecord.versionNumber,
          changeDescription: sourceRecord.changeDescription,
          parentVersionId: sourceRecord.parentVersionId,
          serverParentVersionId: sourceRecord.serverParentVersionId
        })
      )
      await syncPromptAfterLocalSave(id)
      await queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
    },
    [
      buildPromptUpdatePayload,
      editId,
      getPromptRecordById,
      guardPrivateMode,
      queryClient,
      recipePersistenceAvailable,
      syncPromptAfterLocalSave
    ]
  )

  const handleDuplicatePrompt = React.useCallback(
    (record: any) => {
      const recipe = classifyPromptRecipe(record)
      if (recipe.kind === "quarantined_recipe") return
      if (recipe.kind === "recipe" && !recipePersistenceAvailable) return
      savePromptMutation({
        title: `${record.title || record.name} (Copy)`,
        name: `${record.name || record.title} (Copy)`,
        content: record.content,
        is_system: record.is_system,
        keywords: getPromptKeywords(record),
        tags: getPromptKeywords(record),
        favorite: !!record?.favorite,
        author: record?.author,
        details: record?.details,
        system_prompt: record?.system_prompt,
        user_prompt: record?.user_prompt,
        ...(recipe.kind === "recipe"
          ? buildRecipePromptFields(recipe.definition)
          : {})
      })
    },
    [getPromptKeywords, recipePersistenceAvailable, savePromptMutation]
  )

  const handleDeletePrompt = React.useCallback(
    async (record: any) => {
      const ok = await confirmDanger({
        title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
        content: t("managePrompts.confirm.delete"),
        okText: t("common:delete", { defaultValue: "Delete" }),
        cancelText: t("common:cancel", { defaultValue: "Cancel" })
      })
      if (!ok) return
      deletePrompt(record.id)
    },
    [confirmDanger, deletePrompt, t]
  )

  const handleTogglePromptFavorite = React.useCallback(
    (promptId: string, nextFavorite: boolean) => {
      const promptRecord = getPromptRecordById(promptId)
      if (!promptRecord) return
      updatePromptDirect(
        buildPromptUpdatePayload(promptRecord, {
          favorite: nextFavorite
        })
      )
    },
    [buildPromptUpdatePayload, getPromptRecordById, updatePromptDirect]
  )

  const handleEditPromptById = React.useCallback(
    (promptId: string) => {
      const promptRecord = getPromptRecordById(promptId)
      if (!promptRecord) return
      openFullEditor(promptRecord)
    },
    [getPromptRecordById, openFullEditor]
  )

  return {
    // state
    drawerOpen,
    setDrawerOpen,
    drawerMode,
    editId,
    setEditId,
    drawerInitialValues,
    setDrawerInitialValues,
    fullEditorOpen,
    fullEditorMode,
    fullEditorInitialValues,
    // callbacks
    normalizePromptPayload,
    markPromptAsUsed,
    buildPromptUpdatePayload,
    openCreateDrawer,
    openFullEditor,
    closeFullEditor,
    openEditDrawer,
    handleDrawerSubmit,
    handleFullEditorSubmit,
    handleSaveRecipeAsNew,
    handleUpdateRecipe,
    handleDuplicatePrompt,
    handleDeletePrompt,
    handleTogglePromptFavorite,
    handleEditPromptById,
    // mutations
    savePromptMutation,
    savePromptLoading,
    updatePromptDirect,
    updatePromptMutation,
    isUpdatingPrompt,
    deletePrompt,
    restorePromptMutation,
    permanentDeletePromptMutation,
    emptyTrashMutation,
    isEmptyingTrash
  }
}
