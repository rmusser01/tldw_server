import React from "react"
import { useMutation, type QueryClient } from "@tanstack/react-query"
import { tldwClient, type ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { useNavigate } from "react-router-dom"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { focusComposer } from "@/hooks/useComposerFocus"
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { exportCharacterToJSON, exportCharacterToPNG } from "@/utils/character-export"
import {
  sanitizeServerErrorMessage,
  buildServerLogHint
} from "@/utils/server-error-message"
import {
  buildCharacterSelectionPayload,
  resolveCharacterSelectionId,
  resolveCharacterNumericId,
  normalizeWorldBookIds,
  buildCharacterPayload,
  normalizeAlternateGreetings,
  getCharacterVisibleTags,
  getCharacterFolderIdFromTags,
  readPromptPresetFromExtensions,
  readDefaultAuthorNoteFromRecord,
  readGenerationSettingsFromRecord,
  readFavoriteFromRecord,
  applyFavoriteToExtensions,
  hasAdvancedData,
  emitCharacterRecoveryTelemetry,
  characterIdentifier,
  resolveChatWorkspaceUrl,
  type PersonaProfileSummary,
  type PersonaGardenAction,
  type PersonaGardenActionContext
} from "../utils"
import { buildPersonaGardenRoute } from "@/utils/persona-garden-route"
import { createAvatarValue } from "../AvatarField"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import {
  buildCharacterChatReadiness,
  type CharacterChatReadiness
} from "@/utils/chat-model-availability"
import { buildCharacterChatPath } from "@/routes/route-paths"

export type CharacterChatIntentBlocker = {
  record: any
  characterSelection: any
  readiness: CharacterChatReadiness
  selectedModel: string | null
  availableChatModels: Array<{ model?: unknown; name?: unknown }>
}

export interface UseCharacterCrudDeps {
  t: (key: string, opts?: Record<string, any>) => string
  notification: {
    error: (args: { message: string; description?: any; duration?: number }) => void
    warning: (args: { message: string; description?: any; duration?: number }) => void
    success: (args: { message: string; description?: any; duration?: number }) => void
    info: (args: { message: string; description?: any; duration?: number }) => void
  }
  qc: QueryClient
  createForm: any
  editForm: any
  editId: string | null
  setEditId: (id: string | null) => void
  editVersion: number | null
  editCharacterNumericId: number | null
  setOpen: (open: boolean) => void
  setOpenEdit: (open: boolean) => void
  setConversationsOpen: (open: boolean) => void
  setConversationCharacter: (character: any) => void
  setPreviewCharacter: React.Dispatch<React.SetStateAction<any>>
  setShowTemplates: (show: boolean) => void
  setShowCreateSystemPromptExample: (show: boolean | ((prev: boolean) => boolean)) => void
  setShowEditSystemPromptExample: (show: boolean | ((prev: boolean) => boolean)) => void
  setShowCreateAdvanced: (show: boolean) => void
  setShowEditAdvanced: (show: boolean) => void
  setCreateFormDirty: (dirty: boolean) => void
  setEditFormDirty: (dirty: boolean) => void
  setExporting: (id: string | null) => void
  newButtonRef: React.RefObject<HTMLButtonElement | null>
  lastEditTriggerRef: React.MutableRefObject<HTMLButtonElement | null>
  editWorldBooksInitializedRef: React.MutableRefObject<boolean>
  clearCreateDraft: () => void
  clearEditDraft: () => void
  /** data from useCharacterData */
  data: any[]
  effectiveDefaultCharacterId: string | undefined
  defaultCharacterSelection: any
  setDefaultCharacterSelection: (value: any) => Promise<void> | void
  activeChatModel: string | null
  availableChatModels: Array<{ model?: unknown; name?: unknown }> | null | undefined
  setChatIntentBlocker: (value: CharacterChatIntentBlocker | null) => void
}

export function useCharacterCrud(deps: UseCharacterCrudDeps) {
  const {
    t,
    notification,
    qc,
    createForm,
    editForm,
    editId,
    setEditId,
    editVersion,
    editCharacterNumericId,
    setOpen,
    setOpenEdit,
    setConversationsOpen,
    setConversationCharacter,
    setPreviewCharacter,
    setShowTemplates,
    setShowCreateSystemPromptExample,
    setShowEditSystemPromptExample,
    setShowCreateAdvanced,
    setShowEditAdvanced,
    setCreateFormDirty,
    setEditFormDirty,
    setExporting,
    newButtonRef,
    lastEditTriggerRef,
    editWorldBooksInitializedRef,
    clearCreateDraft,
    clearEditDraft,
    data,
    effectiveDefaultCharacterId,
    setDefaultCharacterSelection,
    activeChatModel,
    availableChatModels,
    setChatIntentBlocker
  } = deps

  const navigate = useNavigate()
  const [, setSelectedCharacter] = useSelectedCharacter<any>(null)

  // Conversation state
  const [characterChats, setCharacterChats] = React.useState<ServerChatSummary[]>([])
  const [chatsError, setChatsError] = React.useState<string | null>(null)
  const [loadingChats, setLoadingChats] = React.useState(false)
  const [resumingChatId, setResumingChatId] = React.useState<string | null>(null)

  // Store message option actions for conversations drawer resume-chat flow
  const {
    setHistory,
    setMessages,
    setHistoryId,
    setServerChatId,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef
  } = useStoreMessageOption(
    (state) => ({
      setHistory: state.setHistory,
      setMessages: state.setMessages,
      setHistoryId: state.setHistoryId,
      setServerChatId: state.setServerChatId,
      setServerChatState: state.setServerChatState,
      setServerChatTopic: state.setServerChatTopic,
      setServerChatClusterId: state.setServerChatClusterId,
      setServerChatSource: state.setServerChatSource,
      setServerChatExternalRef: state.setServerChatExternalRef
    }),
    shallow
  )

  // --- Create mutation ---
  const { mutate: createCharacter, isPending: creating } = useMutation({
    mutationFn: async (values: any) => {
      const selectedWorldBookIds = normalizeWorldBookIds(
        values?.world_book_ids
      )
      const createdCharacter = await tldwClient.createCharacter(
        buildCharacterPayload(values)
      )
      if (selectedWorldBookIds.length === 0) {
        return createdCharacter
      }

      const characterId = resolveCharacterNumericId(createdCharacter)
      if (characterId == null) {
        throw new Error(
          t("settings:manageCharacters.form.worldBooks.unsupportedCharacter", {
            defaultValue:
              "World books can only be attached to saved server characters."
          })
        )
      }

      await syncWorldBookSelection(characterId, selectedWorldBookIds)
      return createdCharacter
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      qc.invalidateQueries({ queryKey: ["tldw:characterPreviewWorldBooks"] })
      setOpen(false)
      createForm.resetFields()
      clearCreateDraft()
      setShowTemplates(false)
      setShowCreateSystemPromptExample(false)
      notification.success({
        message: t("settings:manageCharacters.notification.addSuccess", {
          defaultValue: "Character created"
        })
      })
      setTimeout(() => {
        newButtonRef.current?.focus()
      }, 0)
    },
    onError: (e: any) =>
      notification.error({
        message: t("settings:manageCharacters.notification.error", {
          defaultValue: "Error"
        }),
        description:
          e?.message ||
          t("settings:manageCharacters.notification.someError", {
            defaultValue: "Something went wrong. Please try again later"
          })
      })
  })

  // --- Update mutation ---
  const { mutate: updateCharacter, isPending: updating } = useMutation({
    mutationFn: async (values: any) => {
      if (!editId) {
        throw new Error("No character selected for editing")
      }
      const updatedCharacter = await tldwClient.updateCharacter(
        editId,
        buildCharacterPayload(values),
        editVersion ?? undefined
      )
      if (typeof values?.world_book_ids !== "undefined") {
        const selectedWorldBookIds = normalizeWorldBookIds(
          values?.world_book_ids
        )
        const editCharacterId = Number(editId)
        if (Number.isFinite(editCharacterId) && editCharacterId > 0) {
          await syncWorldBookSelection(
            Math.trunc(editCharacterId),
            selectedWorldBookIds
          )
        } else if (selectedWorldBookIds.length > 0) {
          throw new Error(
            t("settings:manageCharacters.form.worldBooks.unsupportedCharacter", {
              defaultValue:
                "World books can only be attached to saved server characters."
            })
          )
        }
      }
      return updatedCharacter
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      qc.invalidateQueries({ queryKey: ["tldw:characterPreviewWorldBooks"] })
      qc.invalidateQueries({
        queryKey: ["tldw:characterEditWorldBooks", editCharacterNumericId]
      })
      setOpenEdit(false)
      editForm.resetFields()
      setEditId(null)
      editWorldBooksInitializedRef.current = false
      clearEditDraft()
      setShowEditSystemPromptExample(false)
      notification.success({
        message: t("settings:manageCharacters.notification.updatedSuccess", {
          defaultValue: "Character updated"
        })
      })
      setTimeout(() => {
        lastEditTriggerRef.current?.focus()
      }, 0)
    },
    onError: (e: any) =>
      notification.error({
        message: t("settings:manageCharacters.notification.error", {
          defaultValue: "Error"
        }),
        description:
          e?.message ||
          t("settings:manageCharacters.notification.someError", {
            defaultValue: "Something went wrong. Please try again later"
          })
      })
  })

  // --- Delete state and mutation ---
  const [pendingDelete, setPendingDelete] = React.useState<{
    character: any
    timeoutId: ReturnType<typeof setTimeout>
  } | null>(null)
  const undoDeleteRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const bulkUndoDeleteRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const { mutate: deleteCharacter, isPending: deleting } = useMutation({
    mutationFn: async ({ id, expectedVersion }: { id: string; expectedVersion?: number }) =>
      tldwClient.deleteCharacter(id, expectedVersion),
    onSuccess: (_data, _variables, context: any) => {
      // Don't invalidate immediately - wait for undo timeout
    },
    onError: (e: any, _variables, context: any) => {
      if (context?.character) {
        qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      }
      notification.error({
        message: t("settings:manageCharacters.notification.error", {
          defaultValue: "Error"
        }),
        description:
          e?.message ||
          t("settings:manageCharacters.notification.someError", {
            defaultValue: "Something went wrong. Please try again later"
          })
      })
    }
  })

  const { mutate: restoreCharacter } = useMutation({
    mutationFn: async ({ id, version }: { id: string; version: number }) =>
      tldwClient.restoreCharacter(id, version),
    onSuccess: (_data, variables) => {
      emitCharacterRecoveryTelemetry("restore", {
        character_id: variables?.id ?? null
      })
      qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      notification.success({
        message: t("settings:manageCharacters.notification.restored", {
          defaultValue: "Character restored"
        })
      })
    },
    onError: (e: any) => {
      emitCharacterRecoveryTelemetry("restore_failed", {
        reason: e?.message ?? "unknown_error"
      })
      qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      const detail = sanitizeServerErrorMessage(
        e,
        t("settings:manageCharacters.notification.restoreErrorDetail", {
          defaultValue: "Try refreshing the list and restoring again."
        })
      )
      notification.error({
        message: t("settings:manageCharacters.notification.restoreError", {
          defaultValue: "Failed to restore character"
        }),
        description: `${detail} ${buildServerLogHint(
          e,
          t("settings:manageCharacters.notification.restoreLogHint", {
            defaultValue: "If this keeps happening, check server logs."
          })
        )}`
      })
    }
  })

  // Cleanup pending delete timeout on unmount
  React.useEffect(() => {
    return () => {
      if (undoDeleteRef.current) {
        clearTimeout(undoDeleteRef.current)
      }
      if (bulkUndoDeleteRef.current) {
        clearTimeout(bulkUndoDeleteRef.current)
      }
    }
  }, [])

  // --- World book sync helper ---
  const getWorldBookSyncError = React.useCallback(
    (error: unknown): Error => {
      const statusRaw = (error as { status?: unknown } | null)?.status
      const statusCode =
        typeof statusRaw === "number"
          ? statusRaw
          : typeof statusRaw === "string"
            ? Number(statusRaw)
            : Number.NaN

      if (statusCode === 401 || statusCode === 403) {
        return new Error(
          t("settings:manageCharacters.worldBooks.permissionDenied", {
            defaultValue: "You do not have permission to modify world-book attachments."
          })
        )
      }

      if (statusCode === 404) {
        return new Error(
          t("settings:manageCharacters.worldBooks.referenceMissing", {
            defaultValue: "A selected world book could not be found. Refresh and try again."
          })
        )
      }

      if (error instanceof Error) return error
      return new Error(
        t("settings:manageCharacters.worldBooks.syncFailed", {
          defaultValue: "Failed to update world-book attachments."
        })
      )
    },
    [t]
  )

  const syncWorldBookSelection = React.useCallback(
    async (characterId: number, nextWorldBookIds: number[]) => {
      await tldwClient.initialize()
      let currentLinks: any[] = []
      try {
        const response = await tldwClient.listCharacterWorldBooks(characterId)
        currentLinks = Array.isArray(response) ? response : []
      } catch (error) {
        throw getWorldBookSyncError(error)
      }
      const currentIds = normalizeWorldBookIds(
        currentLinks.map((book: any) => book?.world_book_id ?? book?.id)
      )
      const desiredIds = normalizeWorldBookIds(nextWorldBookIds)
      const currentSet = new Set(currentIds)
      const desiredSet = new Set(desiredIds)
      const toAttach = desiredIds.filter((id) => !currentSet.has(id))
      const toDetach = currentIds.filter((id) => !desiredSet.has(id))

      if (toAttach.length > 0) {
        for (const worldBookId of toAttach) {
          try {
            await tldwClient.attachWorldBookToCharacter(characterId, worldBookId)
          } catch (error) {
            throw getWorldBookSyncError(error)
          }
        }
      }

      if (toDetach.length > 0) {
        for (const worldBookId of toDetach) {
          try {
            await tldwClient.detachWorldBookFromCharacter(characterId, worldBookId)
          } catch (error) {
            throw getWorldBookSyncError(error)
          }
        }
      }
    },
    [getWorldBookSyncError]
  )

  // --- Load conversations effect ---
  const conversationsLoadErrorMessageRef = React.useRef(
    t("settings:manageCharacters.conversations.error", {
      defaultValue: "Unable to load conversations for this character."
    })
  )

  React.useEffect(() => {
    conversationsLoadErrorMessageRef.current = t(
      "settings:manageCharacters.conversations.error",
      {
        defaultValue: "Unable to load conversations for this character."
      }
    )
  }, [t])

  // --- Export handler ---
  const handleExport = React.useCallback(async (record: any, format: 'json' | 'png' = 'json') => {
    const id = record.id || record.slug || record.name
    const name = record.name || record.title || record.slug || "character"
    try {
      setExporting(id)
      const data = await tldwClient.exportCharacter(id, { format: 'v3' })

      if (format === 'png') {
        await exportCharacterToPNG(data, {
          avatarUrl: record.avatar_url,
          avatarBase64: record.image_base64,
          filename: `${name.replace(/[^a-z0-9]/gi, '_')}_character.png`
        })
      } else {
        exportCharacterToJSON(data, `${name.replace(/[^a-z0-9]/gi, '_')}_character.json`)
      }

      notification.success({
        message: t("settings:manageCharacters.notification.exported", {
          defaultValue: "Character exported"
        })
      })
    } catch (e: any) {
      notification.error({
        message: t("settings:manageCharacters.notification.exportError", {
          defaultValue: "Failed to export character"
        }),
        description: e?.message
      })
    } finally {
      setExporting(null)
    }
  }, [notification, setExporting, t])

  // --- Chat handler ---
  const handleChat = React.useCallback(async (record: any) => {
    const characterSelection = buildCharacterSelectionPayload(record)
    await setSelectedCharacter(characterSelection)

    const readiness = buildCharacterChatReadiness({
      selectedCharacter: characterSelection,
      selectedModel: activeChatModel,
      availableModels: availableChatModels
    })
    if (!readiness.canStart && readiness.missingRequirement === "chat-model") {
      setChatIntentBlocker({
        record,
        characterSelection,
        readiness,
        selectedModel: activeChatModel,
        availableChatModels: availableChatModels ?? []
      })
      return
    }

    setChatIntentBlocker(null)
    navigate(buildCharacterChatPath({ characterId: characterSelection.id }))
    setTimeout(() => {
      focusComposer()
    }, 0)
  }, [
    activeChatModel,
    availableChatModels,
    navigate,
    setChatIntentBlocker,
    setSelectedCharacter
  ])

  // --- Chat in new tab ---
  const handleChatInNewTab = React.useCallback(
    async (record: any) => {
      const characterSelection = buildCharacterSelectionPayload(record)
      await setSelectedCharacter(characterSelection)
      const opened = window.open(
        resolveChatWorkspaceUrl({ characterId: characterSelection.id }),
        "_blank",
        "noopener,noreferrer"
      )
      if (!opened) {
        notification.warning({
          message: t("settings:manageCharacters.notification.chatTabBlocked", {
            defaultValue: "Popup blocked"
          }),
          description: t(
            "settings:manageCharacters.notification.chatTabBlockedDesc",
            {
              defaultValue:
                "Allow popups for this site or use Chat to open in the current tab."
            }
          )
        })
      }
    },
    [notification, setSelectedCharacter, t]
  )

  // --- Edit handler ---
  const handleEdit = React.useCallback((record: any, triggerRef?: HTMLButtonElement | null) => {
    if (triggerRef) {
      lastEditTriggerRef.current = triggerRef
    }
    setEditId(record.id || record.slug || record.name)
    const ex = record.extensions
    const extensionsValue =
      ex && typeof ex === "object" && !Array.isArray(ex)
        ? JSON.stringify(ex, null, 2)
        : typeof ex === "string"
          ? ex
          : ""
    const promptPreset = readPromptPresetFromExtensions(record.extensions)
    const defaultAuthorNote = readDefaultAuthorNoteFromRecord(record)
    const generationSettings = readGenerationSettingsFromRecord(record)
    const visibleTags = getCharacterVisibleTags(record?.tags)
    const assignedFolderId = getCharacterFolderIdFromTags(record?.tags)
    editWorldBooksInitializedRef.current = false
    editForm.setFieldsValue({
      name: record.name,
      description: record.description,
      avatar: createAvatarValue(record.avatar_url, record.image_base64),
      tags: visibleTags,
      folder_id: assignedFolderId,
      greeting:
        record.greeting ||
        record.first_message ||
        record.greet,
      system_prompt: record.system_prompt,
      personality: record.personality,
      scenario: record.scenario,
      post_history_instructions:
        record.post_history_instructions,
      message_example: record.message_example,
      creator_notes: record.creator_notes,
      alternate_greetings: normalizeAlternateGreetings(
        record.alternate_greetings
      ),
      creator: record.creator,
      character_version: record.character_version,
      prompt_preset: promptPreset,
      default_author_note: defaultAuthorNote,
      generation_temperature: generationSettings.temperature,
      generation_top_p: generationSettings.top_p,
      generation_repetition_penalty: generationSettings.repetition_penalty,
      generation_stop_strings: generationSettings.stop?.join("\n") || "",
      extensions: extensionsValue,
      world_book_ids: []
    })
    setShowEditAdvanced(hasAdvancedData(record, extensionsValue))
    setOpenEdit(true)
  }, [editForm, editWorldBooksInitializedRef, lastEditTriggerRef, setEditId, setOpenEdit, setShowEditAdvanced])

  // --- Duplicate handler ---
  const handleDuplicate = React.useCallback((record: any) => {
    const ex = record.extensions
    const extensionsValue =
      ex && typeof ex === "object" && !Array.isArray(ex)
        ? JSON.stringify(ex, null, 2)
        : typeof ex === "string"
          ? ex
          : ""
    const promptPreset = readPromptPresetFromExtensions(record.extensions)
    const defaultAuthorNote = readDefaultAuthorNoteFromRecord(record)
    const generationSettings = readGenerationSettingsFromRecord(record)
    const visibleTags = getCharacterVisibleTags(record?.tags)
    const assignedFolderId = getCharacterFolderIdFromTags(record?.tags)
    createForm.setFieldsValue({
      name: `${record.name || ""} (copy)`,
      description: record.description,
      avatar: createAvatarValue(record.avatar_url, record.image_base64),
      tags: visibleTags,
      folder_id: assignedFolderId,
      greeting:
        record.greeting ||
        record.first_message ||
        record.greet,
      system_prompt: record.system_prompt,
      personality: record.personality,
      scenario: record.scenario,
      post_history_instructions:
        record.post_history_instructions,
      message_example: record.message_example,
      creator_notes: record.creator_notes,
      alternate_greetings: normalizeAlternateGreetings(
        record.alternate_greetings
      ),
      creator: record.creator,
      character_version: record.character_version,
      prompt_preset: promptPreset,
      default_author_note: defaultAuthorNote,
      generation_temperature: generationSettings.temperature,
      generation_top_p: generationSettings.top_p,
      generation_repetition_penalty: generationSettings.repetition_penalty,
      generation_stop_strings: generationSettings.stop?.join("\n") || "",
      extensions: extensionsValue,
      world_book_ids: []
    })
    setShowCreateAdvanced(hasAdvancedData(record, extensionsValue))
    setOpen(true)

    const name = record.name || record.title || record.slug || ""
    notification.info({
      message: t("settings:manageCharacters.notification.duplicated", {
        defaultValue: "Duplicated '{{name}}'. Editing copy.",
        name
      })
    })
  }, [createForm, notification, setOpen, setShowCreateAdvanced, t])

  // --- Delete handler ---
  const handleDelete = React.useCallback(async (record: any) => {
    const name = record?.name || record?.title || record?.slug || ""
    const characterId = String(record.id || record.slug || record.name)
    const characterVersion = record.version

    if (undoDeleteRef.current) {
      clearTimeout(undoDeleteRef.current)
      undoDeleteRef.current = null
    }
    if (pendingDelete?.timeoutId) {
      clearTimeout(pendingDelete.timeoutId)
    }

    deleteCharacter({ id: characterId, expectedVersion: characterVersion }, {
      onSuccess: () => {
        emitCharacterRecoveryTelemetry("delete", {
          character_id: characterId
        })
        const timeoutId = setTimeout(() => {
          setPendingDelete(null)
          undoDeleteRef.current = null
          qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
        }, 10000)

        undoDeleteRef.current = timeoutId
        setPendingDelete({ character: record, timeoutId })

        notification.info({
          message: t("settings:manageCharacters.notification.deletedWithUndo", {
            defaultValue: "Character '{{name}}' deleted",
            name
          }),
          description: (
            <button
              type="button"
              className="mt-1 text-sm font-medium text-primary hover:underline"
              onClick={() => {
                if (undoDeleteRef.current) {
                  clearTimeout(undoDeleteRef.current)
                  undoDeleteRef.current = null
                }
                setPendingDelete(null)
                emitCharacterRecoveryTelemetry("undo", {
                  character_id: characterId
                })
                restoreCharacter({ id: characterId, version: (characterVersion ?? 0) + 1 })
              }}>
              {t("common:undo", { defaultValue: "Undo" })}
            </button>
          ),
          duration: 10
        })
      }
    })
  }, [deleteCharacter, notification, t, qc, pendingDelete, restoreCharacter])

  // --- View conversations ---
  const handleViewConversations = React.useCallback((record: any) => {
    setConversationCharacter(record)
    setCharacterChats([])
    setChatsError(null)
    setConversationsOpen(true)
  }, [setConversationCharacter, setConversationsOpen])

  // --- Restore from trash ---
  const handleRestoreFromTrash = React.useCallback(
    (record: any) => {
      const characterId = String(record?.id || record?.slug || record?.name || "")
      const characterVersion = Number(record?.version)
      if (!characterId) return
      if (!Number.isFinite(characterVersion)) {
        notification.error({
          message: t("settings:manageCharacters.notification.restoreError", {
            defaultValue: "Failed to restore character"
          }),
          description: t("settings:manageCharacters.notification.restoreVersionMissing", {
            defaultValue: "Missing character version. Refresh and try again."
          })
        })
        return
      }
      restoreCharacter({ id: characterId, version: characterVersion })
    },
    [notification, restoreCharacter, t]
  )

  // --- Default character ---
  const isDefaultCharacterRecord = React.useCallback(
    (record: any) => {
      if (!effectiveDefaultCharacterId) return false
      const recordId = resolveCharacterSelectionId({
        id: record?.id || record?.slug || record?.name
      } as any)
      return recordId === effectiveDefaultCharacterId
    },
    [effectiveDefaultCharacterId]
  )

  const handleSetDefaultCharacter = React.useCallback(
    async (record: any) => {
      try {
        const nextSelection = buildCharacterSelectionPayload(record)
        const nextDefaultId = resolveCharacterSelectionId(nextSelection)
        if (!nextDefaultId) {
          throw new Error(
            t("settings:manageCharacters.notification.defaultSetError", {
              defaultValue: "Couldn't set default character"
            })
          )
        }

        let serverWriteError: any = null
        try {
          await tldwClient.setDefaultCharacterPreference(nextDefaultId)
        } catch (serverError) {
          serverWriteError = serverError
        }

        await setDefaultCharacterSelection(nextSelection)

        if (serverWriteError) {
          notification.warning({
            message: t(
              "settings:manageCharacters.notification.defaultSetLocalOnly",
              {
                defaultValue: "Default character saved locally only"
              }
            ),
            description:
              serverWriteError?.message ||
              t(
                "settings:manageCharacters.notification.defaultSetLocalOnlyDesc",
                {
                  defaultValue:
                    "Server preference update failed. This device will still use your selected default."
                }
              )
          })
          return
        }

        notification.success({
          message: t("settings:manageCharacters.notification.defaultSet", {
            defaultValue: "Default character set"
          }),
          description: t("settings:manageCharacters.notification.defaultSetDesc", {
            defaultValue: "{{name}} will be preselected for new chats.",
            name: record?.name || record?.title || record?.slug || ""
          })
        })
      } catch (error: any) {
        notification.error({
          message: t("settings:manageCharacters.notification.defaultSetError", {
            defaultValue: "Couldn't set default character"
          }),
          description:
            error?.message ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
      }
    },
    [notification, setDefaultCharacterSelection, t]
  )

  const handleClearDefaultCharacter = React.useCallback(async () => {
    try {
      let serverWriteError: any = null
      try {
        await tldwClient.setDefaultCharacterPreference(null)
      } catch (serverError) {
        serverWriteError = serverError
      }

      await setDefaultCharacterSelection(null)

      if (serverWriteError) {
        notification.warning({
          message: t(
            "settings:manageCharacters.notification.defaultClearedLocalOnly",
            {
              defaultValue: "Default cleared locally only"
            }
          ),
          description:
            serverWriteError?.message ||
            t(
              "settings:manageCharacters.notification.defaultClearedLocalOnlyDesc",
              {
                defaultValue:
                  "Server preference update failed. This device will still clear the default."
              }
            )
        })
        return
      }

      notification.info({
        message: t("settings:manageCharacters.notification.defaultCleared", {
          defaultValue: "Default character cleared"
        })
      })
    } catch (error: any) {
      notification.error({
        message: t("settings:manageCharacters.notification.defaultSetError", {
          defaultValue: "Couldn't set default character"
        }),
        description:
          error?.message ||
          t("settings:manageCharacters.notification.someError", {
            defaultValue: "Something went wrong. Please try again later"
            })
      })
    }
  }, [notification, setDefaultCharacterSelection, t])

  // --- Favorites ---
  const isCharacterFavoriteRecord = React.useCallback(
    (record: any) => readFavoriteFromRecord(record),
    []
  )

  const handleToggleFavorite = React.useCallback(
    async (record: any) => {
      const id = String(record?.id || record?.slug || record?.name || "")
      if (!id) return

      const nextFavorite = !isCharacterFavoriteRecord(record)
      const nextExtensions = applyFavoriteToExtensions(
        record?.extensions,
        nextFavorite
      )
      if (nextExtensions === null) {
        notification.warning({
          message: t("settings:manageCharacters.notification.favoriteInvalidExtensions", {
            defaultValue: "Couldn't update favorite"
          }),
          description: t(
            "settings:manageCharacters.notification.favoriteInvalidExtensionsDesc",
            {
              defaultValue:
                "This character has invalid extensions JSON. Fix the extensions field before toggling favorite."
            }
          )
        })
        return
      }

      let previousData: unknown = undefined
      let previousPreview: any = undefined
      try {
        previousData = qc.getQueryData?.(["tldw:listCharacters"])
        qc.setQueryData?.(["tldw:listCharacters"], (old: any) => {
          if (!Array.isArray(old)) return old
          return old.map((c: any) => {
            const cId = String(c?.id || c?.slug || c?.name || "")
            if (cId !== id) return c
            return { ...c, extensions: nextExtensions ?? {} }
          })
        })
      } catch {
        // Optimistic update not available
      }
      setPreviewCharacter((current: any) => {
        previousPreview = current
        if (!current) return current
        const currentId = String(
          current?.id || current?.slug || current?.name || ""
        )
        if (currentId !== id) return current
        return {
          ...current,
          extensions: nextExtensions ?? {}
        }
      })

      try {
        await tldwClient.updateCharacter(
          id,
          { extensions: nextExtensions ?? {} },
          record?.version
        )
        qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      } catch (error: any) {
        if (previousData !== undefined) {
          try { qc.setQueryData?.(["tldw:listCharacters"], previousData) } catch { /* noop */ }
        }
        if (previousPreview !== undefined) {
          setPreviewCharacter(previousPreview)
        }
        notification.error({
          message: t("settings:manageCharacters.notification.error", {
            defaultValue: "Error"
          }),
          description:
            error?.message ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
      }
    },
    [isCharacterFavoriteRecord, notification, qc, setPreviewCharacter, t]
  )

  // --- Persona Garden ---
  const pendingPersonaCreateIdsRef = React.useRef<Set<number>>(new Set())
  const [pendingPersonaCreateIds, setPendingPersonaCreateIds] = React.useState<number[]>([])
  const pendingPersonaCreateIdSet = React.useMemo(
    () => new Set(pendingPersonaCreateIds),
    [pendingPersonaCreateIds]
  )

  const getCharacterNumericId = React.useCallback((record: any): number | null => {
    const parsed = Number(record?.id)
    return Number.isFinite(parsed) && parsed > 0 ? Math.trunc(parsed) : null
  }, [])

  const getCharacterDisplayName = React.useCallback(
    (record: any): string =>
      String(record?.name || record?.title || record?.slug || "Character").trim() ||
      "Character",
    []
  )

  const listPersonaProfiles = React.useCallback(async (): Promise<PersonaProfileSummary[]> => {
    const response = await tldwClient.fetchWithAuth(
      "/api/v1/persona/profiles?limit=200" as any,
      { method: "GET" }
    )
    if (!response.ok) {
      throw new Error(response.error || "Failed to load persona profiles")
    }
    const payload = await response.json()
    return Array.isArray(payload) ? (payload as PersonaProfileSummary[]) : []
  }, [])

  const findPersonaForCharacter = React.useCallback(
    (
      profiles: PersonaProfileSummary[],
      characterId: number
    ): PersonaProfileSummary | null =>
      profiles.find((profile) => {
        const originCharacterId = Number(profile?.origin_character_id)
        const linkedCharacterId = Number(profile?.character_card_id)
        return originCharacterId === characterId || linkedCharacterId === characterId
      }) || null,
    []
  )

  const buildSuggestedPersonaName = React.useCallback(
    (record: any, profiles: PersonaProfileSummary[]): string => {
      const baseName = `${getCharacterDisplayName(record)} Persona`
      const existingNames = new Set(
        profiles
          .map((profile) => String(profile?.name || "").trim().toLowerCase())
          .filter(Boolean)
      )
      if (!existingNames.has(baseName.toLowerCase())) {
        return baseName
      }
      let suffix = 2
      while (existingNames.has(`${baseName} ${suffix}`.toLowerCase())) {
        suffix += 1
      }
      return `${baseName} ${suffix}`
    },
    [getCharacterDisplayName]
  )

  const setPersonaCreatePending = React.useCallback(
    (characterId: number, pending: boolean) => {
      const nextPendingIds = new Set(pendingPersonaCreateIdsRef.current)
      if (pending) {
        nextPendingIds.add(characterId)
      } else {
        nextPendingIds.delete(characterId)
      }
      pendingPersonaCreateIdsRef.current = nextPendingIds
      setPendingPersonaCreateIds(Array.from(nextPendingIds))
    },
    []
  )

  const isPersonaCreatePending = React.useCallback(
    (record: any): boolean => {
      const characterId = getCharacterNumericId(record)
      return characterId != null && pendingPersonaCreateIdSet.has(characterId)
    },
    [getCharacterNumericId, pendingPersonaCreateIdSet]
  )

  const getCreatePersonaActionLabel = React.useCallback(
    (record: any): string =>
      isPersonaCreatePending(record)
        ? t("settings:manageCharacters.actions.creatingPersonaFromCharacter", {
            defaultValue: "Creating Persona..."
          })
        : t("settings:manageCharacters.actions.createPersonaFromCharacter", {
            defaultValue: "Create Persona from Character"
          }),
    [isPersonaCreatePending, t]
  )

  const loadPersonaGardenActionContext = React.useCallback(
    async (
      record: any,
      action: PersonaGardenAction
    ): Promise<PersonaGardenActionContext | null> => {
      const characterId = getCharacterNumericId(record)
      const characterName = getCharacterDisplayName(record)
      if (characterId == null) {
        notification.warning({
          message: t("settings:manageCharacters.personaGarden.invalidCharacter", {
            defaultValue: "Character missing a numeric ID"
          }),
          description: t(
            "settings:manageCharacters.personaGarden.invalidCharacterDesc",
            {
              defaultValue:
                action === "create"
                  ? "Save {{name}} to the server before creating a linked persona."
                  : "Save {{name}} to the server before opening a linked persona.",
              name: characterName
            }
          )
        })
        return null
      }
      try {
        const profiles = await listPersonaProfiles()
        return {
          characterId,
          characterName,
          profiles,
          existingPersona: findPersonaForCharacter(profiles, characterId)
        }
      } catch (error: any) {
        notification.error({
          message: t(
            action === "create"
              ? "settings:manageCharacters.personaGarden.createError"
              : "settings:manageCharacters.personaGarden.openError",
            {
              defaultValue:
                action === "create"
                  ? "Failed to create persona"
                  : "Failed to open Persona Garden"
            }
          ),
          description:
            sanitizeServerErrorMessage(
              error?.message,
              t("settings:manageCharacters.notification.someError", {
                defaultValue: "Something went wrong. Please try again later"
              })
            ) ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
        return null
      }
    },
    [
      findPersonaForCharacter,
      getCharacterDisplayName,
      getCharacterNumericId,
      listPersonaProfiles,
      notification,
      t
    ]
  )

  const openPersonaGardenForCharacter = React.useCallback(
    async (record: any) => {
      const context = await loadPersonaGardenActionContext(record, "open")
      if (!context) {
        return
      }
      const { characterName, existingPersona } = context
      if (existingPersona?.id != null) {
        navigate(
          buildPersonaGardenRoute({
            personaId: String(existingPersona.id),
            tab: "profiles"
          })
        )
        return
      }
      notification.info({
        message: t("settings:manageCharacters.personaGarden.noneFound", {
          defaultValue: "No linked persona yet"
        }),
        description: t(
          "settings:manageCharacters.personaGarden.noneFoundDesc",
          {
            defaultValue:
              "Open Persona Garden to create a persona derived from {{name}}.",
            name: characterName
          }
        )
      })
      navigate(buildPersonaGardenRoute({ tab: "profiles" }))
    },
    [
      loadPersonaGardenActionContext,
      navigate,
      notification,
      t
    ]
  )

  const createPersonaFromCharacter = React.useCallback(
    async (record: any) => {
      const characterId = getCharacterNumericId(record)
      if (characterId == null || pendingPersonaCreateIdsRef.current.has(characterId)) {
        return
      }
      setPersonaCreatePending(characterId, true)
      try {
        const context = await loadPersonaGardenActionContext(record, "create")
        if (!context) {
          return
        }
        const { characterId: resolvedCharacterId, characterName, profiles, existingPersona } =
          context
        if (existingPersona?.id != null) {
          notification.info({
            message: t("settings:manageCharacters.personaGarden.existingPersona", {
              defaultValue: "Persona already exists"
            }),
            description: t(
              "settings:manageCharacters.personaGarden.existingPersonaDesc",
              {
                defaultValue:
                  "Opened the existing persona derived from {{name}}.",
                name: characterName
              }
            )
          })
          navigate(
            buildPersonaGardenRoute({
              personaId: String(existingPersona.id),
              tab: "profiles"
            })
          )
          return
        }

        const personaName = buildSuggestedPersonaName(record, profiles)
        const response = await tldwClient.fetchWithAuth("/api/v1/persona/profiles" as any, {
          method: "POST",
          body: {
            name: personaName,
            character_card_id: resolvedCharacterId,
            mode: "persistent_scoped"
          }
        })
        if (!response.ok) {
          throw new Error(response.error || "Failed to create persona from character")
        }
        const payload = await response.json()
        const personaId = String(payload?.id || "").trim()
        if (!personaId) {
          throw new Error("Persona creation response missing id")
        }
        notification.success({
          message: t("settings:manageCharacters.personaGarden.created", {
            defaultValue: "Persona created"
          }),
          description: t("settings:manageCharacters.personaGarden.createdDesc", {
            defaultValue:
              "Created {{personaName}} from {{characterName}} and opened it in Persona Garden.",
            personaName,
            characterName
          })
        })
        navigate(
          buildPersonaGardenRoute({
            personaId,
            tab: "profiles"
          })
        )
      } catch (error: any) {
        notification.error({
          message: t("settings:manageCharacters.personaGarden.createError", {
            defaultValue: "Failed to create persona"
          }),
          description:
            sanitizeServerErrorMessage(
              error?.message,
              t("settings:manageCharacters.notification.someError", {
                defaultValue: "Something went wrong. Please try again later"
              })
            ) ||
            t("settings:manageCharacters.notification.someError", {
              defaultValue: "Something went wrong. Please try again later"
            })
        })
      } finally {
        setPersonaCreatePending(characterId, false)
      }
    },
    [
      buildSuggestedPersonaName,
      getCharacterNumericId,
      loadPersonaGardenActionContext,
      navigate,
      notification,
      setPersonaCreatePending,
      t
    ]
  )

  return {
    // Mutations
    createCharacter,
    creating,
    updateCharacter,
    updating,
    deleteCharacter,
    deleting,
    restoreCharacter,

    // World book sync
    syncWorldBookSelection,

    // Action handlers
    handleExport,
    handleChat,
    handleChatInNewTab,
    handleEdit,
    handleDuplicate,
    handleDelete,
    handleViewConversations,
    handleRestoreFromTrash,

    // Default character
    isDefaultCharacterRecord,
    handleSetDefaultCharacter,
    handleClearDefaultCharacter,

    // Favorites
    isCharacterFavoriteRecord,
    handleToggleFavorite,

    // Persona Garden
    isPersonaCreatePending,
    getCreatePersonaActionLabel,
    openPersonaGardenForCharacter,
    createPersonaFromCharacter,

    // Conversation state
    characterChats,
    setCharacterChats,
    chatsError,
    setChatsError,
    loadingChats,
    setLoadingChats,
    resumingChatId,
    setResumingChatId,

    // Store message option setters (for conversations drawer)
    setHistory,
    setMessages,
    setHistoryId,
    setServerChatId,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef,

    // Delete state (for undo)
    pendingDelete,
    undoDeleteRef,
    bulkUndoDeleteRef,

    // Conversation loading
    conversationsLoadErrorMessageRef
  }
}
