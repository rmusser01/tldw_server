/**
 * Hook: useWritingContextComposition
 *
 * Manages context sources (memory block, author note, world info),
 * context assembly, context preview, and context order configuration.
 */

import React from "react"
import type { TFunction } from "i18next"
import {
  buildContextSystemMessages,
  composeContextPrompt,
  injectSystemMessages,
  type WritingAuthorNote,
  type WritingContextBlock,
  type WritingWorldInfoEntry,
  type WritingWorldInfoSettings
} from "../writing-context-utils"
import { buildWorldInfoExportPayload } from "../writing-world-info-transfer-utils"
import { moveWorldInfoEntry } from "../writing-world-info-utils"
import {
  buildContextPreviewFilename,
  serializeContextPreviewJson
} from "../writing-context-preview-utils"
import {
  applyFimTemplate,
  buildChatMessages,
  buildFillPrompt,
  createWorldInfoId,
  resolveGenerationPlan,
  toNumber,
  type NormalizedTemplate,
  type WritingSessionSettings
} from "./utils"

export interface UseWritingContextCompositionDeps {
  settings: WritingSessionSettings
  editorText: string
  chatMode: boolean
  effectiveTemplate: NormalizedTemplate
  settingsDisabled: boolean
  updateSetting: (partial: Partial<WritingSessionSettings>) => void
  t: TFunction
}

export function useWritingContextComposition(deps: UseWritingContextCompositionDeps) {
  const {
    settings,
    editorText,
    chatMode,
    effectiveTemplate,
    settingsDisabled,
    updateSetting,
    t
  } = deps

  const [contextPreviewModalOpen, setContextPreviewModalOpen] = React.useState(false)

  const memoryBlock = settings.memory_block
  const authorNote = settings.author_note
  const worldInfo = settings.world_info
  const worldInfoEntries = worldInfo.entries

  const updateMemoryBlock = React.useCallback(
    (patch: Partial<WritingContextBlock>) => {
      updateSetting({
        memory_block: {
          ...memoryBlock,
          ...patch
        }
      })
    },
    [memoryBlock, updateSetting]
  )

  const updateAuthorNote = React.useCallback(
    (patch: Partial<WritingAuthorNote>) => {
      updateSetting({
        author_note: {
          ...authorNote,
          ...patch,
          insertion_depth: Math.max(
            1,
            Math.floor(
              toNumber(
                patch.insertion_depth ?? authorNote.insertion_depth,
                authorNote.insertion_depth
              )
            )
          )
        }
      })
    },
    [authorNote, updateSetting]
  )

  const updateWorldInfo = React.useCallback(
    (patch: Partial<WritingWorldInfoSettings>) => {
      updateSetting({
        world_info: {
          ...worldInfo,
          ...patch,
          search_range: Math.max(
            0,
            Math.floor(
              toNumber(
                patch.search_range ?? worldInfo.search_range,
                worldInfo.search_range
              )
            )
          )
        }
      })
    },
    [updateSetting, worldInfo]
  )

  const addWorldInfoEntry = React.useCallback(() => {
    const entry: WritingWorldInfoEntry = {
      id: createWorldInfoId(),
      display_name: "",
      enabled: true,
      keys: [],
      content: "",
      use_regex: false,
      case_sensitive: false,
      search_range: undefined
    }
    updateWorldInfo({
      entries: [...worldInfoEntries, entry]
    })
  }, [updateWorldInfo, worldInfoEntries])

  const updateWorldInfoEntry = React.useCallback(
    (entryId: string, patch: Partial<WritingWorldInfoEntry>) => {
      updateWorldInfo({
        entries: worldInfoEntries.map((entry) =>
          entry.id === entryId
            ? {
                ...entry,
                ...patch
              }
            : entry
        )
      })
    },
    [updateWorldInfo, worldInfoEntries]
  )

  const removeWorldInfoEntry = React.useCallback(
    (entryId: string) => {
      updateWorldInfo({
        entries: worldInfoEntries.filter((entry) => entry.id !== entryId)
      })
    },
    [updateWorldInfo, worldInfoEntries]
  )

  const moveWorldInfoEntryById = React.useCallback(
    (entryId: string, direction: "up" | "down") => {
      updateWorldInfo({
        entries: moveWorldInfoEntry(worldInfoEntries, entryId, direction)
      })
    },
    [updateWorldInfo, worldInfoEntries]
  )

  const handleWorldInfoExport = React.useCallback(() => {
    const payload = buildWorldInfoExportPayload(worldInfo)
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json"
    })
    const now = new Date()
    const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(
      2,
      "0"
    )}-${String(now.getDate()).padStart(2, "0")}`
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `writing-world-info-${stamp}.json`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }, [worldInfo])

  const handleWorldInfoImported = React.useCallback(
    (
      nextWorldInfo: WritingWorldInfoSettings,
      _mode: "append" | "replace"
    ) => {
      updateWorldInfo(nextWorldInfo)
    },
    [updateWorldInfo]
  )

  const handleWorldInfoImportError = React.useCallback(
    (_detail: string) => {
      // Error reporting is handled by the component
    },
    []
  )

  // --- Context preview ---
  const contextPreviewMessages = React.useMemo(() => {
    const plan = resolveGenerationPlan(editorText)
    const fimPrompt =
      plan.mode === "fill"
        ? applyFimTemplate(effectiveTemplate, plan.prefix, plan.suffix)
        : null
    const promptText =
      plan.mode === "fill"
        ? fimPrompt ?? buildFillPrompt(plan.prefix, plan.suffix)
        : plan.prefix
    const contextSettings = {
      memory_block: settings.memory_block,
      author_note: settings.author_note,
      world_info: settings.world_info,
      context_order: settings.context_order,
      context_length: settings.context_length,
      author_note_depth_mode: settings.author_note_depth_mode
    }
    const contextComposedPrompt = chatMode
      ? promptText
      : composeContextPrompt(promptText, contextSettings)
    const baseMessages = buildChatMessages(
      contextComposedPrompt,
      effectiveTemplate,
      chatMode
    )
    const contextMessages = chatMode
      ? buildContextSystemMessages(editorText, contextSettings)
      : []
    return injectSystemMessages(baseMessages, contextMessages)
  }, [
    chatMode,
    editorText,
    effectiveTemplate,
    settings.author_note,
    settings.author_note_depth_mode,
    settings.context_length,
    settings.context_order,
    settings.memory_block,
    settings.world_info
  ])

  const contextPreviewJson = React.useMemo(
    () => serializeContextPreviewJson(contextPreviewMessages),
    [contextPreviewMessages]
  )

  const handleCopyContextPreview = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(contextPreviewJson)
    } catch {
      // Error handling is done by the caller
    }
  }, [contextPreviewJson])

  const handleExportContextPreview = React.useCallback(() => {
    const blob = new Blob([contextPreviewJson], {
      type: "application/json"
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = buildContextPreviewFilename(new Date())
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }, [contextPreviewJson])

  return {
    // state
    contextPreviewModalOpen, setContextPreviewModalOpen,
    // derived
    memoryBlock,
    authorNote,
    worldInfo,
    worldInfoEntries,
    contextPreviewMessages,
    contextPreviewJson,
    // callbacks
    updateMemoryBlock,
    updateAuthorNote,
    updateWorldInfo,
    addWorldInfoEntry,
    updateWorldInfoEntry,
    removeWorldInfoEntry,
    moveWorldInfoEntryById,
    handleWorldInfoExport,
    handleWorldInfoImported,
    handleWorldInfoImportError,
    handleCopyContextPreview,
    handleExportContextPreview
  }
}
