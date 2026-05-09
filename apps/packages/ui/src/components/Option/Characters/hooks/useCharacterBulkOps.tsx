import React from "react"
import { type QueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { exportCharactersToJSON } from "@/utils/character-export"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import {
  emitCharacterRecoveryTelemetry,
  normalizeCharacterComparisonValue,
  formatCharacterComparisonValue,
  toComparisonFilenameSegment,
  CHARACTER_COMPARISON_FIELDS,
  type CharacterTableSortOrder,
  type CharacterComparisonRow
} from "../utils"

export interface UseCharacterBulkOpsDeps {
  t: (key: string, opts?: Record<string, any>) => string
  notification: {
    error: (args: { message: string; description?: any; duration?: number }) => void
    warning: (args: { message: string; description?: any; duration?: number }) => void
    success: (args: { message: string; description?: any; duration?: number }) => void
    info: (args: { message: string; description?: any; duration?: number }) => void
  }
  qc: QueryClient
  /** Current page data from useCharacterData */
  data: any[]
  /** Filter/page deps for clearing selection on change */
  characterListScope: "active" | "deleted"
  creatorFilter: string | undefined
  currentPage: number
  debouncedSearchTerm: string
  filterTags: string[]
  folderFilterId: string | undefined
  favoritesOnly: boolean
  hasConversationsOnly: boolean
  matchAllTags: boolean
  pageSize: number
  sortColumn: string | null
  sortOrder: CharacterTableSortOrder
  /** From useCharacterCrud */
  bulkUndoDeleteRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>
  setBulkOperationLoading: (loading: boolean) => void
  /** From useCharacterModalState */
  compareModalOpen: boolean
  setCompareModalOpen: (open: boolean) => void
  compareCharacters: [any, any] | null
  setCompareCharacters: (chars: [any, any] | null) => void
  closeCompareModal: () => void
  /** From useCharacterTagManagement */
  handleBulkAddTags: (selectedIds: Set<string>, data: any[]) => Promise<void>
}

export function useCharacterBulkOps(deps: UseCharacterBulkOpsDeps) {
  const {
    t,
    notification,
    qc,
    data,
    characterListScope,
    creatorFilter,
    currentPage,
    debouncedSearchTerm,
    filterTags,
    folderFilterId,
    favoritesOnly,
    hasConversationsOnly,
    matchAllTags,
    pageSize,
    sortColumn,
    sortOrder,
    bulkUndoDeleteRef,
    setBulkOperationLoading,
    compareModalOpen,
    setCompareModalOpen,
    compareCharacters,
    setCompareCharacters,
    closeCompareModal,
    handleBulkAddTags
  } = deps

  const confirmDanger = useConfirmDanger()

  // --- Bulk selection state ---
  const [selectedCharacterIds, setSelectedCharacterIds] = React.useState<Set<string>>(new Set())

  const toggleCharacterSelection = React.useCallback((id: string) => {
    setSelectedCharacterIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const selectAllOnPage = React.useCallback(() => {
    if (!Array.isArray(data)) return
    const pageIds = data.map((c: any) => String(c.id || c.slug || c.name))
    setSelectedCharacterIds((prev) => new Set([...prev, ...pageIds]))
  }, [data])

  const clearSelection = React.useCallback(() => {
    setSelectedCharacterIds(new Set())
  }, [])

  const selectedCount = selectedCharacterIds.size
  const hasSelection = selectedCount > 0

  const selectedCharacters = React.useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return []
    return data.filter((character: any) =>
      selectedCharacterIds.has(String(character.id || character.slug || character.name))
    )
  }, [data, selectedCharacterIds])

  const allOnPageSelected = React.useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return false
    const pageIds = data.map((c: any) => String(c.id || c.slug || c.name))
    return pageIds.length > 0 && pageIds.every((id) => selectedCharacterIds.has(id))
  }, [data, selectedCharacterIds])

  const someOnPageSelected = React.useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return false
    const pageIds = data.map((c: any) => String(c.id || c.slug || c.name))
    const selectedOnPage = pageIds.filter((id) => selectedCharacterIds.has(id)).length
    return selectedOnPage > 0 && selectedOnPage < pageIds.length
  }, [data, selectedCharacterIds])

  // Clear selection when the active query/page changes
  React.useEffect(() => {
    setSelectedCharacterIds(new Set())
  }, [
    characterListScope,
    creatorFilter,
    currentPage,
    debouncedSearchTerm,
    filterTags,
    folderFilterId,
    favoritesOnly,
    hasConversationsOnly,
    matchAllTags,
    pageSize,
    sortColumn,
    sortOrder
  ])

  // --- Restore bulk-deleted characters ---
  const restoreBulkDeletedCharacters = React.useCallback(
    async (deletedCharacters: Array<{ id: string; version?: number }>) => {
      let restoredCount = 0
      let failedCount = 0

      for (const deletedCharacter of deletedCharacters) {
        try {
          await tldwClient.restoreCharacter(
            deletedCharacter.id,
            (deletedCharacter.version ?? 0) + 1
          )
          restoredCount++
        } catch {
          failedCount++
        }
      }

      qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })

      if (failedCount === 0) {
        emitCharacterRecoveryTelemetry("bulk_restore", {
          restored_count: restoredCount
        })
        notification.success({
          message: t("settings:manageCharacters.bulk.restoreSuccess", {
            defaultValue: "Restored {{count}} characters",
            count: restoredCount
          })
        })
      } else {
        emitCharacterRecoveryTelemetry("bulk_restore_failed", {
          restored_count: restoredCount,
          failed_count: failedCount
        })
        notification.warning({
          message: t("settings:manageCharacters.bulk.restorePartial", {
            defaultValue: "Restored {{success}} characters, {{fail}} failed",
            success: restoredCount,
            fail: failedCount
          })
        })
      }
    },
    [notification, qc, t]
  )

  // --- Bulk delete ---
  const handleBulkDelete = React.useCallback(async () => {
    if (selectedCharacterIds.size === 0) return

    const selectedChars = (data || []).filter((c: any) =>
      selectedCharacterIds.has(String(c.id || c.slug || c.name))
    )

    const ok = await confirmDanger({
      title: t("settings:manageCharacters.bulk.deleteTitle", {
        defaultValue: "Delete {{count}} characters?",
        count: selectedChars.length
      }),
      content: t("settings:manageCharacters.bulk.deleteContent", {
        defaultValue:
          "This will soft-delete {{count}} characters. You can undo for 10 seconds.",
        count: selectedChars.length
      }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })

    if (!ok) return

    setBulkOperationLoading(true)
    const deletedCharacters: Array<{ id: string; version?: number }> = []
    let successCount = 0
    let failCount = 0

    for (const char of selectedChars) {
      try {
        const id = String(char.id || char.slug || char.name)
        await tldwClient.deleteCharacter(id, char.version)
        deletedCharacters.push({ id, version: char.version })
        successCount++
      } catch {
        failCount++
      }
    }

    setBulkOperationLoading(false)
    setSelectedCharacterIds(new Set())
    qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })

    if (successCount > 0) {
      emitCharacterRecoveryTelemetry("bulk_delete", {
        deleted_count: successCount,
        failed_count: failCount
      })
      if (bulkUndoDeleteRef.current) {
        clearTimeout(bulkUndoDeleteRef.current)
        bulkUndoDeleteRef.current = null
      }

      const timeoutId = setTimeout(() => {
        bulkUndoDeleteRef.current = null
        qc.invalidateQueries({ queryKey: ["tldw:listCharacters"] })
      }, 10000)
      bulkUndoDeleteRef.current = timeoutId

      notification.info({
        message:
          failCount === 0
            ? t("settings:manageCharacters.bulk.deleteSuccess", {
                defaultValue: "Deleted {{count}} characters",
                count: successCount
              })
            : t("settings:manageCharacters.bulk.deletePartial", {
                defaultValue: "Deleted {{success}} characters, {{fail}} failed",
                success: successCount,
                fail: failCount
              }),
        description: (
          <button
            type="button"
            className="mt-1 text-sm font-medium text-primary hover:underline"
            onClick={() => {
              if (bulkUndoDeleteRef.current) {
                clearTimeout(bulkUndoDeleteRef.current)
                bulkUndoDeleteRef.current = null
              }
              emitCharacterRecoveryTelemetry("bulk_undo", {
                deleted_count: deletedCharacters.length
              })
              void restoreBulkDeletedCharacters(deletedCharacters)
            }}>
            {t("common:undo", { defaultValue: "Undo" })}
          </button>
        ),
        duration: 10
      })
    } else {
      notification.warning({
        message: t("settings:manageCharacters.bulk.deleteFailure", {
          defaultValue: "Unable to delete selected characters"
        })
      })
    }
  }, [
    selectedCharacterIds,
    data,
    confirmDanger,
    t,
    setBulkOperationLoading,
    notification,
    qc,
    bulkUndoDeleteRef,
    restoreBulkDeletedCharacters
  ])

  // --- Bulk export ---
  const handleBulkExport = React.useCallback(async () => {
    if (selectedCharacterIds.size === 0) return

    setBulkOperationLoading(true)
    const selectedChars = (data || []).filter((c: any) =>
      selectedCharacterIds.has(String(c.id || c.slug || c.name))
    )

    const exportedCharacters: any[] = []
    let failCount = 0

    try {
      for (const char of selectedChars) {
        try {
          const exported = await tldwClient.exportCharacter(
            String(char.id || char.slug || char.name),
            { format: 'v3' }
          )
          exportedCharacters.push(exported)
        } catch {
          failCount++
        }
      }

      if (exportedCharacters.length > 0) {
        exportCharactersToJSON(exportedCharacters)
      }
    } finally {
      setBulkOperationLoading(false)
    }

    if (failCount === 0) {
      notification.success({
        message: t("settings:manageCharacters.bulk.exportSuccess", {
          defaultValue: "Exported {{count}} characters",
          count: exportedCharacters.length
        })
      })
    } else {
      notification.warning({
        message: t("settings:manageCharacters.bulk.exportPartial", {
          defaultValue: "Exported {{success}} characters, {{fail}} failed",
          success: exportedCharacters.length,
          fail: failCount
        })
      })
    }
  }, [selectedCharacterIds, data, setBulkOperationLoading, notification, t])

  // --- Bulk add tags bridge ---
  const handleBulkAddTagsForSelection = React.useCallback(async () => {
    await handleBulkAddTags(selectedCharacterIds, data || [])
  }, [handleBulkAddTags, selectedCharacterIds, data])

  // --- Comparison modal ---
  const comparisonRows = React.useMemo<CharacterComparisonRow[]>(() => {
    if (!compareCharacters) return []
    const [leftCharacter, rightCharacter] = compareCharacters
    return CHARACTER_COMPARISON_FIELDS.map((fieldDef) => {
      const leftRawValue = fieldDef.getValue(leftCharacter)
      const rightRawValue = fieldDef.getValue(rightCharacter)
      return {
        field: fieldDef.field,
        label: fieldDef.label,
        leftValue: formatCharacterComparisonValue(leftRawValue),
        rightValue: formatCharacterComparisonValue(rightRawValue),
        different:
          normalizeCharacterComparisonValue(leftRawValue) !==
          normalizeCharacterComparisonValue(rightRawValue)
      }
    })
  }, [compareCharacters])

  const changedComparisonRows = React.useMemo(
    () => comparisonRows.filter((row) => row.different),
    [comparisonRows]
  )

  const comparisonSummaryText = React.useMemo(() => {
    if (!compareCharacters) return ""
    const [leftCharacter, rightCharacter] = compareCharacters
    const leftName = String(leftCharacter?.name || leftCharacter?.id || "Character A")
    const rightName = String(rightCharacter?.name || rightCharacter?.id || "Character B")

    const lines: string[] = [
      "Character comparison summary",
      `Left: ${leftName} (${leftCharacter?.id ?? "n/a"})`,
      `Right: ${rightName} (${rightCharacter?.id ?? "n/a"})`,
      `Different fields: ${changedComparisonRows.length}/${comparisonRows.length}`,
      ""
    ]

    if (changedComparisonRows.length === 0) {
      lines.push("No tracked differences found.")
      return lines.join("\n")
    }

    for (const row of changedComparisonRows) {
      lines.push(`[${row.label}]`)
      lines.push(`Left: ${row.leftValue}`)
      lines.push(`Right: ${row.rightValue}`)
      lines.push("")
    }

    return lines.join("\n")
  }, [changedComparisonRows, compareCharacters, comparisonRows.length])

  const handleOpenCompareModal = React.useCallback(() => {
    if (selectedCharacters.length !== 2) {
      notification.warning({
        message: t("settings:manageCharacters.compare.selectTwo", {
          defaultValue: "Select exactly two characters to compare."
        })
      })
      return
    }

    setCompareCharacters([selectedCharacters[0], selectedCharacters[1]])
    setCompareModalOpen(true)
  }, [notification, selectedCharacters, setCompareCharacters, setCompareModalOpen, t])

  const handleCopyComparisonSummary = React.useCallback(async () => {
    if (!comparisonSummaryText.trim()) return

    try {
      if (
        typeof navigator === "undefined" ||
        !navigator.clipboard ||
        typeof navigator.clipboard.writeText !== "function"
      ) {
        throw new Error("Clipboard API unavailable")
      }
      await navigator.clipboard.writeText(comparisonSummaryText)
      notification.success({
        message: t("settings:manageCharacters.compare.copySuccess", {
          defaultValue: "Copied comparison summary"
        })
      })
    } catch {
      notification.warning({
        message: t("settings:manageCharacters.compare.copyFailure", {
          defaultValue: "Unable to copy comparison summary"
        })
      })
    }
  }, [comparisonSummaryText, notification, t])

  const handleExportComparisonSummary = React.useCallback(() => {
    if (!compareCharacters || !comparisonSummaryText.trim()) return

    if (
      typeof URL === "undefined" ||
      typeof URL.createObjectURL !== "function" ||
      typeof URL.revokeObjectURL !== "function"
    ) {
      notification.warning({
        message: t("settings:manageCharacters.compare.exportFailure", {
          defaultValue: "Unable to export comparison summary"
        })
      })
      return
    }

    try {
      const [leftCharacter, rightCharacter] = compareCharacters
      const leftSegment = toComparisonFilenameSegment(
        leftCharacter?.name || leftCharacter?.id
      )
      const rightSegment = toComparisonFilenameSegment(
        rightCharacter?.name || rightCharacter?.id
      )
      const fileName = `character-compare-${leftSegment}-vs-${rightSegment}.txt`

      const blob = new Blob([comparisonSummaryText], {
        type: "text/plain;charset=utf-8"
      })
      const objectUrl = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = objectUrl
      link.download = fileName
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(objectUrl)

      notification.success({
        message: t("settings:manageCharacters.compare.exportSuccess", {
          defaultValue: "Exported comparison summary"
        })
      })
    } catch {
      notification.warning({
        message: t("settings:manageCharacters.compare.exportFailure", {
          defaultValue: "Unable to export comparison summary"
        })
      })
    }
  }, [compareCharacters, comparisonSummaryText, notification, t])

  // Auto-close compare modal when selection changes
  React.useEffect(() => {
    if (!compareModalOpen) return
    if (selectedCount !== 2) {
      closeCompareModal()
    }
  }, [closeCompareModal, compareModalOpen, selectedCount])

  return {
    // Selection state
    selectedCharacterIds,
    setSelectedCharacterIds,
    toggleCharacterSelection,
    selectAllOnPage,
    clearSelection,
    selectedCount,
    hasSelection,
    selectedCharacters,
    allOnPageSelected,
    someOnPageSelected,

    // Bulk actions
    handleBulkDelete,
    handleBulkExport,
    handleBulkAddTagsForSelection,

    // Comparison
    comparisonRows,
    changedComparisonRows,
    comparisonSummaryText,
    handleOpenCompareModal,
    handleCopyComparisonSummary,
    handleExportComparisonSummary
  }
}
