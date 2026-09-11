import { resolveRecipePersistenceOwnerView } from "@/services/recipe-persistence-uncertainty"
import React, { useMemo, useState } from "react"
import { useMutation, type QueryClient } from "@tanstack/react-query"
import { notification } from "antd"
import {
  deletePromptById,
  updatePrompt,
  restorePrompt,
  exportPrompts
} from "@/db/dexie/helpers"
import { autoSyncPrompt } from "@/services/prompt-sync"
import { buildBulkCountSummary, collectFailedIds } from "../bulk-result-utils"

export interface UsePromptBulkActionsDeps {
  queryClient: QueryClient
  data: any[] | undefined
  isOnline: boolean
  isFireFoxPrivateMode: boolean
  t: (key: string, opts?: Record<string, any>) => string
  guardPrivateMode: () => boolean
  getPromptKeywords: (prompt: any) => string[]
  buildPromptUpdatePayload: (prompt: any, overrides?: Partial<any>) => any
  confirmDanger: (options: any) => Promise<boolean>
}

export function usePromptBulkActions(deps: UsePromptBulkActionsDeps) {
  const {
    queryClient,
    data,
    isOnline,
    t,
    guardPrivateMode,
    getPromptKeywords,
    buildPromptUpdatePayload,
    confirmDanger
  } = deps

  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])
  const [trashSelectedRowKeys, setTrashSelectedRowKeys] = useState<React.Key[]>([])
  const [bulkKeywordModalOpen, setBulkKeywordModalOpen] = useState(false)
  const [bulkKeywordValue, setBulkKeywordValue] = useState("")

  const selectedPromptRows = useMemo(() => {
    const selectedIds = new Set(selectedRowKeys.map((key) => String(key)))
    return (data || []).filter((prompt: any) => selectedIds.has(prompt.id))
  }, [data, selectedRowKeys])

  const allSelectedAreFavorite = useMemo(() => {
    return (
      selectedPromptRows.length > 0 &&
      selectedPromptRows.every((prompt: any) => !!prompt?.favorite)
    )
  }, [selectedPromptRows])

  const { mutate: bulkDeletePrompts, isPending: isBulkDeleting } = useMutation({
    mutationFn: async (ids: string[]) => {
      const results = await Promise.allSettled(ids.map((id) => deletePromptById(id)))
      const failedIds = collectFailedIds(ids, results)
      const counts = buildBulkCountSummary(ids.length, failedIds.length)
      return {
        total: counts.total,
        deleted: counts.succeeded,
        failedIds
      }
    },
    onSuccess: ({ total, deleted, failedIds }) => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
      queryClient.invalidateQueries({
        queryKey: ["fetchDeletedPrompts"]
      })
      if (failedIds.length > 0) {
        setSelectedRowKeys(failedIds)
        notification.warning({
          message: t("managePrompts.notification.bulkDeletePartial", {
            defaultValue: "Bulk delete completed with issues"
          }),
          description: t("managePrompts.notification.bulkDeletePartialDesc", {
            defaultValue: "Deleted {{deleted}} of {{total}} prompts. {{failed}} failed.",
            deleted,
            total,
            failed: failedIds.length
          })
        })
      } else {
        setSelectedRowKeys([])
        notification.success({
          message: t("managePrompts.notification.bulkDeletedSuccess", { defaultValue: "Prompts deleted" }),
          description: t("managePrompts.notification.bulkDeletedSuccessDesc", { defaultValue: "Selected prompts have been deleted." })
        })
      }
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.notification.error"),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: bulkToggleFavorite, isPending: isBulkFavoriting } = useMutation({
    mutationFn: async ({
      ids,
      favorite
    }: {
      ids: string[]
      favorite: boolean
    }) => {
      const promptById = new Map(
        (data || []).map((prompt: any) => [String(prompt.id), prompt])
      )
      const results = await Promise.allSettled(
        ids.map(async (id) => {
          const prompt = promptById.get(id)
          if (!prompt) {
            throw new Error("Prompt not found")
          }
          await updatePrompt(
            buildPromptUpdatePayload(prompt, {
              favorite
            })
          )
        })
      )
      const failedIds: string[] = []
      for (let index = 0; index < results.length; index += 1) {
        if (results[index]?.status === "rejected") {
          failedIds.push(ids[index]!)
        }
      }
      return {
        total: ids.length,
        updated: ids.length - failedIds.length,
        failedIds
      }
    },
    onSuccess: ({ total, updated, failedIds }, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
      if (failedIds.length > 0) {
        setSelectedRowKeys(failedIds)
        notification.warning({
          message: t("managePrompts.bulk.favoritePartial", {
            defaultValue: "Bulk favorite update completed with issues"
          }),
          description: t("managePrompts.bulk.favoritePartialDesc", {
            defaultValue:
              "Updated {{updated}} of {{total}} prompts. {{failed}} failed.",
            updated,
            total,
            failed: failedIds.length
          })
        })
        return
      }
      notification.success({
        message: variables.favorite
          ? t("managePrompts.bulk.favoriteSuccess", {
              defaultValue: "Selected prompts favorited"
            })
          : t("managePrompts.bulk.unfavoriteSuccess", {
              defaultValue: "Selected prompts unfavorited"
            }),
        description: t("managePrompts.bulk.favoriteSuccessDesc", {
          defaultValue: "Updated {{count}} prompts.",
          count: updated
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

  const { mutate: bulkAddKeyword, isPending: isBulkAddingKeyword } = useMutation({
    mutationFn: async ({
      ids,
      keyword
    }: {
      ids: string[]
      keyword: string
    }) => {
      const trimmedKeyword = keyword.trim()
      if (!trimmedKeyword) {
        throw new Error(
          t("managePrompts.tags.keywordRequired", {
            defaultValue: "Keyword is required."
          })
        )
      }

      const promptById = new Map(
        (data || []).map((prompt: any) => [String(prompt.id), prompt])
      )
      const results = await Promise.allSettled(
        ids.map(async (id) => {
          const prompt = promptById.get(id)
          if (!prompt) {
            throw new Error("Prompt not found")
          }
          const existingKeywords = getPromptKeywords(prompt) || []
          if (existingKeywords.includes(trimmedKeyword)) {
            return { skipped: true }
          }
          const nextKeywords = [...existingKeywords, trimmedKeyword]
          await updatePrompt(
            buildPromptUpdatePayload(prompt, {
              keywords: nextKeywords,
              tags: nextKeywords
            })
          )
          return { skipped: false }
        })
      )

      let updated = 0
      let skipped = 0
      const failedIds: string[] = []
      for (let index = 0; index < results.length; index += 1) {
        const result = results[index]
        if (result?.status === "rejected") {
          failedIds.push(ids[index]!)
          continue
        }
        if (result.value?.skipped) {
          skipped += 1
        } else {
          updated += 1
        }
      }
      return {
        total: ids.length,
        updated,
        skipped,
        failedIds,
        keyword: trimmedKeyword
      }
    },
    onSuccess: ({ total, updated, skipped, failedIds, keyword }) => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
      setBulkKeywordModalOpen(false)
      setBulkKeywordValue("")

      if (failedIds.length > 0) {
        setSelectedRowKeys(failedIds)
        notification.warning({
          message: t("managePrompts.bulk.keywordPartial", {
            defaultValue: "Bulk keyword update completed with issues"
          }),
          description: t("managePrompts.bulk.keywordPartialDesc", {
            defaultValue:
              "Updated {{updated}}, skipped {{skipped}}, failed {{failed}} of {{total}} prompts.",
            updated,
            skipped,
            failed: failedIds.length,
            total
          })
        })
        return
      }

      notification.success({
        message: t("managePrompts.bulk.keywordSuccess", {
          defaultValue: "Keyword added to selected prompts"
        }),
        description: t("managePrompts.bulk.keywordSuccessDesc", {
          defaultValue:
            "Added '{{keyword}}' to {{updated}} prompts ({{skipped}} already had it).",
          keyword,
          updated,
          skipped
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

  const { mutate: bulkPushToServer, isPending: isBulkPushing } = useMutation({
    mutationFn: async (ids: string[]) => {
      const owner = await resolveRecipePersistenceOwnerView()
      const persistenceInput = owner ? { expectedOwnerId: owner.ownerId } : undefined
      const results = await Promise.allSettled(
        ids.map(async (id) => {
          const result = await autoSyncPrompt(id, undefined, persistenceInput)
          if (!result.success) {
            throw new Error(
              result.error ||
                t("managePrompts.sync.pendingTooltip", {
                  defaultValue: "Local changes not yet synced"
                })
            )
          }
        })
      )

      const failedIds: string[] = []
      for (let index = 0; index < results.length; index += 1) {
        if (results[index]?.status === "rejected") {
          failedIds.push(ids[index]!)
        }
      }
      return {
        total: ids.length,
        synced: ids.length - failedIds.length,
        failedIds
      }
    },
    onSuccess: ({ total, synced, failedIds }) => {
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      if (failedIds.length > 0) {
        setSelectedRowKeys(failedIds)
        notification.warning({
          message: t("managePrompts.sync.bulkPushPartial", {
            defaultValue: "Bulk sync completed with issues"
          }),
          description: t("managePrompts.sync.bulkPushPartialDesc", {
            defaultValue:
              "Synced {{synced}} of {{total}} prompts. {{failed}} failed.",
            synced,
            total,
            failed: failedIds.length
          })
        })
        return
      }
      notification.success({
        message: t("managePrompts.sync.bulkPushSuccess", {
          defaultValue: "Selected prompts synced"
        }),
        description: t("managePrompts.sync.bulkPushSuccessDesc", {
          defaultValue: "Synced {{count}} prompts to the server.",
          count: synced
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

  const { mutate: bulkRestorePrompts, isPending: isBulkRestoring } = useMutation({
    mutationFn: async (ids: string[]) => {
      const results = await Promise.allSettled(ids.map((id) => restorePrompt(id)))
      const failedIds = collectFailedIds(ids, results)
      const counts = buildBulkCountSummary(ids.length, failedIds.length)
      return {
        total: counts.total,
        restored: counts.succeeded,
        failedIds
      }
    },
    onSuccess: ({ total, restored, failedIds }) => {
      queryClient.invalidateQueries({
        queryKey: ["fetchAllPrompts"]
      })
      queryClient.invalidateQueries({
        queryKey: ["fetchDeletedPrompts"]
      })

      if (failedIds.length > 0) {
        setTrashSelectedRowKeys(failedIds)
        notification.warning({
          message: t("managePrompts.trash.bulkRestorePartial", {
            defaultValue: "Bulk restore completed with issues"
          }),
          description: t("managePrompts.trash.bulkRestorePartialDesc", {
            defaultValue: "Restored {{restored}} of {{total}} prompts. {{failed}} failed.",
            restored,
            total,
            failed: failedIds.length
          })
        })
        return
      }

      setTrashSelectedRowKeys([])
      notification.success({
        message: t("managePrompts.trash.bulkRestoreSuccess", {
          defaultValue: "Prompts restored"
        }),
        description: t("managePrompts.trash.bulkRestoreSuccessDesc", {
          defaultValue: "Restored {{count}} prompts from trash.",
          count: restored
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

  const triggerBulkExport = React.useCallback(async () => {
    try {
      if (guardPrivateMode()) return
      const selectedItems = (data || []).filter((p: any) => selectedRowKeys.includes(p.id))
      const blob = new Blob([JSON.stringify(selectedItems, null, 2)], {
        type: "application/json"
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `prompts_selected_${new Date().toISOString()}.json`
      a.click()
      URL.revokeObjectURL(url)
      setSelectedRowKeys([])
    } catch (e) {
      notification.error({
        message: t("managePrompts.notification.error"),
        description: t("managePrompts.notification.someError")
      })
    }
  }, [data, guardPrivateMode, selectedRowKeys, t])

  return {
    // state
    selectedRowKeys,
    setSelectedRowKeys,
    trashSelectedRowKeys,
    setTrashSelectedRowKeys,
    bulkKeywordModalOpen,
    setBulkKeywordModalOpen,
    bulkKeywordValue,
    setBulkKeywordValue,
    // computed
    selectedPromptRows,
    allSelectedAreFavorite,
    // mutations
    bulkDeletePrompts,
    isBulkDeleting,
    bulkToggleFavorite,
    isBulkFavoriting,
    bulkAddKeyword,
    isBulkAddingKeyword,
    bulkPushToServer,
    isBulkPushing,
    bulkRestorePrompts,
    isBulkRestoring,
    // callbacks
    triggerBulkExport
  }
}
