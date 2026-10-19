import React, { useRef, useState } from "react"
import { notification } from "antd"
import type { QueryClient } from "@tanstack/react-query"
import {
  exportPrompts,
  importPromptsV2
} from "@/db/dexie/helpers"
import { exportPromptsServer } from "@/services/prompts-api"
import {
  getPromptImportNotificationCopy,
  normalizePromptImportCounts
} from "../prompt-import-utils"
import {
  getPromptImportErrorNotice,
  parseImportPromptsPayload
} from "../prompt-import-error-utils"

export interface UsePromptImportExportDeps {
  expectedOwnerId?: string | null
  queryClient: QueryClient
  data: any[] | undefined
  isOnline: boolean
  t: (key: string, opts?: Record<string, any>) => string
  guardPrivateMode: () => boolean
  confirmDanger: (options: any) => Promise<boolean>
}

export function usePromptImportExport(deps: UsePromptImportExportDeps) {
  const {
    expectedOwnerId,
    queryClient,
    data,
    isOnline,
    t,
    guardPrivateMode,
    confirmDanger
  } = deps

  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [importMode, setImportMode] = useState<"merge" | "replace">("merge")
  const [exportFormat, setExportFormat] = useState<"json" | "csv" | "markdown">("json")
  const lifetime = useRef({ active: true, ownerId: expectedOwnerId, generation: 0 })
  if (lifetime.current.ownerId !== expectedOwnerId) {
    lifetime.current.ownerId = expectedOwnerId
    lifetime.current.generation++
  }
  React.useEffect(() => {
    const currentLifetime = lifetime.current
    currentLifetime.active = true
    const retire = () => { currentLifetime.generation++ }
    const configChanged = (event: Event) => {
      if ((event as CustomEvent<{ authorityChanged?: boolean }>).detail?.authorityChanged !== false) retire()
    }
    window.addEventListener("tldw:auth-principal-changed", retire)
    window.addEventListener("tldw:config-updated", configChanged)
    return () => {
      currentLifetime.active = false
      window.removeEventListener("tldw:auth-principal-changed", retire)
      window.removeEventListener("tldw:config-updated", configChanged)
    }
  }, [])

  const triggerExport = React.useCallback(async () => {
    const generation = lifetime.current.generation
    const isCurrent = () => lifetime.current.active && lifetime.current.ownerId === expectedOwnerId && lifetime.current.generation === generation
    try {
      if (guardPrivateMode()) return
      if (exportFormat === "json") {
        const items = await exportPrompts(expectedOwnerId)
        if (!isCurrent()) return
        const blob = new Blob([JSON.stringify(items, null, 2)], {
          type: "application/json"
        })
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        const safeStamp = new Date().toISOString().replace(/[:]/g, "-")
        a.download = `prompts_${safeStamp}.json`
        a.click()
        URL.revokeObjectURL(url)
        return
      }

      if (!isOnline) {
        notification.warning({
          message: t("managePrompts.exportOffline", {
            defaultValue: "Server export unavailable offline"
          }),
          description: t("managePrompts.exportOfflineDesc", {
            defaultValue: "Reconnect to export CSV or Markdown."
          })
        })
        return
      }

      const response = await exportPromptsServer(exportFormat, expectedOwnerId)
      if (!isCurrent()) return
      if (!response?.file_content_b64) {
        notification.info({
          message: t("managePrompts.exportEmpty", {
            defaultValue: "Nothing to export"
          }),
          description:
            response?.message ||
            t("managePrompts.exportEmptyDesc", {
              defaultValue: "No prompts matched the export criteria."
            })
        })
        return
      }

      const binary = atob(response.file_content_b64)
      const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
      const fileExtension = exportFormat === "csv" ? "csv" : "md"
      const mimeType = exportFormat === "csv" ? "text/csv" : "text/markdown"
      const blob = new Blob([bytes], {
        type: `${mimeType};charset=utf-8`
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      const safeStamp = new Date().toISOString().replace(/[:]/g, "-")
      a.download = `prompts_${safeStamp}.${fileExtension}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      if (!isCurrent()) return
      notification.error({
        message: t("managePrompts.notification.error"),
        description: t("managePrompts.notification.someError")
      })
    }
  }, [expectedOwnerId, exportFormat, guardPrivateMode, isOnline, t])

  const handleImportFile = React.useCallback(async (file: File) => {
    const generation = lifetime.current.generation
    const isCurrent = () => lifetime.current.active && lifetime.current.ownerId === expectedOwnerId && lifetime.current.generation === generation
    try {
      if (guardPrivateMode()) return
      const text = await file.text()
      if (!isCurrent()) return
      const prompts = parseImportPromptsPayload(text)

      if (importMode === "replace") {
        const currentPrompts = data || []
        const currentCount = currentPrompts.length

        const ok = await confirmDanger({
          title: t("managePrompts.importMode.replaceTitle", { defaultValue: "Replace all prompts?" }),
          content: t("managePrompts.importMode.replaceConfirmWithCount", {
            defaultValue:
              "This will delete {{currentCount}} existing prompts and import {{newCount}} new prompts. A backup will be downloaded automatically before replacing.",
            currentCount,
            newCount: prompts.length
          }),
          okText: t("managePrompts.importMode.replaceAndBackup", { defaultValue: "Backup & Replace" }),
          cancelText: t("common:cancel", { defaultValue: "Cancel" })
        })
        if (!ok || !isCurrent()) return

        if (currentCount > 0) {
          try {
            const backupItems = await exportPrompts(expectedOwnerId)
            if (!isCurrent()) return
            const blob = new Blob([JSON.stringify(backupItems, null, 2)], {
              type: "application/json"
            })
            const url = URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            const safeStamp = new Date().toISOString().replace(/[:]/g, "-")
            a.download = `prompts_backup_before_replace_${safeStamp}.json`
            a.click()
            URL.revokeObjectURL(url)
            await new Promise(resolve => setTimeout(resolve, 100))
          } catch (backupError) {
            if (!isCurrent()) return
            notification.warning({
              message: t("managePrompts.notification.backupFailed", { defaultValue: "Backup failed" }),
              description: t("managePrompts.notification.backupFailedDesc", {
                defaultValue: "Could not create backup, but proceeding with import."
              })
            })
          }
        }
      }

      if (!isCurrent()) return
      const importResult = await importPromptsV2(prompts, {
        expectedOwnerId,
        replaceExisting: importMode === "replace",
        mergeData: importMode === "merge"
      })
      if (!isCurrent()) return
      const importCounts = normalizePromptImportCounts(importResult, prompts.length)
      const importNotificationCopy = getPromptImportNotificationCopy(
        importMode,
        importCounts
      )
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      queryClient.invalidateQueries({ queryKey: ["fetchDeletedPrompts"] })
      notification.success({
        message: t("managePrompts.notification.addSuccess"),
        description: t(importNotificationCopy.key, {
          defaultValue: importNotificationCopy.defaultValue,
          ...importNotificationCopy.values
        })
      })
    } catch (e) {
      if (!isCurrent()) return
      const importErrorNotice = getPromptImportErrorNotice(e)
      if (importErrorNotice) {
        notification.error({
          message: t(importErrorNotice.titleKey, {
            defaultValue: importErrorNotice.titleDefaultValue
          }),
          description: t(importErrorNotice.descriptionKey, {
            defaultValue: importErrorNotice.descriptionDefaultValue,
            ...(importErrorNotice.values || {})
          })
        })
        return
      }
      notification.error({
        message: t("managePrompts.notification.error"),
        description: t("managePrompts.notification.someError")
      })
    }
  }, [confirmDanger, data, expectedOwnerId, guardPrivateMode, importMode, queryClient, t])

  return {
    fileInputRef,
    importMode,
    setImportMode,
    exportFormat,
    setExportFormat,
    triggerExport,
    handleImportFile
  }
}
