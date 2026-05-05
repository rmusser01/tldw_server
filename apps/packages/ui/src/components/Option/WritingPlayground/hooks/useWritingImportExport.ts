/**
 * Hook: useWritingImportExport
 *
 * Manages import/export of writing sessions and snapshots, including
 * file handling, session import, snapshot import/export.
 */

import React from "react"
import { useQueryClient } from "@tanstack/react-query"
import { Modal } from "antd"
import type { TFunction } from "i18next"
import {
  createWritingSession,
  exportWritingSnapshot,
  getWritingSession,
  importWritingSnapshot,
  type WritingSessionListItem
} from "@/services/writing-playground"
import {
  extractImportedSessionItems,
  getImportedSessionModelHint,
  getImportedSessionProviderHint,
  parseImportedSessionPayload
} from "../writing-session-import-utils"
import { extractImportedTemplateItems } from "../writing-template-import-utils"
import { extractImportedThemeItems } from "../writing-theme-import-utils"
import {
  getSnapshotImportInvalidationKeys,
  resolveSnapshotImportAction,
  type SnapshotImportMode
} from "../writing-snapshot-import-utils"
import { isRecord } from "./utils"

export interface UseWritingImportExportDeps {
  isOnline: boolean
  hasWriting: boolean
  hasSnapshots: boolean
  sessions: WritingSessionListItem[]
  sessionsLoading: boolean
  activeSessionId: string | null
  setActiveSessionId: (id: string | null) => void
  setActiveSessionName: (name: string | null) => void
  selectedModel: string | undefined
  setSelectedModel: (model: string) => Promise<void> | void
  apiProviderOverride: string | undefined
  setApiProvider: (provider: string) => void
  t: TFunction
}

export function useWritingImportExport(deps: UseWritingImportExportDeps) {
  const {
    isOnline,
    hasWriting,
    hasSnapshots,
    sessions,
    sessionsLoading,
    activeSessionId,
    setActiveSessionId,
    setActiveSessionName,
    selectedModel,
    setSelectedModel,
    apiProviderOverride,
    setApiProvider,
    t
  } = deps

  const queryClient = useQueryClient()

  const [sessionImporting, setSessionImporting] = React.useState(false)
  const [snapshotImporting, setSnapshotImporting] = React.useState(false)
  const [snapshotExporting, setSnapshotExporting] = React.useState(false)
  const [snapshotImportMode, setSnapshotImportMode] =
    React.useState<SnapshotImportMode>("merge")

  const sessionFileInputRef = React.useRef<HTMLInputElement | null>(null)
  const snapshotFileInputRef = React.useRef<HTMLInputElement | null>(null)

  // --- Export session ---
  const exportSession = React.useCallback(
    async (session: WritingSessionListItem) => {
      const detail = await getWritingSession(session.id)
      const payload = {
        name: detail.name,
        payload: detail.payload,
        schema_version: detail.schema_version
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json"
      })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `${detail.name || "session"}.json`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    },
    []
  )

  // --- Export snapshot ---
  const exportSnapshot = React.useCallback(async () => {
    setSnapshotExporting(true)
    try {
      const snapshot = await exportWritingSnapshot()
      const stamp = new Date().toISOString().replace(/[:.]/g, "-")
      const blob = new Blob([JSON.stringify(snapshot, null, 2)], {
        type: "application/json"
      })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = url
      link.download = `writing-snapshot-${stamp}.json`
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } finally {
      setSnapshotExporting(false)
    }
  }, [])

  // --- Snapshot import ---
  const openSnapshotImportPicker = React.useCallback(
    (mode: SnapshotImportMode) => {
      const action = resolveSnapshotImportAction(mode, {
        title: t(
          "option:writingPlayground.snapshotReplaceConfirmTitle",
          "Replace existing writing data?"
        ),
        body: t(
          "option:writingPlayground.snapshotReplaceConfirmBody",
          "This will replace current sessions, templates, and themes with the imported snapshot."
        ),
        action: t(
          "option:writingPlayground.snapshotReplaceConfirmAction",
          "Choose file"
        ),
        cancel: t("common:cancel", "Cancel")
      })

      if (action.type === "confirm-replace") {
        Modal.confirm({
          title: action.title,
          content: action.content,
          okText: action.okText,
          cancelText: action.cancelText,
          okButtonProps: { danger: action.danger },
          onOk: () => {
            setSnapshotImportMode(action.mode)
            snapshotFileInputRef.current?.click()
          }
        })
        return
      }
      setSnapshotImportMode(action.mode)
      snapshotFileInputRef.current?.click()
    },
    [t]
  )

  const handleSnapshotImport = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return
      setSnapshotImporting(true)
      try {
        const importMode = snapshotImportMode
        const raw = await file.text()
        const parsed = JSON.parse(raw)
        const source =
          isRecord(parsed) && isRecord(parsed.snapshot) ? parsed.snapshot : parsed
        if (!isRecord(source)) return

        const importedSessions = extractImportedSessionItems(source)
        const importedTemplates = extractImportedTemplateItems(source)
        const importedThemes = extractImportedThemeItems(source)

        if (
          importedSessions.length === 0 &&
          importedTemplates.length === 0 &&
          importedThemes.length === 0
        ) return

        let importedModelHint: string | null = null
        let importedProviderHint: string | null = null
        const normalizedSessions = importedSessions
          .filter(isRecord)
          .map((item) => {
            const payload = parseImportedSessionPayload(item)
            if (!importedModelHint) {
              importedModelHint = getImportedSessionModelHint(payload)
            }
            if (!importedProviderHint) {
              importedProviderHint = getImportedSessionProviderHint(payload)
            }
            const id =
              typeof item.id === "string" && item.id.trim().length > 0
                ? item.id.trim()
                : undefined
            const nameCandidate =
              typeof item.name === "string"
                ? item.name
                : typeof item.title === "string"
                  ? item.title
                  : ""
            const name = nameCandidate.trim() || `Imported session ${Date.now()}`
            const schemaVersion =
              typeof item.schema_version === "number"
                ? item.schema_version
                : typeof item.schemaVersion === "number"
                  ? item.schemaVersion
                  : 1
            const versionParentId =
              typeof item.version_parent_id === "string"
                ? item.version_parent_id
                : typeof item.versionParentId === "string"
                  ? item.versionParentId
                  : null
            return {
              id,
              name,
              payload,
              schema_version: schemaVersion,
              version_parent_id: versionParentId
            }
          })

        const normalizedTemplates = importedTemplates.map((item) => ({
          name: item.name,
          payload: item.payload,
          schema_version: item.schemaVersion,
          is_default: item.isDefault
        }))
        const normalizedThemes = importedThemes.map((item) => ({
          name: item.name,
          class_name: item.className,
          css: item.css,
          schema_version: item.schemaVersion,
          is_default: item.isDefault,
          order: item.order
        }))

        await importWritingSnapshot({
          mode: importMode,
          snapshot: {
            sessions: normalizedSessions,
            templates: normalizedTemplates,
            themes: normalizedThemes
          }
        })

        if (importMode === "replace") {
          setActiveSessionId(null)
          setActiveSessionName(null)
        }

        if (!selectedModel && importedModelHint) {
          void setSelectedModel(importedModelHint)
        }
        if (!apiProviderOverride && importedProviderHint) {
          setApiProvider(importedProviderHint)
        }

        await Promise.all(
          getSnapshotImportInvalidationKeys(importMode, activeSessionId).map((queryKey) =>
            queryClient.invalidateQueries({ queryKey })
          )
        )
      } finally {
        setSnapshotImporting(false)
        setSnapshotImportMode("merge")
        event.target.value = ""
      }
    },
    [
      apiProviderOverride,
      activeSessionId,
      queryClient,
      selectedModel,
      snapshotImportMode,
      setApiProvider,
      setActiveSessionId,
      setActiveSessionName,
      setSelectedModel
    ]
  )

  // --- Session import ---
  const handleSessionImport = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file) return
      setSessionImporting(true)
      try {
        const raw = await file.text()
        const parsed = JSON.parse(raw)
        const items = extractImportedSessionItems(parsed)
        if (!Array.isArray(items) || items.length === 0) return
        const existingNames = new Set(sessions.map((session) => session.name))
        let importedModelHint: string | null = null
        let importedProviderHint: string | null = null
        const resolveName = (base: string) => {
          if (!existingNames.has(base)) return base
          let idx = 1
          let candidate = `${base} (imported)`
          while (existingNames.has(candidate)) {
            idx += 1
            candidate = `${base} (imported ${idx})`
          }
          return candidate
        }
        for (const item of items) {
          if (!isRecord(item)) continue
          const rawName = String(item.name || item.title || "").trim()
          const name = resolveName(rawName || `Imported session ${Date.now()}`)
          existingNames.add(name)
          const payload = parseImportedSessionPayload(item)
          if (!importedModelHint) {
            importedModelHint = getImportedSessionModelHint(payload)
          }
          if (!importedProviderHint) {
            importedProviderHint = getImportedSessionProviderHint(payload)
          }
          const schemaVersion =
            typeof item.schema_version === "number" ? item.schema_version : 1
          await createWritingSession({
            name,
            payload,
            schema_version: schemaVersion
          })
        }
        if (!selectedModel && importedModelHint) {
          void setSelectedModel(importedModelHint)
        }
        if (!apiProviderOverride && importedProviderHint) {
          setApiProvider(importedProviderHint)
        }
        queryClient.invalidateQueries({ queryKey: ["writing-sessions"] })
      } finally {
        setSessionImporting(false)
        event.target.value = ""
      }
    },
    [
      apiProviderOverride,
      queryClient,
      selectedModel,
      sessions,
      setApiProvider,
      setSelectedModel
    ]
  )

  // --- Derived ---
  const sessionImportDisabled =
    !hasWriting || sessionsLoading || sessionImporting
  const snapshotImportDisabled =
    !hasSnapshots || snapshotImporting || snapshotExporting
  const snapshotExportDisabled =
    !hasSnapshots || snapshotExporting || snapshotImporting

  return {
    // state
    sessionImporting,
    snapshotImporting,
    snapshotExporting,
    snapshotImportMode,
    // refs
    sessionFileInputRef,
    snapshotFileInputRef,
    // callbacks
    exportSession,
    exportSnapshot,
    openSnapshotImportPicker,
    handleSnapshotImport,
    handleSessionImport,
    // derived
    sessionImportDisabled,
    snapshotImportDisabled,
    snapshotExportDisabled
  }
}
