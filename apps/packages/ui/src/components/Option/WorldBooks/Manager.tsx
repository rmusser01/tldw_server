import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Button, Drawer, Form, Input, InputNumber, Modal, Skeleton, Switch, Table, Tooltip, Tag, Select, Descriptions, Empty, Popover, Divider, Checkbox, Grid, Progress, Upload } from "antd"
import React from "react"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { List, Upload as UploadIcon, HelpCircle } from "lucide-react"
import { useServerOnline } from "@/hooks/useServerOnline"
import FeatureEmptyState from "../../Common/FeatureEmptyState"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import {
  WORLD_BOOK_FORM_DEFAULTS,
  buildWorldBookMutationErrorMessage,
  getWorldBookStarterTemplate,
  isWorldBookVersionConflictError,
  toWorldBookFormValues
} from "./worldBookFormUtils"
import {
  getWorldBookImportFormatLabel,
  WORLD_BOOK_IMPORT_MERGE_HELP_TEXT,
} from "./worldBookInteropUtils"
import {
  getBudgetUtilizationBand,
  getBudgetUtilizationColor,
  getBudgetUtilizationPercent,
  getTokenEstimatorNote
} from "./worldBookStatsUtils"
import {
  buildGlobalWorldBookStatistics,
  type GlobalWorldBookStatistics
} from "./worldBookGlobalStatsUtils"
import { useWorldBookFiltering } from "./hooks/useWorldBookFiltering"
import { useWorldBookBulkActions } from "./hooks/useWorldBookBulkActions"
import { useWorldBookImportExport } from "./hooks/useWorldBookImportExport"
import { useWorldBookTestMatching } from "./hooks/useWorldBookTestMatching"
import { WorldBookForm } from "./WorldBookForm"
import { WorldBookTestMatchingModal } from "./WorldBookTestMatchingModal"
import {
  WorldBookEntryManager as EntryManager,
  type EntryFilterPreset,
  DEFAULT_ENTRY_FILTER_PRESET
} from "./WorldBookEntryManager"
import {
  ATTACHMENT_MATRIX_CHARACTER_THRESHOLD,
  ATTACHMENT_LIST_PAGE_SIZE,
  ATTACHMENT_FEEDBACK_DURATION_MS,
  ATTACHMENT_PULSE_DURATION_MS,
  MODAL_BODY_SCROLL_STYLE,
  ACCESSIBLE_SWITCH_TEXT_PROPS,
  type EditWorldBookConflictState,
} from "./worldBookManagerUtils"
import { WorldBookEmptyState } from "./WorldBookEmptyState"
import { WorldBookToolbar } from "./WorldBookToolbar"
import { WorldBookListPanel } from "./WorldBookListPanel"
import { WorldBookDetailPanel, type WorldBookDetailTabKey } from "./WorldBookDetailPanel"

export { WorldBookForm } from "./WorldBookForm"

export const WorldBooksManager: React.FC = () => {
  const isOnline = useServerOnline()
  const { t } = useTranslation(["option"])
  const screens = Grid.useBreakpoint()
  const qc = useQueryClient()
  const notification = useAntdNotification()
  const { showUndoNotification } = useUndoNotification()
  const [open, setOpen] = React.useState(false)
  const [selectedWorldBookId, setSelectedWorldBookId] = React.useState<number | null>(null)
  const [openEdit, setOpenEdit] = React.useState(false)
  const [openEntries, setOpenEntries] = React.useState<
    null | { id: number; name: string; entryCount?: number; tokenBudget?: number }
  >(null)
  const [openAttach, setOpenAttach] = React.useState<null | number>(null)
  const [editId, setEditId] = React.useState<number | null>(null)
  const [editExpectedVersion, setEditExpectedVersion] = React.useState<number | null>(null)
  const [editConflict, setEditConflict] = React.useState<EditWorldBookConflictState | null>(null)
  const [openMatrix, setOpenMatrix] = React.useState(false)
  const [attachmentsHydrationRequested, setAttachmentsHydrationRequested] = React.useState(false)
  const [openGlobalStats, setOpenGlobalStats] = React.useState(false)
  const [statsFor, setStatsFor] = React.useState<any | null>(null)
  const [statsLoadingId, setStatsLoadingId] = React.useState<number | null>(null)
  const [createForm] = Form.useForm()
  const [editForm] = Form.useForm()
  const [entryForm] = Form.useForm()
  const [settingsForm] = Form.useForm()
  const [attachForm] = Form.useForm()
  const [detailActiveTab, setDetailActiveTab] = React.useState<WorldBookDetailTabKey>("entries")
  const importFormatHelpContentId = React.useId()
  const importErrorDetailsContentId = React.useId()
  const importPreviewEntriesContentId = React.useId()
  const confirmDanger = useConfirmDanger()
  const entriesFocusReturnRef = React.useRef<HTMLElement | null>(null)
  const matrixFocusReturnRef = React.useRef<HTMLElement | null>(null)
  const matrixBaselineKeysRef = React.useRef<Set<string>>(new Set())
  const matrixPulseTimersRef = React.useRef<Record<string, any>>({})
  const matrixFeedbackTimerRef = React.useRef<any>(null)
  const [matrixPending, setMatrixPending] = React.useState<Record<string, boolean>>({})
  const [matrixSessionDeltas, setMatrixSessionDeltas] = React.useState<
    Record<string, "attached" | "detached">
  >({})
  const [matrixSuccessPulse, setMatrixSuccessPulse] = React.useState<Record<string, boolean>>({})
  const [matrixMetaPopoverOpenKey, setMatrixMetaPopoverOpenKey] = React.useState<string | null>(null)
  const [matrixMetaDrafts, setMatrixMetaDrafts] = React.useState<
    Record<string, { enabled: boolean; priority: number }>
  >({})
  const [matrixFeedback, setMatrixFeedback] = React.useState<{
    kind: "success" | "error"
    message: string
  } | null>(null)
  const [matrixBookFilter, setMatrixBookFilter] = React.useState('')
  const [matrixCharacterFilter, setMatrixCharacterFilter] = React.useState('')
  const [matrixListPage, setMatrixListPage] = React.useState(1)
  const [entryFilterPreset, setEntryFilterPreset] = React.useState<EntryFilterPreset>(
    DEFAULT_ENTRY_FILTER_PRESET
  )

  const getActiveFocusableElement = React.useCallback((): HTMLElement | null => {
    if (typeof document === "undefined") return null
    const activeElement = document.activeElement
    return activeElement instanceof HTMLElement ? activeElement : null
  }, [])

  const restoreFocusToElement = React.useCallback((target: HTMLElement | null) => {
    if (!target) return
    window.setTimeout(() => {
      if (!target.isConnected) return
      if (target instanceof HTMLButtonElement && target.disabled) return
      if (target instanceof HTMLInputElement && target.disabled) return
      target.focus()
    }, 0)
  }, [])

  const { data, status } = useQuery({
    queryKey: ['tldw:listWorldBooks'],
    queryFn: async () => {
      await tldwClient.initialize()
      const res = await tldwClient.listWorldBooks(false)
      return res?.world_books || []
    },
    enabled: isOnline
  })

  const { data: worldBookRuntimeConfig } = useQuery({
    queryKey: ["tldw:worldBookRuntimeConfig"],
    queryFn: async () => {
      await tldwClient.initialize()
      return await tldwClient.getWorldBookRuntimeConfig()
    },
    enabled: isOnline
  })

  const { data: characters } = useQuery({
    queryKey: ['tldw:listCharactersForWB'],
    queryFn: async () => {
      await tldwClient.initialize()
      return await tldwClient.listCharacters()
    },
    enabled: isOnline
  })

  const { data: attachmentsByBook, isLoading: attachmentsLoading } = useQuery({
    queryKey: ['tldw:worldBookAttachments', (characters || []).map((c: any) => c.id).join(',')],
    queryFn: async () => {
      if (!characters || characters.length === 0) return {}
      await tldwClient.initialize()
      const results: Array<{ character: any; books: any[] }> = []
      for (const character of characters || []) {
        try {
          const books = await tldwClient.listCharacterWorldBooks(character.id)
          results.push({ character, books: books || [] })
        } catch {
          results.push({ character, books: [] })
        }
      }
      const map: Record<number, any[]> = {}
      results.forEach(({ character, books }) => {
        (books || []).forEach((b: any) => {
          const wid = b.world_book_id ?? b.id
          if (!map[wid]) map[wid] = []
          map[wid].push({
            id: character.id,
            name: character.name,
            attachment_enabled: b.attachment_enabled,
            attachment_priority: b.attachment_priority
          })
        })
      })
      return map
    },
    enabled: isOnline && attachmentsHydrationRequested && !!characters && characters.length > 0
  })

  const requestAttachmentHydration = React.useCallback(() => {
    setAttachmentsHydrationRequested(true)
  }, [])

  const maxRecursiveDepth =
    typeof worldBookRuntimeConfig?.max_recursive_depth === "number" &&
    Number.isFinite(worldBookRuntimeConfig.max_recursive_depth)
      ? worldBookRuntimeConfig.max_recursive_depth
      : 10

  const activeCharacterIds = React.useMemo(() => {
    const ids = new Set<number>()
    ;(characters || []).forEach((character: any) => {
      const id = Number(character?.id)
      if (Number.isFinite(id) && id > 0) ids.add(id)
    })
    return ids
  }, [characters])

  const reconciledAttachmentsByBook = React.useMemo(() => {
    const source = attachmentsByBook as Record<number, any[]> | undefined
    if (!source || typeof source !== "object") return {}

    const next: Record<number, any[]> = {}
    Object.entries(source).forEach(([worldBookIdRaw, attachedCharacters]) => {
      const worldBookId = Number(worldBookIdRaw)
      if (!Number.isFinite(worldBookId) || worldBookId <= 0) return
      const list = Array.isArray(attachedCharacters) ? attachedCharacters : []
      const filtered = list.filter((character: any) => {
        const characterId = Number(character?.id)
        if (!Number.isFinite(characterId) || characterId <= 0) return false
        if (activeCharacterIds.size === 0) return true
        return activeCharacterIds.has(characterId)
      })
      if (filtered.length > 0) next[worldBookId] = filtered
    })
    return next
  }, [activeCharacterIds, attachmentsByBook])

  const globalStatsQuerySignature = React.useMemo(
    () =>
      ((data || []) as any[])
        .map((book: any) => `${book?.id}:${book?.last_modified || ""}`)
        .join("|"),
    [data]
  )

  const {
    data: globalStats,
    status: globalStatsStatus,
    isFetching: globalStatsFetching
  } = useQuery<GlobalWorldBookStatistics>({
    queryKey: ["tldw:worldBookGlobalStatistics", globalStatsQuerySignature],
    queryFn: async () => {
      await tldwClient.initialize()
      const books = Array.isArray(data) ? (data as any[]) : []
      const entriesByBook: Record<number, unknown> = {}
      await Promise.all(
        books.map(async (book: any) => {
          const worldBookId = Number(book?.id)
          if (!Number.isFinite(worldBookId) || worldBookId <= 0) return
          const response = await tldwClient.listWorldBookEntries(worldBookId, false)
          entriesByBook[worldBookId] = Array.isArray(response?.entries) ? response.entries : []
        })
      )
      return buildGlobalWorldBookStatistics(books, entriesByBook)
    },
    enabled: isOnline && openGlobalStats && Array.isArray(data) && data.length > 0
  })

  const quickAttachWorldBookName = React.useMemo(() => {
    if (!openAttach) return null
    const match = ((data || []) as any[]).find(
      (book: any) => Number(book?.id) === Number(openAttach)
    )
    return match?.name || null
  }, [data, openAttach])

  const getAttachedCharacters = React.useCallback(
    (worldBookId: number) => reconciledAttachmentsByBook[worldBookId] || [],
    [reconciledAttachmentsByBook]
  )

  const selectedWorldBookRecord = React.useMemo(
    () => ((data || []) as any[]).find((book: any) => Number(book?.id) === Number(selectedWorldBookId)) || null,
    [data, selectedWorldBookId]
  )
  const selectedWorldBookVersion =
    typeof selectedWorldBookRecord?.version === "number" ? selectedWorldBookRecord.version : null

  const selectedWorldBookAttached = React.useMemo(
    () => selectedWorldBookId ? getAttachedCharacters(selectedWorldBookId) : [],
    [selectedWorldBookId, getAttachedCharacters]
  )

  const {
    data: selectedWorldBookStats,
    status: selectedWorldBookStatsStatus,
    error: selectedWorldBookStatsError,
    isFetching: selectedWorldBookStatsFetching
  } = useQuery({
    queryKey: ["tldw:selectedWorldBookStatistics", selectedWorldBookId],
    queryFn: async () => {
      if (selectedWorldBookId == null) return null
      await tldwClient.initialize()
      return await tldwClient.worldBookStatistics(selectedWorldBookId)
    },
    enabled: isOnline && selectedWorldBookId != null && detailActiveTab === "stats"
  })

  const selectWorldBookInDetail = React.useCallback(
    (id: number, tab: WorldBookDetailTabKey = "entries") => {
      setSelectedWorldBookId(id)
      setDetailActiveTab(tab)
      requestAttachmentHydration()
    },
    [requestAttachmentHydration]
  )

  const { mutate: createWB, isPending: creating } = useMutation({
    mutationFn: async (values: any) => {
      const templateKey =
        typeof values?.template_key === "string" ? values.template_key : undefined
      const payload = { ...(values || {}) }
      delete payload.template_key

      const created = await tldwClient.createWorldBook(payload)
      const template = getWorldBookStarterTemplate(templateKey)
      const createdId = Number(created?.id)

      if (template && Number.isFinite(createdId) && createdId > 0) {
        for (const entry of template.entries) {
          await tldwClient.addWorldBookEntry(createdId, {
            keywords: entry.keywords,
            content: entry.content,
            priority: typeof entry.priority === "number" ? entry.priority : 0,
            enabled: typeof entry.enabled === "boolean" ? entry.enabled : true,
            case_sensitive: !!entry.case_sensitive,
            regex_match: !!entry.regex_match,
            whole_word_match:
              typeof entry.whole_word_match === "boolean"
                ? entry.whole_word_match
                : true,
            appendable: !!entry.appendable
          })
        }
      }

      return created
    },
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tldw:listWorldBooks'] }); setOpen(false); createForm.resetFields() },
    onError: (e: any, values: any) =>
      notification.error({
        message: "Error",
        description: buildWorldBookMutationErrorMessage(e, {
          attemptedName: values?.name,
          fallback: "Failed to create world book"
        })
      })
  })
  const { mutate: updateWB, isPending: updating } = useMutation({
    mutationFn: (values: any) => {
      const targetId = editId ?? selectedWorldBookId
      if (targetId == null) return Promise.resolve(null)
      const versionToUse =
        typeof editExpectedVersion === "number"
          ? editExpectedVersion
          : editId == null
            ? selectedWorldBookVersion
            : null
      if (typeof versionToUse === "number") {
        return tldwClient.updateWorldBook(targetId, values, {
          expectedVersion: versionToUse
        })
      }
      return tldwClient.updateWorldBook(targetId, values)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tldw:listWorldBooks'] })
      setEditConflict(null)
      if (editId != null) {
        setOpenEdit(false)
        editForm.resetFields()
        setEditId(null)
        setEditExpectedVersion(null)
        setEditConflict(null)
      }
    },
    onError: (e: any, values: any) => {
      const description = buildWorldBookMutationErrorMessage(e, {
        attemptedName: values?.name,
        fallback: "Failed to update world book"
      })
      notification.error({
        message: "Error",
        description
      })
      if (isWorldBookVersionConflictError(e)) {
        setDetailActiveTab("settings")
        setEditConflict({
          attemptedValues: { ...(values || {}) },
          message: description
        })
        void qc.invalidateQueries({ queryKey: ["tldw:listWorldBooks"] })
      }
    }
  })
  const { mutate: deleteWB, isPending: deleting } = useMutation({
    mutationFn: (id: number) => tldwClient.deleteWorldBook(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['tldw:listWorldBooks'] }) },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Failed to delete world book' })
  })
  const { mutate: doImport, isPending: importing } = useMutation({
    mutationFn: (payload: any) => tldwClient.importWorldBook(payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tldw:listWorldBooks'] })
      resetImportAfterSuccess()
    },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Failed to import world book' })
  })

  const { mutateAsync: attachWB, isPending: attaching } = useMutation({
    mutationFn: ({
      characterId,
      worldBookId,
      enabled,
      priority
    }: {
      characterId: number
      worldBookId: number
      enabled?: boolean
      priority?: number
    }) => {
      const hasEnabled = typeof enabled === "boolean"
      const hasPriority = typeof priority === "number" && Number.isFinite(priority)
      if (hasEnabled || hasPriority) {
        return tldwClient.attachWorldBookToCharacter(characterId, worldBookId, {
          ...(hasEnabled ? { enabled } : {}),
          ...(hasPriority ? { priority } : {})
        })
      }
      return tldwClient.attachWorldBookToCharacter(characterId, worldBookId)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tldw:worldBookAttachments'] })
    },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Failed to attach world book' })
  })

  const { mutateAsync: detachWB } = useMutation({
    mutationFn: ({ characterId, worldBookId }: { characterId: number; worldBookId: number }) =>
      tldwClient.detachWorldBookFromCharacter(characterId, worldBookId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tldw:worldBookAttachments'] })
    },
    onError: (e: any) => notification.error({ message: 'Error', description: e?.message || 'Failed to detach world book' })
  })

  const [detachingFor, setDetachingFor] = React.useState<{ characterId: number; worldBookId: number } | null>(null)

  const attachmentKeyFor = React.useCallback((worldBookId: number, characterId: number) => {
    return `${worldBookId}:${characterId}`
  }, [])

  const applyMatrixFeedback = React.useCallback((kind: "success" | "error", message: string) => {
    if (matrixFeedbackTimerRef.current) {
      clearTimeout(matrixFeedbackTimerRef.current)
    }
    setMatrixFeedback({ kind, message })
    matrixFeedbackTimerRef.current = setTimeout(() => {
      setMatrixFeedback((current) => (current?.message === message ? null : current))
      matrixFeedbackTimerRef.current = null
    }, ATTACHMENT_FEEDBACK_DURATION_MS)
  }, [])

  const initializeMatrixSession = React.useCallback(() => {
    const baseline = new Set<string>()
    const source = reconciledAttachmentsByBook as Record<string, any> | undefined
    if (source && typeof source === "object") {
      Object.entries(source).forEach(([worldBookIdRaw, attachedCharacters]) => {
        const worldBookId = Number(worldBookIdRaw)
        if (!Number.isFinite(worldBookId) || worldBookId <= 0) return
        const list = Array.isArray(attachedCharacters) ? attachedCharacters : []
        list.forEach((character: any) => {
          const characterId = Number(character?.id)
          if (!Number.isFinite(characterId) || characterId <= 0) return
          baseline.add(attachmentKeyFor(worldBookId, characterId))
        })
      })
    }

    matrixBaselineKeysRef.current = baseline
    Object.values(matrixPulseTimersRef.current).forEach((timerId) => clearTimeout(timerId))
    matrixPulseTimersRef.current = {}
    if (matrixFeedbackTimerRef.current) {
      clearTimeout(matrixFeedbackTimerRef.current)
      matrixFeedbackTimerRef.current = null
    }
    setMatrixSessionDeltas({})
    setMatrixSuccessPulse({})
    setMatrixMetaDrafts({})
    setMatrixMetaPopoverOpenKey(null)
    setMatrixFeedback(null)
  }, [attachmentKeyFor, reconciledAttachmentsByBook])

  const handleOpenMatrix = React.useCallback(() => {
    matrixFocusReturnRef.current = getActiveFocusableElement()
    requestAttachmentHydration()
    initializeMatrixSession()
    setOpenMatrix(true)
  }, [getActiveFocusableElement, initializeMatrixSession, requestAttachmentHydration])

  const handleCloseMatrix = React.useCallback(() => {
    const focusTarget = matrixFocusReturnRef.current
    matrixFocusReturnRef.current = null
    setOpenMatrix(false)
    initializeMatrixSession()
    restoreFocusToElement(focusTarget)
  }, [initializeMatrixSession, restoreFocusToElement])

  const openFullMatrixFromQuickAttach = React.useCallback(() => {
    setOpenAttach(null)
    handleOpenMatrix()
  }, [handleOpenMatrix])

  const isAttached = React.useCallback((worldBookId: number, characterId: number) => {
    const attached = getAttachedCharacters(worldBookId)
    return attached.some((c: any) => c.id === characterId)
  }, [getAttachedCharacters])

  const getAttachmentMetadata = React.useCallback(
    (worldBookId: number, characterId: number) => {
      const attached = getAttachedCharacters(worldBookId).find(
        (character: any) => Number(character?.id) === characterId
      )
      return {
        enabled:
          typeof attached?.attachment_enabled === "boolean"
            ? attached.attachment_enabled
            : true,
        priority:
          typeof attached?.attachment_priority === "number" &&
          Number.isFinite(attached.attachment_priority)
            ? attached.attachment_priority
            : 0
      }
    },
    [getAttachedCharacters]
  )

  const handleMatrixToggle = async (worldBookId: number, characterId: number, next: boolean) => {
    const key = attachmentKeyFor(worldBookId, characterId)
    if (matrixPending[key]) return
    setMatrixPending((prev) => ({ ...prev, [key]: true }))
    try {
      if (next) {
        await attachWB({ characterId, worldBookId })
      } else {
        await detachWB({ characterId, worldBookId })
      }

      const baselineHadAttachment = matrixBaselineKeysRef.current.has(key)
      if (baselineHadAttachment === next) {
        setMatrixSessionDeltas((prev) => {
          const copy = { ...prev }
          delete copy[key]
          return copy
        })
      } else {
        setMatrixSessionDeltas((prev) => ({
          ...prev,
          [key]: next ? "attached" : "detached"
        }))
      }

      const worldBookName =
        ((data || []) as any[]).find((book: any) => Number(book?.id) === worldBookId)?.name ||
        `World Book ${worldBookId}`
      const characterName =
        ((characters || []) as any[]).find((character: any) => Number(character?.id) === characterId)
          ?.name || `Character ${characterId}`
      applyMatrixFeedback(
        "success",
        `${next ? "Attached" : "Detached"} ${characterName} ${next ? "to" : "from"} ${worldBookName}.`
      )

      if (matrixPulseTimersRef.current[key]) {
        clearTimeout(matrixPulseTimersRef.current[key])
      }
      setMatrixSuccessPulse((prev) => ({ ...prev, [key]: true }))
      matrixPulseTimersRef.current[key] = setTimeout(() => {
        setMatrixSuccessPulse((prev) => {
          const copy = { ...prev }
          delete copy[key]
          return copy
        })
        delete matrixPulseTimersRef.current[key]
      }, ATTACHMENT_PULSE_DURATION_MS)
    } catch (error: any) {
      const worldBookName =
        ((data || []) as any[]).find((book: any) => Number(book?.id) === worldBookId)?.name ||
        `World Book ${worldBookId}`
      const characterName =
        ((characters || []) as any[]).find((character: any) => Number(character?.id) === characterId)
          ?.name || `Character ${characterId}`
      const errorDetails = String(error?.message || "Unknown error")
      applyMatrixFeedback(
        "error",
        `Could not ${next ? "attach" : "detach"} ${characterName} ${next ? "to" : "from"} ${worldBookName}. Changes were reverted. ${errorDetails}`
      )
    } finally {
      setMatrixPending((prev) => {
        const copy = { ...prev }
        delete copy[key]
        return copy
      })
    }
  }

  const openMatrixMetadataEditor = React.useCallback(
    (worldBookId: number, characterId: number) => {
      const key = attachmentKeyFor(worldBookId, characterId)
      const defaults = getAttachmentMetadata(worldBookId, characterId)
      setMatrixMetaDrafts((prev) => ({
        ...prev,
        [key]:
          prev[key] || {
            enabled: defaults.enabled,
            priority: defaults.priority
          }
      }))
      setMatrixMetaPopoverOpenKey(key)
    },
    [attachmentKeyFor, getAttachmentMetadata]
  )

  const updateMatrixMetadataDraft = React.useCallback(
    (
      worldBookId: number,
      characterId: number,
      patch: Partial<{ enabled: boolean; priority: number }>
    ) => {
      const key = attachmentKeyFor(worldBookId, characterId)
      setMatrixMetaDrafts((prev) => {
        const baseline = prev[key] || getAttachmentMetadata(worldBookId, characterId)
        return {
          ...prev,
          [key]: {
            enabled:
              typeof patch.enabled === "boolean" ? patch.enabled : baseline.enabled,
            priority:
              typeof patch.priority === "number" && Number.isFinite(patch.priority)
                ? patch.priority
                : baseline.priority
          }
        }
      })
    },
    [attachmentKeyFor, getAttachmentMetadata]
  )

  const saveMatrixMetadata = React.useCallback(
    async (worldBookId: number, characterId: number) => {
      const key = attachmentKeyFor(worldBookId, characterId)
      if (matrixPending[key]) return

      const draft = matrixMetaDrafts[key] || getAttachmentMetadata(worldBookId, characterId)
      const nextPriority = Number.isFinite(draft.priority) ? draft.priority : 0
      setMatrixPending((prev) => ({ ...prev, [key]: true }))
      try {
        await attachWB({
          characterId,
          worldBookId,
          enabled: draft.enabled,
          priority: nextPriority
        })

        const worldBookName =
          ((data || []) as any[]).find((book: any) => Number(book?.id) === worldBookId)?.name ||
          `World Book ${worldBookId}`
        const characterName =
          ((characters || []) as any[]).find(
            (character: any) => Number(character?.id) === characterId
          )?.name || `Character ${characterId}`
        applyMatrixFeedback(
          "success",
          `Updated attachment settings for ${characterName} in ${worldBookName}.`
        )
        setMatrixMetaPopoverOpenKey(null)
      } catch (error: any) {
        const worldBookName =
          ((data || []) as any[]).find((book: any) => Number(book?.id) === worldBookId)?.name ||
          `World Book ${worldBookId}`
        const characterName =
          ((characters || []) as any[]).find(
            (character: any) => Number(character?.id) === characterId
          )?.name || `Character ${characterId}`
        const errorDetails = String(error?.message || "Unknown error")
        applyMatrixFeedback(
          "error",
          `Could not update attachment settings for ${characterName} in ${worldBookName}. ${errorDetails}`
        )
      } finally {
        setMatrixPending((prev) => {
          const copy = { ...prev }
          delete copy[key]
          return copy
        })
      }
    },
    [
      attachmentKeyFor,
      attachWB,
      characters,
      data,
      getAttachmentMetadata,
      matrixMetaDrafts,
      matrixPending,
      applyMatrixFeedback
    ]
  )

  const normalizeCharacterIds = React.useCallback((values: Array<number | string>) => {
    return Array.from(
      new Set(
        values
          .map((value) => Number(value))
          .filter((id) => Number.isFinite(id) && id > 0)
      )
    )
  }, [])

  const filteredBooks = React.useMemo(() => {
    const q = matrixBookFilter.trim().toLowerCase()
    if (!q) return data || []
    return (data || []).filter((b: any) => (b.name || '').toLowerCase().includes(q))
  }, [data, matrixBookFilter])

  const filteredCharacters = React.useMemo(() => {
    const q = matrixCharacterFilter.trim().toLowerCase()
    if (!q) return characters || []
    return (characters || []).filter((c: any) => (c.name || '').toLowerCase().includes(q))
  }, [characters, matrixCharacterFilter])

  const useAttachmentListView = !screens.md || filteredCharacters.length > ATTACHMENT_MATRIX_CHARACTER_THRESHOLD

  const focusMatrixCheckboxAt = React.useCallback((rowIndex: number, colIndex: number) => {
    if (typeof document === "undefined") return false
    const selector =
      `input[data-matrix-checkbox="true"]` +
      `[data-matrix-row-index="${rowIndex}"]` +
      `[data-matrix-col-index="${colIndex}"]`
    const target = document.querySelector<HTMLInputElement>(selector)
    if (!target) return false
    target.focus()
    return true
  }, [])

  const handleMatrixCellKeyDown = React.useCallback(
    (
      event: React.KeyboardEvent<HTMLElement>,
      rowIndex: number,
      colIndex: number
    ) => {
      if (useAttachmentListView) return

      const maxRow = filteredBooks.length - 1
      const maxCol = filteredCharacters.length - 1
      if (maxRow < 0 || maxCol < 0) return

      let nextRow = rowIndex
      let nextCol = colIndex

      switch (event.key) {
        case "ArrowLeft":
          nextCol = Math.max(0, colIndex - 1)
          break
        case "ArrowRight":
          nextCol = Math.min(maxCol, colIndex + 1)
          break
        case "ArrowUp":
          nextRow = Math.max(0, rowIndex - 1)
          break
        case "ArrowDown":
          nextRow = Math.min(maxRow, rowIndex + 1)
          break
        case "Home":
          nextCol = 0
          break
        case "End":
          nextCol = maxCol
          break
        default:
          return
      }

      if (nextRow === rowIndex && nextCol === colIndex) return
      event.preventDefault()
      void focusMatrixCheckboxAt(nextRow, nextCol)
    },
    [filteredBooks.length, filteredCharacters.length, focusMatrixCheckboxAt, useAttachmentListView]
  )

  React.useEffect(() => {
    setMatrixListPage(1)
  }, [matrixBookFilter, matrixCharacterFilter, useAttachmentListView])

  const handleListAttachmentChange = React.useCallback(
    async (worldBookId: number, nextValues: Array<number | string>) => {
      const nextIds = normalizeCharacterIds(nextValues)
      const currentIds = normalizeCharacterIds(
        getAttachedCharacters(worldBookId).map((character: any) => character?.id)
      )
      const nextSet = new Set(nextIds)
      const currentSet = new Set(currentIds)

      const attachIds = nextIds.filter((id) => !currentSet.has(id))
      const detachIds = currentIds.filter((id) => !nextSet.has(id))
      if (attachIds.length === 0 && detachIds.length === 0) return

      await Promise.all([
        ...attachIds.map((id) => handleMatrixToggle(worldBookId, id, true)),
        ...detachIds.map((id) => handleMatrixToggle(worldBookId, id, false))
      ])
    },
    [getAttachedCharacters, handleMatrixToggle, normalizeCharacterIds]
  )

  // --- Hooks ---

  const filtering = useWorldBookFiltering({
    data,
    reconciledAttachmentsByBook,
    attachmentsLoading,
  })
  const {
    listSearch, setListSearch,
    enabledFilter, setEnabledFilter,
    attachmentFilter, setAttachmentFilter,
    selectedWorldBookKeys, setSelectedWorldBookKeys,
    tableSort,
    filteredWorldBooks,
    hasActiveListFilters,
    clearListFilters,
    handleTableSortChange,
  } = filtering

  const bulkActions = useWorldBookBulkActions({
    data,
    qc,
    notification,
    confirmDanger,
    showUndoNotification,
    deleteWB,
    deleting,
    attachmentsLoading,
    getAttachedCharacters,
    selectedWorldBookKeys,
    setSelectedWorldBookKeys,
  })
  const {
    bulkWorldBookAction,
    pendingDeleteIds,
    cancelPendingWorldBookDeletes,
    requestDeleteWorldBook,
    handleBulkWorldBookAction,
  } = bulkActions

  const importExport = useWorldBookImportExport({
    data,
    qc,
    notification,
    selectedWorldBookKeys,
  })
  const {
    openImport,
    importFormatHelpOpen, setImportFormatHelpOpen,
    importErrorDetailsOpen, setImportErrorDetailsOpen,
    importPreviewEntriesOpen, setImportPreviewEntriesOpen,
    mergeOnConflict, setMergeOnConflict,
    importPreview,
    importPayload,
    importError,
    importErrorDetails,
    importFileName,
    exportingId,
    bulkExportMode,
    duplicatingId,
    exportSingleWorldBook,
    exportWorldBookBundle,
    duplicateWorldBook,
    handleImportUpload,
    openImportModal,
    closeImportModal,
    resetImportAfterSuccess,
  } = importExport

  const testMatching = useWorldBookTestMatching()
  const {
    openTestMatching,
    testMatchingWorldBookId,
    openTestMatchingModal,
    closeTestMatchingModal,
  } = testMatching

  // --- End Hooks ---

  // Escape key clears selection when no modal/drawer is open
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.key === "Escape" &&
        selectedWorldBookId != null &&
        !open &&
        !openEdit &&
        !openImport &&
        !openMatrix &&
        !openGlobalStats &&
        !openTestMatching &&
        !openAttach &&
        !openEntries
      ) {
        setSelectedWorldBookId(null)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [selectedWorldBookId, open, openEdit, openImport, openMatrix, openGlobalStats, openTestMatching, openAttach, openEntries])

  // Reduced motion preference for animation fallbacks
  const prefersReducedMotion = React.useMemo(() => {
    if (typeof window === "undefined") return false
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches
  }, [])

  const handleCloseCreate = async () => {
    if (createForm.isFieldsTouched()) {
      const ok = await confirmDanger({
        title: "Discard changes?",
        content: "You have unsaved changes. Are you sure you want to close?",
        okText: "Discard",
        cancelText: "Keep editing"
      })
      if (!ok) return
    }
    setOpen(false)
    createForm.resetFields()
  }

  const handleCloseEdit = async () => {
    if (editForm.isFieldsTouched()) {
      const ok = await confirmDanger({
        title: "Discard changes?",
        content: "You have unsaved changes. Are you sure you want to close?",
        okText: "Discard",
        cancelText: "Keep editing"
      })
      if (!ok) return
    }
    setOpenEdit(false)
    editForm.resetFields()
    setEditId(null)
    setEditExpectedVersion(null)
    setEditConflict(null)
  }

  const handleCloseEntries = async () => {
    if (entryForm.isFieldsTouched()) {
      const ok = await confirmDanger({
        title: "Discard changes?",
        content: "You have unsaved changes in the entry form. Are you sure you want to close?",
        okText: "Discard",
        cancelText: "Keep editing"
      })
      if (!ok) return
    }
    const focusTarget = entriesFocusReturnRef.current
    entriesFocusReturnRef.current = null
    setOpenEntries(null)
    entryForm.resetFields()
    restoreFocusToElement(focusTarget)
  }

  const openEntriesWithPreset = React.useCallback(
    (
      book: { id: number; name: string; entryCount?: number; tokenBudget?: number },
      preset: EntryFilterPreset = DEFAULT_ENTRY_FILTER_PRESET
    ) => {
      entriesFocusReturnRef.current = getActiveFocusableElement()
      setEntryFilterPreset({ ...DEFAULT_ENTRY_FILTER_PRESET, ...(preset || {}) })
      setOpenEntries({
        id: book.id,
        name: book.name,
        entryCount: book.entryCount,
        tokenBudget: book.tokenBudget
      })
    },
    [getActiveFocusableElement]
  )

  const openEntriesFromStats = React.useCallback(
    (preset: EntryFilterPreset) => {
      const worldBookId = Number(statsFor?.world_book_id)
      if (!Number.isFinite(worldBookId) || worldBookId <= 0) return
      const worldBookName = String(statsFor?.name || `World Book ${worldBookId}`)
      const entryCount =
        typeof statsFor?.total_entries === "number" ? statsFor.total_entries : undefined
      const tokenBudget =
        typeof statsFor?.token_budget === "number" ? statsFor.token_budget : undefined
      entriesFocusReturnRef.current = getActiveFocusableElement()
      setEntryFilterPreset({ ...DEFAULT_ENTRY_FILTER_PRESET, ...(preset || {}) })
      setStatsFor(null)
      setOpenEntries({ id: worldBookId, name: worldBookName, entryCount, tokenBudget })
    },
    [getActiveFocusableElement, statsFor]
  )

  const openEntriesFromGlobalStats = React.useCallback(
    (worldBookId: number, keyword?: string) => {
      const source = ((data || []) as any[]).find(
        (book: any) => Number(book?.id) === Number(worldBookId)
      )
      if (!source) return
      entriesFocusReturnRef.current = getActiveFocusableElement()
      setOpenGlobalStats(false)
      setEntryFilterPreset({
        ...DEFAULT_ENTRY_FILTER_PRESET,
        searchText: String(keyword || "").trim()
      })
      setOpenEntries({
        id: Number(source.id),
        name: String(source.name || `World Book ${source.id}`),
        entryCount: typeof source.entry_count === "number" ? source.entry_count : undefined,
        tokenBudget: typeof source.token_budget === "number" ? source.token_budget : undefined
      })
    },
    [data, getActiveFocusableElement]
  )

  const activeEditTargetId = editId ?? selectedWorldBookId

  const latestEditRecord = React.useMemo(
    () =>
      ((data || []) as any[]).find(
        (book: any) => Number(book?.id) === Number(activeEditTargetId)
      ) || null,
    [activeEditTargetId, data]
  )
  const latestEditVersion =
    typeof latestEditRecord?.version === "number" ? latestEditRecord.version : null

  const handleLoadLatestEditValues = React.useCallback(() => {
    if (!latestEditRecord) return
    const targetForm = editId != null ? editForm : settingsForm
    targetForm.setFieldsValue(toWorldBookFormValues(latestEditRecord))
    setEditExpectedVersion(
      typeof latestEditRecord.version === "number" ? latestEditRecord.version : null
    )
    setEditConflict(null)
    notification.info({
      message: "Latest values loaded",
      description: "Review the refreshed values, reapply any local edits, then save again."
    })
  }, [editForm, editId, latestEditRecord, notification, settingsForm])

  const handleReapplyEditDraft = React.useCallback(() => {
    if (!latestEditRecord || !editConflict) return
    const targetForm = editId != null ? editForm : settingsForm
    targetForm.setFieldsValue({
      ...toWorldBookFormValues(latestEditRecord),
      ...editConflict.attemptedValues
    })
    setEditExpectedVersion(
      typeof latestEditRecord.version === "number" ? latestEditRecord.version : null
    )
    setEditConflict(null)
    notification.info({
      message: "Edits reapplied",
      description: "Your unsaved edits were merged onto the latest version. Save to retry."
    })
  }, [editConflict, editForm, editId, latestEditRecord, notification, settingsForm])

  const renderEditConflictBanner = React.useCallback(() => {
    if (!editConflict) return null

    return (
      <div
        data-testid="world-book-edit-conflict"
        className="mb-3 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800"
      >
        <p>{editConflict.message}</p>
        <p className="mt-1 text-xs text-amber-700">
          Reload the latest version and then reapply your edits before saving again.
          {latestEditVersion != null ? ` Latest version: ${latestEditVersion}.` : ""}
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          <Button
            size="small"
            onClick={handleLoadLatestEditValues}
            disabled={!latestEditRecord}
          >
            Load latest values
          </Button>
          <Button
            size="small"
            onClick={handleReapplyEditDraft}
            disabled={!latestEditRecord}
          >
            Reapply my edits
          </Button>
        </div>
      </div>
    )
  }, [
    editConflict,
    handleLoadLatestEditValues,
    handleReapplyEditDraft,
    latestEditRecord,
    latestEditVersion
  ])

  React.useEffect(() => {
    return () => {
      Object.values(matrixPulseTimersRef.current).forEach((t) => clearTimeout(t))
      matrixPulseTimersRef.current = {}
      if (matrixFeedbackTimerRef.current) {
        clearTimeout(matrixFeedbackTimerRef.current)
        matrixFeedbackTimerRef.current = null
      }
    }
  }, [])

  const openEditWorldBook = (record: any) => {
    setEditId(record.id)
    setEditExpectedVersion(
      typeof record?.version === "number" ? record.version : null
    )
    setEditConflict(null)
    editForm.setFieldsValue(toWorldBookFormValues(record))
    setOpenEdit(true)
  }

  const openWorldBookStatistics = async (record: any) => {
    setStatsLoadingId(record.id)
    try {
      const stats = await tldwClient.worldBookStatistics(record.id)
      setStatsFor(stats)
    } catch (error: any) {
      notification.error({ message: "Stats failed", description: error?.message })
    } finally {
      setStatsLoadingId(null)
    }
  }

  const handleRowAction = React.useCallback((action: string, record: any) => {
    switch (action) {
      case "entries":
        selectWorldBookInDetail(record.id, "entries")
        break
      case "duplicate":
        void duplicateWorldBook(record)
        break
      case "attach":
        requestAttachmentHydration()
        setOpenAttach(record.id)
        break
      case "export":
        void exportSingleWorldBook(record)
        break
      case "stats":
        void openWorldBookStatistics(record)
        break
      case "delete":
        void requestDeleteWorldBook(record)
        break
    }
  }, [duplicateWorldBook, exportSingleWorldBook, requestDeleteWorldBook, openWorldBookStatistics, selectWorldBookInDetail])

  const layoutMode: "desktop" | "tablet" | "mobile" = screens.lg
    ? "desktop"
    : screens.md
      ? "tablet"
      : "mobile"

  if (!isOnline) {
    return (
      <FeatureEmptyState
        title={t("option:worldBooksEmpty.offlineTitle", {
          defaultValue: "World Books are offline"
        })}
        description={t("option:worldBooksEmpty.offlineDescription", {
          defaultValue:
            "Connect to your tldw server from the main settings page to view and edit World Books."
        })}
      />
    )
  }

  return (
    <div className="space-y-4" data-testid="world-books-manager">
      <WorldBookToolbar
        listSearch={listSearch}
        onSearchChange={setListSearch}
        enabledFilter={enabledFilter}
        onEnabledFilterChange={setEnabledFilter}
        attachmentFilter={attachmentFilter}
        onAttachmentFilterChange={(value) => {
          if (value !== "all") requestAttachmentHydration()
          setAttachmentFilter(value)
        }}
        onNewWorldBook={() => setOpen(true)}
        onOpenTestMatching={() => openTestMatchingModal()}
        onOpenMatrix={handleOpenMatrix}
        onOpenGlobalStats={() => setOpenGlobalStats(true)}
        onImport={openImportModal}
        onExportAll={() => void exportWorldBookBundle("all")}
        onExportSelected={selectedWorldBookKeys.length > 0 ? () => void exportWorldBookBundle("selected") : undefined}
        hasWorldBooks={Array.isArray(data) && data.length > 0}
        hasSelection={selectedWorldBookKeys.length > 0}
        globalStatsFetching={openGlobalStats && globalStatsFetching}
        bulkExportAllLoading={bulkExportMode === "all"}
        bulkExportSelectedLoading={bulkExportMode === "selected"}
        compact={layoutMode === "mobile"}
      />

      {pendingDeleteIds.length > 0 && (
        <div
          data-testid="world-book-pending-delete-banner"
          className="rounded border border-amber-300 bg-amber-50 px-3 py-2"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="text-sm text-amber-800">
                <strong>{pendingDeleteIds.length}</strong>{" "}
                {pendingDeleteIds.length === 1 ? "world book" : "world books"} pending deletion.
              </p>
              <p className="text-xs text-amber-700">
                Timers run only in this tab. Refreshing or navigating away cancels pending deletions.
              </p>
            </div>
            <Button
              size="small"
              onClick={cancelPendingWorldBookDeletes}
              aria-label="Cancel pending world book deletions"
            >
              Cancel pending
            </Button>
          </div>
        </div>
      )}

      {selectedWorldBookKeys.length > 0 && (
        <div className="rounded border border-border bg-surface-secondary px-3 py-2 flex flex-wrap items-center justify-between gap-2">
          <span className="text-sm">
            <strong>{selectedWorldBookKeys.length}</strong> selected
          </span>
          <div className="flex items-center gap-2">
            <Button size="small" loading={bulkExportMode === "selected"} onClick={() => void exportWorldBookBundle("selected")}>Export selected</Button>
            <Button size="small" loading={bulkWorldBookAction === "enable"} onClick={() => void handleBulkWorldBookAction("enable")}>Enable</Button>
            <Button size="small" loading={bulkWorldBookAction === "disable"} onClick={() => void handleBulkWorldBookAction("disable")}>Disable</Button>
            <Button size="small" danger loading={bulkWorldBookAction === "delete"} onClick={() => void handleBulkWorldBookAction("delete")}>Delete</Button>
          </div>
        </div>
      )}

      {status === "pending" && <Skeleton active paragraph={{ rows: 6 }} />}

      {status === "success" && (
        Array.isArray(data) && data.length === 0 && !hasActiveListFilters ? (
          <WorldBookEmptyState
            onCreateNew={() => setOpen(true)}
            onCreateFromTemplate={(key) => {
              createForm.setFieldsValue({ template_key: key })
              setOpen(true)
            }}
            onImport={openImportModal}
          />
        ) : (
          <>
            {layoutMode === "desktop" && (
              <div className="flex gap-4" data-testid="world-books-two-panel">
                <div className="w-[35%] min-w-[280px] shrink-0">
                  <WorldBookListPanel
                    worldBooks={filteredWorldBooks}
                    selectedWorldBookId={selectedWorldBookId}
                    onSelectWorldBook={(id) => selectWorldBookInDetail(id, "entries")}
                    selectedRowKeys={selectedWorldBookKeys}
                    onSelectedRowKeysChange={setSelectedWorldBookKeys}
                    pendingDeleteIds={pendingDeleteIds}
                    onEditWorldBook={(record) => selectWorldBookInDetail(record.id, "settings")}
                    onRowAction={handleRowAction}
                    tableSort={tableSort}
                    onTableSortChange={handleTableSortChange}
                    loading={false}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <WorldBookDetailPanel
                    worldBook={selectedWorldBookRecord}
                    attachedCharacters={selectedWorldBookAttached}
                    allWorldBooks={(data || []) as any[]}
                    allCharacters={(characters || []) as any[]}
                    activeTab={detailActiveTab}
                    onActiveTabChange={setDetailActiveTab}
                    onUpdateWorldBook={updateWB}
                    onAttachCharacter={async (characterId) => {
                      if (selectedWorldBookId) {
                        await attachWB({ characterId, worldBookId: selectedWorldBookId })
                      }
                    }}
                    onDetachCharacter={async (characterId) => {
                      if (selectedWorldBookId) {
                        await detachWB({ characterId, worldBookId: selectedWorldBookId })
                      }
                    }}
                    onOpenTestMatching={(id) => openTestMatchingModal(id)}
                    maxRecursiveDepth={maxRecursiveDepth}
                    updating={updating}
                    entryFormInstance={entryForm}
                    settingsFormInstance={settingsForm}
                    entryFilterPreset={entryFilterPreset}
                    settingsBanner={editId == null ? renderEditConflictBanner() : null}
                    statsData={selectedWorldBookStats}
                    statsLoading={
                      detailActiveTab === "stats" &&
                      (selectedWorldBookStatsStatus === "pending" || selectedWorldBookStatsFetching)
                    }
                    statsError={
                      detailActiveTab === "stats"
                        ? (selectedWorldBookStatsError as any)?.message || null
                        : null
                    }
                  />
                </div>
              </div>
            )}

            {layoutMode === "tablet" && (
              <div className="space-y-4" data-testid="world-books-stacked">
                <WorldBookListPanel
                  worldBooks={filteredWorldBooks}
                  selectedWorldBookId={selectedWorldBookId}
                  onSelectWorldBook={(id) => selectWorldBookInDetail(id, "entries")}
                  selectedRowKeys={selectedWorldBookKeys}
                  onSelectedRowKeysChange={setSelectedWorldBookKeys}
                  pendingDeleteIds={pendingDeleteIds}
                  onEditWorldBook={(record) => selectWorldBookInDetail(record.id, "settings")}
                  onRowAction={handleRowAction}
                  tableSort={tableSort}
                  onTableSortChange={handleTableSortChange}
                  loading={false}
                  collapsible
                />
                <WorldBookDetailPanel
                  worldBook={selectedWorldBookRecord}
                  attachedCharacters={selectedWorldBookAttached}
                  allWorldBooks={(data || []) as any[]}
                  allCharacters={(characters || []) as any[]}
                  activeTab={detailActiveTab}
                  onActiveTabChange={setDetailActiveTab}
                  onUpdateWorldBook={updateWB}
                  onAttachCharacter={async (characterId) => {
                    if (selectedWorldBookId) {
                      await attachWB({ characterId, worldBookId: selectedWorldBookId })
                    }
                  }}
                  onDetachCharacter={async (characterId) => {
                    if (selectedWorldBookId) {
                      await detachWB({ characterId, worldBookId: selectedWorldBookId })
                    }
                  }}
                  onOpenTestMatching={(id) => openTestMatchingModal(id)}
                  maxRecursiveDepth={maxRecursiveDepth}
                  updating={updating}
                  entryFormInstance={entryForm}
                  settingsFormInstance={settingsForm}
                  entryFilterPreset={entryFilterPreset}
                  settingsBanner={editId == null ? renderEditConflictBanner() : null}
                  statsData={selectedWorldBookStats}
                  statsLoading={
                    detailActiveTab === "stats" &&
                    (selectedWorldBookStatsStatus === "pending" || selectedWorldBookStatsFetching)
                  }
                  statsError={
                    detailActiveTab === "stats"
                      ? (selectedWorldBookStatsError as any)?.message || null
                      : null
                  }
                />
              </div>
            )}

            {layoutMode === "mobile" && (
              <div data-testid="world-books-mobile">
                {selectedWorldBookId === null ? (
                  <WorldBookListPanel
                    worldBooks={filteredWorldBooks}
                    selectedWorldBookId={selectedWorldBookId}
                    onSelectWorldBook={(id) => selectWorldBookInDetail(id, "entries")}
                    selectedRowKeys={selectedWorldBookKeys}
                    onSelectedRowKeysChange={setSelectedWorldBookKeys}
                    pendingDeleteIds={pendingDeleteIds}
                    onEditWorldBook={(record) => selectWorldBookInDetail(record.id, "settings")}
                    onRowAction={handleRowAction}
                    tableSort={tableSort}
                    onTableSortChange={handleTableSortChange}
                    loading={false}
                  />
                ) : (
                  <WorldBookDetailPanel
                    worldBook={selectedWorldBookRecord}
                    attachedCharacters={selectedWorldBookAttached}
                    allWorldBooks={(data || []) as any[]}
                    allCharacters={(characters || []) as any[]}
                    activeTab={detailActiveTab}
                    onActiveTabChange={setDetailActiveTab}
                    onUpdateWorldBook={updateWB}
                    onAttachCharacter={async (characterId) => {
                      if (selectedWorldBookId) {
                        await attachWB({ characterId, worldBookId: selectedWorldBookId })
                      }
                    }}
                    onDetachCharacter={async (characterId) => {
                      if (selectedWorldBookId) {
                        await detachWB({ characterId, worldBookId: selectedWorldBookId })
                      }
                    }}
                    onOpenTestMatching={(id) => openTestMatchingModal(id)}
                    maxRecursiveDepth={maxRecursiveDepth}
                    updating={updating}
                    entryFormInstance={entryForm}
                    settingsFormInstance={settingsForm}
                    entryFilterPreset={entryFilterPreset}
                    settingsBanner={editId == null ? renderEditConflictBanner() : null}
                    statsData={selectedWorldBookStats}
                    statsLoading={
                      detailActiveTab === "stats" &&
                      (selectedWorldBookStatsStatus === "pending" || selectedWorldBookStatsFetching)
                    }
                    statsError={
                      detailActiveTab === "stats"
                        ? (selectedWorldBookStatsError as any)?.message || null
                        : null
                    }
                    onBack={() => setSelectedWorldBookId(null)}
                  />
                )}
              </div>
            )}
          </>
        )
      )}

      <Modal
        title="Create World Book"
        open={open}
        onCancel={handleCloseCreate}
        footer={null}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        <WorldBookForm
          mode="create"
          form={createForm}
          worldBooks={(data || []) as any[]}
          submitting={creating}
          maxRecursiveDepth={maxRecursiveDepth}
          onSubmit={createWB}
        />
      </Modal>

      <Modal
        title="Import World Book (JSON)"
        open={openImport}
        onCancel={closeImportModal}
        footer={null}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        <div className="space-y-3">
          <details
            className="rounded border border-border px-3 py-2"
            open={importFormatHelpOpen}
            onToggle={(event) => {
              const nextOpen = (event.currentTarget as HTMLDetailsElement).open
              setImportFormatHelpOpen(nextOpen)
            }}
          >
            <summary
              className="cursor-pointer text-sm font-medium"
              aria-expanded={importFormatHelpOpen}
              aria-controls={importFormatHelpContentId}
            >
              Format help
            </summary>
            <div id={importFormatHelpContentId} className="mt-2 space-y-2 text-xs text-text-muted">
              <p>Expected tldw JSON shape:</p>
              <pre className="overflow-auto rounded bg-surface-secondary p-2 text-[11px] leading-5 text-text">
{`{
  "world_book": { "name": "My Lorebook", "description": "...", "scan_depth": 3, "token_budget": 500 },
  "entries": [{ "keywords": ["keyword"], "content": "Lore content" }]
}`}
              </pre>
              <p>
                Required fields: <code>world_book.name</code>, at least one
                <code> entries[]</code> item, and each entry needs
                <code> keywords[]</code> + <code>content</code>.
              </p>
              <p>Also supported: SillyTavern and Kobold export formats.</p>
            </div>
          </details>
          <Upload
            data-testid="world-book-import-upload"
            accept=".json,application/json"
            maxCount={1}
            showUploadList={false}
            beforeUpload={(file) => {
              void handleImportUpload(file as File)
              return false
            }}
          >
            <Button icon={<UploadIcon className="h-4 w-4" />} aria-label="Import world book JSON file">
              Choose JSON file
            </Button>
          </Upload>
          <label className="inline-flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={mergeOnConflict}
              onChange={(ev) => setMergeOnConflict(ev.target.checked)}
            />
            Merge on conflict
            <Tooltip title={WORLD_BOOK_IMPORT_MERGE_HELP_TEXT}>
              <HelpCircle
                role="img"
                aria-label="Merge on conflict help"
                className="h-4 w-4 text-text-muted cursor-help"
              />
            </Tooltip>
          </label>
          {importFileName && <p className="text-xs text-text-muted">Selected: {importFileName}</p>}
          {importError && <p className="text-sm text-danger">{importError}</p>}
          {importError && importErrorDetails && (
            <details
              className="rounded border border-border px-3 py-2 text-xs text-text-muted"
              data-testid="import-error-details"
              open={importErrorDetailsOpen}
              onToggle={(event) => {
                const nextOpen = (event.currentTarget as HTMLDetailsElement).open
                setImportErrorDetailsOpen(nextOpen)
              }}
            >
              <summary
                className="cursor-pointer font-medium"
                aria-expanded={importErrorDetailsOpen}
                aria-controls={importErrorDetailsContentId}
              >
                More details
              </summary>
              <pre
                id={importErrorDetailsContentId}
                className="mt-2 whitespace-pre-wrap break-words text-[11px] leading-5 text-text-muted"
              >
                {importErrorDetails}
              </pre>
            </details>
          )}
          {importPreview && !importError && (
            <div className="p-3 rounded bg-surface-secondary text-sm space-y-1">
              <p><strong>Will import:</strong> {importPreview.name}</p>
              <p><strong>Entries:</strong> {importPreview.entryCount}</p>
              {importPreview.format && (
                <p><strong>Detected format:</strong> {getWorldBookImportFormatLabel(importPreview.format)}</p>
              )}
              {importPreview.settings && (
                <div className="pt-1 space-y-1">
                  <p><strong>Settings:</strong></p>
                  <p>Scan depth: {importPreview.settings.scanDepth ?? "Default"}</p>
                  <p>Token budget: {importPreview.settings.tokenBudget ?? "Default"}</p>
                  <p>
                    Recursive scanning:{" "}
                    {typeof importPreview.settings.recursiveScanning === "boolean"
                      ? importPreview.settings.recursiveScanning ? "Enabled" : "Disabled"
                      : "Default"}
                  </p>
                  <p>
                    World book enabled:{" "}
                    {typeof importPreview.settings.enabled === "boolean"
                      ? importPreview.settings.enabled ? "Enabled" : "Disabled"
                      : "Default"}
                  </p>
                </div>
              )}
              {(importPreview.previewEntries || []).length > 0 && (
                <details
                  className="pt-1"
                  data-testid="import-preview-entries"
                  open={importPreviewEntriesOpen}
                  onToggle={(event) => {
                    const nextOpen = (event.currentTarget as HTMLDetailsElement).open
                    setImportPreviewEntriesOpen(nextOpen)
                  }}
                >
                  <summary
                    className="cursor-pointer text-sm font-medium"
                    aria-expanded={importPreviewEntriesOpen}
                    aria-controls={importPreviewEntriesContentId}
                  >
                    Preview first {importPreview.previewEntries?.length} entries
                  </summary>
                  <div id={importPreviewEntriesContentId} className="mt-2 space-y-2">
                    {(importPreview.previewEntries || []).map((entry, index) => (
                      <div
                        key={`${index}-${entry.keywords.join(",")}`}
                        className="rounded border border-border px-2 py-1"
                        data-testid={`import-preview-entry-${index + 1}`}
                      >
                        <div className="flex flex-wrap gap-1 mb-1">
                          {(entry.keywords || []).slice(0, 5).map((keyword) => (
                            <Tag key={`${index}-${keyword}`}>{keyword}</Tag>
                          ))}
                        </div>
                        <p className="text-xs text-text-muted break-words">
                          {entry.contentPreview || "(No content preview)"}
                        </p>
                      </div>
                    ))}
                    {importPreview.entryCount > (importPreview.previewEntries || []).length && (
                      <p className="text-xs text-text-muted">
                        Showing first {(importPreview.previewEntries || []).length} of{" "}
                        {importPreview.entryCount} entries.
                      </p>
                    )}
                  </div>
                </details>
              )}
              {importPreview.conflict && (
                <p className="text-warning">
                  Name conflict detected. Enable "Merge on conflict" to append imported entries to the existing world book.
                </p>
              )}
              {(importPreview.warnings || []).length > 0 && (
                <div className="space-y-1">
                  <p className="font-medium">Conversion warnings:</p>
                  {(importPreview.warnings || []).slice(0, 5).map((warning, index) => (
                    <p key={`${warning}-${index}`} className="text-xs text-warning">- {warning}</p>
                  ))}
                </div>
              )}
            </div>
          )}
          <Button
            type="primary"
            className="w-full"
            loading={importing}
            disabled={!importPayload || !!importError}
            onClick={() => {
              if (!importPayload) return
              doImport({ ...importPayload, merge_on_conflict: mergeOnConflict })
            }}
          >
            Import
          </Button>
        </div>
      </Modal>

      <Modal
        title="World Book Statistics"
        open={!!statsFor}
        onCancel={() => setStatsFor(null)}
        footer={null}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        {statsFor && (
          <div className="space-y-2">
            {(() => {
              const worldBookId = Number(statsFor.world_book_id)
              const matchingBook = ((data || []) as any[]).find(
                (book: any) => Number(book?.id) === worldBookId
              )
              const tokenBudget =
                typeof statsFor?.token_budget === "number"
                  ? statsFor.token_budget
                  : matchingBook?.token_budget
              const utilizationPercent = getBudgetUtilizationPercent(
                statsFor.estimated_tokens,
                tokenBudget
              )
              const utilizationBand = getBudgetUtilizationBand(utilizationPercent)
              const utilizationColor = getBudgetUtilizationColor(utilizationBand)
              const estimatorNote = getTokenEstimatorNote(statsFor)

              return (
                <>
                  <p className="text-xs text-text-muted">
                    {estimatorNote}
                  </p>
                  <p className="text-xs text-text-muted">
                    Tip: Click linked metrics to open the entries drawer with matching filters.
                  </p>
                  <Descriptions size="small" bordered column={1}>
                    <Descriptions.Item label="ID">{statsFor.world_book_id}</Descriptions.Item>
                    <Descriptions.Item label="Name">{statsFor.name}</Descriptions.Item>
                    <Descriptions.Item label="Total Entries">{statsFor.total_entries}</Descriptions.Item>
                    <Descriptions.Item label="Enabled Entries">
                      {Number(statsFor.enabled_entries || 0) > 0 ? (
                        <Button
                          type="link"
                          size="small"
                          className="px-0"
                          aria-label="Open enabled entries"
                          onClick={() =>
                            openEntriesFromStats({
                              enabledFilter: "enabled",
                              matchFilter: "all",
                              searchText: ""
                            })
                          }
                        >
                          {statsFor.enabled_entries}
                        </Button>
                      ) : (
                        statsFor.enabled_entries
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="Disabled Entries">
                      {Number(statsFor.disabled_entries || 0) > 0 ? (
                        <Button
                          type="link"
                          size="small"
                          className="px-0"
                          aria-label="Open disabled entries"
                          onClick={() =>
                            openEntriesFromStats({
                              enabledFilter: "disabled",
                              matchFilter: "all",
                              searchText: ""
                            })
                          }
                        >
                          {statsFor.disabled_entries}
                        </Button>
                      ) : (
                        statsFor.disabled_entries
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="Total Keywords">{statsFor.total_keywords}</Descriptions.Item>
                    <Descriptions.Item label="Regex Entries">
                      {Number(statsFor.regex_entries || 0) > 0 ? (
                        <Button
                          type="link"
                          size="small"
                          className="px-0"
                          aria-label="Open regex entries"
                          onClick={() =>
                            openEntriesFromStats({
                              enabledFilter: "all",
                              matchFilter: "regex",
                              searchText: ""
                            })
                          }
                        >
                          {statsFor.regex_entries}
                        </Button>
                      ) : (
                        statsFor.regex_entries
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="Case Sensitive Entries">{statsFor.case_sensitive_entries}</Descriptions.Item>
                    <Descriptions.Item label="Average Priority">{statsFor.average_priority}</Descriptions.Item>
                    <Descriptions.Item label="Total Content Length">{statsFor.total_content_length}</Descriptions.Item>
                    <Descriptions.Item label="Estimated Tokens">{statsFor.estimated_tokens}</Descriptions.Item>
                    <Descriptions.Item label="Token Budget">
                      {typeof tokenBudget === "number" && Number.isFinite(tokenBudget)
                        ? tokenBudget
                        : <span className="text-text-muted">Not configured</span>}
                    </Descriptions.Item>
                    <Descriptions.Item label="Budget Utilization">
                      {typeof utilizationPercent === "number" ? (
                        <div className="space-y-1">
                          <p>
                            {statsFor.estimated_tokens}/{tokenBudget} ({utilizationPercent.toFixed(1)}%)
                          </p>
                          <Progress
                            percent={Math.min(utilizationPercent, 100)}
                            status={utilizationPercent > 100 ? "exception" : "normal"}
                            strokeColor={utilizationColor}
                            size="small"
                          />
                          {utilizationPercent > 100 && (
                            <div className="space-y-0.5">
                              <p className="text-xs text-danger">
                                Estimated token usage exceeds the configured budget.
                              </p>
                              <p className="text-xs text-text-muted">
                                Reduce entry content or increase token budget.
                              </p>
                            </div>
                          )}
                        </div>
                      ) : (
                        <span className="text-text-muted">Budget unavailable</span>
                      )}
                    </Descriptions.Item>
                  </Descriptions>
                </>
              )
            })()}
          </div>
        )}
      </Modal>

      <Modal
        title="Global World Book Statistics"
        open={openGlobalStats}
        onCancel={() => setOpenGlobalStats(false)}
        footer={null}
        width={780}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        {globalStatsStatus === "pending" && (
          <Skeleton active paragraph={{ rows: 8 }} />
        )}
        {globalStatsStatus === "success" && globalStats && (() => {
          const utilizationPercent = getBudgetUtilizationPercent(
            globalStats.totalEstimatedTokens,
            globalStats.totalTokenBudget
          )
          const utilizationBand = getBudgetUtilizationBand(utilizationPercent)
          const utilizationColor = getBudgetUtilizationColor(utilizationBand)

          return (
            <div className="space-y-3">
              <Descriptions size="small" bordered column={1}>
                <Descriptions.Item label="Total World Books">
                  {globalStats.totalBooks}
                </Descriptions.Item>
                <Descriptions.Item label="Total Entries">
                  {globalStats.totalEntries}
                </Descriptions.Item>
                <Descriptions.Item label="Total Keywords">
                  {globalStats.totalKeywords}
                </Descriptions.Item>
                <Descriptions.Item label="Estimated Tokens">
                  {globalStats.totalEstimatedTokens}
                </Descriptions.Item>
                <Descriptions.Item label="Aggregate Token Budget">
                  {globalStats.totalTokenBudget > 0
                    ? globalStats.totalTokenBudget
                    : <span className="text-text-muted">Not configured</span>}
                </Descriptions.Item>
                <Descriptions.Item label="Budget Utilization">
                  {typeof utilizationPercent === "number" ? (
                    <div className="space-y-1">
                      <p>
                        {globalStats.totalEstimatedTokens}/{globalStats.totalTokenBudget} ({utilizationPercent.toFixed(1)}%)
                      </p>
                      <Progress
                        percent={Math.min(utilizationPercent, 100)}
                        status={utilizationPercent > 100 ? "exception" : "normal"}
                        strokeColor={utilizationColor}
                        size="small"
                      />
                      {utilizationPercent > 100 && (
                        <div className="space-y-0.5">
                          <p className="text-xs text-danger">
                            Estimated token usage exceeds the configured budget.
                          </p>
                          <p className="text-xs text-text-muted">
                            Reduce entry content or increase token budget.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <span className="text-text-muted">Budget unavailable</span>
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="Shared Keywords Across Books">
                  {globalStats.sharedKeywordCount}
                </Descriptions.Item>
                <Descriptions.Item label="Cross-book Keyword Conflicts">
                  {globalStats.conflictKeywordCount}
                </Descriptions.Item>
              </Descriptions>

              <Divider className="my-2" />

              <div className="space-y-2">
                <p className="text-xs text-text-muted">
                  Click a book under a keyword conflict to open entries filtered by that keyword.
                </p>
                {globalStats.conflicts.length === 0 ? (
                  <p className="text-sm text-text-muted">No cross-book keyword conflicts detected.</p>
                ) : (
                  <div className="space-y-2 max-h-72 overflow-auto pr-1">
                    {globalStats.conflicts.slice(0, 20).map((conflict) => (
                      <div
                        key={`${conflict.keyword}-${conflict.occurrenceCount}`}
                        className="rounded border border-border px-3 py-2"
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <Tag color="volcano">{conflict.keyword}</Tag>
                          <span className="text-xs text-text-muted">
                            {conflict.affectedBooks.length} books, {conflict.variantCount} content variants
                          </span>
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {conflict.affectedBooks.map((book) => (
                            <Button
                              key={`${conflict.keyword}-${book.id}`}
                              type="link"
                              size="small"
                              className="px-0"
                              aria-label={`Open conflict keyword ${conflict.keyword} in ${book.name}`}
                              onClick={() => openEntriesFromGlobalStats(book.id, conflict.keyword)}
                            >
                              {book.name}
                            </Button>
                          ))}
                        </div>
                      </div>
                    ))}
                    {globalStats.conflicts.length > 20 && (
                      <p className="text-xs text-text-muted">
                        Showing first 20 conflicts of {globalStats.conflicts.length}.
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          )
        })()}
      </Modal>

      <WorldBookTestMatchingModal
        open={openTestMatching}
        onClose={closeTestMatchingModal}
        worldBooks={Array.isArray(data) ? (data as any[]) : []}
        initialWorldBookId={testMatchingWorldBookId}
      />

      <Modal
        title="Edit World Book"
        open={openEdit}
        onCancel={handleCloseEdit}
        footer={null}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        {renderEditConflictBanner()}
        <WorldBookForm
          mode="edit"
          form={editForm}
          worldBooks={(data || []) as any[]}
          submitting={updating}
          currentWorldBookId={editId}
          maxRecursiveDepth={maxRecursiveDepth}
          onSubmit={updateWB}
        />
      </Modal>

      <Drawer
        title={(
          <div className="space-y-1">
            <div className="text-xs text-text-muted">World Books &gt; {openEntries?.name || ''} &gt; Entries</div>
            <div className="font-semibold">Entries: {openEntries?.name || ''}</div>
          </div>
        )}
        placement="right"
        size={screens.md ? "60vw" : "100%"}
        open={!!openEntries}
        onClose={handleCloseEntries}
        destroyOnHidden
      >
        {openEntries && (
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <Tag color="blue">Editing: {openEntries.name}</Tag>
              <Tag>Entries: {openEntries.entryCount ?? '—'}</Tag>
              <Tag>Attached: {getAttachedCharacters(openEntries.id).length}</Tag>
              {typeof openEntries.tokenBudget === "number" && (
                <Tag>Budget: {openEntries.tokenBudget}</Tag>
              )}
            </div>
            <Button
              size="small"
              aria-label="Test keywords for this world book"
              onClick={() => openTestMatchingModal(openEntries.id)}
            >
              Test Keywords
            </Button>
          </div>
        )}
        <EntryManager
          worldBookId={openEntries?.id!}
          worldBookName={openEntries?.name}
          tokenBudget={openEntries?.tokenBudget}
          worldBooks={(data || []) as any[]}
          entryFilterPreset={entryFilterPreset}
          form={entryForm}
        />
      </Drawer>

      <Modal
        title="World Book ↔ Character Matrix"
        open={openMatrix}
        onCancel={handleCloseMatrix}
        footer={null}
        width="90vw"
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        <div className="text-sm text-text-muted mb-3">
          Toggle checkboxes to attach or detach world books from characters.
        </div>
        {matrixFeedback && (
          <div
            role="status"
            aria-live="polite"
            className={`mb-3 rounded border px-2 py-1 text-xs ${
              matrixFeedback.kind === "success"
                ? "border-blue-300 bg-blue-50 text-blue-700"
                : "border-rose-300 bg-rose-50 text-rose-700"
            }`}
          >
            {matrixFeedback.message}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <Input
            placeholder="Filter world books…"
            value={matrixBookFilter}
            onChange={(e) => setMatrixBookFilter(e.target.value)}
            allowClear
            className="max-w-xs"
          />
          <Input
            placeholder="Filter characters…"
            value={matrixCharacterFilter}
            onChange={(e) => setMatrixCharacterFilter(e.target.value)}
            allowClear
            className="max-w-xs"
          />
          <Button
            size="small"
            onClick={() => {
              setMatrixBookFilter('')
              setMatrixCharacterFilter('')
            }}
            disabled={!matrixBookFilter && !matrixCharacterFilter}
          >
            Clear filters
          </Button>
        </div>
        {filteredCharacters.length === 0 && (
          <Empty description="No characters match this filter" />
        )}
        <div className="text-xs text-text-muted mb-2" aria-live="polite">
          {useAttachmentListView
            ? `List view active (${filteredCharacters.length} characters).`
            : `Matrix view active (${filteredCharacters.length} characters).`}
        </div>
        {useAttachmentListView ? (
          <div className="border border-border rounded">
            <Table
              size="small"
              rowKey={(r: any) => r.id}
              dataSource={filteredBooks}
              pagination={{
                current: matrixListPage,
                pageSize: ATTACHMENT_LIST_PAGE_SIZE,
                total: filteredBooks.length,
                onChange: (page) => setMatrixListPage(page),
                showSizeChanger: false
              }}
              columns={[
                {
                  title: "World Book",
                  dataIndex: "name",
                  key: "name",
                  width: 220
                },
                {
                  title: "Attached Characters",
                  key: "attached_characters",
                  render: (_: any, record: any) => {
                    const attachedIds = normalizeCharacterIds(
                      getAttachedCharacters(record.id).map((character: any) => character?.id)
                    )
                    const rowDeltaStates = (filteredCharacters || [])
                      .map((character: any) => matrixSessionDeltas[attachmentKeyFor(record.id, character.id)])
                      .filter(
                        (value): value is "attached" | "detached" =>
                          value === "attached" || value === "detached"
                      )
                    const attachedDeltaCount = rowDeltaStates.filter(
                      (value) => value === "attached"
                    ).length
                    const detachedDeltaCount = rowDeltaStates.filter(
                      (value) => value === "detached"
                    ).length
                    const rowPending = filteredCharacters.some(
                      (character: any) => matrixPending[attachmentKeyFor(record.id, character.id)]
                    )
                    return (
                      <div className="space-y-1">
                        <Select
                          mode="multiple"
                          allowClear
                          showSearch
                          optionFilterProp="label"
                          aria-label={`Attachment selector for ${record?.name || "world book"}`}
                          placeholder="Select characters"
                          className="w-full"
                          value={attachedIds}
                          options={(filteredCharacters || []).map((character: any) => ({
                            label: character.name,
                            value: character.id
                          }))}
                          disabled={attachmentsLoading || rowPending || filteredCharacters.length === 0}
                          onChange={(values) => {
                            void handleListAttachmentChange(
                              record.id,
                              values as Array<number | string>
                            )
                          }}
                        />
                        <div className="text-xs text-text-muted">
                          {attachedIds.length} attached
                        </div>
                        {(attachedDeltaCount > 0 || detachedDeltaCount > 0) && (
                          <div className="flex flex-wrap items-center gap-1 text-[11px]">
                            {attachedDeltaCount > 0 && (
                              <Tag color="blue">+{attachedDeltaCount} new</Tag>
                            )}
                            {detachedDeltaCount > 0 && (
                              <Tag color="orange">-{detachedDeltaCount} removed</Tag>
                            )}
                          </div>
                        )}
                        <Button
                          size="small"
                          type="text"
                          aria-label={`Detach all characters from ${record?.name || "world book"}`}
                          disabled={attachedIds.length === 0 || attachmentsLoading || rowPending}
                          onClick={() => {
                            void handleListAttachmentChange(record.id, [])
                          }}
                        >
                          Detach all
                        </Button>
                      </div>
                    )
                  }
                }
              ] as any}
            />
          </div>
        ) : (
          <div
            className="overflow-x-auto border border-border rounded"
            role="grid"
            aria-label="World book attachment matrix"
            aria-rowcount={filteredBooks.length}
            aria-colcount={filteredCharacters.length + 1}
          >
            <Table
              size="small"
              pagination={false}
              scroll={{ x: "max-content" }}
              rowKey={(r: any) => r.id}
              dataSource={filteredBooks}
              columns={[
                { title: 'World Book', dataIndex: 'name', key: 'name', fixed: 'left', width: 200 },
                ...(filteredCharacters || []).map((c: any, characterColumnIndex: number) => ({
                  title: (
                    <Tooltip title={c.name}>
                      <a
                        href={`/characters?from=world-books&focusCharacterId=${encodeURIComponent(
                          String(c.id)
                        )}`}
                        className="truncate max-w-[140px] inline-block text-primary hover:underline"
                        onClick={(event) => event.stopPropagation()}
                        aria-label={`Open character ${c.name || `Character ${c.id}`}`}
                      >
                        {c.name}
                      </a>
                    </Tooltip>
                  ),
                  key: `char-${c.id}`,
                  width: 120,
                  render: (_: any, record: any, rowIndex: number) => {
                    const checked = isAttached(record.id, c.id)
                    const key = attachmentKeyFor(record.id, c.id)
                    const pending = !!matrixPending[key]
                    const deltaState = matrixSessionDeltas[key] || "none"
                    const pulse = !!matrixSuccessPulse[key]
                    const metadata = getAttachmentMetadata(record.id, c.id)
                    const metadataDraft = matrixMetaDrafts[key] || metadata
                    const isMetaPopoverOpen = matrixMetaPopoverOpenKey === key
                    const normalizedPriority = Number.isFinite(metadata.priority)
                      ? metadata.priority
                      : 0
                    const normalizedDraftPriority = Number.isFinite(metadataDraft.priority)
                      ? metadataDraft.priority
                      : 0
                    return (
                      <div
                        data-testid={`matrix-cell-${record.id}-${c.id}`}
                        data-delta-state={deltaState}
                        className={`inline-flex items-center justify-center gap-1 rounded px-1 py-1 transition-all ${
                          deltaState === "attached"
                            ? "ring-2 ring-blue-400 bg-blue-50"
                            : deltaState === "detached"
                              ? "ring-2 ring-amber-400 bg-amber-50"
                              : ""
                        } ${pulse && !prefersReducedMotion ? "animate-pulse" : pulse ? "ring-2 ring-blue-400" : ""}`}
                      >
                        <Checkbox
                          aria-label={`Toggle attachment ${record?.name || "world book"} / ${c?.name || "character"}`}
                          checked={checked}
                          disabled={pending || attachmentsLoading}
                          onChange={(e) => handleMatrixToggle(record.id, c.id, e.target.checked)}
                          data-matrix-checkbox="true"
                          data-matrix-row-index={rowIndex}
                          data-matrix-col-index={characterColumnIndex}
                          onKeyDown={(event) =>
                            handleMatrixCellKeyDown(event, rowIndex, characterColumnIndex)
                          }
                        />
                        {checked && (
                          <>
                            <span
                              className={`text-[10px] ${
                                metadata.enabled ? "text-text-muted" : "text-warning"
                              }`}
                            >
                              P{normalizedPriority}
                            </span>
                            <Popover
                              trigger="click"
                              placement="bottomRight"
                              open={isMetaPopoverOpen}
                              onOpenChange={(nextOpen) => {
                                if (nextOpen) {
                                  openMatrixMetadataEditor(record.id, c.id)
                                } else if (isMetaPopoverOpen) {
                                  setMatrixMetaPopoverOpenKey(null)
                                }
                              }}
                              content={(
                                <div className="w-56 space-y-2">
                                  <div className="text-xs font-medium">Attachment settings</div>
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="text-xs">Enabled</span>
                                    <Switch
                                      size="small"
                                      aria-label={`Attachment enabled ${record?.name || "world book"} / ${c?.name || "character"}`}
                                      checked={metadataDraft.enabled}
                                      disabled={pending}
                                      onChange={(nextEnabled) =>
                                        updateMatrixMetadataDraft(record.id, c.id, {
                                          enabled: nextEnabled
                                        })
                                      }
                                      {...ACCESSIBLE_SWITCH_TEXT_PROPS}
                                    />
                                  </div>
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="text-xs">Priority</span>
                                    <InputNumber
                                      size="small"
                                      aria-label={`Attachment priority ${record?.name || "world book"} / ${c?.name || "character"}`}
                                      value={normalizedDraftPriority}
                                      disabled={pending}
                                      onChange={(value) => {
                                        const nextPriority = Number(value)
                                        updateMatrixMetadataDraft(record.id, c.id, {
                                          priority: Number.isFinite(nextPriority)
                                            ? nextPriority
                                            : 0
                                        })
                                      }}
                                    />
                                  </div>
                                  <div className="flex justify-end gap-2">
                                    <Button
                                      size="small"
                                      onClick={() => setMatrixMetaPopoverOpenKey(null)}
                                      disabled={pending}
                                    >
                                      Cancel
                                    </Button>
                                    <Button
                                      size="small"
                                      type="primary"
                                      loading={pending}
                                      onClick={() => {
                                        void saveMatrixMetadata(record.id, c.id)
                                      }}
                                    >
                                      Save
                                    </Button>
                                  </div>
                                </div>
                              )}
                            >
                              <Button
                                size="small"
                                type="text"
                                icon={<List className="h-3 w-3" />}
                                aria-label={`Edit attachment settings ${record?.name || "world book"} / ${c?.name || "character"}`}
                                disabled={pending}
                              />
                            </Popover>
                          </>
                        )}
                      </div>
                    )
                  }
                }))
              ] as any}
            />
          </div>
        )}
      </Modal>

      <Modal
        title={
          quickAttachWorldBookName
            ? `Quick attach: ${quickAttachWorldBookName}`
            : "Quick attach characters"
        }
        open={!!openAttach}
        onCancel={() => setOpenAttach(null)}
        footer={null}
        styles={{ body: MODAL_BODY_SCROLL_STYLE }}
      >
        <div className="space-y-4">
          <div className="rounded border border-border bg-background-subtle px-3 py-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-sm font-medium">Need bulk controls?</div>
                <div className="text-xs text-text-muted">
                  Open the full matrix to manage many world books and characters at once.
                </div>
              </div>
              <Button
                size="small"
                aria-label="Open full attachment matrix"
                onClick={openFullMatrixFromQuickAttach}
              >
                Open full matrix
              </Button>
            </div>
          </div>
          <div>
            <h4 className="text-sm font-medium mb-2">Currently attached</h4>
            {openAttach && getAttachedCharacters(openAttach).length > 0 ? (
              <div className="space-y-2">
                {getAttachedCharacters(openAttach).map((c: any) => (
                  <div key={c.id} className="flex items-center justify-between gap-2">
                    <a
                      href={`/characters?from=world-books&focusCharacterId=${encodeURIComponent(
                        String(c.id)
                      )}&focusWorldBookId=${encodeURIComponent(String(openAttach))}`}
                      className="text-sm text-primary hover:underline"
                      aria-label={`Open character ${c.name || `Character ${c.id}`}`}
                    >
                      {c.name || `Character ${c.id}`}
                    </a>
                    <Button
                      size="small"
                      danger
                      loading={detachingFor?.characterId === c.id && detachingFor?.worldBookId === openAttach}
                      onClick={async () => {
                        setDetachingFor({ characterId: c.id, worldBookId: openAttach })
                        try {
                          await detachWB({ characterId: c.id, worldBookId: openAttach })
                          notification.success({ message: 'Detached' })
                        } finally {
                          setDetachingFor(null)
                        }
                      }}
                    >
                      Detach
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <Empty description="No characters attached" />
            )}
          </div>
          <Divider className="my-2" />
          <Form
            layout="vertical"
            form={attachForm}
            onFinish={async (v) => {
              if (openAttach && v.character_id) {
                await attachWB({ characterId: v.character_id, worldBookId: openAttach })
                notification.success({ message: 'Attached' })
                attachForm.resetFields()
              }
            }}
          >
            <Form.Item name="character_id" label="Attach character" rules={[{ required: true }]}>
              <Select
                showSearch
                optionFilterProp="label"
                options={(characters || []).map((c: any) => ({
                  label: c.name,
                  value: c.id,
                  disabled: openAttach ? getAttachedCharacters(openAttach).some((a: any) => a.id === c.id) : false
                }))}
              />
            </Form.Item>
            <Button type="primary" htmlType="submit" className="w-full" loading={attaching}>
              Attach character
            </Button>
          </Form>
        </div>
      </Modal>
    </div>
  )
}
