import React from "react"
import { useQueryClient } from "@tanstack/react-query"
import {
  Alert,
  Badge,
  Button,
  Checkbox,
  Form,
  Drawer,
  Empty,
  Input,
  List,
  Modal,
  Pagination,
  Popover,
  Progress,
  Segmented,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography
} from "antd"
import { Filter, Plus, LayoutList, List as ListIcon, Keyboard, Check, CheckCheck } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { trackFlashcardsShortcutHintTelemetry } from "@/utils/flashcards-shortcut-hint-telemetry"
import { processInChunks } from "@/utils/chunk-processing"
import {
  DOCUMENT_VIEW_SUPPORTED_SORTS,
  getFlashcardDocumentQueryKey,
  useDecksQuery,
  useFlashcardDocumentQuery,
  type DocumentManageSortBy,
  useManageQuery,
  useUpdateFlashcardMutation,
  useUpdateFlashcardsBulkMutation,
  useUpdateDeckMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useCardsKeyboardNav,
  useTagSuggestionsQuery,
  getManageServerOrderBy,
  type DueStatus,
  type ManageSortBy
} from "../hooks"
import {
  FlashcardActionsMenu,
  FlashcardCreateDrawer,
  FlashcardEditDrawer,
  MarkdownWithBoundary,
  FlashcardMarkdownSnippet
} from "../components"
import { FlashcardDocumentView } from "../components/FlashcardDocumentView"
import { FLASHCARDS_DRAWER_WIDTH_PX } from "../constants"
import { formatCardType } from "../utils/model-type-labels"
import { FlashcardQueueStateBadge } from "../utils/queue-state-badges"
import { getFlashcardSourceMeta } from "../utils/source-reference"
import {
  formatDeckHierarchyLabel,
  getDeckDescendantIds
} from "../utils/deck-display"
import {
  formatFlashcardAbsoluteDateTime,
  formatFlashcardRelativeTime,
  isFlashcardTimestampBefore
} from "../utils/date-display"
import {
  formatFlashcardsUiErrorMessage,
  mapFlashcardsUiError
} from "../utils/error-taxonomy"
import { trackFlashcardsErrorRecoveryTelemetry } from "@/utils/flashcards-error-recovery-telemetry"
import { useFlashcardsShortcutHintDensity } from "../hooks/useFlashcardsShortcutHintDensity"
import {
  createFlashcard,
  deleteFlashcard,
  getFlashcard,
  listFlashcards,
  updateFlashcard,
  type Flashcard,
  type FlashcardUpdate
} from "@/services/flashcards"

const { Text } = Typography

const BULK_MUTATION_CHUNK_SIZE = 50
const DELETE_UNDO_SECONDS = 30
const DELETE_UNDO_MS = DELETE_UNDO_SECONDS * 1000

type PendingDeletion = {
  card: Flashcard
  expiresAt: number
  batchId: string
}

type MoveUndoSnapshot = {
  uuid: string
  previousDeckId: number | null
}

interface ManageTabProps {
  onNavigateToImport: () => void
  onReviewCard: (card: Flashcard) => void
  openCreateSignal?: number
  isActive: boolean
  initialDeckId?: number
  initialShowWorkspaceDecks?: boolean
}

export const buildFlashcardsWorkspaceVisibilityOptions = (
  showWorkspaceDecks: boolean,
  selectedWorkspaceId?: string | null
) => ({
  workspaceId: selectedWorkspaceId ?? null,
  workspace_id: selectedWorkspaceId ?? null,
  includeWorkspaceItems: selectedWorkspaceId == null ? showWorkspaceDecks : false,
  include_workspace_items: selectedWorkspaceId == null ? showWorkspaceDecks : false
})

/**
 * Cards tab for browsing, filtering, creating, editing, and bulk operations on flashcards.
 * Includes a FAB for quick card creation via a drawer.
 */
export const ManageTab: React.FC<ManageTabProps> = ({
  onNavigateToImport,
  onReviewCard,
  openCreateSignal,
  isActive,
  initialDeckId,
  initialShowWorkspaceDecks = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const qc = useQueryClient()
  const message = useAntdMessage()
  const confirmDanger = useConfirmDanger()
  const { showUndoNotification } = useUndoNotification()
  const reportUiError = React.useCallback(
    (error: unknown, operation: string, fallback: string) => {
      const mapped = mapFlashcardsUiError(error, {
        operation,
        fallback
      })
      console.warn("[flashcards:error]", {
        code: mapped.code,
        status: mapped.status,
        operation,
        raw: mapped.rawMessage
      })
      void trackFlashcardsErrorRecoveryTelemetry({
        type: "flashcards_mutation_failed",
        surface: "cards",
        operation,
        error_code: mapped.code,
        status: mapped.status ?? null,
        retriable: mapped.code !== "FLASHCARDS_VALIDATION"
      })
      message.error(formatFlashcardsUiErrorMessage(mapped))
    },
    [message]
  )

  // Track pending deletions for soft-delete with undo + trash view
  const [pendingDeletions, setPendingDeletions] = React.useState<Record<string, PendingDeletion>>({})
  const pendingDeletionsRef = React.useRef<Record<string, PendingDeletion>>({})
  const pendingDeletionBatchesRef = React.useRef<Map<string, { uuids: Set<string>; timeoutId: number }>>(new Map())

  // Track focused card index for keyboard navigation
  const [focusedIndex, setFocusedIndex] = React.useState<number>(-1)
  const [viewMode, setViewMode] = React.useState<"cards" | "trash">("cards")
  const [nowMs, setNowMs] = React.useState(() => Date.now())
  const [showWorkspaceDecks, setShowWorkspaceDecks] = React.useState(initialShowWorkspaceDecks)
  const [selectedWorkspaceId, setSelectedWorkspaceId] = React.useState<string | null>(null)
  const [deckScopeOpen, setDeckScopeOpen] = React.useState(false)
  const [deckScopeForm] = Form.useForm()

  // Shared: decks
  const workspaceVisibilityOptions = React.useMemo(
    () => buildFlashcardsWorkspaceVisibilityOptions(showWorkspaceDecks, selectedWorkspaceId),
    [selectedWorkspaceId, showWorkspaceDecks]
  ) as any
  const decksQuery = useDecksQuery(workspaceVisibilityOptions)
  const updateDeckMutation = useUpdateDeckMutation()

  // Filter state
  const [mDeckId, setMDeckId] = React.useState<number | null | undefined>(initialDeckId)
  const [mQuery, setMQuery] = React.useState("")
  const [mQueryInput, setMQueryInput] = React.useState("")
  const [mTags, setMTags] = React.useState<string[]>([])
  const [mTagInput, setMTagInput] = React.useState("")
  const [mDue, setMDue] = React.useState<DueStatus>("all")
  const [mSort, setMSort] = React.useState<ManageSortBy>("due")
  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(20)
  const [listDensity, setListDensity] = React.useState<"compact" | "expanded" | "document">("compact")
  const [shortcutHintDensity, setShortcutHintDensity] = useFlashcardsShortcutHintDensity()
  React.useEffect(() => {
    if (!isActive || viewMode !== "cards" || shortcutHintDensity === "hidden") return
    void trackFlashcardsShortcutHintTelemetry({
      type: "flashcards_shortcut_hints_exposed",
      surface: "cards",
      density: shortcutHintDensity
    })
  }, [isActive, viewMode, shortcutHintDensity])
  const cycleShortcutHintDensity = React.useCallback(() => {
    void setShortcutHintDensity((prev) => {
      const next =
        prev === "expanded"
          ? "compact"
          : prev === "compact"
            ? "hidden"
            : "expanded"
      void trackFlashcardsShortcutHintTelemetry({
        type: "flashcards_shortcut_hint_density_changed",
        surface: "cards",
        from_density: prev,
        to_density: next
      })
      if (next === "hidden" && prev !== "hidden") {
        void trackFlashcardsShortcutHintTelemetry({
          type: "flashcards_shortcut_hints_dismissed",
          surface: "cards",
          from_density: prev
        })
      }
      return next
    })
  }, [setShortcutHintDensity])
  const shortcutHintToggleLabel =
    shortcutHintDensity === "expanded"
      ? t("option:flashcards.shortcutHintsCompact", {
          defaultValue: "Compact hints"
        })
      : shortcutHintDensity === "compact"
        ? t("option:flashcards.shortcutHintsHide", {
            defaultValue: "Hide hints"
          })
        : t("option:flashcards.shortcutHintsShow", {
            defaultValue: "Show hints"
          })
  const isDocumentMode = viewMode === "cards" && listDensity === "document"
  const isDocumentSortSupported = DOCUMENT_VIEW_SUPPORTED_SORTS.includes(
    mSort as (typeof DOCUMENT_VIEW_SUPPORTED_SORTS)[number]
  )
  const documentSort: DocumentManageSortBy = isDocumentSortSupported
    ? (mSort as DocumentManageSortBy)
    : "due"

  React.useEffect(() => {
    if (!isDocumentMode || isDocumentSortSupported) return
    setMSort("due")
  }, [isDocumentMode, isDocumentSortSupported])

  const setPresentationMode = React.useCallback(
    (mode: "compact" | "expanded" | "document") => {
      setListDensity(mode)
      if (
        mode === "document" &&
        !DOCUMENT_VIEW_SUPPORTED_SORTS.includes(
          mSort as (typeof DOCUMENT_VIEW_SUPPORTED_SORTS)[number]
        )
      ) {
        setMSort("due")
      }
    },
    [mSort]
  )

  // Check if any filters are active
  const hasActiveFilters = !!(mQuery || mTags.length > 0 || mDeckId != null || mDue !== "all")

  // Clear all filters
  const clearAllFilters = () => {
    setMQuery("")
    setMQueryInput("")
    setMTags([])
    setMTagInput("")
    setMDeckId(undefined)
    setMDue("all")
    setMSort("due")
    setPage(1)
  }

  const addTagFilter = React.useCallback((rawValue: string) => {
    const normalized = rawValue.trim()
    if (!normalized) return
    setMTags((prev) => {
      const exists = prev.some((tag) => tag.toLowerCase() === normalized.toLowerCase())
      if (exists) return prev
      return [...prev, normalized]
    })
    setMTagInput("")
    setPage(1)
  }, [])

  const removeTagFilter = React.useCallback((targetTag: string) => {
    setMTags((prev) =>
      prev.filter((tag) => tag.toLowerCase() !== targetTag.toLowerCase())
    )
    setPage(1)
  }, [])

  const tagSuggestionsQuery = useTagSuggestionsQuery(mDeckId, workspaceVisibilityOptions)

  const filteredTagSuggestions = React.useMemo(() => {
    const selected = new Set(mTags.map((tag) => tag.toLowerCase()))
    const input = mTagInput.trim().toLowerCase()
    return (tagSuggestionsQuery.data || [])
      .filter((tag) => !selected.has(tag.toLowerCase()))
      .filter((tag) => (input ? tag.toLowerCase().includes(input) : true))
      .slice(0, 8)
  }, [mTags, mTagInput, tagSuggestionsQuery.data])

  const selectedDeck = React.useMemo(
    () => {
      const decks = decksQuery.data || []
      if (mDeckId != null) {
        return decks.find((deck) => deck.id === mDeckId) ?? null
      }
      if (decks.length === 1) {
        return decks[0] ?? null
      }
      return null
    },
    [decksQuery.data, mDeckId]
  )
  const deckMap = React.useMemo(
    () => new Map((decksQuery.data || []).map((deck) => [deck.id, deck])),
    [decksQuery.data]
  )
  const resolveDeckLabel = React.useCallback(
    (deckId: number | null | undefined) => {
      if (deckId == null) {
        return t("option:flashcards.noDeck", { defaultValue: "No deck" })
      }
      return formatDeckHierarchyLabel(deckMap.get(deckId), deckMap, `Deck ${deckId}`)
    },
    [deckMap, t]
  )
  const deckParentOptions = React.useMemo(() => {
    if (!selectedDeck) {
      return []
    }
    const decks = decksQuery.data || []
    const blockedDeckIds = getDeckDescendantIds(decks, selectedDeck.id)
    blockedDeckIds.add(selectedDeck.id)
    return decks
      .filter((deck) => !blockedDeckIds.has(deck.id))
      .map((deck) => ({
        label: formatDeckHierarchyLabel(deck, deckMap, `Deck ${deck.id}`),
        value: deck.id
      }))
  }, [deckMap, decksQuery.data, selectedDeck])
  const workspaceFilterOptions = React.useMemo(() => {
    const workspaceIds = new Set<string>()
    ;(decksQuery.data || []).forEach((deck) => {
      const workspaceId = deck.workspace_id?.trim()
      if (workspaceId) {
        workspaceIds.add(workspaceId)
      }
    })
    if (selectedWorkspaceId) {
      workspaceIds.add(selectedWorkspaceId)
    }
    return Array.from(workspaceIds)
      .sort((left, right) => left.localeCompare(right))
      .map((workspaceId) => ({
        label: workspaceId,
        value: workspaceId
      }))
  }, [decksQuery.data, selectedWorkspaceId])

  // Selection state
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set())
  const [previewOpen, setPreviewOpen] = React.useState<Set<string>>(new Set())
  const [selectAllAcross, setSelectAllAcross] = React.useState<boolean>(false)

  const updatePendingDeletions = React.useCallback(
    (updater: (prev: Record<string, PendingDeletion>) => Record<string, PendingDeletion>) => {
      setPendingDeletions((prev) => {
        const next = updater(prev)
        pendingDeletionsRef.current = next
        return next
      })
    },
    []
  )

  const pendingDeletionCount = Object.keys(pendingDeletions).length

  React.useEffect(() => {
    pendingDeletionsRef.current = pendingDeletions
  }, [pendingDeletions])

  React.useEffect(() => {
    if (pendingDeletionCount === 0) return
    const interval = window.setInterval(() => setNowMs(Date.now()), 1000)
    return () => window.clearInterval(interval)
  }, [pendingDeletionCount])

  // Bulk operation progress state
  const [bulkProgress, setBulkProgress] = React.useState<{
    open: boolean
    current: number
    total: number
    action: string
  } | null>(null)

  // Type-to-confirm modal state for large bulk deletes
  const [bulkDeleteConfirmOpen, setBulkDeleteConfirmOpen] = React.useState(false)
  const [bulkDeleteInput, setBulkDeleteInput] = React.useState("")
  const [bulkDeleteCount, setBulkDeleteCount] = React.useState(0)
  const [pendingDeleteItems, setPendingDeleteItems] = React.useState<Flashcard[]>([])

  const manageQuery = useManageQuery({
    deckId: mDeckId,
    query: mQuery,
    tags: mTags,
    dueStatus: mDue,
    sortBy: mSort,
    page,
    pageSize
  }, workspaceVisibilityOptions)

  const documentQuery = useFlashcardDocumentQuery(
    {
      deckId: mDeckId,
      query: mQuery,
      tags: mTags,
      dueStatus: mDue,
      sortBy: documentSort,
      includeWorkspaceItems: selectedWorkspaceId == null ? showWorkspaceDecks : false,
      workspaceId: selectedWorkspaceId
    },
    {
      enabled: viewMode === "cards" && listDensity === "document"
    }
  )
  const documentFilterContext = React.useMemo(
    () => ({
      deckId: mDeckId,
      query: mQuery,
      tags: mTags,
      dueStatus: mDue,
      sortBy: documentSort,
      workspaceId: selectedWorkspaceId,
      includeWorkspaceItems: selectedWorkspaceId == null ? showWorkspaceDecks : false
    }),
    [documentSort, mDeckId, mDue, mQuery, mTags, selectedWorkspaceId, showWorkspaceDecks]
  )
  const documentQueryKey = React.useMemo(
    () =>
      getFlashcardDocumentQueryKey(
        {
          deckId: mDeckId,
          query: mQuery,
          tags: mTags,
          dueStatus: mDue,
          sortBy: documentSort,
          workspaceId: selectedWorkspaceId,
          includeWorkspaceItems: selectedWorkspaceId == null ? showWorkspaceDecks : false
        },
        {
          sortBy: documentSort,
          dueStatus: mDue,
          workspaceId: selectedWorkspaceId,
          includeWorkspaceItems: selectedWorkspaceId == null ? showWorkspaceDecks : false
        }
      ),
    [documentSort, mDeckId, mDue, mQuery, mTags, selectedWorkspaceId, showWorkspaceDecks]
  )

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      if (mQueryInput === mQuery) return
      setMQuery(mQueryInput)
      setPage(1)
    }, 300)
    return () => window.clearTimeout(timer)
  }, [mQueryInput, mQuery])

  React.useEffect(() => {
    if (!selectAllAcross) {
      setSelectedIds(new Set())
    }
  }, [page, pageSize, selectAllAcross])

  React.useEffect(() => {
    setSelectedIds(new Set())
    setSelectAllAcross(false)
  }, [mDeckId, mQuery, mTags, mDue, mSort])

  React.useEffect(() => {
    return () => {
      pendingDeletionBatchesRef.current.forEach((batch) => {
        window.clearTimeout(batch.timeoutId)
      })
      pendingDeletionBatchesRef.current.clear()
    }
  }, [])

  const toggleSelect = (uuid: string, checked: boolean) => {
    if (selectAllAcross) return
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (checked) next.add(uuid)
      else next.delete(uuid)
      return next
    })
  }

  const selectAllOnPage = () => {
    const ids = visibleItems.map((i) => i.uuid)
    setSelectAllAcross(false)
    setSelectedIds(new Set([...(selectedIds || new Set()), ...ids]))
  }

  const clearSelection = React.useCallback(() => {
    setSelectedIds(new Set())
    setSelectAllAcross(false)
  }, [])

  const selectAllAcrossResults = () => {
    if (selectAllAcrossDisabled) return
    setSelectAllAcross(true)
    setSelectedIds(new Set())
  }

  const togglePreview = (uuid: string) => {
    setPreviewOpen((prev) => {
      const next = new Set(prev)
      if (next.has(uuid)) next.delete(uuid)
      else next.add(uuid)
      return next
    })
  }

  // Selection across results helpers - filter out pending deletions
  const pageItems = (manageQuery.data?.items || []).filter(
    (item) => !pendingDeletions[item.uuid]
  )
  const documentItems = documentQuery.items.filter(
    (item) => !pendingDeletions[item.uuid]
  )
  const visibleItems = isDocumentMode ? documentItems : pageItems
  const documentTotalCount =
    documentQuery.data && documentQuery.data.pages.length > 0
      ? (documentQuery.data.pages[documentQuery.data.pages.length - 1]?.total ?? documentItems.length)
      : documentItems.length
  const pageCount = visibleItems.length
  const selectedOnPageCount = selectAllAcross
    ? pageCount
    : visibleItems.filter((item) => selectedIds.has(item.uuid)).length
  const totalCount = isDocumentMode
    ? documentTotalCount
    : manageQuery.data?.total ?? manageQuery.data?.count ?? 0
  const selectedCount = selectAllAcross ? totalCount : selectedIds.size
  const anySelection = selectedCount > 0
  const allOnPageSelected = pageCount > 0 && selectedOnPageCount === pageCount
  const someOnPageSelected = selectedOnPageCount > 0 && selectedOnPageCount < pageCount
  const selectAllAcrossDisabled = isDocumentMode && documentQuery.isTruncated

  const updateMutation = useUpdateFlashcardMutation()
  const bulkUpdateMutation = useUpdateFlashcardsBulkMutation()
  const resetSchedulingMutation = useResetFlashcardSchedulingMutation()
  const deleteMutation = useDeleteFlashcardMutation()

  const compactSchedulingLabels = React.useCallback(
    (card: Flashcard) => ({
      memoryStrength: t("option:flashcards.schedulingMemoryStrengthCompact", {
        defaultValue: "Memory {{value}}",
        value: card.ef.toFixed(2)
      }),
      nextReviewGap: t("option:flashcards.schedulingNextReviewGapCompact", {
        defaultValue: "Next gap {{count}}d",
        count: Math.max(0, card.interval_days)
      }),
      recallRuns: t("option:flashcards.schedulingRecallRunsCompact", {
        defaultValue: "Recall runs {{count}}",
        count: Math.max(0, card.repetitions)
      }),
      relearns: t("option:flashcards.schedulingRelearnsCompact", {
        defaultValue: "Relearns {{count}}",
        count: Math.max(0, card.lapses)
      })
    }),
    [t]
  )

  const expandedSchedulingLabels = React.useCallback(
    (card: Flashcard) => ({
      memoryStrength: t("option:flashcards.schedulingMemoryStrengthExpanded", {
        defaultValue: "Memory strength {{value}}",
        value: card.ef.toFixed(2)
      }),
      nextReviewGap: t("option:flashcards.schedulingNextReviewGapExpanded", {
        defaultValue: "Next review gap {{count}}d",
        count: Math.max(0, card.interval_days)
      }),
      recallRuns: t("option:flashcards.schedulingRecallRunsExpanded", {
        defaultValue: "Recall runs {{count}}",
        count: Math.max(0, card.repetitions)
      }),
      relearns: t("option:flashcards.schedulingRelearnsExpanded", {
        defaultValue: "Relearns {{count}}",
        count: Math.max(0, card.lapses)
      })
    }),
    [t]
  )

  // Reset focused index when page or filters change
  React.useEffect(() => {
    setFocusedIndex(-1)
  }, [page, pageSize, listDensity, mDeckId, mQuery, mTags, mDue, mSort])

  async function fetchAllItemsAcrossFilters(): Promise<Flashcard[]> {
    const items: Flashcard[] = []
    const maxPerPage = 1000
    const MAX_ITEMS_CAP = 10000
    const primaryTag = mTags[0]
    const remainingTags = new Set(mTags.slice(1).map((tag) => tag.toLowerCase()))
    const total = totalCount
    if (total > MAX_ITEMS_CAP) {
      message.warning(
        t("option:flashcards.bulkLimitWarning", {
          defaultValue: "Operation limited to first {{limit}} items.",
          limit: MAX_ITEMS_CAP
        })
      )
    }
    for (
      let offset = 0;
      offset < Math.min(total, MAX_ITEMS_CAP);
      offset += maxPerPage
    ) {
      const res = await listFlashcards({
        deck_id: mDeckId ?? undefined,
        q: mQuery || undefined,
        tag: primaryTag,
        due_status: mDue,
        workspace_id: selectedWorkspaceId ?? undefined,
        limit: maxPerPage,
        offset,
        order_by: getManageServerOrderBy(mSort),
        include_workspace_items: selectedWorkspaceId == null ? showWorkspaceDecks : false
      })
      const chunkItems = res.items || []
      if (remainingTags.size === 0) {
        items.push(...chunkItems)
      } else {
        items.push(
          ...chunkItems.filter((card) => {
            const tagSet = new Set(
              (card.tags || []).map((tag) => String(tag || "").trim().toLowerCase())
            )
            for (const tag of remainingTags) {
              if (!tagSet.has(tag)) return false
            }
            return true
          })
        )
      }
      if (!res.items || res.items.length < maxPerPage) break
    }
    return items
  }

  async function getSelectedItems(): Promise<Flashcard[]> {
    if (!selectAllAcross) {
      return visibleItems.filter((i) => selectedIds.has(i.uuid))
    }
    const all = await fetchAllItemsAcrossFilters()
    return all
  }

  // Move modal state
  const [moveOpen, setMoveOpen] = React.useState(false)
  const [moveCard, setMoveCard] = React.useState<Flashcard | null>(null)
  const [moveDeckId, setMoveDeckId] = React.useState<number | null>(null)
  const [bulkTagOpen, setBulkTagOpen] = React.useState(false)
  const [bulkTagMode, setBulkTagMode] = React.useState<"add" | "remove">("add")
  const [bulkTagInput, setBulkTagInput] = React.useState("")

  const openBulkMove = () => {
    if (!anySelection) return
    setMoveCard(null)
    setMoveDeckId(null)
    setMoveOpen(true)
  }

  const openDeckScopeEditor = () => {
    if (!selectedDeck) return
    deckScopeForm.setFieldsValue({
      workspaceId: selectedDeck.workspace_id ?? "",
      parentDeckId: selectedDeck.parent_deck_id ?? undefined
    })
    setDeckScopeOpen(true)
  }

  const closeDeckScopeEditor = () => {
    setDeckScopeOpen(false)
    deckScopeForm.resetFields()
  }

  const openBulkTagEditor = (mode: "add" | "remove") => {
    if (!anySelection) return
    setBulkTagMode(mode)
    setBulkTagInput("")
    setBulkTagOpen(true)
  }

  const submitDeckScopeEdit = async () => {
    if (!selectedDeck) return
    try {
      const values = await deckScopeForm.validateFields()
      const workspaceId = typeof values.workspaceId === "string" ? values.workspaceId.trim() : ""
      const parentDeckId = typeof values.parentDeckId === "number" ? values.parentDeckId : null
      await updateDeckMutation.mutateAsync({
        deckId: selectedDeck.id,
        update: {
          workspace_id: workspaceId.length > 0 ? workspaceId : null,
          parent_deck_id: parentDeckId,
          expected_version: selectedDeck.version
        }
      })
      message.success(
        t("option:flashcards.deckScopeUpdated", {
          defaultValue: "Deck scope updated."
        })
      )
      closeDeckScopeEditor()
    } catch (error: unknown) {
      if (typeof error === "object" && error && "errorFields" in error) return
      reportUiError(
        error,
        "updating deck scope",
        t("option:flashcards.deckScopeUpdateFailed", {
          defaultValue: "Failed to update deck scope."
        })
      )
    }
  }

  const submitBulkTagEdit = async () => {
    try {
      const parsedTags = Array.from(
        new Set(
          bulkTagInput
            .split(/[,\s]+/)
            .map((tag) => tag.trim())
            .filter(Boolean)
        )
      )
      if (parsedTags.length === 0) {
        message.error(
          t("option:flashcards.bulkTagValidation", {
            defaultValue: "Enter at least one tag."
          })
        )
        return
      }

      const selectedItems = await getSelectedItems()
      if (selectedItems.length === 0) {
        message.info(
          t("option:flashcards.bulkTagNoSelection", {
            defaultValue: "No cards selected."
          })
        )
        return
      }

      let changedCount = 0
      await processInChunks(
        selectedItems,
        BULK_MUTATION_CHUNK_SIZE,
        async (chunk) => {
          const results = await Promise.allSettled(
            chunk.map(async (card) => {
              const full = await getFlashcard(card.uuid)
              const currentTags = Array.isArray(full.tags) ? full.tags : []
              const currentTagSet = new Set(currentTags.map((tag) => tag.trim()))
              let nextTags = currentTags
              if (bulkTagMode === "add") {
                for (const tag of parsedTags) currentTagSet.add(tag)
                nextTags = Array.from(currentTagSet)
              } else {
                const removeSet = new Set(parsedTags.map((tag) => tag.toLowerCase()))
                nextTags = currentTags.filter(
                  (tag) => !removeSet.has(String(tag || "").trim().toLowerCase())
                )
              }
              const changed =
                nextTags.length !== currentTags.length ||
                nextTags.some((tag, index) => tag !== currentTags[index])
              if (!changed) return
              await updateFlashcard(card.uuid, {
                tags: nextTags,
                expected_version: full.version
              })
              changedCount += 1
            })
          )
          const failures = results.filter((result) => result.status === "rejected")
          if (failures.length > 0) {
            console.warn(`${failures.length} bulk tag updates failed in chunk`)
          }
        }
      )

      clearSelection()
      setBulkTagOpen(false)
      setBulkTagInput("")
      await qc.invalidateQueries({ queryKey: ["flashcards:list"] })

      if (changedCount === 0) {
        message.info(
          t("option:flashcards.bulkTagNoChanges", {
            defaultValue: "No tag changes were needed."
          })
        )
        return
      }

      message.success(
        bulkTagMode === "add"
          ? t("option:flashcards.bulkTagAddSuccess", {
              defaultValue: "Added tags on {{count}} cards.",
              count: changedCount
            })
          : t("option:flashcards.bulkTagRemoveSuccess", {
              defaultValue: "Removed tags on {{count}} cards.",
              count: changedCount
            })
      )
    } catch (error: unknown) {
      message.error(error instanceof Error ? error.message : "Bulk tag update failed")
    }
  }

  const executeBulkDelete = React.useCallback(
    async (
      items: Flashcard[],
      options?: { showProgress?: boolean; showSuccessMessage?: boolean; clearSelection?: boolean }
    ) => {
      const showProgress = options?.showProgress ?? true
      const showSuccessMessage = options?.showSuccessMessage ?? true
      const shouldClearSelection = options?.clearSelection ?? true
      const total = items.length
      if (showProgress) {
        setBulkProgress({
          open: true,
          current: 0,
          total,
          action: t("option:flashcards.bulkProgressDeleting", {
            defaultValue: "Deleting cards"
          })
        })
      }
      try {
        let processed = 0
        const failedIds = new Set<string>()
        await processInChunks(items, BULK_MUTATION_CHUNK_SIZE, async (chunk) => {
          const results = await Promise.allSettled(
            chunk.map((c) => deleteFlashcard(c.uuid, c.version))
          )
          let chunkFailures = 0
          results.forEach((result, index) => {
            if (result.status === "rejected") {
              failedIds.add(chunk[index].uuid)
              chunkFailures += 1
            }
          })
          if (chunkFailures > 0) {
            console.warn(`${chunkFailures} deletions failed in chunk`)
          }
          processed += chunk.length
          if (showProgress) {
            setBulkProgress((prev) =>
              prev ? { ...prev, current: Math.min(processed, total) } : null
            )
          }
        })
        const failedCount = failedIds.size
        const successCount = Math.max(0, total - failedCount)
        if (failedCount > 0) {
          if (showSuccessMessage) {
            message.warning(
              t("option:flashcards.bulkDeleteResult", {
                defaultValue: "{{success}} deleted · {{failed}} failed",
                success: successCount,
                failed: failedCount
              })
            )
          }
          if (shouldClearSelection) clearSelection()
        } else {
          if (showSuccessMessage) {
            message.success(t("common:deleted", { defaultValue: "Deleted" }))
          }
          if (shouldClearSelection) clearSelection()
        }
        await qc.invalidateQueries({ queryKey: ["flashcards:list"] })
      } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : "Bulk delete failed"
        message.error(errorMessage)
      } finally {
        if (showProgress) {
          setBulkProgress(null)
        }
      }
    },
    [clearSelection, message, qc, t]
  )

  const removePendingDeletions = React.useCallback(
    (uuids: string[]) => {
      if (!uuids.length) return
      updatePendingDeletions((prev) => {
        let changed = false
        const next = { ...prev }
        uuids.forEach((uuid) => {
          if (next[uuid]) {
            delete next[uuid]
            changed = true
          }
        })
        return changed ? next : prev
      })
    },
    [updatePendingDeletions]
  )

  const undoDeletionBatch = React.useCallback(
    (batchId: string) => {
      const batch = pendingDeletionBatchesRef.current.get(batchId)
      if (!batch) return
      window.clearTimeout(batch.timeoutId)
      pendingDeletionBatchesRef.current.delete(batchId)
      removePendingDeletions(Array.from(batch.uuids))
    },
    [removePendingDeletions]
  )

  const finalizeDeletionBatch = React.useCallback(
    async (batchId: string) => {
      const batch = pendingDeletionBatchesRef.current.get(batchId)
      if (!batch) return
      pendingDeletionBatchesRef.current.delete(batchId)

      const pending = pendingDeletionsRef.current
      const cardsToDelete = Array.from(batch.uuids)
        .map((uuid) => pending[uuid]?.card)
        .filter((card): card is Flashcard => !!card)

      try {
        if (cardsToDelete.length > 0) {
          await executeBulkDelete(cardsToDelete, {
            showProgress: false,
            showSuccessMessage: true,
            clearSelection: false
          })
        }
      } finally {
        removePendingDeletions(Array.from(batch.uuids))
      }
    },
    [executeBulkDelete, removePendingDeletions]
  )

  const queueDeletionBatch = React.useCallback(
    (cards: Flashcard[], notification: { title: string; description?: string }) => {
      if (!cards.length) return
      const batchId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(16).slice(2)}`
      const expiresAt = Date.now() + DELETE_UNDO_MS
      const uuids = new Set(cards.map((card) => card.uuid))

      updatePendingDeletions((prev) => {
        const next = { ...prev }
        cards.forEach((card) => {
          next[card.uuid] = {
            card,
            expiresAt,
            batchId
          }
        })
        return next
      })

      const timeoutId = window.setTimeout(() => {
        finalizeDeletionBatch(batchId)
      }, DELETE_UNDO_MS)
      pendingDeletionBatchesRef.current.set(batchId, { uuids, timeoutId })

      showUndoNotification({
        title: notification.title,
        description: notification.description,
        duration: DELETE_UNDO_SECONDS,
        onUndo: async () => {
          undoDeletionBatch(batchId)
        }
      })
    },
    [finalizeDeletionBatch, showUndoNotification, undoDeletionBatch, updatePendingDeletions]
  )

  const undoSinglePendingDeletion = React.useCallback(
    (uuid: string) => {
      const pending = pendingDeletionsRef.current[uuid]
      if (!pending) return
      const batch = pendingDeletionBatchesRef.current.get(pending.batchId)
      if (batch) {
        batch.uuids.delete(uuid)
        if (batch.uuids.size === 0) {
          window.clearTimeout(batch.timeoutId)
          pendingDeletionBatchesRef.current.delete(pending.batchId)
        }
      }
      removePendingDeletions([uuid])
    },
    [removePendingDeletions]
  )

  const undoAllPendingDeletions = React.useCallback(() => {
    pendingDeletionBatchesRef.current.forEach((batch) => {
      window.clearTimeout(batch.timeoutId)
    })
    pendingDeletionBatchesRef.current.clear()
    updatePendingDeletions(() => ({}))
  }, [updatePendingDeletions])

  const handleBulkDelete = async () => {
    const toDelete = await getSelectedItems()
    if (!toDelete.length) return

    const count = toDelete.length
    const LARGE_DELETE_THRESHOLD = 100

    if (count > LARGE_DELETE_THRESHOLD) {
      setBulkDeleteCount(count)
      setPendingDeleteItems(toDelete)
      setBulkDeleteInput("")
      setBulkDeleteConfirmOpen(true)
    } else {
      const ok = await confirmDanger({
        title: t("common:confirmTitle", { defaultValue: "Please confirm" }),
        content: t("option:flashcards.bulkDeleteConfirm", {
          defaultValue:
            "Delete {{count}} selected cards? You can undo for {{seconds}} seconds.",
          count,
          seconds: DELETE_UNDO_SECONDS
        }),
        okText: t("common:delete", { defaultValue: "Delete" }),
        cancelText: t("common:cancel", { defaultValue: "Cancel" })
      })
      if (!ok) return
      const undoHint = t("option:flashcards.deleteUndoHint", {
        defaultValue: "Undo within {{seconds}}s to cancel deletion.",
        seconds: DELETE_UNDO_SECONDS
      })
      queueDeletionBatch(toDelete, {
        title: t("option:flashcards.cardsDeleted", {
          defaultValue: "{{count}} cards deleted",
          count
        }),
        description: undoHint
      })
      clearSelection()
    }
  }

  const confirmLargeBulkDelete = async () => {
    const items = pendingDeleteItems
    setBulkDeleteConfirmOpen(false)
    setBulkDeleteInput("")
    setPendingDeleteItems([])
    setBulkDeleteCount(0)
    if (!items.length) return
    const undoHint = t("option:flashcards.deleteUndoHint", {
      defaultValue: "Undo within {{seconds}}s to cancel deletion.",
      seconds: DELETE_UNDO_SECONDS
    })
    queueDeletionBatch(items, {
      title: t("option:flashcards.cardsDeleted", {
        defaultValue: "{{count}} cards deleted",
        count: items.length
      }),
      description: undoHint
    })
    clearSelection()
  }

  const handleExportSelected = async () => {
    try {
      const items = await getSelectedItems()
      if (!items.length) return
      const header = ["Deck", "Front", "Back", "Tags", "Notes"]
      const decks = decksQuery.data || []
      const nameById = new Map<number, string>()
      decks.forEach((d) => nameById.set(d.id, d.name))
      const rows = items.map((i) =>
        [
          i.deck_id != null
            ? nameById.get(i.deck_id) || `Deck ${i.deck_id}`
            : "",
          i.front || "",
          i.back || "",
          Array.isArray(i.tags) ? i.tags.join(" ") : "",
          i.notes || ""
        ].join("\t")
      )
      const text = [header.join("\t"), ...rows].join("\n")
      const blob = new Blob([text], {
        type: "text/tab-separated-values;charset=utf-8"
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "flashcards-selected.tsv"
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Export failed"
      message.error(errorMessage)
    }
  }

  // Edit drawer
  const [editOpen, setEditOpen] = React.useState(false)
  const [editing, setEditing] = React.useState<Flashcard | null>(null)

  // Create drawer
  const [createOpen, setCreateOpen] = React.useState(false)

  React.useEffect(() => {
    if (!openCreateSignal) return
    setViewMode("cards")
    setCreateOpen(true)
  }, [openCreateSignal])

  // Quick actions: duplicate
  const duplicateCard = async (card: Flashcard) => {
    try {
      await createFlashcard({
        deck_id: card.deck_id ?? undefined,
        front: card.front,
        back: card.back,
        notes: card.notes || undefined,
        extra: card.extra || undefined,
        is_cloze: card.is_cloze,
        tags: card.tags || undefined,
        model_type: card.model_type,
        reverse: card.reverse
      })
      await qc.invalidateQueries({ queryKey: ["flashcards:list"] })
      message.success(t("common:created", { defaultValue: "Created" }))
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Duplicate failed"
      message.error(errorMessage)
    }
  }

  const openMove = async (card: Flashcard) => {
    try {
      setMoveCard(card)
      setMoveDeckId(card.deck_id ?? null)
      setMoveOpen(true)
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Failed to load card"
      message.error(errorMessage)
    }
  }

  const submitMove = async () => {
    try {
      const undoSnapshots: MoveUndoSnapshot[] = []
      let movedCount = 0
      if (moveCard) {
        const full = await getFlashcard(moveCard.uuid)
        const previousDeckId = full.deck_id ?? null
        const targetDeckId = moveDeckId ?? null
        if (previousDeckId !== targetDeckId) {
          await updateFlashcard(moveCard.uuid, {
            deck_id: targetDeckId,
            expected_version: full.version
          })
          undoSnapshots.push({
            uuid: moveCard.uuid,
            previousDeckId
          })
          movedCount += 1
        }
      } else {
        if (moveDeckId == null) {
          message.error(
            t("option:flashcards.bulkMoveSelectDeck", {
              defaultValue: "Select a target deck before moving cards."
            })
          )
          return
        }
        const toMove = await getSelectedItems()
        if (toMove.length) {
          await processInChunks(
            toMove,
            BULK_MUTATION_CHUNK_SIZE,
            async (chunk) => {
              const results = await Promise.allSettled(
                chunk.map(async (c) => {
                  const full = await getFlashcard(c.uuid)
                  const previousDeckId = full.deck_id ?? null
                  if (previousDeckId === moveDeckId) {
                    return
                  }
                  await updateFlashcard(c.uuid, {
                    deck_id: moveDeckId,
                    expected_version: full.version
                  })
                  undoSnapshots.push({
                    uuid: c.uuid,
                    previousDeckId
                  })
                  movedCount += 1
                })
              )
              const failures = results.filter((r) => r.status === "rejected")
              if (failures.length > 0) {
                console.warn(`${failures.length} move updates failed in chunk`)
              }
            }
          )
        }
        clearSelection()
      }
      setMoveOpen(false)
      setMoveCard(null)
      setMoveDeckId(null)
      await qc.invalidateQueries({ queryKey: ["flashcards:list"] })
      if (movedCount === 0) {
        message.info(
          t("option:flashcards.bulkMoveNoChanges", {
            defaultValue: "No cards needed moving."
          })
        )
        return
      }
      showUndoNotification({
        title: moveCard
          ? t("option:flashcards.cardMoved", {
              defaultValue: "Card moved"
            })
          : t("option:flashcards.cardsMoved", {
              defaultValue: "{{count}} cards moved",
              count: movedCount
            }),
        description: t("option:flashcards.moveUndoHint", {
          defaultValue: "Undo within {{seconds}}s to revert this move.",
          seconds: DELETE_UNDO_SECONDS
        }),
        duration: DELETE_UNDO_SECONDS,
        onUndo: async () => {
          await processInChunks(
            undoSnapshots,
            BULK_MUTATION_CHUNK_SIZE,
            async (chunk) => {
              const results = await Promise.allSettled(
                chunk.map(async (snapshot) => {
                  const latest = await getFlashcard(snapshot.uuid)
                  await updateFlashcard(snapshot.uuid, {
                    deck_id: snapshot.previousDeckId,
                    expected_version: latest.version
                  })
                })
              )
              const failed = results.filter((r) => r.status === "rejected")
              if (failed.length > 0) {
                throw new Error(
                  t("option:flashcards.moveUndoPartialFailure", {
                    defaultValue: "Some cards could not be restored."
                  })
                )
              }
            }
          )
          await qc.invalidateQueries({ queryKey: ["flashcards:list"] })
        }
      })
      message.success(t("common:updated", { defaultValue: "Updated" }))
    } catch (e: unknown) {
      reportUiError(
        e,
        "moving cards",
        t("option:flashcards.moveFailed", {
          defaultValue: "Move failed."
        })
      )
    }
  }

  const openEdit = (card: Flashcard) => {
    setEditing(card)
    setEditOpen(true)
  }

  // Keyboard navigation for Cards tab
  useCardsKeyboardNav({
    enabled: isActive && viewMode === "cards" && !isDocumentMode && !editOpen && !createOpen && !moveOpen,
    itemCount: pageCount,
    focusedIndex,
    onFocusChange: setFocusedIndex,
    onEdit: (index) => {
      const card = visibleItems[index]
      if (card) openEdit(card)
    },
    onToggleSelect: (index) => {
      const card = visibleItems[index]
      if (card && !selectAllAcross) {
        toggleSelect(card.uuid, !selectedIds.has(card.uuid))
      }
    },
    onDelete: async (index) => {
      const card = visibleItems[index]
      if (card) {
        // Open edit drawer and trigger delete from there for consistency
        setEditing(card)
        setEditOpen(true)
      }
    }
  })

  const doUpdate = async (values: FlashcardUpdate) => {
    try {
      if (!editing) return
      const editingUuid = editing.uuid
      const previous = await getFlashcard(editingUuid)
      await updateMutation.mutateAsync({
        uuid: editingUuid,
        update: values
      })
      showUndoNotification({
        title: t("option:flashcards.cardUpdated", {
          defaultValue: "Card updated"
        }),
        description: t("option:flashcards.editUndoHint", {
          defaultValue: "Undo within {{seconds}}s to restore previous content.",
          seconds: DELETE_UNDO_SECONDS
        }),
        duration: DELETE_UNDO_SECONDS,
        onUndo: async () => {
          const latest = await getFlashcard(editingUuid)
          await updateMutation.mutateAsync({
            uuid: editingUuid,
            update: {
              deck_id: previous.deck_id ?? null,
              front: previous.front,
              back: previous.back,
              notes: previous.notes ?? null,
              extra: previous.extra ?? null,
              tags: previous.tags ?? [],
              is_cloze: previous.is_cloze,
              model_type: previous.model_type,
              reverse: previous.reverse,
              expected_version: latest.version
            }
          })
        }
      })
      message.success(t("common:updated", { defaultValue: "Updated" }))
      setEditOpen(false)
      setEditing(null)
    } catch (e: unknown) {
      if (typeof e === "object" && e && "errorFields" in e) {
        const { errorFields } = e as { errorFields?: unknown }
        if (errorFields) return
      }
      const mapped = mapFlashcardsUiError(e, {
        operation: "updating this card",
        fallback: t("option:flashcards.updateFailed", {
          defaultValue: "Update failed."
        })
      })
      if (mapped.code === "FLASHCARDS_VERSION_CONFLICT" && editing) {
        try {
          const latest = await getFlashcard(editing.uuid)
          setEditing(latest)
          message.warning(
            t("option:flashcards.updateConflictReloaded", {
              defaultValue:
                "This card changed elsewhere. Reloaded latest data; review and save again. [FLASHCARDS_VERSION_CONFLICT]"
            })
          )
          void trackFlashcardsErrorRecoveryTelemetry({
            type: "flashcards_recovered_by_reload",
            surface: "cards",
            operation: "updating this card",
            error_code: mapped.code
          })
          return
        } catch (reloadError: unknown) {
          reportUiError(
            reloadError,
            "reloading the latest card state",
            t("option:flashcards.cardReloadFailed", {
              defaultValue: "Failed to reload card."
            })
          )
          return
        }
      }
      console.warn("[flashcards:error]", {
        code: mapped.code,
        status: mapped.status,
        operation: "updating this card",
        raw: mapped.rawMessage
      })
      message.error(formatFlashcardsUiErrorMessage(mapped))
    }
  }

  const doDelete = async () => {
    try {
      if (!editing) return
      if (typeof editing.version !== "number") {
        message.error("Missing version; reload and try again")
        return
      }
      const cardToDelete = { ...editing }

      // Close drawer and mark as pending deletion (soft-delete)
      setEditOpen(false)
      setEditing(null)

      // Show undo notification with 30 second timeout
      const preview = cardToDelete.front.slice(0, 60)
      const undoHint = t("option:flashcards.deleteUndoHint", {
        defaultValue: "Undo within {{seconds}}s to cancel deletion.",
        seconds: DELETE_UNDO_SECONDS
      })
      queueDeletionBatch([cardToDelete], {
        title: t("option:flashcards.cardDeleted", { defaultValue: "Card deleted" }),
        description: preview
          ? `${preview}${cardToDelete.front.length > 60 ? "…" : ""} · ${undoHint}`
          : undoHint
      })
    } catch (e: unknown) {
      reportUiError(
        e,
        "deleting this card",
        t("option:flashcards.deleteFailed", {
          defaultValue: "Delete failed."
        })
      )
    }
  }

  const doResetScheduling = async () => {
    try {
      if (!editing) return
      await resetSchedulingMutation.mutateAsync({
        uuid: editing.uuid,
        expectedVersion: editing.version
      })
      message.success(
        t("option:flashcards.schedulingResetSuccess", {
          defaultValue: "Scheduling reset to new-card defaults."
        })
      )
      setEditOpen(false)
      setEditing(null)
    } catch (e: unknown) {
      reportUiError(
        e,
        "resetting scheduling",
        t("option:flashcards.resetSchedulingFailed", {
          defaultValue: "Reset scheduling failed."
        })
      )
    }
  }

  const manageSortOptions = React.useMemo(() => {
    const options: Array<{ value: ManageSortBy; label: string }> = [
      {
        value: "due",
        label: t("option:flashcards.sortDueDate", {
          defaultValue: "Sort: Due date"
        })
      },
      {
        value: "created",
        label: t("option:flashcards.sortCreatedDate", {
          defaultValue: "Sort: Created"
        })
      },
      {
        value: "ease",
        label: t("option:flashcards.sortEaseFactor", {
          defaultValue: "Sort: Ease factor"
        })
      },
      {
        value: "last_reviewed",
        label: t("option:flashcards.sortLastReviewed", {
          defaultValue: "Sort: Last reviewed"
        })
      },
      {
        value: "front_alpha",
        label: t("option:flashcards.sortFrontAlpha", {
          defaultValue: "Sort: Front (A-Z)"
        })
      }
    ]

    if (!isDocumentMode) return options

    const supported = new Set(documentQuery.supportedSorts)
    return options.filter((option) => supported.has(option.value as "due" | "created"))
  }, [documentQuery.supportedSorts, isDocumentMode, t])

  return (
    <>
      <div>
        <div
          className="mb-3 flex items-center justify-between gap-2"
          data-testid="flashcards-manage-topbar"
        >
          <div className="flex items-center gap-3">
            <Segmented
              value={viewMode}
              onChange={(value) => {
                setViewMode(value as "cards" | "trash")
              }}
              options={[
                {
                  label: t("option:flashcards.cards", { defaultValue: "Cards" }),
                  value: "cards"
                },
                {
                  label: (
                    <span className="inline-flex items-center gap-2">
                      {t("option:flashcards.trash", { defaultValue: "Trash" })}
                      {pendingDeletionCount > 0 && (
                        <Badge count={pendingDeletionCount} size="small" />
                      )}
                    </span>
                  ),
                  value: "trash"
                }
              ]}
            />
            {/* Keyboard shortcut hint */}
            {viewMode === "cards" && (
              <div
                className="hidden md:flex items-center gap-1.5"
                data-testid="flashcards-manage-shortcut-chips"
              >
                <Tooltip
                  title={t("option:flashcards.keyboardShortcutsHint", {
                    defaultValue: "Press ? for keyboard shortcuts"
                  })}
                >
                  <span className="inline-flex items-center gap-1 text-xs text-text-muted cursor-help">
                    <Keyboard className="size-3.5" aria-hidden="true" />
                    <span>?</span>
                  </span>
                </Tooltip>
                {shortcutHintDensity === "expanded" && (
                  <>
                    <Tag className="!m-0 text-[11px]">
                      {t("option:flashcards.shortcutChipManageNav", {
                        defaultValue: "J/K Navigate"
                      })}
                    </Tag>
                    <Tag className="!m-0 text-[11px]">
                      {t("option:flashcards.shortcutChipManageEdit", {
                        defaultValue: "Enter Edit"
                      })}
                    </Tag>
                    <Tag className="!m-0 text-[11px]">
                      {t("option:flashcards.shortcutChipManageSelect", {
                        defaultValue: "Space Select"
                      })}
                    </Tag>
                    <Tag className="!m-0 text-[11px]">
                      {t("option:flashcards.shortcutChipManageDelete", {
                        defaultValue: "Delete Remove"
                      })}
                    </Tag>
                  </>
                )}
                {shortcutHintDensity === "compact" && (
                  <Tag className="!m-0 text-[11px]">
                    {t("option:flashcards.shortcutChipManageCompact", {
                      defaultValue: "J/K · Enter · Space · Delete"
                    })}
                  </Tag>
                )}
                <Button
                  type="link"
                  size="small"
                  className="!h-auto !px-0 text-xs"
                  onClick={cycleShortcutHintDensity}
                  data-testid="flashcards-manage-shortcut-hints-toggle"
                >
                  {shortcutHintToggleLabel}
                </Button>
              </div>
            )}
          </div>
          {viewMode === "trash" && pendingDeletionCount > 0 && (
            <Button size="small" onClick={undoAllPendingDeletions}>
              {t("option:flashcards.trashUndoAll", { defaultValue: "Undo all" })}
            </Button>
          )}
        </div>
        {viewMode === "trash" && (
          <Text type="secondary" className="block text-xs mb-2">
            {t("option:flashcards.trashEmptyDescription", {
              defaultValue: "Deleted cards appear here for 30 seconds."
            })}
          </Text>
        )}

        {/* Simplified Filter UI */}
        {viewMode === "cards" && (
        <div className="mb-3 space-y-3">
          {/* Primary filters: Search + Deck (always visible) */}
          <div className="flex items-center gap-2 flex-wrap">
            <Input.Search
              placeholder={t("common:search", { defaultValue: "Search" })}
              allowClear
              onSearch={() => {
                setMQuery(mQueryInput)
                setPage(1)
              }}
              value={mQueryInput}
              onChange={(e) => setMQueryInput(e.target.value)}
              className="max-w-64"
              data-testid="flashcards-manage-search"
            />
            <Select<number>
              placeholder={t("option:flashcards.deck", { defaultValue: "All decks" })}
              allowClear
              loading={decksQuery.isLoading}
              value={mDeckId ?? undefined}
              onChange={(v) => {
                setMDeckId(v ?? undefined)
                setPage(1)
              }}
              className="min-w-44"
              data-testid="flashcards-manage-deck-select"
              options={(decksQuery.data || []).map((d) => ({
                label: formatDeckHierarchyLabel(d, deckMap, `Deck ${d.id}`),
                value: d.id
              }))}
            />
            <Button
              onClick={openDeckScopeEditor}
              disabled={!selectedDeck}
              data-testid="flashcards-manage-move-scope"
            >
              {t("option:flashcards.moveScope", { defaultValue: "Move scope" })}
            </Button>
            <Checkbox
              checked={showWorkspaceDecks}
              onChange={(event) => {
                setShowWorkspaceDecks(event.target.checked)
                if (!event.target.checked) {
                  setSelectedWorkspaceId(null)
                }
                setPage(1)
              }}
              aria-label={t("option:flashcards.showWorkspaceDecks", {
                defaultValue: "Show workspace decks"
              })}
              data-testid="flashcards-manage-show-workspace-decks"
            >
              {t("option:flashcards.showWorkspaceDecks", {
                defaultValue: "Show workspace decks"
              })}
            </Checkbox>
            <Select<string>
              allowClear
              showSearch
              placeholder={t("option:flashcards.filterWorkspace", {
                defaultValue: "Filter workspace"
              })}
              value={selectedWorkspaceId ?? undefined}
              onChange={(value) => {
                setSelectedWorkspaceId(value ?? null)
                setPage(1)
              }}
              disabled={!showWorkspaceDecks && selectedWorkspaceId == null}
              options={workspaceFilterOptions}
              className="min-w-44"
              data-testid="flashcards-manage-workspace-filter"
            />
            {/* Tag filter in popover */}
            <Popover
              trigger="click"
              placement="bottomLeft"
              content={
                <div className="w-72 space-y-2">
                  <div className="text-sm font-medium text-text-muted">
                    {t("option:flashcards.filterByTag", {
                      defaultValue: "Filter by tag"
                    })}
                  </div>
                  <Input
                    placeholder={t("option:flashcards.tagPlaceholder", {
                      defaultValue: "Type a tag and press Enter"
                    })}
                    value={mTagInput}
                    onChange={(e) => {
                      setMTagInput(e.target.value)
                    }}
                    onPressEnter={() => addTagFilter(mTagInput)}
                    onKeyDown={(event) => {
                      if (event.key === ",") {
                        event.preventDefault()
                        addTagFilter(mTagInput)
                      }
                    }}
                    allowClear
                    data-testid="flashcards-manage-tag-input"
                  />
                  <div className="flex flex-wrap gap-1" data-testid="flashcards-manage-tag-selected">
                    {mTags.length > 0 ? (
                      mTags.map((tag) => (
                        <Tag
                          key={tag.toLowerCase()}
                          closable
                          onClose={(event) => {
                            event.preventDefault()
                            removeTagFilter(tag)
                          }}
                          className="!m-0"
                        >
                          {tag}
                        </Tag>
                      ))
                    ) : (
                      <Text type="secondary" className="text-xs">
                        {t("option:flashcards.tagFilterMatchAllHint", {
                          defaultValue: "Add one or more tags. Cards must match all selected tags."
                        })}
                      </Text>
                    )}
                  </div>
                  {filteredTagSuggestions.length > 0 && (
                    <div className="flex flex-wrap gap-1" data-testid="flashcards-manage-tag-suggestions">
                      {filteredTagSuggestions.map((tag) => (
                        <Button
                          key={tag.toLowerCase()}
                          size="small"
                          type="default"
                          onClick={() => addTagFilter(tag)}
                          className="!h-6 !px-2 !text-xs"
                        >
                          {tag}
                        </Button>
                      ))}
                    </div>
                  )}
                </div>
              }
            >
              <Badge dot={mTags.length > 0} offset={[-4, 4]}>
                <Button icon={<Filter className="size-4" />}>
                  {t("option:flashcards.moreFilters", { defaultValue: "More" })}
                </Button>
              </Badge>
            </Popover>
            {hasActiveFilters && (
              <Button size="small" type="link" onClick={clearAllFilters}>
                {t("option:flashcards.clearFilters", { defaultValue: "Clear all" })}
              </Button>
            )}
          </div>

          {mTags.length > 0 && (
            <div
              className="flex flex-wrap items-center gap-1.5"
              data-testid="flashcards-manage-active-tag-filters"
            >
              <Text type="secondary" className="text-xs">
                {t("option:flashcards.activeTagFilters", {
                  defaultValue: "Tag filters:"
                })}
              </Text>
              {mTags.map((tag) => (
                <Tag
                  key={`active-${tag.toLowerCase()}`}
                  closable
                  onClose={(event) => {
                    event.preventDefault()
                    removeTagFilter(tag)
                  }}
                  className="!m-0"
                >
                  {tag}
                </Tag>
              ))}
            </div>
          )}

          {/* Due status as segmented control + density toggle */}
          <div className="flex items-center justify-between gap-2">
            <Segmented
              value={mDue}
              onChange={(value) => {
                setMDue(value as DueStatus)
                setPage(1)
              }}
              data-testid="flashcards-manage-due-status"
              options={[
                {
                  label: t("option:flashcards.dueAll", { defaultValue: "All" }),
                  value: "all"
                },
                {
                  label: t("option:flashcards.dueDue", { defaultValue: "Due" }),
                  value: "due"
                },
                {
                  label: t("option:flashcards.dueNew", { defaultValue: "New" }),
                  value: "new"
                },
                {
                  label: t("option:flashcards.dueLearning", { defaultValue: "Learning" }),
                  value: "learning"
                }
              ]}
            />
            <div className="flex items-center gap-2">
              <div data-testid="flashcards-manage-sort-select">
                <Select<ManageSortBy>
                  value={mSort}
                  className="min-w-44"
                  onChange={(value) => {
                    setMSort(value)
                    setPage(1)
                  }}
                  options={manageSortOptions}
                />
              </div>
              <div
                className="flex items-center gap-1 rounded-lg border border-border bg-surface px-1 py-1"
                aria-label={t("option:flashcards.presentationMode", {
                  defaultValue: "Presentation mode"
                })}
              >
                <Tooltip
                  title={t("option:flashcards.compactView", { defaultValue: "Compact view" })}
                >
                  <Button
                    type={listDensity === "compact" ? "default" : "text"}
                    icon={<ListIcon className="size-4" />}
                    onClick={() => setPresentationMode("compact")}
                    data-testid="flashcards-density-toggle-compact"
                  />
                </Tooltip>
                <Tooltip
                  title={t("option:flashcards.expandedView", { defaultValue: "Expanded view" })}
                >
                  <Button
                    type={listDensity === "expanded" ? "default" : "text"}
                    icon={<LayoutList className="size-4" />}
                    onClick={() => setPresentationMode("expanded")}
                    data-testid="flashcards-density-toggle"
                  />
                </Tooltip>
                <Tooltip
                  title={t("option:flashcards.documentView", { defaultValue: "Document view" })}
                >
                  <Button
                    type={listDensity === "document" ? "default" : "text"}
                    onClick={() => setPresentationMode("document")}
                    data-testid="flashcards-density-toggle-document"
                  >
                    {t("option:flashcards.document", { defaultValue: "Doc" })}
                  </Button>
                </Tooltip>
              </div>
            </div>
          </div>
        </div>
        )}

        {/* Selection Summary Bar - simplified to two modes */}
        {viewMode === "cards" && (
        <div
          className="mb-2 flex items-center gap-3"
          data-testid="flashcards-manage-selection-summary"
        >
          {/* 44px touch target wrapper for checkbox */}
          <span className="inline-flex items-center justify-center min-w-11 min-h-11">
            <Checkbox
              indeterminate={someOnPageSelected}
              checked={allOnPageSelected}
              onChange={(e) => {
                if (e.target.checked) selectAllOnPage()
                else clearSelection()
              }}
              aria-label={t("option:flashcards.selectAllOnPage", { defaultValue: "Select all cards on this page" })}
            />
          </span>
          <Text>
            {selectedCount === 0 ? (
              <span className="text-text-muted">
                {totalCount} {t("option:flashcards.cards", { defaultValue: "cards" })}
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <Badge
                  count={
                    <span className="flex items-center gap-1">
                      {/* Icon indicator for colorblind differentiation */}
                      {selectAllAcross ? (
                        <CheckCheck className="size-3" aria-hidden="true" />
                      ) : (
                        <Check className="size-3" aria-hidden="true" />
                      )}
                      {selectedCount}
                    </span>
                  }
                  showZero={false}
                  className="mr-1"
                  style={{ backgroundColor: selectAllAcross ? "rgb(var(--color-primary))" : "rgb(var(--color-success))" }}
                  title={selectAllAcross ? t("option:flashcards.allResults", { defaultValue: "All results" }) : t("option:flashcards.thisPage", { defaultValue: "This page" })}
                />
                <span className="text-text-muted flex items-center gap-1">
                  {selectAllAcross
                    ? t("option:flashcards.selectedAcrossAll", {
                        defaultValue: "selected across all results"
                      })
                    : t("option:flashcards.selectedOnPage", {
                        defaultValue: "selected on this page"
                      })}
                </span>
                {!selectAllAcross && selectedCount > 0 && totalCount > selectedCount && (
                  <button
                    className="text-primary hover:underline text-sm"
                    onClick={selectAllAcrossResults}
                    disabled={selectAllAcrossDisabled}
                    data-testid="flashcards-select-all-across"
                  >
                    {t("option:flashcards.selectAllCount", {
                      defaultValue: "Select all {{count}}",
                      count: totalCount
                    })}
                  </button>
                )}
                <button
                  className="text-text-muted hover:text-text text-sm"
                  onClick={clearSelection}
                >
                  {t("option:flashcards.clear", { defaultValue: "Clear" })}
                </button>
              </span>
            )}
          </Text>
        </div>
        )}

        {viewMode === "cards" ? (
        isDocumentMode ? (
          <FlashcardDocumentView
            items={documentItems}
            decks={decksQuery.data || []}
            isLoading={documentQuery.isLoading}
            isFetchingNextPage={documentQuery.isFetchingNextPage}
            hasNextPage={Boolean(documentQuery.hasNextPage)}
            isTruncated={documentQuery.isTruncated}
            selectedIds={selectedIds}
            selectAllAcross={selectAllAcross}
            filterContext={documentFilterContext}
            queryKey={documentQueryKey}
            onToggleSelect={toggleSelect}
            onLoadMore={() => {
              if (documentQuery.hasNextPage && !documentQuery.isFetchingNextPage) {
                void documentQuery.fetchNextPage()
              }
            }}
            onOpenDrawer={openEdit}
            bulkUpdate={bulkUpdateMutation.mutateAsync}
          />
        ) : (
          <List
            loading={manageQuery.isFetching}
            dataSource={pageItems}
            locale={{
              emptyText: (
                <Empty
                  description={t("option:flashcards.noCardsTitle", {
                    defaultValue:
                      mQuery || mTags.length > 0 || mDeckId != null || mDue !== "all"
                        ? "No cards match your filters"
                        : "No flashcards yet"
                  })}
                >
                  <Space orientation="vertical" align="center">
                    <Text type="secondary">
                      {t("option:flashcards.noCardsDescription", {
                        defaultValue:
                          mQuery || mTags.length > 0 || mDeckId != null || mDue !== "all"
                            ? "Try adjusting your search, deck, tag, or due filters."
                            : "Create cards from your notes and media, or import an existing deck."
                      })}
                    </Text>
                    <Space>
                      {mQuery || mTags.length > 0 || mDeckId != null || mDue !== "all" ? (
                        <Button onClick={clearAllFilters}>
                          {t("option:flashcards.clearFilters", {
                            defaultValue: "Clear filters"
                          })}
                        </Button>
                      ) : (
                        <>
                          <Button type="primary" onClick={() => setCreateOpen(true)}>
                            {t("option:flashcards.noCardsCreateCta", {
                              defaultValue: "Create card"
                            })}
                          </Button>
                          <Button onClick={onNavigateToImport}>
                            {t("option:flashcards.noCardsImportCta", {
                              defaultValue: "Import flashcards"
                            })}
                          </Button>
                        </>
                      )}
                    </Space>
                  </Space>
                </Empty>
              )
            }}
            renderItem={(item, index) => {
            const isFocused = index === focusedIndex
            const compactSchedule = compactSchedulingLabels(item)
            const expandedSchedule = expandedSchedulingLabels(item)
            const sourceMeta = getFlashcardSourceMeta(item)
            const dueRelativeLabel = item.due_at
              ? formatFlashcardRelativeTime(item.due_at)
              : null
            const dueAbsoluteLabel = item.due_at
              ? formatFlashcardAbsoluteDateTime(item.due_at)
              : null
            const isDue = item.due_at
              ? isFlashcardTimestampBefore(item.due_at)
              : false
            return (
            <List.Item
              data-testid={`flashcard-item-${item.uuid}`}
              className={`cursor-pointer hover:bg-surface2/50 ${isFocused ? "ring-2 ring-primary ring-inset bg-surface2/30" : ""}`}
              onClick={() => {
                setFocusedIndex(index)
                togglePreview(item.uuid)
              }}
              actions={[
                <span
                  key="sel"
                  className="inline-flex items-center justify-center min-w-11 min-h-11"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Checkbox
                    checked={selectAllAcross ? true : selectedIds.has(item.uuid)}
                    disabled={selectAllAcross}
                    onChange={(e) => {
                      e.stopPropagation()
                      toggleSelect(item.uuid, e.target.checked)
                    }}
                    aria-label={`Select card: ${item.front.slice(0, 80)}`}
                    data-testid={`flashcard-item-${item.uuid}-select`}
                  />
                </span>,
                <FlashcardActionsMenu
                  key="actions"
                  card={item}
                  onEdit={() => openEdit(item)}
                  onReview={() => onReviewCard(item)}
                  onDuplicate={() => duplicateCard(item)}
                  onMove={() => openMove(item)}
                />
              ]}
            >
              {listDensity === "compact" ? (
                /* Compact mode: Front text + due indicator + deck name */
                <List.Item.Meta
                  title={
                    <div className="flex items-center gap-2">
                      {/* Due status indicator */}
                      {isDue && (
                        <Tooltip title={t("option:flashcards.dueNow", { defaultValue: "Due now" })}>
                          <span className="inline-block w-2 h-2 rounded-full bg-success" />
                        </Tooltip>
                      )}
                      <div className="min-w-0 flex-1 text-sm text-text">
                        <div className="line-clamp-1">
                          <FlashcardMarkdownSnippet
                            content={item.front}
                          />
                        </div>
                      </div>
                    </div>
                  }
                  description={
                    <div className="flex flex-col gap-0.5 text-xs">
                      <div className="flex items-center gap-2">
                        {item.deck_id != null && (
                          <span className="text-text-muted">
                            {resolveDeckLabel(item.deck_id)}
                          </span>
                        )}
                        {item.due_at && (
                          <span className="text-text-subtle">
                            {dueRelativeLabel ?? item.due_at}
                          </span>
                        )}
                        {sourceMeta && (
                          sourceMeta.href ? (
                            <a
                              href={sourceMeta.href}
                              onClick={(event) => event.stopPropagation()}
                              className="text-primary hover:underline"
                              title={t("option:flashcards.sourceOpenLink", {
                                defaultValue: "Open source"
                              })}
                            >
                              {sourceMeta.label}
                            </a>
                          ) : (
                            <span className="text-text-subtle">
                              {sourceMeta.label}
                            </span>
                          )
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-text-subtle">
                        <Tooltip
                          title={t("option:flashcards.schedulingMemoryStrengthHelp", {
                            defaultValue: "SM-2 ease factor (how fast review gaps grow)."
                          })}
                        >
                          <span>{compactSchedule.memoryStrength}</span>
                        </Tooltip>
                        <Tooltip
                          title={t("option:flashcards.schedulingNextGapHelp", {
                            defaultValue: "SM-2 interval (days until next review)."
                          })}
                        >
                          <span>{compactSchedule.nextReviewGap}</span>
                        </Tooltip>
                        <Tooltip
                          title={t("option:flashcards.schedulingRecallRunsHelp", {
                            defaultValue: "SM-2 repetitions (successful recalls)."
                          })}
                        >
                          <span>{compactSchedule.recallRuns}</span>
                        </Tooltip>
                        <Tooltip
                          title={t("option:flashcards.schedulingRelearnsHelp", {
                            defaultValue: "SM-2 lapses (times forgotten)."
                          })}
                        >
                          <span>{compactSchedule.relearns}</span>
                        </Tooltip>
                      </div>
                    </div>
                  }
                />
              ) : (
                /* Expanded mode: Full details with front/back preview */
                <>
                  <List.Item.Meta
                    title={
                      <div className="flex min-w-0 items-start gap-2">
                        <div className="min-w-0 flex-1 text-sm font-semibold text-text">
                          <div className="line-clamp-1">
                            <FlashcardMarkdownSnippet
                              content={item.front}
                            />
                          </div>
                        </div>
                        <span className="text-text-subtle">-</span>
                        <div className="min-w-0 flex-1 text-sm text-text-muted">
                          <div className="line-clamp-1">
                            <FlashcardMarkdownSnippet
                              content={item.back}
                            />
                          </div>
                        </div>
                      </div>
                    }
                    description={
                      <div className="flex items-center gap-2 flex-wrap">
                        {item.deck_id != null && (
                          <Tag color="blue">
                            {resolveDeckLabel(item.deck_id)}
                          </Tag>
                        )}
                        <Tag>{formatCardType(item, t)}</Tag>
                        <FlashcardQueueStateBadge
                          card={item}
                          testId={`flashcards-manage-queue-state-${item.uuid}`}
                        />
                        {(item.tags || []).map((tg) => (
                          <Tag key={tg}>{tg}</Tag>
                        ))}
                        {sourceMeta && (
                          <Tag
                            color={
                              sourceMeta.unavailable
                                ? "default"
                                : sourceMeta.type === "media"
                                  ? "blue"
                                  : sourceMeta.type === "note"
                                    ? "gold"
                                    : "green"
                            }
                          >
                            {sourceMeta.href ? (
                              <a href={sourceMeta.href} onClick={(event) => event.stopPropagation()}>
                                {sourceMeta.label}
                              </a>
                            ) : (
                              sourceMeta.label
                            )}
                          </Tag>
                        )}
                        {item.due_at && (
                          <Tag color="green">
                            {t("option:flashcards.due", { defaultValue: "Due" })}:{" "}
                            {dueRelativeLabel ?? item.due_at} (
                            {dueAbsoluteLabel ?? item.due_at})
                          </Tag>
                        )}
                        <Tooltip
                          title={t("option:flashcards.schedulingMemoryStrengthHelp", {
                            defaultValue: "SM-2 ease factor (how fast review gaps grow)."
                          })}
                        >
                          <Tag>{expandedSchedule.memoryStrength}</Tag>
                        </Tooltip>
                        <Tooltip
                          title={t("option:flashcards.schedulingNextGapHelp", {
                            defaultValue: "SM-2 interval (days until next review)."
                          })}
                        >
                          <Tag>{expandedSchedule.nextReviewGap}</Tag>
                        </Tooltip>
                        <Tooltip
                          title={t("option:flashcards.schedulingRecallRunsHelp", {
                            defaultValue: "SM-2 repetitions (successful recalls)."
                          })}
                        >
                          <Tag>{expandedSchedule.recallRuns}</Tag>
                        </Tooltip>
                        <Tooltip
                          title={t("option:flashcards.schedulingRelearnsHelp", {
                            defaultValue: "SM-2 lapses (times forgotten)."
                          })}
                        >
                          <Tag>{expandedSchedule.relearns}</Tag>
                        </Tooltip>
                      </div>
                    }
                  />
                  {previewOpen.has(item.uuid) && (
                    <div className="mt-2">
                      <div className="border rounded p-2 bg-surface text-xs sm:text-sm">
                        <MarkdownWithBoundary
                          content={item.back}
                          size="xs"
                          className="sm:prose-sm"
                        />
                      </div>
                      {item.extra && (
                        <div className="opacity-80 text-xs mt-1">
                          <MarkdownWithBoundary content={item.extra} size="xs" />
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </List.Item>
          )}}
        />
        )
        ) : (
          <List
            dataSource={Object.values(pendingDeletions).sort(
              (a, b) => a.expiresAt - b.expiresAt
            )}
            locale={{
              emptyText: (
                <Empty
                  description={t("option:flashcards.trashEmptyDescription", {
                    defaultValue: "Deleted cards appear here for 30 seconds."
                  })}
                />
              )
            }}
            renderItem={(item) => {
              const remainingSeconds = Math.max(
                0,
                Math.ceil((item.expiresAt - nowMs) / 1000)
              )
              return (
                <List.Item
                  data-testid={`flashcard-trash-${item.card.uuid}`}
                  actions={[
                    <Button
                      key="undo"
                      className="min-h-11 min-w-11"
                      onClick={() => undoSinglePendingDeletion(item.card.uuid)}
                    >
                      {t("option:flashcards.trashUndo", { defaultValue: "Undo" })}
                    </Button>,
                    <Tag
                      key="expires"
                      color="volcano"
                      role="timer"
                      aria-live={remainingSeconds <= 10 ? "assertive" : "off"}
                      aria-label={t("option:flashcards.trashExpiresInAria", {
                        defaultValue: "Permanently deletes in {{seconds}} seconds",
                        seconds: remainingSeconds
                      })}
                    >
                      {t("option:flashcards.trashExpiresIn", {
                        defaultValue: "Deletes in {{seconds}}s",
                        seconds: remainingSeconds
                      })}
                    </Tag>
                  ]}
                >
                  <List.Item.Meta
                    title={<Text>{item.card.front}</Text>}
                    description={
                      item.card.back ? (
                        <Text type="secondary" className="text-xs">
                          {item.card.back.slice(0, 120)}
                          {item.card.back.length > 120 ? "…" : ""}
                        </Text>
                      ) : null
                    }
                  />
                </List.Item>
              )
            }}
          />
        )}

        {viewMode === "cards" && !isDocumentMode && (
        <div className="mt-3 flex justify-end">
          <Pagination
            current={page}
            pageSize={pageSize}
            onChange={(p, ps) => {
              setPage(p)
              setPageSize(ps)
            }}
            total={totalCount}
            showSizeChanger
            pageSizeOptions={[10, 20, 50, 100]}
          />
        </div>
        )}
      </div>

      {/* Floating Action Bar - appears when items are selected */}
      {viewMode === "cards" && anySelection && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 bg-surface border border-border rounded-lg shadow-lg px-4 py-3 flex items-center gap-4">
          <Badge
            count={
              <span className="flex items-center gap-1">
                {/* Icon indicator for colorblind differentiation */}
                {selectAllAcross ? (
                  <CheckCheck className="size-3" aria-hidden="true" />
                ) : (
                  <Check className="size-3" aria-hidden="true" />
                )}
                {selectedCount}
              </span>
            }
            showZero={false}
            style={{ backgroundColor: selectAllAcross ? "rgb(var(--color-primary))" : "rgb(var(--color-success))" }}
          />
          <span className="text-sm text-text-muted">
            {selectAllAcross
              ? t("option:flashcards.selectedAcrossAll", { defaultValue: "selected across all results" })
              : t("option:flashcards.selected", { defaultValue: "selected" })}
          </span>
          <div className="h-4 w-px bg-border" />
          <Space>
            <Button size="small" onClick={openBulkMove}>
              {t("option:flashcards.bulkMove", { defaultValue: "Move" })}
            </Button>
            <Button size="small" onClick={() => openBulkTagEditor("add")}>
              {t("option:flashcards.bulkAddTag", { defaultValue: "Add tag" })}
            </Button>
            <Button size="small" onClick={() => openBulkTagEditor("remove")}>
              {t("option:flashcards.bulkRemoveTag", { defaultValue: "Remove tag" })}
            </Button>
            <Button size="small" onClick={handleExportSelected}>
              {t("option:flashcards.export", { defaultValue: "Export" })}
            </Button>
            <Button size="small" danger onClick={handleBulkDelete}>
              {t("option:flashcards.bulkDelete", { defaultValue: "Delete" })}
            </Button>
          </Space>
          <div className="h-4 w-px bg-border" />
          <Button type="text" size="small" onClick={clearSelection}>
            {t("option:flashcards.clear", { defaultValue: "Clear" })}
          </Button>
        </div>
      )}

      <Modal
        open={bulkTagOpen}
        title={
          bulkTagMode === "add"
            ? t("option:flashcards.bulkAddTagTitle", {
                defaultValue: "Add tags to selected cards"
              })
            : t("option:flashcards.bulkRemoveTagTitle", {
                defaultValue: "Remove tags from selected cards"
              })
        }
        onCancel={() => {
          setBulkTagOpen(false)
          setBulkTagInput("")
        }}
        onOk={submitBulkTagEdit}
        okText={
          bulkTagMode === "add"
            ? t("option:flashcards.bulkAddTag", { defaultValue: "Add tag" })
            : t("option:flashcards.bulkRemoveTag", { defaultValue: "Remove tag" })
        }
      >
        <Space orientation="vertical" className="w-full">
          <Text type="secondary">
            {t("option:flashcards.bulkTagDescription", {
              defaultValue:
                "Enter one or more tags separated by commas or spaces."
            })}
          </Text>
          <Input
            value={bulkTagInput}
            onChange={(event) => setBulkTagInput(event.target.value)}
            onPressEnter={submitBulkTagEdit}
            placeholder={t("option:flashcards.bulkTagPlaceholder", {
              defaultValue: "example-tag, chapter-1"
            })}
            data-testid="flashcards-bulk-tag-input"
          />
        </Space>
      </Modal>

      <Modal
        open={deckScopeOpen}
        title={t("option:flashcards.deckScopeTitle", {
          defaultValue: "Move deck scope"
        })}
        onCancel={closeDeckScopeEditor}
        onOk={() => {
          void submitDeckScopeEdit()
        }}
        okText={t("common:save", { defaultValue: "Save" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={updateDeckMutation.isPending}
      >
        <Form form={deckScopeForm} layout="vertical">
          <Form.Item
            name="workspaceId"
            label={t("option:flashcards.workspaceId", { defaultValue: "Workspace ID" })}
          >
            <Input
              placeholder={t("option:flashcards.workspaceIdPlaceholder", {
                defaultValue: "Leave blank for general scope"
              })}
            />
          </Form.Item>
          <Form.Item
            name="parentDeckId"
            label={t("option:flashcards.parentDeck", { defaultValue: "Parent deck" })}
          >
            <Select<number>
              allowClear
              placeholder={t("option:flashcards.topLevelDeck", {
                defaultValue: "Top-level deck"
              })}
              options={deckParentOptions}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* Move Drawer */}
      <Drawer
        title={
          moveCard
            ? t("option:flashcards.moveToDeck", { defaultValue: "Move to deck" })
            : t("option:flashcards.bulkMove", { defaultValue: "Bulk Move" })
        }
        placement="right"
        styles={{ wrapper: { width: FLASHCARDS_DRAWER_WIDTH_PX } }}
        open={moveOpen}
        onClose={() => {
          setMoveOpen(false)
          setMoveCard(null)
          setMoveDeckId(null)
        }}
        footer={
          <div className="flex justify-end">
            <Space>
              <Button
                onClick={() => {
                  setMoveOpen(false)
                  setMoveCard(null)
                  setMoveDeckId(null)
                }}
              >
                {t("common:cancel", { defaultValue: "Cancel" })}
              </Button>
            <Button
              type="primary"
              onClick={submitMove}
              disabled={!moveCard && moveDeckId == null}
            >
              {t("option:flashcards.move", { defaultValue: "Move" })}
            </Button>
            </Space>
          </div>
        }
      >
        <Select<number>
          className="w-full"
          allowClear
          loading={decksQuery.isLoading}
          value={moveDeckId ?? undefined}
          onChange={(v) => setMoveDeckId(v ?? null)}
          options={(decksQuery.data || []).map((d) => ({
            label: formatDeckHierarchyLabel(d, deckMap, `Deck ${d.id}`),
            value: d.id
          }))}
        />
      </Drawer>

      {/* Edit Drawer */}
      <FlashcardEditDrawer
        open={editOpen}
        onClose={() => {
          setEditOpen(false)
          setEditing(null)
        }}
        card={editing}
        onSave={doUpdate}
        onDelete={doDelete}
        onResetScheduling={doResetScheduling}
        isLoading={updateMutation.isPending || resetSchedulingMutation.isPending}
        decks={decksQuery.data || []}
        decksLoading={decksQuery.isLoading}
      />

      {/* Create Drawer */}
      <FlashcardCreateDrawer
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        decks={decksQuery.data || []}
        decksLoading={decksQuery.isLoading}
        includeWorkspaceItems={workspaceVisibilityOptions.includeWorkspaceItems}
        workspaceId={workspaceVisibilityOptions.workspaceId}
      />

      {/* Floating Action Button for creating cards */}
      {viewMode === "cards" && !anySelection && (
        <Tooltip title={t("option:flashcards.createCard", { defaultValue: "Create card" })}>
          <Button
            type="primary"
            shape="circle"
            size="large"
            icon={<Plus className="size-5" />}
            className="fixed bottom-6 right-6 z-50 shadow-lg !w-14 !h-14 flex items-center justify-center"
            onClick={() => setCreateOpen(true)}
            data-testid="flashcards-fab-create"
          />
        </Tooltip>
      )}

      {/* Bulk operation progress modal */}
      <Modal
        open={bulkProgress?.open ?? false}
        closable={false}
        footer={null}
        centered
        title={
          bulkProgress?.action ||
          t("option:flashcards.bulkProgressTitle", { defaultValue: "Processing" })
        }
      >
        <div className="flex flex-col items-center gap-4 py-4">
          <Spin size="large" />
          <div className="text-center">
            <Text className="block text-lg">
              {bulkProgress?.current ?? 0} / {bulkProgress?.total ?? 0}
            </Text>
            <Text type="secondary" className="block mt-1">
              {t("option:flashcards.bulkProgressPleaseWait", {
                defaultValue: "Please wait..."
              })}
            </Text>
          </div>
          <Progress
            percent={Math.round(
              ((bulkProgress?.current ?? 0) / (bulkProgress?.total || 1)) * 100
            )}
            status="active"
            className="w-full max-w-xs"
          />
        </div>
      </Modal>

      {/* Type-to-confirm modal for large bulk deletes */}
      <Modal
        open={bulkDeleteConfirmOpen}
        title={t("option:flashcards.bulkDeleteLargeTitle", {
          defaultValue: "Delete {{count}} cards?",
          count: bulkDeleteCount
        })}
        onCancel={() => {
          setBulkDeleteConfirmOpen(false)
          setBulkDeleteInput("")
          setPendingDeleteItems([])
          setBulkDeleteCount(0)
        }}
        okText={t("common:delete", { defaultValue: "Delete" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        okButtonProps={{
          danger: true,
          disabled: bulkDeleteInput.toUpperCase() !== "DELETE"
        }}
        onOk={confirmLargeBulkDelete}
        centered
      >
        <div className="space-y-4">
          <Alert
            type="warning"
            showIcon
            title={t("option:flashcards.bulkDeleteLargeWarning", {
              defaultValue: "These cards will move to Trash for {{seconds}} seconds.",
              seconds: DELETE_UNDO_SECONDS
            })}
          />
          <p className="text-text-muted">
            {t("option:flashcards.bulkDeleteLargeContent", {
              defaultValue:
                "After {{seconds}} seconds, {{count}} cards will be permanently deleted.",
              count: bulkDeleteCount,
              seconds: DELETE_UNDO_SECONDS
            })}
          </p>
          <div className="pt-2">
            <p className="text-sm font-medium text-text-muted mb-2">
              {t("option:flashcards.typeDeleteToConfirm", {
                defaultValue: "Type DELETE to confirm:"
              })}
            </p>
            <Input
              value={bulkDeleteInput}
              onChange={(e) => setBulkDeleteInput(e.target.value)}
              placeholder="DELETE"
              className="font-mono"
              autoFocus
            />
          </div>
        </div>
      </Modal>
    </>
  )
}

export default ManageTab
