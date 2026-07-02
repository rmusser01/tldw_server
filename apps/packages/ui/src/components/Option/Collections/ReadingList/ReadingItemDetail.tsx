import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import DOMPurify from "dompurify"
import {
  Button,
  Drawer,
  Empty,
  message,
  Select,
  Spin,
  Tabs,
  Tooltip,
  Input,
  Modal
} from "antd"
import type { TabsProps } from "antd"
import {
  Star,
  ExternalLink,
  Trash2,
  Clock,
  Calendar,
  Globe,
  Sparkles,
  Volume2,
  FileDown,
  Highlighter,
  StickyNote,
  X
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useCollectionsStore } from "@/store/collections"
import { safeExternalUrl } from "@/utils/safe-external-url"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import type { Highlight, HighlightColor, ReadingNoteLink, ReadingStatus } from "@/types/collections"
import { TagSelector } from "../common/TagSelector"
import { HighlightCard } from "../Highlights/HighlightCard"

const { TextArea } = Input

interface ReadingItemDetailProps {
  onRefresh?: () => void
}

export const ReadingItemDetail: React.FC<ReadingItemDetailProps> = ({
  onRefresh
}) => {
  const { t } = useTranslation(["collections", "common"])
  const api = useTldwApiClient()

  const selectedItemId = useCollectionsStore((s) => s.selectedItemId)
  const currentItem = useCollectionsStore((s) => s.currentItem)
  const currentItemLoading = useCollectionsStore((s) => s.currentItemLoading)
  const itemDetailOpen = useCollectionsStore((s) => s.itemDetailOpen)
  const readingNoteLinksEnabled = useCollectionsStore((s) => s.readingNoteLinksEnabled)

  const setCurrentItem = useCollectionsStore((s) => s.setCurrentItem)
  const setCurrentItemLoading = useCollectionsStore((s) => s.setCurrentItemLoading)
  const setReadingNoteLinksEnabled = useCollectionsStore((s) => s.setReadingNoteLinksEnabled)
  const closeItemDetail = useCollectionsStore((s) => s.closeItemDetail)
  const updateItemInList = useCollectionsStore((s) => s.updateItemInList)
  const removeItem = useCollectionsStore((s) => s.removeItem)
  const addHighlight = useCollectionsStore((s) => s.addHighlight)
  const removeHighlight = useCollectionsStore((s) => s.removeHighlight)
  const updateHighlightInList = useCollectionsStore((s) => s.updateHighlightInList)
  const openHighlightEditor = useCollectionsStore((s) => s.openHighlightEditor)
  const highlightEditorOpen = useCollectionsStore((s) => s.highlightEditorOpen)
  const editingHighlight = useCollectionsStore((s) => s.editingHighlight)

  const [actionLoading, setActionLoading] = useState(false)
  const [summarizing, setSummarizing] = useState(false)
  const [generatingTts, setGeneratingTts] = useState(false)
  const [editingNotes, setEditingNotes] = useState(false)
  const [notesValue, setNotesValue] = useState("")
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [itemHighlights, setItemHighlights] = useState<Highlight[]>([])
  const [itemHighlightsLoading, setItemHighlightsLoading] = useState(false)
  const [itemHighlightsError, setItemHighlightsError] = useState<string | null>(null)
  const [highlightQuote, setHighlightQuote] = useState("")
  const [highlightNote, setHighlightNote] = useState("")
  const [highlightColor, setHighlightColor] = useState<HighlightColor>("yellow")
  const [highlightSaving, setHighlightSaving] = useState(false)
  const [highlightDeleteOpen, setHighlightDeleteOpen] = useState(false)
  const [highlightDeleteId, setHighlightDeleteId] = useState<string | null>(null)
  const [highlightDeleteLoading, setHighlightDeleteLoading] = useState(false)
  const [selectedQuote, setSelectedQuote] = useState("")
  const [selectedNote, setSelectedNote] = useState("")
  const [selectedColor, setSelectedColor] = useState<HighlightColor>("yellow")
  const [selectedMatchId, setSelectedMatchId] = useState<string | null>(null)
  const [linkedNotes, setLinkedNotes] = useState<ReadingNoteLink[]>([])
  const [linkedNotesLoading, setLinkedNotesLoading] = useState(false)
  const [linkNoteId, setLinkNoteId] = useState("")
  const [linkingNote, setLinkingNote] = useState(false)
  const [unlinkingNoteId, setUnlinkingNoteId] = useState<string | null>(null)
  const [notesDirty, setNotesDirty] = useState(false)
  const [notesSaving, setNotesSaving] = useState(false)
  const [notesSaveState, setNotesSaveState] = useState<"idle" | "dirty" | "saving" | "saved" | "error">("idle")
  const contentRef = useRef<HTMLDivElement | null>(null)
  const notesAutosaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastEditedItemId = useRef<string | null>(null)

  // Fetch item details
  useEffect(() => {
    if (!selectedItemId || !itemDetailOpen) return

    const fetchItem = async () => {
      setCurrentItemLoading(true)
      try {
        const item = await api.getReadingItem(selectedItemId)
        setCurrentItem(item)
        setNotesValue(item.notes || "")
        setNotesDirty(false)
        setNotesSaveState("idle")
      } catch (error: any) {
        message.error(error?.message || "Failed to load article details")
        closeItemDetail()
      } finally {
        setCurrentItemLoading(false)
      }
    }

    fetchItem()
  }, [
    selectedItemId,
    itemDetailOpen,
    api,
    setCurrentItem,
    setCurrentItemLoading,
    closeItemDetail
  ])

  const fetchItemHighlights = useCallback(async () => {
    if (!selectedItemId || !itemDetailOpen) return
    setItemHighlightsLoading(true)
    setItemHighlightsError(null)
    try {
      const highlights = await api.getHighlights(selectedItemId)
      const enriched = highlights.map((highlight: any) => ({
        ...highlight,
        item_title: highlight.item_title || currentItem?.title
      }))
      setItemHighlights(enriched)
    } catch (error: any) {
      const errorMsg = error?.message || "Failed to load highlights"
      setItemHighlightsError(errorMsg)
    } finally {
      setItemHighlightsLoading(false)
    }
  }, [api, selectedItemId, itemDetailOpen, currentItem?.title])

  useEffect(() => {
    fetchItemHighlights()
  }, [fetchItemHighlights])

  const fetchLinkedNotes = useCallback(async () => {
    if (!selectedItemId || !itemDetailOpen || !readingNoteLinksEnabled) {
      setLinkedNotes([])
      return
    }
    setLinkedNotesLoading(true)
    try {
      const links = await api.listReadingItemNoteLinks(selectedItemId)
      setLinkedNotes(links)
    } catch (error: any) {
      const errorMsg = error?.message || "Failed to load linked notes"
      if (errorMsg.includes("reading_note_links_disabled")) {
        setReadingNoteLinksEnabled(false)
        setLinkedNotes([])
      } else {
        message.error(errorMsg)
      }
    } finally {
      setLinkedNotesLoading(false)
    }
  }, [api, itemDetailOpen, readingNoteLinksEnabled, selectedItemId, setReadingNoteLinksEnabled])

  useEffect(() => {
    void fetchLinkedNotes()
  }, [fetchLinkedNotes])

  useEffect(() => {
    if (highlightEditorOpen && editingHighlight?.item_id) {
      lastEditedItemId.current = String(editingHighlight.item_id)
    }
  }, [highlightEditorOpen, editingHighlight])

  useEffect(() => {
    if (!highlightEditorOpen && lastEditedItemId.current) {
      if (currentItem?.id === lastEditedItemId.current) {
        fetchItemHighlights()
      }
      lastEditedItemId.current = null
    }
  }, [highlightEditorOpen, currentItem?.id, fetchItemHighlights])

  const normalizeQuote = useCallback((value: string) => {
    return value.replace(/\s+/g, " ").trim().toLowerCase()
  }, [])

  const clearSelectionDraft = useCallback(() => {
    setSelectedQuote("")
    setSelectedNote("")
    setSelectedColor("yellow")
    setSelectedMatchId(null)
    if (typeof window !== "undefined") {
      const selection = window.getSelection()
      if (selection && selection.rangeCount > 0) {
        selection.removeAllRanges()
      }
    }
  }, [])

  const clearNotesAutosaveTimer = useCallback(() => {
    if (notesAutosaveTimerRef.current) {
      clearTimeout(notesAutosaveTimerRef.current)
      notesAutosaveTimerRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => {
      clearNotesAutosaveTimer()
    }
  }, [clearNotesAutosaveTimer])

  useEffect(() => {
    if (!currentItem) {
      setNotesDirty(false)
      setNotesSaveState("idle")
      return
    }
    const baseline = currentItem.notes || ""
    const dirty = editingNotes && notesValue !== baseline
    setNotesDirty(dirty)
    if (dirty) {
      setNotesSaveState("dirty")
    } else if (editingNotes) {
      setNotesSaveState("saved")
    } else {
      setNotesSaveState("idle")
    }
  }, [currentItem, editingNotes, notesValue])

  const saveNotes = useCallback(
    async (showSuccess: boolean): Promise<boolean> => {
      if (!currentItem) return true
      setNotesSaving(true)
      setNotesSaveState("saving")
      try {
        await api.updateReadingItem(currentItem.id, { notes: notesValue })
        const updated = { ...currentItem, notes: notesValue }
        setCurrentItem(updated)
        updateItemInList(currentItem.id, { notes: notesValue })
        setNotesDirty(false)
        setNotesSaveState("saved")
        if (showSuccess) {
          message.success(t("collections:reading.notesSaved", "Notes saved"))
        }
        return true
      } catch (error: any) {
        setNotesSaveState("error")
        message.error(error?.message || "Failed to save notes")
        return false
      } finally {
        setNotesSaving(false)
      }
    },
    [api, currentItem, notesValue, setCurrentItem, t, updateItemInList]
  )

  useEffect(() => {
    if (!editingNotes || !notesDirty || notesSaving || !currentItem) return
    clearNotesAutosaveTimer()
    notesAutosaveTimerRef.current = setTimeout(() => {
      void saveNotes(false)
    }, 1000)
    return () => {
      clearNotesAutosaveTimer()
    }
  }, [clearNotesAutosaveTimer, currentItem, editingNotes, notesDirty, notesSaving, saveNotes])

  const captureContentSelection = useCallback(() => {
    if (!contentRef.current) return
    const selection = window.getSelection()
    if (!selection || selection.rangeCount === 0) return
    const raw = selection.toString()
    const quote = raw.replace(/\s+/g, " ").trim()
    if (!quote) {
      clearSelectionDraft()
      return
    }
    const range = selection.getRangeAt(0)
    if (!contentRef.current.contains(range.commonAncestorContainer)) return
    const trimmedQuote = quote.slice(0, 2000)
    const matched = itemHighlights.find(
      (highlight) => normalizeQuote(highlight.quote) === normalizeQuote(trimmedQuote)
    )
    setSelectedQuote(trimmedQuote)
    setSelectedColor((matched?.color as HighlightColor) || "yellow")
    setSelectedNote(matched?.note || "")
    setSelectedMatchId(matched?.id || null)
  }, [clearSelectionDraft, itemHighlights, normalizeQuote])

  const handleQuickHighlightSave = useCallback(async () => {
    if (!currentItem || !selectedQuote.trim()) return
    setHighlightSaving(true)
    try {
      if (selectedMatchId) {
        const updated = await api.updateHighlight(selectedMatchId, {
          color: selectedColor,
          note: selectedNote.trim() || undefined,
          state: "active"
        })
        setItemHighlights((prev) =>
          prev.map((highlight) =>
            highlight.id === selectedMatchId
              ? { ...highlight, ...updated, item_title: highlight.item_title || currentItem.title }
              : highlight
          )
        )
        updateHighlightInList(selectedMatchId, {
          color: updated.color,
          note: updated.note,
          state: updated.state
        })
        message.success(t("collections:highlights.updated", "Highlight updated"))
      } else {
        const created = await api.createHighlight({
          item_id: currentItem.id,
          quote: selectedQuote.trim(),
          note: selectedNote.trim() || undefined,
          color: selectedColor
        })
        const normalized = created.item_title
          ? created
          : { ...created, item_title: currentItem.title }
        setItemHighlights((prev) => [normalized, ...prev])
        addHighlight(normalized)
        message.success(t("collections:highlights.created", "Highlight created"))
      }
      clearSelectionDraft()
    } catch (error: any) {
      message.error(error?.message || "Failed to save highlight")
    } finally {
      setHighlightSaving(false)
    }
  }, [
    addHighlight,
    api,
    clearSelectionDraft,
    currentItem,
    selectedColor,
    selectedMatchId,
    selectedNote,
    selectedQuote,
    t,
    updateHighlightInList
  ])

  const handleQuickHighlightDelete = useCallback(async () => {
    if (!selectedMatchId) return
    setHighlightDeleteLoading(true)
    try {
      await api.deleteHighlight(selectedMatchId)
      setItemHighlights((prev) => prev.filter((highlight) => highlight.id !== selectedMatchId))
      removeHighlight(selectedMatchId)
      message.success(t("collections:highlights.deleted", "Highlight deleted"))
      clearSelectionDraft()
    } catch (error: any) {
      message.error(error?.message || "Failed to delete highlight")
    } finally {
      setHighlightDeleteLoading(false)
    }
  }, [api, clearSelectionDraft, removeHighlight, selectedMatchId, t])

  const notesSaveStateLabel = useMemo(() => {
    if (!editingNotes) return null
    if (notesSaveState === "saving") {
      return t("collections:reading.notesSaving", "Saving…")
    }
    if (notesSaveState === "dirty") {
      return t("collections:reading.notesDirty", "Unsaved changes")
    }
    if (notesSaveState === "error") {
      return t("collections:reading.notesSaveError", "Save failed")
    }
    return t("collections:reading.notesSavedState", "All changes saved")
  }, [editingNotes, notesSaveState, t])

  const handleClose = useCallback(() => {
    clearNotesAutosaveTimer()
    if (!editingNotes || !notesDirty) {
      closeItemDetail()
      setEditingNotes(false)
      clearSelectionDraft()
      return
    }
    void (async () => {
      const saved = await saveNotes(false)
      if (saved) {
        closeItemDetail()
        setEditingNotes(false)
        clearSelectionDraft()
        return
      }
      Modal.confirm({
        title: t("collections:reading.notesDiscardConfirmTitle", "Discard unsaved notes?"),
        content: t(
          "collections:reading.notesDiscardConfirmMessage",
          "Autosave failed. You can discard your unsaved note edits or stay and retry."
        ),
        okText: t("collections:reading.discard", "Discard"),
        okButtonProps: { danger: true },
        cancelText: t("common:cancel", "Cancel"),
        onOk: () => {
          closeItemDetail()
          setEditingNotes(false)
          clearSelectionDraft()
        }
      })
    })()
  }, [
    clearNotesAutosaveTimer,
    editingNotes,
    notesDirty,
    closeItemDetail,
    clearSelectionDraft,
    saveNotes,
    t
  ])

  const handleToggleFavorite = useCallback(async () => {
    if (!currentItem) return
    setActionLoading(true)
    try {
      await api.updateReadingItem(currentItem.id, {
        favorite: !currentItem.favorite
      })
      const updated = { ...currentItem, favorite: !currentItem.favorite }
      setCurrentItem(updated)
      updateItemInList(currentItem.id, { favorite: updated.favorite })
    } catch (error: any) {
      message.error(error?.message || "Failed to update favorite status")
    } finally {
      setActionLoading(false)
    }
  }, [api, currentItem, setCurrentItem, updateItemInList])

  const handleStatusChange = useCallback(
    async (newStatus: ReadingStatus) => {
      if (!currentItem) return
      setActionLoading(true)
      try {
        await api.updateReadingItem(currentItem.id, { status: newStatus })
        const updated = { ...currentItem, status: newStatus }
        setCurrentItem(updated)
        updateItemInList(currentItem.id, { status: newStatus })
        message.success(
          t("collections:reading.statusUpdated", "Status updated to {{status}}", {
            status: newStatus
          })
        )
      } catch (error: any) {
        message.error(error?.message || "Failed to update status")
      } finally {
        setActionLoading(false)
      }
    },
    [api, currentItem, setCurrentItem, updateItemInList, t]
  )

  const handleTagsChange = useCallback(
    async (tags: string[]) => {
      if (!currentItem) return
      try {
        await api.updateReadingItem(currentItem.id, { tags })
        const updated = { ...currentItem, tags }
        setCurrentItem(updated)
        updateItemInList(currentItem.id, { tags })
      } catch (error: any) {
        message.error(error?.message || "Failed to update tags")
      }
    },
    [api, currentItem, setCurrentItem, updateItemInList]
  )

  const handleSaveNotes = useCallback(async () => {
    const saved = await saveNotes(true)
    if (saved) {
      setEditingNotes(false)
    }
  }, [saveNotes])

  const handleLinkNote = useCallback(async () => {
    if (!currentItem) return
    const noteId = linkNoteId.trim()
    if (!noteId) {
      message.warning(t("collections:reading.linkedNotes.noteIdRequired", "Enter a note ID"))
      return
    }
    setLinkingNote(true)
    try {
      await api.linkReadingItemToNote(currentItem.id, noteId)
      setLinkNoteId("")
      await fetchLinkedNotes()
      message.success(t("collections:reading.linkedNotes.linked", "Note linked"))
    } catch (error: any) {
      const errorMsg = error?.message || "Failed to link note"
      if (errorMsg.includes("reading_note_links_disabled")) {
        setReadingNoteLinksEnabled(false)
        setLinkedNotes([])
      } else {
        message.error(errorMsg)
      }
    } finally {
      setLinkingNote(false)
    }
  }, [
    api,
    currentItem,
    fetchLinkedNotes,
    linkNoteId,
    setReadingNoteLinksEnabled,
    t
  ])

  const handleUnlinkNote = useCallback(
    async (noteId: string) => {
      if (!currentItem) return
      setUnlinkingNoteId(noteId)
      try {
        await api.unlinkReadingItemNote(currentItem.id, noteId)
        setLinkedNotes((prev) => prev.filter((link) => link.note_id !== noteId))
        message.success(t("collections:reading.linkedNotes.unlinked", "Note unlinked"))
      } catch (error: any) {
        const errorMsg = error?.message || "Failed to unlink note"
        if (errorMsg.includes("reading_note_links_disabled")) {
          setReadingNoteLinksEnabled(false)
          setLinkedNotes([])
        } else {
          message.error(errorMsg)
        }
      } finally {
        setUnlinkingNoteId(null)
      }
    },
    [api, currentItem, setReadingNoteLinksEnabled, t]
  )

  const handleSummarize = useCallback(async () => {
    if (!currentItem) return
    setSummarizing(true)
    try {
      const result = await api.summarizeReadingItem(currentItem.id)
      const updated = { ...currentItem, summary: result.summary }
      setCurrentItem(updated)
      message.success(t("collections:reading.summarized", "Summary generated"))
    } catch (error: any) {
      message.error(error?.message || "Failed to generate summary")
    } finally {
      setSummarizing(false)
    }
  }, [api, currentItem, setCurrentItem, t])

  const handleGenerateTts = useCallback(async () => {
    if (!currentItem) return
    setGeneratingTts(true)
    try {
      const result = await api.generateReadingItemTts(currentItem.id)
      const updated = { ...currentItem, tts_audio_url: result.audio_url }
      setCurrentItem(updated)
      message.success(t("collections:reading.ttsGenerated", "Audio generated"))
    } catch (error: any) {
      message.error(error?.message || "Failed to generate audio")
    } finally {
      setGeneratingTts(false)
    }
  }, [api, currentItem, setCurrentItem, t])

  const handleDelete = useCallback(async () => {
    if (!currentItem) return
    setActionLoading(true)
    try {
      await api.deleteReadingItem(currentItem.id, { hard: true })
      removeItem(currentItem.id)
      message.success(t("collections:reading.deleted", "Article deleted"))
      handleClose()
      onRefresh?.()
    } catch (error: any) {
      message.error(error?.message || "Failed to delete article")
    } finally {
      setActionLoading(false)
      setDeleteModalOpen(false)
    }
  }, [api, currentItem, removeItem, handleClose, onRefresh, t])

  const handleCreateHighlight = useCallback(async () => {
    if (!currentItem) return
    const quote = highlightQuote.trim()
    if (!quote) {
      message.warning(
        t("collections:highlights.quoteRequired", "Please enter the highlight text")
      )
      return
    }
    setHighlightSaving(true)
    try {
      const created = await api.createHighlight({
        item_id: currentItem.id,
        quote,
        note: highlightNote.trim() || undefined,
        color: highlightColor
      })
      const normalized = created.item_title
        ? created
        : { ...created, item_title: currentItem.title }
      setItemHighlights((prev) => [normalized, ...prev])
      addHighlight(normalized)
      setHighlightQuote("")
      setHighlightNote("")
      message.success(t("collections:highlights.created", "Highlight created"))
    } catch (error: any) {
      message.error(error?.message || "Failed to create highlight")
    } finally {
      setHighlightSaving(false)
    }
  }, [addHighlight, api, currentItem, highlightColor, highlightNote, highlightQuote, t])

  const handleHighlightDeleteClick = useCallback((id: string) => {
    setHighlightDeleteId(id)
    setHighlightDeleteOpen(true)
  }, [])

  const handleHighlightDeleteConfirm = useCallback(async () => {
    if (!highlightDeleteId) return
    setHighlightDeleteLoading(true)
    try {
      await api.deleteHighlight(highlightDeleteId)
      setItemHighlights((prev) => prev.filter((h) => h.id !== highlightDeleteId))
      removeHighlight(highlightDeleteId)
      message.success(t("collections:highlights.deleted", "Highlight deleted"))
      setHighlightDeleteOpen(false)
      setHighlightDeleteId(null)
    } catch (error: any) {
      message.error(error?.message || "Failed to delete highlight")
    } finally {
      setHighlightDeleteLoading(false)
    }
  }, [api, highlightDeleteId, removeHighlight, t])

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric"
    })
  }

  const tabItems: TabsProps["items"] = [
    {
      key: "content",
      label: (
        <span className="flex items-center gap-1">
          <Globe className="h-4 w-4" />
          {t("collections:reading.tabs.content", "Content")}
        </span>
      ),
      children: (
        <div className="space-y-3">
          {selectedQuote && (
            <div className="rounded-lg border border-primary/30 bg-primary/10 p-3">
              <p className="text-xs font-medium text-primary">
                {selectedMatchId
                  ? t("collections:highlights.selectionMatched", "Selected text matches an existing highlight")
                  : t("collections:highlights.selectionCaptured", "Selected text captured")}
              </p>
              <blockquote className="mt-1 text-sm italic text-text">
                "{selectedQuote}"
              </blockquote>
              <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
                <Select
                  value={selectedColor}
                  onChange={(value) => setSelectedColor(value as HighlightColor)}
                  options={[
                    { value: "yellow", label: t("collections:colors.yellow", "Yellow") },
                    { value: "green", label: t("collections:colors.green", "Green") },
                    { value: "blue", label: t("collections:colors.blue", "Blue") },
                    { value: "pink", label: t("collections:colors.pink", "Pink") },
                    { value: "purple", label: t("collections:colors.purple", "Purple") }
                  ]}
                  className="w-32"
                  size="small"
                />
                <Input
                  value={selectedNote}
                  onChange={(e) => setSelectedNote(e.target.value)}
                  placeholder={t("collections:highlights.notePlaceholder", "Add context or why this matters...")}
                  size="small"
                  className="flex-1"
                />
                <div className="flex items-center gap-2">
                  <Button
                    type="primary"
                    size="small"
                    loading={highlightSaving}
                    onClick={handleQuickHighlightSave}
                  >
                    {selectedMatchId
                      ? t("collections:highlights.update", "Update")
                      : t("collections:highlights.add", "Add Highlight")}
                  </Button>
                  {selectedMatchId && (
                    <Button
                      danger
                      size="small"
                      loading={highlightDeleteLoading}
                      onClick={handleQuickHighlightDelete}
                    >
                      {t("common:delete", "Delete")}
                    </Button>
                  )}
                  <Button size="small" onClick={clearSelectionDraft}>
                    {t("common:clear", "Clear")}
                  </Button>
                </div>
              </div>
            </div>
          )}

          <div
            ref={contentRef}
            onMouseUp={captureContentSelection}
            onKeyUp={captureContentSelection}
            className="prose prose-sm dark:prose-invert max-w-none"
          >
            {currentItem?.clean_html ? (
              <div
                role="region"
                aria-label={t("collections:readingItem.contentRegion", { defaultValue: "Article content" })}
                dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(currentItem.clean_html, { USE_PROFILES: { html: true } }) }}
              />
            ) : currentItem?.text ? (
              <pre className="whitespace-pre-wrap text-sm text-text">
                {currentItem.text}
              </pre>
            ) : (
              <p className="text-text-muted">
                {t("collections:reading.noContent", "Content not available")}
              </p>
            )}
          </div>
        </div>
      )
    },
    {
      key: "summary",
      label: (
        <span className="flex items-center gap-1">
          <Sparkles className="h-4 w-4" />
          {t("collections:reading.tabs.summary", "Summary")}
        </span>
      ),
      children: (
        <div className="space-y-4">
          {currentItem?.summary ? (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <p>{currentItem.summary}</p>
            </div>
          ) : (
            <div className="text-center py-8">
              <Sparkles className="h-8 w-8 mx-auto text-text-subtle mb-2" />
              <p className="text-text-muted mb-4">
                {t("collections:reading.noSummary", "No summary yet")}
              </p>
              <Button
                type="primary"
                icon={<Sparkles className="h-4 w-4" />}
                onClick={handleSummarize}
                loading={summarizing}
              >
                {t("collections:reading.generateSummary", "Generate Summary")}
              </Button>
            </div>
          )}
        </div>
      )
    },
    {
      key: "highlights",
      label: (
        <span className="flex items-center gap-1">
          <Highlighter className="h-4 w-4" />
          {t("collections:reading.tabs.highlights", "Highlights")}
        </span>
      ),
      children: (
        <div className="space-y-4">
          <div className="rounded-lg border border-border p-4">
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-sm font-medium text-text-muted">
                  {t("collections:highlights.quoteLabel", "Quote")}
                </label>
                <TextArea
                  rows={3}
                  value={highlightQuote}
                  onChange={(e) => setHighlightQuote(e.target.value)}
                  placeholder={t(
                    "collections:highlights.quotePlaceholder",
                    "Paste the highlighted text..."
                  )}
                  aria-label={t("collections:highlights.quoteLabel", "Quote")}
                />
              </div>
              <div>
                <label className="mb-1 block text-sm font-medium text-text-muted">
                  {t("collections:highlights.noteLabel", "Note (optional)")}
                </label>
                <TextArea
                  rows={2}
                  value={highlightNote}
                  onChange={(e) => setHighlightNote(e.target.value)}
                  placeholder={t(
                    "collections:highlights.notePlaceholder",
                    "Add context or why this matters..."
                  )}
                  aria-label={t("collections:highlights.noteLabel", "Note (optional)")}
                />
              </div>
              <div className="flex items-center gap-3">
                <Select
                  value={highlightColor}
                  onChange={(value) => setHighlightColor(value as HighlightColor)}
                  options={[
                    { value: "yellow", label: t("collections:colors.yellow", "Yellow") },
                    { value: "green", label: t("collections:colors.green", "Green") },
                    { value: "blue", label: t("collections:colors.blue", "Blue") },
                    { value: "pink", label: t("collections:colors.pink", "Pink") },
                    { value: "purple", label: t("collections:colors.purple", "Purple") }
                  ]}
                  className="w-32"
                  size="small"
                />
                <Button
                  type="primary"
                  icon={<Highlighter className="h-4 w-4" />}
                  onClick={handleCreateHighlight}
                  loading={highlightSaving}
                >
                  {t("collections:highlights.add", "Add Highlight")}
                </Button>
              </div>
            </div>
          </div>

          {itemHighlightsLoading ? (
            <div className="flex items-center justify-center py-6">
              <Spin size="large" />
            </div>
          ) : itemHighlightsError ? (
            <Empty
              description={itemHighlightsError}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            >
              <Button onClick={fetchItemHighlights}>{t("common:retry", "Retry")}</Button>
            </Empty>
          ) : itemHighlights.length === 0 ? (
            <Empty
              description={t(
                "collections:highlights.emptyForItem",
                "No highlights for this article yet"
              )}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            <div className="space-y-3">
              {itemHighlights.map((highlight) => (
                <HighlightCard
                  key={highlight.id}
                  highlight={highlight}
                  onDelete={handleHighlightDeleteClick}
                  onEdit={openHighlightEditor}
                />
              ))}
            </div>
          )}
        </div>
      )
    },
    {
      key: "notes",
      label: (
        <span className="flex items-center gap-1">
          <StickyNote className="h-4 w-4" />
          {t("collections:reading.tabs.notes", "Notes")}
        </span>
      ),
      children: (
        <div className="space-y-4">
          {editingNotes ? (
            <>
              <TextArea
                value={notesValue}
                onChange={(e) => setNotesValue(e.target.value)}
                rows={6}
                placeholder={t("collections:reading.notesPlaceholder", "Add your notes...")}
              />
              <div className="flex items-center justify-between text-xs text-text-muted">
                <span>{notesSaveStateLabel}</span>
                {notesSaveState === "saving" && <Spin size="small" />}
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  onClick={() => {
                    setNotesValue(currentItem?.notes || "")
                    setEditingNotes(false)
                    setNotesDirty(false)
                    setNotesSaveState("idle")
                    clearNotesAutosaveTimer()
                  }}
                >
                  {t("common:cancel", "Cancel")}
                </Button>
                <Button type="primary" onClick={handleSaveNotes} loading={notesSaving}>
                  {t("common:save", "Save")}
                </Button>
              </div>
            </>
          ) : (
            <>
              {currentItem?.notes ? (
                <div className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap">
                  {currentItem.notes}
                </div>
              ) : (
                <p className="text-text-muted">
                  {t("collections:reading.noNotes", "No notes yet")}
                </p>
              )}
              <Button onClick={() => setEditingNotes(true)}>
                {currentItem?.notes
                  ? t("collections:reading.editNotes", "Edit Notes")
                  : t("collections:reading.addNotes", "Add Notes")}
              </Button>
            </>
          )}

          {readingNoteLinksEnabled && (
            <div className="rounded-lg border border-border p-4">
              <h4 className="mb-2 text-sm font-medium text-text">
                {t("collections:reading.linkedNotes.title", "Linked Notes")}
              </h4>
              <div className="mb-3 flex items-center gap-2">
                <Input
                  value={linkNoteId}
                  onChange={(e) => setLinkNoteId(e.target.value)}
                  placeholder={t(
                    "collections:reading.linkedNotes.placeholder",
                    "Enter note ID to link"
                  )}
                  onPressEnter={() => void handleLinkNote()}
                />
                <Button
                  type="primary"
                  onClick={() => void handleLinkNote()}
                  loading={linkingNote}
                >
                  {t("collections:reading.linkedNotes.link", "Link")}
                </Button>
              </div>

              {linkedNotesLoading ? (
                <div className="flex items-center justify-center py-2">
                  <Spin size="small" />
                </div>
              ) : linkedNotes.length === 0 ? (
                <p className="text-sm text-text-muted">
                  {t("collections:reading.linkedNotes.empty", "No linked notes")}
                </p>
              ) : (
                <div className="space-y-2">
                  {linkedNotes.map((link) => (
                    <div
                      key={link.note_id}
                      className="flex items-center justify-between gap-2 rounded border border-border px-3 py-2"
                    >
                      <span className="text-sm text-text">{link.note_id}</span>
                      <Button
                        size="small"
                        onClick={() => void handleUnlinkNote(link.note_id)}
                        loading={unlinkingNoteId === link.note_id}
                      >
                        {t("collections:reading.linkedNotes.unlink", "Unlink")}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )
    }
  ]

  return (
    <>
      <Drawer
        title={null}
        open={itemDetailOpen}
        onClose={handleClose}
        size={640}
        className="reading-item-detail-drawer"
        styles={{ body: { padding: 0 } }}
      >
        {currentItemLoading ? (
          <div className="flex items-center justify-center py-12">
            <Spin size="large" />
          </div>
        ) : currentItem ? (
          <div className="flex flex-col h-full">
            {/* Header */}
            <div className="border-b border-border p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <h2 className="text-xl font-semibold text-text">
                    {currentItem.title}
                  </h2>
                  <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-text-muted">
                    {currentItem.domain && (
                      <a
                        href={safeExternalUrl(currentItem.url) ?? undefined}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 hover:text-primary"
                      >
                        <Globe className="h-3 w-3" />
                        {currentItem.domain}
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                    {currentItem.published_at && (
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {formatDate(currentItem.published_at)}
                      </span>
                    )}
                    {currentItem.reading_time_minutes && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {t("collections:reading.readingTime", "{{count}} min read", {
                          count: currentItem.reading_time_minutes
                        })}
                      </span>
                    )}
                  </div>
                </div>
                <Button
                  type="text"
                  icon={<X className="h-5 w-5" />}
                  onClick={handleClose}
                />
              </div>

              {/* Status & Tags Row */}
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <Select
                  value={currentItem.status}
                  onChange={handleStatusChange}
                  options={[
                    { value: "saved", label: t("collections:status.saved", "Saved") },
                    { value: "reading", label: t("collections:status.reading", "Reading") },
                    { value: "read", label: t("collections:status.read", "Read") },
                    { value: "archived", label: t("collections:status.archived", "Archived") }
                  ]}
                  size="small"
                  className="w-28"
                  loading={actionLoading}
                />

                <Tooltip
                  title={
                    currentItem.favorite
                      ? t("collections:reading.unfavorite", "Unfavorite")
                      : t("collections:reading.favorite", "Favorite")
                  }
                >
                  <Button
                    type={currentItem.favorite ? "primary" : "default"}
                    size="small"
                    icon={
                      <Star
                        className={`h-4 w-4 ${currentItem.favorite ? "fill-current" : ""}`}
                      />
                    }
                    onClick={handleToggleFavorite}
                    loading={actionLoading}
                  />
                </Tooltip>

                <div className="flex-1">
                  <TagSelector
                    tags={currentItem.tags}
                    onChange={handleTagsChange}
                    placeholder={t("collections:reading.addTag", "Add tag...")}
                  />
                </div>
              </div>
            </div>

            {/* Content Tabs */}
            <div className="flex-1 overflow-auto p-4">
              <Tabs items={tabItems} defaultActiveKey="content" />
            </div>

            {/* Footer Actions */}
            <div className="border-t border-border p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Button
                    icon={<Sparkles className="h-4 w-4" />}
                    onClick={handleSummarize}
                    loading={summarizing}
                  >
                    {t("collections:reading.summarize", "Summarize")}
                  </Button>
                  <Button
                    icon={<Volume2 className="h-4 w-4" />}
                    onClick={handleGenerateTts}
                    loading={generatingTts}
                  >
                    {t("collections:reading.generateTts", "Generate Audio")}
                  </Button>
                  {currentItem.tts_audio_url && (
                    <Button
                      icon={<FileDown className="h-4 w-4" />}
                      href={currentItem.tts_audio_url}
                      target="_blank"
                    >
                      {t("collections:reading.downloadAudio", "Download Audio")}
                    </Button>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    danger
                    icon={<Trash2 className="h-4 w-4" />}
                    onClick={() => setDeleteModalOpen(true)}
                  >
                    {t("common:delete", "Delete")}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </Drawer>

      {/* Delete Confirmation Modal */}
      <Modal
        title={t("collections:reading.deleteConfirm.title", "Delete Article")}
        open={deleteModalOpen}
        onCancel={() => setDeleteModalOpen(false)}
        onOk={handleDelete}
        okText={t("common:delete", "Delete")}
        okButtonProps={{ danger: true, loading: actionLoading }}
        cancelText={t("common:cancel", "Cancel")}
      >
        <p>
          {t(
            "collections:reading.deleteConfirm.message",
            "Are you sure you want to delete this article? This action cannot be undone."
          )}
        </p>
      </Modal>

      <Modal
        title={t("collections:highlights.deleteConfirm.title", "Delete Highlight")}
        open={highlightDeleteOpen}
        onCancel={() => setHighlightDeleteOpen(false)}
        onOk={handleHighlightDeleteConfirm}
        okText={t("common:delete", "Delete")}
        okButtonProps={{ danger: true, loading: highlightDeleteLoading }}
        cancelText={t("common:cancel", "Cancel")}
      >
        <p>
          {t(
            "collections:highlights.deleteConfirm.message",
            "Are you sure you want to delete this highlight?"
          )}
        </p>
      </Modal>
    </>
  )
}
