import React, { useState, useMemo, useCallback, useRef } from "react"
import { useTranslation } from "react-i18next"
import {
  Empty,
  Spin,
  Button,
  Dropdown,
  Input,
  Modal,
  Tooltip,
  Segmented,
  message
} from "antd"
import type { MenuProps } from "antd"
import {
  Highlighter,
  StickyNote,
  Filter,
  SortAsc,
  SortDesc,
  Edit3,
  Trash2,
  ChevronRight,
  Search,
  X,
  Download
} from "lucide-react"
import { ExportAnnotationsModal } from "./ExportAnnotationsModal"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import {
  useAnnotations,
  useUpdateAnnotation,
  useDeleteAnnotation
} from "@/hooks/document-workspace"
import type { Annotation, AnnotationColor, AnnotationType, DocumentType } from "../types"
import { COLOR_BADGES } from "../config"

const { TextArea } = Input

type SortOption = "date-desc" | "date-asc" | "page-asc" | "page-desc"
type TypeFilter = "all" | "highlight" | "page_note"

interface AnnotationCardProps {
  annotation: Annotation
  documentType: DocumentType | null
  onNavigate: () => void
  onEdit: () => void
  onDelete: () => void
}

const AnnotationCard: React.FC<AnnotationCardProps> = ({
  annotation,
  documentType,
  onNavigate,
  onEdit,
  onDelete
}) => {
  const { t } = useTranslation(["option", "common"])
  const colorInfo = COLOR_BADGES[annotation.color]

  // For PDF: location is page number
  // For EPUB: location is CFI string with optional chapter title
  const isEpub = documentType === "epub"
  const pageNumber = typeof annotation.location === "number"
    ? annotation.location
    : parseInt(String(annotation.location), 10) || 0
  const isPageNote = annotation.annotationType === "page_note"

  // Format location label
  let locationLabel: string
  if (isEpub) {
    if (annotation.chapterTitle) {
      locationLabel = annotation.chapterTitle
    } else if (annotation.percentage !== undefined) {
      locationLabel = `${Math.round(annotation.percentage)}%`
    } else {
      locationLabel = t("option:documentWorkspace.location", "Location")
    }
  } else {
    locationLabel = t("option:documentWorkspace.page", "Page") + " " + pageNumber
  }

  return (
    <div
      className={`group relative rounded-lg border p-3 transition-colors ${colorInfo.bg} ${colorInfo.border} hover:border-primary/50`}
    >
      {/* Header with location, type badge, and actions */}
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            onClick={onNavigate}
            className="flex items-center gap-1 text-xs text-text-secondary hover:text-primary"
          >
            <span className="font-medium">
              {locationLabel}
            </span>
            <ChevronRight className="h-3 w-3" />
          </button>
          {isPageNote && (
            <span className="inline-flex items-center gap-1 rounded bg-surface px-1.5 py-0.5 text-[10px] text-text-secondary">
              <StickyNote className="h-2.5 w-2.5" />
              {t("option:documentWorkspace.note", "Note")}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 text-text-muted hover:text-text transition-colors">
          <Tooltip title={t("common:edit", "Edit")}>
            <button
              onClick={onEdit}
              className="rounded p-1 hover:bg-hover"
            >
              <Edit3 className="h-3.5 w-3.5 text-text-secondary" />
            </button>
          </Tooltip>
          <Tooltip title={t("common:delete", "Delete")}>
            <button
              onClick={onDelete}
              className="rounded p-1 hover:bg-hover"
            >
              <Trash2 className="h-3.5 w-3.5 text-text-secondary hover:text-danger" />
            </button>
          </Tooltip>
        </div>
      </div>

      {/* Content: For highlights show quoted text, for page notes show the note directly */}
      {isPageNote ? (
        // Page note: show note as main content
        <p className="line-clamp-4 text-sm leading-relaxed">
          {annotation.note || t("option:documentWorkspace.emptyNote", "(No note text)")}
        </p>
      ) : (
        <>
          {/* Highlighted text */}
          <p className="line-clamp-3 text-sm leading-relaxed">
            "{annotation.text}"
          </p>

          {/* Note if present */}
          {annotation.note && (
            <div className="mt-2 flex items-start gap-1.5 rounded bg-surface/50 p-2">
              <StickyNote className="mt-0.5 h-3 w-3 shrink-0 text-text-secondary" />
              <p className="line-clamp-2 text-xs text-text-secondary">
                {annotation.note}
              </p>
            </div>
          )}
        </>
      )}

      {/* Timestamp */}
      <p className="mt-2 text-[10px] text-text-secondary">
        {new Date(annotation.createdAt).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit"
        })}
      </p>
    </div>
  )
}

interface EditNoteModalProps {
  annotation: Annotation | null
  open: boolean
  onClose: () => void
  onSave: (note: string) => void
  saving: boolean
}

const EditNoteModal: React.FC<EditNoteModalProps> = ({
  annotation,
  open,
  onClose,
  onSave,
  saving
}) => {
  const { t } = useTranslation(["option", "common"])
  const [note, setNote] = useState(annotation?.note || "")

  React.useEffect(() => {
    setNote(annotation?.note || "")
  }, [annotation])

  return (
    <Modal
      title={t("option:documentWorkspace.editNote", "Edit Note")}
      open={open}
      onCancel={onClose}
      onOk={() => onSave(note)}
      okText={t("common:save", "Save")}
      cancelText={t("common:cancel", "Cancel")}
      confirmLoading={saving}
    >
      {annotation && (
        <div className="space-y-4">
          <div className="rounded-lg border border-border bg-surface-hover p-3">
            <p className="line-clamp-3 text-sm italic">"{annotation.text}"</p>
          </div>
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              {t("option:documentWorkspace.noteLabel", "Note")}
            </label>
            <TextArea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder={t(
                "option:documentWorkspace.notePlaceholder",
                "Add a note about this highlight..."
              )}
              autoSize={{ minRows: 3, maxRows: 6 }}
            />
          </div>
        </div>
      )}
    </Modal>
  )
}

/**
 * AnnotationsPanel - Displays and manages document annotations.
 *
 * Features:
 * - List all highlights/notes grouped by page
 * - Filter by color
 * - Sort by date or page
 * - Click to navigate to annotation location
 * - Edit/delete annotations
 */
export const AnnotationsPanel: React.FC = () => {
  const { t } = useTranslation(["option", "common"])

  // Store state
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const activeDocumentType = useDocumentWorkspaceStore((s) => s.activeDocumentType)
  const annotations = useDocumentWorkspaceStore((s) => s.annotations)
  const setCurrentPage = useDocumentWorkspaceStore((s) => s.setCurrentPage)

  // Local state
  const [colorFilter, setColorFilter] = useState<AnnotationColor | "all">("all")
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all")
  const [sortOption, setSortOption] = useState<SortOption>("date-desc")
  const [searchQuery, setSearchQuery] = useState("")
  const [editingAnnotation, setEditingAnnotation] = useState<Annotation | null>(null)
  const [exportModalOpen, setExportModalOpen] = useState(false)

  // Get document title for export
  const openDocuments = useDocumentWorkspaceStore((s) => s.openDocuments)
  const activeDocument = openDocuments.find((d) => d.id === activeDocumentId)
  const documentTitle = activeDocument?.title || "Document"

  // Query hooks
  const { isLoading, error } = useAnnotations(activeDocumentId)
  const updateMutation = useUpdateAnnotation()
  const deleteMutation = useDeleteAnnotation()

  // Filter and sort annotations
  const filteredAnnotations = useMemo(() => {
    let result = [...annotations]

    // Filter by type
    if (typeFilter !== "all") {
      result = result.filter((ann) => {
        const annType = ann.annotationType ?? "highlight"
        return annType === typeFilter
      })
    }

    // Filter by color
    if (colorFilter !== "all") {
      result = result.filter((ann) => ann.color === colorFilter)
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      result = result.filter(
        (ann) =>
          ann.text.toLowerCase().includes(query) ||
          (ann.note && ann.note.toLowerCase().includes(query))
      )
    }

    // Sort
    result.sort((a, b) => {
      switch (sortOption) {
        case "date-desc":
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        case "date-asc":
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        case "page-asc": {
          // For EPUB: use percentage for sorting (CFI strings can't be reliably compared)
          // For PDF: use page number
          const isEpubA = typeof a.location === "string"
          const isEpubB = typeof b.location === "string"
          if (isEpubA && isEpubB) {
            // Both EPUB: sort by percentage
            const pctA = a.percentage ?? 0
            const pctB = b.percentage ?? 0
            return pctA - pctB
          }
          const pageA = typeof a.location === "number" ? a.location : parseInt(String(a.location), 10) || 0
          const pageB = typeof b.location === "number" ? b.location : parseInt(String(b.location), 10) || 0
          return pageA - pageB
        }
        case "page-desc": {
          // For EPUB: use percentage for sorting (CFI strings can't be reliably compared)
          // For PDF: use page number
          const isEpubA = typeof a.location === "string"
          const isEpubB = typeof b.location === "string"
          if (isEpubA && isEpubB) {
            // Both EPUB: sort by percentage (descending)
            const pctA = a.percentage ?? 0
            const pctB = b.percentage ?? 0
            return pctB - pctA
          }
          const pageA = typeof a.location === "number" ? a.location : parseInt(String(a.location), 10) || 0
          const pageB = typeof b.location === "number" ? b.location : parseInt(String(b.location), 10) || 0
          return pageB - pageA
        }
        default:
          return 0
      }
    })

    return result
  }, [annotations, typeFilter, colorFilter, sortOption, searchQuery])

  // Handlers
  const handleNavigate = useCallback((annotation: Annotation) => {
    if (activeDocumentType === "epub" && typeof annotation.location === "string") {
      // EPUB: navigate to CFI location
      window.dispatchEvent(
        new CustomEvent("epub-navigate-cfi", {
          detail: { cfi: annotation.location, documentId: activeDocumentId }
        })
      )
    } else {
      // PDF: navigate to page
      const page = typeof annotation.location === "number"
        ? annotation.location
        : parseInt(String(annotation.location), 10) || 1
      setCurrentPage(page)
    }
  }, [activeDocumentType, activeDocumentId, setCurrentPage])

  const handleEdit = useCallback((annotation: Annotation) => {
    setEditingAnnotation(annotation)
  }, [])

  const handleSaveNote = useCallback(async (note: string) => {
    if (!editingAnnotation || !activeDocumentId) return

    await updateMutation.mutateAsync({
      mediaId: activeDocumentId,
      annotationId: editingAnnotation.id,
      updates: { note: note || undefined }
    })
    setEditingAnnotation(null)
  }, [editingAnnotation, activeDocumentId, updateMutation])

  // Soft-delete with undo: remove from UI immediately, delay server delete by 5s
  const pendingDeleteTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  const handleDelete = useCallback((annotation: Annotation) => {
    if (!activeDocumentId) return

    // Capture full annotation for undo (preserves original id)
    const savedAnnotation = { ...annotation }

    // Remove from local store immediately
    const removeAnnotation = useDocumentWorkspaceStore.getState().removeAnnotation
    removeAnnotation(annotation.id)

    // Show undo toast
    const key = `delete-${annotation.id}`
    message.info({
      key,
      content: (
        <span>
          {t("option:documentWorkspace.annotationDeleted", "Annotation deleted")}{" "}
          <Button
            type="link"
            size="small"
            onClick={() => {
              // Cancel the pending server delete
              const timer = pendingDeleteTimers.current.get(annotation.id)
              if (timer) {
                clearTimeout(timer)
                pendingDeleteTimers.current.delete(annotation.id)
              }
              // Restore with original id to stay in sync with server
              useDocumentWorkspaceStore.setState((state) => ({
                annotations: [...state.annotations, savedAnnotation]
              }))
              message.destroy(key)
            }}
          >
            {t("common:undo", "Undo")}
          </Button>
        </span>
      ),
      duration: 5
    })

    // Schedule server-side delete after 5s
    const timer = setTimeout(async () => {
      pendingDeleteTimers.current.delete(annotation.id)
      try {
        await deleteMutation.mutateAsync({
          mediaId: activeDocumentId,
          annotationId: annotation.id
        })
      } catch {
        // If server delete fails, the annotation was already removed from UI
        // It will re-appear on next sync
      }
    }, 5000)
    pendingDeleteTimers.current.set(annotation.id, timer)
  }, [activeDocumentId, deleteMutation, t])

  // Color filter menu
  const colorFilterItems: MenuProps["items"] = [
    { key: "all", label: t("option:documentWorkspace.allColors", "All colors") },
    { type: "divider" },
    ...Object.entries(COLOR_BADGES).map(([color, info]) => ({
      key: color,
      label: (
        <span className="flex items-center gap-2">
          <span className={`h-3 w-3 rounded-full ${info.bg} ${info.border} border`} />
          {info.label}
        </span>
      )
    }))
  ]

  // Sort menu
  const sortItems: MenuProps["items"] = [
    { key: "date-desc", label: t("option:documentWorkspace.newestFirst", "Newest first") },
    { key: "date-asc", label: t("option:documentWorkspace.oldestFirst", "Oldest first") },
    { type: "divider" },
    { key: "page-asc", label: t("option:documentWorkspace.pageAsc", "Page (low to high)") },
    { key: "page-desc", label: t("option:documentWorkspace.pageDesc", "Page (high to low)") }
  ]

  // No document selected
  if (!activeDocumentId) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          description={t(
            "option:documentWorkspace.noDocumentSelected",
            "No document selected"
          )}
        />
      </div>
    )
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spin />
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center">
        <p className="text-sm text-danger">
          {t("option:documentWorkspace.loadAnnotationsError", "Failed to load annotations")}
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex flex-col gap-2 border-b border-border p-3">
        {/* Type filter - segmented control */}
        <Segmented
          value={typeFilter}
          onChange={(value) => setTypeFilter(value as TypeFilter)}
          options={[
            { value: "all", label: t("option:documentWorkspace.allTypes", "All") },
            { value: "highlight", label: t("option:documentWorkspace.highlights", "Highlights") },
            { value: "page_note", label: t("option:documentWorkspace.notes", "Notes") }
          ]}
          size="small"
          block
        />

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-text-secondary" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t("option:documentWorkspace.searchAnnotations", "Search annotations...")}
            className="pl-8 pr-8"
            size="small"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2"
            >
              <X className="h-4 w-4 text-text-secondary hover:text-text-primary" />
            </button>
          )}
        </div>

        {/* Filter and Sort */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Dropdown
              menu={{
                items: colorFilterItems,
                onClick: ({ key }) => setColorFilter(key as AnnotationColor | "all")
              }}
              trigger={["click"]}
            >
              <Button
                size="small"
                icon={<Filter className="h-3.5 w-3.5" />}
                className="flex items-center gap-1"
              >
                {colorFilter === "all"
                  ? t("option:documentWorkspace.filter", "Filter")
                  : COLOR_BADGES[colorFilter].label}
              </Button>
            </Dropdown>

            <Dropdown
              menu={{
                items: sortItems,
                onClick: ({ key }) => setSortOption(key as SortOption)
              }}
              trigger={["click"]}
            >
              <Button
                size="small"
                icon={
                  sortOption.includes("desc") ? (
                    <SortDesc className="h-3.5 w-3.5" />
                  ) : (
                    <SortAsc className="h-3.5 w-3.5" />
                  )
                }
                className="flex items-center gap-1"
              >
                {t("option:documentWorkspace.sort", "Sort")}
              </Button>
            </Dropdown>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">
              {filteredAnnotations.length} {t("option:documentWorkspace.annotationsCount", "items")}
            </span>
            <Tooltip title={t("option:documentWorkspace.exportAnnotations", "Export")}>
              <Button
                size="small"
                type="text"
                icon={<Download className="h-3.5 w-3.5" />}
                onClick={() => setExportModalOpen(true)}
                disabled={annotations.length === 0}
              />
            </Tooltip>
          </div>
        </div>
      </div>

      {/* Annotations list */}
      <div className="flex-1 overflow-y-auto p-3">
        {filteredAnnotations.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <div className="rounded-full bg-surface-hover p-4">
              {typeFilter === "page_note" ? (
                <StickyNote className="h-8 w-8 text-text-secondary" />
              ) : (
                <Highlighter className="h-8 w-8 text-text-secondary" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium">
                {searchQuery || colorFilter !== "all" || typeFilter !== "all"
                  ? t("option:documentWorkspace.noMatchingAnnotations", "No matching highlights or notes")
                  : t("option:documentWorkspace.noAnnotations", "No highlights or notes yet")}
              </p>
              <p className="mt-1 text-xs text-text-secondary">
                {searchQuery || colorFilter !== "all" || typeFilter !== "all"
                  ? t("option:documentWorkspace.tryDifferentFilter", "Try a different filter")
                  : t(
                      "option:documentWorkspace.selectTextToHighlight",
                      "Select text in the document to highlight"
                    )}
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredAnnotations.map((annotation) => (
              <AnnotationCard
                key={annotation.id}
                annotation={annotation}
                documentType={activeDocumentType}
                onNavigate={() => handleNavigate(annotation)}
                onEdit={() => handleEdit(annotation)}
                onDelete={() => handleDelete(annotation)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Edit note modal */}
      <EditNoteModal
        annotation={editingAnnotation}
        open={!!editingAnnotation}
        onClose={() => setEditingAnnotation(null)}
        onSave={handleSaveNote}
        saving={updateMutation.isPending}
      />

      {/* Export annotations modal */}
      <ExportAnnotationsModal
        open={exportModalOpen}
        onClose={() => setExportModalOpen(false)}
        annotations={annotations}
        documentTitle={documentTitle}
        documentType={activeDocumentType}
      />
    </div>
  )
}
