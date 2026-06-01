import { Button } from "antd"
import { AlertTriangle, Loader2, Upload, WifiOff } from "lucide-react"
import React, { useCallback } from "react"
import { useTranslation } from "react-i18next"
import {
  QUICK_INGEST_ACCEPT_STRING,
  QUICK_INGEST_MAX_FILE_SIZE_LABEL,
  QUICK_INGEST_MAX_FILE_SIZE
} from "../constants"

interface FileDropZoneProps {
  /** Callback when valid files are dropped or selected */
  onFilesAdded: (files: File[]) => void
  /** Callback when files are rejected (size/type) */
  onFilesRejected?: (errors: string[]) => void
  /** Whether processing is running */
  running?: boolean
  /** Whether server is online for ingest */
  isOnlineForIngest?: boolean
  /** Focus the drop zone when it is first shown. */
  autoFocus?: boolean
  /** Additional CSS classes */
  className?: string
}

/**
 * Drag-and-drop file zone for Quick Ingest.
 * Based on pattern from Flashcards/FileDropZone.tsx with:
 * - Drag counter pattern (prevents flickering on nested elements)
 * - Visual feedback for drag states
 * - Size and type validation
 * - Disabled states for processing and server offline
 */
export const FileDropZone: React.FC<FileDropZoneProps> = ({
  onFilesAdded,
  onFilesRejected,
  running = false,
  isOnlineForIngest = true,
  autoFocus = false,
  className
}) => {
  const { t } = useTranslation(["option"])
  const [isDragging, setIsDragging] = React.useState(false)
  const [isDragReject, setIsDragReject] = React.useState(false)
  const [dragFileCount, setDragFileCount] = React.useState(0)
  const dropZoneRef = React.useRef<HTMLDivElement>(null)
  const inputRef = React.useRef<HTMLInputElement>(null)
  // Drag counter prevents flickering when dragging over nested elements
  const dragCounterRef = React.useRef(0)

  const disabled = running || !isOnlineForIngest

  React.useEffect(() => {
    if (!autoFocus || disabled) return
    dropZoneRef.current?.focus()
  }, [autoFocus, disabled])

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      t(`quickIngest.${key}`, defaultValue, options),
    [t]
  )

  const acceptEntries = React.useMemo(() => {
    return QUICK_INGEST_ACCEPT_STRING.split(",")
      .map((entry) => entry.trim().toLowerCase())
      .filter(Boolean)
  }, [])

  // Validate extensions from accept string
  const validExtensions = React.useMemo(() => {
    return acceptEntries.filter((entry) => entry.startsWith("."))
  }, [acceptEntries])

  const validMimeTypes = React.useMemo(() => {
    return acceptEntries.filter((entry) => entry.includes("/"))
  }, [acceptEntries])

  const isMimeTypeAllowed = useCallback(
    (mimeType: string) => {
      const normalized = mimeType.toLowerCase()
      if (!normalized) return false
      if (validMimeTypes.includes(normalized)) return true
      const category = normalized.split("/")[0]
      return validMimeTypes.includes(`${category}/*`)
    },
    [validMimeTypes]
  )

  const validateFile = useCallback(
    (file: File): string | null => {
      // Size check
      if (file.size > QUICK_INGEST_MAX_FILE_SIZE) {
        return `${file.name} exceeds ${QUICK_INGEST_MAX_FILE_SIZE_LABEL} quick-ingest limit`
      }

      // Type check - allow if extension matches OR MIME type matches
      const fileExt = "." + (file.name.split(".").pop()?.toLowerCase() || "")
      const isValidExtension = validExtensions.includes(fileExt)
      const isValidMime = file.type ? isMimeTypeAllowed(file.type) : false

      if (!isValidExtension && !isValidMime) {
        return `${file.name} is not a supported file type`
      }

      return null
    },
    [isMimeTypeAllowed, validExtensions]
  )

  const resetDragState = useCallback(() => {
    dragCounterRef.current = 0
    setIsDragging(false)
    setIsDragReject(false)
    setDragFileCount(0)
  }, [])

  const getDragSummary = useCallback(
    (dataTransfer: DataTransfer | null) => {
      if (!dataTransfer) return { count: 0, isReject: false }
      const items = Array.from(dataTransfer.items || [])
      const fileItems = items.filter((item) => item.kind === "file")
      const count = fileItems.length || dataTransfer.files?.length || 0
      let isReject = false
      for (const item of fileItems) {
        const type = item.type?.toLowerCase()
        if (type && !isMimeTypeAllowed(type)) {
          isReject = true
          break
        }
      }
      return { count, isReject }
    },
    [isMimeTypeAllowed]
  )

  const processFiles = useCallback(
    (files: File[]) => {
      if (disabled) return

      const validFiles: File[] = []
      const errors: string[] = []

      for (const file of files) {
        const error = validateFile(file)
        if (error) {
          errors.push(error)
        } else {
          validFiles.push(file)
        }
      }

      if (validFiles.length > 0) {
        onFilesAdded(validFiles)
      }

      if (errors.length > 0) {
        onFilesRejected?.(errors)
      }
    },
    [disabled, validateFile, onFilesAdded, onFilesRejected]
  )

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (disabled) {
      resetDragState()
      return
    }
    dragCounterRef.current += 1
    if (dragCounterRef.current === 1) {
      const summary = getDragSummary(e.dataTransfer)
      setDragFileCount(summary.count)
      setIsDragReject(summary.isReject)
      setIsDragging(true)
    }
  }, [disabled, getDragSummary, resetDragState])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (disabled) {
      resetDragState()
      return
    }
    dragCounterRef.current = Math.max(0, dragCounterRef.current - 1)
    if (dragCounterRef.current === 0) {
      setIsDragging(false)
      setIsDragReject(false)
      setDragFileCount(0)
    }
  }, [disabled, resetDragState])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (disabled) return
    const summary = getDragSummary(e.dataTransfer)
    setDragFileCount(summary.count)
    setIsDragReject(summary.isReject)
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = summary.isReject ? "none" : "copy"
    }
  }, [disabled, getDragSummary])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    resetDragState()

    if (disabled) return

    const files = Array.from(e.dataTransfer?.files || [])
    if (files.length > 0) {
      processFiles(files)
    }
  }, [disabled, processFiles, resetDragState])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      processFiles(files)
    }
    // Reset input so same file can be selected again
    if (e.target) e.target.value = ""
  }

  const handleBrowseClick = useCallback((
    event?: React.MouseEvent | React.KeyboardEvent
  ) => {
    event?.stopPropagation()
    if (!disabled) {
      inputRef.current?.click()
    }
  }, [disabled])

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.currentTarget !== event.target) return
    if (event.key !== "Enter" && event.key !== " ") return
    event.preventDefault()
    handleBrowseClick(event)
  }

  // Determine visual state
  const getStateStyles = () => {
    if (disabled) {
      return "border-border bg-surface2 opacity-50 cursor-not-allowed"
    }
    if (isDragging && isDragReject) {
      return "border-danger/50 bg-danger/10 cursor-pointer"
    }
    if (isDragging) {
      return "border-primary bg-primary/5 cursor-pointer"
    }
    return "border-border bg-surface2 hover:border-primary/50 cursor-pointer"
  }

  const releaseLabel =
    dragFileCount > 1
      ? qi("dropzoneReleaseMultiple", "Release to add {{count}} files", {
          count: dragFileCount
        })
      : dragFileCount === 1
      ? qi("dropzoneReleaseSingle", "Release to add 1 file")
      : qi("dropzoneRelease", "Release to add files")

  const ariaStatus = running
    ? qi("dropzoneProcessing", "Processing in progress...")
    : !isOnlineForIngest
    ? qi("dropzoneOffline", "Server not connected")
    : isDragging && isDragReject
    ? qi("dropzoneRejectTitle", "Some files not supported")
    : isDragging
    ? releaseLabel
    : ""

  return (
    <div
      ref={dropZoneRef}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleBrowseClick}
      onKeyDown={handleKeyDown}
      className={`mt-3 w-full rounded-md border border-dashed px-4 py-4 text-center transition-colors ${getStateStyles()} ${className || ""}`}
      data-testid="qi-file-dropzone"
      role="button"
      aria-label={qi(
        "dropzoneAriaLabel",
        "File upload zone. Drag and drop files or press Enter to browse."
      )}
      aria-disabled={disabled}
      tabIndex={disabled ? -1 : 0}
    >
      <div className="flex flex-col gap-2 items-center justify-center">
        {/* Icon based on state */}
        {running ? (
          <Loader2 className="size-6 animate-spin text-text-muted" />
        ) : !isOnlineForIngest ? (
          <WifiOff className="size-6 text-text-muted" />
        ) : isDragging && isDragReject ? (
          <AlertTriangle className="size-6 text-danger" />
        ) : isDragging ? (
          <Upload className="size-6 text-primary" />
        ) : (
          <Upload className="size-6 text-text-muted" />
        )}

        {/* Text based on state */}
        {running ? (
          <>
            <p className="text-base font-medium text-text-muted">
              {qi("dropzoneProcessing", "Processing in progress...")}
            </p>
            <p className="text-xs text-text-subtle">
              {qi("dropzoneWaitProcessing", "Wait for completion to add more")}
            </p>
          </>
        ) : !isOnlineForIngest ? (
          <>
            <p className="text-base font-medium text-text-muted">
              {qi("dropzoneOffline", "Server not connected")}
            </p>
            <p className="text-xs text-text-subtle">
              {qi("dropzoneConnectHint", "Connect to add files")}
            </p>
          </>
        ) : isDragging && isDragReject ? (
          <>
            <p className="text-base font-medium text-danger">
              {qi("dropzoneRejectTitle", "Some files not supported")}
            </p>
            <p className="text-xs text-text-subtle">
              {qi("dropzoneRejectHint", "Check file type and size (max {{maxSize}})", {
                maxSize: QUICK_INGEST_MAX_FILE_SIZE_LABEL
              })}
            </p>
          </>
        ) : isDragging ? (
          <p className="text-base font-medium text-primary">{releaseLabel}</p>
        ) : (
          <>
            <p className="text-base font-medium">
              {qi("dragAndDrop", "Drag and drop files")}
            </p>
            <p className="text-xs text-text-subtle">
              {qi(
                "dragAndDropHint",
                "Docs, PDFs, HTML/XML/JSON, audio, and video are all welcome."
              )}
            </p>
          </>
        )}

        {/* Browse button - only show when not dragging */}
        {!isDragging && (
          <Button onClick={handleBrowseClick} disabled={disabled}>
            {qi("addFiles", "Browse files")}
          </Button>
        )}
      </div>
      <span className="sr-only" aria-live="polite" aria-atomic="true">
        {ariaStatus}
      </span>

      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={QUICK_INGEST_ACCEPT_STRING}
        onChange={handleFileSelect}
        className="hidden"
        data-testid="qi-file-input"
      />
    </div>
  )
}

export default FileDropZone
