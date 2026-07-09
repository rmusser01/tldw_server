import React from "react"
import { useTranslation } from "react-i18next"
import {
  ChevronDown,
  ChevronUp,
  ImageIcon,
  FileText,
  Globe,
  X,
  Trash2,
  Settings2
} from "lucide-react"
import { Image, Tooltip } from "antd"
import { DocumentChip } from "@/components/Common/Playground/DocumentChip"
import type { UploadedFile as StoredUploadedFile } from "@/db/dexie/types"

interface Document {
  id: number
  title: string
  url: string
  favIconUrl?: string
}

type UploadedFile = Pick<StoredUploadedFile, "id" | "filename" | "size"> &
  Partial<
    Pick<
      StoredUploadedFile,
      "processingMode" | "processingStatus" | "processingBlockedReason"
    >
  >

interface AttachmentsSummaryProps {
  /** Base64 image string */
  image: string
  /** Selected browser tabs/documents */
  documents: Document[]
  /** Uploaded files */
  files: UploadedFile[]
  /** Callback to remove the image */
  onRemoveImage: () => void
  /** Callback to remove a document */
  onRemoveDocument: (id: number) => void
  /** Callback to clear all documents */
  onClearDocuments: () => void
  /** Callback to remove a file */
  onRemoveFile: (id: string) => void
  /** Callback to clear all files */
  onClearFiles: () => void
  /** Callback to open Knowledge Panel for managing context */
  onOpenKnowledgePanel?: () => void
  /** Render as read-only summary (no remove actions). */
  readOnly?: boolean
}

export const AttachmentsSummary: React.FC<AttachmentsSummaryProps> = ({
  image,
  documents,
  files,
  onRemoveImage,
  onRemoveDocument,
  onClearDocuments,
  onRemoveFile,
  onClearFiles,
  onOpenKnowledgePanel,
  readOnly = false
}) => {
  const { t } = useTranslation(["playground", "common", "option"])
  const [expanded, setExpanded] = React.useState(false)

  const imageCount = image ? 1 : 0
  const docCount = documents.length
  const fileCount = files.length
  const totalFileBytes = files.reduce((total, file) => total + file.size, 0)
  const totalCount = imageCount + docCount + fileCount
  const largeAttachmentThresholdBytes = 20 * 1024 * 1024
  const singleLargeFileThresholdBytes = 10 * 1024 * 1024
  const hasLargeAttachments =
    totalFileBytes >= largeAttachmentThresholdBytes ||
    files.some((file) => file.size >= singleLargeFileThresholdBytes)

  if (totalCount === 0) {
    return null
  }

  const formatFileSize = (bytes: number) => {
    return new Intl.NumberFormat(undefined, {
      style: "unit",
      unit: "megabyte",
      maximumFractionDigits: 2
    }).format(bytes / (1024 * 1024))
  }

  const fileProcessingLabel = (file: UploadedFile) => {
    if (file.processingStatus === "blocked") {
      return t("playground:attachments.processingBlocked", "Blocked")
    }
    if (file.processingStatus === "failed") {
      return t("playground:attachments.processingFailed", "Failed")
    }
    if (file.processingStatus === "processing") {
      return t("playground:attachments.processing", "Processing")
    }
    if (file.processingStatus === "preflighting") {
      return t("playground:attachments.processingPreflight", "Checking")
    }
    if (file.processingMode === "ocr_pages") {
      return t("playground:attachments.processingOcr", "OCR")
    }
    if (file.processingMode === "ingest_to_library") {
      return t("playground:attachments.processingIngest", "Library ingest")
    }
    if (file.processingMode === "add_to_chat") {
      return t("playground:attachments.processingChatOnly", "Chat only")
    }
    return null
  }

  const handleClearAll = () => {
    if (image) onRemoveImage()
    if (documents.length > 0) onClearDocuments()
    if (files.length > 0) onClearFiles()
  }

  return (
    <div className="border-b border-border/70">
      {/* Collapsed summary bar */}
      <div className="flex w-full items-center px-3 py-2 transition-colors hover:bg-surface2/50">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex min-w-0 flex-1 items-center justify-between gap-2 text-left"
          aria-expanded={expanded}
          aria-controls="attachments-panel"
          title={`${t("playground:attachments.title", "Attachments")} (${totalCount})`}
        >
          <div className="flex min-w-0 items-center gap-3">
            <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-text-subtle">
              {t("playground:attachments.title", "Attachments")} ({totalCount})
            </span>
            <div className="flex items-center gap-2 text-[11px] text-text-muted">
              {imageCount > 0 && (
                <span className="inline-flex items-center gap-1">
                  <ImageIcon className="h-3 w-3" aria-hidden="true" />
                  {imageCount}
                </span>
              )}
              {fileCount > 0 && (
                <span className="inline-flex items-center gap-1">
                  <FileText className="h-3 w-3" aria-hidden="true" />
                  {fileCount}
                </span>
              )}
              {docCount > 0 && (
                <span className="inline-flex items-center gap-1">
                  <Globe className="h-3 w-3" aria-hidden="true" />
                  {docCount}
                </span>
              )}
            </div>
          </div>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-text-subtle" />
          ) : (
            <ChevronDown className="h-4 w-4 text-text-subtle" />
          )}
        </button>
        {!readOnly && (
          <Tooltip title={t("playground:attachments.clearAll", "Clear all")}>
            <button
              type="button"
              onClick={handleClearAll}
              className="ml-2 rounded p-1 text-text-subtle hover:bg-surface2 hover:text-text"
              aria-label={t("playground:attachments.clearAll", "Clear all") as string}
              title={t("playground:attachments.clearAll", "Clear all") as string}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
        )}
      </div>

      {/* Expanded panel */}
      {expanded && (
        <div
          id="attachments-panel"
          className="space-y-3 px-3 pb-3"
        >
          {/* Image section */}
          {image && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] text-text-muted">
                <span className="font-medium">
                  {t("playground:attachments.image", "Image")}
                </span>
                {!readOnly && (
                  <button
                    type="button"
                    onClick={onRemoveImage}
                    className="text-text-subtle hover:text-text"
                    aria-label={t("common:remove", "Remove") as string}
                    title={t("common:remove", "Remove") as string}
                  >
                    {t("common:remove", "Remove")}
                  </button>
                )}
              </div>
              <div className="relative inline-block">
                {!readOnly && (
                  <button
                    type="button"
                    onClick={onRemoveImage}
                    className="absolute -top-1 -left-1 z-10 flex items-center justify-center rounded-full border border-border bg-surface p-0.5 text-text hover:bg-surface2"
                    aria-label={t("common:remove", "Remove") as string}
                    title={t("common:remove", "Remove") as string}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
                <Image
                  src={image}
                  alt="Attached image"
                  preview={false}
                  className="rounded-md max-h-20"
                />
              </div>
            </div>
          )}

          {/* Tabs/Documents section */}
          {documents.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] text-text-muted">
                <span className="font-medium">
                  {t("playground:attachments.tabs", "Web pages")} ({documents.length})
                </span>
                {!readOnly && (
                  <button
                    type="button"
                    onClick={onClearDocuments}
                    className="text-text-subtle hover:text-text"
                    title={t("playground:composer.clearTabs", "Remove all") as string}
                  >
                    {t("playground:composer.clearTabs", "Remove all")}
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                {documents.map((doc) => (
                  <DocumentChip
                    key={doc.id}
                    document={doc}
                    variant="compact"
                    onRemove={readOnly ? undefined : onRemoveDocument}
                    removeLabel={t("option:remove", "Remove") as string}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Files section */}
          {files.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] text-text-muted">
                <span className="font-medium">
                  {t("playground:attachments.files", "Uploaded files")} ({files.length})
                </span>
                {!readOnly && (
                  <button
                    type="button"
                    onClick={onClearFiles}
                    className="text-text-subtle hover:text-text"
                    title={t("playground:composer.clearFiles", "Remove all") as string}
                  >
                    {t("playground:composer.clearFiles", "Remove all")}
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                {files.map((file) => {
                  const processingLabel = fileProcessingLabel(file)
                  return (
                    <div
                      key={file.id}
                      className="group relative flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5 text-xs"
                    >
                      <FileText className="h-3.5 w-3.5 text-text-subtle" />
                      <div className="flex flex-col">
                        <span className="font-medium text-text line-clamp-1 max-w-[150px]">
                          {file.filename}
                        </span>
                        <span className="text-[10px] text-text-muted">
                          {formatFileSize(file.size)}
                        </span>
                        {processingLabel && (
                          <span className="text-[10px] font-medium text-text-subtle">
                            {processingLabel}
                          </span>
                        )}
                      </div>
                      {!readOnly && (
                        <button
                          type="button"
                          onClick={() => onRemoveFile(file.id)}
                          className="absolute -top-1 -right-1 invisible rounded-full border border-border bg-surface p-0.5 text-text shadow-sm group-hover:visible hover:bg-surface2"
                          aria-label={t("common:remove", "Remove") as string}
                          title={t("common:remove", "Remove") as string}
                        >
                          <X className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {hasLargeAttachments && (
            <div
              className="rounded-md border border-warn/40 bg-warn/10 px-2 py-2 text-xs text-warn"
              data-testid="attachments-large-warning"
            >
              <p className="font-medium">
                {t(
                  "playground:attachments.largeFileWarning",
                  "Large file attachments can increase latency and context usage."
                )}
              </p>
              <p className="mt-1 text-[11px] text-warn">
                {t(
                  "playground:attachments.largeFileHint",
                  "Consider pinning focused excerpts or using tab context for long sources."
                )}
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {onOpenKnowledgePanel && (
                  <button
                    type="button"
                    onClick={onOpenKnowledgePanel}
                    className="rounded border border-warn/40 bg-surface px-2 py-0.5 text-[11px] font-medium text-warn hover:bg-warn/10"
                  >
                    {t(
                      "playground:attachments.reviewInKnowledge",
                      "Review in context panel"
                    )}
                  </button>
                )}
                {!readOnly && fileCount > 0 && (
                  <button
                    type="button"
                    onClick={onClearFiles}
                    className="rounded border border-warn/40 bg-surface px-2 py-0.5 text-[11px] font-medium text-warn hover:bg-warn/10"
                  >
                    {t(
                      "playground:attachments.clearLargeFiles",
                      "Clear files"
                    )}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Manage in Knowledge Panel link */}
          {onOpenKnowledgePanel && (
            <div className="pt-2 border-t border-border/50">
              <button
                type="button"
                onClick={onOpenKnowledgePanel}
                className="flex items-center gap-1.5 text-[11px] text-accent hover:text-accent/80 transition-colors"
                title={t("playground:attachments.manageContext", "Manage in Knowledge Panel") as string}
              >
                <Settings2 className="h-3.5 w-3.5" />
                <span>{t("playground:attachments.manageContext", "Manage in Knowledge Panel")}</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
