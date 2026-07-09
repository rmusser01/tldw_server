import React from "react"
import { AlertCircle, CheckCircle2, FileText, Loader2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import type {
  DocumentProcessingRecoveryAction,
  DocumentProcessingStatus,
  DocumentProcessingTurnMetadata
} from "@/db/dexie/types"
import {
  documentProcessingModeLabel,
  documentProcessingStatusLabel,
  toDisplayText
} from "./document-processing-labels"

type Props = {
  metadata?: DocumentProcessingTurnMetadata | null
}

const recoveryLabel = (
  action: DocumentProcessingRecoveryAction,
  t: (key: string, fallback: string) => unknown
): string => {
  switch (action) {
    case "retry":
      return toDisplayText(t("playground:documentProcessing.retry", "Retry"))
    case "cancel":
      return toDisplayText(t("playground:documentProcessing.cancel", "Cancel"))
    case "switch_to_add_to_chat":
      return toDisplayText(
        t("playground:documentProcessing.switchToChat", "Switch to chat")
      )
    case "switch_to_ocr":
      return toDisplayText(
        t("playground:documentProcessing.switchToOcr", "Switch to OCR")
      )
    case "switch_to_ingest":
      return toDisplayText(
        t(
          "playground:documentProcessing.switchToIngest",
          "Switch to ingest"
        )
      )
    case "use_chat_scoped_retrieval":
      return toDisplayText(
        t(
          "playground:documentProcessing.useChatRetrieval",
          "Use chat retrieval"
        )
      )
    case "remove":
      return toDisplayText(
        t("playground:documentProcessing.removeFile", "Remove file")
      )
    default:
      return action
  }
}

const fileCountLabel = (
  count: number,
  t: (key: string, fallback: string, options?: Record<string, unknown>) => unknown
): string =>
  toDisplayText(
    count === 1
      ? t("playground:documentProcessing.fileCountOne", "1 file")
      : t("playground:documentProcessing.fileCount", "{{count}} files", {
          count
        })
  )

const isAttentionStatus = (status?: DocumentProcessingStatus) =>
  status === "blocked" || status === "failed"

export const DocumentProcessingTurn: React.FC<Props> = ({ metadata }) => {
  const { t } = useTranslation(["playground"])
  if (!metadata || metadata.files.length === 0) {
    return null
  }

  const attentionFiles = metadata.files.filter((file) =>
    isAttentionStatus(file.status)
  )
  const visibleFiles = attentionFiles.length > 0 ? attentionFiles : metadata.files
  const Icon =
    metadata.status === "ready"
      ? CheckCircle2
      : metadata.status === "blocked" || metadata.status === "failed"
        ? AlertCircle
        : Loader2

  return (
    <div className="w-full max-w-[calc(100%-1.75rem)] rounded-lg border border-border bg-surface/80 px-3 py-2 text-xs text-text">
      <div className="flex flex-wrap items-center gap-2">
        <Icon className="h-3.5 w-3.5 text-text-subtle" aria-hidden="true" />
        <span className="font-medium">
          {documentProcessingStatusLabel(metadata.status, t)}
        </span>
        <span className="text-text-muted">
          {fileCountLabel(metadata.files.length, t)}
        </span>
      </div>
      <div className="mt-2 space-y-1">
        {visibleFiles.map((file) => (
          <div
            key={file.id}
            className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]"
          >
            <FileText className="h-3 w-3 text-text-subtle" aria-hidden="true" />
            <span className="font-medium text-text line-clamp-1">
              {file.filename}
            </span>
            <span className="text-text-muted">
              {documentProcessingModeLabel(file.mode, t)}
            </span>
            {(file.summary || file.error) && (
              <span className="basis-full pl-5 text-text-muted">
                {file.summary || file.error}
              </span>
            )}
          </div>
        ))}
      </div>
      {metadata.recoveryActions && metadata.recoveryActions.length > 0 ? (
        <div className="mt-2 flex flex-wrap gap-1">
          {metadata.recoveryActions.map((action) => (
            <span
              key={action}
              className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text-muted"
            >
              {recoveryLabel(action, t)}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}
