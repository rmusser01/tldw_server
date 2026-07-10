import React from "react"
import { useTranslation } from "react-i18next"
import {
  Library,
  MessageSquare,
  ScanText,
  SlidersHorizontal
} from "lucide-react"
import type {
  DocumentProcessingMode,
  UploadedFile
} from "@/db/dexie/types"
import {
  setBatchDocumentProcessingMode,
  setFileDocumentProcessingMode
} from "@/services/chat-document-processing"

type Props = {
  files: UploadedFile[]
  onChangeFiles: (files: UploadedFile[]) => void
}

type ModeOption = {
  mode: DocumentProcessingMode
  label: string
  description: string
  icon: React.ComponentType<{ className?: string }>
}

const unavailableReason = (
  files: UploadedFile[],
  mode: DocumentProcessingMode
): string | null => {
  const reasons = Array.from(
    new Set(
      files
        .map((file) => file.processingCapabilities?.[mode])
        .filter((capability) => capability && !capability.available)
        .map((capability) => capability?.reason)
        .filter((reason): reason is string => Boolean(reason))
    )
  )
  if (reasons.length > 0) return reasons.join("; ")
  return files.some((file) => {
    const capability = file.processingCapabilities?.[mode]
    return capability && !capability.available
  })
    ? "Unavailable for one or more files"
    : null
}

const statusSummary = (
  files: UploadedFile[],
  t: (key: string, fallback: string, options?: Record<string, unknown>) => unknown
): string => {
  const blocked = files.filter(
    (file) => file.processingStatus === "blocked"
  ).length
  const failed = files.filter((file) => file.processingStatus === "failed").length
  const ready = Math.max(0, files.length - blocked - failed)
  const parts = [
    String(
      t("playground:documentProcessing.readyCount", "{{count}} ready", {
        count: ready
      })
    )
  ]
  if (blocked > 0) {
    parts.push(
      String(
        t("playground:documentProcessing.blockedCount", "{{count}} blocked", {
          count: blocked
        })
      )
    )
  }
  if (failed > 0) {
    parts.push(
      String(
        t("playground:documentProcessing.failedCount", "{{count}} failed", {
          count: failed
        })
      )
    )
  }
  return parts.join(", ")
}

export const DocumentProcessingChoices: React.FC<Props> = ({
  files,
  onChangeFiles
}) => {
  const { t } = useTranslation(["playground"])
  const [showPerFile, setShowPerFile] = React.useState(false)

  if (files.length === 0) {
    return null
  }

  const modeOptions: ModeOption[] = [
    {
      mode: "add_to_chat",
      label: t("playground:documentProcessing.addToChat", "Add to chat"),
      description: t(
        "playground:documentProcessing.addToChatDescription",
        "Chat only context"
      ),
      icon: MessageSquare
    },
    {
      mode: "ocr_pages",
      label: t("playground:documentProcessing.ocrPages", "OCR pages"),
      description: t(
        "playground:documentProcessing.ocrPagesDescription",
        "Convert pages to images"
      ),
      icon: ScanText
    },
    {
      mode: "ingest_to_library",
      label: t(
        "playground:documentProcessing.ingestToLibrary",
        "Ingest to library"
      ),
      description: t(
        "playground:documentProcessing.ingestToLibraryDescription",
        "Durable library source"
      ),
      icon: Library
    }
  ]

  const activeMode =
    files.length > 0 &&
    files.every((file) => file.processingMode === files[0]?.processingMode)
      ? files[0]?.processingMode
      : null

  return (
    <div className="border-b border-border/70 px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-text-subtle">
            {t("playground:documentProcessing.title", "Document handling")}
          </div>
          <div
            className="text-[11px] text-text-muted"
            data-testid="document-processing-summary"
          >
            <span>{statusSummary(files, t)}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowPerFile((value) => !value)}
          aria-expanded={showPerFile}
          className="inline-flex min-h-7 items-center gap-1.5 rounded border border-border bg-surface px-2 text-[11px] text-text hover:bg-surface2"
        >
          <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
          {t("playground:documentProcessing.adjustPerFile", "Adjust per file")}
        </button>
      </div>

      <div className="mt-2 grid gap-2 md:grid-cols-3">
        {modeOptions.map((option) => {
          const reason = unavailableReason(files, option.mode)
          const disabled = Boolean(reason)
          const Icon = option.icon
          const selected = activeMode === option.mode
          return (
            <div key={option.mode} className="min-w-0">
              <button
                type="button"
                disabled={disabled}
                aria-pressed={selected}
                onClick={() =>
                  onChangeFiles(
                    setBatchDocumentProcessingMode(files, option.mode)
                  )
                }
                className={[
                  "flex min-h-11 w-full items-center gap-2 rounded border px-2 py-1.5 text-left text-xs",
                  selected
                    ? "border-accent bg-accent/10 text-text"
                    : "border-border bg-surface text-text hover:bg-surface2",
                  disabled ? "cursor-not-allowed opacity-50 hover:bg-surface" : ""
                ].join(" ")}
              >
                <Icon className="h-4 w-4 shrink-0 text-text-subtle" />
                <span className="min-w-0">
                  <span className="block font-medium">{option.label}</span>
                  <span className="block text-[10px] text-text-muted">
                    {option.description}
                  </span>
                </span>
              </button>
              {reason ? (
                <div className="mt-1 text-[10px] text-warn">{reason}</div>
              ) : null}
            </div>
          )
        })}
      </div>

      {showPerFile ? (
        <div className="mt-2 space-y-2 border-t border-border/60 pt-2">
          {files.map((file) => (
            <div
              key={file.id}
              className="flex flex-wrap items-center justify-between gap-2 text-xs"
            >
              <div className="min-w-0 font-medium text-text line-clamp-1">
                {file.filename}
              </div>
              <div className="flex flex-wrap gap-1">
                {modeOptions.map((option) => {
                  const capability = file.processingCapabilities?.[option.mode]
                  const disabled = capability ? !capability.available : false
                  const selected = file.processingMode === option.mode
                  return (
                    <button
                      key={option.mode}
                      type="button"
                      disabled={disabled}
                      aria-pressed={selected}
                      onClick={() =>
                        onChangeFiles(
                          setFileDocumentProcessingMode(
                            files,
                            file.id,
                            option.mode
                          )
                        )
                      }
                      className={[
                        "min-h-7 rounded border px-2 text-[11px]",
                        selected
                          ? "border-accent bg-accent/10 text-text"
                          : "border-border bg-surface text-text-muted hover:bg-surface2",
                        disabled
                          ? "cursor-not-allowed opacity-50 hover:bg-surface"
                          : ""
                      ].join(" ")}
                    >
                      {option.label}
                    </button>
                  )
                })}
              </div>
              {file.processingBlockedReason ? (
                <div className="basis-full text-[10px] text-warn">
                  {file.processingBlockedReason}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
