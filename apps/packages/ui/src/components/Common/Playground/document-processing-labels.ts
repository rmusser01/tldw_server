import type {
  DocumentProcessingMode,
  DocumentProcessingStatus,
  DocumentProcessingTurnMetadata
} from "@/db/dexie/types"

type Translate = (
  key: string,
  fallback: string,
  options?: Record<string, unknown>
) => unknown

export const toDisplayText = (value: unknown): string =>
  typeof value === "string" ? value : String(value ?? "")

export const documentProcessingStatusLabel = (
  status: DocumentProcessingTurnMetadata["status"] | DocumentProcessingStatus | undefined,
  t: Translate
): string => {
  switch (status) {
    case "waiting_for_files":
      return toDisplayText(
        t("playground:documentProcessing.waiting", "Waiting for files")
      )
    case "preflighting":
      return toDisplayText(
        t("playground:documentProcessing.preflighting", "Checking documents")
      )
    case "pending":
      return toDisplayText(
        t("playground:documentProcessing.pending", "Ready to process")
      )
    case "processing":
      return toDisplayText(
        t("playground:documentProcessing.processing", "Processing documents")
      )
    case "ready":
      return toDisplayText(
        t("playground:documentProcessing.ready", "Documents ready")
      )
    case "blocked":
      return toDisplayText(
        t(
          "playground:documentProcessing.blocked",
          "Document processing blocked"
        )
      )
    case "failed":
      return toDisplayText(
        t("playground:documentProcessing.failed", "Document processing failed")
      )
    case "sending_prompt":
      return toDisplayText(
        t("playground:documentProcessing.sending", "Sending prompt")
      )
    case "cancelled":
      return toDisplayText(
        t("playground:documentProcessing.cancelled", "Processing cancelled")
      )
    default:
      return toDisplayText(
        t("playground:documentProcessing.processing", "Processing documents")
      )
  }
}

export const documentProcessingModeLabel = (
  mode: DocumentProcessingMode | undefined,
  t: Translate
): string => {
  if (mode === "ocr_pages") {
    return toDisplayText(
      t("playground:documentProcessing.ocrPages", "OCR pages")
    )
  }
  if (mode === "ingest_to_library") {
    return toDisplayText(
      t(
        "playground:documentProcessing.ingestToLibrary",
        "Ingest to library"
      )
    )
  }
  return toDisplayText(
    t("playground:documentProcessing.addToChat", "Add to chat")
  )
}
