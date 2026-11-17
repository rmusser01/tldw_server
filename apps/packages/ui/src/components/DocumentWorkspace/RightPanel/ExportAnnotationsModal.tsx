import { useDocumentOperation } from "@/hooks/document-workspace/useDocumentOperation"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { createDocumentClient } from "@/services/document-generation"
import { downloadBlob } from "@/utils/download-blob"
import { formatErrorMessage } from "@/utils/format-error-message"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import { Modal, Radio, Button, Alert } from "antd"
import type { RadioChangeEvent } from "antd"
import { FileText, FileJson, File } from "lucide-react"
import type { Annotation, DocumentType } from "../types"
import {
  downloadAnnotations,
  type ExportFormat
} from "../utils/annotationExport"

interface ExportAnnotationsModalProps {
  open: boolean
  onClose: () => void
  annotations: Annotation[]
  documentTitle: string
  documentType: DocumentType | null
  documentId?: number | null
  hasPendingWrites?: boolean
}

const FORMAT_OPTIONS: Array<{
  value: ExportFormat
  label: string
  description: string
  icon: React.ReactNode
}> = [
  {
    value: "markdown",
    label: "Markdown",
    description: "Formatted text with headings and quotes",
    icon: <FileText className="h-5 w-5" />
  },
  {
    value: "json",
    label: "JSON",
    description: "Structured data for programmatic use",
    icon: <FileJson className="h-5 w-5" />
  },
  {
    value: "text",
    label: "Plain Text",
    description: "Simple text format",
    icon: <File className="h-5 w-5" />
  }
]

/**
 * Modal for selecting annotation export format and triggering download.
 */
export const ExportAnnotationsModal: React.FC<ExportAnnotationsModalProps> = ({
  open,
  onClose,
  annotations,
  documentTitle,
  documentType,
  documentId,
  hasPendingWrites = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const [format, setFormat] = useState<ExportFormat | "pdf">("markdown")

  const handleFormatChange = (e: RadioChangeEvent) => {
    setFormat(e.target.value)
  }

  const operation = useDocumentOperation()
  const pending = useDocumentWorkspaceStore((state) => state.pendingAnnotations)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const work = React.useRef<AbortController | null>(null)
  React.useEffect(() => {
    setExporting(false)
    setError(null)
    return () => {
      work.current?.abort()
      work.current = null
    }
  }, [open, documentId, operation])
  const unsaved =
    hasPendingWrites || pending.some((item) => item.documentId === documentId)
  const disabled =
    exporting ||
    (format === "pdf"
      ? unsaved || !documentId || !operation?.isCurrent()
      : annotations.length === 0)
  const handleExport = async () => {
    if (work.current || disabled || !operation?.isCurrent()) return
    const controller = new AbortController()
    work.current = controller
    setExporting(true)
    setError(null)
    try {
      if (format === "pdf") {
        if (documentType !== "pdf" || !documentId || !operation?.isCurrent())
          return
        const blob = await createDocumentClient(operation).exportPdf(
          documentId,
          controller.signal
        )
        if (!operation.isCurrent() || controller.signal.aborted) return
        downloadBlob(blob, `document-${documentId}-annotated.pdf`)
      } else {
        downloadAnnotations(annotations, documentTitle, documentType, format)
      }
      if (!controller.signal.aborted) onClose()
    } catch (failure) {
      if (!controller.signal.aborted && operation?.isCurrent())
        setError(
          formatErrorMessage(
            failure,
            "Export failed. The original PDF and saved annotations are unchanged."
          )
        )
    } finally {
      if (work.current === controller) {
        work.current = null
        setExporting(false)
      }
    }
  }
  const formats =
    documentType === "pdf"
      ? [
          {
            value: "pdf" as const,
            label: "PDF with saved annotations",
            description:
              "Original pages with saved highlights and notes as PDF page comments.",
            icon: <File className="h-5 w-5" />
          },
          ...FORMAT_OPTIONS
        ]
      : FORMAT_OPTIONS

  return (
    <Modal
      title={t(
        "option:documentWorkspace.exportAnnotations",
        "Export Annotations"
      )}
      open={open}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>
          {t("common:cancel", "Cancel")}
        </Button>,
        <Button
          key="export"
          type="primary"
          onClick={handleExport}
          disabled={disabled}
          loading={exporting}
        >
          {t("option:documentWorkspace.export", "Export")}
        </Button>
      ]}
      width={420}
    >
      <div className="space-y-4">
        {error && <Alert type="error" showIcon title={error} />}
        {format === "pdf" && unsaved && (
          <Alert
            type="warning"
            showIcon
            title="Wait for annotation changes to save before exporting the saved PDF."
          />
        )}

        {/* Annotation count */}
        <div className="rounded-lg border border-border bg-surface-hover p-3">
          <p className="text-sm">
            <span className="font-medium">{annotations.length}</span>{" "}
            {t(
              "option:documentWorkspace.annotationsToExport",
              "annotations to export"
            )}
          </p>
          <p className="mt-1 text-xs text-text-secondary">{documentTitle}</p>
        </div>

        {/* Format selection */}
        <div>
          <p className="mb-2 text-sm font-medium">
            {t("option:documentWorkspace.selectFormat", "Select format")}
          </p>
          <Radio.Group
            value={format}
            onChange={handleFormatChange}
            className="w-full"
          >
            <div className="space-y-2">
              {formats.map((option) => (
                <Radio
                  key={option.value}
                  value={option.value}
                  className={`!m-0 !flex w-full cursor-pointer !items-start gap-3 rounded-lg border p-3 transition-colors ${
                    format === option.value
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-text-secondary">{option.icon}</span>
                    <div>
                      <p className="font-medium">{option.label}</p>
                      <p className="text-xs text-text-secondary">
                        {option.description}
                      </p>
                    </div>
                  </div>
                </Radio>
              ))}
            </div>
          </Radio.Group>
        </div>

        {/* Empty state warning */}
        {annotations.length === 0 && (
          <div className="rounded border border-warn/30 bg-warn/10 p-3 text-sm text-warn">
            {t(
              "option:documentWorkspace.noAnnotationsToExport",
              "No annotations to export. Add some highlights or notes first."
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}

export default ExportAnnotationsModal
