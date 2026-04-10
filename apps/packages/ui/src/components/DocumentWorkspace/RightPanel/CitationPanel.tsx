import React, { useState, useCallback } from "react"
import { useTranslation } from "react-i18next"
import { Empty, Spin, Segmented, Button, Tooltip, message } from "antd"
import { Copy, Check, Quote } from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import {
  useCitation,
  CITATION_FORMAT_INFO,
  type CitationFormat
} from "@/hooks/document-workspace/useCitation"

/**
 * CitationPanel - Generate and copy document citations.
 *
 * Features:
 * - Multiple citation formats (MLA, APA, Chicago, Harvard, IEEE)
 * - One-click copy to clipboard
 * - Uses document metadata for accurate citations
 */
export const CitationPanel: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)

  const [format, setFormat] = useState<CitationFormat>("apa")
  const [copied, setCopied] = useState(false)

  const { getCitation, isLoading, error, metadata } = useCitation(activeDocumentId)

  const citation = getCitation(format)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(citation)
      setCopied(true)
      message.success(t("option:documentWorkspace.citationCopied", "Citation copied to clipboard"))
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      message.error(t("option:documentWorkspace.copyFailed", "Failed to copy"))
    }
  }, [citation, t])

  const handleFormatChange = (value: string | number) => {
    setFormat(value as CitationFormat)
    setCopied(false)
  }

  // No document selected
  if (!activeDocumentId) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Empty
          image={<Quote className="h-12 w-12 text-muted mx-auto mb-2" />}
          description={t(
            "option:documentWorkspace.noDocumentForCitation",
            "Generate formatted citations (APA, MLA, Chicago, and more) from your document's metadata."
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
          {t("option:documentWorkspace.loadMetadataError", "Failed to load document metadata")}
        </p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      {/* Format selector */}
      <div className="border-b border-border p-3">
        <div className="mb-2 text-xs font-medium text-text-secondary">
          {t("option:documentWorkspace.citationFormat", "Citation Format")}
        </div>
        <Segmented
          value={format}
          onChange={handleFormatChange}
          options={Object.entries(CITATION_FORMAT_INFO).map(([key, info]) => ({
            value: key,
            label: info.label
          }))}
          size="small"
          block
        />
        <p className="mt-1.5 text-xs text-text-muted">
          {CITATION_FORMAT_INFO[format].description}
        </p>
      </div>

      {/* Citation display */}
      <div className="flex-1 overflow-y-auto p-3">
        <div className="space-y-3">
          {/* Document info */}
          {metadata && (
            <div className="rounded-lg border border-border bg-surface-hover p-3">
              <p className="text-sm font-medium">{metadata.title}</p>
              {metadata.authors && metadata.authors.length > 0 && (
                <p className="mt-1 text-xs text-text-secondary">
                  {metadata.authors.join(", ")}
                </p>
              )}
            </div>
          )}

          {/* Citation text */}
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-xs font-medium text-text-secondary">
                {CITATION_FORMAT_INFO[format].label} {t("option:documentWorkspace.citation", "Citation")}
              </span>
              <Tooltip title={copied ? t("common:copied", "Copied!") : t("common:copy", "Copy")}>
                <Button
                  size="small"
                  type="text"
                  icon={
                    copied ? (
                      <Check className="h-3.5 w-3.5 text-success" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )
                  }
                  onClick={handleCopy}
                />
              </Tooltip>
            </div>
            <div
              className="rounded-lg border border-border bg-surface p-3 text-sm leading-relaxed cursor-pointer hover:border-primary/50 transition-colors"
              onClick={handleCopy}
            >
              {citation}
            </div>
            <p className="mt-2 text-[10px] text-text-muted text-center">
              {t("option:documentWorkspace.clickToCopy", "Click to copy")}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CitationPanel
