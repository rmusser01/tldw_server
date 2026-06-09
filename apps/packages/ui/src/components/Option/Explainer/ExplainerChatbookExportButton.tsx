import { Download } from "lucide-react"

type ExplainerChatbookExportButtonProps = {
  disabled?: boolean
  isExporting?: boolean
  message?: string | null
  onExport: () => void
}

export const ExplainerChatbookExportButton = ({
  disabled = false,
  isExporting = false,
  message,
  onExport
}: ExplainerChatbookExportButtonProps) => (
  <div className="flex flex-wrap items-center gap-2">
    <button
      type="button"
      className="inline-flex h-9 items-center gap-2 rounded-md border border-border bg-surface px-3 text-sm font-semibold text-text transition-colors hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60"
      disabled={disabled || isExporting}
      onClick={onExport}
    >
      <Download className="h-4 w-4" aria-hidden="true" />
      Export to Chatbook
    </button>
    {message ? (
      <span role="status" className="text-xs font-medium text-text-muted">
        {message}
      </span>
    ) : null}
  </div>
)
