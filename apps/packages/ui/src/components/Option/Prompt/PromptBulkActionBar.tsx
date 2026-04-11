import React from "react"
import { Download, Star, Tag, Trash2 } from "lucide-react"

type PromptBulkActionBarV1Props = {
  mode?: "v1"
  selectedCount: number
  disabled?: boolean
  onBulkExport?: () => void
  onBulkTag?: () => void
  onBulkFavoriteToggle?: () => void
  onBulkDelete?: () => void
  onClearSelection: () => void
}

type PromptBulkActionBarLegacyProps = {
  mode: "legacy"
  children: React.ReactNode
  className?: string
  testId?: string
}

type PromptBulkActionBarProps =
  | PromptBulkActionBarV1Props
  | PromptBulkActionBarLegacyProps

export const PromptBulkActionBar: React.FC<PromptBulkActionBarProps> = (props) => {
  if (props.mode === "legacy") {
    return (
      <div
        data-testid={props.testId || "prompts-bulk-action-bar-legacy"}
        className={
          props.className ||
          "flex flex-wrap items-center gap-2 rounded-md border border-primary/30 bg-primary/10 p-2"
        }
      >
        {props.children}
      </div>
    )
  }

  const {
    selectedCount,
    disabled = false,
    onBulkExport,
    onBulkTag,
    onBulkFavoriteToggle,
    onBulkDelete,
    onClearSelection
  } = props

  if (selectedCount === 0) {
    return null
  }

  return (
    <div
      data-testid="prompts-bulk-action-bar-scaffold"
      className="flex flex-wrap items-center gap-2 rounded-md border border-primary/30 bg-primary/10 p-2"
    >
      <span className="text-sm text-primary">{selectedCount} selected</span>
      <button
        type="button"
        disabled={disabled}
        onClick={onBulkExport}
        className="inline-flex items-center gap-1 rounded border border-primary/30 px-2 py-1 text-sm text-primary hover:bg-primary/10 disabled:opacity-50"
      >
        <Download className="size-3" />
        Export
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={onBulkTag}
        className="inline-flex items-center gap-1 rounded border border-primary/30 px-2 py-1 text-sm text-primary hover:bg-primary/10 disabled:opacity-50"
      >
        <Tag className="size-3" />
        Add tag
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={onBulkFavoriteToggle}
        className="inline-flex items-center gap-1 rounded border border-primary/30 px-2 py-1 text-sm text-primary hover:bg-primary/10 disabled:opacity-50"
      >
        <Star className="size-3" />
        Favorite
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={onBulkDelete}
        className="inline-flex items-center gap-1 rounded border border-danger/30 px-2 py-1 text-sm text-danger hover:bg-danger/10 disabled:opacity-50"
      >
        <Trash2 className="size-3" />
        Delete
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={onClearSelection}
        className="ml-auto rounded px-2 py-1 text-sm text-text-muted hover:text-text disabled:opacity-50"
      >
        Clear selection
      </button>
    </div>
  )
}
