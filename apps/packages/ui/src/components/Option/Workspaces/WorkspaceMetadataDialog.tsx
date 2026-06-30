import React from "react"
import { Button } from "@/components/Common/Button"
import type { WorkspaceManagerItem } from "./workspace-manager-models"

type WorkspaceMetadataDialogProps = {
  item: WorkspaceManagerItem | null
  submitting?: boolean
  error?: string | null
  onClose: () => void
  onSubmit: (item: WorkspaceManagerItem, name: string) => Promise<void> | void
}

export const WorkspaceMetadataDialog = ({
  item,
  submitting = false,
  error,
  onClose,
  onSubmit
}: WorkspaceMetadataDialogProps) => {
  const [name, setName] = React.useState("")

  React.useEffect(() => {
    setName(item?.name ?? "")
  }, [item])

  if (!item) return null

  const trimmedName = name.trim()

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="workspace-metadata-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <form
        className="w-full max-w-md rounded-lg border border-border bg-bg p-4 shadow-xl"
        onSubmit={(event) => {
          event.preventDefault()
          if (!trimmedName || submitting) return
          void onSubmit(item, trimmedName)
        }}
      >
        <div className="space-y-1">
          <h2
            id="workspace-metadata-title"
            className="text-base font-semibold text-text"
          >
            Edit Workspace
          </h2>
          <p className="text-sm text-text-muted">
            Rename the server-backed Workspace without changing sources, notes, or roots.
          </p>
        </div>

        <label
          className="mt-4 block text-sm font-medium text-text"
          htmlFor="workspace-metadata-name"
        >
          Workspace name
        </label>
        <input
          id="workspace-metadata-name"
          className={[
            "mt-1 w-full rounded-md border border-border bg-surface",
            "px-3 py-2 text-sm text-text outline-none",
            "focus:border-primary focus:ring-2 focus:ring-primary/20"
          ].join(" ")}
          value={name}
          onChange={(event) => setName(event.target.value)}
          autoFocus
        />

        {error && <p className="mt-3 text-sm text-danger">{error}</p>}

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose} disabled={submitting}>
            Cancel
          </Button>
          <Button
            variant="primary"
            htmlType="submit"
            disabled={!trimmedName}
            loading={submitting}
          >
            Save metadata
          </Button>
        </div>
      </form>
    </div>
  )
}
