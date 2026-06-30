import React from "react"
import { Button } from "@/components/Common/Button"
import type { WorkspaceProfile } from "@/services/tldw/domains/workspace-api"

type WorkspaceCreateDialogProps = {
  open: boolean
  profile: WorkspaceProfile
  submitting?: boolean
  error?: string | null
  onClose: () => void
  onSubmit: (name: string, profile: WorkspaceProfile) => Promise<void> | void
}

const profileTitle: Record<WorkspaceProfile, string> = {
  research: "New Research Workspace",
  project: "New Project Workspace"
}

const profileDescription: Record<WorkspaceProfile, string> = {
  research: "Create a server-backed research workspace for sources, notes, and grounded QA.",
  project: "Create a Project Workspace shell. Root setup happens from the project panel."
}

export const WorkspaceCreateDialog = ({
  open,
  profile,
  submitting = false,
  error,
  onClose,
  onSubmit
}: WorkspaceCreateDialogProps) => {
  const [name, setName] = React.useState("")

  React.useEffect(() => {
    if (open) setName("")
  }, [open, profile])

  if (!open) return null

  const trimmedName = name.trim()

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="workspace-create-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <form
        className="w-full max-w-md rounded-lg border border-border bg-bg p-4 shadow-xl"
        onSubmit={(event) => {
          event.preventDefault()
          if (!trimmedName || submitting) return
          void onSubmit(trimmedName, profile)
        }}
      >
        <div className="space-y-1">
          <h2 id="workspace-create-title" className="text-base font-semibold text-text">
            {profileTitle[profile]}
          </h2>
          <p className="text-sm text-text-muted">{profileDescription[profile]}</p>
        </div>

        <label className="mt-4 block text-sm font-medium text-text" htmlFor="workspace-create-name">
          Workspace name
        </label>
        <input
          id="workspace-create-name"
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
            Create Workspace
          </Button>
        </div>
      </form>
    </div>
  )
}
