import { useMemo, useState } from "react"
import { useCreateToken } from "@/hooks/useSharing"
import { useCreateOwnerBranchSession, useCreatePrototypeWorkspace } from "@/hooks/usePrototypeWorkspaces"
import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import type { PrototypeWorkspaceDetail } from "@/types/prototype-workspace"

interface PrototypeWorkspaceOwnerViewProps {
  prototypeWorkspaceId?: string | null
  workspace?: PrototypeWorkspaceDetail | null
}

export const PrototypeWorkspaceOwnerView = ({
  prototypeWorkspaceId,
  workspace
}: PrototypeWorkspaceOwnerViewProps) => {
  const activeWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.activeWorkspaceId
  )
  const setActiveWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.setActiveWorkspaceId
  )
  const ownerSessionId = usePrototypeWorkspaceStore((state) => state.ownerSessionId)
  const setOwnerSessionId = usePrototypeWorkspaceStore(
    (state) => state.setOwnerSessionId
  )

  const [title, setTitle] = useState("Prototype Workspace")
  const [prompt, setPrompt] = useState("")
  const [shareToken, setShareToken] = useState<string | null>(null)
  const createWorkspace = useCreatePrototypeWorkspace()

  const resolvedWorkspaceId = prototypeWorkspaceId ?? activeWorkspaceId
  const createOwnerSession = useCreateOwnerBranchSession(resolvedWorkspaceId ?? "")
  const createToken = useCreateToken()

  const shareUrl = useMemo(() => {
    if (!shareToken) {
      return null
    }
    const origin =
      typeof window !== "undefined" ? window.location.origin : "http://localhost"
    return `${origin}/share/${shareToken}`
  }, [shareToken])

  const handleCreateWorkspace = async () => {
    const workspace = await createWorkspace.mutateAsync({
      title,
      creation_source: "prompt",
      prompt: prompt || undefined
    })
    setActiveWorkspaceId(workspace.id)
  }

  const handleCreateOwnerSession = async () => {
    if (!resolvedWorkspaceId) {
      return
    }
    const session = await createOwnerSession.mutateAsync({})
    setOwnerSessionId(session.prototype_session_id)
  }

  const handleCreateShareLink = async () => {
    if (!resolvedWorkspaceId) {
      return
    }
    const token = await createToken.mutateAsync({
      resource_type: "prototype_workspace",
      resource_id: resolvedWorkspaceId,
      access_level: "full_edit",
      allow_clone: false
    })
    setShareToken(token.raw_token ?? token.token ?? token.token_prefix)
  }

  return (
    <div
      data-testid="prototype-workspace-owner-view"
      className="flex h-full w-full flex-col gap-4 overflow-auto p-4"
    >
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold">Prototype Workspace</h1>
        <p className="text-sm text-muted-foreground">
          Create a canonical prototype, start an owner session, and mint a private
          stakeholder link.
        </p>
      </div>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Create workspace</h2>
          <p className="text-sm text-muted-foreground">
            Seed a new prototype workspace from a prompt-oriented artifact brief.
          </p>
        </div>
        <div className="grid gap-3">
          <label className="grid gap-1 text-sm">
            <span>Title</span>
            <input
              className="rounded border px-3 py-2"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>
          <label className="grid gap-1 text-sm">
            <span>Prompt</span>
            <textarea
              className="min-h-24 rounded border px-3 py-2"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Describe the B2B artifact you want the prototype workspace to seed."
            />
          </label>
          <div className="flex flex-wrap gap-3">
            <button
              className="rounded bg-primary px-4 py-2 text-primary-foreground"
              onClick={() => void handleCreateWorkspace()}
              disabled={createWorkspace.isPending || !title.trim()}
            >
              {createWorkspace.isPending ? "Creating..." : "Create workspace"}
            </button>
            {resolvedWorkspaceId ? (
              <span className="self-center text-sm text-muted-foreground">
                Workspace: {resolvedWorkspaceId}
              </span>
            ) : null}
          </div>
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Canonical preview</h2>
          <p className="text-sm text-muted-foreground">
            Track the currently promoted prototype surface and its preview health.
          </p>
        </div>
        <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-3">
          <span>Preview status: {workspace?.canonical_preview_status ?? "unknown"}</span>
          <span>Canonical snapshot: {workspace?.canonical_snapshot_id ?? "not set"}</span>
          <span>Last known good: {workspace?.last_known_good_snapshot_id ?? "not set"}</span>
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Owner session</h2>
          <p className="text-sm text-muted-foreground">
            Boot an ACP-backed owner branch session against the canonical snapshot.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            className="rounded border px-4 py-2"
            onClick={() => void handleCreateOwnerSession()}
            disabled={createOwnerSession.isPending || !resolvedWorkspaceId}
          >
            {createOwnerSession.isPending ? "Starting..." : "Start owner session"}
          </button>
          {ownerSessionId ? (
            <span className="self-center text-sm text-muted-foreground">
              Session: {ownerSessionId}
            </span>
          ) : null}
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Private share link</h2>
          <p className="text-sm text-muted-foreground">
            Generate a prototype-specific private link that external stakeholders can
            exchange for their own collaborator session.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            className="rounded border px-4 py-2"
            onClick={() => void handleCreateShareLink()}
            disabled={createToken.isPending || !resolvedWorkspaceId}
          >
            {createToken.isPending ? "Creating..." : "Create private link"}
          </button>
          {shareUrl ? (
            <code className="self-center break-all rounded bg-muted px-2 py-1 text-xs">
              {shareUrl}
            </code>
          ) : null}
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Branch inventory</h2>
          <p className="text-sm text-muted-foreground">
            Review active branch sessions attached to this prototype workspace.
          </p>
        </div>
        <div className="space-y-2 text-sm">
          {(workspace?.sessions ?? []).length === 0 ? (
            <p className="text-muted-foreground">No branch sessions yet.</p>
          ) : (
            workspace?.sessions.map((session) => (
              <div
                key={session.id}
                className="rounded border px-3 py-2"
              >
                <div className="font-medium">{session.id}</div>
                <div className="text-muted-foreground">
                  {session.actor_type} · {session.runtime_status} · preview {session.preview_status}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Candidate revisions</h2>
          <p className="text-sm text-muted-foreground">
            Snapshot inventory for owner review and promotion decisions.
          </p>
        </div>
        <div className="space-y-2 text-sm">
          {(workspace?.snapshots ?? []).length === 0 ? (
            <p className="text-muted-foreground">No snapshots recorded yet.</p>
          ) : (
            workspace?.snapshots.map((snapshot) => (
              <div
                key={snapshot.snapshot_id}
                className="rounded border px-3 py-2"
              >
                <div className="font-medium">{snapshot.snapshot_id}</div>
                <div className="text-muted-foreground">
                  {snapshot.is_canonical ? "canonical" : "candidate"} ·{" "}
                  {snapshot.prompt_summary ?? "No summary provided"}
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  )
}
