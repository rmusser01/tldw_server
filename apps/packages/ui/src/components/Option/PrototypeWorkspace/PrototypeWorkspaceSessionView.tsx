import { useState } from "react"
import { usePrototypePrivateLinkExchange } from "@/hooks/useSharing"
import {
  usePrototypeWorkspace,
  useCreateCollaboratorBranchSession,
  useCreatePromotionRequest
} from "@/hooks/usePrototypeWorkspaces"
import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import type { PrototypeWorkspaceDetail } from "@/types/prototype-workspace"

interface PrototypeWorkspaceSessionViewProps {
  prototypeWorkspaceId?: string | null
  sessionToken?: string | null
  shareToken?: string | null
  workspace?: PrototypeWorkspaceDetail | null
}

export const PrototypeWorkspaceSessionView = ({
  prototypeWorkspaceId,
  sessionToken,
  shareToken,
  workspace
}: PrototypeWorkspaceSessionViewProps) => {
  const activeWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.activeWorkspaceId
  )
  const collaboratorSessionId = usePrototypeWorkspaceStore(
    (state) => state.collaboratorSessionId
  )
  const collaboratorSessionToken = usePrototypeWorkspaceStore(
    (state) => state.collaboratorSessionToken
  )
  const setActiveWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.setActiveWorkspaceId
  )
  const setCollaboratorEntry = usePrototypeWorkspaceStore(
    (state) => state.setCollaboratorEntry
  )
  const lastPromotionRequestId = usePrototypeWorkspaceStore(
    (state) => state.lastPromotionRequestId
  )
  const setLastPromotionRequestId = usePrototypeWorkspaceStore(
    (state) => state.setLastPromotionRequestId
  )

  const [displayName, setDisplayName] = useState("Stakeholder")
  const [password, setPassword] = useState("")
  const [candidateSnapshotId, setCandidateSnapshotId] = useState("")
  const [requestReason, setRequestReason] = useState("")

  const exchangeLink = usePrototypePrivateLinkExchange()
  const createSession = useCreateCollaboratorBranchSession()
  const createPromotion = useCreatePromotionRequest()

  const effectiveSessionToken = sessionToken ?? collaboratorSessionToken
  const resolvedWorkspaceId =
    prototypeWorkspaceId ??
    activeWorkspaceId ??
    createSession.data?.prototype_workspace_id ??
    null
  const resolvedSessionId =
    collaboratorSessionId ?? createSession.data?.prototype_session_id ?? null
  const workspaceQuery = usePrototypeWorkspace(
    workspace ? null : resolvedWorkspaceId
  )
  const resolvedWorkspace = workspace ?? workspaceQuery.data ?? null

  const handleExchangeLink = async () => {
    if (!shareToken) {
      return
    }
    const result = await exchangeLink.mutateAsync({
      token: shareToken,
      display_name: displayName,
      password: password || undefined
    })
    setCollaboratorEntry({
      collaboratorSessionToken: result.session_token,
      collaboratorShareToken: shareToken,
      sharedActorId: result.shared_actor_id
    })
  }

  const handleCreateSession = async () => {
    if (!effectiveSessionToken) {
      return
    }
    const result = await createSession.mutateAsync({
      session_token: effectiveSessionToken
    })
    setActiveWorkspaceId(result.prototype_workspace_id)
    setCollaboratorEntry({
      collaboratorSessionId: result.prototype_session_id,
      collaboratorSessionToken: effectiveSessionToken,
      collaboratorShareToken: shareToken,
      sharedActorId: result.shared_actor_id ?? null
    })
  }

  const handleCreatePromotionRequest = async () => {
    if (!effectiveSessionToken || !resolvedWorkspaceId || !resolvedSessionId) {
      return
    }
    const promotion = await createPromotion.mutateAsync({
      prototype_workspace_id: resolvedWorkspaceId,
      prototype_session_id: resolvedSessionId,
      candidate_snapshot_id: candidateSnapshotId,
      session_token: effectiveSessionToken,
      request_reason: requestReason || undefined
    })
    setLastPromotionRequestId(promotion.id)
  }

  return (
    <div
      data-testid="prototype-workspace-session-view"
      className="flex h-full w-full flex-col gap-4 overflow-auto p-4"
    >
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold">Collaborator Session</h1>
        <p className="text-sm text-muted-foreground">
          Exchange a private stakeholder link, create a collaborator branch session,
          and submit a promotion request back to the canonical prototype.
        </p>
      </div>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Private link exchange</h2>
          <p className="text-sm text-muted-foreground">
            Use the public prototype share token to mint an external collaborator
            session token.
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="grid gap-1 text-sm">
            <span>Display name</span>
            <input
              className="rounded border px-3 py-2"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
            />
          </label>
          <label className="grid gap-1 text-sm">
            <span>Password</span>
            <input
              className="rounded border px-3 py-2"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap gap-3">
          <button
            className="rounded border px-4 py-2"
            onClick={() => void handleExchangeLink()}
            disabled={exchangeLink.isPending || !shareToken}
          >
            {exchangeLink.isPending ? "Exchanging..." : "Exchange private link"}
          </button>
          {effectiveSessionToken ? (
            <code className="self-center break-all rounded bg-muted px-2 py-1 text-xs">
              Session token: {effectiveSessionToken}
            </code>
          ) : null}
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Canonical preview</h2>
          <p className="text-sm text-muted-foreground">
            The current promoted prototype remains visible while collaborator
            revisions are developed in isolated branch sessions.
          </p>
        </div>
        <div className="grid gap-2 text-sm text-muted-foreground md:grid-cols-3">
          <span>Preview status: {resolvedWorkspace?.canonical_preview_status ?? "unknown"}</span>
          <span>Canonical snapshot: {resolvedWorkspace?.canonical_snapshot_id ?? "not set"}</span>
          <span>Last known good: {resolvedWorkspace?.last_known_good_snapshot_id ?? "not set"}</span>
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Collaborator branch session</h2>
          <p className="text-sm text-muted-foreground">
            Start or reuse the ACP-backed branch session for this stakeholder actor.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            className="rounded border px-4 py-2"
            onClick={() => void handleCreateSession()}
            disabled={createSession.isPending || !effectiveSessionToken}
          >
            {createSession.isPending ? "Starting..." : "Start collaborator session"}
          </button>
          {resolvedWorkspaceId ? (
            <span className="self-center text-sm text-muted-foreground">
              Workspace: {resolvedWorkspaceId}
            </span>
          ) : null}
          {resolvedSessionId ? (
            <span className="self-center text-sm text-muted-foreground">
              Session: {resolvedSessionId}
            </span>
          ) : null}
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Promotion request</h2>
          <p className="text-sm text-muted-foreground">
            Submit a candidate snapshot back to the owner review path once the
            collaborator branch produces a viable artifact revision.
          </p>
        </div>
        <div className="grid gap-3">
          <label className="grid gap-1 text-sm">
            <span>Candidate snapshot id</span>
            {resolvedWorkspace && resolvedWorkspace.snapshots.length > 0 ? (
              <select
                className="rounded border px-3 py-2"
                value={candidateSnapshotId}
                onChange={(event) => setCandidateSnapshotId(event.target.value)}
              >
                <option value="">Select a candidate snapshot</option>
                {resolvedWorkspace.snapshots
                  .filter((snapshot) => !snapshot.is_canonical)
                  .map((snapshot) => (
                    <option key={snapshot.snapshot_id} value={snapshot.snapshot_id}>
                      {snapshot.snapshot_id}
                    </option>
                  ))}
              </select>
            ) : (
              <input
                className="rounded border px-3 py-2"
                value={candidateSnapshotId}
                onChange={(event) => setCandidateSnapshotId(event.target.value)}
                placeholder="psnap_..."
              />
            )}
          </label>
          <label className="grid gap-1 text-sm">
            <span>Request reason</span>
            <textarea
              className="min-h-20 rounded border px-3 py-2"
              value={requestReason}
              onChange={(event) => setRequestReason(event.target.value)}
              placeholder="Summarize the stakeholder change request and why it should be promoted."
            />
          </label>
          <div className="flex flex-wrap gap-3">
            <button
              className="rounded border px-4 py-2"
              onClick={() => void handleCreatePromotionRequest()}
              disabled={
                createPromotion.isPending ||
                !effectiveSessionToken ||
                !resolvedWorkspaceId ||
                !resolvedSessionId ||
                !candidateSnapshotId.trim()
              }
            >
              {createPromotion.isPending
                ? "Submitting..."
                : "Submit promotion request"}
            </button>
            {lastPromotionRequestId ? (
              <span className="self-center text-sm text-muted-foreground">
                Promotion: {lastPromotionRequestId}
              </span>
            ) : null}
          </div>
        </div>
      </section>

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Snapshot candidates</h2>
          <p className="text-sm text-muted-foreground">
            Review candidate revisions that can be promoted back into the
            canonical prototype.
          </p>
        </div>
        <div className="space-y-2 text-sm">
          {(resolvedWorkspace?.snapshots ?? []).filter((snapshot) => !snapshot.is_canonical)
            .length === 0 ? (
            <p className="text-muted-foreground">No candidate snapshots available yet.</p>
          ) : (
            resolvedWorkspace?.snapshots
              .filter((snapshot) => !snapshot.is_canonical)
              .map((snapshot) => (
                <button
                  key={snapshot.snapshot_id}
                  type="button"
                  className="block w-full rounded border px-3 py-2 text-left"
                  onClick={() => setCandidateSnapshotId(snapshot.snapshot_id)}
                >
                  <div className="font-medium">{snapshot.snapshot_id}</div>
                  <div className="text-muted-foreground">
                    {snapshot.prompt_summary ?? "No summary provided"}
                  </div>
                </button>
              ))
          )}
        </div>
      </section>
    </div>
  )
}
