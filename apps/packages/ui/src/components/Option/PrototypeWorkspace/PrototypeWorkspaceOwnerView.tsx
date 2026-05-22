import { useMemo, useState } from "react"
import { useCreateToken } from "@/hooks/useSharing"
import {
  useCreateOwnerBranchSession,
  useCreatePrototypeWorkspace,
  useReviewPrototypePromotionRequest
} from "@/hooks/usePrototypeWorkspaces"
import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import type {
  PrototypePromotionRequest,
  PrototypeWorkspaceDetail,
  PrototypeWorkspaceSessionSummary,
  PrototypeWorkspaceSnapshotSummary
} from "@/types/prototype-workspace"

interface PrototypeWorkspaceOwnerViewProps {
  prototypeWorkspaceId?: string | null
  workspace?: PrototypeWorkspaceDetail | null
}

const prototypeReviewStateCopy: Record<string, string> = {
  pending: "Pending owner review",
  approved: "Approved",
  rejected: "Rejected",
  promoted: "Promoted",
  stale: "Stale candidate",
  promotion_stale: "Stale candidate",
  conflict: "Promotion conflict",
  promotion_conflict: "Promotion conflict",
  validation_running: "Validation running",
  validation_failed: "Validation failed",
  promotion_validation_failed: "Validation failed",
  failed: "Promotion failed",
  promotion_failed: "Promotion failed"
}

const normalizePrototypeState = (value: string | null | undefined) =>
  (value ?? "").trim().toLowerCase().replace(/[-\s]+/g, "_")

const toDisplayLabel = (value: string) =>
  value
    .split(/[_-\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")

const getPrototypeReviewStateCopy = (status: string | null | undefined) => {
  const normalized = normalizePrototypeState(status)
  return (
    prototypeReviewStateCopy[normalized] ??
    (normalized ? toDisplayLabel(normalized) : null)
  )
}

const getPromotionStatusCopy = (status: string | null | undefined) =>
  getPrototypeReviewStateCopy(status) ?? "Unknown review state"

const getReviewResultCopy = (
  status: string | null | undefined,
  failureCode?: string | null
) => {
  const failureState = normalizePrototypeState(failureCode)
  if (
    failureState === "publish_validation_failed" ||
    failureState === "promotion_validation_failed" ||
    failureState === "validation_failed"
  ) {
    return "Validation failed"
  }
  return getPrototypeReviewStateCopy(status) ?? "Review completed"
}

const isSessionActionable = (session: PrototypeWorkspaceSessionSummary) => {
  if (session.is_revoked || session.revoked_at) {
    return false
  }
  if (!session.expires_at) {
    return true
  }
  return new Date(session.expires_at).getTime() > Date.now()
}

const getReviewResultReason = (details: Record<string, unknown> | undefined) => {
  const reason = details?.reason
  return typeof reason === "string" && reason.trim() ? reason : null
}

const getReviewErrorTitle = (detail?: {
  frontend_state?: string
  category?: string
}) =>
  getPrototypeReviewStateCopy(detail?.frontend_state) ??
  getPrototypeReviewStateCopy(detail?.category) ??
  "Review failed"

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
  const reviewPromotion = useReviewPrototypePromotionRequest()

  const sessionsById = useMemo(() => {
    const map = new Map<string, PrototypeWorkspaceSessionSummary>()
    for (const session of workspace?.sessions ?? []) {
      map.set(session.id, session)
    }
    return map
  }, [workspace?.sessions])

  const snapshotsById = useMemo(() => {
    const map = new Map<string, PrototypeWorkspaceSnapshotSummary>()
    for (const snapshot of workspace?.snapshots ?? []) {
      map.set(snapshot.snapshot_id, snapshot)
    }
    return map
  }, [workspace?.snapshots])

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

  const canApprovePromotion = (promotion: PrototypePromotionRequest) => {
    const status = promotion.status.toLowerCase()
    const session = sessionsById.get(promotion.prototype_session_id)
    return (
      status === "pending" &&
      !!session &&
      isSessionActionable(session) &&
      !!snapshotsById.get(promotion.candidate_snapshot_id) &&
      !reviewPromotion.isPending &&
      !!resolvedWorkspaceId
    )
  }

  const canRejectPromotion = (promotion: PrototypePromotionRequest) =>
    promotion.status.toLowerCase() === "pending" &&
    !reviewPromotion.isPending &&
    !!resolvedWorkspaceId

  const handleReviewPromotion = async (
    promotion: PrototypePromotionRequest,
    decision: "approve" | "reject"
  ) => {
    const canSubmit =
      decision === "approve"
        ? canApprovePromotion(promotion)
        : canRejectPromotion(promotion)
    if (!resolvedWorkspaceId || !canSubmit) {
      return
    }
    await reviewPromotion.mutateAsync({
      promotion_request_id: promotion.id,
      prototype_workspace_id: resolvedWorkspaceId,
      decision,
      ...(decision === "approve" && workspace?.canonical_snapshot_id
        ? { review_baseline_snapshot_id: workspace.canonical_snapshot_id }
        : {})
    })
  }

  const reviewError = reviewPromotion.error as
    | {
        detail?: {
          message?: string
          frontend_state?: string
          category?: string
        }
        message?: string
      }
    | null
  const reviewResultReason = reviewPromotion.data
    ? getReviewResultReason(reviewPromotion.data.details)
    : null

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
                data-testid={`prototype-branch-session-${session.id}`}
                className="rounded border px-3 py-2"
              >
                <div className="font-medium">{session.id}</div>
                <div className="grid gap-1 text-muted-foreground md:grid-cols-4">
                  <span>{session.actor_type}</span>
                  <span>Runtime {session.runtime_status}</span>
                  <span>Preview {session.preview_status}</span>
                  <span>{isSessionActionable(session) ? "Actionable" : "Not actionable"}</span>
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

      <section className="rounded-lg border p-4">
        <div className="mb-3 space-y-1">
          <h2 className="font-medium">Promotion review</h2>
          <p className="text-sm text-muted-foreground">
            Review collaborator candidate snapshots without mixing review state with
            runtime or preview health.
          </p>
        </div>
        {reviewPromotion.data ? (
          <div
            data-testid="prototype-promotion-review-result"
            className="mb-3 rounded border px-3 py-2 text-sm"
          >
            <div className="font-medium">
              {getReviewResultCopy(
                reviewPromotion.data.status,
                reviewPromotion.data.failure_code
              )}
            </div>
            <div className="text-muted-foreground">
              Candidate {reviewPromotion.data.candidate_snapshot_id}
              {reviewPromotion.data.failure_code
                ? ` · ${reviewPromotion.data.failure_code}`
                : null}
              {reviewPromotion.data.preview_handle
                ? ` · preview ${reviewPromotion.data.preview_handle}`
                : null}
            </div>
            {reviewResultReason ? (
              <div className="text-muted-foreground">
                {reviewResultReason}
              </div>
            ) : null}
          </div>
        ) : null}
        {reviewError ? (
          <div className="mb-3 rounded border border-destructive px-3 py-2 text-sm">
            <div className="font-medium">
              {getReviewErrorTitle(reviewError.detail)}
            </div>
            <div className="text-muted-foreground">
              {reviewError.detail?.message ?? reviewError.message ?? "Promotion review failed"}
            </div>
            {reviewError.detail?.category ? (
              <div className="text-muted-foreground">
                {getPrototypeReviewStateCopy(reviewError.detail.category) ??
                  reviewError.detail.category}
              </div>
            ) : null}
          </div>
        ) : null}
        <div className="space-y-2 text-sm">
          {(workspace?.promotion_requests ?? []).length === 0 ? (
            <p className="text-muted-foreground">No promotion requests yet.</p>
          ) : (
            workspace?.promotion_requests.map((promotion) => {
              const candidate = snapshotsById.get(promotion.candidate_snapshot_id)
              const session = sessionsById.get(promotion.prototype_session_id)
              const canApprove = canApprovePromotion(promotion)
              const canReject = canRejectPromotion(promotion)
              return (
                <div
                  key={promotion.id}
                  data-testid={`prototype-promotion-request-${promotion.id}`}
                  className="rounded border px-3 py-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="font-medium">{promotion.id}</div>
                      <div className="text-muted-foreground">
                        {getPromotionStatusCopy(promotion.status)}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        className="rounded border px-3 py-1"
                        aria-label={`Approve promotion ${promotion.id}`}
                        onClick={() => void handleReviewPromotion(promotion, "approve")}
                        disabled={!canApprove}
                      >
                        Approve
                      </button>
                      <button
                        className="rounded border px-3 py-1"
                        aria-label={`Reject promotion ${promotion.id}`}
                        onClick={() => void handleReviewPromotion(promotion, "reject")}
                        disabled={!canReject}
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                  <div className="mt-2 grid gap-1 text-muted-foreground md:grid-cols-2">
                    <span>Candidate {promotion.candidate_snapshot_id}</span>
                    <span>Session {promotion.prototype_session_id}</span>
                    <span>
                      {promotion.requested_by_shared_actor_id
                        ? `Shared actor ${promotion.requested_by_shared_actor_id}`
                        : `User ${promotion.requested_by_user_id ?? "unknown"}`}
                    </span>
                    <span>
                      {candidate ? "Candidate snapshot present" : "Candidate snapshot missing"}
                    </span>
                    <span>
                      {session ? "Branch session present" : "Branch session missing"}
                    </span>
                    {promotion.review_notes ? <span>{promotion.review_notes}</span> : null}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </section>
    </div>
  )
}
