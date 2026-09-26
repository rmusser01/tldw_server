import { useEffect, useMemo, useState } from "react"
import { useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceRenameDraft } from "@/store/owned-workspace-state"
import { createOwnedWorkspaceMetadataContext } from "@/services/owned-workspace-opening"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"

type RenameError = "conflict" | "accountChanged" | "saveFailed" | "reviewFailed"
type Review = {
  workspace: WorkspaceApiResponse
  expectedWorkspace: WorkspaceApiResponse
}

/** The store owns the editor; requests and inspected conflicts belong to this mount. */
export function useOwnedWorkspaceRename() {
  const origin = useWorkspaceStore((s) => s.activeWorkspaceOrigin)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const attempt = useWorkspaceStore((s) => s.ownedWorkspaceAttempt)
  const draft = useWorkspaceStore((s) => s.ownedWorkspaceRenameDraft)
  const session = useWorkspaceStore((s) => s.ownedWorkspaceRenameSession)
  const storageStatus = useWorkspaceStore((s) => s.ownedWorkspaceDraftStatus)
  const enabled = origin?.kind === "server-owned"
  const lifetime = useMemo(
    () => ({
      origin,
      workspaceId,
      attempt,
      controller: new AbortController(),
      pending: false
    }),
    [origin, workspaceId, attempt]
  )
  const [activity, setActivity] = useState<typeof lifetime | null>(null)
  const [feedback, setFeedback] = useState<{
    lifetime: typeof lifetime
    session: number
    error: RenameError | null
    review: Review | null
  } | null>(null)

  useEffect(() => {
    if (!enabled) return
    // React StrictMode may set up this same effect again after its cleanup.
    if (lifetime.controller.signal.aborted)
      lifetime.controller = new AbortController()
    const controller = lifetime.controller
    const unsubscribe = useWorkspaceStore.subscribe((state) => {
      if (
        state.activeWorkspaceOrigin !== origin ||
        state.workspaceId !== workspaceId ||
        state.ownedWorkspaceAttempt !== attempt
      )
        controller.abort()
    })
    return () => {
      controller.abort()
      unsubscribe()
    }
  }, [enabled, lifetime, origin, workspaceId, attempt])

  const current = () => {
    const state = useWorkspaceStore.getState()
    return (
      enabled &&
      !lifetime.controller.signal.aborted &&
      state.activeWorkspaceOrigin === lifetime.origin &&
      state.workspaceId === lifetime.workspaceId &&
      state.ownedWorkspaceAttempt === lifetime.attempt &&
      !lifetime.attempt
    )
  }
  const editorCurrent = () =>
    current() &&
    useWorkspaceStore.getState().ownedWorkspaceRenameSession === session
  const visible =
    feedback?.lifetime === lifetime && feedback.session === session
      ? feedback
      : null
  const error = visible?.error ?? null
  const review = visible?.review ?? null
  const report = (error: RenameError | null, review: Review | null = null) => {
    if (editorCurrent()) setFeedback({ lifetime, session, error, review })
  }
  const setDraft = (next: OwnedWorkspaceRenameDraft | null) => {
    if (!editorCurrent()) return false
    const state = useWorkspaceStore.getState()
    return state.setOwnedWorkspaceRenameDraft({
      origin,
      workspaceId,
      expectedDraft: state.ownedWorkspaceRenameDraft,
      draft: next
    })
  }
  const start = () => {
    if (!editorCurrent()) return
    const state = useWorkspaceStore.getState()
    const workspace = state.ownedWorkspaceBundle?.workspace
    if (!state.ownedWorkspaceRenameDraft && workspace)
      setDraft({ name: workspace.name, baseVersion: workspace.version })
  }
  const change = (name: string) => {
    if (!editorCurrent()) return
    const draft = useWorkspaceStore.getState().ownedWorkspaceRenameDraft
    if (draft) setDraft({ ...draft, name })
  }
  const cancel = () => {
    setDraft(null)
  }

  const save = async () => {
    if (
      !editorCurrent() ||
      lifetime.pending ||
      error === "conflict" ||
      error === "reviewFailed" ||
      error === "accountChanged" ||
      review
    )
      return
    const state = useWorkspaceStore.getState()
    const submittedDraft = state.ownedWorkspaceRenameDraft
    const expectedWorkspace = state.ownedWorkspaceBundle?.workspace
    if (
      !submittedDraft?.name.trim() ||
      !expectedWorkspace ||
      origin.kind !== "server-owned"
    )
      return
    const controller = lifetime.controller
    lifetime.pending = true
    setActivity(lifetime)
    report(null)
    try {
      const context = await createOwnedWorkspaceMetadataContext(
        workspaceId,
        origin.scope,
        controller.signal
      )
      if (!editorCurrent() || controller.signal.aborted) return
      const workspace = await context.patch({
        name: submittedDraft.name.trim(),
        version: submittedDraft.baseVersion
      })
      if (!current() || controller.signal.aborted) return
      const result = useWorkspaceStore
        .getState()
        .applyOwnedWorkspaceRenameReceipt({
          origin,
          workspaceId,
          expectedWorkspace,
          submittedDraft,
          editorSession: session,
          workspace
        })
      if (result === "stale") report("saveFailed")
    } catch (error) {
      if (!current() || controller.signal.aborted) return
      const status = (error as { status?: number })?.status
      report(
        status === 409
          ? "conflict"
          : status === 412
            ? "accountChanged"
            : "saveFailed"
      )
    } finally {
      lifetime.pending = false
      if (current() && !controller.signal.aborted) setActivity(null)
    }
  }

  const inspect = async () => {
    if (!editorCurrent() || lifetime.pending || origin.kind !== "server-owned")
      return
    const state = useWorkspaceStore.getState()
    const expectedWorkspace = state.ownedWorkspaceBundle?.workspace
    if (!state.ownedWorkspaceRenameDraft || !expectedWorkspace) return
    const controller = lifetime.controller
    lifetime.pending = true
    setActivity(lifetime)
    try {
      const context = await createOwnedWorkspaceMetadataContext(
        workspaceId,
        origin.scope,
        controller.signal
      )
      if (!editorCurrent() || controller.signal.aborted) return
      const workspace = await context.get()
      if (!editorCurrent() || controller.signal.aborted) return
      const state = useWorkspaceStore.getState()
      if (
        state.ownedWorkspaceBundle?.workspace !== expectedWorkspace ||
        !state.ownedWorkspaceRenameDraft ||
        workspace.version < state.ownedWorkspaceRenameDraft.baseVersion ||
        workspace.version < expectedWorkspace.version
      ) {
        report("reviewFailed")
        return
      }
      report(
        workspace.version > state.ownedWorkspaceRenameDraft.baseVersion
          ? "conflict"
          : null,
        { workspace, expectedWorkspace }
      )
    } catch (error) {
      if (!editorCurrent() || controller.signal.aborted) return
      report(
        (error as { status?: number })?.status === 412
          ? "accountChanged"
          : "reviewFailed"
      )
    } finally {
      lifetime.pending = false
      if (current() && !controller.signal.aborted) setActivity(null)
    }
  }
  const keepMine = () => {
    if (!editorCurrent() || lifetime.pending || !review) return
    const state = useWorkspaceStore.getState()
    const draft = state.ownedWorkspaceRenameDraft
    if (
      !draft ||
      state.ownedWorkspaceBundle?.workspace !== review.expectedWorkspace
    ) {
      report("reviewFailed")
      return
    }
    if (setDraft({ ...draft, baseVersion: review.workspace.version }))
      report(null)
  }
  const useServer = () => {
    if (!editorCurrent() || lifetime.pending || !review) return
    const state = useWorkspaceStore.getState()
    const submittedDraft = state.ownedWorkspaceRenameDraft
    if (!submittedDraft) return
    try {
      const result = state.acceptOwnedWorkspaceRenameReview({
        origin,
        workspaceId,
        expectedWorkspace: review.expectedWorkspace,
        submittedDraft,
        editorSession: session,
        workspace: review.workspace
      })
      if (result === "stale") report("reviewFailed")
    } catch {
      report("reviewFailed")
    }
  }

  const pending = activity === lifetime
  return {
    enabled,
    draft,
    pending,
    error,
    remote: review?.workspace ?? null,
    storageUnavailable:
      !!draft &&
      (storageStatus === "unavailable" || storageStatus === "invalid"),
    canSave:
      !!draft?.name.trim() &&
      !pending &&
      !review &&
      error !== "conflict" &&
      error !== "reviewFailed" &&
      error !== "accountChanged",
    start,
    change,
    cancel,
    save,
    review: inspect,
    keepMine,
    useServer
  }
}
