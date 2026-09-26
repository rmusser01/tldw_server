import { useEffect, useMemo, useReducer } from "react"
import { useNavigate } from "react-router-dom"
import { useWorkspaceStore } from "@/store/workspace"
import { createOwnedWorkspaceLifecycleContext } from "@/services/owned-workspace-opening"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"
import { WORKSPACES_PATH } from "@/routes/route-paths"

type ArchiveError =
  | "conflict"
  | "accountChanged"
  | "uncertain"
  | "reviewFailed"
  | "storageUnavailable"
  | "completionFailed"
type Confirmation = { workspace: WorkspaceApiResponse }

/** A dismissed confirmation cannot erase uncertainty about an already dispatched write. */
export function useOwnedWorkspaceArchive() {
  const navigate = useNavigate()
  const origin = useWorkspaceStore((s) => s.activeWorkspaceOrigin)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const attempt = useWorkspaceStore((s) => s.ownedWorkspaceAttempt)
  const enabled = origin?.kind === "server-owned"
  const [, render] = useReducer((n: number) => n + 1, 0)
  const lifetime = useMemo(
    () => ({
      origin,
      workspaceId,
      attempt,
      controller: new AbortController(),
      editor: null as Confirmation | null,
      pending: false,
      needsReview: false,
      error: null as ArchiveError | null,
      remote: null as WorkspaceApiResponse | null,
      receipt: null as WorkspaceApiResponse | null
    }),
    [origin, workspaceId, attempt]
  )

  useEffect(() => {
    if (!enabled) return
    if (lifetime.controller.signal.aborted)
      lifetime.controller = new AbortController()
    const controller = lifetime.controller
    const unsubscribe = useWorkspaceStore.subscribe((state) => {
      if (
        state.activeWorkspaceOrigin !== origin ||
        state.workspaceId !== workspaceId ||
        state.ownedWorkspaceAttempt !== attempt
      ) {
        controller.abort()
        lifetime.editor = null
      }
    })
    return () => {
      controller.abort()
      lifetime.editor = null
      unsubscribe()
    }
  }, [enabled, lifetime, origin, workspaceId, attempt])

  const current = () => {
    const state = useWorkspaceStore.getState()
    return (
      enabled &&
      !lifetime.controller.signal.aborted &&
      !lifetime.attempt &&
      state.activeWorkspaceOrigin === lifetime.origin &&
      state.workspaceId === lifetime.workspaceId &&
      state.ownedWorkspaceAttempt === lifetime.attempt
    )
  }
  const editor = lifetime.editor
  const editorCurrent = (
    candidate: Confirmation | null
  ): candidate is Confirmation =>
    !!candidate && current() && lifetime.editor === candidate
  // Capturing this record also invalidates callbacks from before a status review.
  const reviewedWorkspace = editor?.workspace
  const report = (error: ArchiveError | null) => {
    lifetime.error = error
    if (current()) render()
  }
  const complete = (
    candidate: Confirmation,
    workspace: WorkspaceApiResponse
  ) => {
    if (!editorCurrent(candidate)) return
    lifetime.needsReview = true
    try {
      const accepted = useWorkspaceStore
        .getState()
        .completeOwnedWorkspaceArchive({
          origin,
          workspaceId,
          workspace
        })
      // Successful completion intentionally clears the owned activation.
      if (accepted) navigate(WORKSPACES_PATH)
      else if (editorCurrent(candidate)) {
        const currentVersion =
          useWorkspaceStore.getState().ownedWorkspaceBundle?.workspace.version
        if (currentVersion == null || workspace.version < currentVersion) {
          lifetime.receipt = null
          report("reviewFailed")
        } else {
          lifetime.receipt = workspace
          report("completionFailed")
        }
      }
    } catch {
      if (editorCurrent(candidate)) {
        lifetime.receipt = null
        report("reviewFailed")
      }
    }
  }
  const open = () => {
    if (!current() || lifetime.editor) return
    const workspace =
      lifetime.remote ??
      useWorkspaceStore.getState().ownedWorkspaceBundle?.workspace
    if (!workspace) return
    lifetime.editor = { workspace }
    if (!lifetime.needsReview) lifetime.error = null
    render()
  }
  const cancel = () => {
    if (!editorCurrent(editor) || lifetime.receipt) return
    lifetime.editor = null
    render()
  }
  const confirm = async () => {
    if (
      !editorCurrent(editor) ||
      editor.workspace !== reviewedWorkspace ||
      lifetime.pending ||
      lifetime.needsReview ||
      lifetime.receipt ||
      origin.kind !== "server-owned"
    )
      return
    const controller = lifetime.controller
    lifetime.pending = true
    report(null)
    try {
      const context = await createOwnedWorkspaceLifecycleContext(
        workspaceId,
        origin.scope,
        controller.signal
      )
      if (!editorCurrent(editor) || controller.signal.aborted) return
      if (
        !useWorkspaceStore
          .getState()
          .prepareOwnedWorkspaceArchive({ origin, workspaceId })
      ) {
        report("storageUnavailable")
        return
      }
      if (!editorCurrent(editor) || controller.signal.aborted) return
      lifetime.needsReview = true
      lifetime.error = "uncertain"
      const workspace = await context.setArchived(
        true,
        reviewedWorkspace.version
      )
      if (!editorCurrent(editor) || controller.signal.aborted) return
      complete(editor, workspace)
    } catch (error) {
      if (!current() || controller.signal.aborted) return
      // Even a cancelled modal retains the write barrier until a successful GET.
      lifetime.needsReview = true
      const status = (error as { status?: number })?.status
      report(
        status === 409
          ? "conflict"
          : status === 412
            ? "accountChanged"
            : "uncertain"
      )
    } finally {
      lifetime.pending = false
      if (current()) render()
    }
  }
  const checkStatus = async () => {
    if (
      !editorCurrent(editor) ||
      lifetime.pending ||
      lifetime.receipt ||
      origin.kind !== "server-owned"
    )
      return
    const controller = lifetime.controller
    lifetime.pending = true
    lifetime.needsReview = true
    render()
    try {
      const context = await createOwnedWorkspaceLifecycleContext(
        workspaceId,
        origin.scope,
        controller.signal
      )
      if (!editorCurrent(editor) || controller.signal.aborted) return
      const workspace = await context.get()
      if (!editorCurrent(editor) || controller.signal.aborted) return
      const currentVersion =
        useWorkspaceStore.getState().ownedWorkspaceBundle?.workspace.version
      if (
        currentVersion == null ||
        workspace.version < Math.max(currentVersion, editor.workspace.version)
      ) {
        report("reviewFailed")
        return
      }
      if (workspace.archived) {
        complete(editor, workspace)
      } else {
        lifetime.remote = workspace
        editor.workspace = workspace
        lifetime.needsReview = false
        report(null)
      }
    } catch (error) {
      if (editorCurrent(editor) && !controller.signal.aborted)
        report(
          (error as { status?: number })?.status === 412
            ? "accountChanged"
            : "reviewFailed"
        )
    } finally {
      lifetime.pending = false
      if (current()) render()
    }
  }
  const finish = () => {
    if (editorCurrent(editor) && !lifetime.pending && lifetime.receipt)
      complete(editor, lifetime.receipt)
  }
  const visible = editorCurrent(editor)
  return {
    enabled,
    isOpen: visible,
    pending: visible && lifetime.pending,
    error: visible ? lifetime.error : null,
    remote: visible ? lifetime.remote : null,
    completionBlocked: visible && !!lifetime.receipt,
    canConfirm:
      visible &&
      !lifetime.pending &&
      !lifetime.needsReview &&
      !lifetime.receipt,
    needsReview: visible && lifetime.needsReview && !lifetime.receipt,
    open,
    cancel,
    confirm,
    checkStatus,
    finish
  }
}
