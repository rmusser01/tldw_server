import { useEffect, useMemo, useReducer } from "react"
import { sanitizeWorkspaceBanner, useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceBannerDraft } from "@/store/owned-workspace-state"
import { createOwnedWorkspaceMetadataContext } from "@/services/owned-workspace-opening"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"

type BannerError =
  | "conflict"
  | "accountChanged"
  | "loadFailed"
  | "saveFailed"
  | "reviewFailed"
type Review = {
  workspace: WorkspaceApiResponse
  expectedWorkspace: WorkspaceApiResponse
}
type Editor = {
  session: number
  pending: "load" | "save" | "review" | null
  error: BannerError | null
  workspace: WorkspaceApiResponse | null
  review: Review | null
}

/** Persist text and its base version; request identity belongs to this modal only. */
export function useOwnedWorkspaceBanner() {
  const origin = useWorkspaceStore((s) => s.activeWorkspaceOrigin)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const attempt = useWorkspaceStore((s) => s.ownedWorkspaceAttempt)
  const draft = useWorkspaceStore((s) => s.ownedWorkspaceBannerDraft)
  const session = useWorkspaceStore((s) => s.ownedWorkspaceBannerSession)
  const storageStatus = useWorkspaceStore((s) => s.ownedWorkspaceDraftStatus)
  const enabled = origin?.kind === "server-owned"
  const [, render] = useReducer((n: number) => n + 1, 0)
  const lifetime = useMemo(
    () => ({
      origin,
      workspaceId,
      attempt,
      controller: new AbortController(),
      editor: null as Editor | null
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
      state.activeWorkspaceOrigin === origin &&
      state.workspaceId === workspaceId &&
      state.ownedWorkspaceAttempt === attempt
    )
  }
  const editor = lifetime.editor
  const editorCurrent = (candidate: Editor | null): candidate is Editor =>
    !!candidate &&
    current() &&
    lifetime.editor === candidate &&
    useWorkspaceStore.getState().ownedWorkspaceBannerSession ===
      candidate.session
  const updateDraft = (
    candidate: Editor,
    next: OwnedWorkspaceBannerDraft | null
  ) => {
    if (!editorCurrent(candidate)) return false
    const state = useWorkspaceStore.getState()
    return state.setOwnedWorkspaceBannerDraft({
      origin,
      workspaceId,
      expectedDraft: state.ownedWorkspaceBannerDraft,
      draft: next
    })
  }
  const report = (candidate: Editor, error: BannerError | null) => {
    if (!editorCurrent(candidate)) return
    candidate.error = error
    if (error) candidate.review = null
    render()
  }
  const close = (candidate: Editor) => {
    if (lifetime.editor !== candidate) return
    lifetime.editor = null
    render()
  }

  const load = async (candidate: Editor, inspect: boolean) => {
    if (
      !editorCurrent(candidate) ||
      candidate.pending ||
      origin.kind !== "server-owned"
    )
      return
    const expectedWorkspace =
      useWorkspaceStore.getState().ownedWorkspaceBundle?.workspace
    if (!expectedWorkspace) return
    candidate.pending = inspect ? "review" : "load"
    candidate.review = null
    render()
    try {
      const context = await createOwnedWorkspaceMetadataContext(
        workspaceId,
        origin.scope,
        lifetime.controller.signal
      )
      if (!editorCurrent(candidate)) return
      const workspace = await context.get()
      if (!editorCurrent(candidate)) return
      const state = useWorkspaceStore.getState()
      const savedDraft = state.ownedWorkspaceBannerDraft
      if (
        state.ownedWorkspaceBundle?.workspace !== expectedWorkspace ||
        workspace.version < expectedWorkspace.version ||
        (savedDraft && workspace.version < savedDraft.baseVersion)
      ) {
        report(candidate, "reviewFailed")
        return
      }
      candidate.workspace = workspace
      if (!savedDraft) {
        if (
          !updateDraft(candidate, {
            title: workspace.banner_title ?? "",
            subtitle: workspace.banner_subtitle ?? "",
            baseVersion: workspace.version
          })
        )
          return
        candidate.session =
          useWorkspaceStore.getState().ownedWorkspaceBannerSession
        candidate.error = null
      } else {
        // A restored editor keeps its original base; even equal-version recovery is explicit.
        candidate.review = { workspace, expectedWorkspace }
        candidate.error =
          workspace.version > savedDraft.baseVersion ? "conflict" : null
      }
      render()
    } catch (error) {
      report(
        candidate,
        (error as { status?: number })?.status === 412
          ? "accountChanged"
          : inspect
            ? "reviewFailed"
            : "loadFailed"
      )
    } finally {
      candidate.pending = null
      if (editorCurrent(candidate)) render()
    }
  }

  const open = async () => {
    if (!current() || editorCurrent(lifetime.editor)) return
    const candidate: Editor = {
      session: useWorkspaceStore.getState().ownedWorkspaceBannerSession,
      pending: null,
      error: null,
      workspace: null,
      review: null
    }
    lifetime.editor = candidate
    render()
    await load(candidate, false)
  }
  const cancel = () => {
    if (!editorCurrent(editor)) return
    if (updateDraft(editor, null)) close(editor)
  }
  const change = (
    fields: Partial<Pick<OwnedWorkspaceBannerDraft, "title" | "subtitle">>
  ) => {
    if (!editorCurrent(editor)) return
    const draft = useWorkspaceStore.getState().ownedWorkspaceBannerDraft
    if (!draft) return
    updateDraft(editor, { ...draft, ...fields })
    render()
  }
  const maySave = (
    candidate: Editor | null,
    draft: OwnedWorkspaceBannerDraft | null
  ) =>
    editorCurrent(candidate) &&
    !!draft &&
    !!candidate.workspace &&
    !candidate.pending &&
    !candidate.error &&
    !candidate.review

  const save = async (reset = false) => {
    const state = useWorkspaceStore.getState()
    const submittedDraft = state.ownedWorkspaceBannerDraft
    const expectedWorkspace = state.ownedWorkspaceBundle?.workspace
    if (
      !editor ||
      !submittedDraft ||
      !expectedWorkspace ||
      !maySave(editor, submittedDraft) ||
      origin.kind !== "server-owned"
    )
      return
    const editorSession = editor.session
    editor.pending = "save"
    render()
    try {
      const context = await createOwnedWorkspaceMetadataContext(
        workspaceId,
        origin.scope,
        lifetime.controller.signal
      )
      if (
        !editorCurrent(editor) ||
        useWorkspaceStore.getState().ownedWorkspaceBannerDraft !==
          submittedDraft
      )
        return
      const text = sanitizeWorkspaceBanner({ ...submittedDraft, image: null })
      const workspace = await context.patch({
        version: submittedDraft.baseVersion,
        banner_title: reset ? "" : text.title,
        banner_subtitle: reset ? "" : text.subtitle
      })
      if (!current()) return
      // Receipt acceptance is separate from modal identity: Cancel cannot undo a dispatched write.
      const result = useWorkspaceStore
        .getState()
        .applyOwnedWorkspaceBannerReceipt({
          origin,
          workspaceId,
          expectedWorkspace,
          submittedDraft,
          editorSession,
          workspace
        })
      if (result === "saved") close(editor)
      else if (result === "stale") report(editor, "saveFailed")
      else if (editorCurrent(editor)) {
        editor.workspace = workspace
        render()
      }
    } catch (error) {
      const status = (error as { status?: number })?.status
      report(
        editor,
        status === 409
          ? "conflict"
          : status === 412
            ? "accountChanged"
            : "saveFailed"
      )
    } finally {
      editor.pending = null
      if (editorCurrent(editor)) render()
    }
  }

  const review = async () => {
    if (editorCurrent(editor) && editor.error !== "accountChanged")
      await load(editor, true)
  }
  const keepMine = () => {
    if (!editorCurrent(editor) || editor.pending || !editor.review) return
    const state = useWorkspaceStore.getState()
    const draft = state.ownedWorkspaceBannerDraft
    if (
      !draft ||
      state.ownedWorkspaceBundle?.workspace !== editor.review.expectedWorkspace
    ) {
      report(editor, "reviewFailed")
      return
    }
    if (
      updateDraft(editor, {
        ...draft,
        baseVersion: editor.review.workspace.version
      })
    ) {
      editor.review = null
      report(editor, null)
    }
  }
  const useServer = () => {
    if (!editorCurrent(editor) || editor.pending || !editor.review) return
    const state = useWorkspaceStore.getState()
    const submittedDraft = state.ownedWorkspaceBannerDraft
    if (!submittedDraft) return
    try {
      const result = state.acceptOwnedWorkspaceBannerReview({
        origin,
        workspaceId,
        expectedWorkspace: editor.review.expectedWorkspace,
        submittedDraft,
        editorSession: editor.session,
        workspace: editor.review.workspace
      })
      if (result === "saved") close(editor)
      else report(editor, "reviewFailed")
    } catch {
      report(editor, "reviewFailed")
    }
  }

  const visible =
    editor?.session === session && editorCurrent(editor) ? editor : null
  return {
    enabled,
    draft,
    isOpen: !!visible,
    loading: visible?.pending === "load" || visible?.pending === "review",
    saving: visible?.pending === "save",
    pending: !!visible?.pending,
    error: visible?.error ?? null,
    remote: visible?.review?.workspace ?? null,
    workspace: visible?.workspace ?? null,
    canSave: maySave(visible, draft),
    storageUnavailable:
      !!draft &&
      (storageStatus === "unavailable" || storageStatus === "invalid"),
    open,
    cancel,
    changeTitle: (title: string) => change({ title }),
    changeSubtitle: (subtitle: string) => change({ subtitle }),
    save,
    review,
    keepMine,
    useServer
  }
}
